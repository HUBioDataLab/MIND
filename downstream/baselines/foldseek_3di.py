"""Foldseek 3Di states for the exact residues a processed MIND split kept (CPU data processing).

    python -m downstream.baselines.foldseek_3di --task ec --splits train valid test \\
        --foldseek <path to the foldseek binary> --threads 16 --workers 8

Foldseek runs once per split in batch mode (``structureto3didescriptor`` with
``--chain-name-mode 1``, the call SaProt's own ``get_struc_seq`` makes) over a
directory of symlinks to the split's structure files. Its output has one amino
acid and one 3Di letter per residue it read, but no residue numbers, so the
residues are matched as follows (any violation raises ``AlignmentError``; there
is no truncation or fuzzy alignment):

1. The structure file is parsed once with Biopython exactly like MIND
   processing (first model, altloc/point-mutation resolution of
   ``structure_io``); residues are taken in file order.
2. Foldseek reads gemmi's first conformer: of consecutive residues sharing a
   residue number (microheterogeneity such as CYS/DDZ or SER/PYR at one
   position) only the first in file order is read; the others are
   "shadowed" and have no Foldseek state. A Biopython point-mutation residue
   is read as its first-listed variant. The readable residues with a CA are
   the candidates (water excluded).
3. The candidates are aligned to Foldseek's amino-acid string without
   mismatches: a standard candidate (20 amino acids, MSE read as MET) must meet
   the same Foldseek letter; a nonstandard candidate meets any letter (Foldseek
   maps some modified residues to their parent, others to X) and is the only
   kind that may be absent from Foldseek's output. Foldseek residues absent
   from the candidates are not allowed.
4. ``missing_atoms.apply_policy`` gives the kept residues (MIND drops residues
   missing N/CA/C/O), identified by ``(resseq, insertion code, residue name)``.
   A shadowed kept residue gets ``#``; any other must be a candidate with
   exactly one outcome over all valid alignments: one Foldseek position, or
   absent (``#``). Foldseek writes nothing for chains with fewer than 4 read
   residues; all their kept residues get ``#``. More than one outcome, no valid alignment, or a kept
   sequence that differs from the processed split raises.

The cache stores, per chain, the kept sequence, a 3Di string over the kept
residues (Foldseek's letter unchanged, including its invalid-geometry state at
chain ends and gaps, which SaProt also saw in training; ``#`` for a kept
residue Foldseek has no state for) and the indices of residues MIND masks for
an incomplete sidechain. ``saprot.py`` writes ``<aa>#`` (SaProt's
unknown-structure letter) for both.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from downstream.common.missing_atoms import apply_policy, parse_policy
from downstream.common.registry import sha256_file
from downstream.common.structure_io import (
    THREE_TO_ONE, _select_residue, chain_records_from_model, load_first_model,
)

THREE_DI_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
NO_STATE = "#"
MIN_FOLDSEEK_RESIDUES = 4      # Foldseek writes no entry for chains with fewer residues (checked on 10-941cd33)
WATER_NAMES = frozenset(("HOH", "WAT", "DOD"))
CACHE_FORMAT = "mind_foldseek_3di_v1"
ResidueKey = Tuple[int, str, str]          # (resseq, insertion code, residue name)


@dataclass(frozen=True)
class Candidate:
    """A residue Foldseek may have read: ``letter`` is None for nonstandard residues."""

    key: ResidueKey
    letter: Optional[str]
    may_be_absent: bool


class AlignmentError(ValueError):
    """Foldseek output and the MIND residue list disagree."""


# ---------------------------------------------------------------------------
# Foldseek
# ---------------------------------------------------------------------------

def foldseek_version(binary: str) -> str:
    out = subprocess.run([binary, "version"], check=True, capture_output=True, text=True)
    return out.stdout.strip()


def foldseek_command(binary: str, input_dir: Path, output: Path, threads: int) -> List[str]:
    return [binary, "structureto3didescriptor", str(input_dir), str(output),
            "--threads", str(threads), "--chain-name-mode", "1", "-v", "1"]


def parse_descriptor_tsv(path: Path) -> Dict[str, Tuple[str, str]]:
    """``entry name -> (amino acids, 3Di)`` from Foldseek's descriptor TSV."""
    out: Dict[str, Tuple[str, str]] = {}
    with open(path) as handle:
        for number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise AlignmentError(f"{path}:{number}: expected 4 columns, got {len(fields)}")
            name, aa, three_di = fields[0], fields[1], fields[2]
            if name in out:
                raise AlignmentError(f"{path}:{number}: duplicate entry {name}")
            if len(aa) != len(three_di):
                raise AlignmentError(f"{path}:{number}: {name} has {len(aa)} residues but {len(three_di)} 3Di states")
            bad = set(three_di) - THREE_DI_ALPHABET
            if bad:
                raise AlignmentError(f"{path}:{number}: {name} has 3Di letters outside the alphabet: {sorted(bad)}")
            out[name] = (aa, three_di)
    return out


def run_foldseek(binary: str, files: Sequence[Path], *, threads: int,
                 workdir: Path) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """Run Foldseek once over ``files``; return ``{file basename: {chain: (aa, 3Di)}}``."""
    names = [Path(f).name for f in files]
    duplicates = [n for n, c in Counter(names).items() if c > 1]
    if duplicates:
        raise ValueError(f"structure file names are not unique: {duplicates[:5]}")
    links = workdir / "structures"
    links.mkdir()
    for path in files:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        os.symlink(Path(path).resolve(), links / Path(path).name)
    output = workdir / "descriptor.tsv"
    subprocess.run(foldseek_command(binary, links, output, threads), check=True)
    parsed = parse_descriptor_tsv(output)
    by_file: Dict[str, Dict[str, Tuple[str, str]]] = {}
    known = set(names)
    for name, (aa, three_di) in parsed.items():
        base, sep, chain = name.rpartition("_")
        if not sep or base not in known:
            raise AlignmentError(f"cannot map Foldseek entry {name!r} to an input file")
        by_file.setdefault(base, {})[chain] = (aa, three_di)
    return by_file


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def candidate_residues(bio_chain) -> Tuple[List[Candidate], set]:
    """Residues of a Biopython chain Foldseek can read, and the keys of residues it cannot see.

    Foldseek reads gemmi's first conformer: of consecutive residues with the same
    number (microheterogeneity) only the first in file order is read, and a
    point-mutation residue is read as its first-listed variant. Candidates are the
    read, non-water residues with a CA. Keys use the residue name MIND selects
    (``structure_io._select_residue``); the letter is the variant Foldseek reads.
    """
    candidates, shadowed, previous = [], set(), None
    for raw in bio_chain:
        residue, _ = _select_residue(raw)
        first = raw.disordered_get_list()[0] if raw.is_disordered() == 2 else residue
        hetflag, resseq, icode = residue.get_id()
        number = (int(resseq), icode.strip())
        key = (*number, residue.get_resname().strip().upper())
        if number == previous:
            shadowed.add(key)
            continue
        previous = number
        first_name = first.get_resname().strip().upper()
        if hetflag == "W" or first_name in WATER_NAMES or "CA" not in first:
            continue
        letter = "M" if first_name == "MSE" else THREE_TO_ONE.get(first_name)
        candidates.append(Candidate(key, letter, letter is None))
    return candidates, shadowed


def _compatible(candidate: Candidate, letter: str) -> bool:
    return candidate.letter is None or candidate.letter == letter


def align_kept_residues(
    chain: str,
    candidates: Sequence[Candidate],
    foldseek_aa: str,
    foldseek_3di: str,
    kept_keys: Sequence[ResidueKey],
    kept_sequence: str,
    shadowed: Optional[set] = None,
) -> str:
    """3Di letters of the kept residues (``#`` where Foldseek has none); raises ``AlignmentError``.

    Alignment states are ``(i, s)``: ``i`` candidates consumed, ``s`` of them absent
    from Foldseek, so candidate ``i`` meets Foldseek position ``i - s``.
    """
    if len(foldseek_aa) != len(foldseek_3di):
        raise AlignmentError(f"{chain}: Foldseek gave {len(foldseek_aa)} residues and {len(foldseek_3di)} 3Di states")
    n, m = len(candidates), len(foldseek_aa)
    absent = n - m
    if absent < 0:
        raise AlignmentError(f"{chain}: Foldseek read {m} residues, more than the {n} parsed residues with a CA")
    if absent > sum(c.may_be_absent for c in candidates):
        raise AlignmentError(f"{chain}: Foldseek read {m} of {n} residues; only "
                             f"{sum(c.may_be_absent for c in candidates)} may be absent")

    def match(i: int, s: int) -> bool:
        j = i - s
        return 0 <= j < m and _compatible(candidates[i], foldseek_aa[j])

    def skip(i: int, s: int) -> bool:
        return s < absent and candidates[i].may_be_absent

    forward = [[False] * (absent + 1) for _ in range(n + 1)]
    forward[0][0] = True
    for i in range(n):
        for s in range(absent + 1):
            if forward[i][s]:
                if match(i, s):
                    forward[i + 1][s] = True
                if skip(i, s):
                    forward[i + 1][s + 1] = True
    if not forward[n][absent]:
        raise AlignmentError(f"{chain}: no mismatch-free alignment of {n} parsed residues to Foldseek's {m}")
    backward = [[False] * (absent + 1) for _ in range(n + 1)]
    backward[n][absent] = True
    for i in range(n - 1, -1, -1):
        for s in range(absent + 1):
            backward[i][s] = ((match(i, s) and backward[i + 1][s])
                              or (skip(i, s) and backward[i + 1][s + 1]))

    index = {}
    for position, candidate in enumerate(candidates):
        if candidate.key in index:
            index[candidate.key] = None          # ambiguous key; fails only if a kept residue uses it
        else:
            index[candidate.key] = position
    if len(kept_keys) != len(kept_sequence):
        raise AlignmentError(f"{chain}: {len(kept_keys)} kept residues but sequence length {len(kept_sequence)}")
    shadowed = shadowed or set()
    letters = []
    for key in kept_keys:
        if key in shadowed:
            if key in index:
                raise AlignmentError(f"{chain}: kept residue {key} is both read and shadowed")
            letters.append(NO_STATE)
            continue
        i = index.get(key, "missing")
        if i == "missing" or i is None:
            raise AlignmentError(f"{chain}: kept residue {key} is {'not' if i == 'missing' else 'not uniquely'} a parsed CA residue")
        outcomes = set()
        for s in range(absent + 1):
            if forward[i][s]:
                if match(i, s) and backward[i + 1][s]:
                    outcomes.add(i - s)
                if skip(i, s) and backward[i + 1][s + 1]:
                    outcomes.add(None)
        if len(outcomes) != 1:
            raise AlignmentError(f"{chain}: kept residue {key} aligns ambiguously ({sorted(outcomes, key=str)})")
        j = outcomes.pop()
        letters.append(NO_STATE if j is None else foldseek_3di[j])
    return "".join(letters)


def chain_3di(job) -> dict:
    """Parse one structure file and align Foldseek's states to the residues MIND kept."""
    chain, path, chain_id, foldseek_chains, expected_sequence, settings = job
    model, num_models = load_first_model(Path(path))
    bio_chains = {c.id: c for c in model}
    if chain_id is None:
        if len(bio_chains) != 1 or len(foldseek_chains) > 1:
            raise AlignmentError(
                f"{chain}: expected a single-chain file, Biopython has {sorted(bio_chains)}, "
                f"Foldseek has {sorted(foldseek_chains)}"
            )
        chain_id = next(iter(bio_chains))
        entry = next(iter(foldseek_chains.values()), None)
    else:
        if chain_id not in bio_chains:
            raise AlignmentError(f"{chain}: chain {chain_id} missing (file has {sorted(bio_chains)})")
        entry = foldseek_chains.get(chain_id)
    candidates, shadowed = candidate_residues(bio_chains[chain_id])
    if entry is None:
        if len(candidates) >= MIN_FOLDSEEK_RESIDUES:
            raise AlignmentError(f"{chain}: Foldseek wrote no entry for a chain with {len(candidates)} residues")
        entry = ("", "")
        candidates = [Candidate(c.key, c.letter, True) for c in candidates]   # all absent -> '#'
    foldseek_aa, foldseek_3di = entry
    record = chain_records_from_model(model, Path(path).name.split(".")[0], num_models,
                                      mse_to_met=settings["mse_to_met"])[chain_id]
    outcome = apply_policy(record, parse_policy(settings["missing_atoms"]))
    if outcome.dropped_reason:
        raise AlignmentError(f"{chain}: the policy now drops this chain ({outcome.dropped_reason})")
    kept_sequence = "".join(THREE_TO_ONE[r.resname] for r in outcome.residues)
    if kept_sequence != expected_sequence:
        raise AlignmentError(f"{chain}: re-parsed kept sequence differs from the processed split")
    three_di = align_kept_residues(chain, candidates, foldseek_aa, foldseek_3di,
                                   [(r.resseq, r.icode, r.source_resname) for r in outcome.residues],
                                   kept_sequence, shadowed)
    return {"chain": chain, "seq": kept_sequence, "3di": three_di,
            "masked": [i for i, flag in enumerate(outcome.masked) if flag],
            "foldseek_residues": len(foldseek_aa), "candidates": len(candidates)}


def _safe_chain_3di(job):
    try:
        return chain_3di(job)
    except Exception as exc:  # collected and reported; any failure stops the split
        return {"chain": job[0], "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_dir(config: dict) -> Path:
    """``<data_root>/foldseek_3di[_<processed_revision>]``: a cache belongs to one processed revision."""
    from downstream.tasks.ec.dataset import task_paths
    revision = str(config["processed_revision"])
    return task_paths(config)["root"] / ("foldseek_3di" + (f"_{revision}" if revision else ""))


def load_cache(path: Path, chains: Sequence[str], sequences: Sequence[str]) -> Tuple[List[dict], dict]:
    """``(records, provenance)``; records follow the processed split, whose chains and sequences must match."""
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != CACHE_FORMAT:
        raise ValueError(f"{path}: format {payload.get('format')!r}, expected {CACHE_FORMAT}")
    records = payload["chains"]
    if [r["chain"] for r in records] != list(chains):
        raise ValueError(f"{path}: chain list differs from the processed split")
    for record, sequence in zip(records, sequences):
        if record["seq"] != sequence or len(record["3di"]) != len(sequence):
            raise ValueError(f"{path}: {record['chain']} sequence/3Di differs from the processed split")
    return records, payload["provenance"]


def build_split_cache(config: dict, split: str, *, binary: str, threads: int, workers: int) -> Path:
    from downstream.tasks.ec.dataset import ProcessedSplit, processed_dir, structure_locations

    out = cache_dir(config) / f"{split}.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    source = processed_dir(config) / f"{split}.pt"
    processed = ProcessedSplit(source)
    locations = structure_locations(config)
    files = sorted({str(locations[c][0]) for c in processed.chains})
    version = foldseek_version(binary)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"foldseek_{split}_") as tmp:
        command = foldseek_command(binary, Path("<symlink dir>"), Path("<tmp>/descriptor.tsv"), threads)
        foldseek_out = run_foldseek(binary, [Path(f) for f in files], threads=threads, workdir=Path(tmp))
    foldseek_seconds = time.monotonic() - started
    settings = {"missing_atoms": config["missing_atoms"], "mse_to_met": bool(config["structure"]["mse_to_met"])}
    jobs = []
    for chain, sequence in zip(processed.chains, processed.sequences):
        path, chain_id = locations[chain]
        name = Path(path).name
        if name not in foldseek_out:
            jobs.append((chain, str(path), chain_id, {}, sequence, settings))
        else:
            jobs.append((chain, str(path), chain_id, foldseek_out[name], sequence, settings))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_safe_chain_3di, jobs, chunksize=16))
    errors = [r for r in results if "error" in r]
    if errors:
        for r in errors[:20]:
            print(f"[{split}] MISMATCH {r['chain']}: {r['error']}", flush=True)
        raise AlignmentError(f"[{split}] {len(errors)}/{len(results)} chains failed alignment; cache not written")
    residues = sum(len(r["seq"]) for r in results)
    masked = sum(len(r["masked"]) for r in results)
    payload = {
        "format": CACHE_FORMAT,
        "provenance": {
            "split": split, "created_utc": datetime.now(timezone.utc).isoformat(),
            "processed": str(source), "processed_sha256": sha256_file(source),
            "foldseek_binary": binary, "foldseek_binary_sha256": sha256_file(Path(binary)),
            "foldseek_version": version, "foldseek_command": " ".join(command),
            "foldseek_seconds": round(foldseek_seconds, 1), "structure_files": len(files),
            "missing_atoms": config["missing_atoms"],
            "alignment": "first-conformer (gemmi) residues of the first model with a CA, aligned to Foldseek's "
                         "sequence without mismatches; only nonstandard residues may be absent; kept residues "
                         "shadowed by microheterogeneity or absent get '#'; each kept residue has exactly one "
                         "outcome; see downstream/baselines/foldseek_3di.py",
            "chains": len(results), "residues": residues, "masked_residues": masked,
            "masked_fraction": round(masked / max(residues, 1), 6),
            "no_foldseek_state_residues": sum(r["3di"].count(NO_STATE) for r in results),
            "foldseek_residues_total": sum(r["foldseek_residues"] for r in results),
            "parsed_ca_residues_total": sum(r["candidates"] for r in results),
            "three_di_counts": dict(sorted(Counter("".join(r["3di"] for r in results)).items())),
            "mismatches": 0,
        },
        "chains": [{k: r[k] for k in ("chain", "seq", "3di", "masked")} for r in results],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    partial.write_text(json.dumps(payload))
    os.replace(partial, out)
    print(json.dumps(payload["provenance"], indent=1), flush=True)
    return out


def main(argv=None) -> int:
    from downstream.tasks.ec.dataset import TASK_YAML, load_task_config

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, choices=["ec"])
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--foldseek", required=True, help="path to the foldseek binary")
    parser.add_argument("--threads", type=int, default=16, help="Foldseek threads")
    parser.add_argument("--workers", type=int, default=8, help="Biopython parsing processes")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_task_config(TASK_YAML, data_root=args.data_root)
    for split in args.splits or config["splits"]:
        build_split_cache(config, split, binary=args.foldseek, threads=args.threads, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
