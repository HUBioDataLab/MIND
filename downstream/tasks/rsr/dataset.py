"""ATOM3D RSR: LMDB reading, decoy -> RNA graph processing with QC, processed-split access.

``build_processed`` turns the three ATOM3D split LMDBs into ``processed/<revision>/<split>.pt``: per candidate model one
compact edge-free RNA graph (``downstream.common.rna_graph``), its RMSD to the native structure and its puzzle. Nothing
is skipped silently: every model that cannot be converted is counted with a reason (``qc_<split>.tsv`` / ``.json``), and
a split whose failed or dropped fraction exceeds ``qc.max_failed_structure_fraction`` (or that loses a whole puzzle, or
whose puzzle count differs from ``expected_puzzles``) gets only its QC report and makes the run exit 1.

The LMDB is read without the ``atom3d`` package: each value is a gzip-compressed JSON or msgpack dict whose DataFrame
entries are ``DataFrame.to_dict(orient="split")`` dicts (``columns`` / ``index`` / ``data``). A pickle is refused.
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from data_loading.data_root import expand_data_root
from downstream.common import rna_graph as rg

TASK_YAML = Path(__file__).with_name("task.yaml")
KIND = "ranking_regression"
HYDROGEN_ELEMENTS = frozenset({"H", "D"})
SERIALIZATION_FORMATS = ("json", "msgpack")
ID_SEPARATOR = "/"
REQUIRED_KEYS = (
    "task", "modality", "data_root", "kind", "source", "sources", "splits", "expected_counts", "expected_puzzles",
    "metric", "primary_subset", "target_name", "target_unit", "puzzle_from", "structure", "missing_atoms",
    "min_nucleotides", "qc", "processed_revision", "embedding", "probe", "finetune", "finetune_seconds_per_epoch",
)
ATOM_COLUMNS = ("chain", "residue", "resname", "x", "y", "z", "element", "name")
OPTIONAL_COLUMNS = ("insertion_code", "altloc", "hetero")


# ---------------------------------------------------------------------------
# Config and paths
# ---------------------------------------------------------------------------

def load_task_config(path: Path = TASK_YAML, *, data_root: Optional[Path] = None) -> dict:
    with open(path) as handle:
        config = expand_data_root(yaml.safe_load(handle))
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if config["kind"] != KIND:
        raise ValueError(f"{path}: kind is {config['kind']!r}, expected {KIND!r}")
    if config["source"] not in config["sources"]:
        raise ValueError(f"source {config['source']!r} is not configured under sources")
    if list(config["splits"]) != ["train", "val", "test"]:
        raise ValueError(f"splits must be [train, val, test] (train first, validation second), got {config['splits']}")
    for key in ("expected_counts", "expected_puzzles"):
        if set(config[key]) != set(config["splits"]):
            raise ValueError(f"{key} must name exactly the splits {config['splits']}, got {sorted(config[key])}")
    if config["puzzle_from"] != "id_tuple_first":
        raise ValueError("puzzle_from: only 'id_tuple_first' is implemented")
    structure, missing_atoms = config["structure"], config["missing_atoms"]
    if structure.get("hydrogens") != "drop":
        raise ValueError("structure.hydrogens: only 'drop' is implemented")
    if structure.get("altloc") != "keep_blank_or_A":
        raise ValueError("structure.altloc: only 'keep_blank_or_A' is implemented")
    if missing_atoms != {"backbone": "keep_quarantined", "incomplete_base": "adapter_padding",
                         "unexpected_atoms": "remove_and_count"}:
        raise ValueError(f"missing_atoms {missing_atoms}: only the pretraining adapter rules are implemented "
                         "(backbone keep_quarantined, incomplete_base adapter_padding, unexpected_atoms remove_and_count)")
    placeholders = structure.get("placeholder_atoms")
    if not isinstance(placeholders, list) or not all(isinstance(name, str) and name for name in placeholders):
        raise ValueError("structure.placeholder_atoms must be a list of atom names (may be empty)")
    aliases = structure.get("resname_aliases") or {}
    bad = {key: value for key, value in aliases.items() if value not in rg.KNOWN_NUCLEOTIDES}
    if bad:
        raise ValueError(f"structure.resname_aliases must map to one of {rg.KNOWN_NUCLEOTIDES}, got {bad}")
    if int(config["min_nucleotides"]) < 1:
        raise ValueError("min_nucleotides must be positive")
    for key in ("max_failed_structure_fraction", "max_incomplete_base_fraction"):
        if not 0 <= float(config["qc"][key]) <= 1:
            raise ValueError(f"qc.{key} must lie in [0, 1]")
    if config["metric"] != "spearman_rho" or config["primary_subset"] != "mean_per_puzzle":
        raise ValueError("the primary cell of rsr is mean_per_puzzle spearman_rho")
    if data_root is not None:
        config["data_root"] = str(data_root)
    return config


def task_paths(config: dict) -> Dict[str, Path]:
    root = Path(config["data_root"])
    return {
        "root": root, "raw": root / "raw", "extracted": root / "raw" / "extracted",
        "processed": root / "processed", "provenance": root / "provenance.json",
    }


def processed_dir(config: dict) -> Path:
    return task_paths(config)["processed"] / str(config["processed_revision"])


# ---------------------------------------------------------------------------
# LMDB reading (no atom3d dependency)
# ---------------------------------------------------------------------------

def find_lmdb_splits(root: Path, splits: Sequence[str]) -> Dict[str, Path]:
    """The LMDB directory of each split below ``root``: a directory named like the split that holds ``data.mdb``.

    The archive layout is not documented, so it is discovered; anything but exactly one directory per split
    raises (a layout with several candidates would otherwise be resolved by guessing).
    """
    found: Dict[str, List[Path]] = {split: [] for split in splits}
    for data_file in sorted(Path(root).rglob("data.mdb")):
        if data_file.parent.name in found:
            found[data_file.parent.name].append(data_file.parent)
    problems = {split: [str(p) for p in paths] for split, paths in found.items() if len(paths) != 1}
    if problems:
        raise ValueError(f"{root}: expected exactly one LMDB directory per split {list(splits)} (a directory with "
                         f"that name holding data.mdb), found {problems}")
    return {split: paths[0] for split, paths in found.items()}


def lmdb_dirs(config: dict) -> Dict[str, Path]:
    return find_lmdb_splits(task_paths(config)["extracted"], config["splits"])


def _lmdb_header(txn, path: Path) -> Tuple[int, str]:
    """``(num_examples, serialization_format)`` of an ATOM3D LMDB, checked against its key count."""
    fmt = txn.get(b"serialization_format")
    if fmt is None or fmt.decode() not in SERIALIZATION_FORMATS:
        raise ValueError(f"{path}: serialization_format is {fmt!r}, expected one of {SERIALIZATION_FORMATS} "
                         "(a pickle is refused: it would run code from a third-party file)")
    raw = txn.get(b"num_examples")
    if raw is None:
        raise ValueError(f"{path}: LMDB has no num_examples key")
    count = int(raw.decode())
    entries = int(txn.stat()["entries"])
    if entries != count + 3:  # items + num_examples + serialization_format + id_to_idx
        raise ValueError(f"{path}: {entries} LMDB keys for num_examples={count} (expected {count + 3})")
    return count, fmt.decode()


def lmdb_length(path: Path) -> int:
    import lmdb

    env = lmdb.open(str(path), readonly=True, lock=False, max_readers=1)
    try:
        with env.begin() as txn:
            return _lmdb_header(txn, path)[0]
    finally:
        env.close()


def decode_value(raw: bytes, serialization_format: str) -> dict:
    payload = gzip.decompress(raw)
    if serialization_format == "json":
        item = json.loads(payload.decode())
    elif serialization_format == "msgpack":
        try:
            import msgpack
        except ImportError as exc:
            raise ImportError("this LMDB is msgpack-serialized; install the msgpack package") from exc
        item = msgpack.unpackb(payload, raw=False)
    else:
        raise ValueError(f"unsupported serialization_format {serialization_format!r}")
    if not isinstance(item, dict):
        raise ValueError(f"LMDB value decoded to {type(item).__name__}, expected a dict")
    return item


def iter_lmdb(path: Path, start: int = 0, stop: Optional[int] = None) -> Iterator[Tuple[int, dict]]:
    """Yield ``(index, item)`` for keys ``start .. stop-1`` (default: every example) in numeric order."""
    import lmdb

    env = lmdb.open(str(path), readonly=True, lock=False, max_readers=1)
    try:
        with env.begin() as txn:
            count, fmt = _lmdb_header(txn, path)
            for index in range(start, count if stop is None else min(stop, count)):
                raw = txn.get(str(index).encode())
                if raw is None:
                    raise KeyError(f"{path}: LMDB declares {count} examples but has no key {index!r}")
                yield index, decode_value(raw, fmt)
    finally:
        env.close()


def frame(item: dict, name: str, *, label: str) -> Tuple[Dict[str, int], List[list]]:
    """``(column -> position, rows)`` of one ``to_dict(orient='split')`` DataFrame."""
    value = item.get(name)
    if not isinstance(value, dict) or "columns" not in value or "data" not in value:
        raise ValueError(f"{label}: {name!r} is not a split-oriented DataFrame")
    return {str(column): index for index, column in enumerate(value["columns"])}, value["data"]


# ---------------------------------------------------------------------------
# One candidate model
# ---------------------------------------------------------------------------

@dataclass
class Decoy:
    puzzle: str
    nucleotides: List[rg.NucleotideRecord]
    target: float
    source_id: str
    counts: Dict[str, int] = field(default_factory=dict)


def _text(value) -> str:
    return "" if value is None else str(value).strip()


def parse_puzzle(item: dict) -> Tuple[str, str]:
    """``(puzzle, raw id)`` of an item. The LMDB ``id`` is the text of a tuple ``(puzzle name, model name)``, e.g.
    ``"('rna_puzzle_14_bound', 'S_000001_002')"``; it is parsed with ``ast.literal_eval`` (literals only). The atoms'
    ``ensemble`` column is the model's own file name, not the puzzle."""
    raw = item.get("id")
    try:
        value = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"bad_puzzle: id {raw!r} is not the text of a (puzzle, model) tuple") from exc
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or not all(isinstance(part, str) and part for part in value)):
        raise ValueError(f"bad_puzzle: id {raw!r} is not a (puzzle, model) pair of non-empty strings")
    puzzle = value[0]
    if ID_SEPARATOR in puzzle:
        raise ValueError(f"bad_puzzle: puzzle name {puzzle!r} contains {ID_SEPARATOR!r}")
    return puzzle, str(raw)


def read_decoy(item: dict, settings: dict, *, label: str) -> Decoy:
    """One LMDB item -> puzzle, nucleotides and RMSD; every dropped atom is counted in ``counts``.

    Raises ``ValueError`` with a reason prefix (``unsupported_residue``, ``duplicate_atoms``, ``bad_target``, ...) that
    ``process_item`` turns into a counted failure. ``Decoy.puzzle`` is known before any atom is read, so a failed
    model still reports its puzzle (see ``PuzzleError``).
    """
    puzzle, source_id = parse_puzzle(item)
    try:
        return _read_atoms(item, settings, label, puzzle, source_id)
    except ValueError as exc:
        raise PuzzleError(puzzle, source_id, str(exc)) from exc


class PuzzleError(ValueError):
    """A conversion failure that knows the puzzle and raw id of its model (for the counted report)."""

    def __init__(self, puzzle: str, source_id: str, message: str):
        super().__init__(message)
        self.puzzle, self.source_id = puzzle, source_id


def _read_atoms(item: dict, settings: dict, label: str, puzzle: str, source_id: str) -> Decoy:
    scores = item.get("scores")
    if not isinstance(scores, dict) or settings["target_name"] not in scores:
        raise ValueError(f"bad_target: scores has no {settings['target_name']!r} key "
                         f"(keys {sorted(scores) if isinstance(scores, dict) else scores!r})")
    target = float(scores[settings["target_name"]])
    if not math.isfinite(target) or target < 0:
        raise ValueError(f"bad_target: {settings['target_name']}={target!r} is not a finite non-negative RMSD")
    columns, rows = frame(item, "atoms", label=label)
    missing = [name for name in ATOM_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"bad_frame: atoms lacks columns {missing} (has {sorted(columns)})")
    optional = {name: columns.get(name) for name in OPTIONAL_COLUMNS}
    aliases = settings["resname_aliases"]
    placeholders = frozenset(settings["placeholder_atoms"])
    counts: Counter = Counter({key: 0 for key in (
        "atoms_read", "hydrogens_skipped", "placeholder_atoms_removed", "altloc_atoms_skipped",
        "unexpected_atoms_removed")})
    order: List[tuple] = []
    atoms: Dict[tuple, list] = {}
    resnames: Dict[tuple, str] = {}
    for row in rows:
        element = _text(row[columns["element"]]).upper()
        counts["atoms_read"] += 1
        if not element:
            # Rosetta placeholder atoms (RPA1, RPB1, ...) have no element; only the listed names are dropped, any
            # other atom without an element fails the model.
            if _text(row[columns["name"]]) not in placeholders:
                raise ValueError(f"bad_frame: atom {_text(row[columns['name']])!r} has no element and is not one of "
                                 f"the listed placeholder atoms {sorted(placeholders)}")
            counts["placeholder_atoms_removed"] += 1
            continue
        if element in HYDROGEN_ELEMENTS:
            counts["hydrogens_skipped"] += 1
            continue
        altloc = _text(row[optional["altloc"]]) if optional["altloc"] is not None else ""
        if altloc not in ("", "A"):
            counts["altloc_atoms_skipped"] += 1
            continue
        key = (_text(row[columns["chain"]]), _text(row[columns["residue"]]),
               _text(row[optional["insertion_code"]]) if optional["insertion_code"] is not None else "")
        resname = _text(row[columns["resname"]]).upper()
        resname = aliases.get(resname, resname)
        if resname not in rg.KNOWN_NUCLEOTIDES:
            raise ValueError(f"unsupported_residue: residue {key} is {resname!r}, not one of {rg.KNOWN_NUCLEOTIDES}")
        if key not in atoms:
            order.append(key)
            atoms[key], resnames[key] = [], resname
        elif resnames[key] != resname:
            raise ValueError(f"bad_frame: residue {key} changes its name from {resnames[key]} to {resname}")
        name = rg.canonical_atom_name(_text(row[columns["name"]]))
        position = (float(row[columns["x"]]), float(row[columns["y"]]), float(row[columns["z"]]))
        if not all(math.isfinite(value) for value in position):
            raise ValueError(f"bad_frame: non-finite coordinate in residue {key} atom {name}")
        atoms[key].append((name, rg.canonical_element(element, name), position))
    nucleotides = []
    for key in order:
        known = [entry for entry in atoms[key] if entry[0] in rg.KNOWN_POSITIONS]
        counts["unexpected_atoms_removed"] += len(atoms[key]) - len(known)
        names = [name for name, _, _ in known]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate_atoms: residue {key} repeats {sorted({n for n in names if names.count(n) > 1})}")
        if not known:
            raise ValueError(f"bad_frame: residue {key} keeps no recognised atom")
        nucleotides.append(rg.NucleotideRecord(resnames[key], known, label=f"{key[0]}:{resnames[key]}{key[1]}{key[2]}"))
    counts["chains"] = len({key[0] for key in order})
    return Decoy(puzzle, nucleotides, target, source_id, dict(counts))


@dataclass
class DecoyResult:
    decoy_id: str
    status: str                 # ok | failed | dropped
    reason: str
    report: dict
    sample: object = None
    target: float = float("nan")
    puzzle: str = ""
    source_id: str = ""


def process_item(item: dict, index: int, settings: dict) -> DecoyResult:
    """Convert one item; a failure is a counted result, never an exception (the QC thresholds decide)."""
    label = f"item {index}"
    try:
        decoy = read_decoy(item, settings, label=label)
    except PuzzleError as exc:
        return DecoyResult(f"{exc.puzzle}{ID_SEPARATOR}{index:06d}", "failed", _reason(exc), {}, puzzle=exc.puzzle,
                           source_id=exc.source_id)
    except Exception as exc:
        return DecoyResult(f"?{ID_SEPARATOR}{index:06d}", "failed", _reason(exc), {}, source_id=_text(item.get("id")))
    decoy_id = f"{decoy.puzzle}{ID_SEPARATOR}{index:06d}"
    report = dict(decoy.counts)
    report["nucleotides"] = len(decoy.nucleotides)
    common = dict(puzzle=decoy.puzzle, source_id=decoy.source_id, target=decoy.target)
    if len(decoy.nucleotides) < int(settings["min_nucleotides"]):
        return DecoyResult(decoy_id, "dropped", f"too_few_nucleotides: {len(decoy.nucleotides)}", report, **common)
    reports = [rg.nucleotide_report(n) for n in decoy.nucleotides]
    report["nucleotides_missing_backbone"] = sum(bool(r.missing_backbone) for r in reports)
    report["nucleotides_missing_base"] = sum(bool(r.missing_base) for r in reports)
    report["base_atoms_missing"] = sum(len(r.missing_base) for r in reports)
    report["terminal_atoms_dropped"] = sum(len(r.terminal) for r in reports)
    try:
        node = rg.build_node_sample(decoy.nucleotides, decoy_id)
        compact = rg.compact_sample(node)
    except Exception as exc:
        return DecoyResult(decoy_id, "failed", f"graph_error: {_reason(exc)}", report, **common)
    report["real_atoms"] = int(sum(len(n.atoms) for n in decoy.nucleotides))
    report["stored_nodes"] = int(compact.z.numel())
    return DecoyResult(decoy_id, "ok", "", report, compact, **common)


def _reason(exc: Exception) -> str:
    text = str(exc)
    head = text.split(":", 1)[0]
    return text if head.isidentifier() and head.islower() else f"{type(exc).__name__}: {text}"


def _process_range(arguments: tuple) -> List[DecoyResult]:
    path, start, stop, settings = arguments
    torch.set_num_threads(1)
    return [process_item(item, index, settings) for index, item in iter_lmdb(Path(path), start, stop)]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

QC_COLUMNS = (
    "decoy", "split", "status", "reason", "puzzle", "source_id", "rmsd", "nucleotides", "chains", "atoms_read",
    "hydrogens_skipped", "placeholder_atoms_removed", "altloc_atoms_skipped", "unexpected_atoms_removed",
    "terminal_atoms_dropped",
    "nucleotides_missing_backbone", "nucleotides_missing_base", "base_atoms_missing", "real_atoms", "stored_nodes",
)


def _percentiles(values: Sequence[float]) -> dict:
    if not len(values):
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "p25": float(np.percentile(array, 25)), "median": float(np.median(array)),
            "p75": float(np.percentile(array, 75)), "max": float(array.max()), "mean": float(array.mean())}


def processing_settings(config: dict) -> dict:
    return {"target_name": config["target_name"], "resname_aliases": dict(config["structure"]["resname_aliases"] or {}),
            "placeholder_atoms": list(config["structure"]["placeholder_atoms"]),
            "min_nucleotides": int(config["min_nucleotides"])}


def summarize_split(config: dict, split: str, results: Sequence[DecoyResult]) -> dict:
    """The QC summary of one split and the reasons it must not be written (empty list = fine)."""
    kept = [r for r in results if r.status == "ok"]
    status = Counter(r.status for r in results)
    bad = len(results) - len(kept)
    fraction = bad / max(len(results), 1)
    incomplete = sum(r.report.get("nucleotides_missing_base", 0) > 0 for r in kept) / max(len(kept), 1)
    per_puzzle_kept = Counter(r.puzzle for r in kept)
    per_puzzle_seen = Counter(r.puzzle for r in results if r.puzzle)
    unknown_puzzle = sum(not r.puzzle for r in results)
    summary = {
        "task": "rsr", "split": split, "models": len(results), "status": dict(status),
        "failed_or_dropped_fraction": round(fraction, 6), "limit": config["qc"]["max_failed_structure_fraction"],
        "reasons": dict(Counter(r.reason.split(":")[0] for r in results if r.status != "ok")),
        "puzzles": len(per_puzzle_kept), "expected_puzzles": int(config["expected_puzzles"][split]),
        "models_per_puzzle_kept": dict(per_puzzle_kept), "models_per_puzzle_read": dict(per_puzzle_seen),
        "models_without_puzzle": int(unknown_puzzle),
        "incomplete_base_fraction": round(incomplete, 6), "incomplete_base_limit": config["qc"]["max_incomplete_base_fraction"],
        "nucleotides": _percentiles([r.report.get("nucleotides", 0) for r in kept]),
        "stored_nodes": _percentiles([r.report.get("stored_nodes", 0) for r in kept]),
        "rmsd": _percentiles([r.target for r in kept]),
        "totals": {key: int(sum(r.report.get(key, 0) for r in results)) for key in (
            "atoms_read", "hydrogens_skipped", "placeholder_atoms_removed", "altloc_atoms_skipped",
            "unexpected_atoms_removed",
            "terminal_atoms_dropped", "nucleotides_missing_backbone", "nucleotides_missing_base", "base_atoms_missing")},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    problems = []
    if fraction > config["qc"]["max_failed_structure_fraction"]:
        problems.append(f"{bad}/{len(results)} models failed or were dropped (limit "
                        f"{config['qc']['max_failed_structure_fraction']})")
    if incomplete > config["qc"]["max_incomplete_base_fraction"]:
        problems.append(f"{incomplete:.3f} of the models have an incomplete base (limit "
                        f"{config['qc']['max_incomplete_base_fraction']})")
    if len(per_puzzle_kept) != int(config["expected_puzzles"][split]):
        problems.append(f"{len(per_puzzle_kept)} puzzles, expected {config['expected_puzzles'][split]}")
    lost = sorted(set(per_puzzle_seen) - set(per_puzzle_kept))
    if lost:
        problems.append(f"puzzles with no usable model: {lost}")
    return summary, problems


def build_processed(config: dict, *, splits: Sequence[str], workers: int = 1, chunk: int = 250) -> int:
    """Process the requested splits; return the exit code (0 ok, 1 QC failure)."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    out_dir = processed_dir(config)
    schema = rg.edge_free_schema({"builder": "downstream.tasks.rsr.dataset", "source": config["source"],
                                  "revision": str(config["processed_revision"]), "missing_atoms": config["missing_atoms"]})
    settings = processing_settings(config)
    sources = lmdb_dirs(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    puzzles_by_split: Dict[str, set] = {}
    for split in splits:
        target_path = out_dir / f"{split}.pt"
        if target_path.exists():
            raise FileExistsError(f"refusing to overwrite {target_path}")
        source = sources[split]
        expected = int(config["expected_counts"][split])
        length = lmdb_length(source)
        if length != expected:
            raise ValueError(f"{source}: {length} entries, task.yaml expects {expected} for {split}")
        jobs = [(str(source), start, min(start + chunk, length), settings) for start in range(0, length, chunk)]
        results: List[DecoyResult] = []
        if workers == 1:
            for job in jobs:
                results += _process_range(job)
                print(f"[{split}] {len(results)}/{length} models", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for part in pool.map(_process_range, jobs):
                    results += part
                    print(f"[{split}] {len(results)}/{length} models", flush=True)
        if len(results) != expected:
            raise ValueError(f"{source}: {len(results)} results for {expected} expected models of {split}")
        summary, problems = summarize_split(config, split, results)
        kept = [r for r in results if r.status == "ok"]
        puzzles_by_split[split] = {r.puzzle for r in kept}
        with open(out_dir / f"qc_{split}.tsv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QC_COLUMNS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for result in results:
                writer.writerow({"decoy": result.decoy_id, "split": split, "status": result.status,
                                 "reason": result.reason, "puzzle": result.puzzle, "source_id": result.source_id,
                                 "rmsd": result.target, **result.report})
        (out_dir / f"qc_{split}.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps({k: v for k, v in summary.items() if not k.startswith("models_per")}, indent=1), flush=True)
        if problems:
            print(f"[{split}] QC FAILED: {'; '.join(problems)}; {target_path} not written", flush=True)
            exit_code = 1
            continue
        payload = {
            "schema": schema, "split": split, "revision": str(config["processed_revision"]),
            "ids": [r.decoy_id for r in kept], "source_ids": [r.source_id for r in kept],
            "puzzles": [r.puzzle for r in kept],
            "targets": torch.tensor([r.target for r in kept], dtype=torch.float32),
            "samples": [r.sample for r in kept], "qc": summary,
        }
        partial = target_path.with_name(target_path.name + ".partial")
        torch.save(payload, partial)
        os.replace(partial, target_path)
        print(f"[{split}] wrote {target_path} ({len(kept)} models, {len(puzzles_by_split[split])} puzzles)", flush=True)
    names = list(puzzles_by_split)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            shared = puzzles_by_split[first] & puzzles_by_split[second]
            if shared:
                print(f"SPLIT LEAK: puzzles {sorted(shared)} occur in both {first} and {second}", flush=True)
                exit_code = 1
    if len(names) < len(config["splits"]):
        print(f"note: only {names} were processed, so puzzle overlap between splits was checked among those", flush=True)
    return exit_code


# ---------------------------------------------------------------------------
# Processed split access
# ---------------------------------------------------------------------------

def group_labels(chains: Sequence[str]) -> np.ndarray:
    """Puzzle of each model id (``<puzzle>/<index>``); an id without the separator raises."""
    labels = []
    for chain in chains:
        puzzle, separator, _ = str(chain).partition(ID_SEPARATOR)
        if not separator or not puzzle:
            raise ValueError(f"model id {chain!r} is not '<puzzle>{ID_SEPARATOR}<index>'")
        labels.append(puzzle)
    return np.asarray(labels)


class ProcessedSplit:
    """Models, RMSD targets and stored RNA graphs of one processed split (one runtime graph per model)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        for key in ("schema", "ids", "targets", "samples", "puzzles", "source_ids"):
            if key not in payload:
                raise ValueError(f"{self.path}: processed split lacks {key!r}")
        self.payload = payload
        self.schema = payload["schema"]
        self.ids: List[str] = list(payload["ids"])
        self.chains = self.ids            # embed.py writes this column
        self.targets: np.ndarray = payload["targets"].numpy().astype(np.float32)
        counts = {len(self.ids), len(payload["samples"]), len(payload["puzzles"]), len(payload["source_ids"]),
                  int(self.targets.shape[0])}
        if len(counts) != 1:
            raise ValueError(f"{self.path}: inconsistent model counts {counts}")
        self.groups = group_labels(self.ids)
        if list(self.groups) != list(payload["puzzles"]):
            raise ValueError(f"{self.path}: model ids and stored puzzles disagree")
        self.labels = self.targets.reshape(-1, 1)
        self.sample_chain = torch.arange(len(self))
        self.report = payload.get("qc")

    def __len__(self) -> int:
        return len(self.ids)

    def runtime(self, index: int, model_config, *, seed: int):
        """The decoded graph; RNA graphs have no random edges, so ``seed`` changes nothing."""
        return rg.runtime_sample(self.payload["samples"][index], self.schema, model_config)

    def iter_runtime(self, model_config, *, seed: int) -> Iterator:
        for index in range(len(self)):
            yield index, self.runtime(index, model_config, seed=seed)

    def chain_embeddings(self, sample_embeddings: np.ndarray) -> np.ndarray:
        if sample_embeddings.shape[0] != len(self):
            raise ValueError(f"{self.path}: {sample_embeddings.shape[0]} embeddings for {len(self)} models")
        return sample_embeddings.astype(np.float32)
