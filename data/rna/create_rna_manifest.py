#!/usr/bin/env python3
"""Create the RNA manifest for ``make build-edge-free DOMAIN=rna`` from a prepared RNA3DB release.

Input is the directory written by ``data/rna/download_rna3db.py`` (``raw_structures/``,
``rna3db-jsons/split.json`` and ``download_provenance.json``). Chains are selected in two stages.

1. RNA3DB chain filters, with the defaults of the release code (``rna3db/filter.py``):
   sequence length >= 32 nt, resolution <= 9.0 A (NMR has infinite resolution and is
   therefore rejected), no single nucleotide above 80 % of the sequence and no more than
   30 % ``N``. The chain selection itself (every ``polyribonucleotide`` chain, DNA/RNA
   hybrids included, pure DNA excluded) is made by RNA3DB before ``split.json`` is
   written and cannot be re-checked from it. A published ``split.json`` already passed
   these filters, so on an untouched release they reject nothing; they matter when the
   thresholds are tightened and they guard against a damaged ``split.json``.
2. The builder's own acceptance test: every kept chain is converted by the production
   worker of ``data_loading/build_edge_free_chunks.py``. A chain it rejects (for example a
   P-only phosphate trace) is dropped here with its reason, because each builder failure
   counts against the per-chunk failure limit. Any other conversion error is fatal.
   Chains the worker accepts are then checked for mirror-image (left-handed) chirality with the
   criterion of ``data_loading/audit_rna_anchor_selection.py``: in every block with an exact
   12-atom backbone, the sugar O2' is placed relative to the frame built from C2', C3', C4'; a
   molecule is a mirror image when more than half of those blocks have O2' on the non-positive
   side. Mirror-image chains are dropped and counted.

Outputs (never overwritten): the manifest CSV (column ``name`` is the structure id the
builder reads; rows are sorted by name, so the file is deterministic for one release and
set of thresholds) and ``<manifest stem>_provenance.json`` with the inputs, thresholds,
counts per rejection reason and the rejected chains.

Example (after the download step)::

    python data/rna/create_rna_manifest.py --prepared-dir <dir> --workers 24

With ``--verify-existing`` an already finished manifest is re-verified against the current inputs
(release, split.json digest, raw file count, thresholds, manifest digest and chain set) and the
run exits 0 without writing; any mismatch is an error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MANIFEST_COLUMNS = (
    "name", "pdb_id", "chain", "rna3db_set", "release_date", "structure_method", "resolution", "length",
)
# Builder rejections that are properties of the structure; any other reason is an error.
MIRROR_REASON = "mirror-image (left-handed) molecule"
EXPECTED_BUILDER_REASONS = (
    "no parsable residues", "contains a P-only nucleotide block", "no atoms", MIRROR_REASON,
)
MIRROR_EPSILON = 1.0e-6
REQUIRED_META = ("release_date", "structure_method", "resolution", "sequence")


class ManifestError(RuntimeError):
    """Inconsistent inputs or an unexpected conversion failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split(split_json: Path) -> dict[str, dict]:
    """Return ``{chain name: metadata + rna3db_set}`` from a release ``split.json``."""
    with open(split_json) as handle:
        split = json.load(handle)
    chains: dict[str, dict] = {}
    try:
        for set_name, components in split.items():
            for component in components.values():
                for representative in component.values():
                    for name, meta in representative.items():
                        if name in chains:
                            raise ManifestError(f"chain {name!r} occurs twice in {split_json}")
                        missing = [key for key in REQUIRED_META if key not in meta]
                        if missing:
                            raise ManifestError(f"chain {name!r} lacks {missing} in {split_json}")
                        resolution = meta["resolution"]
                        if isinstance(resolution, bool) or not isinstance(resolution, (int, float)):
                            raise ManifestError(f"chain {name!r} has a non-numeric resolution {resolution!r}")
                        if not isinstance(meta["sequence"], str) or not meta["sequence"]:
                            raise ManifestError(f"chain {name!r} has no sequence")
                        chains[name] = dict(meta, rna3db_set=set_name)
    except (AttributeError, TypeError) as error:
        raise ManifestError(
            f"{split_json} does not have the set/component/representative/chain layout: {error}"
        ) from error
    if not chains:
        raise ManifestError(f"{split_json} lists no chains")
    return chains


def rna3db_filter_reasons(
    meta: dict, *, min_length: int, max_resolution: float,
    single_ratio_cutoff: float, max_unknown_ratio: float,
) -> list[str]:
    """Every RNA3DB filter a chain fails, in the order of ``rna3db/filter.py``."""
    sequence = meta["sequence"]
    length = len(sequence)
    reasons = []
    if length < min_length:
        reasons.append("length_below_minimum")
    if meta["resolution"] > max_resolution or math.isnan(meta["resolution"]):
        reasons.append("resolution_above_maximum")
    if max(sequence.count(nt) for nt in set(sequence)) / length > single_ratio_cutoff:
        reasons.append("single_nucleotide_ratio")
    if sequence.count("N") / length > max_unknown_ratio:
        reasons.append("unknown_nucleotide_ratio")
    return reasons


# ---------------------------------------------------------------------------
# Builder acceptance test
# ---------------------------------------------------------------------------

def _init_check_worker() -> None:
    from data_loading import build_edge_free_chunks as builder

    builder._init_worker("rna")


def mirror_image_reason(molecule) -> Optional[str]:
    """``MIRROR_REASON`` when the molecule is a left-handed (mirror-image) RNA, else ``None``.

    Same criterion and thresholds as ``audit_rna_anchor_selection`` (chirality policy), applied
    to the adapter's blocks: the signed height of O2' above the C2'/C3'/C4' frame, counted over
    blocks with an exact backbone and a defined frame.
    """
    from data_loading.atom_position_vocab import rna_backbone_contract_violations
    from data_loading.audit_rna_anchor_selection import (
        PROVISIONAL_CHIRALITY_ANCHORS,
        build_candidate_frame,
        is_mirror_rna_molecule,
    )

    exact = nonpositive = 0
    for block in molecule.blocks:
        missing, duplicates = rna_backbone_contract_violations(atom.pos_code for atom in block.atoms)
        if missing or duplicates:
            continue
        try:
            frame = build_candidate_frame(block.atoms, PROVISIONAL_CHIRALITY_ANCHORS, epsilon=MIRROR_EPSILON)
        except ValueError:
            continue  # degenerate anchors (coincident or collinear): chirality undefined, block not counted
        exact += 1
        nonpositive += int(frame.orientation_reference_signed_z <= MIRROR_EPSILON)
    return MIRROR_REASON if is_mirror_rna_molecule(nonpositive, exact) else None


def _check_one(path: str) -> tuple[str, Optional[str]]:
    """Run the production node worker, then the chirality check, on one file; samples are discarded."""
    from data_loading import build_edge_free_chunks as builder

    structure_id, _sample, reason = builder.build_node_sample(path)
    if reason is not None:
        return structure_id, reason
    try:
        molecule = builder._WORKER[1].convert_to_universal(Path(path))
        return structure_id, mirror_image_reason(molecule)
    except Exception as exc:  # reported by the caller as an unexpected, fatal reason
        return structure_id, f"{type(exc).__name__}: {exc}"


def builder_rejections(
    paths: list[Path], *, workers: int, log: Callable[[str], None] = print, progress_seconds: float = 30.0,
) -> dict[str, str]:
    """Map the id of every chain the builder worker rejects to its reason."""
    rejected: dict[str, str] = {}
    started = time.monotonic()
    last = started

    def consume(results) -> None:
        nonlocal last
        for done, (structure_id, reason) in enumerate(results, start=1):
            if reason is not None:
                rejected[structure_id] = reason
            now = time.monotonic()
            if now - last >= progress_seconds:
                last = now
                log(f"  builder check {done:,}/{len(paths):,} | rejected so far {len(rejected)} | "
                    f"elapsed {now - started:.0f}s")

    strings = [str(path) for path in paths]
    if workers <= 1:
        _init_check_worker()
        consume(map(_check_one, strings))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_check_worker) as executor:
            consume(executor.map(_check_one, strings, chunksize=4))
    return rejected


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as handle:
        handle.write(text)
    os.replace(tmp, path)


def load_inputs(prepared_dir: Path) -> tuple[dict, Path, dict[str, dict], Path, set[str]]:
    """Validate the download output; return ``(download record, split path, chains, raw dir, file stems)``."""
    download_record_path = prepared_dir / "download_provenance.json"
    if not download_record_path.exists():
        raise ManifestError(f"{download_record_path} missing; run data/rna/download_rna3db.py first")
    with open(download_record_path) as handle:
        download_record = json.load(handle)
    if download_record.get("status") != "complete":
        raise ManifestError(f"{download_record_path} reports status {download_record.get('status')!r}, not 'complete'")

    split_path = prepared_dir / "rna3db-jsons" / "split.json"
    chains = read_split(split_path)
    if len(chains) != download_record.get("split_json_chains"):
        raise ManifestError(
            f"split.json has {len(chains)} chains but the download recorded {download_record.get('split_json_chains')}"
        )
    raw_dir = prepared_dir / "raw_structures"
    on_disk = {path.stem for path in raw_dir.glob("*.cif")}
    missing = sorted(set(chains) - on_disk)
    unlisted = sorted(on_disk - set(chains))
    if missing or unlisted:
        raise ManifestError(
            f"{raw_dir} and split.json disagree: {len(missing)} chains without a file (first {missing[:5]}), "
            f"{len(unlisted)} files not in split.json (first {unlisted[:5]})"
        )
    return download_record, split_path, chains, raw_dir, on_disk


def recorded_filters(thresholds: dict) -> dict:
    """The filter record stored in the provenance: RNA3DB thresholds plus the mirror-image criterion."""
    from data_loading.audit_rna_anchor_selection import (
        MIRROR_MOLECULE_NONPOSITIVE_FRACTION_THRESHOLD,
        PROVISIONAL_CHIRALITY_ANCHORS,
    )

    return dict(
        thresholds,
        mirror_anchors=list(PROVISIONAL_CHIRALITY_ANCHORS),
        mirror_nonpositive_fraction=MIRROR_MOLECULE_NONPOSITIVE_FRACTION_THRESHOLD,
    )


def verify_existing_manifest(
    prepared_dir: Path, manifest_path: Path, thresholds: dict, *, log: Callable[[str], None] = print,
) -> dict:
    """Check a finished manifest against the current inputs; raise on any mismatch, never write."""
    provenance_path = manifest_path.with_name(manifest_path.stem + "_provenance.json")
    present = [path.exists() for path in (manifest_path, provenance_path)]
    if not all(present):
        raise ManifestError(f"{manifest_path} and {provenance_path} must both exist to be verified")
    download_record, split_path, chains, raw_dir, on_disk = load_inputs(prepared_dir)
    with open(provenance_path) as handle:
        record = json.load(handle)

    def require(label: str, recorded, current) -> None:
        if recorded != current:
            raise ManifestError(
                f"{provenance_path} does not match the current inputs: {label} recorded {recorded!r}, now {current!r}"
            )

    require("release", record.get("release"), download_record["release"])
    require("download archives", record.get("download_archives"), download_record["archives"])
    require("split.json sha256", record.get("inputs", {}).get("split_json_sha256"), sha256_file(split_path))
    require("raw file count", record.get("inputs", {}).get("raw_files"), len(on_disk))
    require("filters", record.get("filters"), recorded_filters(thresholds))
    require("manifest sha256", record.get("manifest", {}).get("sha256"), sha256_file(manifest_path))
    with open(manifest_path, newline="") as handle:
        listed = [row["name"] for row in csv.DictReader(handle)]
    expected = sorted(set(chains) - {entry["name"] for entry in record.get("rejected", [])})
    require("manifest chain set", listed, expected)
    log(f"manifest verified against current inputs: {len(listed):,} chains in {manifest_path}")
    return record


def create_manifest(
    prepared_dir: Path,
    manifest_path: Path,
    *,
    min_length: int = 32,
    max_resolution: float = 9.0,
    single_ratio_cutoff: float = 0.8,
    max_unknown_ratio: float = 0.3,
    workers: int = 1,
    check: Optional[Callable[[list[Path]], dict[str, str]]] = None,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> dict:
    """Write the manifest and its provenance; return the provenance record.

    ``check`` maps raw files to ``{id: reason}`` for chains the builder would reject; it
    defaults to the production worker run on ``workers`` processes.
    """
    provenance_path = manifest_path.with_name(manifest_path.stem + "_provenance.json")
    for path in (manifest_path, provenance_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if min_length < 1 or max_resolution <= 0 or not 0 < single_ratio_cutoff <= 1 or not 0 <= max_unknown_ratio <= 1:
        raise ValueError("invalid filter thresholds")

    download_record, split_path, chains, raw_dir, on_disk = load_inputs(prepared_dir)

    thresholds = {
        "min_length": min_length, "max_resolution": max_resolution,
        "single_ratio_cutoff": single_ratio_cutoff, "max_unknown_ratio": max_unknown_ratio,
    }
    names = sorted(chains)
    rejected: dict[str, str] = {}
    filter_hits: Counter = Counter()
    for name in names:
        reasons = rna3db_filter_reasons(chains[name], **thresholds)
        filter_hits.update(reasons)
        if reasons:
            rejected[name] = reasons[0]
    after_filters = [name for name in names if name not in rejected]
    log(f"RNA3DB filters: {len(after_filters):,} of {len(names):,} chains kept")

    log(f"builder acceptance test on {len(after_filters):,} chains ({workers} workers)")
    paths = [raw_dir / f"{name}.cif" for name in after_filters]
    builder_result = check(paths) if check is not None else builder_rejections(paths, workers=workers, log=log)
    unexpected = {name: reason for name, reason in builder_result.items() if reason not in EXPECTED_BUILDER_REASONS}
    if unexpected:
        first = dict(sorted(unexpected.items())[:3])
        raise ManifestError(f"{len(unexpected)} chains failed conversion for an unexpected reason; first: {first}")
    unknown_ids = sorted(set(builder_result) - set(after_filters))
    if unknown_ids:
        raise ManifestError(f"builder check returned ids that were not checked: {unknown_ids[:5]}")
    for name, reason in builder_result.items():
        rejected[name] = "mirror_image" if reason == MIRROR_REASON else f"builder: {reason}"

    kept = [name for name in names if name not in rejected]
    if not kept:
        raise ManifestError("no chain survived the filters")

    rows = []
    for name in kept:
        meta = chains[name]
        entry, _, chain = name.partition("_")
        rows.append({
            "name": name, "pdb_id": entry, "chain": chain, "rna3db_set": meta["rna3db_set"],
            "release_date": meta["release_date"], "structure_method": meta["structure_method"],
            "resolution": meta["resolution"], "length": len(meta["sequence"]),
        })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(manifest_path, buffer.getvalue())

    record = {
        "tool": "data/rna/create_rna_manifest.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release": download_record["release"],
        "release_tag": download_record["tag"],
        "download_archives": download_record["archives"],
        "inputs": {
            "split_json_sha256": sha256_file(split_path),
            "raw_dir": str(raw_dir),
            "raw_files": len(on_disk),
        },
        "chain_selection": (
            "RNA3DB keeps every chain whose _entity_poly.type contains 'polyribonucleotide' "
            "(DNA/RNA hybrids included, pure DNA excluded); applied upstream, not re-checked here"
        ),
        "filters": recorded_filters(thresholds),
        "counts": {
            "mirror_image_chains": sum(reason == "mirror_image" for reason in rejected.values()),
            "release_chains": len(names),
            "manifest_chains": len(kept),
            "rejected_total": len(rejected),
            "rejected_by_first_reason": dict(sorted(Counter(rejected.values()).items())),
            "rna3db_filter_hits_independent": dict(sorted(filter_hits.items())),
            "manifest_chains_per_rna3db_set": dict(sorted(Counter(chains[n]["rna3db_set"] for n in kept).items())),
            "manifest_entries": len({n.partition("_")[0] for n in kept}),
        },
        "rejected": [{"name": name, "reason": rejected[name]} for name in sorted(rejected)],
        "manifest": {"file": manifest_path.name, "sha256": sha256_file(manifest_path), "columns": list(MANIFEST_COLUMNS)},
    }
    write_text_atomic(provenance_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    log(f"manifest: {len(kept):,} chains -> {manifest_path}")
    log(f"rejected: {dict(sorted(Counter(rejected.values()).items()))}")
    return record


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", required=True, type=Path,
                        help="directory written by data/rna/download_rna3db.py")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="output CSV (default: <prepared-dir>/manifest_rna.csv); must not exist")
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--max-resolution", type=float, default=9.0)
    parser.add_argument("--single-ratio-cutoff", type=float, default=0.8)
    parser.add_argument("--max-unknown-ratio", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=8, help="processes for the builder acceptance test")
    parser.add_argument("--verify-existing", action="store_true",
                        help="if the manifest and its provenance already exist, verify them against the "
                             "current inputs and exit 0 without writing (any mismatch is an error)")
    args = parser.parse_args(argv)
    manifest = args.manifest or args.prepared_dir / "manifest_rna.csv"
    thresholds = {
        "min_length": args.min_length, "max_resolution": args.max_resolution,
        "single_ratio_cutoff": args.single_ratio_cutoff, "max_unknown_ratio": args.max_unknown_ratio,
    }
    provenance = manifest.with_name(manifest.stem + "_provenance.json")
    if args.verify_existing and (manifest.exists() or provenance.exists()):
        verify_existing_manifest(args.prepared_dir, manifest, thresholds)
        return 0
    create_manifest(args.prepared_dir, manifest, workers=args.workers, **thresholds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
