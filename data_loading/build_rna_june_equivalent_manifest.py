#!/usr/bin/env python3
"""Build an immutable June-policy-equivalent manifest from current RNA PTs.

The historical policy counts real non-hydrogen atoms. Current RNA graphs are
padded with virtual nodes, so neither ``num_nodes`` nor nucleotide length is an
equivalent filter. This audit reads current tensors without rewriting them and
records stable source locations for every selected molecule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

try:
    from data_loading.atom_position_vocab import VIRTUAL_ATOMIC_NUMBER
except ModuleNotFoundError:
    from atom_position_vocab import VIRTUAL_ATOMIC_NUMBER


HYDROGEN_ATOMIC_NUMBER = 1
DEFAULT_RAW_ONLY_IDS = ("4r8i_B", "4wb3_D", "4wb3_E")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def _metadata_path(pt_path: Path) -> Path:
    return pt_path.parent / f"{pt_path.stem}_metadata.json"


def _find_chunk_pt(source_base: Path, chunk_index: int) -> Path:
    chunk_dir = source_base.parent / f"{source_base.name}_chunk_{chunk_index:03d}"
    candidates = sorted(
        path
        for path in (chunk_dir / "processed").glob("*.pt")
        if not path.name.startswith("pre_")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one PT file in {chunk_dir / 'processed'}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _read_unique_names(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "name" not in reader.fieldnames:
            raise ValueError(f"Manifest must contain a 'name' column: {path}")
        names = [str(row["name"]).strip() for row in reader]
    if any(not name for name in names):
        raise ValueError(f"Manifest contains an empty name: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Manifest contains duplicate names: {path}")
    return names


def _expect(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, observed {actual!r}")


def real_non_hydrogen_count(atomic_numbers: torch.Tensor) -> int:
    """Count real non-H nodes in a current padded graph."""
    if atomic_numbers.ndim != 1:
        raise ValueError(f"Expected a 1-D atomic-number tensor, got {atomic_numbers.shape}")
    real_non_h = (atomic_numbers != HYDROGEN_ATOMIC_NUMBER) & (
        atomic_numbers != VIRTUAL_ATOMIC_NUMBER
    )
    return int(torch.count_nonzero(real_non_h))


def scan_current_chunks(
    *,
    source_base: Path,
    first_chunk: int,
    last_chunk: int,
    max_real_non_h_atoms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit current PT chunks and partition graphs under the June predicate."""
    if first_chunk < 0 or last_chunk < first_chunk:
        raise ValueError("chunk range must satisfy 0 <= first_chunk <= last_chunk")
    if max_real_non_h_atoms <= 0:
        raise ValueError("max_real_non_h_atoms must be positive")

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    chunk_reports: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for chunk_index in range(first_chunk, last_chunk + 1):
        source_pt = _find_chunk_pt(source_base, chunk_index)
        metadata_path = _metadata_path(source_pt)
        metadata = json.loads(metadata_path.read_text())
        source_before = _stat_fingerprint(source_pt)
        data, slices = torch.load(
            source_pt, map_location="cpu", weights_only=False, mmap=True
        )
        graph_count = len(slices["z"]) - 1
        if graph_count != int(metadata["num_samples"]):
            raise ValueError(f"Source graph count mismatch for {source_pt}")
        mol_ids = [str(value) for value in getattr(data, "mol_id", [])]
        if len(mol_ids) != graph_count:
            raise ValueError(f"Source mol_id count mismatch for {source_pt}")

        chunk_selected = chunk_rejected = 0
        for sample_meta in metadata["samples"]:
            source_index = int(sample_meta["idx"])
            if source_index < 0 or source_index >= graph_count:
                raise ValueError(f"Invalid source index {source_index} in {metadata_path}")
            molecule_id = mol_ids[source_index]
            if molecule_id in seen_ids:
                raise ValueError(f"Duplicate current molecule ID across chunks: {molecule_id}")
            seen_ids.add(molecule_id)

            start = int(slices["z"][source_index])
            end = int(slices["z"][source_index + 1])
            num_nodes = int(sample_meta["num_atoms"])
            if end - start != num_nodes:
                raise ValueError(f"Node slice mismatch for {molecule_id}")
            atom_count = real_non_hydrogen_count(data.z[start:end])
            record = {
                "name": molecule_id,
                "real_non_hydrogen_atom_count": atom_count,
                "num_nodes": num_nodes,
                "source_chunk": chunk_index,
                "source_idx": source_index,
            }
            if 0 < atom_count <= max_real_non_h_atoms:
                selected.append(record)
                chunk_selected += 1
            else:
                rejected.append(
                    {
                        **record,
                        "reason": (
                            "no_real_non_hydrogen_atoms"
                            if atom_count <= 0
                            else "above_real_non_hydrogen_atom_limit"
                        ),
                    }
                )
                chunk_rejected += 1

        if _stat_fingerprint(source_pt) != source_before:
            raise RuntimeError(f"Source PT changed while it was being audited: {source_pt}")
        chunk_reports.append(
            {
                "source": source_before,
                "source_metadata_sha256": _sha256(metadata_path),
                "graphs": graph_count,
                "selected": chunk_selected,
                "rejected": chunk_rejected,
            }
        )
        print(
            f"Audited chunk {chunk_index:03d}: graphs={graph_count} "
            f"selected={chunk_selected} rejected={chunk_rejected}",
            flush=True,
        )
        del data, slices

    return selected, rejected, chunk_reports


def build_manifest(
    *,
    source_base: Path,
    historical_raw_manifest: Path,
    legacy_filtered_dir: Path,
    current_manifest: Path,
    output_manifest: Path,
    output_report: Path,
    first_chunk: int = 0,
    last_chunk: int = 66,
    max_real_non_h_atoms: int = 5000,
    historical_june_graphs: int = 9783,
    expected_historical_raw: int = 12316,
    expected_current: int = 12869,
    expected_raw_intersection: int = 12313,
    expected_raw_only: int = 3,
    expected_current_only: int = 556,
    expected_stale_cifs: int = 567,
    expected_raw_only_ids: tuple[str, ...] = DEFAULT_RAW_ONLY_IDS,
) -> dict[str, Any]:
    """Build the current-population manifest under the exact June filter policy."""
    for path in (historical_raw_manifest, legacy_filtered_dir, current_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (output_manifest, output_report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable output: {path}")

    raw_names = _read_unique_names(historical_raw_manifest)
    current_names = _read_unique_names(current_manifest)
    raw_set = set(raw_names)
    current_set = set(current_names)
    _expect("historical raw manifest count", len(raw_names), expected_historical_raw)
    _expect("current manifest count", len(current_names), expected_current)
    _expect("raw/current intersection", len(raw_set & current_set), expected_raw_intersection)
    _expect("raw-only count", len(raw_set - current_set), expected_raw_only)
    _expect("current-only count", len(current_set - raw_set), expected_current_only)
    _expect("raw-only IDs", sorted(raw_set - current_set), sorted(expected_raw_only_ids))

    filtered_cifs = {path.stem for path in legacy_filtered_dir.glob("*.cif")}
    stale_cifs = sorted(filtered_cifs - raw_set)
    missing_filtered_cifs = sorted(raw_set - filtered_cifs)
    _expect("stale legacy filtered CIF count", len(stale_cifs), expected_stale_cifs)
    _expect("historical IDs missing from legacy filtered directory", missing_filtered_cifs, [])

    selected, rejected, chunk_reports = scan_current_chunks(
        source_base=source_base,
        first_chunk=first_chunk,
        last_chunk=last_chunk,
        max_real_non_h_atoms=max_real_non_h_atoms,
    )
    scanned_ids = {record["name"] for record in selected + rejected}
    _expect("PT/current manifest missing IDs", sorted(current_set - scanned_ids), [])
    _expect("PT IDs absent from current manifest", sorted(scanned_ids - current_set), [])

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = output_manifest.with_name(f".{output_manifest.name}.{os.getpid()}.building")
    report_tmp = output_report.with_name(f".{output_report.name}.{os.getpid()}.building")
    try:
        with manifest_tmp.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)

        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy": (
                "select current 23-slot RNA graphs with 0 < real non-H atom count "
                "<= 5000; exclude z=1 hydrogen and z=120 virtual nodes; do not use num_nodes"
            ),
            "source_base": str(source_base.resolve()),
            "chunk_range": [first_chunk, last_chunk],
            "max_real_non_h_atoms": max_real_non_h_atoms,
            "historical_june_graphs_comparison_only": historical_june_graphs,
            "counts": {
                "historical_raw_manifest": len(raw_names),
                "current_manifest": len(current_names),
                "raw_current_intersection": len(raw_set & current_set),
                "raw_only": len(raw_set - current_set),
                "current_only": len(current_set - raw_set),
                "legacy_filtered_stale_cifs": len(stale_cifs),
                "pt_graphs_scanned": len(scanned_ids),
                "selected": len(selected),
                "rejected": len(rejected),
                "selected_minus_historical_june": len(selected) - historical_june_graphs,
            },
            "raw_only_ids": sorted(raw_set - current_set),
            "legacy_filtered_stale_cif_sample": stale_cifs[:100],
            "rejected_graphs": rejected,
            "chunks": chunk_reports,
            "input_manifests": {
                "historical_raw": {
                    "path": str(historical_raw_manifest.resolve()),
                    "sha256": _sha256(historical_raw_manifest),
                },
                "current": {
                    "path": str(current_manifest.resolve()),
                    "sha256": _sha256(current_manifest),
                },
            },
            "output_manifest": {
                "path": str(output_manifest.resolve()),
                "sha256": _sha256(manifest_tmp),
            },
            "go_for_derived_pt": True,
        }
        report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest_tmp.replace(output_manifest)
        report_tmp.replace(output_report)
    finally:
        manifest_tmp.unlink(missing_ok=True)
        report_tmp.unlink(missing_ok=True)

    print(
        "RNA_JUNE_MANIFEST=GO "
        f"selected={len(selected)} historical_comparison={historical_june_graphs} "
        f"manifest={output_manifest}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-base", type=Path, required=True)
    parser.add_argument("--historical-raw-manifest", type=Path, required=True)
    parser.add_argument("--legacy-filtered-dir", type=Path, required=True)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--first-chunk", type=int, default=0)
    parser.add_argument("--last-chunk", type=int, default=66)
    parser.add_argument("--max-real-non-h-atoms", type=int, default=5000)
    parser.add_argument("--historical-june-graphs", type=int, default=9783)
    args = parser.parse_args()
    build_manifest(**vars(args))


if __name__ == "__main__":
    main()
