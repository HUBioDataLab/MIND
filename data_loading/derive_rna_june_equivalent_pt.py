#!/usr/bin/env python3
"""Derive immutable RNA PyG chunks from a reviewed ID manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate

try:
    from data_loading.create_chunk_metadata import assign_bucket
    from data_loading.build_rna_june_equivalent_manifest import real_non_hydrogen_count
except ModuleNotFoundError:
    from create_chunk_metadata import assign_bucket
    from build_rna_june_equivalent_manifest import real_non_hydrogen_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def read_selection_manifest(
    path: Path,
    *,
    expected_selected_count: int | None,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "name",
            "real_non_hydrogen_atom_count",
            "num_nodes",
            "source_chunk",
            "source_idx",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Selection manifest lacks columns: {sorted(missing)}")
        rows = list(reader)

    names = [row["name"].strip() for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Selection manifest contains duplicate molecule IDs")
    if expected_selected_count is not None and len(rows) != expected_selected_count:
        raise ValueError(
            f"Expected {expected_selected_count} selected rows, found {len(rows)}"
        )
    selected = {
        row["name"].strip(): {
            "real_non_hydrogen_atom_count": int(row["real_non_hydrogen_atom_count"]),
            "num_nodes": int(row["num_nodes"]),
            "source_chunk": int(row["source_chunk"]),
            "source_idx": int(row["source_idx"]),
        }
        for row in rows
    }
    if any(
        not 0 < record["real_non_hydrogen_atom_count"] <= 5000
        for record in selected.values()
    ):
        raise ValueError("Selection manifest contains a real atom count outside 1..5000")
    return selected, {"selected_rows": len(rows)}


def _validate_graph(sample: Any, *, expected_nodes: int, expected_edges: int) -> None:
    required = {"pos", "z", "edge_index", "block_idx", "entity_idx", "pos_code"}
    missing = sorted(required - set(sample.keys()))
    molecule_id = str(getattr(sample, "mol_id", "<unknown>"))
    if missing:
        raise ValueError(f"Graph {molecule_id} lacks fields: {missing}")
    if int(sample.num_nodes) != expected_nodes:
        raise ValueError(f"Node count mismatch for {molecule_id}")
    if sample.pos.shape != (expected_nodes, 3) or sample.z.numel() != expected_nodes:
        raise ValueError(f"Invalid node tensors for {molecule_id}")
    if sample.edge_index.ndim != 2 or sample.edge_index.shape != (2, expected_edges):
        raise ValueError(f"Invalid edge_index for {molecule_id}")
    if expected_edges:
        if int(sample.edge_index.min()) < 0 or int(sample.edge_index.max()) >= expected_nodes:
            raise ValueError(f"Out-of-range edge for {molecule_id}")
    if not bool(torch.isfinite(sample.pos).all()):
        raise ValueError(f"Non-finite coordinates for {molecule_id}")
    edge_attr = getattr(sample, "edge_attr", None)
    if edge_attr is not None:
        if edge_attr.shape[0] != expected_edges or not bool(torch.isfinite(edge_attr).all()):
            raise ValueError(f"Invalid edge_attr for {molecule_id}")


def validate_output_chunk(output_pt: Path, metadata: dict[str, Any]) -> None:
    data, slices = torch.load(
        output_pt, map_location="cpu", weights_only=False, mmap=True
    )
    graph_count = len(slices["pos"]) - 1
    if graph_count != metadata["num_samples"]:
        raise ValueError(f"Output graph count mismatch for {output_pt}")

    observed_ids = []
    total_nodes = total_edges = 0
    for sample_meta in metadata["samples"]:
        sample = separate(
            cls=data.__class__,
            batch=data,
            idx=int(sample_meta["idx"]),
            slice_dict=slices,
            inc_dict=None,
            decrement=False,
        )
        expected_nodes = int(sample_meta["num_atoms"])
        expected_edges = int(sample_meta["num_edges"])
        _validate_graph(
            sample, expected_nodes=expected_nodes, expected_edges=expected_edges
        )
        molecule_id = str(getattr(sample, "mol_id", ""))
        if molecule_id != sample_meta["mol_id"]:
            raise ValueError(f"Output molecule ID mismatch at {sample_meta['idx']}")
        observed_ids.append(molecule_id)
        total_nodes += expected_nodes
        total_edges += expected_edges
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError(f"Duplicate molecule IDs in output chunk: {output_pt}")
    if total_nodes != metadata["total_atoms"] or total_edges != metadata["total_edges"]:
        raise ValueError(f"Output totals mismatch for {output_pt}")


def derive_chunk(
    source_pt: Path,
    output_pt: Path,
    *,
    selected_records: dict[str, dict[str, int]],
    source_chunk_index: int,
    manifest_sha256: str,
) -> dict[str, Any]:
    source_metadata_path = _metadata_path(source_pt)
    source_metadata = json.loads(source_metadata_path.read_text())
    source_before = _stat_fingerprint(source_pt)
    data, slices = torch.load(
        source_pt, map_location="cpu", weights_only=False, mmap=True
    )
    graph_count = len(slices["pos"]) - 1
    if graph_count != source_metadata["num_samples"]:
        raise ValueError(f"Source graph count mismatch for {source_pt}")
    mol_ids = [str(value) for value in getattr(data, "mol_id", [])]
    if len(mol_ids) != graph_count:
        raise ValueError(f"Source mol_id count mismatch for {source_pt}")

    kept_graphs = []
    output_samples = []
    total_nodes = total_edges = 0
    for sample_meta in source_metadata["samples"]:
        source_index = int(sample_meta["idx"])
        molecule_id = mol_ids[source_index]
        if molecule_id not in selected_records:
            continue
        selection_record = selected_records[molecule_id]
        num_nodes = int(sample_meta["num_atoms"])
        num_edges = int(sample_meta["num_edges"])
        if selection_record["source_chunk"] != source_chunk_index:
            raise ValueError(f"Manifest source chunk mismatch for {molecule_id}")
        if selection_record["source_idx"] != source_index:
            raise ValueError(f"Manifest source index mismatch for {molecule_id}")
        if selection_record["num_nodes"] != num_nodes:
            raise ValueError(f"Manifest node count mismatch for {molecule_id}")
        sample = separate(
            cls=data.__class__,
            batch=data,
            idx=source_index,
            slice_dict=slices,
            inc_dict=None,
            decrement=False,
        )
        _validate_graph(sample, expected_nodes=num_nodes, expected_edges=num_edges)
        real_atom_count = real_non_hydrogen_count(sample.z)
        if real_atom_count != selection_record["real_non_hydrogen_atom_count"]:
            raise ValueError(f"Manifest real atom count mismatch for {molecule_id}")
        output_index = len(kept_graphs)
        kept_graphs.append(sample)
        output_samples.append(
            {
                "idx": output_index,
                "mol_id": molecule_id,
                "num_atoms": num_nodes,
                "num_edges": num_edges,
                "real_non_hydrogen_atom_count": real_atom_count,
                "bucket": assign_bucket(num_nodes),
                "source_chunk": source_chunk_index,
                "source_idx": source_index,
            }
        )
        total_nodes += num_nodes
        total_edges += num_edges

    if not kept_graphs:
        return {
            "source": source_before,
            "source_metadata_sha256": _sha256(source_metadata_path),
            "source_graphs": graph_count,
            "kept_graphs": 0,
            "kept_nodes": 0,
            "kept_edges": 0,
            "kept_ids": [],
            "output": None,
        }

    output_pt.parent.mkdir(parents=True, exist_ok=False)
    collated, output_slices, _ = collate(
        cls=kept_graphs[0].__class__,
        data_list=kept_graphs,
        increment=False,
        add_batch=False,
    )
    building_pt = output_pt.with_name(f".{output_pt.name}.building")
    torch.save((collated, output_slices), building_pt)
    building_pt.replace(output_pt)

    output_metadata = {
        "chunk_file": output_pt.name,
        "num_samples": len(kept_graphs),
        "dataset_type": "rna",
        "total_atoms": total_nodes,
        "total_edges": total_edges,
        "selection_policy": "membership in reviewed June-equivalent manifest",
        "selection_manifest_sha256": manifest_sha256,
        "source_chunk_file": str(source_pt.resolve()),
        "samples": output_samples,
    }
    _metadata_path(output_pt).write_text(
        json.dumps(output_metadata, indent=2, sort_keys=True) + "\n"
    )
    validate_output_chunk(output_pt, output_metadata)
    if _stat_fingerprint(source_pt) != source_before:
        raise RuntimeError(f"Source PT changed while deriving chunk: {source_pt}")

    return {
        "source": source_before,
        "source_metadata_sha256": _sha256(source_metadata_path),
        "source_graphs": graph_count,
        "kept_graphs": len(kept_graphs),
        "kept_nodes": total_nodes,
        "kept_edges": total_edges,
        "kept_ids": [item["mol_id"] for item in output_samples],
        "output": {
            **_stat_fingerprint(output_pt),
            "sha256": _sha256(output_pt),
            "metadata_sha256": _sha256(_metadata_path(output_pt)),
        },
    }


def derive_all_chunks(
    *,
    source_base: Path,
    output_root: Path,
    manifest_path: Path,
    first_chunk: int = 0,
    last_chunk: int = 66,
    expected_selected_count: int | None = None,
) -> dict[str, Any]:
    if first_chunk < 0 or last_chunk < first_chunk:
        raise ValueError("chunk range must satisfy 0 <= first_chunk <= last_chunk")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output root: {output_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    selected_records, selection = read_selection_manifest(
        manifest_path,
        expected_selected_count=expected_selected_count,
    )
    selected_ids = set(selected_records)
    manifest_sha256 = _sha256(manifest_path)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f".{output_root.name}.building-{uuid.uuid4().hex}"
    staging_root.mkdir(mode=0o755)
    chunk_reports = []
    found_ids: set[str] = set()
    try:
        staged_base = staging_root / "processed_23slot"
        for chunk_index in range(first_chunk, last_chunk + 1):
            source_pt = _find_chunk_pt(source_base, chunk_index)
            output_dir = staging_root / f"{staged_base.name}_chunk_{chunk_index:03d}"
            output_pt = (
                output_dir
                / "processed"
                / f"rna_june_equivalent_v2_chunk_{chunk_index:03d}.pt"
            )
            print(f"Deriving chunk {chunk_index:03d} from {source_pt}", flush=True)
            chunk_report = derive_chunk(
                source_pt,
                output_pt,
                selected_records=selected_records,
                source_chunk_index=chunk_index,
                manifest_sha256=manifest_sha256,
            )
            duplicate_ids = found_ids & set(chunk_report["kept_ids"])
            if duplicate_ids:
                raise ValueError(f"IDs repeated across source chunks: {sorted(duplicate_ids)}")
            found_ids.update(chunk_report["kept_ids"])
            if chunk_report["output"] is not None:
                staged_output = Path(str(chunk_report["output"]["path"]))
                chunk_report["output"]["path"] = str(
                    output_root / staged_output.relative_to(staging_root.resolve())
                )
            chunk_reports.append(chunk_report)
            print(
                f"  kept={chunk_report['kept_graphs']} "
                f"nodes={chunk_report['kept_nodes']:,}",
                flush=True,
            )

        missing = sorted(selected_ids - found_ids)
        unexpected = sorted(found_ids - selected_ids)
        if missing or unexpected or len(found_ids) != len(selected_ids):
            raise ValueError(
                "Derived membership mismatch: "
                f"found={len(found_ids)} missing={missing[:20]} unexpected={unexpected[:20]}"
            )

        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy": "immutable ID-membership derivation; graph node count is not a filter",
            "source_base": str(source_base.resolve()),
            "output_chunk_base": str((output_root / "processed_23slot").resolve()),
            "selection_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_sha256,
                **selection,
            },
            "chunk_range": [first_chunk, last_chunk],
            "aggregate": {
                "source_graphs": sum(item["source_graphs"] for item in chunk_reports),
                "kept_graphs": sum(item["kept_graphs"] for item in chunk_reports),
                "kept_nodes": sum(item["kept_nodes"] for item in chunk_reports),
                "kept_edges": sum(item["kept_edges"] for item in chunk_reports),
                "unique_kept_ids": len(found_ids),
            },
            "chunks": chunk_reports,
            "go": True,
        }
        (staging_root / "derivation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        staging_root.replace(output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print(
        f"RNA_JUNE_DERIVED_PT=GO graphs={len(found_ids)} output_root={output_root}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--first-chunk", type=int, default=0)
    parser.add_argument("--last-chunk", type=int, default=66)
    parser.add_argument("--expected-selected-count", type=int)
    args = parser.parse_args()
    derive_all_chunks(**vars(args))


if __name__ == "__main__":
    main()
