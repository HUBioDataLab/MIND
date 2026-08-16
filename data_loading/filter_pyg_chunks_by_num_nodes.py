#!/usr/bin/env python3
"""Filter existing PyG chunks by graph node count without rebuilding graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate

try:
    from data_loading.create_chunk_metadata import assign_bucket
except ModuleNotFoundError:
    from create_chunk_metadata import assign_bucket


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


def _find_chunk_pt(base_path: Path, chunk_index: int) -> Path:
    chunk_dir = base_path.parent / f"{base_path.name}_chunk_{chunk_index:03d}"
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


def _metadata_path(pt_path: Path) -> Path:
    return pt_path.parent / f"{pt_path.stem}_metadata.json"


def _validate_graph(sample: Any, expected_nodes: int, expected_edges: int) -> None:
    required = {"pos", "z", "edge_index", "block_idx", "entity_idx", "pos_code"}
    missing = sorted(required - set(sample.keys()))
    if missing:
        raise ValueError(f"Graph {getattr(sample, 'mol_id', '<unknown>')} lacks {missing}")
    if int(sample.num_nodes) != expected_nodes:
        raise ValueError(
            f"Node metadata mismatch for {getattr(sample, 'mol_id', '<unknown>')}: "
            f"{sample.num_nodes} != {expected_nodes}"
        )
    if sample.pos.shape != (expected_nodes, 3) or sample.z.numel() != expected_nodes:
        raise ValueError(f"Invalid node tensor shapes for {getattr(sample, 'mol_id', '<unknown>')}")
    if sample.edge_index.ndim != 2 or sample.edge_index.shape[0] != 2:
        raise ValueError(f"Invalid edge_index shape for {getattr(sample, 'mol_id', '<unknown>')}")
    if sample.edge_index.shape[1] != expected_edges:
        raise ValueError(
            f"Edge metadata mismatch for {getattr(sample, 'mol_id', '<unknown>')}: "
            f"{sample.edge_index.shape[1]} != {expected_edges}"
        )
    if expected_edges:
        edge_min = int(sample.edge_index.min())
        edge_max = int(sample.edge_index.max())
        if edge_min < 0 or edge_max >= expected_nodes:
            raise ValueError(
                f"Out-of-range edge for {getattr(sample, 'mol_id', '<unknown>')}: "
                f"[{edge_min}, {edge_max}] with {expected_nodes} nodes"
            )
    if not bool(torch.isfinite(sample.pos).all()):
        raise ValueError(f"Non-finite coordinates in {getattr(sample, 'mol_id', '<unknown>')}")
    if hasattr(sample, "edge_attr") and sample.edge_attr is not None:
        if sample.edge_attr.shape[0] != expected_edges:
            raise ValueError(f"edge_attr length mismatch for {getattr(sample, 'mol_id', '<unknown>')}")
        if not bool(torch.isfinite(sample.edge_attr).all()):
            raise ValueError(f"Non-finite edge_attr in {getattr(sample, 'mol_id', '<unknown>')}")


def validate_output_chunk(output_pt: Path, metadata: dict[str, Any]) -> None:
    """Reload a filtered chunk and verify every serialized graph."""
    data, slices = torch.load(
        output_pt, map_location="cpu", weights_only=False, mmap=True
    )
    graph_count = len(slices["pos"]) - 1
    if graph_count != metadata["num_samples"]:
        raise ValueError(
            f"Output metadata count mismatch for {output_pt}: "
            f"{metadata['num_samples']} != {graph_count}"
        )

    total_nodes = 0
    total_edges = 0
    limit = int(metadata["max_graph_atoms"])
    for sample_meta in metadata["samples"]:
        output_index = int(sample_meta["idx"])
        sample = separate(
            cls=data.__class__,
            batch=data,
            idx=output_index,
            slice_dict=slices,
            inc_dict=None,
            decrement=False,
        )
        num_nodes = int(sample_meta["num_atoms"])
        num_edges = int(sample_meta["num_edges"])
        _validate_graph(sample, num_nodes, num_edges)
        if num_nodes > limit:
            raise ValueError(
                f"Output graph exceeds max_graph_atoms: "
                f"{sample_meta.get('mol_id', '<unknown>')} has {num_nodes} > {limit}"
            )
        if str(getattr(sample, "mol_id", "")) != str(sample_meta["mol_id"]):
            raise ValueError(f"Molecule ID mismatch at output index {output_index}")
        total_nodes += num_nodes
        total_edges += num_edges

    if total_nodes != metadata["total_atoms"]:
        raise ValueError(f"Output node total mismatch for {output_pt}")
    if total_edges != metadata["total_edges"]:
        raise ValueError(f"Output edge total mismatch for {output_pt}")


def filter_chunk(
    source_pt: Path,
    output_pt: Path,
    *,
    max_nodes: int,
    source_chunk_index: int,
) -> dict[str, Any]:
    source_metadata_path = _metadata_path(source_pt)
    source_metadata = json.loads(source_metadata_path.read_text())
    source_before = _stat_fingerprint(source_pt)
    source_sha256 = _sha256(source_pt)

    data, slices = torch.load(
        source_pt, map_location="cpu", weights_only=False, mmap=True
    )
    graph_count = len(slices["pos"]) - 1
    if graph_count != source_metadata["num_samples"]:
        raise ValueError(
            f"Source metadata count mismatch for {source_pt}: "
            f"{source_metadata['num_samples']} != {graph_count}"
        )

    kept_graphs = []
    kept_records = []
    dropped_records = []
    output_samples = []
    total_nodes = 0
    total_edges = 0

    mol_ids = getattr(data, "mol_id", [None] * graph_count)
    for sample_meta in source_metadata["samples"]:
        source_index = int(sample_meta["idx"])
        num_nodes = int(sample_meta["num_atoms"])
        num_edges = int(sample_meta["num_edges"])
        mol_id = str(mol_ids[source_index])
        record = {
            "mol_id": mol_id,
            "source_chunk": source_chunk_index,
            "source_index": source_index,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
        }
        if num_nodes > max_nodes:
            dropped_records.append(record)
            continue

        sample = separate(
            cls=data.__class__,
            batch=data,
            idx=source_index,
            slice_dict=slices,
            inc_dict=None,
            decrement=False,
        )
        _validate_graph(sample, num_nodes, num_edges)
        output_index = len(kept_graphs)
        kept_graphs.append(sample)
        kept_records.append({**record, "output_index": output_index})
        output_samples.append(
            {
                "idx": output_index,
                "num_atoms": num_nodes,
                "num_edges": num_edges,
                "bucket": assign_bucket(num_nodes),
                "mol_id": mol_id,
                "source_chunk": source_chunk_index,
                "source_idx": source_index,
            }
        )
        total_nodes += num_nodes
        total_edges += num_edges

    if not kept_graphs:
        raise ValueError(f"Filter removed every graph from {source_pt}")

    output_pt.parent.mkdir(parents=True, exist_ok=False)
    collated, output_slices, _ = collate(
        cls=kept_graphs[0].__class__,
        data_list=kept_graphs,
        increment=False,
        add_batch=False,
    )
    temporary_pt = output_pt.with_suffix(".pt.tmp")
    torch.save((collated, output_slices), temporary_pt)
    temporary_pt.replace(output_pt)

    output_metadata = {
        "chunk_file": output_pt.name,
        "num_samples": len(kept_graphs),
        "dataset_type": "rna",
        "total_atoms": total_nodes,
        "total_edges": total_edges,
        "max_graph_atoms": max_nodes,
        "source_chunk_file": str(source_pt.resolve()),
        "samples": output_samples,
    }
    _metadata_path(output_pt).write_text(
        json.dumps(output_metadata, indent=2, sort_keys=True) + "\n"
    )
    validate_output_chunk(output_pt, output_metadata)

    source_after = _stat_fingerprint(source_pt)
    if source_before != source_after:
        raise AssertionError(f"Source PT changed while filtering: {source_pt}")

    return {
        "source": {**source_before, "sha256": source_sha256},
        "output": {
            **_stat_fingerprint(output_pt),
            "sha256": _sha256(output_pt),
        },
        "source_graphs": graph_count,
        "kept_graphs": len(kept_graphs),
        "dropped_graphs": len(dropped_records),
        "kept_nodes": total_nodes,
        "kept_edges": total_edges,
        "kept": kept_records,
        "dropped": dropped_records,
    }


def filter_chunks(
    source_base: Path,
    output_base: Path,
    *,
    first_chunk: int,
    last_chunk: int,
    max_nodes: int,
) -> dict[str, Any]:
    if first_chunk < 0 or last_chunk < first_chunk:
        raise ValueError("chunk range must satisfy 0 <= first_chunk <= last_chunk")
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    if source_base.resolve() == output_base.resolve():
        raise ValueError("source and output bases must differ")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    chunk_reports = []
    for chunk_index in range(first_chunk, last_chunk + 1):
        source_pt = _find_chunk_pt(source_base, chunk_index)
        output_dir = output_base.parent / f"{output_base.name}_chunk_{chunk_index:03d}"
        output_pt = output_dir / "processed" / f"rna_5k_chunk_{chunk_index:03d}.pt"
        print(
            f"Filtering chunk {chunk_index:03d}: {source_pt.name} -> {output_pt}",
            flush=True,
        )
        report = filter_chunk(
            source_pt,
            output_pt,
            max_nodes=max_nodes,
            source_chunk_index=chunk_index,
        )
        chunk_reports.append(report)
        print(
            f"  kept={report['kept_graphs']} dropped={report['dropped_graphs']} "
            f"nodes={report['kept_nodes']:,}",
            flush=True,
        )

    aggregate = {
        "source_graphs": sum(item["source_graphs"] for item in chunk_reports),
        "kept_graphs": sum(item["kept_graphs"] for item in chunk_reports),
        "dropped_graphs": sum(item["dropped_graphs"] for item in chunk_reports),
        "kept_nodes": sum(item["kept_nodes"] for item in chunk_reports),
        "kept_edges": sum(item["kept_edges"] for item in chunk_reports),
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": "keep graphs with num_nodes <= max_nodes",
        "source_base": str(source_base.resolve()),
        "output_base": str(output_base.resolve()),
        "chunk_range": [first_chunk, last_chunk],
        "max_nodes": max_nodes,
        "aggregate": aggregate,
        "chunks": chunk_reports,
    }
    report_path = output_base.parent / f"{output_base.name}_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"FILTER_PYG_CHUNKS=GO manifest={report_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-base", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--first-chunk", type=int, required=True)
    parser.add_argument("--last-chunk", type=int, required=True)
    parser.add_argument("--max-nodes", type=int, required=True)
    args = parser.parse_args()
    filter_chunks(
        args.source_base,
        args.output_base,
        first_chunk=args.first_chunk,
        last_chunk=args.last_chunk,
        max_nodes=args.max_nodes,
    )


if __name__ == "__main__":
    main()
