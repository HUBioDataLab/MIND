#!/usr/bin/env python3
"""Create a new PT cache with visible virtual coordinates/edges prepositioned."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.separate import separate

from core.masked_coordinate_corruption import (
    build_masked_coordinate_corruption,
    rebuild_template_incident_edges,
)
from core.masked_slot_mapping import flatten_pos_codes
from data_loading.atom_position_vocab import is_virtual_position_code
from data_loading.lazy_universal_dataset import normalize_legacy_atom_ids


def _fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def precompute_cache(
    input_path: Path,
    output_path: Path,
    *,
    domain: str,
    cutoff: float,
    max_neighbors: int,
    start_index: int = 0,
    end_index: int | None = None,
) -> dict:
    before = _fingerprint(input_path)
    data, slices = torch.load(
        input_path, map_location="cpu", weights_only=False, mmap=True
    )
    graph_count = len(slices["pos"]) - 1
    selected_end = graph_count if end_index is None else end_index
    if not 0 <= start_index < selected_end <= graph_count:
        raise ValueError(
            "graph range must satisfy "
            f"0 <= start_index < end_index <= {graph_count}; "
            f"got [{start_index}, {selected_end})"
        )
    selected_count = selected_end - start_index
    processed = []
    counts = Counter()
    quarantine_reasons = Counter()

    for graph_index in range(start_index, selected_end):
        sample = separate(
            cls=data.__class__,
            batch=data,
            idx=graph_index,
            slice_dict=slices,
            inc_dict=None,
            decrement=False,
        ).clone()
        sample = normalize_legacy_atom_ids(sample)
        original_pos = sample.pos.clone()
        original_dataset_type = getattr(sample, "dataset_type", domain)
        sample.dataset_type = domain
        sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long)
        sample.num_graphs = 1
        sample.mlm_mask = torch.zeros(sample.num_nodes, dtype=torch.bool)

        result = build_masked_coordinate_corruption(
            sample,
            domains=(domain,),
            relocate_context_virtuals=True,
        )
        if result.mapping_quarantines or result.coordinate_quarantines:
            raise AssertionError("an unmasked preprocessing sample produced target quarantine")

        codes = flatten_pos_codes(sample.pos_code, sample.num_nodes)
        original_virtual_mask = torch.tensor(
            [is_virtual_position_code(code) for code in codes], dtype=torch.bool
        )
        if bool((result.context_virtual_node_mask & ~original_virtual_mask).any()):
            raise AssertionError("visible-template preprocessing touched a real atom")
        if not torch.equal(
            result.input_pos[~result.context_virtual_node_mask],
            original_pos[~result.context_virtual_node_mask],
        ):
            raise AssertionError("visible-template preprocessing moved a non-V atom")

        rebuilt = rebuild_template_incident_edges(
            sample,
            result.input_pos,
            result.edge_rebuild_node_mask,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
        )
        sample.pos = result.input_pos
        sample.edge_index = rebuilt.edge_index
        if rebuilt.edge_attr is not None:
            sample.edge_attr = rebuilt.edge_attr

        if bool(result.context_virtual_node_mask.any()):
            new_distances = torch.linalg.vector_norm(
                sample.pos[sample.edge_index[0]] - sample.pos[sample.edge_index[1]],
                dim=1,
            )
            touches_moved = (
                result.context_virtual_node_mask[sample.edge_index[0]]
                | result.context_virtual_node_mask[sample.edge_index[1]]
            )
            if bool((touches_moved & (new_distances <= 1.0e-7)).any()):
                raise AssertionError("prepositioned V has a zero-distance edge")

        counts["graphs"] += 1
        counts["nodes"] += int(sample.num_nodes)
        counts["visible_virtual_blocks"] += len(result.context_virtual_blocks)
        counts["visible_virtual_nodes"] += int(
            result.context_virtual_node_mask.sum().item()
        )
        counts["mapping_quarantines"] += len(result.context_mapping_quarantines)
        counts["coordinate_quarantines"] += len(
            result.context_coordinate_quarantines
        )
        counts["removed_edges"] += rebuilt.removed_edges
        counts["added_edges"] += rebuilt.added_edges
        for item in result.context_mapping_quarantines:
            quarantine_reasons.update(item.reasons)
        for item in result.context_coordinate_quarantines:
            quarantine_reasons[item.reason] += 1

        sample.dataset_type = original_dataset_type
        del sample.batch
        del sample.mlm_mask
        if "num_graphs" in sample:
            del sample.num_graphs
        processed.append(sample)

        if counts["graphs"] % 25 == 0:
            print(f"processed={counts['graphs']}/{selected_count}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    collated, output_slices = InMemoryDataset.collate(processed)
    torch.save((collated, output_slices), output_path)
    after = _fingerprint(input_path)
    if before != after:
        raise AssertionError("source PT fingerprint changed during preprocessing")

    return {
        "domain": domain,
        "source_graphs": graph_count,
        "graph_range": {"start": start_index, "end": selected_end},
        "source_before": before,
        "source_after": after,
        "output": _fingerprint(output_path),
        "counts": dict(counts),
        "quarantine_reasons": dict(quarantine_reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", choices=("protein", "rna"), required=True)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-neighbors", type=int, default=48)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive graph index; defaults to the source graph count.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("output must be a new PT path; source caches are immutable")

    report = precompute_cache(
        args.input,
        args.output,
        domain=args.domain,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"VISIBLE_VIRTUAL_PRECOMPUTE=GO report={args.report}", flush=True)


if __name__ == "__main__":
    main()
