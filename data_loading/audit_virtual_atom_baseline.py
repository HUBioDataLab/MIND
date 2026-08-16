#!/usr/bin/env python3
"""Read-only audit for fixed-slot protein/RNA PyG cache files.

The script deliberately processes one cache at a time.  It records enough
identity and geometry information to make virtual-atom experiments comparable
without rewriting any production cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from data_loading.atom_position_vocab import (
    RARE_ATOMIC_NUMBER,
    VIRTUAL_ATOMIC_NUMBER,
    is_virtual_position_code,
)


ZERO_DISTANCE_TOLERANCE_ANGSTROM = 1.0e-7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile_summary(values: torch.Tensor) -> dict[str, float | int]:
    if values.numel() == 0:
        return {"min": 0, "p25": 0.0, "p50": 0.0, "p75": 0.0,
                "p95": 0.0, "p99": 0.0, "max": 0, "mean": 0.0}
    float_values = values.to(torch.float64)
    quantiles = torch.quantile(
        float_values,
        torch.tensor([0.25, 0.50, 0.75, 0.95, 0.99], dtype=float_values.dtype),
    ).tolist()
    return {
        "min": int(values.min().item()),
        "p25": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": int(values.max().item()),
        "mean": float(float_values.mean().item()),
    }


def _metadata_audit(metadata_path: Path | None) -> dict[str, Any] | None:
    if metadata_path is None:
        return None
    metadata = json.loads(metadata_path.read_text())
    return {
        "path": str(metadata_path.resolve()),
        "sha256": _sha256(metadata_path),
        "num_samples": metadata.get("num_samples"),
        "total_atoms": metadata.get("total_atoms"),
        "total_edges": metadata.get("total_edges"),
    }


def audit_cache(
    name: str,
    cache_path: Path,
    metadata_path: Path | None,
    expected_slots: int | None,
    edge_chunk_size: int,
) -> dict[str, Any]:
    cache_path = cache_path.resolve()
    stat = cache_path.stat()
    metadata = _metadata_audit(metadata_path.resolve() if metadata_path else None)

    print(f"[{name}] hashing {cache_path}", flush=True)
    cache_sha256 = _sha256(cache_path)
    print(f"[{name}] loading cache", flush=True)
    loaded = torch.load(cache_path, weights_only=False, map_location="cpu")
    if not (isinstance(loaded, tuple) and len(loaded) == 2):
        raise TypeError("Expected torch cache payload (data, slices)")
    data, slices = loaded

    required = ("pos", "z", "block_idx", "entity_idx", "edge_index", "edge_attr", "pos_code")
    missing = [key for key in required if not hasattr(data, key)]
    if missing:
        raise AssertionError(f"Missing required cache fields: {missing}")

    atom_count = int(data.z.numel())
    edge_count = int(data.edge_index.shape[1])
    graph_count = len(data.pos_code)
    lengths = {
        "pos": int(data.pos.shape[0]),
        "z": atom_count,
        "block_idx": int(data.block_idx.numel()),
        "entity_idx": int(data.entity_idx.numel()),
        "pos_code_from_graph_lists": int(sum(map(len, data.pos_code))),
        "final_pos_slice": int(slices["pos"][-1]),
        "edge_index": edge_count,
        "edge_attr": int(data.edge_attr.shape[0]),
    }
    atom_length_keys = ("pos", "z", "block_idx", "entity_idx", "pos_code_from_graph_lists", "final_pos_slice")
    if any(lengths[key] != atom_count for key in atom_length_keys):
        raise AssertionError(f"Atom-aligned length mismatch: {lengths}")
    if lengths["edge_attr"] != edge_count:
        raise AssertionError(f"Edge-aligned length mismatch: {lengths}")
    if len(slices["pos"]) != graph_count + 1:
        raise AssertionError("Graph count and position slices disagree")
    if data.edge_index.shape[0] != 2:
        raise AssertionError(f"Expected edge_index [2,E], got {tuple(data.edge_index.shape)}")
    print(f"[{name}] auditing graph/block/slot contracts", flush=True)
    slot_histogram: Counter[int] = Counter()
    position_histogram: Counter[str] = Counter()
    virtual_position_count = 0
    virtual_mask = torch.zeros(atom_count, dtype=torch.bool)
    nonvirtual_z119_positions: Counter[str] = Counter()
    block_count = 0
    block_symbol_count = 0
    block_symbol_mismatch_graphs = 0
    duplicate_block_key_graphs = 0
    graph_atom_counts: list[int] = []
    graph_pair_total = 0
    isolated_graph_long_range_support = 0

    for graph_idx, codes in enumerate(data.pos_code):
        start = int(slices["pos"][graph_idx])
        end = int(slices["pos"][graph_idx + 1])
        graph_atoms = end - start
        graph_atom_counts.append(graph_atoms)
        if len(codes) != graph_atoms:
            raise AssertionError(f"Graph {graph_idx}: pos_code length mismatch")
        position_histogram.update(map(str, codes))
        local_virtual_list = [is_virtual_position_code(code) for code in codes]
        local_virtual_mask = torch.tensor(local_virtual_list, dtype=torch.bool)
        virtual_mask[start:end] = local_virtual_mask
        virtual_position_count += int(local_virtual_mask.sum().item())
        local_z119_nonvirtual = (
            (data.z[start:end] == RARE_ATOMIC_NUMBER) & ~local_virtual_mask
        )
        for local_idx in torch.nonzero(
            local_z119_nonvirtual, as_tuple=False
        ).flatten().tolist():
            nonvirtual_z119_positions[str(codes[local_idx])] += 1

        entity = data.entity_idx[start:end]
        block = data.block_idx[start:end]
        if graph_atoms:
            block_keys = torch.stack((entity, block), dim=1)
            unique_keys, counts = torch.unique(block_keys, dim=0, return_counts=True)
            local_blocks = int(unique_keys.shape[0])
            block_count += local_blocks
            slot_histogram.update(map(int, counts.tolist()))

            consecutive_changes = torch.ones(graph_atoms, dtype=torch.bool)
            if graph_atoms > 1:
                consecutive_changes[1:] = torch.any(block_keys[1:] != block_keys[:-1], dim=1)
            if int(consecutive_changes.sum()) != local_blocks:
                duplicate_block_key_graphs += 1

            if hasattr(data, "block_symbols"):
                symbols = data.block_symbols[graph_idx]
                symbol_count = len(symbols)
                block_symbol_count += symbol_count
                if symbol_count != local_blocks:
                    block_symbol_mismatch_graphs += 1

        pair_count = graph_atoms * (graph_atoms - 1) // 2
        graph_pair_total += pair_count
        isolated_graph_long_range_support += pair_count if graph_atoms <= 50 else min(1000, pair_count)

    z119_count = int((data.z == RARE_ATOMIC_NUMBER).sum().item())
    z120_count = int((data.z == VIRTUAL_ATOMIC_NUMBER).sum().item())
    graph_atoms_tensor = torch.tensor(graph_atom_counts, dtype=torch.long)
    virtual_legacy_z119 = int(
        (virtual_mask & (data.z == RARE_ATOMIC_NUMBER)).sum().item()
    )
    virtual_z120 = int(
        (virtual_mask & (data.z == VIRTUAL_ATOMIC_NUMBER)).sum().item()
    )
    virtual_with_unrecognized_id = (
        virtual_position_count - virtual_legacy_z119 - virtual_z120
    )
    if virtual_with_unrecognized_id:
        raise AssertionError(
            f"{virtual_with_unrecognized_id} V positional labels use neither "
            f"legacy z={RARE_ATOMIC_NUMBER} nor z={VIRTUAL_ATOMIC_NUMBER}"
        )
    nonvirtual_z119_count = int(
        ((~virtual_mask) & (data.z == RARE_ATOMIC_NUMBER)).sum().item()
    )
    nonvirtual_z120_count = int(
        ((~virtual_mask) & (data.z == VIRTUAL_ATOMIC_NUMBER)).sum().item()
    )
    if virtual_legacy_z119 + nonvirtual_z119_count != z119_count:
        raise AssertionError("z==119 accounting does not match V and non-V labels")
    if nonvirtual_z120_count:
        raise AssertionError(
            f"{nonvirtual_z120_count} non-virtual atoms incorrectly use z==120"
        )

    print(f"[{name}] auditing {edge_count:,} edges graph by graph", flush=True)
    # InMemoryDataset.collate stores every graph's edge_index in local 0..N-1
    # coordinates.  The edge slices, unlike a runtime Batch, are not globally
    # incremented.  Respect graph boundaries for geometry and degree counts.
    if "edge_index" not in slices or len(slices["edge_index"]) != graph_count + 1:
        raise AssertionError("Graph count and edge_index slices disagree")
    source_degree = torch.empty(atom_count, dtype=torch.long)
    target_degree = torch.empty(atom_count, dtype=torch.long)
    zero_edges = 0
    zero_edges_touching_virtual = 0
    self_loops = 0
    nonfinite_edge_attr = 0
    edge_attr_distance_mismatch = 0
    maximum_edge_attr_error = 0.0
    tolerance_squared = ZERO_DISTANCE_TOLERANCE_ANGSTROM ** 2

    for graph_idx in range(graph_count):
        node_start = int(slices["pos"][graph_idx])
        node_end = int(slices["pos"][graph_idx + 1])
        local_atom_count = node_end - node_start
        edge_start = int(slices["edge_index"][graph_idx])
        edge_end = int(slices["edge_index"][graph_idx + 1])
        local_edges = data.edge_index[:, edge_start:edge_end]
        if local_edges.numel() and (
            int(local_edges.min()) < 0 or int(local_edges.max()) >= local_atom_count
        ):
            raise AssertionError(f"Graph {graph_idx}: edge_index is outside local atom range")

        source_degree[node_start:node_end] = torch.bincount(
            local_edges[0], minlength=local_atom_count
        )
        target_degree[node_start:node_end] = torch.bincount(
            local_edges[1], minlength=local_atom_count
        )
        local_positions = data.pos[node_start:node_end]
        local_virtual_mask = virtual_mask[node_start:node_end]

        for chunk_start in range(edge_start, edge_end, edge_chunk_size):
            chunk_end = min(chunk_start + edge_chunk_size, edge_end)
            relative_start = chunk_start - edge_start
            relative_end = chunk_end - edge_start
            sources = local_edges[0, relative_start:relative_end]
            targets = local_edges[1, relative_start:relative_end]
            squared = torch.sum(
                (local_positions[targets] - local_positions[sources]) ** 2, dim=1
            )
            local_zero = squared <= tolerance_squared
            zero_edges += int(local_zero.sum().item())
            zero_edges_touching_virtual += int(
                (
                    local_zero
                    & (local_virtual_mask[sources] | local_virtual_mask[targets])
                ).sum().item()
            )
            self_loops += int((sources == targets).sum().item())

            attr = data.edge_attr[chunk_start:chunk_end].reshape(-1)
            finite = torch.isfinite(attr)
            nonfinite_edge_attr += int((~finite).sum().item())
            if bool(finite.any()):
                recomputed = torch.sqrt(squared[finite])
                errors = torch.abs(attr[finite].to(recomputed.dtype) - recomputed)
                edge_attr_distance_mismatch += int((errors > 1.0e-5).sum().item())
                maximum_edge_attr_error = max(
                    maximum_edge_attr_error, float(errors.max().item())
                )

    total_degree = source_degree + target_degree

    finite_position_values = int(torch.isfinite(data.pos).sum().item())
    position_value_count = int(data.pos.numel())
    metadata_matches = None
    if metadata:
        metadata_matches = {
            "graphs": metadata["num_samples"] == graph_count,
            "atoms": metadata["total_atoms"] == atom_count,
            "edges": metadata["total_edges"] == edge_count,
        }

    exact_expected_blocks = slot_histogram.get(expected_slots, 0) if expected_slots else None
    non_exact_expected_blocks = (
        block_count - exact_expected_blocks if expected_slots else None
    )
    report = {
        "name": name,
        "cache": {
            "path": str(cache_path),
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "sha256": cache_sha256,
        },
        "metadata": metadata,
        "metadata_matches_cache": metadata_matches,
        "integrity": {
            "aligned_lengths": lengths,
            "finite_position_values": finite_position_values,
            "position_value_count": position_value_count,
            "nonfinite_position_values": position_value_count - finite_position_values,
            "nonfinite_edge_attr_values": nonfinite_edge_attr,
            "edge_attr_distance_mismatch_over_1e-5": edge_attr_distance_mismatch,
            "maximum_edge_attr_distance_error": maximum_edge_attr_error,
        },
        "counts": {
            "graphs": graph_count,
            "atoms": atom_count,
            "edges_directed": edge_count,
            "blocks": block_count,
            "block_symbols": block_symbol_count,
            "block_symbol_mismatch_graphs": block_symbol_mismatch_graphs,
            "noncontiguous_duplicate_block_key_graphs": duplicate_block_key_graphs,
            "virtual_position_codes": virtual_position_count,
            "z119": z119_count,
            "z120": z120_count,
            "virtual_legacy_z119": virtual_legacy_z119,
            "virtual_z120": virtual_z120,
            "nonvirtual_z119": nonvirtual_z119_count,
            "nonvirtual_z119_positions": dict(nonvirtual_z119_positions),
        },
        "slots": {
            "expected_per_block": expected_slots,
            "histogram": {str(key): value for key, value in sorted(slot_histogram.items())},
            "exact_expected_blocks": exact_expected_blocks,
            "non_exact_expected_blocks": non_exact_expected_blocks,
        },
        "graph_atom_count": _quantile_summary(graph_atoms_tensor),
        "degree": {
            "source": _quantile_summary(source_degree),
            "target": _quantile_summary(target_degree),
            "incident": _quantile_summary(total_degree),
            "zero_source_degree_atoms": int((source_degree == 0).sum().item()),
            "zero_target_degree_atoms": int((target_degree == 0).sum().item()),
        },
        "geometry": {
            "zero_distance_tolerance_angstrom": ZERO_DISTANCE_TOLERANCE_ANGSTROM,
            "zero_distance_directed_edges": zero_edges,
            "zero_distance_directed_edges_touching_virtual": zero_edges_touching_virtual,
            "self_loops": self_loops,
        },
        "distance_loss_support": {
            "short_range_finite_directed_edges_validation": edge_count - nonfinite_edge_attr,
            "short_range_expected_sampled_directed_edges_train_at_15pct": 0.15 * (edge_count - nonfinite_edge_attr),
            "long_range_unique_within_graph_pairs": graph_pair_total,
            "long_range_pairs_if_each_graph_were_processed_alone_current_cap": isolated_graph_long_range_support,
            "actual_current_long_range_cap_per_collated_model_batch": 1000,
            "note": "Current long-range loss samples after collation; the 1000-pair cap applies to the full model batch.",
        },
        "top_position_codes": dict(position_histogram.most_common(40)),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--expected-slots", type=int)
    parser.add_argument("--edge-chunk-size", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_cache(
        name=args.name,
        cache_path=args.pt,
        metadata_path=args.metadata,
        expected_slots=args.expected_slots,
        edge_chunk_size=args.edge_chunk_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
