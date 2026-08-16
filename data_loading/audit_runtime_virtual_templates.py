#!/usr/bin/env python3
"""Read-only Stage 4 audit of runtime virtual templates on real PT chunks."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch_geometric.data.separate import separate

from core.block_masking import legacy_oxt_eligible_node_mask, select_whole_blocks
from core.masked_coordinate_corruption import apply_masked_slot_template_
from core.pretraining_config import create_pretraining_config
from core.masked_slot_mapping import flatten_pos_codes
from data_loading.atom_position_vocab import is_virtual_position_code
from data_loading.lazy_universal_dataset import normalize_legacy_atom_ids


ZERO_TOLERANCE_ANGSTROM = 1.0e-7


def _fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def _distribution(values: list[torch.Tensor]) -> dict[str, float]:
    merged = torch.cat(values).float()
    return {
        "mean": float(merged.mean().item()),
        "p50": float(torch.quantile(merged, 0.50).item()),
        "p95": float(torch.quantile(merged, 0.95).item()),
        "max": float(merged.max().item()),
    }


def _exact_uniform_blocks(sample, expected_nodes: int) -> bool:
    unique_blocks, counts = torch.unique(sample.block_idx, return_counts=True)
    return bool(unique_blocks.numel()) and bool((counts == expected_nodes).all())


def _prepare_mask(
    sample, *, graph_index: int, ratio: float, domain: str
) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(70_001 + graph_index)
    eligible_node_mask = None
    if domain == "protein":
        eligible_node_mask = legacy_oxt_eligible_node_mask(
            sample.block_idx,
            flatten_pos_codes(sample.pos_code, sample.num_nodes),
        )
    selection = select_whole_blocks(
        sample.block_idx,
        ratio,
        generator=generator,
        eligible_node_mask=eligible_node_mask,
    )
    sample.original_types = sample.z.clone()
    sample.mlm_mask = selection.selected_node_mask
    sample.masked_types = sample.z.clone()
    sample.masked_types[sample.mlm_mask] = 0
    sample.z = sample.masked_types


def _edge_distances(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(
        pos[edge_index[0]] - pos[edge_index[1]], dim=1
    )


def audit_domain(
    path: Path,
    *,
    domain: str,
    requested_graphs: int,
    mask_ratio: float,
    config,
) -> dict:
    expected_nodes = 14 if domain == "protein" else 23
    before = _fingerprint(path)
    started = time.perf_counter()
    data, slices = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    total_graphs = len(slices["pos"]) - 1
    counts = Counter()
    rejection_reasons = Counter()
    baseline_degrees = []
    template_degrees = []
    accepted_ids = []
    rejected_graphs = []
    short_baseline_support = 0
    short_eligible_support = 0
    long_baseline_support = 0
    long_eligible_support = 0

    for graph_index in range(total_graphs):
        if counts["accepted_graphs"] >= requested_graphs:
            break
        counts["inspected_graphs"] += 1
        sample = separate(
            cls=data.__class__,
            batch=data,
            idx=graph_index,
            slice_dict=slices,
            inc_dict=None,
            decrement=False,
        ).clone()
        sample = normalize_legacy_atom_ids(sample)
        if not _exact_uniform_blocks(sample, expected_nodes):
            counts["skipped_nonuniform_graphs"] += 1
            continue

        sample.dataset_type = domain
        sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long)
        sample.num_graphs = 1
        original_pos = sample.pos.clone()
        original_edges = sample.edge_index.clone()
        original_edge_attr = sample.edge_attr.clone()
        original_block_idx = sample.block_idx.clone()
        original_pos_codes = flatten_pos_codes(sample.pos_code, sample.num_nodes)
        original_virtual_mask = torch.tensor(
            [is_virtual_position_code(code) for code in original_pos_codes],
            dtype=torch.bool,
        )
        _prepare_mask(
            sample, graph_index=graph_index, ratio=mask_ratio, domain=domain
        )
        original_targets = sample.original_types.clone()

        try:
            stats = apply_masked_slot_template_(sample, config)
        except RuntimeError as error:
            counts["runtime_rejected_graphs"] += 1
            rejection_reasons[str(error)] += 1
            rejected_graphs.append(
                {
                    "graph_index": graph_index,
                    "molecule_id": str(sample.mol_id),
                    "reason": str(error),
                }
            )
            continue

        if stats["templated_blocks"] == 0:
            counts["runtime_empty_graphs"] += 1
            continue

        if sample.num_nodes != original_pos.shape[0]:
            raise AssertionError("runtime template changed graph node count")
        if not torch.equal(sample.clean_pos, original_pos):
            raise AssertionError("clean_pos differs from loaded coordinates")
        if not torch.equal(sample.original_types, original_targets):
            raise AssertionError("runtime template changed MLM targets")
        if not torch.equal(sample.block_idx, original_block_idx):
            raise AssertionError("runtime template changed block membership")
        if not torch.equal(
            sample.pos[~sample.template_node_mask],
            original_pos[~sample.template_node_mask],
        ):
            raise AssertionError("runtime template changed a non-template node")
        if bool((sample.target_template_node_mask & ~sample.mlm_mask).any()):
            raise AssertionError("target template moved an unmasked node")
        if bool((sample.context_virtual_node_mask & sample.mlm_mask).any()):
            raise AssertionError("context virtual relocation touched a masked node")
        if bool((sample.context_virtual_node_mask & ~original_virtual_mask).any()):
            raise AssertionError("context virtual relocation touched a real atom")
        if bool(sample.context_virtual_node_mask.any()):
            context_displacements = torch.linalg.vector_norm(
                sample.pos[sample.context_virtual_node_mask]
                - original_pos[sample.context_virtual_node_mask],
                dim=1,
            )
            if bool((context_displacements <= ZERO_TOLERANCE_ANGSTROM).any()):
                raise AssertionError("a context virtual node was not relocated")
        if not torch.isfinite(sample.pos).all():
            raise AssertionError("runtime template produced non-finite coordinates")
        if sample.edge_attr is not None and not torch.isfinite(sample.edge_attr).all():
            raise AssertionError("runtime template produced non-finite edge_attr")

        baseline_distances = _edge_distances(original_pos, original_edges)
        new_distances = _edge_distances(sample.pos, sample.edge_index)
        touches_template = (
            sample.template_node_mask[sample.edge_index[0]]
            | sample.template_node_mask[sample.edge_index[1]]
        )
        template_zero_edges = touches_template & (
            new_distances <= ZERO_TOLERANCE_ANGSTROM
        )
        counts["baseline_zero_edges"] += int(
            (baseline_distances <= ZERO_TOLERANCE_ANGSTROM).sum().item()
        )
        counts["new_zero_edges"] += int(
            (new_distances <= ZERO_TOLERANCE_ANGSTROM).sum().item()
        )
        counts["new_zero_edges_touching_template"] += int(
            template_zero_edges.sum().item()
        )
        if bool(template_zero_edges.any()):
            raise AssertionError("template node has a zero-distance runtime edge")

        if sample.edge_attr is not None:
            attr_error = torch.abs(sample.edge_attr.reshape(-1) - new_distances)
            counts["edge_attr_mismatches"] += int(
                (attr_error > 1.0e-5).sum().item()
            )

        baseline_degree = torch.bincount(
            original_edges[0], minlength=sample.num_nodes
        ) + torch.bincount(original_edges[1], minlength=sample.num_nodes)
        new_degree = torch.bincount(
            sample.edge_index[0], minlength=sample.num_nodes
        ) + torch.bincount(sample.edge_index[1], minlength=sample.num_nodes)
        baseline_degrees.append(baseline_degree)
        template_degrees.append(new_degree)

        node_count = int(sample.num_nodes)
        eligible_nodes = node_count - int(sample.template_node_mask.sum().item())
        baseline_physical_edges = ~(
            original_virtual_mask[original_edges[0]]
            | original_virtual_mask[original_edges[1]]
        )
        short_baseline_support += int((
            torch.isfinite(baseline_distances) & baseline_physical_edges
        ).sum().item())
        short_eligible_support += int(
            (torch.isfinite(new_distances) & ~touches_template).sum().item()
        )
        physical_node_count = node_count - int(original_virtual_mask.sum().item())
        long_baseline_support += physical_node_count * (physical_node_count - 1) // 2
        long_eligible_support += eligible_nodes * (eligible_nodes - 1) // 2

        counts["accepted_graphs"] += 1
        counts["nodes"] += node_count
        counts["baseline_edges"] += int(original_edges.shape[1])
        counts["runtime_edges"] += int(sample.edge_index.shape[1])
        counts["templated_blocks"] += int(stats["templated_blocks"])
        counts["templated_nodes"] += int(stats["templated_nodes"])
        counts["target_template_nodes"] += int(stats["target_template_nodes"])
        counts["context_virtual_blocks"] += int(stats["context_virtual_blocks"])
        counts["context_virtual_nodes"] += int(stats["context_virtual_nodes"])
        counts["context_mapping_quarantines"] += int(
            stats["context_mapping_quarantines"]
        )
        counts["context_coordinate_quarantines"] += int(
            stats["context_coordinate_quarantines"]
        )
        counts["removed_edges"] += int(stats["removed_edges"])
        counts["added_edges"] += int(stats["added_edges"])
        accepted_ids.append(str(sample.mol_id))

        if counts["accepted_graphs"] % 10 == 0:
            print(
                f"[{domain}] accepted={counts['accepted_graphs']}/{requested_graphs} "
                f"inspected={counts['inspected_graphs']}",
                flush=True,
            )

    del data, slices
    gc.collect()
    after = _fingerprint(path)
    if before != after:
        raise AssertionError(f"source chunk fingerprint changed: {path}")
    if counts["accepted_graphs"] < requested_graphs:
        raise AssertionError(
            f"{domain}: accepted only {counts['accepted_graphs']}/{requested_graphs} graphs"
        )

    short_ratio = short_eligible_support / short_baseline_support
    long_ratio = long_eligible_support / long_baseline_support
    return {
        "source_before": before,
        "source_after": after,
        "counts": dict(counts),
        "rejection_reasons": dict(rejection_reasons),
        "runtime_rejected_graphs": rejected_graphs,
        "accepted_molecule_ids": accepted_ids,
        "degree": {
            "baseline": _distribution(baseline_degrees),
            "runtime_template": _distribution(template_degrees),
        },
        "distance_support": {
            "short_baseline_edges": short_baseline_support,
            "short_eligible_edges": short_eligible_support,
            "short_support_ratio": short_ratio,
            "long_baseline_pairs": long_baseline_support,
            "long_eligible_pairs": long_eligible_support,
            "long_support_ratio": long_ratio,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein-pt", type=Path, required=True)
    parser.add_argument("--rna-pt", type=Path, required=True)
    parser.add_argument("--graphs", type=int, default=100)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.graphs < 1:
        raise ValueError("--graphs must be positive")

    config = create_pretraining_config(
        pretraining_tasks=["long_range_distance", "short_range_distance", "mlm"],
        protein_block_mlm=True,
        protein_block_mlm_ratio=args.mask_ratio,
        rna_block_mlm=True,
        rna_block_mlm_ratio=args.mask_ratio,
        masked_slot_template={"enabled": True, "protein": True, "rna": True},
        cutoff_distance=5.0,
        max_spatial_neighbors=48,
    )
    report = {
        "requested_graphs_per_domain": args.graphs,
        "mask_ratio": args.mask_ratio,
        "protein": audit_domain(
            args.protein_pt,
            domain="protein",
            requested_graphs=args.graphs,
            mask_ratio=args.mask_ratio,
            config=config,
        ),
        "rna": audit_domain(
            args.rna_pt,
            domain="rna",
            requested_graphs=args.graphs,
            mask_ratio=args.mask_ratio,
            config=config,
        ),
    }
    gate_failures = []
    for domain in ("protein", "rna"):
        support = report[domain]["distance_support"]
        if support["short_support_ratio"] < 0.70:
            gate_failures.append(f"{domain}: short distance support below 70%")
        if support["long_support_ratio"] < 0.70:
            gate_failures.append(f"{domain}: long distance support below 70%")
        counts = report[domain]["counts"]
        if counts.get("new_zero_edges_touching_template", 0):
            gate_failures.append(f"{domain}: zero-distance template edge found")
        if counts.get("edge_attr_mismatches", 0):
            gate_failures.append(f"{domain}: rebuilt edge_attr mismatch found")
        if counts.get("runtime_rejected_graphs", 0):
            gate_failures.append(
                f"{domain}: {counts['runtime_rejected_graphs']} selected graphs "
                "contained quarantined template blocks"
            )

    report["gate"] = {
        "status": "STOP" if gate_failures else "GO",
        "failures": gate_failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if gate_failures:
        raise AssertionError(
            "Stage 4 gate STOP: " + "; ".join(gate_failures)
            + f"; report={args.output}"
        )
    print(f"STAGE4_AUDIT=GO output={args.output}", flush=True)


if __name__ == "__main__":
    main()
