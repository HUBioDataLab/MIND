#!/usr/bin/env python3
"""Read-only real-source audit for fixed protein/RNA local templates.

The audit streams UniversalMolecule pickle files, applies the Stage 1 target
eligibility policy in memory, and checks the synthetic point clouds against
the retained backbone atoms.  It never writes CIF, PKL, PT, adapter, or
training data; only the requested JSON report is created.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data_loading.data_types as _data_types_module
from core.local_frame import (
    LocalFrameError,
    protein_n_ca_c_frame,
    rna_c2_c3_c4_frame,
)
from core.virtual_atom_templates import (
    MINIMUM_TEMPLATE_SEPARATION_ANGSTROM,
    protein_sidechain_template,
    rna_base_template,
)
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    RNA_BACKBONE_ATOMS,
    normalize_rna_position_code,
    rna_backbone_contract_violations,
    rna_molecule_contains_p_only_block,
)
from data_loading.audit_rna_anchor_selection import is_mirror_rna_molecule


# Historical streams may refer to the old top-level module name.
sys.modules.setdefault("data_types", _data_types_module)

COLLISION_TOLERANCE_ANGSTROM = 1.0e-7
SPATIAL_CUTOFF_ANGSTROM = 5.0
CHIRALITY_EPSILON_ANGSTROM = 1.0e-6


def _fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def _iter_pickle(path: Path):
    with path.open("rb") as handle:
        while True:
            try:
                yield pickle.load(handle)
            except EOFError:
                return


def _update_minimum(current: float, values: torch.Tensor) -> float:
    return min(current, float(values.min().item()))


def _new_geometry_counts() -> Counter:
    return Counter(
        template_backbone_zero_or_near_collisions=0,
        template_backbone_below_minimum_separation=0,
        template_points_without_direct_backbone_spatial_edge=0,
        blocks_without_any_template_backbone_spatial_edge=0,
        nonfinite_template_global_coordinates=0,
    )


def _observe_local_geometry_batch(
    template_local: torch.Tensor,
    backbone_local_batch: list[torch.Tensor],
    anchor_indices: tuple[int, ...],
    counts: Counter,
    minima: dict[str, float],
) -> None:
    if not backbone_local_batch:
        return
    backbone_local = torch.stack(backbone_local_batch)
    if not torch.isfinite(backbone_local).all():
        counts["nonfinite_template_global_coordinates"] += int(
            (~torch.isfinite(backbone_local).all(dim=(1, 2))).sum().item()
        )
        return

    # One batched operation per molecule is far faster than a separate cdist
    # dispatch for every residue while producing the same rigid-invariant
    # distances in local coordinates.
    distances = torch.linalg.vector_norm(
        template_local[None, :, None, :] - backbone_local[:, None, :, :], dim=-1
    )
    minima["template_to_backbone_angstrom"] = _update_minimum(
        minima["template_to_backbone_angstrom"], distances
    )
    minima["template_to_anchor_angstrom"] = _update_minimum(
        minima["template_to_anchor_angstrom"], distances[:, :, anchor_indices]
    )
    counts["template_backbone_zero_or_near_collisions"] += int(
        (distances <= COLLISION_TOLERANCE_ANGSTROM).sum().item()
    )
    counts["template_backbone_below_minimum_separation"] += int(
        (distances < MINIMUM_TEMPLATE_SEPARATION_ANGSTROM).sum().item()
    )
    nearest_backbone = distances.min(dim=2).values
    counts["template_points_without_direct_backbone_spatial_edge"] += int(
        (nearest_backbone > SPATIAL_CUTOFF_ANGSTROM).sum().item()
    )
    counts["blocks_without_any_template_backbone_spatial_edge"] += int(
        (distances > SPATIAL_CUTOFF_ANGSTROM).all(dim=(1, 2)).sum().item()
    )


def audit_protein(path: Path, progress_every: int) -> dict:
    template = protein_sidechain_template(dtype=torch.float64)
    counts = _new_geometry_counts()
    minima = {
        "template_to_template_angstrom": float(torch.pdist(template).min()),
        "template_to_anchor_angstrom": float("inf"),
        "template_to_backbone_angstrom": float("inf"),
    }

    for molecule_index, molecule in enumerate(_iter_pickle(path), start=1):
        counts["input_molecules"] += 1
        molecule_backbones_local = []
        for block in molecule.blocks:
            counts["input_blocks"] += 1
            positions_by_code: dict[str, list[torch.Tensor]] = {}
            for atom in block.atoms:
                positions_by_code.setdefault(str(atom.pos_code), []).append(
                    torch.as_tensor(atom.position, dtype=torch.float64)
                )
            invalid = tuple(
                code
                for code in PROTEIN_BACKBONE_ATOMS
                if len(positions_by_code.get(code, ())) != 1
            )
            if invalid:
                counts["quarantined_backbone_contract_blocks"] += 1
                continue
            try:
                frame = protein_n_ca_c_frame(
                    {code: positions_by_code[code][0] for code in ("N", "CA", "C")}
                )
            except LocalFrameError as error:
                counts["frame_failures"] += 1
                counts[f"frame_failure:{error.reason}"] += 1
                continue

            backbone = torch.stack(
                [positions_by_code[code][0] for code in PROTEIN_BACKBONE_ATOMS]
            )
            molecule_backbones_local.append(frame.global_to_local(backbone))
            counts["accepted_target_blocks"] += 1

        _observe_local_geometry_batch(
            template,
            molecule_backbones_local,
            (0, 1, 2),
            counts,
            minima,
        )

        if progress_every and molecule_index % progress_every == 0:
            print(
                f"[protein template audit] molecules={molecule_index:,} "
                f"blocks={counts['input_blocks']:,}",
                flush=True,
            )

    return {"counts": dict(counts), "minimum_distances": minima}


def _rna_block_frame(block):
    codes = [normalize_rna_position_code(atom.pos_code) for atom in block.atoms]
    missing, duplicates = rna_backbone_contract_violations(codes)
    if missing or duplicates:
        return codes, None, None
    positions = torch.tensor(
        [atom.position for atom in block.atoms], dtype=torch.float64
    )
    frame = rna_c2_c3_c4_frame(codes, positions)
    o2_index = codes.index("O2'")
    signed_z = float(frame.global_to_local(positions[o2_index])[2].item())
    return codes, positions, (frame, signed_z)


def audit_rna(paths: list[Path], progress_every: int) -> dict:
    template = rna_base_template(dtype=torch.float64)
    counts = _new_geometry_counts()
    minima = {
        "template_to_template_angstrom": float(torch.pdist(template).min()),
        "template_to_anchor_angstrom": float("inf"),
        "template_to_backbone_angstrom": float("inf"),
    }
    source_summaries = []

    for path in paths:
        source_counts = Counter()
        for molecule_index, molecule in enumerate(_iter_pickle(path), start=1):
            counts["input_molecules"] += 1
            source_counts["input_molecules"] += 1
            block_count = len(molecule.blocks)
            counts["input_blocks"] += block_count
            source_counts["input_blocks"] += block_count
            if rna_molecule_contains_p_only_block(molecule):
                counts["rejected_p_only_molecules"] += 1
                counts["rejected_p_only_molecule_blocks"] += block_count
                source_counts["rejected_p_only_molecules"] += 1
                source_counts["rejected_p_only_molecule_blocks"] += block_count
                continue

            inspected = []
            nonpositive = 0
            valid_orientation = 0
            for block in molecule.blocks:
                try:
                    codes, positions, frame_result = _rna_block_frame(block)
                except LocalFrameError as error:
                    counts["frame_failures"] += 1
                    counts[f"frame_failure:{error.reason}"] += 1
                    inspected.append((codes, None, None))
                    continue
                inspected.append((codes, positions, frame_result))
                if frame_result is not None:
                    valid_orientation += 1
                    nonpositive += int(frame_result[1] <= CHIRALITY_EPSILON_ANGSTROM)

            if is_mirror_rna_molecule(nonpositive, valid_orientation):
                counts["rejected_mirror_molecules"] += 1
                counts["rejected_mirror_molecule_blocks"] += block_count
                source_counts["rejected_mirror_molecules"] += 1
                source_counts["rejected_mirror_molecule_blocks"] += block_count
                continue

            molecule_backbones_local = []
            for codes, positions, frame_result in inspected:
                if frame_result is None:
                    counts["quarantined_backbone_or_frame_blocks"] += 1
                    source_counts["quarantined_backbone_or_frame_blocks"] += 1
                    continue
                frame, signed_z = frame_result
                if signed_z <= CHIRALITY_EPSILON_ANGSTROM:
                    counts["quarantined_noncanonical_orientation_blocks"] += 1
                    source_counts["quarantined_noncanonical_orientation_blocks"] += 1
                    continue

                index_by_code = {code: codes.index(code) for code in RNA_BACKBONE_ATOMS}
                backbone = torch.stack(
                    [positions[index_by_code[code]] for code in RNA_BACKBONE_ATOMS]
                )
                molecule_backbones_local.append(frame.global_to_local(backbone))
                counts["accepted_target_blocks"] += 1
                source_counts["accepted_target_blocks"] += 1

            _observe_local_geometry_batch(
                template,
                molecule_backbones_local,
                (9, 7, 5),
                counts,
                minima,
            )

            if progress_every and molecule_index % progress_every == 0:
                print(
                    f"[RNA template audit:{path.name}] "
                    f"molecules={molecule_index:,}",
                    flush=True,
                )

        source_summaries.append(
            {"source_pkl": str(path.resolve()), "counts": dict(source_counts)}
        )

    return {
        "counts": dict(counts),
        "minimum_distances": minima,
        "source_summaries": source_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-pkl", type=Path, required=True)
    parser.add_argument("--rna-pkl", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    sources = [args.protein_pkl, *args.rna_pkl]
    before = [_fingerprint(path) for path in sources]
    protein = audit_protein(args.protein_pkl, args.progress_every)
    rna = audit_rna(args.rna_pkl, args.progress_every)
    after = [_fingerprint(path) for path in sources]

    failures = {
        "source_fingerprints_changed": before != after,
        "protein_frame_failures": protein["counts"].get("frame_failures", 0) > 0,
        "rna_frame_failures": rna["counts"].get("frame_failures", 0) > 0,
        "protein_collision": protein["counts"].get(
            "template_backbone_below_minimum_separation", 0
        ) > 0,
        "rna_collision": rna["counts"].get(
            "template_backbone_below_minimum_separation", 0
        ) > 0,
        "protein_disconnected_template_blocks": protein["counts"].get(
            "blocks_without_any_template_backbone_spatial_edge", 0
        ) > 0,
        "rna_disconnected_template_blocks": rna["counts"].get(
            "blocks_without_any_template_backbone_spatial_edge", 0
        ) > 0,
    }
    report = {
        "stage": "stage1_type_independent_local_template_points",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GO" if not any(failures.values()) else "STOP",
        "thresholds_angstrom": {
            "minimum_separation": MINIMUM_TEMPLATE_SEPARATION_ANGSTROM,
            "zero_or_near_collision": COLLISION_TOLERANCE_ANGSTROM,
            "spatial_edge_cutoff": SPATIAL_CUTOFF_ANGSTROM,
        },
        "protein": protein,
        "rna": rna,
        "stop_conditions": failures,
        "source_fingerprints_before": before,
        "source_fingerprints_after": after,
        "production_cif_pkl_pt_files_rewritten": False,
        "adapter_masking_edges_distance_loss_training_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "stop_conditions": failures}, indent=2))
    if report["status"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
