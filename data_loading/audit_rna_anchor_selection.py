#!/usr/bin/env python3
"""Select the RNA local-frame anchor triple after P-only molecule filtering.

This is a read-only Stage 1 audit. It streams the exact UniversalMolecule PKL
sources, rejects whole molecules containing P-only blocks in memory, and never
rewrites source PKL/CIF/PT files.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data_loading.data_types as _data_types_module
from data_loading.atom_position_vocab import (
    normalize_rna_position_code,
    rna_backbone_contract_violations,
    rna_molecule_contains_p_only_block,
)
from data_loading.audit_rna_frame_anchors import ReservoirSummary


# Historical pickle streams may reference the pre-package module name.
sys.modules.setdefault("data_types", _data_types_module)


SELECTION_CANDIDATES = {
    # Role order: (plane anchor, origin, positive-x anchor).
    "c2_c3_c4": ("C2'", "C3'", "C4'"),
    "c1_c2_c3": ("C1'", "C2'", "C3'"),
    "o4_c1_c2": ("O4'", "C1'", "C2'"),
}

ORIENTATION_REFERENCE = "O2'"
PROVISIONAL_CHIRALITY_ANCHORS = SELECTION_CANDIDATES["c2_c3_c4"]
MIRROR_MOLECULE_NONPOSITIVE_FRACTION_THRESHOLD = 0.5


@dataclass(frozen=True)
class CandidateFrame:
    """Numerical frame and conditioning values for one accepted RNA block."""

    origin: np.ndarray
    basis: np.ndarray
    perpendicular_height: float
    sin_angle: float
    determinant: float
    orthonormal_max_abs_error: float
    orientation_reference_signed_z: float


def _position_map(atoms) -> dict[str, list[np.ndarray]]:
    positions: dict[str, list[np.ndarray]] = {}
    for atom in atoms:
        code = normalize_rna_position_code(atom.pos_code)
        positions.setdefault(code, []).append(
            np.asarray(atom.position, dtype=np.float64)
        )
    return positions


def _unique_position(
    positions: dict[str, list[np.ndarray]], position_code: str
) -> np.ndarray:
    values = positions.get(position_code, ())
    if len(values) != 1:
        raise ValueError(
            f"{position_code} must occur exactly once, observed {len(values)}"
        )
    value = values[0]
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{position_code} has invalid/non-finite coordinates")
    return value


def build_candidate_frame(
    atoms,
    anchors: tuple[str, str, str],
    *,
    epsilon: float = 1.0e-6,
    orientation_reference: str = ORIENTATION_REFERENCE,
) -> CandidateFrame:
    """Build the deterministic right-handed frame used by the selection audit."""
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    positions = _position_map(atoms)
    plane_name, origin_name, x_name = anchors
    plane = _unique_position(positions, plane_name)
    origin = _unique_position(positions, origin_name)
    x_anchor = _unique_position(positions, x_name)
    marker = _unique_position(positions, orientation_reference)

    origin_to_plane = plane - origin
    origin_to_x = x_anchor - origin
    plane_norm = float(np.linalg.norm(origin_to_plane))
    x_norm = float(np.linalg.norm(origin_to_x))
    if plane_norm <= epsilon or x_norm <= epsilon:
        raise ValueError("coincident anchors")

    x_axis = origin_to_x / x_norm
    perpendicular = origin_to_plane - np.dot(origin_to_plane, x_axis) * x_axis
    perpendicular_height = float(np.linalg.norm(perpendicular))
    if perpendicular_height <= epsilon:
        raise ValueError("collinear anchors")

    y_axis = perpendicular / perpendicular_height
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    # Match the production local-frame convention and reduce accumulated error.
    y_axis = np.cross(z_axis, x_axis)
    basis = np.stack((x_axis, y_axis, z_axis), axis=1)

    identity_error = basis.T @ basis - np.eye(3, dtype=np.float64)
    return CandidateFrame(
        origin=origin,
        basis=basis,
        perpendicular_height=perpendicular_height,
        sin_angle=perpendicular_height / plane_norm,
        determinant=float(np.linalg.det(basis)),
        orthonormal_max_abs_error=float(np.max(np.abs(identity_error))),
        orientation_reference_signed_z=float(np.dot(marker - origin, z_axis)),
    )


class CandidateSelectionStats:
    def __init__(self, name: str, anchors: tuple[str, str, str]):
        self.name = name
        self.anchors = anchors
        seed = sum(map(ord, name)) + 10_000
        self.counts: Counter[str] = Counter()
        self.perpendicular_height = ReservoirSummary(seed=seed + 1)
        self.sin_angle = ReservoirSummary(seed=seed + 2)
        self.determinant = ReservoirSummary(seed=seed + 3)
        self.orthonormal_error = ReservoirSummary(seed=seed + 4)
        self.orientation_signed_z = ReservoirSummary(seed=seed + 5)
        self.orientation_abs_z = ReservoirSummary(seed=seed + 6)
        self.max_reordered_origin_abs_diff = 0.0
        self.max_reordered_basis_abs_diff = 0.0
        self.orientation_nonpositive_examples: list[dict] = []
        self.valid_blocks_by_molecule: Counter[str] = Counter()
        self.orientation_nonpositive_blocks_by_molecule: Counter[str] = Counter()

    def observe(
        self,
        atoms,
        *,
        source_pkl: Path,
        molecule_id: str,
        block_index: int,
        block_symbol: str,
        epsilon: float,
    ) -> None:
        self.counts["accepted_blocks_seen"] += 1
        try:
            frame = build_candidate_frame(atoms, self.anchors, epsilon=epsilon)
            reversed_frame = build_candidate_frame(
                list(reversed(atoms)), self.anchors, epsilon=epsilon
            )
        except ValueError as error:
            self.counts["invalid_frame_blocks"] += 1
            self.counts[f"invalid:{error}"] += 1
            return

        self.counts["valid_frame_blocks"] += 1
        self.valid_blocks_by_molecule[molecule_id] += 1
        origin_diff = float(np.max(np.abs(frame.origin - reversed_frame.origin)))
        basis_diff = float(np.max(np.abs(frame.basis - reversed_frame.basis)))
        self.max_reordered_origin_abs_diff = max(
            self.max_reordered_origin_abs_diff, origin_diff
        )
        self.max_reordered_basis_abs_diff = max(
            self.max_reordered_basis_abs_diff, basis_diff
        )
        if origin_diff != 0.0 or basis_diff != 0.0:
            self.counts["atom_order_invariance_failures"] += 1

        if frame.determinant <= 0.0:
            self.counts["non_right_handed_frames"] += 1

        signed_z = frame.orientation_reference_signed_z
        if signed_z > epsilon:
            self.counts["orientation_reference_positive"] += 1
        elif signed_z < -epsilon:
            self.counts["orientation_reference_negative"] += 1
        else:
            self.counts["orientation_reference_near_plane"] += 1
        if signed_z <= epsilon and len(self.orientation_nonpositive_examples) < 100:
            self.orientation_nonpositive_examples.append(
                {
                    "source_pkl": str(source_pkl.resolve()),
                    "molecule_id": molecule_id,
                    "block_index": block_index,
                    "block_symbol": block_symbol,
                    "orientation_reference_signed_z": signed_z,
                }
            )
        if signed_z <= epsilon:
            self.orientation_nonpositive_blocks_by_molecule[molecule_id] += 1

        self.perpendicular_height.add(frame.perpendicular_height)
        self.sin_angle.add(frame.sin_angle)
        self.determinant.add(frame.determinant)
        self.orthonormal_error.add(frame.orthonormal_max_abs_error)
        self.orientation_signed_z.add(signed_z)
        self.orientation_abs_z.add(abs(signed_z))

    def report(self) -> dict:
        accepted = self.counts["accepted_blocks_seen"]
        valid = self.counts["valid_frame_blocks"]
        nonpositive_molecules = []
        for molecule_id, nonpositive_count in (
            self.orientation_nonpositive_blocks_by_molecule.most_common()
        ):
            valid_count = self.valid_blocks_by_molecule[molecule_id]
            nonpositive_molecules.append(
                {
                    "molecule_id": molecule_id,
                    "nonpositive_blocks": nonpositive_count,
                    "valid_blocks": valid_count,
                    "nonpositive_percent": 100.0 * nonpositive_count / valid_count,
                }
            )
        return {
            "anchors_plane_origin_x": list(self.anchors),
            "orientation_reference": ORIENTATION_REFERENCE,
            "counts": dict(self.counts),
            "accepted_block_valid_percent": (
                100.0 * valid / accepted if accepted else 0.0
            ),
            "geometry": {
                "perpendicular_height_angstrom": self.perpendicular_height.report(),
                "sin_angle": self.sin_angle.report(),
                "determinant": self.determinant.report(),
                "orthonormal_max_abs_error": self.orthonormal_error.report(),
                "orientation_reference_signed_z_angstrom": self.orientation_signed_z.report(),
                "orientation_reference_abs_z_angstrom": self.orientation_abs_z.report(),
            },
            "atom_order_invariance": {
                "reordered_as_reversed_input_atom_list": True,
                "failures": self.counts["atom_order_invariance_failures"],
                "max_origin_abs_diff": self.max_reordered_origin_abs_diff,
                "max_basis_abs_diff": self.max_reordered_basis_abs_diff,
            },
            "orientation_nonpositive_examples_first_100": (
                self.orientation_nonpositive_examples
            ),
            "orientation_nonpositive_molecules": nonpositive_molecules,
        }


def select_numerically_dominant_candidate(candidate_reports: dict) -> str | None:
    """Select a unique safe candidate that wins both worst-case metrics."""
    safe: dict[str, dict] = {}
    for name, report in candidate_reports.items():
        counts = report["counts"]
        if (
            report["accepted_block_valid_percent"] == 100.0
            and counts.get("atom_order_invariance_failures", 0) == 0
            and counts.get("non_right_handed_frames", 0) == 0
        ):
            safe[name] = report
    if not safe:
        return None

    max_min_height = max(
        report["geometry"]["perpendicular_height_angstrom"]["min"]
        for report in safe.values()
    )
    max_min_sin = max(
        report["geometry"]["sin_angle"]["min"] for report in safe.values()
    )
    winners = [
        name
        for name, report in safe.items()
        if math.isclose(
            report["geometry"]["perpendicular_height_angstrom"]["min"],
            max_min_height,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            report["geometry"]["sin_angle"]["min"],
            max_min_sin,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    return winners[0] if len(winners) == 1 else None


def is_mirror_rna_molecule(
    nonpositive_blocks: int,
    valid_exact_backbone_blocks: int,
    *,
    threshold: float = MIRROR_MOLECULE_NONPOSITIVE_FRACTION_THRESHOLD,
) -> bool:
    """Classify molecule-level mirror chirality without deleting local outliers."""
    if valid_exact_backbone_blocks <= 0:
        return False
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")
    return nonpositive_blocks / valid_exact_backbone_blocks > threshold


def _stat_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def audit_anchor_selection(
    source_pkls: list[Path], *, epsilon: float = 1.0e-6, progress_every: int = 500
) -> dict:
    before = [_stat_fingerprint(path) for path in source_pkls]
    candidates = {
        name: CandidateSelectionStats(name, anchors)
        for name, anchors in SELECTION_CANDIDATES.items()
    }
    counts: Counter[str] = Counter()
    quarantine_patterns: Counter[tuple] = Counter()
    source_summaries: list[dict] = []

    for source_pkl in source_pkls:
        source_counts: Counter[str] = Counter()
        with source_pkl.open("rb") as handle:
            while True:
                try:
                    molecule = pickle.load(handle)
                except EOFError:
                    break
                counts["input_molecules"] += 1
                source_counts["input_molecules"] += 1
                molecule_blocks = list(molecule.blocks)
                counts["input_blocks"] += len(molecule_blocks)
                source_counts["input_blocks"] += len(molecule_blocks)

                if rna_molecule_contains_p_only_block(molecule):
                    counts["rejected_p_only_molecules"] += 1
                    counts["rejected_p_only_molecule_blocks"] += len(molecule_blocks)
                    source_counts["rejected_p_only_molecules"] += 1
                    source_counts["rejected_p_only_molecule_blocks"] += len(
                        molecule_blocks
                    )
                    continue

                counts["retained_molecules"] += 1
                counts["retained_blocks"] += len(molecule_blocks)
                source_counts["retained_molecules"] += 1
                source_counts["retained_blocks"] += len(molecule_blocks)
                molecule_id = str(getattr(molecule, "id", "unknown"))

                for block_index, block in enumerate(molecule_blocks):
                    missing, duplicates = rna_backbone_contract_violations(
                        atom.pos_code for atom in block.atoms
                    )
                    if missing or duplicates:
                        counts["quarantined_blocks"] += 1
                        source_counts["quarantined_blocks"] += 1
                        quarantine_patterns[(missing, tuple(sorted(duplicates.items())))] += 1
                        continue

                    counts["accepted_exact_backbone_blocks"] += 1
                    source_counts["accepted_exact_backbone_blocks"] += 1
                    for candidate in candidates.values():
                        candidate.observe(
                            block.atoms,
                            source_pkl=source_pkl,
                            molecule_id=molecule_id,
                            block_index=block_index,
                            block_symbol=str(getattr(block, "symbol", "")),
                            epsilon=epsilon,
                        )

                if progress_every and counts["input_molecules"] % progress_every == 0:
                    print(
                        "[RNA anchor selection] "
                        f"input={counts['input_molecules']:,} "
                        f"rejected={counts['rejected_p_only_molecules']:,} "
                        f"accepted_blocks={counts['accepted_exact_backbone_blocks']:,}",
                        flush=True,
                    )

        source_summaries.append(
            {"source_pkl": str(source_pkl.resolve()), "counts": dict(source_counts)}
        )

    after = [_stat_fingerprint(path) for path in source_pkls]
    candidate_reports = {
        name: candidate.report() for name, candidate in candidates.items()
    }
    selected = select_numerically_dominant_candidate(candidate_reports)
    retained_blocks = counts["retained_blocks"]
    quarantined_blocks = counts["quarantined_blocks"]
    quarantine_percent = (
        100.0 * quarantined_blocks / retained_blocks if retained_blocks else 0.0
    )

    stop_conditions = {
        "unexpected_p_only_reject_count": counts["rejected_p_only_molecules"] != 55,
        "quarantine_above_2_5_percent": quarantine_percent > 2.5,
        "accepted_candidate_coverage_below_100_percent": any(
            report["accepted_block_valid_percent"] != 100.0
            for report in candidate_reports.values()
        ),
        "atom_order_changes_frame": any(
            report["counts"].get("atom_order_invariance_failures", 0) > 0
            for report in candidate_reports.values()
        ),
        "non_right_handed_frame": any(
            report["counts"].get("non_right_handed_frames", 0) > 0
            for report in candidate_reports.values()
        ),
        "selected_candidate_has_noncanonical_orientation_blocks": (
            selected is not None
            and (
                candidate_reports[selected]["counts"].get(
                    "orientation_reference_negative", 0
                )
                + candidate_reports[selected]["counts"].get(
                    "orientation_reference_near_plane", 0
                )
            )
            > 0
        ),
        "no_unique_worst_case_geometry_winner": selected is None,
        "source_fingerprint_changed": before != after,
    }

    return {
        "stage": "stage1.6_rna_anchor_selection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": "training_source_pkl_streams_read_only",
        "epsilon_angstrom": epsilon,
        "p_only_policy": "reject_entire_molecule_if_any_block_is_p_only",
        "accepted_target_policy": (
            "retain non-P-only molecules as context; only blocks with exactly one "
            "of every canonical 12-position RNA backbone atom are template/MLM targets"
        ),
        "counts": dict(counts),
        "quarantine_percent_of_retained_blocks": quarantine_percent,
        "quarantine_patterns": [
            {
                "count": count,
                "missing": list(missing),
                "duplicates": dict(duplicates),
            }
            for (missing, duplicates), count in quarantine_patterns.most_common(25)
        ],
        "candidate_role_order": ["plane_anchor", "origin", "positive_x_anchor"],
        "orientation_reference": ORIENTATION_REFERENCE,
        "orientation_reference_interpretation": (
            "O2' signed local-z is a stereochemical data diagnostic. A rare "
            "negative sign is a real-coordinate outlier, not an atom-order or "
            "right-handedness ambiguity; examples are retained for review."
        ),
        "candidates": candidate_reports,
        "selection_rule": (
            "Among candidates with 100% accepted coverage, zero atom-order "
            "failures, and right-handed frames, select the unique candidate "
            "with both the largest worst-case perpendicular height and the "
            "largest worst-case sin(angle)."
        ),
        "selected_candidate": selected,
        "selected_anchors_plane_origin_x": (
            list(SELECTION_CANDIDATES[selected]) if selected else None
        ),
        "selection_made": selected is not None and not any(stop_conditions.values()),
        "stop_conditions": stop_conditions,
        "go": not any(stop_conditions.values()),
        "source_summaries": source_summaries,
        "source_fingerprints_before": before,
        "source_fingerprints_after": after,
        "production_cif_pkl_pt_files_rewritten": False,
    }


def audit_chirality_policy_simulation(
    source_pkls: list[Path],
    *,
    epsilon: float = 1.0e-6,
    mirror_threshold: float = MIRROR_MOLECULE_NONPOSITIVE_FRACTION_THRESHOLD,
    progress_every: int = 500,
) -> dict:
    """Apply the proposed D/L policy in memory, then re-run anchor selection."""
    before = [_stat_fingerprint(path) for path in source_pkls]
    candidates = {
        name: CandidateSelectionStats(name, anchors)
        for name, anchors in SELECTION_CANDIDATES.items()
    }
    counts: Counter[str] = Counter()
    quarantine_patterns: Counter[tuple] = Counter()
    mirror_rejections: list[dict] = []
    orientation_quarantine_examples: list[dict] = []
    source_summaries: list[dict] = []

    for source_pkl in source_pkls:
        source_counts: Counter[str] = Counter()
        with source_pkl.open("rb") as handle:
            while True:
                try:
                    molecule = pickle.load(handle)
                except EOFError:
                    break

                counts["input_molecules"] += 1
                source_counts["input_molecules"] += 1
                molecule_blocks = list(molecule.blocks)
                molecule_id = str(getattr(molecule, "id", "unknown"))
                counts["input_blocks"] += len(molecule_blocks)
                source_counts["input_blocks"] += len(molecule_blocks)

                if rna_molecule_contains_p_only_block(molecule):
                    counts["rejected_p_only_molecules"] += 1
                    counts["rejected_p_only_molecule_blocks"] += len(molecule_blocks)
                    source_counts["rejected_p_only_molecules"] += 1
                    source_counts["rejected_p_only_molecule_blocks"] += len(
                        molecule_blocks
                    )
                    continue

                exact_block_records: list[tuple[int, object, CandidateFrame]] = []
                for block_index, block in enumerate(molecule_blocks):
                    missing, duplicates = rna_backbone_contract_violations(
                        atom.pos_code for atom in block.atoms
                    )
                    if missing or duplicates:
                        continue
                    frame = build_candidate_frame(
                        block.atoms,
                        PROVISIONAL_CHIRALITY_ANCHORS,
                        epsilon=epsilon,
                    )
                    exact_block_records.append((block_index, block, frame))

                nonpositive_records = [
                    record
                    for record in exact_block_records
                    if record[2].orientation_reference_signed_z <= epsilon
                ]
                if is_mirror_rna_molecule(
                    len(nonpositive_records),
                    len(exact_block_records),
                    threshold=mirror_threshold,
                ):
                    counts["rejected_mirror_molecules"] += 1
                    counts["rejected_mirror_molecule_blocks"] += len(molecule_blocks)
                    counts["rejected_mirror_exact_backbone_blocks"] += len(
                        exact_block_records
                    )
                    source_counts["rejected_mirror_molecules"] += 1
                    source_counts["rejected_mirror_molecule_blocks"] += len(
                        molecule_blocks
                    )
                    source_counts["rejected_mirror_exact_backbone_blocks"] += len(
                        exact_block_records
                    )
                    mirror_rejections.append(
                        {
                            "source_pkl": str(source_pkl.resolve()),
                            "molecule_id": molecule_id,
                            "blocks": len(molecule_blocks),
                            "exact_backbone_blocks": len(exact_block_records),
                            "nonpositive_orientation_blocks": len(
                                nonpositive_records
                            ),
                            "nonpositive_fraction": (
                                len(nonpositive_records) / len(exact_block_records)
                            ),
                        }
                    )
                    continue

                counts["retained_molecules"] += 1
                counts["retained_blocks"] += len(molecule_blocks)
                source_counts["retained_molecules"] += 1
                source_counts["retained_blocks"] += len(molecule_blocks)
                exact_by_index = {
                    block_index: (block, frame)
                    for block_index, block, frame in exact_block_records
                }

                for block_index, block in enumerate(molecule_blocks):
                    exact_record = exact_by_index.get(block_index)
                    if exact_record is None:
                        missing, duplicates = rna_backbone_contract_violations(
                            atom.pos_code for atom in block.atoms
                        )
                        counts["quarantined_backbone_contract_blocks"] += 1
                        source_counts["quarantined_backbone_contract_blocks"] += 1
                        quarantine_patterns[
                            (missing, tuple(sorted(duplicates.items())))
                        ] += 1
                        continue

                    _, provisional_frame = exact_record
                    if provisional_frame.orientation_reference_signed_z <= epsilon:
                        counts["quarantined_noncanonical_orientation_blocks"] += 1
                        source_counts[
                            "quarantined_noncanonical_orientation_blocks"
                        ] += 1
                        if len(orientation_quarantine_examples) < 100:
                            orientation_quarantine_examples.append(
                                {
                                    "source_pkl": str(source_pkl.resolve()),
                                    "molecule_id": molecule_id,
                                    "block_index": block_index,
                                    "block_symbol": str(
                                        getattr(block, "symbol", "")
                                    ),
                                    "orientation_reference_signed_z": (
                                        provisional_frame.orientation_reference_signed_z
                                    ),
                                }
                            )
                        continue

                    counts["accepted_target_blocks"] += 1
                    source_counts["accepted_target_blocks"] += 1
                    for candidate in candidates.values():
                        candidate.observe(
                            block.atoms,
                            source_pkl=source_pkl,
                            molecule_id=molecule_id,
                            block_index=block_index,
                            block_symbol=str(getattr(block, "symbol", "")),
                            epsilon=epsilon,
                        )

                if progress_every and counts["input_molecules"] % progress_every == 0:
                    print(
                        "[RNA D/L policy simulation] "
                        f"input={counts['input_molecules']:,} "
                        f"p_only={counts['rejected_p_only_molecules']:,} "
                        f"mirror={counts['rejected_mirror_molecules']:,} "
                        f"accepted={counts['accepted_target_blocks']:,}",
                        flush=True,
                    )

        source_summaries.append(
            {"source_pkl": str(source_pkl.resolve()), "counts": dict(source_counts)}
        )

    after = [_stat_fingerprint(path) for path in source_pkls]
    candidate_reports = {
        name: candidate.report() for name, candidate in candidates.items()
    }
    selected = select_numerically_dominant_candidate(candidate_reports)
    quarantine_total = (
        counts["quarantined_backbone_contract_blocks"]
        + counts["quarantined_noncanonical_orientation_blocks"]
    )
    quarantine_percent = (
        100.0 * quarantine_total / counts["retained_blocks"]
        if counts["retained_blocks"]
        else 0.0
    )
    selected_nonpositive = (
        candidate_reports[selected]["counts"].get(
            "orientation_reference_negative", 0
        )
        + candidate_reports[selected]["counts"].get(
            "orientation_reference_near_plane", 0
        )
        if selected
        else -1
    )

    expected_counts = {
        "input_molecules": 9783,
        "rejected_p_only_molecules": 55,
        "rejected_mirror_molecules": 3,
        "retained_molecules": 9725,
        "retained_blocks": 852098,
        "quarantined_backbone_contract_blocks": 5670,
        "quarantined_noncanonical_orientation_blocks": 68,
        "accepted_target_blocks": 846360,
    }
    stop_conditions = {
        "expected_count_mismatch": any(
            counts[key] != expected for key, expected in expected_counts.items()
        ),
        "unexpected_mirror_molecule_ids": {
            record["molecule_id"] for record in mirror_rejections
        }
        != {"4r8i_B", "4wb3_D", "4wb3_E"},
        "quarantine_above_2_5_percent": quarantine_percent > 2.5,
        "accepted_candidate_coverage_below_100_percent": any(
            report["accepted_block_valid_percent"] != 100.0
            for report in candidate_reports.values()
        ),
        "atom_order_changes_frame": any(
            report["counts"].get("atom_order_invariance_failures", 0) > 0
            for report in candidate_reports.values()
        ),
        "non_right_handed_frame": any(
            report["counts"].get("non_right_handed_frames", 0) > 0
            for report in candidate_reports.values()
        ),
        "selected_candidate_changed": selected != "c2_c3_c4",
        "selected_candidate_still_has_noncanonical_orientation": (
            selected_nonpositive != 0
        ),
        "source_fingerprint_changed": before != after,
    }

    return {
        "stage": "stage1.7_rna_chirality_policy_simulation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": "training_source_pkl_streams_read_only",
        "epsilon_angstrom": epsilon,
        "policy": {
            "p_only": "reject entire molecule if any block is P-only",
            "mirror_molecule": (
                "reject entire molecule when more than 50% of exact-backbone "
                "blocks have nonpositive O2' signed-z in the provisional "
                "C2'-C3'-C4' frame"
            ),
            "local_orientation_outlier": (
                "retain molecule/block as graph context but exclude block from "
                "template and MLM target eligibility"
            ),
            "mirror_nonpositive_fraction_threshold": mirror_threshold,
        },
        "counts": dict(counts),
        "expected_counts": expected_counts,
        "quarantine_total_blocks": quarantine_total,
        "quarantine_percent_of_retained_blocks": quarantine_percent,
        "mirror_rejections": mirror_rejections,
        "orientation_quarantine_examples_first_100": (
            orientation_quarantine_examples
        ),
        "backbone_quarantine_patterns": [
            {
                "count": count,
                "missing": list(missing),
                "duplicates": dict(duplicates),
            }
            for (missing, duplicates), count in quarantine_patterns.most_common(25)
        ],
        "candidate_role_order": ["plane_anchor", "origin", "positive_x_anchor"],
        "orientation_reference": ORIENTATION_REFERENCE,
        "candidates": candidate_reports,
        "selection_rule": (
            "Among candidates with 100% accepted coverage, zero atom-order "
            "failures, and right-handed frames, select the unique candidate "
            "with both the largest worst-case perpendicular height and the "
            "largest worst-case sin(angle)."
        ),
        "selected_candidate": selected,
        "selected_anchors_plane_origin_x": (
            list(SELECTION_CANDIDATES[selected]) if selected else None
        ),
        "selected_candidate_nonpositive_orientation_blocks": selected_nonpositive,
        "stop_conditions": stop_conditions,
        "go": not any(stop_conditions.values()),
        "selection_made": selected is not None and not any(stop_conditions.values()),
        "source_summaries": source_summaries,
        "source_fingerprints_before": before,
        "source_fingerprints_after": after,
        "production_cif_pkl_pt_files_rewritten": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pkl", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--simulate-canonical-chirality-policy",
        action="store_true",
        help=(
            "Apply the proposed D/L policy in memory before repeating anchor "
            "selection; source files remain read-only"
        ),
    )
    args = parser.parse_args()

    if args.simulate_canonical_chirality_policy:
        report = audit_chirality_policy_simulation(
            args.source_pkl,
            epsilon=args.epsilon,
            progress_every=args.progress_every,
        )
    else:
        report = audit_anchor_selection(
            args.source_pkl,
            epsilon=args.epsilon,
            progress_every=args.progress_every,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    if not report["go"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
