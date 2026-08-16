#!/usr/bin/env python3
"""Read-only coverage and conditioning audit for RNA local-frame anchors."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_loading.adapters.rna_adapter import RNAAdapter
from data_loading.atom_position_vocab import (
    normalize_rna_position_code,
    rna_backbone_contract_violations,
)
import data_loading.data_types as _data_types_module


# Historical pickle streams may reference the pre-package module name.
sys.modules.setdefault("data_types", _data_types_module)


ANCHOR_CANDIDATES = {
    # Tuple order is (plane anchor, origin, +x anchor), matching the role order
    # used by protein N-CA-C without claiming that any candidate is selected.
    "c1_c2_c3": ("C1'", "C2'", "C3'"),
    "o4_c1_c2": ("O4'", "C1'", "C2'"),
    "c1_o4_c4": ("C1'", "O4'", "C4'"),
    "c2_c3_c4": ("C2'", "C3'", "C4'"),
    "c3_c4_c5": ("C3'", "C4'", "C5'"),
    "c4_c5_o5": ("C4'", "C5'", "O5'"),
    "op1_p_op2": ("OP1", "P", "OP2"),
}


@dataclass
class ReservoirSummary:
    """Exact count/min/max/mean plus deterministic sampled quantiles."""

    capacity: int = 100_000
    seed: int = 0
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    sample: list[float] = field(default_factory=list)

    def __post_init__(self):
        self._random = random.Random(self.seed)

    def add(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        if len(self.sample) < self.capacity:
            self.sample.append(value)
            return
        replacement = self._random.randrange(self.count)
        if replacement < self.capacity:
            self.sample[replacement] = value

    def report(self) -> dict:
        if not self.count:
            return {"count": 0}
        values = np.asarray(self.sample, dtype=np.float64)
        quantile_levels = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
        quantiles = np.quantile(values, quantile_levels)
        return {
            "count": self.count,
            "mean": self.total / self.count,
            "min": self.minimum,
            "q0_1": float(quantiles[0]),
            "q1": float(quantiles[1]),
            "q5": float(quantiles[2]),
            "median": float(quantiles[3]),
            "q95": float(quantiles[4]),
            "q99": float(quantiles[5]),
            "q99_9": float(quantiles[6]),
            "max": self.maximum,
            "quantile_sample_size": len(self.sample),
        }


class CandidateStats:
    def __init__(self, name: str, anchors: tuple[str, str, str], epsilon: float):
        self.name = name
        self.anchors = anchors
        self.epsilon = epsilon
        self.counts: Counter[str] = Counter()
        seed_offset = sum(map(ord, name))
        self.metrics = {
            "origin_to_plane_anchor_angstrom": ReservoirSummary(seed=seed_offset + 1),
            "origin_to_x_anchor_angstrom": ReservoirSummary(seed=seed_offset + 2),
            "perpendicular_height_angstrom": ReservoirSummary(seed=seed_offset + 3),
            "angle_degrees": ReservoirSummary(seed=seed_offset + 4),
            "sin_angle": ReservoirSummary(seed=seed_offset + 5),
        }

    def observe(self, positions: dict[str, list[np.ndarray]], accepted: bool) -> None:
        self.counts["blocks_seen"] += 1
        if accepted:
            self.counts["accepted_blocks_seen"] += 1

        missing = [name for name in self.anchors if len(positions.get(name, ())) == 0]
        duplicate = [name for name in self.anchors if len(positions.get(name, ())) > 1]
        if missing:
            self.counts["missing_anchor_blocks"] += 1
            if accepted:
                self.counts["accepted_missing_anchor_blocks"] += 1
            return
        if duplicate:
            self.counts["duplicate_anchor_blocks"] += 1
            if accepted:
                self.counts["accepted_duplicate_anchor_blocks"] += 1
            return

        plane_name, origin_name, x_name = self.anchors
        plane = positions[plane_name][0]
        origin = positions[origin_name][0]
        x_anchor = positions[x_name][0]
        if not (
            np.isfinite(plane).all()
            and np.isfinite(origin).all()
            and np.isfinite(x_anchor).all()
        ):
            self.counts["nonfinite_anchor_blocks"] += 1
            if accepted:
                self.counts["accepted_nonfinite_anchor_blocks"] += 1
            return

        origin_to_plane = plane - origin
        origin_to_x = x_anchor - origin
        plane_norm = float(np.linalg.norm(origin_to_plane))
        x_norm = float(np.linalg.norm(origin_to_x))
        if plane_norm <= self.epsilon or x_norm <= self.epsilon:
            self.counts["coincident_anchor_blocks"] += 1
            if accepted:
                self.counts["accepted_coincident_anchor_blocks"] += 1
            return

        x_axis = origin_to_x / x_norm
        perpendicular = origin_to_plane - np.dot(origin_to_plane, x_axis) * x_axis
        perpendicular_height = float(np.linalg.norm(perpendicular))
        cosine = float(
            np.clip(np.dot(origin_to_plane, origin_to_x) / (plane_norm * x_norm), -1.0, 1.0)
        )
        angle_degrees = math.degrees(math.acos(cosine))
        sin_angle = perpendicular_height / plane_norm
        if perpendicular_height <= self.epsilon:
            self.counts["collinear_anchor_blocks"] += 1
            if accepted:
                self.counts["accepted_collinear_anchor_blocks"] += 1
            return

        self.counts["valid_frame_blocks"] += 1
        if accepted:
            self.counts["accepted_valid_frame_blocks"] += 1
        for key, value in (
            ("origin_to_plane_anchor_angstrom", plane_norm),
            ("origin_to_x_anchor_angstrom", x_norm),
            ("perpendicular_height_angstrom", perpendicular_height),
            ("angle_degrees", angle_degrees),
            ("sin_angle", sin_angle),
        ):
            self.metrics[key].add(value)

    def report(self) -> dict:
        total = self.counts["blocks_seen"]
        accepted = self.counts["accepted_blocks_seen"]
        valid = self.counts["valid_frame_blocks"]
        accepted_valid = self.counts["accepted_valid_frame_blocks"]
        return {
            "anchors_plane_origin_x": list(self.anchors),
            "counts": dict(self.counts),
            "all_block_valid_percent": 100.0 * valid / total if total else 0.0,
            "accepted_block_valid_percent": (
                100.0 * accepted_valid / accepted if accepted else 0.0
            ),
            "geometry": {key: summary.report() for key, summary in self.metrics.items()},
        }


def _position_map(atoms: Iterable) -> dict[str, list[np.ndarray]]:
    positions: dict[str, list[np.ndarray]] = {}
    for atom in atoms:
        position_code = normalize_rna_position_code(atom.pos_code)
        positions.setdefault(position_code, []).append(
            np.asarray(atom.position, dtype=np.float64)
        )
    return positions


def _observe_backbone_contract(
    atoms: Iterable,
    counts: Counter[str],
    missing_position_counts: Counter[str],
    duplicate_position_counts: Counter[str],
    violation_patterns: Counter[tuple],
) -> bool:
    """Record the same normalized backbone contract used by production code."""
    missing, duplicates = rna_backbone_contract_violations(
        atom.pos_code for atom in atoms
    )
    accepted = not missing and not duplicates
    counts[
        "accepted_exact_backbone_blocks" if accepted else "quarantined_blocks"
    ] += 1
    if accepted:
        return True

    missing_position_counts.update(missing)
    duplicate_position_counts.update(duplicates.keys())
    violation_patterns[(missing, tuple(sorted(duplicates.items())))] += 1
    return False


def _contract_diagnostics(
    missing_position_counts: Counter[str],
    duplicate_position_counts: Counter[str],
    violation_patterns: Counter[tuple],
) -> dict:
    return {
        "missing_backbone_position_counts": dict(
            missing_position_counts.most_common()
        ),
        "duplicate_backbone_position_counts": dict(
            duplicate_position_counts.most_common()
        ),
        "top_violation_patterns": [
            {
                "count": count,
                "missing": list(missing),
                "duplicates": dict(duplicates),
            }
            for (missing, duplicates), count in violation_patterns.most_common(25)
        ],
    }


def audit_rna_anchors(
    root: Path,
    *,
    max_files: int | None = None,
    epsilon: float = 1.0e-6,
    progress_every: int = 250,
) -> dict:
    adapter = RNAAdapter(
        include_modified=True,
        include_ions=False,
        use_23atom_uniform=False,
    )
    files = adapter.load_raw_data(str(root), max_samples=max_files)
    candidates = {
        name: CandidateStats(name, anchors, epsilon)
        for name, anchors in ANCHOR_CANDIDATES.items()
    }
    counts: Counter[str] = Counter()
    nucleotide_counts: Counter[str] = Counter()
    missing_position_counts: Counter[str] = Counter()
    duplicate_position_counts: Counter[str] = Counter()
    violation_patterns: Counter[tuple] = Counter()

    for file_index, path in enumerate(files, start=1):
        blocks = adapter.create_blocks(path)
        counts["files_scanned"] += 1
        if not blocks:
            counts["files_without_parsed_rna_blocks"] += 1
        for block in blocks:
            counts["blocks"] += 1
            nucleotide_counts[str(block.symbol)] += 1
            positions = _position_map(block.atoms)
            accepted = _observe_backbone_contract(
                block.atoms,
                counts,
                missing_position_counts,
                duplicate_position_counts,
                violation_patterns,
            )
            for candidate in candidates.values():
                candidate.observe(positions, accepted)
        if progress_every and file_index % progress_every == 0:
            print(
                f"[RNA anchor audit] files={file_index:,}/{len(files):,} "
                f"blocks={counts['blocks']:,} accepted={counts['accepted_exact_backbone_blocks']:,}",
                flush=True,
            )

    total_blocks = counts["blocks"]
    accepted_blocks = counts["accepted_exact_backbone_blocks"]
    return {
        "stage": "stage1.2_rna_anchor_candidate_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "epsilon_angstrom": epsilon,
        "max_files": max_files,
        "counts": dict(counts),
        "accepted_exact_backbone_percent": (
            100.0 * accepted_blocks / total_blocks if total_blocks else 0.0
        ),
        "nucleotide_counts": dict(nucleotide_counts.most_common()),
        "backbone_contract_diagnostics": _contract_diagnostics(
            missing_position_counts,
            duplicate_position_counts,
            violation_patterns,
        ),
        "candidate_order_semantics": ["plane_anchor", "origin", "positive_x_anchor"],
        "candidates": {name: candidate.report() for name, candidate in candidates.items()},
        "selection_made": False,
        "production_pt_files_rewritten": False,
    }


def audit_rna_anchor_pkls(
    source_pkls: list[Path],
    *,
    epsilon: float = 1.0e-6,
    progress_every: int = 500,
) -> dict:
    """Audit every molecule/block in the exact source streams used for caches."""
    candidates = {
        name: CandidateStats(name, anchors, epsilon)
        for name, anchors in ANCHOR_CANDIDATES.items()
    }
    counts: Counter[str] = Counter()
    nucleotide_counts: Counter[str] = Counter()
    missing_position_counts: Counter[str] = Counter()
    duplicate_position_counts: Counter[str] = Counter()
    violation_patterns: Counter[tuple] = Counter()
    source_summaries: list[dict] = []

    for pkl_path in source_pkls:
        source_counts: Counter[str] = Counter()
        with pkl_path.open("rb") as handle:
            while True:
                try:
                    molecule = pickle.load(handle)
                except EOFError:
                    break
                counts["molecules"] += 1
                source_counts["molecules"] += 1
                for block in molecule.blocks:
                    counts["blocks"] += 1
                    source_counts["blocks"] += 1
                    nucleotide_counts[str(block.symbol)] += 1
                    positions = _position_map(block.atoms)
                    accepted = _observe_backbone_contract(
                        block.atoms,
                        counts,
                        missing_position_counts,
                        duplicate_position_counts,
                        violation_patterns,
                    )
                    source_counts[
                        "accepted_exact_backbone_blocks"
                        if accepted
                        else "quarantined_blocks"
                    ] += 1
                    for candidate in candidates.values():
                        candidate.observe(positions, accepted)
                if progress_every and counts["molecules"] % progress_every == 0:
                    print(
                        f"[RNA anchor PKL audit] molecules={counts['molecules']:,} "
                        f"blocks={counts['blocks']:,} "
                        f"accepted={counts['accepted_exact_backbone_blocks']:,}",
                        flush=True,
                    )
        counts["source_pkl_files"] += 1
        source_blocks = source_counts["blocks"]
        source_accepted = source_counts["accepted_exact_backbone_blocks"]
        source_summaries.append(
            {
                "path": str(pkl_path.resolve()),
                "counts": dict(source_counts),
                "accepted_exact_backbone_percent": (
                    100.0 * source_accepted / source_blocks if source_blocks else 0.0
                ),
            }
        )

    total_blocks = counts["blocks"]
    accepted_blocks = counts["accepted_exact_backbone_blocks"]
    return {
        "stage": "stage1.2_rna_anchor_candidate_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": "training_source_pkl_streams",
        "source_pkls": [str(path.resolve()) for path in source_pkls],
        "epsilon_angstrom": epsilon,
        "counts": dict(counts),
        "accepted_exact_backbone_percent": (
            100.0 * accepted_blocks / total_blocks if total_blocks else 0.0
        ),
        "nucleotide_counts": dict(nucleotide_counts.most_common()),
        "source_summaries": source_summaries,
        "backbone_contract_diagnostics": _contract_diagnostics(
            missing_position_counts,
            duplicate_position_counts,
            violation_patterns,
        ),
        "candidate_order_semantics": ["plane_anchor", "origin", "positive_x_anchor"],
        "candidates": {name: candidate.report() for name, candidate in candidates.items()},
        "selection_made": False,
        "production_pt_files_rewritten": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--root", type=Path)
    sources.add_argument("--source-pkl", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    if args.source_pkl:
        report = audit_rna_anchor_pkls(
            args.source_pkl,
            epsilon=args.epsilon,
            progress_every=args.progress_every,
        )
    else:
        report = audit_rna_anchors(
            args.root,
            max_files=args.max_files,
            epsilon=args.epsilon,
            progress_every=args.progress_every,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
