#!/usr/bin/env python3
"""Trace RNA backbone quarantine patterns to molecules and source CIFs.

This is a read-only audit. It never modifies source PKLs, CIFs, or production
PyG caches.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data_loading.data_types as _data_types_module
from data_loading.atom_position_vocab import (
    is_p_only_rna_block,
    normalize_rna_position_code,
    rna_backbone_contract_violations,
)


# Historical pickle streams may reference the pre-package module name.
sys.modules.setdefault("data_types", _data_types_module)


SUGAR_FRAME_ANCHORS = ("C1'", "C2'", "C3'")
MISSING_PHOSPHATE = ("P", "OP1", "OP2")
MISSING_O2 = ("O2'",)
P_ONLY_MISSING = (
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
)


def classify_block(atoms, block_index: int) -> tuple[str, tuple[str, ...], dict[str, int]]:
    missing, duplicates = rna_backbone_contract_violations(
        atom.pos_code for atom in atoms
    )
    if not missing and not duplicates:
        return "exact_backbone", missing, duplicates
    if is_p_only_rna_block(atoms):
        return "p_only", missing, duplicates
    if missing == MISSING_PHOSPHATE and not duplicates:
        category = (
            "missing_phosphate_at_block_zero"
            if block_index == 0
            else "missing_phosphate_internal"
        )
        return category, missing, duplicates
    if missing == MISSING_O2 and not duplicates:
        return "missing_o2_only", missing, duplicates
    if missing == P_ONLY_MISSING and not duplicates:
        return "p_plus_nonbackbone_only", missing, duplicates
    if duplicates:
        return "duplicate_backbone", missing, duplicates
    return "other_missing_backbone", missing, duplicates


def has_finite_unique_sugar_frame(atoms) -> bool:
    positions: dict[str, list[np.ndarray]] = {}
    for atom in atoms:
        code = normalize_rna_position_code(atom.pos_code)
        if code in SUGAR_FRAME_ANCHORS:
            positions.setdefault(code, []).append(
                np.asarray(atom.position, dtype=np.float64)
            )
    if any(len(positions.get(code, ())) != 1 for code in SUGAR_FRAME_ANCHORS):
        return False
    plane, origin, x_anchor = (
        positions[code][0] for code in SUGAR_FRAME_ANCHORS
    )
    if not all(np.isfinite(point).all() for point in (plane, origin, x_anchor)):
        return False
    origin_to_plane = plane - origin
    origin_to_x = x_anchor - origin
    x_norm = float(np.linalg.norm(origin_to_x))
    if x_norm <= 1.0e-6:
        return False
    x_axis = origin_to_x / x_norm
    perpendicular = origin_to_plane - np.dot(origin_to_plane, x_axis) * x_axis
    return float(np.linalg.norm(perpendicular)) > 1.0e-6


def _find_source_cif(raw_root: Path | None, molecule_id: str) -> Path | None:
    if raw_root is None:
        return None
    exact = raw_root / f"{molecule_id}.cif"
    if exact.exists():
        return exact
    pdb_id = molecule_id[:4]
    suffix = molecule_id[5:] if len(molecule_id) > 5 else ""
    cleaned_suffix = suffix.strip("-")
    candidates = []
    if cleaned_suffix:
        candidates.extend(
            [
                raw_root / f"{pdb_id}_{cleaned_suffix}.cif",
                raw_root / f"{pdb_id}_{cleaned_suffix.upper()}.cif",
                raw_root / f"{pdb_id}_{cleaned_suffix.lower()}.cif",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pdb_matches = sorted(raw_root.glob(f"{pdb_id}_*.cif"))
    return pdb_matches[0] if len(pdb_matches) == 1 else None


def _raw_cif_summary(path: Path | None) -> dict:
    if path is None:
        return {"source_cif_found": False}
    method = None
    resolution = None
    atom_rows = 0
    position_counts: Counter[str] = Counter()
    with path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("_exptl.method "):
                method = stripped.removeprefix("_exptl.method ").strip("'\"")
            elif stripped.startswith("_reflns.d_resolution_high "):
                resolution = stripped.split(maxsplit=1)[1]
            elif stripped.startswith(("ATOM ", "HETATM ")):
                fields = stripped.split()
                atom_rows += 1
                if len(fields) > 3:
                    position_counts[normalize_rna_position_code(fields[3])] += 1
    return {
        "source_cif_found": True,
        "source_cif": str(path.resolve()),
        "experimental_method": method,
        "resolution_angstrom_raw": resolution,
        "atom_rows": atom_rows,
        "position_counts": dict(position_counts.most_common()),
    }


def audit(source_pkls: list[Path], raw_root: Path | None, max_examples: int) -> dict:
    counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_molecules: dict[str, set[str]] = {}
    molecule_rows: list[dict] = []
    examples: dict[str, list[dict]] = {}
    source_summaries: list[dict] = []
    excluded_p_only_structure_blocks = 0
    excluded_p_only_structure_anchor_failures = 0
    excluded_any_p_only_molecule_blocks = 0
    excluded_any_p_only_molecule_anchor_failures = 0
    excluded_any_p_only_molecule_quarantined_blocks = 0

    for source_index, pkl_path in enumerate(source_pkls):
        source_counts: Counter[str] = Counter()
        with pkl_path.open("rb") as handle:
            while True:
                try:
                    molecule = pickle.load(handle)
                except EOFError:
                    break
                counts["molecules"] += 1
                source_counts["molecules"] += 1
                molecule_category_counts: Counter[str] = Counter()
                anchor_eligible = 0
                for block_index, block in enumerate(molecule.blocks):
                    counts["blocks"] += 1
                    source_counts["blocks"] += 1
                    category, missing, duplicates = classify_block(
                        block.atoms, block_index
                    )
                    category_counts[category] += 1
                    molecule_category_counts[category] += 1
                    category_molecules.setdefault(category, set()).add(str(molecule.id))
                    if has_finite_unique_sugar_frame(block.atoms):
                        anchor_eligible += 1
                        counts["sugar_frame_eligible_blocks"] += 1
                    else:
                        counts["sugar_frame_ineligible_blocks"] += 1
                    category_examples = examples.setdefault(category, [])
                    if len(category_examples) < max_examples:
                        category_examples.append(
                            {
                                "source_index": source_index,
                                "molecule_id": str(molecule.id),
                                "block_index": block_index,
                                "blocks_in_molecule": len(molecule.blocks),
                                "symbol": str(block.symbol),
                                "atom_count": len(block.atoms),
                                "position_codes": [
                                    normalize_rna_position_code(atom.pos_code)
                                    for atom in block.atoms
                                ],
                                "missing": list(missing),
                                "duplicates": duplicates,
                            }
                        )

                block_count = len(molecule.blocks)
                all_p_only = (
                    block_count > 0
                    and molecule_category_counts["p_only"] == block_count
                )
                all_missing_o2 = (
                    block_count > 0
                    and molecule_category_counts["missing_o2_only"] == block_count
                )
                if all_p_only:
                    counts["all_p_only_molecules"] += 1
                    excluded_p_only_structure_blocks += block_count
                    excluded_p_only_structure_anchor_failures += block_count - anchor_eligible
                has_p_only = molecule_category_counts["p_only"] > 0
                if has_p_only:
                    counts["molecules_with_p_only"] += 1
                    excluded_any_p_only_molecule_blocks += block_count
                    excluded_any_p_only_molecule_anchor_failures += (
                        block_count - anchor_eligible
                    )
                if all_missing_o2:
                    counts["all_missing_o2_only_molecules"] += 1
                quarantined = block_count - molecule_category_counts["exact_backbone"]
                if has_p_only:
                    excluded_any_p_only_molecule_quarantined_blocks += quarantined
                if quarantined:
                    counts["molecules_with_quarantine"] += 1
                    molecule_rows.append(
                        {
                            "source_index": source_index,
                            "molecule_id": str(molecule.id),
                            "blocks": block_count,
                            "quarantined_blocks": quarantined,
                            "sugar_frame_eligible_blocks": anchor_eligible,
                            "all_p_only": all_p_only,
                            "all_missing_o2_only": all_missing_o2,
                            "category_counts": dict(molecule_category_counts),
                        }
                    )
        source_summaries.append(
            {
                "source_index": source_index,
                "path": str(pkl_path.resolve()),
                "counts": dict(source_counts),
            }
        )

    molecule_rows.sort(
        key=lambda row: (row["quarantined_blocks"], row["blocks"]), reverse=True
    )
    for category, category_examples in examples.items():
        for example in category_examples:
            source_cif = _find_source_cif(raw_root, example["molecule_id"])
            example["raw_cif"] = _raw_cif_summary(source_cif)

    retained_blocks = counts["blocks"] - excluded_p_only_structure_blocks
    retained_anchor_failures = (
        counts["sugar_frame_ineligible_blocks"]
        - excluded_p_only_structure_anchor_failures
    )
    quarantined_blocks = counts["blocks"] - category_counts["exact_backbone"]
    retained_exact_quarantine = quarantined_blocks - excluded_p_only_structure_blocks
    retained_any_p_only_blocks = counts["blocks"] - excluded_any_p_only_molecule_blocks
    retained_any_p_only_anchor_failures = (
        counts["sugar_frame_ineligible_blocks"]
        - excluded_any_p_only_molecule_anchor_failures
    )
    retained_any_p_only_exact_quarantine = (
        quarantined_blocks - excluded_any_p_only_molecule_quarantined_blocks
    )
    return {
        "stage": "stage1.3_rna_quarantine_provenance_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pkls": [str(path.resolve()) for path in source_pkls],
        "raw_root": str(raw_root.resolve()) if raw_root else None,
        "counts": dict(counts),
        "category_block_counts": dict(category_counts.most_common()),
        "category_unique_molecule_counts": {
            category: len(molecules)
            for category, molecules in sorted(category_molecules.items())
        },
        "source_summaries": source_summaries,
        "policy_simulation": {
            "description": (
                "Exclude molecules whose every parsed block is exactly one P atom; "
                "then test C1'-C2'-C3' frame eligibility. No source data was changed."
            ),
            "excluded_all_p_only_molecules": counts["all_p_only_molecules"],
            "excluded_blocks": excluded_p_only_structure_blocks,
            "retained_blocks": retained_blocks,
            "retained_sugar_frame_ineligible_blocks": retained_anchor_failures,
            "retained_sugar_frame_ineligible_percent": (
                100.0 * retained_anchor_failures / retained_blocks
                if retained_blocks
                else 0.0
            ),
            "retained_exact_backbone_quarantined_blocks": retained_exact_quarantine,
            "retained_exact_backbone_quarantine_percent": (
                100.0 * retained_exact_quarantine / retained_blocks
                if retained_blocks
                else 0.0
            ),
        },
        "recommended_policy_simulation": {
            "description": (
                "Exclude an entire molecule when any parsed nucleotide is exactly "
                "one P atom; retain the conservative exact-backbone target rule. "
                "No source data was changed."
            ),
            "excluded_molecules_with_any_p_only_block": counts["molecules_with_p_only"],
            "excluded_blocks": excluded_any_p_only_molecule_blocks,
            "retained_blocks": retained_any_p_only_blocks,
            "retained_sugar_frame_ineligible_blocks": (
                retained_any_p_only_anchor_failures
            ),
            "retained_sugar_frame_ineligible_percent": (
                100.0
                * retained_any_p_only_anchor_failures
                / retained_any_p_only_blocks
                if retained_any_p_only_blocks
                else 0.0
            ),
            "retained_exact_backbone_quarantined_blocks": (
                retained_any_p_only_exact_quarantine
            ),
            "retained_exact_backbone_quarantine_percent": (
                100.0
                * retained_any_p_only_exact_quarantine
                / retained_any_p_only_blocks
                if retained_any_p_only_blocks
                else 0.0
            ),
        },
        "most_affected_molecules": molecule_rows[:250],
        "examples_by_category": examples,
        "production_files_rewritten": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pkl", type=Path, nargs="+", required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    report = audit(args.source_pkl, args.raw_root, args.max_examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
