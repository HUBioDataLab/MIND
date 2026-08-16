#!/usr/bin/env python3
"""Stage 2 geometry-only leakage probes for fixed virtual-atom templates.

The script streams the same protein/RNA UniversalMolecule sources used by the
Stage 1 geometry audit.  Molecules are assigned deterministically to train or
validation before block sampling, so residues from one structure can never
cross the split boundary.  Production CIF/PKL/PT files are read-only.

The absolute leakage gate uses only the fixed template, its ordered slots, and
its internal radius-graph signature.  Features involving the three real frame
anchors or the full visible backbone are evaluated as matched baseline versus
template-augmented pairs: real backbone conformation may legitimately
correlate with block identity, so its absolute accuracy is not called template
leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

import data_loading.data_types as _data_types_module
from core.local_frame import (
    LocalFrameError,
    protein_n_ca_c_frame,
    rna_c2_c3_c4_frame,
)
from core.virtual_atom_templates import (
    protein_sidechain_template,
    rna_base_template,
)
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE,
    RNA_BACKBONE_ATOMS,
    normalize_rna_position_code,
    rna_backbone_contract_violations,
    rna_molecule_contains_p_only_block,
)
from data_loading.audit_rna_anchor_selection import is_mirror_rna_molecule


sys.modules.setdefault("data_types", _data_types_module)

PROTEIN_CLASSES = tuple(sorted(PROTEIN_SIDECHAIN_ATOMS_BY_RESIDUE))
RNA_CLASSES = ("A", "C", "G", "U")
SPLIT_SEED = 20_260_804
VALIDATION_FRACTION = 0.20
SPATIAL_CUTOFF_ANGSTROM = 5.0
CHIRALITY_EPSILON_ANGSTROM = 1.0e-6

GATE_FEATURES = (
    "template_xyz",
    "template_slot_order",
    "template_internal_graph",
)
CONTEXT_CONTROL_FEATURES = (
    "frame_anchors_only",
    "frame_anchors_plus_template_relative",
    "visible_backbone_only",
    "visible_backbone_plus_template_graph",
)


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


def stable_group_split(
    domain: str,
    molecule_id: str,
    *,
    seed: int = SPLIT_SEED,
    validation_fraction: float = VALIDATION_FRACTION,
) -> str:
    """Assign a whole molecule reproducibly to train or validation."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    payload = f"{seed}|{domain}|{molecule_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    fraction = value / float(1 << 64)
    return "validation" if fraction < validation_fraction else "train"


class ClassBalancedReservoir:
    """Bounded deterministic per-class/per-split reservoir of backbone tables."""

    def __init__(self, classes: tuple[str, ...], caps: dict[str, int], domain: str):
        self.classes = classes
        self.caps = caps
        self.domain = domain
        self.records = {
            split: {label: [] for label in classes} for split in caps
        }
        self.seen = {split: Counter() for split in caps}
        self.groups = {split: set() for split in caps}
        self.rngs = {}
        for split in caps:
            for label in classes:
                seed_payload = f"{SPLIT_SEED}|{domain}|{split}|{label}".encode()
                rng_seed = int.from_bytes(
                    hashlib.sha256(seed_payload).digest()[:8], "big"
                )
                self.rngs[(split, label)] = np.random.default_rng(rng_seed)

    def observe(
        self,
        split: str,
        label: str,
        molecule_id: str,
        backbone_local: np.ndarray,
    ) -> None:
        if label not in self.classes:
            return
        self.groups[split].add(str(molecule_id))
        self.seen[split][label] += 1
        seen = self.seen[split][label]
        target = self.records[split][label]
        cap = self.caps[split]
        value = np.asarray(backbone_local, dtype=np.float32)
        if len(target) < cap:
            target.append(value.copy())
            return
        replacement = int(self.rngs[(split, label)].integers(0, seen))
        if replacement < cap:
            target[replacement] = value.copy()

    def balanced_arrays(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        per_class = min(len(self.records[split][label]) for label in self.classes)
        if per_class == 0:
            raise ValueError(f"{self.domain}/{split} has a class with zero support")
        arrays = []
        labels = []
        for class_index, label in enumerate(self.classes):
            arrays.extend(self.records[split][label][:per_class])
            labels.extend([class_index] * per_class)
        return np.stack(arrays), np.asarray(labels, dtype=np.int64)

    def report(self) -> dict:
        overlap = self.groups["train"] & self.groups["validation"]
        return {
            "seen_blocks_by_split_and_class": {
                split: dict(self.seen[split]) for split in self.caps
            },
            "reservoir_blocks_by_split_and_class": {
                split: {
                    label: len(self.records[split][label]) for label in self.classes
                }
                for split in self.caps
            },
            "molecules_by_split": {
                split: len(self.groups[split]) for split in self.caps
            },
            "molecule_overlap_count": len(overlap),
            "molecule_overlap_examples": sorted(overlap)[:20],
        }


def _upper_triangle_distances(points: np.ndarray) -> np.ndarray:
    row, column = np.triu_indices(points.shape[-2], k=1)
    return np.linalg.norm(points[..., row, :] - points[..., column, :], axis=-1)


def build_feature_matrices(
    domain: str, backbone_local: np.ndarray
) -> dict[str, np.ndarray]:
    """Build gate and diagnostic geometry features without using class labels."""
    backbone = np.asarray(backbone_local, dtype=np.float32)
    if domain == "protein":
        template = protein_sidechain_template().cpu().numpy()
        anchor_indices = (0, 1, 2)  # N, CA, C in PROTEIN_BACKBONE_ATOMS
    elif domain == "rna":
        template = rna_base_template().cpu().numpy()
        anchor_indices = (9, 7, 5)  # C2', C3', C4' in RNA_BACKBONE_ATOMS
    else:
        raise ValueError(f"unsupported domain: {domain}")

    sample_count = backbone.shape[0]
    slot_count = template.shape[0]
    slot_order = np.linspace(0.0, 1.0, slot_count, dtype=np.float32)
    template_pair_distances = _upper_triangle_distances(template)
    template_pair_matrix = np.linalg.norm(
        template[:, None, :] - template[None, :, :], axis=-1
    )
    template_adjacency = (
        (template_pair_matrix <= SPATIAL_CUTOFF_ANGSTROM)
        & ~np.eye(slot_count, dtype=bool)
    )
    template_degrees = template_adjacency.sum(axis=1).astype(np.float32)
    internal_edge_distances = template_pair_distances * (
        template_pair_distances <= SPATIAL_CUTOFF_ANGSTROM
    )

    xyz_vector = template.reshape(-1).astype(np.float32)
    slot_vector = np.concatenate((xyz_vector, slot_order))
    internal_vector = np.concatenate(
        (slot_vector, internal_edge_distances, template_degrees)
    ).astype(np.float32)

    anchors = backbone[:, anchor_indices, :]
    template_to_anchors = np.linalg.norm(
        template[None, :, None, :] - anchors[:, None, :, :], axis=-1
    )
    anchor_pair_distances = _upper_triangle_distances(anchors)
    anchors_only = np.concatenate(
        (anchors.reshape(sample_count, -1), anchor_pair_distances), axis=1
    ).astype(np.float32)
    anchors_plus_template = np.concatenate(
        (
            anchors_only,
            template_to_anchors.reshape(sample_count, -1),
        ),
        axis=1,
    ).astype(np.float32)

    backbone_pair_distances = _upper_triangle_distances(backbone)
    backbone_only = np.concatenate(
        (backbone.reshape(sample_count, -1), backbone_pair_distances), axis=1
    ).astype(np.float32)

    template_to_backbone = np.linalg.norm(
        template[None, :, None, :] - backbone[:, None, :, :], axis=-1
    )
    template_backbone_adjacency = (
        template_to_backbone <= SPATIAL_CUTOFF_ANGSTROM
    )
    visible_edge_distances = np.where(
        template_backbone_adjacency, template_to_backbone, 0.0
    )
    template_visible_degrees = template_backbone_adjacency.sum(axis=2)
    backbone_template_degrees = template_backbone_adjacency.sum(axis=1)
    backbone_plus_template = np.concatenate(
        (
            backbone_only,
            template_backbone_adjacency.reshape(sample_count, -1),
            visible_edge_distances.reshape(sample_count, -1),
            template_visible_degrees,
            backbone_template_degrees,
        ),
        axis=1,
    ).astype(np.float32)

    return {
        "template_xyz": np.broadcast_to(
            xyz_vector, (sample_count, xyz_vector.size)
        ),
        "template_slot_order": np.broadcast_to(
            slot_vector, (sample_count, slot_vector.size)
        ),
        "template_internal_graph": np.broadcast_to(
            internal_vector, (sample_count, internal_vector.size)
        ),
        "frame_anchors_only": anchors_only,
        "frame_anchors_plus_template_relative": anchors_plus_template,
        "visible_backbone_only": backbone_only,
        "visible_backbone_plus_template_graph": backbone_plus_template,
    }


def _fit_and_score(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> dict[str, float | int]:
    if model_name == "linear":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=400,
                solver="lbfgs",
                random_state=SPLIT_SEED,
            ),
        )
    elif model_name == "shallow_extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=64,
            max_depth=8,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SPLIT_SEED,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"unsupported probe model: {model_name}")
    model.fit(train_x, train_y)
    predictions = model.predict(validation_x)
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(validation_y, predictions)
        ),
        "accuracy": float(accuracy_score(validation_y, predictions)),
        "predicted_class_count": int(np.unique(predictions).size),
    }


def evaluate_domain(
    domain: str,
    reservoir: ClassBalancedReservoir,
) -> dict:
    train_backbone, train_y = reservoir.balanced_arrays("train")
    validation_backbone, validation_y = reservoir.balanced_arrays("validation")
    train_features = build_feature_matrices(domain, train_backbone)
    validation_features = build_feature_matrices(domain, validation_backbone)

    results = {}
    for feature_name in (*GATE_FEATURES, *CONTEXT_CONTROL_FEATURES):
        models = ["linear"]
        if feature_name in (
            "template_internal_graph",
            "frame_anchors_only",
            "frame_anchors_plus_template_relative",
            "visible_backbone_only",
            "visible_backbone_plus_template_graph",
        ):
            models.append("shallow_extra_trees")
        model_results = {}
        for model_name in models:
            print(
                f"[{domain} leakage probe] feature={feature_name} "
                f"model={model_name}",
                flush=True,
            )
            model_results[model_name] = _fit_and_score(
                model_name,
                train_features[feature_name],
                train_y,
                validation_features[feature_name],
                validation_y,
            )
        first_row_hashes = {}
        per_class = len(validation_y) // len(reservoir.classes)
        for class_index, label in enumerate(reservoir.classes):
            row = validation_features[feature_name][class_index * per_class]
            first_row_hashes[label] = hashlib.sha256(
                np.ascontiguousarray(row).tobytes()
            ).hexdigest()
        results[feature_name] = {
            "feature_dimension": int(train_features[feature_name].shape[1]),
            "first_validation_row_hash_by_class": first_row_hashes,
            "distinct_first_row_hashes_across_classes": len(
                set(first_row_hashes.values())
            ),
            "models": model_results,
        }

    controlled_pairs = {
        "template_relative_over_frame_anchors": (
            "frame_anchors_only",
            "frame_anchors_plus_template_relative",
        ),
        "template_graph_over_visible_backbone": (
            "visible_backbone_only",
            "visible_backbone_plus_template_graph",
        ),
    }
    controlled_deltas = {}
    for comparison_name, (baseline_name, augmented_name) in controlled_pairs.items():
        baseline_scores = results[baseline_name]["models"]
        augmented_scores = results[augmented_name]["models"]
        controlled_deltas[comparison_name] = {
            model_name: (
                augmented_scores[model_name]["balanced_accuracy"]
                - baseline_scores[model_name]["balanced_accuracy"]
            )
            for model_name in baseline_scores
        }
    return {
        "classes": list(reservoir.classes),
        "chance_balanced_accuracy": 1.0 / len(reservoir.classes),
        "train_samples": int(len(train_y)),
        "validation_samples": int(len(validation_y)),
        "split_and_support": reservoir.report(),
        "feature_results": results,
        "context_controlled_balanced_accuracy_deltas": controlled_deltas,
    }


def collect_protein(
    path: Path, train_cap: int, validation_cap: int, progress_every: int
) -> tuple[ClassBalancedReservoir, dict]:
    reservoir = ClassBalancedReservoir(
        PROTEIN_CLASSES,
        {"train": train_cap, "validation": validation_cap},
        "protein",
    )
    counts = Counter()
    for molecule_index, molecule in enumerate(_iter_pickle(path), start=1):
        counts["input_molecules"] += 1
        molecule_id = str(molecule.id)
        split = stable_group_split("protein", molecule_id)
        for block in molecule.blocks:
            counts["input_blocks"] += 1
            label = str(block.symbol)
            if label not in PROTEIN_CLASSES:
                counts["noncanonical_label_blocks"] += 1
                continue
            positions_by_code = {}
            duplicate = False
            for atom in block.atoms:
                code = str(atom.pos_code)
                if code in PROTEIN_BACKBONE_ATOMS and code in positions_by_code:
                    duplicate = True
                positions_by_code[code] = torch.as_tensor(
                    atom.position, dtype=torch.float64
                )
            if duplicate or any(
                code not in positions_by_code for code in PROTEIN_BACKBONE_ATOMS
            ):
                counts["quarantined_backbone_contract_blocks"] += 1
                continue
            try:
                frame = protein_n_ca_c_frame(
                    {code: positions_by_code[code] for code in ("N", "CA", "C")}
                )
            except LocalFrameError as error:
                counts["frame_failures"] += 1
                counts[f"frame_failure:{error.reason}"] += 1
                continue
            backbone = torch.stack(
                [positions_by_code[code] for code in PROTEIN_BACKBONE_ATOMS]
            )
            local = frame.global_to_local(backbone).cpu().numpy()
            reservoir.observe(split, label, molecule_id, local)
            counts["eligible_probe_blocks"] += 1
        if progress_every and molecule_index % progress_every == 0:
            print(
                f"[protein leakage collection] molecules={molecule_index:,} "
                f"blocks={counts['input_blocks']:,}",
                flush=True,
            )
    return reservoir, dict(counts)


def _rna_block_frame(block):
    codes = [normalize_rna_position_code(atom.pos_code) for atom in block.atoms]
    missing, duplicates = rna_backbone_contract_violations(codes)
    if missing or duplicates:
        return codes, None, None
    positions = torch.tensor(
        [atom.position for atom in block.atoms], dtype=torch.float64
    )
    frame = rna_c2_c3_c4_frame(codes, positions)
    signed_z = float(
        frame.global_to_local(positions[codes.index("O2'")])[2].item()
    )
    return codes, positions, (frame, signed_z)


def collect_rna(
    paths: list[Path], train_cap: int, validation_cap: int, progress_every: int
) -> tuple[ClassBalancedReservoir, dict]:
    reservoir = ClassBalancedReservoir(
        RNA_CLASSES,
        {"train": train_cap, "validation": validation_cap},
        "rna",
    )
    counts = Counter()
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
                    inspected.append((block, codes, None, None))
                    continue
                inspected.append((block, codes, positions, frame_result))
                if frame_result is not None:
                    valid_orientation += 1
                    nonpositive += int(frame_result[1] <= CHIRALITY_EPSILON_ANGSTROM)

            if is_mirror_rna_molecule(nonpositive, valid_orientation):
                counts["rejected_mirror_molecules"] += 1
                counts["rejected_mirror_molecule_blocks"] += block_count
                source_counts["rejected_mirror_molecules"] += 1
                continue

            molecule_id = str(molecule.id)
            split = stable_group_split("rna", molecule_id)
            for block, codes, positions, frame_result in inspected:
                if frame_result is None:
                    counts["quarantined_backbone_or_frame_blocks"] += 1
                    continue
                frame, signed_z = frame_result
                if signed_z <= CHIRALITY_EPSILON_ANGSTROM:
                    counts["quarantined_noncanonical_orientation_blocks"] += 1
                    continue
                counts["accepted_target_policy_blocks"] += 1
                label = str(block.symbol)
                if label not in RNA_CLASSES:
                    counts["noncanonical_label_blocks"] += 1
                    continue
                index_by_code = {code: codes.index(code) for code in RNA_BACKBONE_ATOMS}
                backbone = torch.stack(
                    [positions[index_by_code[code]] for code in RNA_BACKBONE_ATOMS]
                )
                local = frame.global_to_local(backbone).cpu().numpy()
                reservoir.observe(split, label, molecule_id, local)
                counts["eligible_probe_blocks"] += 1
                source_counts["eligible_probe_blocks"] += 1

            if progress_every and molecule_index % progress_every == 0:
                print(
                    f"[RNA leakage collection:{path.name}] "
                    f"molecules={molecule_index:,}",
                    flush=True,
                )
        source_summaries.append(
            {"source_pkl": str(path.resolve()), "counts": dict(source_counts)}
        )
    return reservoir, {"counts": dict(counts), "source_summaries": source_summaries}


def _template_gate_maximum(domain_report: dict) -> float:
    values = []
    for feature_name in GATE_FEATURES:
        for result in domain_report["feature_results"][feature_name]["models"].values():
            values.append(result["balanced_accuracy"])
    return max(values)


def _maximum_positive_context_delta(domain_report: dict) -> float:
    values = [0.0]
    for model_deltas in domain_report[
        "context_controlled_balanced_accuracy_deltas"
    ].values():
        values.extend(model_deltas.values())
    return max(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-pkl", type=Path, required=True)
    parser.add_argument("--rna-pkl", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protein-train-per-class", type=int, default=2_000)
    parser.add_argument("--protein-validation-per-class", type=int, default=500)
    parser.add_argument("--rna-train-per-class", type=int, default=5_000)
    parser.add_argument("--rna-validation-per-class", type=int, default=1_250)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    sources = [args.protein_pkl, *args.rna_pkl]
    before = [_fingerprint(path) for path in sources]
    protein_reservoir, protein_collection = collect_protein(
        args.protein_pkl,
        args.protein_train_per_class,
        args.protein_validation_per_class,
        args.progress_every,
    )
    rna_reservoir, rna_collection = collect_rna(
        args.rna_pkl,
        args.rna_train_per_class,
        args.rna_validation_per_class,
        args.progress_every,
    )
    protein = evaluate_domain("protein", protein_reservoir)
    rna = evaluate_domain("rna", rna_reservoir)
    after = [_fingerprint(path) for path in sources]

    protein_max = _template_gate_maximum(protein)
    rna_max = _template_gate_maximum(rna)
    protein_context_delta = _maximum_positive_context_delta(protein)
    rna_context_delta = _maximum_positive_context_delta(rna)
    protein_hash_leakage = any(
        protein["feature_results"][name]["distinct_first_row_hashes_across_classes"]
        != 1
        for name in ("template_xyz", "template_slot_order", "template_internal_graph")
    )
    rna_hash_leakage = any(
        rna["feature_results"][name]["distinct_first_row_hashes_across_classes"]
        != 1
        for name in ("template_xyz", "template_slot_order", "template_internal_graph")
    )
    stop_conditions = {
        "source_fingerprints_changed": before != after,
        "protein_molecule_split_overlap": (
            protein["split_and_support"]["molecule_overlap_count"] != 0
        ),
        "rna_molecule_split_overlap": (
            rna["split_and_support"]["molecule_overlap_count"] != 0
        ),
        "protein_type_conditioned_fixed_template_hash": protein_hash_leakage,
        "rna_type_conditioned_fixed_template_hash": rna_hash_leakage,
        "protein_template_probe_above_stop": protein_max > 0.15,
        "rna_template_probe_above_stop": rna_max > 0.40,
        "protein_context_controlled_delta_above_stop": protein_context_delta > 0.05,
        "rna_context_controlled_delta_above_stop": rna_context_delta > 0.05,
    }
    go_conditions = {
        "protein_template_probe_at_or_below_go": protein_max <= 0.10,
        "rna_template_probe_at_or_below_go": rna_max <= 0.35,
        "protein_context_controlled_delta_at_or_below_go": (
            protein_context_delta <= 0.03
        ),
        "rna_context_controlled_delta_at_or_below_go": rna_context_delta <= 0.03,
    }
    if any(stop_conditions.values()):
        status = "STOP"
    elif not all(go_conditions.values()):
        status = "REVIEW"
    else:
        status = "GO"

    report = {
        "stage": "stage2_geometry_only_leakage_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "split_unit": "whole_molecule_id_sha256",
        "gate_features": list(GATE_FEATURES),
        "context_control_features_compared_as_matched_pairs": list(
            CONTEXT_CONTROL_FEATURES
        ),
        "thresholds": {
            "protein_go": 0.10,
            "protein_stop": 0.15,
            "rna_go": 0.35,
            "rna_stop": 0.40,
        },
        "maximum_gate_balanced_accuracy": {
            "protein": protein_max,
            "rna": rna_max,
        },
        "maximum_positive_context_controlled_delta": {
            "protein": protein_context_delta,
            "rna": rna_context_delta,
        },
        "go_conditions": go_conditions,
        "stop_conditions": stop_conditions,
        "protein_collection": protein_collection,
        "rna_collection": rna_collection,
        "protein": protein,
        "rna": rna,
        "source_fingerprints_before": before,
        "source_fingerprints_after": after,
        "production_cif_pkl_pt_files_rewritten": False,
        "adapter_masking_edges_distance_loss_training_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "maximum_gate_balanced_accuracy": report[
                    "maximum_gate_balanced_accuracy"
                ],
                "maximum_positive_context_controlled_delta": report[
                    "maximum_positive_context_controlled_delta"
                ],
                "go_conditions": go_conditions,
                "stop_conditions": stop_conditions,
            },
            indent=2,
        ),
        flush=True,
    )
    if status != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
