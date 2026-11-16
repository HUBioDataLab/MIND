"""MoleculeNet task family (``mn_*``): Uni-Mol LMDB reading, processing with QC, the processed-split loader.

One task per MoleculeNet data set (QM9 excluded). Each task has its own ``data_root``
(``<source_root>/<task>``) with ``processed/<revision>/{train,valid,test}.pt``, ``qc_<split>.{json,tsv}`` and a
``provenance.json``. The raw Uni-Mol LMDB files are read-only inputs shared by all 14 tasks.

Processing rules (``task.yaml``):

* the Uni-Mol scaffold split of the LMDB files is the split; nothing is re-split;
* the FIRST conformer (index 0) of the 11 per molecule is used; the last one is Uni-Mol's flat fallback and is ignored;
* coordinates are stored in Angstrom: each dataset declares the unit of its LMDB coordinates (QM7: Bohr) and the
  conversion is applied before anything is stored; a split whose median nearest-neighbour distance is not a plausible
  Angstrom value is refused;
* hydrogens are dropped (counted); no atom-count cropping (molecules above 256 heavy atoms are only counted);
* molecules are never dropped for chemistry reasons: flat conformers, rare elements (id 119 at graph-build time),
  a single heavy atom, no heavy atom at all and more than 256 heavy atoms are flagged and counted instead;
* a corrupt record (missing/unknown key, wrong shape, non-finite value, bad label code, wrong count) raises.

The ligand graph is built at load time by ``downstream.common.complex_graph.ligand_runtime_sample``: the same
function as the LBA "separate" ligand path (``dataset_type="unimol"``, radius graph 5 A, at most 64 neighbours).
``data_loading.adapters.UniMolAdapter`` is deliberately not used (it swallows errors and does not drop H).

The LMDB values are pickles from a third-party archive. They are read with a restricted unpickler that only
accepts numpy array reconstruction; any other global raises ``pickle.UnpicklingError``.
"""

from __future__ import annotations

import csv
import io
import json
import pickle
import re
import types
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from data_loading.data_root import expand_data_root
from data_loading.atom_position_vocab import RARE_ATOMIC_NUMBER
from data_loading.cache_to_pyg import element_to_atomic_number
from downstream.common import complex_graph as cg
from downstream.common.provenance import record_process_run

TASK_YAML = Path(__file__).with_name("task.yaml")
CLASSIFICATION_KIND = "multilabel_masked"
REGRESSION_KIND = "multitarget_regression"
MOLNET_KINDS = (CLASSIFICATION_KIND, REGRESSION_KIND)
REQUIRED_KEYS = (
    "family", "modality", "processed_revision", "plausible_median_nn_angstrom", "source_name", "source_root",
    "raw_subdir", "archive", "splits",
    "conformer", "hydrogens", "max_atoms_policy", "report_atom_threshold", "missing_classification_code", "embedding",
    "probe", "finetune", "datasets",
)
DATASET_KEYS = ("task", "kind", "num_targets", "metrics", "primary_metric", "expected_counts", "coordinate_unit")
# Factor to Angstrom of every unit a dataset may declare for its LMDB coordinates (Bohr: CODATA 2018).
COORDINATE_UNIT_TO_ANGSTROM = {"angstrom": 1.0, "bohr": 0.529177210903}
ARCHIVE_KEYS = ("url", "name", "bytes", "md5", "sha256", "extract_subdir")
REQUIRED_RECORD_KEYS = frozenset(("atoms", "coordinates", "smi", "scaffold", "target", "ori_index"))
FLAG_NAMES = (
    "all_conformers_flat", "first_conformer_flat", "rare_element", "no_heavy_atoms", "single_heavy_atom",
    "over_256_heavy_atoms",
)
TSV_COLUMNS = ("id", "ori_index", "smiles", "status", "flags", "total_atoms", "heavy_atoms", "hydrogens_removed",
               "rare_elements")
PAYLOAD_KEYS = (
    "schema", "kind", "task", "dataset", "split", "num_targets", "ids", "smiles", "scaffolds", "ori_index", "labels",
    "label_mask", "atom_offsets", "atom_element_index", "element_table", "atom_positions", "flags", "counts", "qc",
)
_ELEMENT = re.compile(r"^(\*|[A-Z][a-z]{0,2})$")


def _read_yaml(path: Path) -> dict:
    with open(path) as handle:
        return expand_data_root(yaml.safe_load(handle))


def _dataset_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    datasets = _read_yaml(TASK_YAML)["datasets"]
    return ({entry["task"]: key for key, entry in datasets.items()},
            {key: entry["task"] for key, entry in datasets.items()})


DATASET_OF_TASK, TASK_OF_DATASET = _dataset_maps()


# ---------------------------------------------------------------------------
# Config and paths
# ---------------------------------------------------------------------------

def load_task_config(path: Path = TASK_YAML, *, data_root: Optional[Path] = None, dataset: str) -> dict:
    """Task config flattened for ``dataset`` (a key of ``datasets``); every decision is validated."""
    config = _read_yaml(path)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if dataset not in config["datasets"]:
        raise ValueError(f"unknown MoleculeNet dataset {dataset!r}; choose from {sorted(config['datasets'])}")
    absent = [key for key in ARCHIVE_KEYS if not config["archive"].get(key)]
    if absent:
        raise ValueError(f"{path}: archive lacks {absent} (url, size and both checksums must be pinned)")
    entry = config["datasets"][dataset]
    absent = [key for key in DATASET_KEYS if key not in entry]
    if absent:
        raise ValueError(f"{path}: dataset {dataset} lacks {absent}")
    if config["conformer"].get("policy") != "first":
        raise ValueError("conformer.policy: only 'first' (index 0) is implemented")
    if int(config["conformer"].get("expected_per_molecule", 0)) < 1:
        raise ValueError("conformer.expected_per_molecule must be a positive integer")
    if config["hydrogens"] != "drop":
        raise ValueError("hydrogens: only 'drop' is implemented")
    if config["max_atoms_policy"] != "none":
        raise ValueError("max_atoms_policy: only 'none' (no cropping) is implemented")
    if int(config["missing_classification_code"]) != -1:
        raise ValueError("missing_classification_code must be -1 (the Uni-Mol LMDB coding)")
    if list(config["splits"]) != ["train", "valid", "test"]:
        raise ValueError("splits must be [train, valid, test]")
    if entry["kind"] not in MOLNET_KINDS:
        raise ValueError(f"dataset {dataset}: kind must be one of {MOLNET_KINDS}")
    if int(entry["num_targets"]) < 1:
        raise ValueError(f"dataset {dataset}: num_targets must be positive")
    if entry["coordinate_unit"] not in COORDINATE_UNIT_TO_ANGSTROM:
        raise ValueError(f"dataset {dataset}: coordinate_unit {entry['coordinate_unit']!r} is not one of "
                         f"{sorted(COORDINATE_UNIT_TO_ANGSTROM)}")
    low, high = (float(v) for v in config["plausible_median_nn_angstrom"])
    if not 0.0 < low < high:
        raise ValueError(f"{path}: plausible_median_nn_angstrom must be [low, high] with 0 < low < high")
    if sorted(entry["expected_counts"]) != sorted(config["splits"]):
        raise ValueError(f"dataset {dataset}: expected_counts must name every split")
    wanted = {"roc_auc", "prc_auc"} if entry["kind"] == CLASSIFICATION_KIND else {"rmse", "mae"}
    if set(entry["metrics"]) != wanted or entry["primary_metric"] not in wanted:
        raise ValueError(f"dataset {dataset}: metrics must be {sorted(wanted)} and the primary metric one of them")
    config.update(entry)
    config["dataset"] = dataset
    del config["datasets"]
    config["source_root"] = str(config["source_root"])
    config["data_root"] = str(Path(config["source_root"]) / config["task"])
    if data_root is not None:
        config["data_root"] = str(data_root)
        config["source_root"] = str(Path(data_root).parent)
    config["expected_counts"] = {k: int(v) for k, v in entry["expected_counts"].items()}
    config["num_targets"] = int(entry["num_targets"])
    config["coordinate_scale"] = COORDINATE_UNIT_TO_ANGSTROM[entry["coordinate_unit"]]
    config["plausible_median_nn_angstrom"] = (low, high)
    config["processed_revision"] = str(config["processed_revision"])
    return config


def task_paths(config: dict) -> Dict[str, Path]:
    root, source = Path(config["data_root"]), Path(config["source_root"])
    return {"source_root": source, "raw": source / config["raw_subdir"], "root": root, "processed": root / "processed",
            "provenance": root / "provenance.json"}


def processed_dir(config: dict) -> Path:
    return task_paths(config)["processed"] / str(config["processed_revision"])


def lmdb_path(config: dict, split: str) -> Path:
    return task_paths(config)["raw"] / config["dataset"] / f"{split}.lmdb"


# ---------------------------------------------------------------------------
# LMDB reading (restricted unpickler)
# ---------------------------------------------------------------------------

class _NumpyOnlyUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"), ("numpy", "ndarray"),
        ("numpy", "dtype"), ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name} in an LMDB value (numpy arrays only)")


def load_record(raw: bytes) -> object:
    return _NumpyOnlyUnpickler(io.BytesIO(raw)).load()


def iter_lmdb(path: Path, *, expected: int) -> Iterator[Tuple[int, object]]:
    """Yield ``(index, record)`` for keys ``b"0" .. b"N-1"``; the entry count must equal ``expected``."""
    import lmdb

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (expected the Uni-Mol LMDB file, e.g. train.lmdb)")
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)
    try:
        with env.begin() as txn:
            count = int(txn.stat()["entries"])
            if count != int(expected):
                raise ValueError(f"{path}: LMDB holds {count} molecules, expected {expected} (expected_counts)")
            for index in range(count):
                raw = txn.get(str(index).encode())
                if raw is None:
                    raise KeyError(f"{path}: LMDB has {count} entries but no key {str(index).encode()!r}")
                try:
                    yield index, load_record(raw)
                except pickle.UnpicklingError as exc:
                    raise pickle.UnpicklingError(f"{path} key {index}: {exc}") from exc
    finally:
        env.close()


# ---------------------------------------------------------------------------
# One molecule
# ---------------------------------------------------------------------------

@dataclass
class Molecule:
    ori_index: int
    smiles: str
    scaffold: str
    elements: Tuple[str, ...]          # heavy atoms of conformer 0, input order
    positions: np.ndarray              # [L, 3] float32, Angstrom (converted from the dataset's declared unit)
    labels: np.ndarray                 # [T] float32, NaN = missing
    total_atoms: int
    hydrogens_removed: int
    rare_elements: Tuple[str, ...]
    flags: Dict[str, bool]


def _is_integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def convert_record(record: object, *, config: dict, where: str) -> Molecule:
    """Validate one LMDB record and turn it into a ``Molecule``; every deviation raises with ``where``."""
    if not isinstance(record, dict):
        raise ValueError(f"{where}: record is a {type(record).__name__}, expected dict")
    absent = sorted(REQUIRED_RECORD_KEYS - set(record))
    extra = sorted(set(record) - REQUIRED_RECORD_KEYS)
    if absent:
        raise ValueError(f"{where}: record is missing keys {absent}")
    if extra:
        raise ValueError(f"{where}: record has unexpected keys {extra}")
    atoms = record["atoms"]
    if not isinstance(atoms, (list, tuple)) or len(atoms) == 0:
        raise ValueError(f"{where}: molecule has no atoms")
    bad = sorted({str(a) for a in atoms if not isinstance(a, str) or not _ELEMENT.match(a)})
    if bad:
        raise ValueError(f"{where}: atoms are not element symbols: {bad[:5]}")
    conformers = record["coordinates"]
    expected = int(config["conformer"]["expected_per_molecule"])
    if not isinstance(conformers, (list, tuple)) or len(conformers) != expected:
        count = len(conformers) if isinstance(conformers, (list, tuple)) else type(conformers).__name__
        raise ValueError(f"{where}: {count} conformers, expected {expected}")
    flat = []
    for k, frame in enumerate(conformers):
        frame = np.asarray(frame)
        if frame.shape != (len(atoms), 3):
            raise ValueError(f"{where}: coordinates of conformer {k} have shape {frame.shape}, expected {(len(atoms), 3)}")
        if not np.isfinite(frame).all():
            raise ValueError(f"{where}: coordinates of conformer {k} are non-finite")
        flat.append(bool(np.abs(frame[:, 2]).max() == 0))
    smiles, scaffold, ori = record["smi"], record["scaffold"], record["ori_index"]
    if not isinstance(smiles, str) or not smiles:
        raise ValueError(f"{where}: smi must be a non-empty string")
    if not isinstance(scaffold, str):
        raise ValueError(f"{where}: scaffold must be a string")
    if not _is_integer(ori) or int(ori) < 0:
        raise ValueError(f"{where}: ori_index must be a non-negative integer, got {ori!r}")
    try:
        target = np.asarray(record["target"], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: target is not numeric: {record['target']!r}") from exc
    if target.size != config["num_targets"]:
        raise ValueError(f"{where}: {target.size} targets, expected {config['num_targets']}")
    if config["kind"] == CLASSIFICATION_KIND:
        bad_codes = sorted({float(v) for v in target if v not in (-1.0, 0.0, 1.0)})
        if bad_codes:
            raise ValueError(f"{where}: label code {bad_codes[0]:g} is not one of -1 (missing), 0, 1")
        labels = np.where(target == -1.0, np.nan, target).astype(np.float32)
    else:
        if np.isinf(target).any():
            raise ValueError(f"{where}: non-finite target {target.tolist()}")
        labels = target.astype(np.float32)
    heavy = [i for i, atom in enumerate(atoms) if atom != "H"]
    elements = tuple(atoms[i] for i in heavy)
    rare = tuple(sorted({e for e in elements if element_to_atomic_number(e) == RARE_ATOMIC_NUMBER}))
    positions = (np.asarray(conformers[0], dtype=np.float64)[heavy].reshape(len(heavy), 3)
                 * config["coordinate_scale"]).astype(np.float32)
    flags = {
        "all_conformers_flat": all(flat),
        "first_conformer_flat": flat[0],
        "rare_element": bool(rare),
        "no_heavy_atoms": len(heavy) == 0,
        "single_heavy_atom": len(heavy) == 1,
        "over_256_heavy_atoms": len(heavy) > int(config["report_atom_threshold"]),
    }
    return Molecule(int(ori), smiles, scaffold, elements, positions, labels, len(atoms), len(atoms) - len(heavy),
                    rare, flags)


# ---------------------------------------------------------------------------
# One split
# ---------------------------------------------------------------------------

def _label_summary(labels: np.ndarray, kind: str) -> dict:
    observed = ~np.isnan(labels)
    summary = {
        "num_targets": int(labels.shape[1]), "cells": int(labels.size), "labelled": int(observed.sum()),
        "missing": int((~observed).sum()), "missing_fraction": round(float((~observed).mean()), 6),
        "labelled_per_target": observed.sum(axis=0).astype(int).tolist(),
    }
    if kind == CLASSIFICATION_KIND:
        positives = np.nansum(labels, axis=0).astype(int)
        summary["positives_per_target"] = positives.tolist()
        summary["targets_without_positive"] = int(((positives == 0) & (observed.sum(axis=0) > 0)).sum())
        summary["targets_without_negative"] = int(((positives == observed.sum(axis=0)) & (observed.sum(axis=0) > 0)).sum())
    else:
        values = labels[observed]
        summary["min"], summary["max"] = float(values.min()), float(values.max())
    return summary


def _qc_summary(config: dict, split: str, molecules: Sequence[Molecule], labels: np.ndarray) -> dict:
    heavy = np.array([len(m.elements) for m in molecules])
    total = np.array([m.total_atoms for m in molecules])
    threshold = int(config["report_atom_threshold"])
    by_molecule: Counter = Counter()
    by_atom: Counter = Counter()
    for m in molecules:
        by_molecule.update(m.rare_elements)
        by_atom.update(e for e in m.elements if e in m.rare_elements)
    smiles = Counter(m.smiles for m in molecules)
    return {
        "dataset": config["dataset"], "task": config["task"], "split": split, "molecules": len(molecules),
        "expected": config["expected_counts"][split], "dropped": 0,
        "conformer_policy": f"first (index 0 of {config['conformer']['expected_per_molecule']})",
        "hydrogens": {"molecules_with_h": int(sum(m.hydrogens_removed > 0 for m in molecules)),
                      "h_atoms_removed": int(sum(m.hydrogens_removed for m in molecules))},
        "flags": {name: int(sum(m.flags[name] for m in molecules)) for name in FLAG_NAMES},
        "rare_elements": {
            "molecules": int(sum(bool(m.rare_elements) for m in molecules)),
            "by_element_molecules": dict(sorted(by_molecule.items())),
            "by_element_atoms": dict(sorted(by_atom.items())),
            "atoms_mapped_to_119": int(sum(by_atom.values())),
        },
        "atoms": {
            "heavy_min": int(heavy.min()), "heavy_median": float(np.median(heavy)), "heavy_max": int(heavy.max()),
            "total_max": int(total.max()), "heavy_over_256": int((heavy > threshold).sum()),
            "total_over_256": int((total > threshold).sum()),
        },
        "duplicate_smiles_within_split": int(sum(c - 1 for c in smiles.values())),
        "labels": _label_summary(labels, config["kind"]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def median_nearest_neighbour(molecules: Sequence[Molecule]) -> Tuple[float, int]:
    """Median over molecules of the per-molecule median nearest-neighbour distance (heavy atoms, first conformer).

    Molecules with fewer than two heavy atoms and those with a flat first conformer (RDKit failed; their coordinates
    are not a 3-D structure) are skipped. Returns ``(median, number of molecules used)``; ``(nan, 0)`` when none is left.
    """
    per_molecule = []
    for m in molecules:
        if len(m.elements) < 2 or m.flags["first_conformer_flat"]:
            continue
        gaps = np.linalg.norm(m.positions[:, None, :] - m.positions[None, :, :], axis=-1)
        np.fill_diagonal(gaps, np.inf)
        per_molecule.append(float(np.median(gaps.min(axis=1))))
    return (float(np.median(per_molecule)), len(per_molecule)) if per_molecule else (float("nan"), 0)


def check_coordinate_scale(config: dict, split: str, molecules: Sequence[Molecule]) -> dict:
    """Unit guard on the converted coordinates of one split; raises when they are not plausible Angstrom values.

    Bond lengths between heavy atoms are 1.2-1.6 Angstrom, so the median nearest-neighbour distance of a set in
    Angstrom is about 1.4; the same set in Bohr gives about 2.6 and in nanometres about 0.14. The accepted range is
    ``plausible_median_nn_angstrom``. Returns the QC entry (unit, factor, median, range).
    """
    low, high = config["plausible_median_nn_angstrom"]
    median, used = median_nearest_neighbour(molecules)
    if used == 0:
        raise ValueError(f"{config['task']} {split}: no molecule with a 3-D conformer and two heavy atoms, so the "
                         "coordinate unit cannot be checked")
    if not low <= median <= high:
        raise ValueError(
            f"{config['task']} {split}: median nearest-neighbour distance {median:.3f} after converting from "
            f"{config['coordinate_unit']} (factor {config['coordinate_scale']}) is outside the plausible Angstrom range "
            f"[{low}, {high}]; the declared coordinate_unit of dataset {config['dataset']} in task.yaml is probably wrong")
    return {"declared_unit": config["coordinate_unit"], "scale_to_angstrom": config["coordinate_scale"],
            "median_nn_angstrom": round(median, 4), "molecules_checked": used, "plausible_range": [low, high]}


def process_split(config: dict, split: str, owner: Dict[int, str]) -> Tuple[dict, List[Molecule], dict]:
    """Read one LMDB split into ``(payload, molecules, summary)``; ``owner`` tracks ``ori_index`` across splits."""
    path = lmdb_path(config, split)
    molecules: List[Molecule] = []
    for index, record in iter_lmdb(path, expected=config["expected_counts"][split]):
        molecule = convert_record(record, config=config, where=f"{config['dataset']} {split} record {index}")
        if molecule.ori_index in owner:
            raise ValueError(f"{config['dataset']} {split} record {index}: ori_index {molecule.ori_index} appears in both "
                             f"{owner[molecule.ori_index]} and {split}")
        owner[molecule.ori_index] = split
        molecules.append(molecule)
    labels = np.stack([m.labels for m in molecules]).astype(np.float32)
    table = sorted({e for m in molecules for e in m.elements})
    if len(table) > 255:
        raise ValueError(f"{path}: {len(table)} distinct elements do not fit the uint8 element table")
    index_of = {e: i for i, e in enumerate(table)}
    offsets = np.concatenate(([0], np.cumsum([len(m.elements) for m in molecules]))).astype(np.int64)
    positions = (np.concatenate([m.positions for m in molecules]) if offsets[-1]
                 else np.zeros((0, 3), np.float32)).astype(np.float32)
    summary = _qc_summary(config, split, molecules, labels)
    summary["coordinates"] = check_coordinate_scale(config, split, molecules)
    ids = [f"{config['dataset']}_{m.ori_index}" for m in molecules]
    payload = {
        "schema": {"format": "unimol_molnet_v1", "conformer": "first", "hydrogens": "dropped",
                   "source": config["source_name"], "revision": str(config["processed_revision"]),
                   "coordinate_unit": "angstrom", "coordinate_source_unit": config["coordinate_unit"],
                   "coordinate_scale": config["coordinate_scale"]},
        "kind": config["kind"], "task": config["task"], "dataset": config["dataset"], "split": split,
        "num_targets": config["num_targets"], "ids": ids, "smiles": [m.smiles for m in molecules],
        "scaffolds": [m.scaffold for m in molecules],
        "ori_index": torch.tensor([m.ori_index for m in molecules], dtype=torch.long),
        "labels": torch.from_numpy(labels), "label_mask": torch.from_numpy(~np.isnan(labels)),
        "atom_offsets": torch.from_numpy(offsets),
        "atom_element_index": torch.tensor([index_of[e] for m in molecules for e in m.elements], dtype=torch.uint8),
        "element_table": table, "atom_positions": torch.from_numpy(positions),
        "flags": {name: torch.tensor([m.flags[name] for m in molecules], dtype=torch.bool) for name in FLAG_NAMES},
        "counts": {
            "total_atoms": torch.tensor([m.total_atoms for m in molecules], dtype=torch.long),
            "heavy_atoms": torch.tensor([len(m.elements) for m in molecules], dtype=torch.long),
            "hydrogens_removed": torch.tensor([m.hydrogens_removed for m in molecules], dtype=torch.long),
        },
        "qc": summary,
    }
    return payload, molecules, summary


def _write_qc(out_dir: Path, split: str, ids: Sequence[str], molecules: Sequence[Molecule], summary: dict) -> None:
    with open(out_dir / f"qc_{split}.tsv", "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for id_, m in zip(ids, molecules):
            names = [name for name in FLAG_NAMES if m.flags[name]]
            writer.writerow([id_, m.ori_index, m.smiles, "flagged" if names else "ok", ",".join(names), m.total_atoms,
                             len(m.elements), m.hydrogens_removed, ",".join(m.rare_elements)])
    (out_dir / f"qc_{split}.json").write_text(json.dumps(summary, indent=1))


def build_processed(config: dict, *, splits: Sequence[str], config_path: Path, dry_run: bool = False) -> int:
    """Process the requested splits; returns 0 (every failure raises). ``dry_run`` reads and checks, writes nothing.

    ``config_path`` is the task YAML ``config`` was loaded from; the provenance record names and hashes it.
    """
    out_dir = processed_dir(config)
    if not dry_run:
        existing = [out_dir / f"{s}.pt" for s in splits if (out_dir / f"{s}.pt").exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite {existing[0]}")
    started = datetime.now(timezone.utc).isoformat()
    owner: Dict[int, str] = {}
    for split in splits:
        payload, molecules, summary = process_split(config, split, owner)
        print(json.dumps(summary, indent=1), flush=True)
        if dry_run:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_qc(out_dir, split, payload["ids"], molecules, summary)
        target = out_dir / f"{split}.pt"
        partial = target.with_name(target.name + ".partial")
        torch.save(payload, partial)
        partial.replace(target)
        print(f"[{split}] wrote {target} ({len(molecules)} molecules)", flush=True)
    if not dry_run:
        record_process_run(
            task_paths(config)["provenance"], started_utc=started, config_path=config_path, processed_dir=out_dir,
            splits=list(splits), exit_code=0, inputs=[lmdb_path(config, s) for s in splits],
            extra={"dataset": config["dataset"], "task": config["task"],
                   "policy": {"conformer": "first (index 0)", "hydrogens": "dropped", "cropping": "none",
                              "flags_not_dropped": list(FLAG_NAMES)}})
    return 0


# ---------------------------------------------------------------------------
# Processed split
# ---------------------------------------------------------------------------

def check_schema_matches_task_yaml(payload: dict, path: Path) -> None:
    """Refuse a processed split of another revision than task.yaml declares for its dataset, and any split whose
    dataset is declared in a unit other than Angstrom but whose payload does not record the conversion.

    Only QM7 (Bohr) has a revision above the family default; the v1 files of the other 13 datasets are unchanged.
    """
    schema = payload["schema"]
    values = _read_yaml(TASK_YAML)
    if payload["dataset"] not in values["datasets"]:
        raise ValueError(f"{path}: unknown MoleculeNet dataset {payload['dataset']!r}")
    entry = values["datasets"][payload["dataset"]]
    revision = str(entry.get("processed_revision", values["processed_revision"]))
    if str(schema.get("revision")) != revision:
        raise ValueError(
            f"{path}: processed with revision {schema.get('revision')!r} but task.yaml declares {revision!r} for "
            f"{entry['task']}; the older data is not valid for this task (QM7 v1 coordinates are in Bohr, not "
            "Angstrom): reprocess with `make ds-process TASK=" + entry["task"] + "`")
    unit = entry["coordinate_unit"]
    if unit != "angstrom" and (schema.get("coordinate_source_unit") != unit or schema.get("coordinate_unit") != "angstrom"
                               or schema.get("coordinate_scale") != COORDINATE_UNIT_TO_ANGSTROM[unit]):
        raise ValueError(f"{path}: {entry['task']} coordinates are declared in {unit} but the file does not record the "
                         f"conversion to Angstrom (schema {schema}); reprocess it")


class MolnetProcessedSplit:
    """One processed split: ids, labels (NaN = missing) with mask, flags and the first-conformer heavy atoms.

    Interface-compatible with the other tasks' ``ProcessedSplit`` for ``embed``/``probe`` (``chains``, ``labels``,
    ``runtime``, ``iter_runtime``, ``chain_embeddings``). One sample per molecule.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        for key in PAYLOAD_KEYS:
            if key not in payload:
                raise ValueError(f"{self.path}: processed split lacks {key!r}")
        if payload["kind"] not in MOLNET_KINDS:
            raise ValueError(f"{self.path}: unknown kind {payload['kind']!r}")
        check_schema_matches_task_yaml(payload, self.path)
        count = len(payload["ids"])
        targets = int(payload["num_targets"])
        labels, mask = payload["labels"].numpy(), payload["label_mask"].numpy()
        if labels.shape != (count, targets) or targets < 1:
            raise ValueError(f"{self.path}: labels {labels.shape} do not match {count} molecules x {targets} targets")
        if mask.shape != labels.shape or not np.array_equal(mask, ~np.isnan(labels)):
            raise ValueError(f"{self.path}: label_mask disagrees with the NaN (missing) labels")
        if payload["kind"] == CLASSIFICATION_KIND and not np.isin(labels[mask], (0.0, 1.0)).all():
            raise ValueError(f"{self.path}: classification labels must be 0, 1 or NaN")
        offsets = payload["atom_offsets"].numpy()
        atoms = int(payload["atom_positions"].shape[0])
        if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != atoms or (np.diff(offsets) < 0).any():
            raise ValueError(f"{self.path}: atom offsets {offsets.shape} are inconsistent with {count} molecules / {atoms} atoms")
        if payload["atom_element_index"].shape[0] != atoms or (
                atoms and int(payload["atom_element_index"].max()) >= len(payload["element_table"])):
            raise ValueError(f"{self.path}: atom element index does not match the element table")
        if not bool(torch.isfinite(payload["atom_positions"]).all()):
            raise ValueError(f"{self.path}: atom positions are non-finite")
        if sorted(payload["flags"]) != sorted(FLAG_NAMES):
            raise ValueError(f"{self.path}: flags must be exactly {FLAG_NAMES}")
        for name, values in payload["flags"].items():
            if values.shape != (count,) or values.dtype != torch.bool:
                raise ValueError(f"{self.path}: flag {name} has shape {tuple(values.shape)}, expected ({count},) bool")
        for key in ("smiles", "scaffolds"):
            if len(payload[key]) != count:
                raise ValueError(f"{self.path}: {key} has {len(payload[key])} entries for {count} molecules")
        self.payload = payload
        self.schema = payload["schema"]
        self.kind, self.task, self.dataset, self.split = payload["kind"], payload["task"], payload["dataset"], payload["split"]
        self.ids: List[str] = list(payload["ids"])
        self.chains = self.ids
        self.smiles: List[str] = list(payload["smiles"])
        self.scaffolds: List[str] = list(payload["scaffolds"])
        self.ori_index = payload["ori_index"].numpy()
        self.labels = labels.astype(np.float32)
        self.label_mask = mask
        self.flags = {name: values.numpy() for name, values in payload["flags"].items()}
        self.counts = {name: values.numpy() for name, values in payload["counts"].items()}
        self.embeddable = ~self.flags["no_heavy_atoms"]
        self._offsets = offsets
        self._elements = [payload["element_table"][int(i)] for i in payload["atom_element_index"].tolist()]
        self._positions = payload["atom_positions"].numpy().astype(np.float32)
        self.report = {"molecules": count, "not_embeddable": int((~self.embeddable).sum())}

    def __len__(self) -> int:
        return len(self.ids)

    def ligand(self, index: int) -> cg.LigandAtoms:
        lo, hi = int(self._offsets[index]), int(self._offsets[index + 1])
        return cg.ligand_from_rows(self._elements[lo:hi], self._positions[lo:hi])

    def runtime(self, index: int, model_config, *, seed: int):
        if not self.embeddable[index]:
            raise ValueError(
                f"no heavy atom after hydrogen removal: {self.ids[index]} cannot be turned into a graph; the "
                "embedding step must handle such molecules explicitly (none occur in the 14 MoleculeNet data sets)")
        return cg.ligand_runtime_sample(self.ligand(index), self.ids[index], model_config, seed=seed)

    def iter_runtime(self, model_config, *, seed: int) -> Iterator:
        for index in range(len(self)):
            yield index, self.runtime(index, model_config, seed=seed)

    def chain_embeddings(self, sample_embeddings: np.ndarray) -> np.ndarray:
        if len(sample_embeddings) != len(self):
            raise ValueError(f"expected {len(self)} molecule embeddings, got {len(sample_embeddings)}")
        return np.asarray(sample_embeddings, dtype=np.float32)


def test_subsets(config: dict, chains: Sequence[str]) -> Dict[str, np.ndarray]:
    """Row masks of the test molecules per QC flag (only flags with members) plus ``no_flags``.

    Read from ``qc_test.tsv``. A molecule that is not listed raises: a silent False would drop it from every subset.
    """
    path = processed_dir(config) / "qc_test.tsv"
    with open(path, newline="") as handle:
        rows = {row["id"]: set(filter(None, row["flags"].split(","))) for row in csv.DictReader(handle, delimiter="\t")}
    unlisted = [c for c in chains if c not in rows]
    if unlisted:
        raise ValueError(f"{len(unlisted)} test molecules are not in qc_test.tsv, first {unlisted[:5]}")
    subsets = {}
    for name in FLAG_NAMES:
        mask = np.array([name in rows[c] for c in chains])
        if mask.any():
            subsets[name] = mask
    clean = np.array([not rows[c] for c in chains])
    if clean.any():
        subsets["no_flags"] = clean
    return subsets


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------

def task_module(name: str) -> types.ModuleType:
    """A module-like object with the surface ``downstream.tasks.load_task`` and the CLIs expect, for one ``mn_*`` task."""
    if name not in DATASET_OF_TASK:
        raise ValueError(f"unknown MoleculeNet task {name!r}; choose from {sorted(DATASET_OF_TASK)}")
    dataset = DATASET_OF_TASK[name]
    module = types.ModuleType(f"downstream.tasks.molnet.{name}")
    module.DATASET = dataset
    module.TASK_YAML = TASK_YAML
    module.KIND = _read_yaml(TASK_YAML)["datasets"][dataset]["kind"]
    module.load_task_config = lambda path=TASK_YAML, *, data_root=None: load_task_config(
        path, data_root=data_root, dataset=dataset)
    module.processed_dir = processed_dir
    module.test_subsets = test_subsets
    module.ProcessedSplit = MolnetProcessedSplit
    return module
