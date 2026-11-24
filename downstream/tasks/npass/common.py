"""NPASS task family (``npass_*``): CSV reading, split, conformer generation, processing with QC, processed splits.

One task per target of the NaFM NPASS regression benchmark (``task.yaml`` documents the sources, the protocol of the
reference code and every point that is not stated anywhere). Each task has its own ``data_root``
(``<source_root>/<task>``) with ``processed/<tag>/<revision>/{train,valid,test}.pt``, ``qc_<split>.{json,tsv}`` (one
directory per split tag ``seed<k>``, see ``downstream.common.split_replicates``) and a ``provenance.json``; the raw
CSV is a read-only input shared by all eight tasks.

Rules (``task.yaml``):

* rows of one ``target_id`` in CSV order, label ``activity_value`` unchanged; the row count must equal
  ``expected_molecules`` (the paper's count is kept next to it and a difference needs a ``count_note``);
* split: the NaFM reference code's random 80/10/10 permutation, drawn once per seed of ``split.seeds`` (tag
  ``seed<k>`` = the split of run seed ``k``); the sizes must sum to the molecule count (the reference code would drop
  samples silently, here that raises) and every processed split must have exactly the size the rule gives;
* graph input: RDKit ETKDGv3 conformer with explicit hydrogens and a fixed seed, MMFF94 optimization, hydrogens
  dropped (counted), no cropping. A molecule whose embedding fails keeps flat 2-D coordinates and is flagged and
  counted, never dropped; MMFF outcomes are counted;
* a corrupt row (missing column, empty or duplicated id, unparsable SMILES, non-finite value) raises.

The payload and the loader are those of the MoleculeNet family (``downstream.tasks.molnet.common``), so ``embed``,
``probe``, ``finetune`` and the fingerprint baselines treat an NPASS task like a one-target MoleculeNet regression,
with the split tag of the run seed.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml

from data_loading.atom_position_vocab import RARE_ATOMIC_NUMBER
from data_loading.cache_to_pyg import element_to_atomic_number
from downstream.common import split_replicates
from downstream.common.provenance import record_process_run
from downstream.tasks.molnet import common as molnet
from downstream.tasks.molnet.common import Molecule

TASK_YAML = Path(__file__).with_name("task.yaml")
KIND = molnet.REGRESSION_KIND
SCHEMA_FORMAT = "nafm_npass_v1"
REQUIRED_KEYS = (
    "family", "modality", "processed_revision", "plausible_median_nn_angstrom", "source_name", "source_root",
    "raw_subdir", "source_file", "splits", "split", "conformer", "hydrogens", "max_atoms_policy",
    "report_atom_threshold", "embedding", "probe", "finetune", "datasets",
)
DATASET_KEYS = ("task", "target_id", "target_name", "paper_molecules", "expected_molecules", "kind", "num_targets",
                "metrics", "primary_metric", "expected_counts")
SOURCE_KEYS = ("url", "name", "record", "bytes", "md5", "sha256", "rows", "columns")
COLUMN_KEYS = ("id", "smiles", "target", "value")
CONFORMER_STATUS = ("etkdg", "etkdg_random_coords", "failed_2d_fallback")
MMFF_STATUS = ("converged", "not_converged", "no_parameters", "skipped_2d_fallback")


def _read_yaml(path: Path) -> dict:
    from data_loading.data_root import expand_data_root

    with open(path) as handle:
        return expand_data_root(yaml.safe_load(handle))


def _dataset_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    datasets = _read_yaml(TASK_YAML)["datasets"]
    return ({entry["task"]: key for key, entry in datasets.items()},
            {key: entry["task"] for key, entry in datasets.items()})


DATASET_OF_TASK, TASK_OF_DATASET = _dataset_maps()


# ---------------------------------------------------------------------------
# Split (the NaFM reference code, gnn/utils.py train_val_test_split with float sizes)
# ---------------------------------------------------------------------------

def split_sizes(count: int, fractions: Sequence[float]) -> Tuple[int, int, int]:
    """Train / valid / test sizes of ``count`` molecules; raises unless they add up to ``count``.

    ``round(count * fraction)`` per split (Python rounding, as the reference code) and the test size reduced by one
    when the sum exceeds ``count``. The reference code leaves samples out silently when the sum is below ``count``;
    that case raises here.
    """
    train, valid, test = (round(count * float(f)) for f in fractions)
    if train + valid + test > count:
        test -= 1
    if min(train, valid, test) < 0 or train + valid + test != count:
        raise ValueError(f"split fractions {list(fractions)} give sizes {(train, valid, test)} for {count} molecules; "
                         "they must add up to the molecule count")
    return train, valid, test


def split_indices(count: int, fractions: Sequence[float], seed: int) -> Dict[str, np.ndarray]:
    """Row indices (into the target's rows, CSV order) of each split: the reference code's seeded permutation."""
    train, valid, test = split_sizes(count, fractions)
    order = np.random.default_rng(int(seed)).permutation(np.arange(count, dtype=np.int64))
    return {"train": order[:train], "valid": order[train:train + valid], "test": order[train + valid:]}


def check_partition(indices: Dict[str, np.ndarray], count: int, expected: Dict[str, int], where: str) -> None:
    """Raise unless the splits are disjoint, cover rows ``0..count-1`` and have the expected sizes."""
    sizes = {name: int(len(rows)) for name, rows in indices.items()}
    if sizes != {name: int(size) for name, size in expected.items()}:
        raise ValueError(f"{where}: split sizes {sizes}, expected {dict(expected)}")
    covered = np.concatenate([np.asarray(rows, dtype=np.int64) for rows in indices.values()])
    if sorted(covered.tolist()) != list(range(count)):
        raise ValueError(f"{where}: the splits are not a partition of the {count} rows (overlap or gap)")


# ---------------------------------------------------------------------------
# Config and paths
# ---------------------------------------------------------------------------

def load_task_config(path: Path = TASK_YAML, *, data_root: Path | None = None, dataset: str) -> dict:
    """Task config flattened for ``dataset`` (a key of ``datasets``); every decision is validated."""
    config = _read_yaml(path)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if dataset not in config["datasets"]:
        raise ValueError(f"unknown NPASS dataset {dataset!r}; choose from {sorted(config['datasets'])}")
    absent = [key for key in SOURCE_KEYS if not config["source_file"].get(key)]
    if absent:
        raise ValueError(f"{path}: source_file lacks {absent} (url, size and both checksums must be pinned)")
    if sorted(config["source_file"]["columns"]) != sorted(COLUMN_KEYS):
        raise ValueError(f"{path}: source_file.columns must name exactly {list(COLUMN_KEYS)}")
    entry = config["datasets"][dataset]
    absent = [key for key in DATASET_KEYS if key not in entry]
    if absent:
        raise ValueError(f"{path}: dataset {dataset} lacks {absent}")
    if config["conformer"].get("policy") != "etkdg_mmff":
        raise ValueError("conformer.policy: only 'etkdg_mmff' is implemented")
    if int(config["conformer"].get("expected_per_molecule", 0)) != 1:
        raise ValueError("conformer.expected_per_molecule must be 1 (one generated conformer)")
    if config["hydrogens"] != "drop":
        raise ValueError("hydrogens: only 'drop' is implemented")
    if config["max_atoms_policy"] != "none":
        raise ValueError("max_atoms_policy: only 'none' (no cropping) is implemented")
    if list(config["splits"]) != ["train", "valid", "test"]:
        raise ValueError("splits must be [train, valid, test]")
    if config["split"].get("policy") != "random_permutation":
        raise ValueError("split.policy: only 'random_permutation' (the NaFM reference code) is implemented")
    fractions = [float(f) for f in config["split"]["fractions"]]
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split.fractions must be three positive numbers that sum to 1")
    split_seeds = config["split"].get("seeds")
    if (not isinstance(split_seeds, list) or not split_seeds or len(set(split_seeds)) != len(split_seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in split_seeds)):
        raise ValueError("split.seeds must be a non-empty list of distinct non-negative integers (one split per seed)")
    for run in ("probe", "finetune"):
        unsplit = sorted(set(config[run]["seeds"]) - set(split_seeds))
        if unsplit:
            raise ValueError(f"{run}.seeds {unsplit} have no split: the split and the model initialization share the "
                             f"seed, so every {run} seed must be listed in split.seeds {split_seeds}")
    if entry["kind"] != KIND:
        raise ValueError(f"dataset {dataset}: kind must be {KIND!r}")
    if int(entry["num_targets"]) != 1:
        raise ValueError(f"dataset {dataset}: an NPASS task has exactly one target")
    if set(entry["metrics"]) != {"rmse", "mae"} or entry["primary_metric"] not in ("rmse", "mae"):
        raise ValueError(f"dataset {dataset}: metrics must be ['rmse', 'mae'] and the primary metric one of them")
    low, high = (float(v) for v in config["plausible_median_nn_angstrom"])
    if not 0.0 < low < high:
        raise ValueError(f"{path}: plausible_median_nn_angstrom must be [low, high] with 0 < low < high")
    paper, expected = int(entry["paper_molecules"]), int(entry["expected_molecules"])
    if paper != expected and not entry.get("count_note"):
        raise ValueError(f"dataset {dataset}: expected_molecules {expected} differs from the paper's {paper}; "
                         "document the difference in count_note")
    sizes = dict(zip(config["splits"], split_sizes(expected, fractions)))
    if {k: int(v) for k, v in entry["expected_counts"].items()} != sizes:
        raise ValueError(f"dataset {dataset}: expected_counts {entry['expected_counts']} differ from the split rule's "
                         f"sizes {sizes} for {expected} molecules")
    config.update(entry)
    config["dataset"] = dataset
    del config["datasets"]
    config["source_root"] = str(config["source_root"])
    config["data_root"] = str(Path(config["source_root"]) / config["task"])
    if data_root is not None:
        config["data_root"] = str(data_root)
        config["source_root"] = str(Path(data_root).parent)
    config["expected_counts"] = sizes
    config["num_targets"] = 1
    config["coordinate_unit"] = "angstrom"           # RDKit coordinates
    config["coordinate_scale"] = 1.0
    config["plausible_median_nn_angstrom"] = (low, high)
    config["processed_revision"] = str(config["processed_revision"])
    config["split"] = {**config["split"], "fractions": fractions, "seeds": list(split_seeds)}
    config["split_seeds"] = list(split_seeds)
    config["split_tags"] = [split_replicates.tag_for_seed(s) for s in split_seeds]
    return config


def csv_path(config: dict) -> Path:
    return molnet.task_paths(config)["raw"] / config["source_file"]["name"]


task_paths = molnet.task_paths


def processed_dir(config: dict, tag: str) -> Path:
    """``<data_root>/processed/<tag>/<revision>``: the processed files of the split of run seed ``seed_for_tag(tag)``."""
    if tag not in config["split_tags"]:
        raise ValueError(f"unknown split tag {tag!r}; {config['task']} has the tags {config['split_tags']} "
                         "(one per seed of split.seeds)")
    return task_paths(config)["processed"] / tag / str(config["processed_revision"])


# ---------------------------------------------------------------------------
# CSV rows
# ---------------------------------------------------------------------------

def read_target_rows(config: dict, path: Path | None = None) -> List[dict]:
    """The rows of ``config['target_id']`` in file order as ``{row, id, smiles, value}``; corrupt input raises."""
    path = Path(path or csv_path(config))
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make ds-download TASK={config['task']}` first")
    columns = config["source_file"]["columns"]
    rows, seen = [], set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        absent = [name for name in columns.values() if name not in (reader.fieldnames or [])]
        if absent:
            raise ValueError(f"{path}: lacks columns {absent}; found {reader.fieldnames}")
        for index, record in enumerate(reader):
            if record[columns["target"]] != config["target_id"]:
                continue
            where = f"{path} data row {index} (target {config['target_id']})"
            identifier, smiles = record[columns["id"]].strip(), record[columns["smiles"]].strip()
            if not identifier or not smiles:
                raise ValueError(f"{where}: empty id or SMILES")
            if identifier in seen:
                raise ValueError(f"{where}: id {identifier!r} occurs twice within the target")
            seen.add(identifier)
            try:
                value = float(record[columns["value"]])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{where}: activity value {record[columns['value']]!r} is not a number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{where}: non-finite activity value {value}")
            rows.append({"row": index, "id": identifier, "smiles": smiles, "value": value})
    expected = int(config["expected_molecules"])
    if len(rows) != expected:
        raise ValueError(f"{path}: {len(rows)} rows for {config['target_id']} ({config['target_name']}), expected "
                         f"{expected} (paper: {config['paper_molecules']}"
                         f"{'; ' + config['count_note'] if config.get('count_note') else ''})")
    return rows


# ---------------------------------------------------------------------------
# One molecule: SMILES -> conformer
# ---------------------------------------------------------------------------

def generate_conformer(smiles: str, *, config: dict, where: str):
    """``(heavy-atom symbols, positions [L, 3] float32, n_atoms_with_h, conformer status, mmff status)``.

    ETKDGv3 on the molecule with explicit hydrogens (seed ``conformer.seed``, one thread), one retry with random
    initial coordinates, then flat 2-D coordinates (status ``failed_2d_fallback``). MMFF runs on the 3-D conformer; its
    outcome is returned, a failed optimization keeps the ETKDG geometry. Heavy atoms keep the SMILES order.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"{where}: RDKit cannot parse SMILES {smiles!r}")
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if not heavy:
        raise ValueError(f"{where}: SMILES {smiles!r} has no heavy atom")
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy]
    spec = config["conformer"]
    mol_h = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = int(spec["seed"])
    params.numThreads = 1
    status, conformer_id = "etkdg", AllChem.EmbedMolecule(mol_h, params)
    if conformer_id < 0 and spec["embed_retry_random_coords"]:
        params.useRandomCoords = True
        status, conformer_id = "etkdg_random_coords", AllChem.EmbedMolecule(mol_h, params)
    if conformer_id < 0:
        flat = Chem.Mol(mol)
        AllChem.Compute2DCoords(flat)
        xyz = np.asarray(flat.GetConformer().GetPositions(), dtype=np.float64)
        xyz[:, 2] = 0.0
        return symbols, xyz[heavy].astype(np.float32), mol_h.GetNumAtoms(), "failed_2d_fallback", "skipped_2d_fallback"
    if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol_h):
        mmff = "no_parameters"
    else:
        code = rdForceFieldHelpers.MMFFOptimizeMolecule(
            mol_h, mmffVariant=spec["mmff_variant"], maxIters=int(spec["mmff_max_iters"]))
        mmff = {0: "converged", 1: "not_converged"}.get(code)
        if mmff is None:
            raise RuntimeError(f"{where}: MMFFOptimizeMolecule returned {code}, expected 0 or 1")
    xyz = np.asarray(mol_h.GetConformer().GetPositions(), dtype=np.float64)
    positions = xyz[heavy].astype(np.float32)
    if not np.isfinite(positions).all():
        raise ValueError(f"{where}: non-finite coordinates after embedding")
    return symbols, positions, mol_h.GetNumAtoms(), status, mmff


def murcko_scaffold(smiles: str, *, where: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception as exc:  # RDKit raises ValueError/RuntimeError subclasses of varying type
        raise ValueError(f"{where}: Murcko scaffold of {smiles!r} failed: {exc}") from exc


def build_molecule(row: dict, *, config: dict) -> Tuple[Molecule, str, str]:
    """One processed molecule plus its conformer and MMFF status."""
    where = f"{config['task']} {row['id']}"
    symbols, positions, total, status, mmff = generate_conformer(row["smiles"], config=config, where=where)
    rare = tuple(sorted({e for e in symbols if element_to_atomic_number(e) == RARE_ATOMIC_NUMBER}))
    failed = status == "failed_2d_fallback"
    flags = {
        "all_conformers_flat": failed, "first_conformer_flat": failed, "rare_element": bool(rare),
        "no_heavy_atoms": False, "single_heavy_atom": len(symbols) == 1,
        "over_256_heavy_atoms": len(symbols) > int(config["report_atom_threshold"]),
    }
    molecule = Molecule(row["row"], row["smiles"], murcko_scaffold(row["smiles"], where=where), tuple(symbols), positions,
                        np.array([row["value"]], dtype=np.float32), total, total - len(symbols), rare, flags)
    return molecule, status, mmff


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _conformer_summary(statuses: Sequence[Tuple[str, str]], ids: Sequence[str]) -> dict:
    conformer = Counter(s for s, _ in statuses)
    mmff = Counter(m for _, m in statuses)
    return {
        "embedding": {name: int(conformer.get(name, 0)) for name in CONFORMER_STATUS},
        "mmff": {name: int(mmff.get(name, 0)) for name in MMFF_STATUS},
        "embedding_failed_ids": [i for i, (s, _) in zip(ids, statuses) if s == "failed_2d_fallback"],
        "mmff_not_converged_ids": [i for i, (_, m) in zip(ids, statuses) if m == "not_converged"],
        "mmff_no_parameters_ids": [i for i, (_, m) in zip(ids, statuses) if m == "no_parameters"],
    }


def process_task(config: dict, tags: Sequence[str] | None = None) -> Dict[str, Dict[str, Tuple[dict, List[Molecule], dict]]]:
    """Read and convert one task into ``{tag: {split: (payload, molecules, summary)}}``; nothing is written.

    Every molecule is converted once (the conformer does not depend on the split) and shared by the tags. For each tag
    the rows are split with the reference permutation of that tag's seed; a split whose size differs from the one the
    rule gives for the target's row count, or splits that are not a partition of the rows, raise.
    """
    tags = list(tags or config["split_tags"])
    unknown = [tag for tag in tags if tag not in config["split_tags"]]
    if unknown:
        raise ValueError(f"unknown split tags {unknown}; {config['task']} has {config['split_tags']}")
    rows = read_target_rows(config)
    built = {i: build_molecule(row, config=config) for i, row in enumerate(rows)}
    result = {}
    for tag in tags:
        seed = split_replicates.seed_for_tag(tag)
        indices = split_indices(len(rows), config["split"]["fractions"], seed)
        check_partition(indices, len(rows), config["expected_counts"], f"{config['task']} {tag}")
        smiles_split = {}
        for split in config["splits"]:
            for i in indices[split]:
                smiles_split.setdefault(rows[int(i)]["smiles"], set()).add(split)
        result[tag] = {}
        for split in config["splits"]:
            order = [int(i) for i in indices[split]]
            molecules = [built[i][0] for i in order]
            ids = [f"{config['dataset']}_{rows[i]['id']}" for i in order]
            labels = np.stack([m.labels for m in molecules]).astype(np.float32)
            summary = molnet._qc_summary(config, split, molecules, labels)
            summary["split_tag"], summary["split_seed"] = tag, seed
            summary["conformer_policy"] = (f"ETKDGv3 seed {config['conformer']['seed']} + {config['conformer']['mmff_variant']} "
                                           f"(1 conformer; 2-D fallback on embedding failure)")
            summary["coordinates"] = molnet.check_coordinate_scale(config, split, molecules)
            summary["conformer_generation"] = _conformer_summary([built[i][1:] for i in order], ids)
            summary["smiles_shared_with_other_split"] = int(sum(len(smiles_split[m.smiles]) > 1 for m in molecules))
            summary["label_mean_std"] = [round(float(labels.mean()), 6), round(float(labels.std()), 6)]
            summary["split_rule"] = {"policy": config["split"]["policy"], "fractions": config["split"]["fractions"],
                                     "seed": seed, "rows_of_target": len(rows),
                                     "paper_molecules": config["paper_molecules"]}
            schema = {"format": SCHEMA_FORMAT, "conformer": "etkdgv3_mmff", "hydrogens": "dropped",
                      "source": config["source_name"], "revision": str(config["processed_revision"]),
                      "coordinate_unit": "angstrom", "coordinate_source_unit": "angstrom", "coordinate_scale": 1.0,
                      "target_id": config["target_id"], "split_tag": tag, "split_seed": seed,
                      "conformer_seed": int(config["conformer"]["seed"]), "rdkit": _rdkit_version()}
            payload = molnet.payload_from_molecules(config, split, molecules, ids, summary, schema)
            result[tag][split] = (payload, molecules, summary)
    return result


def _rdkit_version() -> str:
    import rdkit

    return str(rdkit.__version__)


def build_processed(config: dict, *, splits: Sequence[str], config_path: Path, tags: Sequence[str] | None = None,
                    dry_run: bool = False) -> int:
    """Process the requested splits of the requested tags (default: every tag); returns 0 (every failure raises).

    No file is written when any target file already exists (nothing is overwritten, and nothing is half-processed
    because of an existing tag): the whole call raises first. ``dry_run`` writes nothing.
    """
    import json

    tags = list(tags or config["split_tags"])
    if not dry_run:
        existing = [processed_dir(config, tag) / f"{s}.pt" for tag in tags for s in splits
                    if (processed_dir(config, tag) / f"{s}.pt").exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite {existing[0]} ({len(existing)} existing file(s); process "
                                  "the missing tags with --tags)")
    started = datetime.now(timezone.utc).isoformat()
    result = process_task(config, tags)
    for tag in tags:
        out_dir = processed_dir(config, tag)
        for split in splits:
            payload, molecules, summary = result[tag][split]
            print(json.dumps(summary, indent=1), flush=True)
            if dry_run:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            molnet._write_qc(out_dir, split, payload["ids"], molecules, summary)
            target = out_dir / f"{split}.pt"
            partial = target.with_name(target.name + ".partial")
            torch.save(payload, partial)
            partial.replace(target)
            print(f"[{tag}/{split}] wrote {target} ({len(molecules)} molecules)", flush=True)
        if not dry_run:
            record_process_run(
                task_paths(config)["provenance"], started_utc=started, config_path=config_path, processed_dir=out_dir,
                splits=list(splits), exit_code=0, inputs=[csv_path(config)],
                extra={"dataset": config["dataset"], "task": config["task"], "target_id": config["target_id"],
                       "split_tag": tag, "split_seed": split_replicates.seed_for_tag(tag),
                       "policy": {"split": config["split"], "conformer": config["conformer"], "hydrogens": "dropped",
                                  "cropping": "none", "flags_not_dropped": list(molnet.FLAG_NAMES),
                                  "molecules_dropped": 0},
                       "rdkit": _rdkit_version()})
    return 0


# ---------------------------------------------------------------------------
# Processed split and registry surface
# ---------------------------------------------------------------------------

def check_schema_matches_task_yaml(payload: dict, path: Path) -> None:
    """Refuse a processed split of another format, revision or target than task.yaml declares for its dataset."""
    schema = payload["schema"]
    values = _read_yaml(TASK_YAML)
    if payload["dataset"] not in values["datasets"]:
        raise ValueError(f"{path}: unknown NPASS dataset {payload['dataset']!r}")
    entry = values["datasets"][payload["dataset"]]
    if schema.get("format") != SCHEMA_FORMAT:
        raise ValueError(f"{path}: schema format {schema.get('format')!r}, expected {SCHEMA_FORMAT!r}")
    if str(schema.get("revision")) != str(values["processed_revision"]):
        raise ValueError(f"{path}: processed with revision {schema.get('revision')!r} but task.yaml declares "
                         f"{values['processed_revision']!r}; reprocess with `make ds-process TASK={entry['task']}`")
    if schema.get("target_id") != entry["target_id"]:
        raise ValueError(f"{path}: file holds target {schema.get('target_id')!r}, task.yaml says {entry['target_id']!r}")
    if schema.get("coordinate_unit") != "angstrom":
        raise ValueError(f"{path}: coordinates are not recorded in Angstrom (schema {schema})")
    tag = schema.get("split_tag")
    if Path(path).parent.parent.name != tag:
        raise ValueError(f"{path}: file holds split {tag!r} but lies under {Path(path).parent.parent.name!r}")
    if tag not in [split_replicates.tag_for_seed(s) for s in values["split"]["seeds"]] or (
            split_replicates.seed_for_tag(tag) != schema.get("split_seed")):
        raise ValueError(f"{path}: split tag {tag!r} / seed {schema.get('split_seed')!r} do not match task.yaml "
                         f"split.seeds {values['split']['seeds']}")


class NpassProcessedSplit(molnet.MolnetProcessedSplit):
    """A processed NPASS split: the MoleculeNet loader with this family's schema check."""

    def _check_schema(self, payload: dict) -> None:
        check_schema_matches_task_yaml(payload, self.path)


def task_module(name: str):
    """A module-like object with the surface ``downstream.tasks.load_task`` and the CLIs expect, for one ``npass_*`` task."""
    import types

    if name not in DATASET_OF_TASK:
        raise ValueError(f"unknown NPASS task {name!r}; choose from {sorted(DATASET_OF_TASK)}")
    dataset = DATASET_OF_TASK[name]
    module = types.ModuleType(f"downstream.tasks.npass.{name}")
    module.DATASET = dataset
    module.TASK_YAML = TASK_YAML
    module.KIND = KIND
    module.load_task_config = lambda path=TASK_YAML, *, data_root=None: molnet.with_finetune_entries(
        load_task_config(path, data_root=data_root, dataset=dataset), path)
    module.processed_dir = processed_dir
    module.test_subsets = lambda config, chains, split_tag: molnet.test_subsets(
        config, chains, directory=processed_dir(config, split_tag))
    module.ProcessedSplit = NpassProcessedSplit
    return module
