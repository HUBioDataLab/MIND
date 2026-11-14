"""Shared machinery of the single-label multi-class protein tasks (Fold, Reaction).

Both tasks come from Hermosilla et al. (IEConv) ``.hdf5`` structure archives with one class id per protein and
metric = accuracy. Chain processing is the EC/GO code path (``downstream.tasks.ec.dataset.process_entry``): missing
atoms are resolved by ``downstream.common.missing_atoms`` before the training adapter sees a residue (the adapter
would pad a missing sidechain atom with V and shift later atoms), the only difference being the reader (``structure_io.read_hdf5_chain``).

A task module (``fold/dataset.py``, ``reaction/dataset.py``) supplies its own label reader; everything else
(config checks, split validation, processing with QC, processed-split view, archive helpers) lives here.

Dropped proteins are never silent: ``qc_<split>.json`` and ``ProcessedSplit.report`` carry listed / kept /
dropped counts per split, and a split above ``qc.max_failed_chain_fraction`` is not written (exit code 1).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tarfile
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from data_loading.data_root import expand_data_root
from downstream.common.graph_builder import edge_free_schema, runtime_sample
from downstream.common.missing_atoms import parse_policy
from downstream.tasks.ec.dataset import QC_COLUMNS, ChainResult, process_entry, read_split

KIND = "multiclass"
LENGTH_POLICIES = ("skip", "crop_windows")
REQUIRED_KEYS = (
    "task", "modality", "source_name", "source_root", "data_root", "source", "sources", "label_count", "id_namespace",
    "splits", "train_split", "valid_split", "test_splits", "expected_chain_counts", "metric", "structure",
    "missing_atoms", "max_residues", "length_policy", "min_residues", "qc", "graph", "embedding", "probe",
    "finetune",
)


# ---------------------------------------------------------------------------
# Config and paths
# ---------------------------------------------------------------------------

def load_config(path: Path, *, expected_task: str, data_root: Optional[Path] = None) -> dict:
    with open(path) as handle:
        config = expand_data_root(yaml.safe_load(handle))
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if config["task"] != expected_task:
        raise ValueError(f"{path}: task is {config['task']!r}, expected {expected_task!r}")
    if config["metric"] != "accuracy":
        raise ValueError(f"{path}: metric must be 'accuracy' for a multiclass task, got {config['metric']!r}")
    if int(config["label_count"]) < 2:
        raise ValueError(f"{path}: label_count must be at least 2")
    if config["length_policy"] not in LENGTH_POLICIES:
        raise ValueError(f"length_policy must be one of {LENGTH_POLICIES}")
    if config["source"] not in config["sources"]:
        raise ValueError(f"source {config['source']!r} is not configured under sources")
    splits = list(config["splits"])
    roles = [config["train_split"], config["valid_split"], *config["test_splits"]]
    if sorted(roles) != sorted(splits) or len(set(splits)) != len(splits):
        raise ValueError(f"{path}: train/valid/test_splits {roles} must name every split of {splits} exactly once")
    if not config["test_splits"]:
        raise ValueError(f"{path}: at least one test split is required")
    if sorted(config["expected_chain_counts"]) != sorted(splits):
        raise ValueError(f"{path}: expected_chain_counts must name every split")
    parse_policy(config["missing_atoms"])  # an absent, unknown or malformed policy stops here
    if data_root is not None:
        config["data_root"] = str(data_root)
        config["source_root"] = str(Path(data_root).parent)
    return config


def task_paths(config: dict) -> Dict[str, Path]:
    root, source = Path(config["data_root"]), Path(config["source_root"])
    return {
        "source_root": source, "raw": source / "raw", "root": root, "splits": root / "splits",
        "processed": root / "processed", "provenance": root / "provenance.json",
    }


def processed_dir(config: dict) -> Path:
    policy = parse_policy(config["missing_atoms"])
    return task_paths(config)["processed"] / (
        f"{config['source']}_{policy.tag}_{config['length_policy']}{config['max_residues']}"
    )


def file_index_path(config: dict) -> Path:
    return task_paths(config)["raw"] / f"{config['task']}_file_index.json"


def split_roles(config: dict) -> Tuple[str, str, List[str]]:
    """``(train, valid, [tests])`` split names of any task config (EC/LBA/GO keep their 3-split lists)."""
    if "test_splits" in config:
        return config["train_split"], config["valid_split"], list(config["test_splits"])
    train, valid, test = config["splits"]
    return train, valid, [test]


# ---------------------------------------------------------------------------
# Inputs: label tables and split lists
# ---------------------------------------------------------------------------

def validate_inputs(config: dict, splits: Mapping[str, Sequence[str]], labels: Mapping[str, int],
                    classes: Sequence[str]) -> dict:
    """Cross-check split lists, label table and class list; raise on any inconsistency."""
    if len(classes) != int(config["label_count"]) or len(set(classes)) != len(classes):
        raise ValueError(f"{len(classes)} classes ({len(set(classes))} unique), expected label_count {config['label_count']}")
    counts = {name: len(ids) for name, ids in splits.items()}
    expected = {k: int(v) for k, v in config["expected_chain_counts"].items()}
    if counts != expected:
        raise ValueError(f"split counts {counts} differ from expected {expected} (expected_chain_counts)")
    owner: Dict[str, str] = {}
    for name, ids in splits.items():
        for chain in ids:
            if chain in owner:
                raise ValueError(f"{chain} is in both {owner[chain]} and {name}")
            owner[chain] = name
    unlabeled = [c for c in owner if c not in labels]
    if unlabeled:
        raise ValueError(f"{len(unlabeled)} split proteins have no label, first {unlabeled[:5]}")
    bad = [c for c in owner if not 0 <= labels[c] < len(classes)]
    if bad:
        raise ValueError(f"{len(bad)} labels outside [0, {len(classes)}), first {bad[:5]}")
    train_classes = {labels[c] for c in splits[config["train_split"]]}
    per_split = {}
    for name, ids in splits.items():
        present = {labels[c] for c in ids}
        per_split[name] = {"classes_present": len(present), "classes_absent_from_train": len(present - train_classes)}
    return {"split_counts": counts, "classes": len(classes), "per_split": per_split}


def structure_locations(config: dict, chains: Sequence[str]) -> Dict[str, Tuple[Path, Optional[str]]]:
    index = json.loads(file_index_path(config).read_text())
    missing = [c for c in chains if c not in index]
    if missing:
        raise ValueError(f"{len(missing)} split proteins have no structure file, first {missing[:5]}")
    return {c: (Path(index[c]), None) for c in chains}


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def build_processed(config: dict, labels: Mapping[str, int], classes: Sequence[str], *, splits: Sequence[str],
                    workers: int, builder: str) -> int:
    """Process the requested splits into ``<split>.pt``; returns 0, or 1 when a split fails QC (file not written)."""
    policy = parse_policy(config["missing_atoms"])
    paths = task_paths(config)
    out_dir = processed_dir(config)
    all_splits = {s: read_split(paths["splits"] / f"{s}.txt") for s in config["splits"]}
    validate_inputs(config, all_splits, labels, classes)
    settings = {
        "missing_atoms": config["missing_atoms"], "mse_to_met": bool(config["structure"]["mse_to_met"]),
        "max_residues": int(config["max_residues"]), "length_policy": config["length_policy"],
        "min_residues": int(config["min_residues"]),
    }
    schema = edge_free_schema(int(config["graph"]["sequence_neighbors_k"]), {
        "builder": builder, "source": config["source"], "policy": policy.tag,
    })
    exit_code = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if split not in all_splits:
            raise ValueError(f"unknown split {split!r}; choose from {sorted(all_splits)}")
        target = out_dir / f"{split}.pt"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        chains = all_splits[split]
        locations = structure_locations(config, chains)
        by_file: Dict[str, List[Tuple[str, Optional[str]]]] = {}
        for chain in chains:
            path, chain_id = locations[chain]
            by_file.setdefault(str(path), []).append((chain, chain_id))
        jobs = [(path, requests, settings) for path, requests in sorted(by_file.items())]
        results: Dict[str, ChainResult] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, batch in enumerate(pool.map(process_entry, jobs, chunksize=8), start=1):
                for result in batch:
                    results[result.chain] = result
                if done % 1000 == 0:
                    print(f"[{split}] {done}/{len(jobs)} structure files", flush=True)
        ordered = [results[chain] for chain in chains]
        status = Counter(r.status for r in ordered)
        bad = len(ordered) - status["ok"]
        fraction = bad / max(len(ordered), 1)
        kept = [r for r in ordered if r.status == "ok"]
        summary = {
            "split": split, "listed_chains": len(ordered), "kept_chains": len(kept), "status": dict(status),
            "failed_or_dropped_fraction": round(fraction, 6), "limit": config["qc"]["max_failed_chain_fraction"],
            "reasons": dict(Counter(r.reason.split(":")[0] for r in ordered if r.status != "ok")),
            "masked_residues": int(sum(r.report.get("masked_residues", 0) for r in ordered)),
            "residues_missing_backbone": int(sum(r.report.get("residues_missing_backbone", 0) for r in ordered)),
            "nonstandard_residues": int(sum(r.report.get("nonstandard_residues", 0) for r in ordered)),
            "classes_kept": int(len({labels[r.chain] for r in kept})),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(out_dir / f"qc_{split}.tsv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QC_COLUMNS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for r in ordered:
                writer.writerow({"chain": r.chain, "split": split, "status": r.status, "reason": r.reason, **r.report})
        with open(out_dir / f"qc_{split}.json", "w") as handle:
            json.dump(summary, handle, indent=1)
        print(json.dumps(summary, indent=1), flush=True)
        if status["strict_violation"] or fraction > config["qc"]["max_failed_chain_fraction"]:
            print(f"[{split}] QC FAILED: {bad}/{len(ordered)} proteins not usable; {target} not written", flush=True)
            exit_code = 1
            continue
        kept_ids = [r.chain for r in kept]
        payload = {
            "schema": schema, "policy": policy.tag, "split": split, "classes": list(classes), "chains": kept_ids,
            "labels": torch.tensor([labels[c] for c in kept_ids], dtype=torch.long),
            "sequences": [r.sequence for r in kept],
            "samples": [s for r in kept for s in r.samples],
            "sample_chain": torch.tensor([i for i, r in enumerate(kept) for _ in r.samples]),
            "sample_residues": torch.tensor([n for r in kept for n in r.window_residues]),
            "qc": summary,
        }
        partial = target.with_name(target.name + ".partial")
        torch.save(payload, partial)
        os.replace(partial, target)
        print(f"[{split}] wrote {target} ({len(kept)} of {len(ordered)} proteins)", flush=True)
    return exit_code


# ---------------------------------------------------------------------------
# Processed split view
# ---------------------------------------------------------------------------

class MulticlassProcessedSplit:
    """Chains, integer class labels and samples of one processed split.

    Interface-compatible with ``downstream.tasks.ec.dataset.ProcessedSplit`` except that ``labels`` is a 1-D
    int64 array of class ids and ``classes`` lists the class names by id.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        for key in ("schema", "chains", "labels", "samples", "sample_chain", "sample_residues", "sequences",
                    "classes", "qc"):
            if key not in payload:
                raise ValueError(f"{self.path}: processed split lacks {key!r}")
        labels = payload["labels"]
        if labels.dtype != torch.long or labels.ndim != 1 or len(labels) != len(payload["chains"]):
            raise ValueError(f"{self.path}: labels must be a 1-D int64 vector with one id per chain, "
                             f"got {tuple(labels.shape)} {labels.dtype}")
        if len(labels) and (int(labels.min()) < 0 or int(labels.max()) >= len(payload["classes"])):
            raise ValueError(f"{self.path}: class id outside [0, {len(payload['classes'])})")
        if len(payload["samples"]) != len(payload["sample_chain"]) or len(payload["sequences"]) != len(payload["chains"]):
            raise ValueError(f"{self.path}: inconsistent chain/sample counts")
        self.schema, self.payload = payload["schema"], payload
        self.classes: List[str] = payload["classes"]
        self.chains: List[str] = payload["chains"]
        self.labels = labels.numpy()
        self.sequences: List[str] = payload["sequences"]
        self.sample_chain = payload["sample_chain"].long()
        self.sample_residues = payload["sample_residues"].long()
        qc = payload["qc"]
        self.report = {"listed_chains": int(qc["listed_chains"]), "kept_chains": len(self.chains),
                       "dropped_chains": int(qc["listed_chains"]) - len(self.chains)}

    def __len__(self) -> int:
        return len(self.payload["samples"])

    def runtime(self, index: int, model_config, *, seed: int):
        return runtime_sample(self.payload["samples"][index], self.schema, model_config, seed=seed)

    def iter_runtime(self, model_config, *, seed: int) -> Iterator:
        for index in range(len(self)):
            yield index, self.runtime(index, model_config, seed=seed)

    def chain_embeddings(self, sample_embeddings: np.ndarray) -> np.ndarray:
        """Residue-weighted mean of window embeddings per chain."""
        if len(sample_embeddings) != len(self):
            raise ValueError(f"expected {len(self)} sample embeddings, got {len(sample_embeddings)}")
        weights = self.sample_residues.numpy().astype(np.float64)
        out = np.zeros((len(self.chains), sample_embeddings.shape[1]), dtype=np.float64)
        totals = np.zeros(len(self.chains))
        np.add.at(out, self.sample_chain.numpy(), sample_embeddings * weights[:, None])
        np.add.at(totals, self.sample_chain.numpy(), weights)
        if (totals == 0).any():
            raise ValueError(f"{self.path}: a chain has no samples")
        return (out / totals[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Archive helpers (data-processing CLIs only)
# ---------------------------------------------------------------------------

def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract_archive(archive: Path, dest: Path) -> None:
    """Extract a zip or tar archive; members escaping ``dest`` raise. Unknown formats raise."""
    dest = dest.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                target = (dest / member).resolve()
                if target != dest and not str(target).startswith(str(dest) + os.sep):
                    raise RuntimeError(f"{archive}: member {member!r} escapes {dest}")
            handle.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                target = (dest / member.name).resolve()
                if target != dest and not str(target).startswith(str(dest) + os.sep):
                    raise RuntimeError(f"{archive}: member {member.name!r} escapes {dest}")
                if member.issym() or member.islnk() or member.isdev():
                    raise RuntimeError(f"{archive}: link or device member {member.name!r} refused")
            handle.extractall(dest)
    else:
        raise RuntimeError(f"{archive} is neither a zip nor a tar archive")


def index_hdf5(root: Path, expected_names: Sequence[str], *, what: str) -> Dict[str, str]:
    """Map every expected protein name to its unique ``<name>.hdf5`` under ``root``; any gap or extra raises."""
    found: Dict[str, str] = {}
    for path in sorted(Path(root).rglob("*.hdf5")):
        name = path.name[: -len(".hdf5")]
        if name in found:
            raise RuntimeError(f"{what}: {name} appears twice ({found[name]} and {path})")
        found[name] = str(path)
    missing = [n for n in expected_names if n not in found]
    if missing:
        raise RuntimeError(f"{what}: {len(missing)} of {len(expected_names)} listed proteins have no hdf5 file under "
                           f"{root}, first {missing[:5]}; {len(found)} hdf5 files found")
    extra = sorted(set(found) - set(expected_names))
    if extra:
        raise RuntimeError(f"{what}: {len(extra)} hdf5 files under {root} are not listed, first {extra[:5]}")
    return {name: found[name] for name in expected_names}
