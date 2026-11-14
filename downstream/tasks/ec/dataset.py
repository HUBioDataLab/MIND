"""EC task data: annotations, splits, chain -> graph processing with QC, and processed splits.

``build_processed`` turns raw structures into ``processed/<source>_<policy_tag>_<length policy><max residues>[_<processed_revision>]/<split>.pt``
(compact edge-free samples + 538-dim multi-hot labels) and writes a per-chain QC
report next to it. A split whose failed/dropped fraction exceeds
``qc.max_failed_chain_fraction`` (or any strict-policy violation) gets only its
QC report and makes the run exit non-zero.
"""

from __future__ import annotations

import csv
import json
import math
import os
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from downstream.common.graph_builder import (
    build_node_sample, compact_sample, edge_free_schema, runtime_sample,
)
from downstream.common.missing_atoms import MissingAtomError, apply_policy, parse_policy
from downstream.common.structure_io import THREE_TO_ONE, read_chains

TASK_YAML = Path(__file__).with_name("task.yaml")
KIND = "multilabel"
REQUIRED_KEYS = (
    "task", "modality", "processed_revision", "data_root", "source", "sources", "splits", "label_count",
    "test_cutoffs", "test_cutoff_unlisted_chains", "structure", "missing_atoms", "max_residues", "length_policy", "min_residues", "qc", "graph",
    "embedding", "probe", "finetune",
)
LENGTH_POLICIES = ("skip", "crop_windows")


def load_task_config(path: Path = TASK_YAML, *, data_root: Optional[Path] = None) -> dict:
    with open(path) as handle:
        config = yaml.safe_load(handle)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if config["length_policy"] not in LENGTH_POLICIES:
        raise ValueError(f"length_policy must be one of {LENGTH_POLICIES}")
    if config["source"] not in config["sources"]:
        raise ValueError(f"source {config['source']!r} is not configured under sources")
    if data_root is not None:
        config["data_root"] = str(data_root)
    return config


def task_paths(config: dict) -> Dict[str, Path]:
    root = Path(config["data_root"])
    return {
        "root": root, "raw": root / "raw", "processed": root / "processed",
        "splits": root / "splits", "provenance": root / "provenance.json",
    }


# ---------------------------------------------------------------------------
# Annotation and split files (DeepFRI formats; TorchDrug uses the same ones)
# ---------------------------------------------------------------------------

def read_annotations(path: Path, expected_terms: int) -> Tuple[List[str], Dict[str, List[int]]]:
    """``### EC-numbers`` header, term line, column header, then ``chain<TAB>t1,t2``."""
    with open(path) as handle:
        lines = [line.rstrip("\n") for line in handle]
    if not lines[0].startswith("### EC-numbers"):
        raise ValueError(f"{path}: unexpected annotation header {lines[0][:40]!r}")
    terms = lines[1].split("\t")
    if len(terms) != expected_terms or len(set(terms)) != len(terms):
        raise ValueError(f"{path}: {len(terms)} terms ({len(set(terms))} unique), expected {expected_terms}")
    index = {term: i for i, term in enumerate(terms)}
    if not lines[2].startswith("### PDB-chain"):
        raise ValueError(f"{path}: unexpected column header {lines[2][:40]!r}")
    labels: Dict[str, List[int]] = {}
    for number, line in enumerate(lines[3:], start=4):
        if not line:
            continue
        chain, _, value = line.partition("\t")
        if chain in labels:
            raise ValueError(f"{path}:{number}: duplicate chain {chain}")
        names = [name for name in value.split(",") if name]
        unknown = [name for name in names if name not in index]
        if unknown or not names:
            raise ValueError(f"{path}:{number}: chain {chain} has unknown/no terms {unknown}")
        labels[chain] = sorted({index[name] for name in names})
    return terms, labels


def read_split(path: Path) -> List[str]:
    with open(path) as handle:
        ids = [line.strip() for line in handle if line.strip()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate chain ids")
    return ids


def read_test_cutoffs(path: Path, cutoffs: Sequence[int]) -> Dict[int, set]:
    """Chains in the ``<X%`` subset have a 1 in that column (TorchDrug drops the 0 rows)."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        columns = {int(name.strip("<%")): i for i, name in enumerate(header[1:], start=1)}
        missing = [c for c in cutoffs if c not in columns]
        if missing:
            raise ValueError(f"{path}: no columns for cutoffs {missing}; header {header}")
        subsets = {c: set() for c in cutoffs}
        for row in reader:
            for cutoff in cutoffs:
                flag = row[columns[cutoff]].strip()
                if flag not in ("0", "1"):
                    raise ValueError(f"{path}: chain {row[0]} has flag {flag!r} for <{cutoff}%")
                if flag == "1":
                    subsets[cutoff].add(row[0])
    return subsets


def split_chain_id(chain: str) -> Tuple[str, str]:
    pdb_id, sep, chain_id = chain.partition("-")
    if not sep or len(pdb_id) != 4 or not chain_id:
        raise ValueError(f"chain id {chain!r} is not <PDB>-<chain>")
    return pdb_id.upper(), chain_id


def multi_hot(labels: Dict[str, List[int]], chains: Sequence[str], count: int) -> np.ndarray:
    out = np.zeros((len(chains), count), dtype=np.uint8)
    for row, chain in enumerate(chains):
        if chain not in labels:
            raise KeyError(f"chain {chain} has no EC annotation")
        out[row, labels[chain]] = 1
    return out


# ---------------------------------------------------------------------------
# Chain processing
# ---------------------------------------------------------------------------

def structure_locations(config: dict) -> Dict[str, Tuple[Path, Optional[str]]]:
    """Map every split chain to ``(structure file, chain id to select or None for single-chain files)``."""
    paths = task_paths(config)
    chains = [c for split in config["splits"] for c in read_split(paths["splits"] / f"{split}.txt")]
    if config["source"] == "deepfri_rcsb":
        out = {}
        for chain in chains:
            pdb_id, chain_id = split_chain_id(chain)
            out[chain] = (paths["raw"] / "structures" / f"{pdb_id}.cif.gz", chain_id)
        return out
    index = json.loads((paths["raw"] / "torchdrug" / "file_index.json").read_text())
    missing = [c for c in chains if c not in index]
    if missing:
        raise ValueError(f"{len(missing)} split chains have no TorchDrug file, first {missing[:5]}")
    return {c: (Path(index[c]), None) for c in chains}


def windows(num_residues: int, max_residues: int) -> List[Tuple[int, int]]:
    count = math.ceil(num_residues / max_residues)
    bounds = np.linspace(0, num_residues, count + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


@dataclass
class ChainResult:
    chain: str
    status: str            # ok | dropped | failed | strict_violation
    reason: str
    report: dict
    samples: list          # compact samples (one per window)
    window_residues: list
    sequence: str


def process_entry(job: Tuple[str, List[Tuple[str, Optional[str]]], dict]) -> List[ChainResult]:
    """Parse one structure file once and process each requested chain."""
    torch.set_num_threads(1)
    path, requests, settings = job
    policy = parse_policy(settings["missing_atoms"])
    if not Path(path).is_file():
        return [ChainResult(c, "failed", "structure_missing: no file", {}, [], [], "") for c, _ in requests]
    try:
        chains = read_chains(Path(path), mse_to_met=settings["mse_to_met"])
    except Exception as exc:  # recorded per chain and bounded by qc.max_failed_chain_fraction
        return [ChainResult(c, "failed", f"parse_error: {type(exc).__name__}: {exc}", {}, [], [], "")
                for c, _ in requests]
    results = []
    for chain, chain_id in requests:
        if chain_id is None:
            if len(chains) != 1:
                results.append(ChainResult(chain, "failed", f"chain_missing: expected one chain, file has {sorted(chains)}", {}, [], [], ""))
                continue
            record = next(iter(chains.values()))
        elif chain_id not in chains:
            results.append(ChainResult(chain, "failed", f"chain_missing: {chain_id} not in file (has {sorted(chains)[:20]})", {}, [], [], ""))
            continue
        else:
            record = chains[chain_id]
        try:
            outcome = apply_policy(record, policy)
        except MissingAtomError as exc:
            results.append(ChainResult(chain, "strict_violation", f"strict_violation: {exc}", dict(record.report), [], [], ""))
            continue
        report = outcome.report
        residues, masked, positions = outcome.residues, outcome.masked, outcome.sequence_index
        if outcome.dropped_reason:
            results.append(ChainResult(chain, "dropped", f"policy_drop: {outcome.dropped_reason}", report, [], [], ""))
            continue
        if len(residues) < settings["min_residues"]:
            results.append(ChainResult(chain, "dropped", f"too_short: {len(residues)} residues < min_residues", report, [], [], ""))
            continue
        if sum(masked) == len(masked):
            results.append(ChainResult(chain, "dropped", "all_masked: every residue is masked", report, [], [], ""))
            continue
        if len(residues) > settings["max_residues"] and settings["length_policy"] == "skip":
            results.append(ChainResult(chain, "dropped", f"too_long: {len(residues)} residues > max_residues", report, [], [], ""))
            continue
        spans = windows(len(residues), settings["max_residues"])
        try:
            samples = [
                compact_sample(build_node_sample(
                    residues[a:b], masked[a:b], chain if len(spans) == 1 else f"{chain}#w{i}",
                    sequence_index=positions[a:b],
                ))
                for i, (a, b) in enumerate(spans)
            ]
        except Exception as exc:  # recorded per chain and bounded by qc.max_failed_chain_fraction
            results.append(ChainResult(chain, "failed", f"graph_error: {type(exc).__name__}: {exc}", report, [], [], ""))
            continue
        report["windows"] = len(spans)
        sequence = "".join(THREE_TO_ONE[residue.resname] for residue in residues)
        results.append(ChainResult(chain, "ok", "", report, samples, [b - a for a, b in spans], sequence))
    return results


QC_COLUMNS = (
    "chain", "split", "status", "reason", "standard_residues", "kept_residues", "masked_residues",
    "residues_missing_backbone", "residues_incomplete_sidechain", "nonstandard_residues",
    "unexpected_atoms_removed", "mse_converted", "altloc_atoms_resolved", "num_models_in_file", "windows",
)


def build_processed(config: dict, *, splits: Sequence[str], workers: int) -> int:
    """Process the requested splits; return the exit code (0 ok, 1 QC failure)."""
    policy = parse_policy(config["missing_atoms"])
    paths = task_paths(config)
    out_dir = processed_dir(config)
    terms, labels = read_annotations(paths["raw"] / "annotations.tsv", config["label_count"])
    locations = structure_locations(config)
    settings = {
        "missing_atoms": config["missing_atoms"], "mse_to_met": bool(config["structure"]["mse_to_met"]),
        "max_residues": int(config["max_residues"]), "length_policy": config["length_policy"],
        "min_residues": int(config["min_residues"]),
    }
    schema = edge_free_schema(int(config["graph"]["sequence_neighbors_k"]), {
        "builder": "downstream.tasks.ec.dataset", "source": config["source"], "policy": policy.tag,
    })
    exit_code = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        target = out_dir / f"{split}.pt"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        chains = read_split(paths["splits"] / f"{split}.txt")
        by_file: Dict[str, List[Tuple[str, Optional[str]]]] = {}
        for chain in chains:
            path, chain_id = locations[chain]
            by_file.setdefault(str(path), []).append((chain, chain_id))
        jobs = [(path, requests, settings) for path, requests in sorted(by_file.items())]
        results: Dict[str, ChainResult] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, batch in enumerate(pool.map(process_entry, jobs, chunksize=4), start=1):
                for result in batch:
                    results[result.chain] = result
                if done % 500 == 0:
                    print(f"[{split}] {done}/{len(jobs)} structure files", flush=True)
        ordered = [results[chain] for chain in chains]
        status = Counter(r.status for r in ordered)
        bad = len(ordered) - status["ok"]
        fraction = bad / max(len(ordered), 1)
        summary = {
            "split": split, "chains": len(ordered), "status": dict(status),
            "failed_or_dropped_fraction": round(fraction, 6),
            "limit": config["qc"]["max_failed_chain_fraction"],
            "reasons": dict(Counter(r.reason.split(":")[0] for r in ordered if r.status != "ok")),
            "masked_residues": int(sum(r.report.get("masked_residues", 0) for r in ordered)),
            "residues_missing_backbone": int(sum(r.report.get("residues_missing_backbone", 0) for r in ordered)),
            "nonstandard_residues": int(sum(r.report.get("nonstandard_residues", 0) for r in ordered)),
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
            print(f"[{split}] QC FAILED: {bad}/{len(ordered)} chains not usable; {target} not written", flush=True)
            exit_code = 1
            continue
        kept = [r for r in ordered if r.status == "ok"]
        payload = {
            "schema": schema, "policy": policy.tag, "split": split, "terms": terms,
            "chains": [r.chain for r in kept],
            "labels": torch.from_numpy(multi_hot(labels, [r.chain for r in kept], len(terms))),
            "sequences": [r.sequence for r in kept],
            "samples": [s for r in kept for s in r.samples],
            "sample_chain": torch.tensor([i for i, r in enumerate(kept) for _ in r.samples]),
            "sample_residues": torch.tensor([n for r in kept for n in r.window_residues]),
            "qc": summary,
        }
        partial = target.with_name(target.name + ".partial")
        torch.save(payload, partial)
        os.replace(partial, target)
        print(f"[{split}] wrote {target} ({len(kept)} chains)", flush=True)
    return exit_code


# ---------------------------------------------------------------------------
# Processed split access
# ---------------------------------------------------------------------------

class ProcessedSplit:
    """Chains, labels and stored samples of one processed split."""

    def __init__(self, path: Path):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        for key in ("schema", "chains", "labels", "samples", "sample_chain", "sample_residues", "sequences"):
            if key not in payload:
                raise ValueError(f"{self.path}: processed split lacks {key!r}")
        self.payload = payload
        self.schema = payload["schema"]
        self.chains: List[str] = payload["chains"]
        self.labels: np.ndarray = payload["labels"].numpy()
        self.sequences: List[str] = payload["sequences"]
        self.sample_chain = payload["sample_chain"].long()
        self.sample_residues = payload["sample_residues"].long()
        if len(self.chains) != len(self.labels) or len(payload["samples"]) != len(self.sample_chain):
            raise ValueError(f"{self.path}: inconsistent chain/sample counts")

    def __len__(self) -> int:
        return len(self.payload["samples"])

    def runtime(self, index: int, model_config, *, seed: int):
        return runtime_sample(self.payload["samples"][index], self.schema, model_config, seed=seed)

    def iter_runtime(self, model_config, *, seed: int) -> Iterator:
        for index in range(len(self)):
            yield index, self.runtime(index, model_config, seed=seed)

    def chain_embeddings(self, sample_embeddings: np.ndarray) -> np.ndarray:
        """Residue-weighted mean of window embeddings per chain (identity for unwindowed chains)."""
        weights = self.sample_residues.numpy().astype(np.float64)
        out = np.zeros((len(self.chains), sample_embeddings.shape[1]), dtype=np.float64)
        totals = np.zeros(len(self.chains))
        np.add.at(out, self.sample_chain.numpy(), sample_embeddings * weights[:, None])
        np.add.at(totals, self.sample_chain.numpy(), weights)
        if (totals == 0).any():
            raise ValueError(f"{self.path}: a chain has no samples")
        return (out / totals[:, None]).astype(np.float32)


def processed_dir(config: dict) -> Path:
    policy = parse_policy(config["missing_atoms"])
    revision = str(config["processed_revision"])
    return task_paths(config)["processed"] / (
        f"{config['source']}_{policy.tag}_{config['length_policy']}{config['max_residues']}"
        + (f"_{revision}" if revision else "")
    )


def read_test_cutoff_chains(path: Path) -> set:
    """Every chain id that has a row in the cutoff table (the first column)."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return {row[0] for row in reader if row}


def test_subsets(config: dict, chains: Sequence[str]) -> Dict[int, np.ndarray]:
    """Row masks of the processed test split for every identity cutoff.

    A test chain that the cutoff table does not list raises (a silent False would drop it from every subset),
    except the chains named in ``test_cutoff_unlisted_chains``: a documented upstream gap of the DeepFRI table.
    Those stay in the full-test-set metric and in no cutoff subset, with a warning. An exception for a chain the
    table does list raises too, so the list cannot go stale.
    """
    path = task_paths(config)["splits"] / "test_cutoffs.csv"
    subsets = read_test_cutoffs(path, config["test_cutoffs"])
    listed = read_test_cutoff_chains(path)
    allowed = list(config["test_cutoff_unlisted_chains"])
    stale = [c for c in allowed if c in listed]
    if stale:
        raise ValueError(f"test_cutoff_unlisted_chains lists {stale}, which are now in test_cutoffs.csv; "
                         "remove them from task.yaml")
    unlisted = [c for c in chains if c not in listed]
    unexpected = [c for c in unlisted if c not in allowed]
    if unexpected:
        shown = unexpected[:50] + ([f"... +{len(unexpected) - 50} more"] if len(unexpected) > 50 else [])
        raise ValueError(f"{len(unexpected)} test chains are missing from test_cutoffs.csv: {shown}")
    if unlisted:
        warnings.warn(f"{unlisted} not in test_cutoffs.csv (documented in test_cutoff_unlisted_chains): "
                      "counted only in the full-test-set 'all' metric, in no cutoff subset", stacklevel=2)
    return {cutoff: np.array([c in members for c in chains]) for cutoff, members in subsets.items()}
