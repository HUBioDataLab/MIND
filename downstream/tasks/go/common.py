"""Gene Ontology task family (go_mf, go_bp, go_cc): annotations, splits, processing, branch views.

Three downstream tasks share one processed file per split (same chains and graphs) that stores one
multi-hot label matrix per ontology branch. ``GoProcessedSplit`` is the per-branch view the generic
``downstream.embed / probe / finetune`` CLIs consume: it keeps only chains with at least one term in
the branch (``unannotated_policy: exclude``), so the CAFA Fmax contract (every evaluated protein has a
true term) holds by construction and is re-checked on load.

Chain processing is the EC code path (``downstream.tasks.ec.dataset.process_entry``): missing atoms are
resolved by ``downstream.common.missing_atoms`` before the training adapter ever sees a residue, because
the adapter pads a missing sidechain atom with V and shifts later atoms into wrong slots (G5).
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from data_loading.data_root import expand_data_root
from downstream.common.graph_builder import edge_free_schema, runtime_sample
from downstream.common.missing_atoms import parse_policy
from downstream.tasks.ec.dataset import (
    QC_COLUMNS, ChainResult, process_entry, read_split, read_test_cutoffs, split_chain_id,
)

TASK_YAML = Path(__file__).with_name("task.yaml")
KIND = "multilabel"
BRANCH_ORDER = ("mf", "bp", "cc")
ONTOLOGY = {"mf": "molecular_function", "bp": "biological_process", "cc": "cellular_component"}
UNANNOTATED_POLICIES = ("exclude",)
REQUIRED_KEYS = (
    "family", "modality", "processed_revision", "source_name", "source_root", "data_root", "source", "sources", "branches", "splits",
    "expected_chain_counts", "test_cutoffs", "unannotated_policy", "structure", "missing_atoms", "max_residues",
    "length_policy", "min_residues", "qc", "graph", "embedding", "probe", "finetune",
)
LENGTH_POLICIES = ("skip", "crop_windows")
_GO_ID = re.compile(r"^GO:\d{7}$")


# ---------------------------------------------------------------------------
# Config and paths
# ---------------------------------------------------------------------------

def load_task_config(path: Path = TASK_YAML, *, data_root: Optional[Path] = None, branch: str) -> dict:
    """Task config flattened for ``branch``; ``config['branches']`` keeps all three for processing."""
    if branch not in ONTOLOGY:
        raise ValueError(f"unknown GO branch {branch!r}; choose from {BRANCH_ORDER}")
    with open(path) as handle:
        config = expand_data_root(yaml.safe_load(handle))
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if sorted(config["branches"]) != sorted(BRANCH_ORDER):
        raise ValueError(f"{path}: branches must be exactly {BRANCH_ORDER}, got {sorted(config['branches'])}")
    for key, expected in ONTOLOGY.items():
        entry = config["branches"][key]
        if entry.get("ontology") != expected or int(entry.get("label_count", 0)) < 1:
            raise ValueError(f"{path}: branch {key} must name ontology {expected!r} with a positive label_count")
    if config["unannotated_policy"] not in UNANNOTATED_POLICIES:
        raise ValueError(f"unannotated_policy must be one of {UNANNOTATED_POLICIES}; "
                         f"got {config['unannotated_policy']!r} (keep_zero_rows is not implemented)")
    if config["length_policy"] not in LENGTH_POLICIES:
        raise ValueError(f"length_policy must be one of {LENGTH_POLICIES}")
    if config["source"] not in config["sources"]:
        raise ValueError(f"source {config['source']!r} is not configured under sources")
    if 95 not in config["test_cutoffs"] or 30 not in config["test_cutoffs"]:
        raise ValueError("test_cutoffs must include 30 (hard subset) and 95 (full test set)")
    if list(config["splits"]) != ["train", "valid", "test"]:
        raise ValueError("splits must be [train, valid, test]")
    if sorted(config["expected_chain_counts"]) != sorted(config["splits"]):
        raise ValueError("expected_chain_counts must name every split")
    parse_policy(config["missing_atoms"])  # UNDECIDED or malformed policies stop here
    if data_root is not None:
        config["data_root"] = str(data_root)
        config["source_root"] = str(Path(data_root).parent)
    config["branch"] = branch
    config["task"] = config["branches"][branch]["task"]
    config["ontology"] = ONTOLOGY[branch]
    config["label_count"] = int(config["branches"][branch]["label_count"])
    return config


def task_paths(config: dict) -> Dict[str, Path]:
    root, source = Path(config["data_root"]), Path(config["source_root"])
    return {
        "source_root": source, "raw": source / "raw", "root": root, "splits": root / "splits",
        "processed": root / "processed", "provenance": root / "provenance.json",
    }


def processed_dir(config: dict) -> Path:
    policy = parse_policy(config["missing_atoms"])
    revision = str(config["processed_revision"])
    return task_paths(config)["processed"] / (
        f"{config['source']}_{policy.tag}_{config['length_policy']}{config['max_residues']}"
        + (f"_{revision}" if revision else "")
    )


# ---------------------------------------------------------------------------
# Annotation file
# ---------------------------------------------------------------------------

def read_go_annotations(path: Path, expected_terms: Mapping[str, int]) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[int]]]]:
    """Parse the DeepFRI GO annotation file.

    Returns ``(terms, labels)``: ``terms[branch]`` is the ordered GO id list of the branch and
    ``labels[chain][branch]`` the sorted term indices (possibly empty). Every deviation raises.
    """
    with open(path) as handle:
        lines = [line.rstrip("\n") for line in handle]
    terms: Dict[str, List[str]] = {}
    cursor = 0
    for branch in BRANCH_ORDER:
        ontology = ONTOLOGY[branch]
        if len(lines) < cursor + 4:
            raise ValueError(f"{path}: file ends inside the {ontology} block")
        if lines[cursor] != f"### GO-terms ({ontology})":
            raise ValueError(f"{path}:{cursor + 1}: expected '### GO-terms ({ontology})', got {lines[cursor][:50]!r}")
        if lines[cursor + 2] != f"### GO-names ({ontology})":
            raise ValueError(f"{path}:{cursor + 3}: expected '### GO-names ({ontology})', got {lines[cursor + 2][:50]!r}")
        names = lines[cursor + 1].split("\t")
        labels_names = lines[cursor + 3].split("\t")
        bad = [t for t in names if not _GO_ID.match(t)]
        if bad:
            raise ValueError(f"{path}:{cursor + 2}: {len(bad)} malformed GO ids in {ontology}, first {bad[:3]}")
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: duplicate GO terms in {ontology}")
        if len(names) != int(expected_terms[branch]):
            raise ValueError(f"{path}: {ontology} has {len(names)} terms, expected {expected_terms[branch]}")
        if len(labels_names) != len(names):
            raise ValueError(f"{path}: {ontology} has {len(names)} terms but {len(labels_names)} names")
        terms[branch] = names
        cursor += 4
    if len(lines) <= cursor or not lines[cursor].startswith("### PDB-chain"):
        raise ValueError(f"{path}:{cursor + 1}: expected the '### PDB-chain' header")
    index = {branch: {term: i for i, term in enumerate(terms[branch])} for branch in BRANCH_ORDER}
    labels: Dict[str, Dict[str, List[int]]] = {}
    for number, line in enumerate(lines[cursor + 1:], start=cursor + 2):
        if not line:
            continue
        cells = line.split("\t")
        if len(cells) > 4:
            raise ValueError(f"{path}:{number}: {len(cells)} columns, expected at most 4")
        chain = cells[0]
        split_chain_id(chain)
        if chain in labels:
            raise ValueError(f"{path}:{number}: duplicate chain {chain}")
        cells += [""] * (4 - len(cells))  # a trailing empty cell may be trimmed by the writer
        per_branch = {}
        for branch, cell in zip(BRANCH_ORDER, cells[1:]):
            names = [name for name in cell.split(",") if name]
            unknown = [name for name in names if name not in index[branch]]
            if unknown:
                raise ValueError(f"{path}:{number}: chain {chain} has terms outside the {ONTOLOGY[branch]} list: {unknown[:3]}")
            per_branch[branch] = sorted({index[branch][name] for name in names})
        labels[chain] = per_branch
    if not labels:
        raise ValueError(f"{path}: no chain rows")
    return terms, labels


def multi_hot_branch(labels: Mapping[str, Mapping[str, List[int]]], chains: Sequence[str], branch: str, count: int) -> np.ndarray:
    out = np.zeros((len(chains), count), dtype=np.uint8)
    for row, chain in enumerate(chains):
        if chain not in labels:
            raise KeyError(f"chain {chain} has no GO annotation row")
        out[row, labels[chain][branch]] = 1
    return out


def validate_inputs(config: dict, splits: Mapping[str, Sequence[str]], labels: Mapping[str, Mapping[str, List[int]]]) -> dict:
    """Cross-check split lists against the annotation table; raise on any inconsistency."""
    counts = {name: len(ids) for name, ids in splits.items()}
    expected = {k: int(v) for k, v in config["expected_chain_counts"].items()}
    if counts != expected:
        raise ValueError(f"split chain counts {counts} differ from expected {expected} (expected_chain_counts)")
    owner: Dict[str, str] = {}
    for name, ids in splits.items():
        for chain in ids:
            if chain in owner:
                raise ValueError(f"chain {chain} is in both {owner[chain]} and {name}")
            owner[chain] = name
    unlabeled = [c for c in owner if c not in labels]
    if unlabeled:
        raise ValueError(f"{len(unlabeled)} split chains have no annotation row, first {unlabeled[:5]}")
    no_term = [c for c in owner if not any(labels[c][b] for b in BRANCH_ORDER)]
    return {
        "split_counts": counts, "chains_without_any_term": len(no_term),
        "annotated_per_branch": {b: {n: int(sum(bool(labels[c][b]) for c in ids)) for n, ids in splits.items()}
                                 for b in BRANCH_ORDER},
    }


# ---------------------------------------------------------------------------
# Structure locations and processing
# ---------------------------------------------------------------------------

def file_index_path(config: dict) -> Path:
    return task_paths(config)["raw"] / f"{config['sources'][config['source']]['extract_dir']}_file_index.json"


def structure_locations(config: dict, chains: Sequence[str]) -> Dict[str, Tuple[Path, Optional[str]]]:
    index = json.loads(file_index_path(config).read_text())
    missing = [c for c in chains if c not in index]
    if missing:
        raise ValueError(f"{len(missing)} split chains have no structure file, first {missing[:5]}")
    return {c: (Path(index[c]), None) for c in chains}


def build_processed(config: dict, *, splits: Sequence[str], workers: int) -> int:
    """Process the requested splits into ONE file per split with all three label matrices.

    Returns 0 on success, 1 when a split exceeds ``qc.max_failed_chain_fraction`` (its file is not written).
    """
    policy = parse_policy(config["missing_atoms"])
    paths = task_paths(config)
    out_dir = processed_dir(config)
    expected_terms = {b: config["branches"][b]["label_count"] for b in BRANCH_ORDER}
    terms, labels = read_go_annotations(
        paths["raw"] / config["sources"][config["source"]]["extract_dir"] / config["sources"][config["source"]]["annotation_file"],
        expected_terms)
    all_splits = {s: read_split(paths["splits"] / f"{s}.txt") for s in config["splits"]}
    validate_inputs(config, all_splits, labels)
    settings = {
        "missing_atoms": config["missing_atoms"], "mse_to_met": bool(config["structure"]["mse_to_met"]),
        "max_residues": int(config["max_residues"]), "length_policy": config["length_policy"],
        "min_residues": int(config["min_residues"]),
    }
    schema = edge_free_schema(int(config["graph"]["sequence_neighbors_k"]), {
        "builder": "downstream.tasks.go.common", "source": config["source"], "policy": policy.tag,
    })
    exit_code = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
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
            for done, batch in enumerate(pool.map(process_entry, jobs, chunksize=4), start=1):
                for result in batch:
                    results[result.chain] = result
                if done % 500 == 0:
                    print(f"[{split}] {done}/{len(jobs)} structure files", flush=True)
        ordered = [results[chain] for chain in chains]
        status = Counter(r.status for r in ordered)
        bad = len(ordered) - status["ok"]
        fraction = bad / max(len(ordered), 1)
        kept = [r for r in ordered if r.status == "ok"]
        summary = {
            "split": split, "chains": len(ordered), "status": dict(status),
            "failed_or_dropped_fraction": round(fraction, 6), "limit": config["qc"]["max_failed_chain_fraction"],
            "reasons": dict(Counter(r.reason.split(":")[0] for r in ordered if r.status != "ok")),
            "masked_residues": int(sum(r.report.get("masked_residues", 0) for r in ordered)),
            "residues_missing_backbone": int(sum(r.report.get("residues_missing_backbone", 0) for r in ordered)),
            "nonstandard_residues": int(sum(r.report.get("nonstandard_residues", 0) for r in ordered)),
            "annotated_per_branch": {b: int(sum(bool(labels[r.chain][b]) for r in kept)) for b in BRANCH_ORDER},
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
        kept_ids = [r.chain for r in kept]
        payload = {
            "schema": schema, "policy": policy.tag, "split": split, "terms": terms, "chains": kept_ids,
            "labels": {b: torch.from_numpy(multi_hot_branch(labels, kept_ids, b, len(terms[b]))) for b in BRANCH_ORDER},
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
# Per-branch view of a processed split
# ---------------------------------------------------------------------------

class GoProcessedSplit:
    """Chains, labels and samples of one processed split restricted to one ontology branch.

    Interface-compatible with ``downstream.tasks.ec.dataset.ProcessedSplit`` (``chains``, ``labels``,
    ``sequences``, ``sample_chain``, ``sample_residues``, ``payload['samples']``, ``runtime``,
    ``iter_runtime``, ``chain_embeddings``) but indices refer to the branch view.
    """

    def __init__(self, path: Path, *, branch: str):
        if branch not in ONTOLOGY:
            raise ValueError(f"unknown GO branch {branch!r}")
        self.path, self.branch = Path(path), branch
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        for key in ("schema", "chains", "labels", "samples", "sample_chain", "sample_residues", "sequences", "terms"):
            if key not in payload:
                raise ValueError(f"{self.path}: processed split lacks {key!r}")
        if not isinstance(payload["labels"], dict) or sorted(payload["labels"]) != sorted(BRANCH_ORDER):
            raise ValueError(f"{self.path}: labels must hold exactly the branches {BRANCH_ORDER}")
        chains: List[str] = payload["chains"]
        for name in BRANCH_ORDER:  # the shared file must be sound for every branch, not only the requested one
            matrix = payload["labels"][name].numpy()
            if matrix.ndim != 2 or matrix.shape[0] != len(chains) or matrix.shape[1] != len(payload["terms"][name]):
                raise ValueError(f"{self.path}: label matrix {matrix.shape} does not match {len(chains)} chains / "
                                 f"{len(payload['terms'][name])} {name} terms")
            if not np.isin(matrix, (0, 1)).all():
                raise ValueError(f"{self.path}: {name} labels must be 0/1")
        labels = payload["labels"][branch].numpy()
        sample_chain = payload["sample_chain"].long()
        if len(payload["samples"]) != len(sample_chain) or len(payload["sequences"]) != len(chains):
            raise ValueError(f"{self.path}: inconsistent chain/sample counts")
        keep = np.where(labels.sum(axis=1) > 0)[0]
        if keep.size == 0:
            raise ValueError(f"{self.path}: no chain has a {branch} term; refusing an empty branch view")
        remap = np.full(len(chains), -1, dtype=np.int64)
        remap[keep] = np.arange(keep.size)
        sample_rows = np.where(remap[sample_chain.numpy()] >= 0)[0]
        self.schema = payload["schema"]
        self.terms: List[str] = payload["terms"][branch]
        self.chains = [chains[i] for i in keep]
        self.labels = labels[keep]
        self.sequences = [payload["sequences"][i] for i in keep]
        self._sample_rows = sample_rows
        self.sample_chain = torch.from_numpy(remap[sample_chain.numpy()[sample_rows]]).long()
        self.sample_residues = payload["sample_residues"].long()[sample_rows]
        self.payload = {"samples": [payload["samples"][i] for i in sample_rows], "schema": self.schema, "qc": payload.get("qc")}
        self.report = {
            "branch": branch, "processed_chains": len(chains), "kept_chains": int(keep.size),
            "excluded_unannotated_chains": int(len(chains) - keep.size),
            "unannotated_policy": "exclude",
        }

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


def test_subsets(config: dict, chains: Sequence[str]) -> Dict[int, np.ndarray]:
    """Row masks of the (branch-filtered) test chains per identity cutoff.

    Unlike EC, a test chain that the cutoff table does not list raises: a silent False would
    drop it from every subset.
    """
    subsets = read_test_cutoffs(task_paths(config)["splits"] / "test_cutoffs.csv", config["test_cutoffs"])
    listed = set()
    with open(task_paths(config)["splits"] / "test_cutoffs.csv", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        listed = {row[0] for row in reader}
    unlisted = [c for c in chains if c not in listed]
    if unlisted:
        raise ValueError(f"{len(unlisted)} test chains are missing from test_cutoffs.csv, first {unlisted[:5]}")
    return {cutoff: np.array([c in members for c in chains]) for cutoff, members in subsets.items()}
