"""Secondary-structure task data (CB513 / TS115 / CASP12, NetSurfP-2.0 training set): config, records, processing, splits.

Per-residue task: the literature records are sequence + label arrays, the encoder reads a structure. ``process`` maps each
dataset chain to its RCSB structure (``mapping.py``), builds the usual edge-free chain graphs (the EC/GO code path:
``apply_policy`` + ``build_node_sample``) and stores, per window of at most ``max_residues`` residues, one label per
structure residue (-1 = no label, with the reason counted). One processed directory serves both tasks; ``ss_q3`` reads
the ``ss3`` labels, ``ss_q8`` the ``ss8`` labels.

Coverage is part of the data: ``qc_<split>.json`` and ``ProcessedSplit.coverage`` give, per split, the literature chains and
residues, those evaluated and every loss by reason. A split whose failed/dropped chain fraction exceeds
``qc.max_failed_chain_fraction`` is not written (exit 1).
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from data_loading.data_root import expand_data_root
from downstream.common.graph_builder import build_node_sample, compact_sample, edge_free_schema, runtime_sample
from downstream.common.missing_atoms import MissingAtomError, apply_policy, parse_policy
from downstream.common.structure_io import read_chains
from downstream.tasks.ec.dataset import windows
from downstream.tasks.ss import mapping as M

TASK_YAML = Path(__file__).with_name("task.yaml")
KIND = "multiclass"
PER_RESIDUE = True  # embedding / fine-tuning rows are residues, not chains
REQUIRED_KEYS = (
    "task", "modality", "processed_revision", "data_root", "source", "sources", "splits", "train_split", "valid_split",
    "test_splits", "expected_chain_counts", "metric", "structure", "missing_atoms", "max_residues", "length_policy",
    "min_residues", "alignment", "qc", "graph", "embedding", "probe", "finetune", "finetune_seconds_per_epoch",
)
LABEL_COUNTS = {name: len(classes) for name, classes in M.LABEL_SETS.items()}
# ``<PDB id>-<chain>``; a chain that occurs twice in one split (CB513 has 513 regions of 434 proteins) gets ``#<k>`` for k >= 2
CHAIN_ID = re.compile(r"^(?P<pdb>[0-9][A-Za-z0-9]{3})[-_](?P<chain>[A-Za-z0-9]+)(?:#\d+)?$")
QC_COLUMNS = ("chain", "split", "status", "reason", "dataset_residues", "structure_residues", "identity", "aligned_fraction",
              "masked_residues",
              *M.RESIDUE_REASONS)


def load_task_config(path: Path = TASK_YAML, *, data_root: Optional[Path] = None, label_set: str = "ss3") -> dict:
    with open(path) as handle:
        config = expand_data_root(yaml.safe_load(handle))
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if label_set not in LABEL_COUNTS:
        raise ValueError(f"label_set must be one of {sorted(LABEL_COUNTS)}, got {label_set!r}")
    if config["metric"] != "accuracy":
        raise ValueError(f"{path}: metric must be 'accuracy', got {config['metric']!r}")
    if config["length_policy"] != "crop_windows":
        raise ValueError(f"{path}: length_policy must be crop_windows (a per-residue label of a skipped chain is a silent loss)")
    if config["source"] not in config["sources"]:
        raise ValueError(f"source {config['source']!r} is not configured under sources")
    splits = list(config["splits"])
    roles = [config["train_split"], config["valid_split"], *config["test_splits"]]
    if sorted(roles) != sorted(splits) or len(set(splits)) != len(splits):
        raise ValueError(f"{path}: train/valid/test_splits {roles} must name every split of {splits} exactly once")
    if sorted(config["expected_chain_counts"]) != sorted(splits):
        raise ValueError(f"{path}: expected_chain_counts must name every split")
    parse_policy(config["missing_atoms"])
    for key in ("min_identity", "min_aligned_fraction"):
        if key not in config["alignment"] or not 0.0 < float(config["alignment"][key]) <= 1.0:
            raise ValueError(f"{path}: alignment.{key} must be in (0, 1]")
    if data_root is not None:
        config["data_root"] = str(data_root)
    config["label_set"] = label_set
    config["label_count"] = LABEL_COUNTS[label_set]
    config["task"] = {"ss3": "ss_q3", "ss8": "ss_q8"}[label_set]
    return config


def task_paths(config: dict) -> Dict[str, Path]:
    root = Path(config["data_root"])
    return {"root": root, "raw": root / "raw", "records": root / "raw" / "records", "structures": root / "raw" / "structures",
            "splits": root / "splits", "processed": root / "processed", "provenance": root / "provenance.json"}


def processed_dir(config: dict) -> Path:
    policy = parse_policy(config["missing_atoms"])
    revision = str(config["processed_revision"])
    return task_paths(config)["processed"] / (
        f"{config['source']}_{policy.tag}_{config['length_policy']}{config['max_residues']}" + (f"_{revision}" if revision else ""))


# ---------------------------------------------------------------------------
# Dataset records (written by ``prepare download`` from the TAPE LMDB files)
# ---------------------------------------------------------------------------

def write_records(path: Path, records: Sequence[dict]) -> str:
    """``records/<split>.jsonl.gz`` (one JSON object per chain); an existing file with other content raises."""
    lines = [json.dumps({"id": r["id"], "primary": r["primary"], "ss3": "".join(map(str, r["ss3"].tolist())),
                         "ss8": "".join(map(str, r["ss8"].tolist())),
                         "observed": "".join(map(str, r["observed"].tolist()))}) for r in records]
    content = ("\n".join(lines) + "\n").encode()
    if path.exists():
        with gzip.open(path, "rb") as handle:
            if handle.read() != content:
                raise FileExistsError(f"{path} exists with different content")
        return "unchanged"
    partial = path.with_name(path.name + ".partial")
    with gzip.GzipFile(partial, "wb", mtime=0) as handle:
        handle.write(content)
    os.replace(partial, path)
    return "written"


def read_records(path: Path) -> List[dict]:
    records, seen = [], set()
    with gzip.open(path, "rt") as handle:
        for number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if raw["id"] in seen:
                raise ValueError(f"{path}:{number}: duplicate chain id {raw['id']}")
            seen.add(raw["id"])
            record = {"id": raw["id"], "primary": raw["primary"]}
            for key in ("ss3", "ss8", "observed"):
                record[key] = np.array(list(raw[key]), dtype=np.int64)
            M.check_label_arrays(raw["id"], record["primary"], record["ss3"], record["ss8"], record["observed"])
            records.append(record)
    return records


def read_split_ids(config: dict, split: str) -> List[str]:
    path = task_paths(config)["splits"] / f"{split}.txt"
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate chain ids")
    return ids


def parse_chain_id(chain: str) -> Optional[Tuple[str, str]]:
    match = CHAIN_ID.match(chain)
    return (match["pdb"].upper(), match["chain"]) if match else None


# ---------------------------------------------------------------------------
# Chain processing
# ---------------------------------------------------------------------------

@dataclass
class ChainResult:
    chain: str
    status: str                     # ok | dropped | failed | strict_violation
    reason: str
    report: dict
    samples: list = field(default_factory=list)
    window_residues: list = field(default_factory=list)
    window_labels: dict = field(default_factory=dict)   # label set -> [int64 array per window], -1 = no label
    sequence: str = ""


def _lost_counts(record: dict, evaluated: np.ndarray) -> Dict[str, List[int]]:
    """Per label set, the class counts of the dataset residues that are not evaluated."""
    lost = ~evaluated
    return {name: np.bincount(record[name][lost], minlength=LABEL_COUNTS[name]).tolist() for name in M.LABEL_SETS}


def _unprocessed(record: dict, status: str, reason: str, report: Optional[dict] = None) -> ChainResult:
    n = len(record["primary"])
    buckets = M.account_residues(record["id"], n, Counter({"chain_not_processed": n}))
    base = {"dataset_residues": n, "buckets": buckets, "lost_labels": _lost_counts(record, np.zeros(n, dtype=bool))}
    return ChainResult(record["id"], status, reason, {**(report or {}), **base})


def process_entry(job: Tuple[str, List[Tuple[str, str, dict]], dict]) -> List[ChainResult]:
    """Parse one structure file once and map every requested dataset chain onto it."""
    torch.set_num_threads(1)
    path, requests, settings = job
    policy = parse_policy(settings["missing_atoms"])
    if not Path(path).is_file():
        return [_unprocessed(r, "failed", "structure_missing: no file") for _, _, r in requests]
    try:
        chains = read_chains(Path(path), mse_to_met=settings["mse_to_met"])
    except Exception as exc:  # recorded per chain and bounded by qc.max_failed_chain_fraction
        return [_unprocessed(r, "failed", f"parse_error: {type(exc).__name__}: {exc}") for _, _, r in requests]
    results = []
    for chain, chain_id, record in requests:
        if chain_id not in chains:
            results.append(_unprocessed(record, "failed", f"chain_missing: {chain_id} not in file (has {sorted(chains)[:20]})"))
            continue
        structure = chains[chain_id]
        if not structure.residues:
            results.append(_unprocessed(record, "failed", "no_standard_residues: the chain has no standard amino-acid residue"))
            continue
        ss = {name: record[name] for name in M.LABEL_SETS}
        mapping = M.map_labels(structure.sequence, record["primary"], ss,
                               [r.source_resname != r.resname for r in structure.residues])
        fraction = mapping.aligned_fraction(len(record["primary"]))
        report = {"structure_residues": len(structure.residues), "identity": round(mapping.identity, 6),
                  "aligned_fraction": round(fraction, 6), "structure_residue_reasons": dict(Counter(mapping.structure_reasons))}
        if mapping.identity < settings["min_identity"] or fraction < settings["min_aligned_fraction"]:
            results.append(_unprocessed(record, "dropped", f"low_alignment_identity: identity {mapping.identity:.3f} (min "
                                        f"{settings['min_identity']}), aligned fraction {fraction:.3f} (min "
                                        f"{settings['min_aligned_fraction']})", report))
            continue
        try:
            outcome = apply_policy(structure, policy)
        except MissingAtomError as exc:
            results.append(_unprocessed(record, "strict_violation", f"strict_violation: {exc}", report))
            continue
        if outcome.dropped_reason:
            results.append(_unprocessed(record, "dropped", f"policy_drop: {outcome.dropped_reason}", report))
            continue
        residues, masked, positions = outcome.residues, outcome.masked, outcome.sequence_index
        kept = np.zeros(len(structure.residues), dtype=bool)
        kept[positions] = True
        labels = {name: mapping.labels[name][positions] for name in M.LABEL_SETS}
        labelled = labels["ss3"] >= 0
        if len(residues) < settings["min_residues"]:
            results.append(_unprocessed(record, "dropped", f"too_short: {len(residues)} residues < min_residues", report))
            continue
        if not labelled.any():
            results.append(_unprocessed(record, "dropped", "no_labelled_residue: every kept residue lacks a label", report))
            continue
        buckets = M.dataset_residue_buckets(len(record["primary"]), record["observed"], mapping, kept,
                                            structure.sequence, record["primary"])
        n_eval = buckets[M.EVALUATED]
        if n_eval != int(labelled.sum()):
            raise ValueError(f"{chain}: {n_eval} evaluated dataset residues but {int(labelled.sum())} labelled structure residues")
        evaluated = np.zeros(len(record["primary"]), dtype=bool)
        evaluated[mapping.pair[np.array(positions)[labelled]]] = True
        spans = windows(len(residues), settings["max_residues"])
        try:
            samples = [compact_sample(build_node_sample(residues[a:b], masked[a:b], chain if len(spans) == 1 else f"{chain}#w{i}",
                                                        sequence_index=positions[a:b])) for i, (a, b) in enumerate(spans)]
        except Exception as exc:  # recorded per chain and bounded by qc.max_failed_chain_fraction
            results.append(_unprocessed(record, "failed", f"graph_error: {type(exc).__name__}: {exc}", report))
            continue
        report.update({"dataset_residues": len(record["primary"]), "buckets": M.account_residues(chain, len(record["primary"]), buckets),
                       "lost_labels": _lost_counts(record, evaluated), "masked_residues": int(sum(masked)),
                       "windows": len(spans), "policy": outcome.report.get("policy"),
                       "residues_missing_backbone": outcome.report.get("residues_missing_backbone")})
        sequence = "".join(structure.sequence[p] for p in positions)
        results.append(ChainResult(chain, "ok", "", report, samples, [b - a for a, b in spans],
                                   {name: [labels[name][a:b] for a, b in spans] for name in M.LABEL_SETS}, sequence))
    return results


def split_coverage(results: Sequence[ChainResult], listed: int, expected_residues: int) -> dict:
    """Coverage of one split against the literature set (chains, residues, every loss by reason)."""
    kept = [r for r in results if r.status == "ok"]
    buckets = Counter()
    lost = {name: np.zeros(LABEL_COUNTS[name], dtype=np.int64) for name in M.LABEL_SETS}
    for r in results:
        buckets.update(r.report["buckets"])
        for name in M.LABEL_SETS:
            lost[name] += np.array(r.report["lost_labels"][name], dtype=np.int64)
    total = int(sum(buckets.values()))
    if total != expected_residues:
        raise ValueError(f"residue buckets sum to {total}, the split has {expected_residues} literature residues")
    evaluated = int(buckets[M.EVALUATED])
    return {
        "literature_chains": listed, "evaluated_chains": len(kept), "chain_coverage": len(kept) / listed,
        "literature_residues": total, "evaluated_residues": evaluated, "residue_coverage": evaluated / total,
        "residue_buckets": {reason: int(buckets.get(reason, 0)) for reason in M.RESIDUE_REASONS},
        "lost_label_counts": {name: lost[name].tolist() for name in lost},
        "masked_residues": int(sum(r.report.get("masked_residues", 0) for r in kept)),
    }


def build_processed(config: dict, *, splits: Sequence[str], workers: int, max_failed_fraction: Optional[float] = None) -> int:
    """Process the requested splits; returns 0, or 1 when a split exceeds the failed-chain limit (file not written)."""
    policy = parse_policy(config["missing_atoms"])
    paths = task_paths(config)
    out_dir = processed_dir(config)
    limit = float(config["qc"]["max_failed_chain_fraction"] if max_failed_fraction is None else max_failed_fraction)
    settings = {"missing_atoms": config["missing_atoms"], "mse_to_met": bool(config["structure"]["mse_to_met"]),
                "max_residues": int(config["max_residues"]), "min_residues": int(config["min_residues"]),
                "min_identity": float(config["alignment"]["min_identity"]),
                "min_aligned_fraction": float(config["alignment"]["min_aligned_fraction"])}
    schema = edge_free_schema(int(config["graph"]["sequence_neighbors_k"]), {
        "builder": "downstream.tasks.ss.dataset", "source": config["source"], "policy": policy.tag})
    exit_code = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if split not in config["splits"]:
            raise ValueError(f"unknown split {split!r}; choose from {config['splits']}")
        target = out_dir / f"{split}.pt"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        records = read_records(paths["records"] / f"{split}.jsonl.gz")
        if [r["id"] for r in records] != read_split_ids(config, split):
            raise ValueError(f"{split}: splits/{split}.txt and records/{split}.jsonl.gz list different chains")
        by_file: Dict[str, list] = {}
        unparsable: List[ChainResult] = []
        for record in records:
            parsed = parse_chain_id(record["id"])
            if parsed is None:
                unparsable.append(_unprocessed(record, "failed", f"id_unparsable: {record['id']!r} is not <PDB id>-<chain>"))
                continue
            pdb_id, chain_id = parsed
            by_file.setdefault(str(paths["structures"] / f"{pdb_id}.cif.gz"), []).append((record["id"], chain_id, record))
        jobs = [(path, requests, settings) for path, requests in sorted(by_file.items())]
        results: Dict[str, ChainResult] = {r.chain: r for r in unparsable}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, batch in enumerate(pool.map(process_entry, jobs, chunksize=4), start=1):
                for result in batch:
                    results[result.chain] = result
                if done % 500 == 0:
                    print(f"[{split}] {done}/{len(jobs)} structure files", flush=True)
        ordered = [results[r["id"]] for r in records]
        status = Counter(r.status for r in ordered)
        bad = len(ordered) - status["ok"]
        fraction = bad / max(len(ordered), 1)
        coverage = split_coverage(ordered, len(ordered), int(sum(len(r["primary"]) for r in records)))
        summary = {"split": split, "listed_chains": len(ordered), "kept_chains": status["ok"], "status": dict(status),
                   "failed_or_dropped_fraction": round(fraction, 6), "limit": limit,
                   "reasons": dict(Counter(r.reason.split(":")[0] for r in ordered if r.status != "ok")),
                   "coverage": coverage, "created_utc": datetime.now(timezone.utc).isoformat()}
        with open(out_dir / f"qc_{split}.tsv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QC_COLUMNS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for r in ordered:
                writer.writerow({"chain": r.chain, "split": split, "status": r.status, "reason": r.reason,
                                 **{k: v for k, v in r.report.items() if k != "buckets"}, **r.report["buckets"]})
        with open(out_dir / f"qc_{split}.json", "w") as handle:
            json.dump(summary, handle, indent=1)
        print(json.dumps(summary, indent=1), flush=True)
        if status["strict_violation"] or fraction > limit:
            print(f"[{split}] QC FAILED: {bad}/{len(ordered)} chains not usable (limit {limit}); {target} not written; "
                  f"see qc_{split}.tsv for the reasons, rerun with --max-failed-fraction to accept the loss", flush=True)
            exit_code = 1
            continue
        kept = [r for r in ordered if r.status == "ok"]
        payload = {
            "schema": schema, "policy": policy.tag, "split": split, "label_sets": {k: list(v) for k, v in M.LABEL_SETS.items()},
            "chains": [r.chain for r in kept], "sequences": [r.sequence for r in kept],
            "samples": [s for r in kept for s in r.samples],
            "sample_chain": torch.tensor([i for i, r in enumerate(kept) for _ in r.samples]),
            "sample_residues": torch.tensor([n for r in kept for n in r.window_residues]),
            "sample_labels": {name: [torch.from_numpy(w.astype(np.int64)) for r in kept for w in r.window_labels[name]]
                              for name in M.LABEL_SETS},
            "qc": summary,
        }
        partial = target.with_name(target.name + ".partial")
        torch.save(payload, partial)
        os.replace(partial, target)
        print(f"[{split}] wrote {target} ({len(kept)} of {len(ordered)} chains, {coverage['evaluated_residues']} of "
              f"{coverage['literature_residues']} residues)", flush=True)
    return exit_code


# ---------------------------------------------------------------------------
# Processed split access
# ---------------------------------------------------------------------------

class SSProcessedSplit:
    """Samples (windows) of one processed split with the residue labels of one label set.

    ``labels`` / ``row_chains`` are the labelled residues in sample order, the row order of the embeddings and of the
    fine-tuning evaluation; ``sample_labels[i]`` has one entry per residue of window ``i`` (-1 = unlabelled, still part of
    the graph).
    """

    def __init__(self, path: Path, label_set: str):
        if label_set not in LABEL_COUNTS:
            raise ValueError(f"unknown label set {label_set!r}")
        self.path, self.label_set = Path(path), label_set
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        for key in ("schema", "chains", "samples", "sample_chain", "sample_residues", "sample_labels", "sequences", "qc"):
            if key not in payload:
                raise ValueError(f"{self.path}: processed split lacks {key!r}")
        if sorted(payload["sample_labels"]) != sorted(LABEL_COUNTS):
            raise ValueError(f"{self.path}: label sets {sorted(payload['sample_labels'])}")
        self.payload, self.schema = payload, payload["schema"]
        self.chains: List[str] = payload["chains"]
        self.sequences: List[str] = payload["sequences"]
        self.sample_chain = payload["sample_chain"].long()
        self.sample_residues = payload["sample_residues"].long()
        self.sample_labels: List[torch.Tensor] = payload["sample_labels"][label_set]
        if not (len(self.sample_labels) == len(payload["samples"]) == len(self.sample_chain) == len(self.sample_residues)):
            raise ValueError(f"{self.path}: inconsistent sample counts")
        for index, (labels, count) in enumerate(zip(self.sample_labels, self.sample_residues.tolist())):
            if labels.numel() != count:
                raise ValueError(f"{self.path}: sample {index} has {labels.numel()} labels for {count} residues")
        if len(self.chains) != int(self.sample_chain.max()) + 1 or len(self.sequences) != len(self.chains):
            raise ValueError(f"{self.path}: inconsistent chain counts")
        other = payload["sample_labels"]["ss8" if label_set == "ss3" else "ss3"]
        if any(not torch.equal(a >= 0, b >= 0) for a, b in zip(self.sample_labels, other)):
            raise ValueError(f"{self.path}: ss3 and ss8 label masks differ")
        self.classes = list(M.LABEL_SETS[label_set])
        labelled = [(labels[labels >= 0], int(chain)) for labels, chain in zip(self.sample_labels, self.sample_chain)]
        self.labels = (torch.cat([l for l, _ in labelled]).numpy() if labelled else np.zeros(0, dtype=np.int64))
        self.row_chains = np.array([self.chains[c] for l, c in labelled for _ in range(len(l))])
        self.coverage = payload["qc"]["coverage"]
        if self.coverage["evaluated_residues"] != len(self.labels) or self.coverage["evaluated_chains"] != len(self.chains):
            raise ValueError(f"{self.path}: coverage record disagrees with the stored labels")

    def __len__(self) -> int:
        return len(self.payload["samples"])

    def runtime(self, index: int, model_config, *, seed: int):
        return runtime_sample(self.payload["samples"][index], self.schema, model_config, seed=seed)

    def iter_runtime(self, model_config, *, seed: int) -> Iterator:
        for index in range(len(self)):
            yield index, self.runtime(index, model_config, seed=seed)


def augment_test_metrics(metrics_test: Dict[str, dict], coverage_by_split: Dict[str, dict], label_set: str) -> None:
    """Add coverage to each test block (in place) so a subset result can never be read as the full literature set.

    ``accuracy`` and the other metrics are computed over the evaluated residues only. ``accuracy_all_residues_coil_fallback``
    counts every other literature residue as predicted coil (an assumption, stated in the name): it has the literature's
    denominator, and the coil fallback is right exactly for the lost residues labelled coil.
    """
    for split, block in metrics_test.items():
        coverage = coverage_by_split[split]
        if int(block["num_proteins"]) != coverage["evaluated_residues"]:
            raise ValueError(f"{split}: {block['num_proteins']} scored residues but coverage lists {coverage['evaluated_residues']}")
        block["num_residues"] = block.pop("num_proteins")
        coil_lost = coverage["lost_label_counts"][label_set][M.COIL_INDEX[label_set]]
        block.update({
            "residue_coverage": coverage["residue_coverage"], "chain_coverage": coverage["chain_coverage"],
            "literature_residues": coverage["literature_residues"], "literature_chains": coverage["literature_chains"],
            "evaluated_chains": coverage["evaluated_chains"],
            "accuracy_all_residues_coil_fallback":
                (block["accuracy"] * block["num_residues"] + coil_lost) / coverage["literature_residues"],
        })


def coverage_text(coverage_by_split: Dict[str, dict]) -> str:
    """One line per split for logs: marks every split below full coverage as a SUBSET."""
    lines = []
    for split, c in coverage_by_split.items():
        mark = "full literature set" if c["evaluated_residues"] == c["literature_residues"] else "SUBSET of the literature set"
        lines.append(f"{split}: {c['evaluated_residues']}/{c['literature_residues']} residues "
                     f"({c['residue_coverage']:.4f}), {c['evaluated_chains']}/{c['literature_chains']} chains: {mark}")
    return "; ".join(lines)
