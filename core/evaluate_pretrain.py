#!/usr/bin/env python3
"""Offline evaluation of a frozen pretraining checkpoint.

    python -m core.evaluate_pretrain --ckpt RUN/last.ckpt --split test --seed 0 \
        --output-dir /path/to/new_dir [--config-yaml-path core/<config>.yaml] \
        [--domains protein,rna] [--max-batches N] [--device cuda:0] \
        [--counterfactual MODE [--counterfactual-sigma ANGSTROM]]
        [--unimol-geometry-hiding checkpoint|on|off [--unimol-bond-threshold ANGSTROM]]

The checkpoint is only read (``torch.load`` with ``map_location='cpu'``); its
sha256, size and mtime are recorded and re-checked at the end. Outputs go to
``--output-dir``, which must not exist and may not lie inside the
checkpoint's directory.

Model architecture and graph-construction settings always come from the
checkpoint. ``--config-yaml-path`` (default: the checkpoint's stored config)
supplies the evaluation data selection (``multi_domain``, ``max_atoms_per_batch``,
the split ``seed``) and the masking protocol (mask ratios, block-MLM flags,
COCONUT walk masking); differing values of other keys are listed, not applied.

Determinism (what is guaranteed):

* The batch plan depends only on the data selection, the split and
  ``max_atoms_per_batch``: batches are domain-pure, samples keep their
  chunk/index order and are packed greedily up to ``max_atoms_per_batch``.
* Every graph carries a seed derived from ``--seed`` and its stable key
  ``<chunk dir>/<chunk file>#<index>``. The MLM mask (block or atom level,
  COCONUT walks) and the runtime random protein Tier-3 edges of the graph
  are drawn from that seed alone, so they do not depend on the checkpoint,
  the batch the graph lands in, or the other graphs of that batch.
* The global RNG is reseeded before each batch from (seed, domain, batch
  ordinal within the domain); the long-range pair sample of a batch (which
  pools the pairs of all its graphs) is therefore reproducible and equal
  across checkpoints, but it depends on the batch composition.
* The embedding-health sample depends on (seed, batch position).

``mask_manifest.csv`` lists every graph's mask digest; ``metrics.json`` holds
the per-domain digest, so two evaluations can be checked for identical masks.

Counterfactual modes (core/counterfactual_eval.py): ``--counterfactual MODE``
corrupts, at evaluation time only, what the encoder/ESA see of the masked UniMol atoms
(position, distances, ESA edges) while the targets, the mask and the distance-loss frame
stay clean. protein/RNA/COCONUT graphs are untouched. The default ``none`` leaves the
evaluation unchanged; ``metrics.json`` gets a ``counterfactual`` block either way.

UniMol masked-geometry hiding (``unimol_masked_geometry_hiding``) is a model setting, so it
comes from the checkpoint by default (a checkpoint without the key: disabled). ``--unimol-geometry-hiding
on --unimol-bond-threshold A`` / ``off`` override it for this evaluation only, so that checkpoints trained
with and without hiding can be measured under one test condition. The effective setting is recorded in
``provenance.unimol_masked_geometry_hiding``; a YAML value is never applied (it is listed as not applied).

A real forward needs a GPU (flash_varlen). The CPU path exists for tests,
which replace the flash kernel with an SDPA stand-in.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

from core.counterfactual_eval import (
    COUNTERFACTUAL_MODES,
    DEFAULT_NOISE_SIGMA,
    MODE_REVISIONS,
    build_counterfactual_settings,
    validate_mode,
)
from core.deterministic import enable_deterministic_algorithms
from core.metrics.domain_mlm_metrics import DOMAIN_ORDER, canonical_domain
from core.metrics.domain_mlm_metrics_callback import DomainMLMMetricsCallback
from core.metrics.protein_per_atom_metrics_callback import ProteinPerAtomMetricsCallback
from core.metrics.rna_per_atom_metrics_callback import PerAtomMetricsCallback
from core.metrics.virtual_slot_diagnostics_callback import VirtualSlotDiagnosticsCallback
from core.pretraining_config import (
    create_pretraining_config,
    load_config_yaml,
    normalize_unimol_masked_geometry_hiding_config,
    validate_unimol_bond_threshold,
)
from core.pretraining_model import PretrainingESAModel
from data_loading.lazy_universal_dataset import LazyUniversalDataset
from data_loading.pt_schema import edge_free_runtime_contract_from_config

# Keys the evaluation YAML may set; everything else comes from the checkpoint.
DATA_SELECTION_KEYS = ("multi_domain", "max_atoms_per_batch", "seed")
MASKING_KEYS = (
    "mlm_mask_ratio",
    "protein_block_mlm",
    "protein_block_mlm_ratio",
    "rna_block_mlm",
    "rna_block_mlm_ratio",
    "protein_block_mlm_exclude_legacy_oxt",
    "coconut_mask_ratio",
    "coconut_masking_method",
    "coconut_num_random_walks",
)
CONFIG_DOMAIN_KEYS = {"pdb": "protein", "metabolite": "coconut"}
ABSENT_FROM_CHECKPOINT = "<absent from checkpoint>"
HIDING_KEY = "unimol_masked_geometry_hiding"
HIDING_MODES = ("checkpoint", "on", "off")
EMBEDDING_NODES_PER_BATCH = 256
EMBEDDING_MAX_NODES_PER_DOMAIN = 4096


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def stable_seed(*parts) -> int:
    """63-bit seed from a SHA-256 of the parts (stable across processes and runs)."""
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def git_provenance(repo: Path) -> dict:
    def run(*args):
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}")
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "repo": str(repo),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_files": status.splitlines(),
    }


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def checkpoint_config_dict(checkpoint: dict) -> dict:
    hyper = checkpoint.get("hyper_parameters")
    if not hyper:
        raise ValueError("checkpoint has no hyper_parameters; cannot rebuild the model")
    if "config" in hyper:
        return dict(hyper["config"])
    # Legacy flat checkpoints: the model constructor normalizes them.
    return dict(vars(PretrainingESAModel(**hyper).config))


def merge_eval_config(model_config: dict, yaml_config: Optional[dict]) -> tuple:
    """Return (eval config dict, ignored yaml differences)."""
    merged = dict(model_config)
    ignored = {}
    if yaml_config is None:
        return merged, ignored
    for key, value in yaml_config.items():
        if key in DATA_SELECTION_KEYS or key in MASKING_KEYS:
            merged[key] = value
        elif model_config.get(key, ABSENT_FROM_CHECKPOINT) != value:
            # also a key the checkpoint does not have (an older run): never dropped silently
            ignored[key] = {"checkpoint": model_config.get(key, ABSENT_FROM_CHECKPOINT), "yaml": value}
    return merged, ignored


def resolve_geometry_hiding(model_config: dict, mode: str, threshold: Optional[float]) -> tuple:
    """Return (model config for this evaluation, provenance record) of the UniMol geometry hiding.

    ``checkpoint`` keeps the checkpoint's setting (absent = disabled); ``on``/``off`` override it for the
    evaluation only. A disabled setting is recorded without a threshold, so equal conditions compare equal.
    """
    if mode not in HIDING_MODES:
        raise ValueError(f"unimol geometry hiding mode must be one of {HIDING_MODES}, got {mode!r}")
    stored = normalize_unimol_masked_geometry_hiding_config(model_config.get(HIDING_KEY))
    if mode == "checkpoint":
        if threshold is not None:
            raise ValueError("--unimol-bond-threshold applies to --unimol-geometry-hiding on only")
        effective = dict(stored)
    elif mode == "on":
        if threshold is None:
            raise ValueError("--unimol-geometry-hiding on needs an explicit --unimol-bond-threshold")
        effective = normalize_unimol_masked_geometry_hiding_config(
            {"enabled": True, "bond_threshold_angstrom": threshold})
    else:
        if threshold is not None:
            raise ValueError("--unimol-bond-threshold applies to --unimol-geometry-hiding on only")
        effective = normalize_unimol_masked_geometry_hiding_config(None)
    if not effective["enabled"]:
        effective = normalize_unimol_masked_geometry_hiding_config(None)
    config = dict(model_config)
    config[HIDING_KEY] = dict(effective)
    return config, {"source": mode, "checkpoint": stored, "effective": effective}


def describe_geometry_hiding(record: dict) -> str:
    def text(value):
        return (f"ENABLED (bond_threshold_angstrom={value['bond_threshold_angstrom']})"
                if value["enabled"] else "DISABLED")
    return (f"UniMol masked-geometry hiding: {text(record['effective'])}; source={record['source']} "
            f"(checkpoint: {text(record['checkpoint'])})")


def resolve_domains(config, requested: Optional[str]) -> List[str]:
    multi_domain = getattr(config, "multi_domain", None) or {}
    if not multi_domain.get("enabled", False):
        raise ValueError("evaluate_pretrain needs a multi_domain config; single-dataset configs are not supported")
    enabled = []
    for key, settings in multi_domain.get("datasets", {}).items():
        if settings.get("enabled", False):
            enabled.append(canonical_domain(CONFIG_DOMAIN_KEYS.get(key, key)))
    if not enabled:
        raise ValueError("no multi_domain dataset is enabled in the evaluation config")
    if requested is None:
        return [d for d in DOMAIN_ORDER if d in enabled]
    wanted = [canonical_domain(name.strip()) for name in requested.split(",") if name.strip()]
    missing = [d for d in wanted if d not in enabled]
    if missing:
        raise ValueError(f"--domains {missing} are not enabled in the evaluation config (enabled: {enabled})")
    return [d for d in DOMAIN_ORDER if d in wanted]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _metadata_search_paths() -> List[str]:
    base = REPO_ROOT / "metadata"
    return [str(sub) for sub in base.iterdir() if sub.is_dir()] if base.exists() else []


def load_split_dataset(config, split: str, domains: Sequence[str]):
    """Return (LazyUniversalDataset, global indices of the split for ``domains``)."""
    from core.train_pretrain import (
        METADATA_DOMAIN_NAMES,
        _discover_chunk_files,
        create_pretraining_data_transforms,
        load_universal_dataset,
        split_dataset,
    )

    datasets = config.multi_domain["datasets"]
    selected = {
        key: settings for key, settings in datasets.items()
        if settings.get("enabled", False) and canonical_domain(CONFIG_DOMAIN_KEYS.get(key, key)) in domains
    }
    if all(settings.get("splits") is not None for settings in selected.values()):
        files, limits, policies = [], {}, {}
        for key, settings in selected.items():
            split_config = settings["splits"].get(split)
            if split_config is None:
                raise ValueError(f"{key}: no '{split}' entry in its fixed splits")
            found = _discover_chunk_files(f"{key}/{split}", split_config.get("chunk_dir", ""),
                                          split_config.get("chunk_range"))
            if not found:
                raise FileNotFoundError(f"{key}/{split}: no chunks found")
            files.extend(found)
            meta_name = METADATA_DOMAIN_NAMES.get(key, key)
            if settings.get("max_graph_atoms") is not None:
                limits[meta_name] = int(settings["max_graph_atoms"])
            if settings.get("oversized_graphs") is not None:
                policies[meta_name] = settings["oversized_graphs"]
        dataset = LazyUniversalDataset(
            chunk_pt_files=files,
            transform=create_pretraining_data_transforms(config),
            max_cache_chunks=int(config.multi_domain.get("max_cache_chunks", 10)),
            verbose=True,
            load_metadata=True,
            metadata_search_paths=_metadata_search_paths() or None,
            max_graph_atoms_by_domain=limits,
            oversized_graph_policy_by_domain=policies,
            edge_free_runtime=edge_free_runtime_contract_from_config(config),
            chunk_split_tags=[split] * len(files),
        )
        indices = [i for i in range(len(dataset)) if i not in dataset.excluded_indices]
        source = "fixed split chunks"
    else:
        # Random-split domains: membership depends on the whole pooled corpus,
        # so the training loader and split are reproduced exactly.
        print("ℹ️  evaluate_pretrain: a selected domain has no fixed splits; "
              "rebuilding the full training dataset and its seeded random split")
        dataset = load_universal_dataset(config, config.dataset, config.dataset_download_dir)
        subsets = dict(zip(("train", "val", "test"), split_dataset(dataset, config.seed)))
        indices = list(subsets[split].indices)
        source = "training split_dataset (random-split domains present)"
    if not getattr(dataset, "metadata_loaded", False):
        raise ValueError("evaluation planning needs chunk metadata (per-sample atom counts)")
    return dataset, indices, source


def sample_entries(dataset: LazyUniversalDataset, indices: Sequence[int], domains: Sequence[str]) -> Dict[str, list]:
    """Per domain, the split's samples in chunk/index order: (global idx, key, atoms)."""
    entries = defaultdict(list)
    for index in sorted(indices):
        meta = dataset.sample_metadata_index[index]
        domain = canonical_domain(meta["dataset_type"])
        if domain not in domains:
            continue
        pt_file = Path(dataset.chunk_pt_files[meta["chunk_idx"]])
        key = f"{pt_file.parent.parent.name}/{pt_file.name}#{int(meta['local_idx'])}"
        entries[domain].append((int(index), key, int(meta["num_atoms"])))
    return entries


def plan_batches(entries: Dict[str, list], max_atoms: int, max_batches: Optional[int]) -> tuple:
    """Domain-pure greedy packing in sample order; returns (plan, per-domain stats)."""
    if max_atoms < 1:
        raise ValueError("max_atoms_per_batch must be at least 1")
    plan, stats = [], {}
    position = 0
    positions = {}
    for domain in DOMAIN_ORDER:
        if domain not in entries:
            continue
        batches, current, atoms, oversized = [], [], 0, 0
        for entry in entries[domain]:
            if current and atoms + entry[2] > max_atoms:
                batches.append(current)
                current, atoms = [], 0
            current.append(entry)
            atoms += entry[2]
            if entry[2] > max_atoms:
                oversized += 1
        if current:
            batches.append(current)
        planned = len(batches)
        if max_batches is not None:
            batches = batches[:max_batches]
        for ordinal, batch in enumerate(batches):
            rows = []
            for entry in batch:
                positions[position] = entry
                rows.append(position)
                position += 1
            plan.append({"domain": domain, "ordinal": ordinal, "positions": rows,
                         "atoms": sum(e[2] for e in batch)})
        stats[domain] = {
            "samples_in_split": len(entries[domain]),
            "planned_batches": planned,
            "evaluated_batches": len(batches),
            "evaluated_graphs": sum(len(b) for b in batches),
            "evaluated_atoms": sum(e[2] for b in batches for e in b),
            "single_graph_batches_over_atom_budget": oversized,
        }
    return plan, stats, positions


class EvalSampleDataset(Dataset):
    """Plan positions -> samples with per-graph seeds (mask and random edges)."""

    def __init__(self, base: LazyUniversalDataset, positions: Dict[int, tuple], seed: int, domains: Dict[int, str]):
        self.base = base
        self.positions = positions
        self.seed = int(seed)
        self.domains = domains

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, position):
        global_index, key, _ = self.positions[position]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stable_seed(self.seed, "runtime_edges", key))
            sample = self.base[global_index]
        domain = canonical_domain(sample.dataset_type)
        if domain != self.domains[position]:
            raise ValueError(f"{key}: dataset_type {sample.dataset_type!r} does not match planned domain")
        sample.eval_mask_seed = torch.tensor([stable_seed(self.seed, "mlm_mask", key)], dtype=torch.long)
        sample.eval_sample_key = key
        return sample


# ---------------------------------------------------------------------------
# Run bookkeeping callback
# ---------------------------------------------------------------------------


class EvalRunCallback(Callback):
    """Per-batch RNG seeding, per-domain losses, mask manifest, slot-diagnostic snapshot."""

    def __init__(self, plan: list, seed: int, virtual_slot_callback: VirtualSlotDiagnosticsCallback):
        super().__init__()
        self.plan = plan
        self.seed = int(seed)
        self.virtual_slot_callback = virtual_slot_callback
        self.loss_sums = defaultdict(Counter)
        self.loss_batches = Counter()
        self.mask_rows = []
        self.virtual_slot_snapshot = None
        self.batches_seen = 0

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        entry = self.plan[batch_idx]
        torch.manual_seed(stable_seed(self.seed, "batch_rng", entry["domain"], entry["ordinal"]))
        domains = {canonical_domain(d) for d in (batch.dataset_type if isinstance(batch.dataset_type, (list, tuple))
                                                  else [batch.dataset_type])}
        if domains != {entry["domain"]}:
            raise ValueError(f"batch {batch_idx}: domains {sorted(domains)} != planned {entry['domain']}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        entry = self.plan[batch_idx]
        domain = entry["domain"]
        self.batches_seen += 1
        self.loss_batches[domain] += 1
        self.loss_sums[domain]["total"] += float(outputs["loss"])
        for name, value in outputs["task_losses"].items():
            self.loss_sums[domain][name] += float(value)

        graph = batch.batch.detach().cpu()
        mask = batch.mlm_mask.detach().bool().cpu()
        keys = list(batch.eval_sample_key)
        counts = torch.bincount(graph, minlength=len(keys))
        starts = torch.cumsum(counts, 0) - counts
        for g, key in enumerate(keys):
            local = torch.where(mask[starts[g]:starts[g] + counts[g]])[0].to(torch.int32).numpy()
            self.mask_rows.append({
                "domain": domain,
                "key": key,
                "n_nodes": int(counts[g]),
                "n_masked": int(local.size),
                "mask_sha1": hashlib.sha1(local.tobytes()).hexdigest(),
            })

    def on_validation_epoch_end(self, trainer, pl_module):
        # Runs before VirtualSlotDiagnosticsCallback prints and resets its state.
        cb = self.virtual_slot_callback
        if not cb.enabled:
            self.virtual_slot_snapshot = {"enabled": False}
            return
        snapshot = {"enabled": True}
        for domain in ("protein", "rna"):
            summary = VirtualSlotDiagnosticsCallback.summarize_domain(cb._slots[domain])
            if summary["rows"]:
                snapshot[domain] = summary
        snapshot["context_audit"] = {
            name: {"mean": cb._audit_weighted_sum[name] / max(cb._audit_support[name], 1),
                   "support": cb._audit_support[name]}
            for name in sorted(cb._audit_support)
        }
        self.virtual_slot_snapshot = snapshot

    def per_domain_losses(self) -> dict:
        return {
            domain: {"n_batches": self.loss_batches[domain],
                     **{name: total / self.loss_batches[domain] for name, total in sorted(sums.items())}}
            for domain, sums in self.loss_sums.items()
        }

    def mask_digests(self) -> dict:
        digests = {}
        for domain in sorted({row["domain"] for row in self.mask_rows}):
            lines = sorted(f"{r['key']}\t{r['mask_sha1']}" for r in self.mask_rows if r["domain"] == domain)
            digests[domain] = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        return digests


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        fields.extend(k for k in row if k not in fields)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(out: Path, metrics: dict, run_cb: EvalRunCallback) -> None:
    tables = out / "tables"
    tables.mkdir()
    for domain, result in metrics["domain_mlm"].items():
        write_csv(tables / f"{domain}_summary.csv", [{"metric": k, "value": v} for k, v in result["summary"].items()])
        write_csv(tables / f"{domain}_elements.csv", result["elements"])
        write_csv(tables / f"{domain}_calibration.csv", result["calibration"])
        confusion = result["confusion"]
        write_csv(tables / f"{domain}_confusion.csv", [
            {"target": label, **dict(zip(confusion["labels"], row))}
            for label, row in zip(confusion["labels"], confusion["matrix"])
        ])
        identity = result.get("block_identity")
        if identity is not None:
            write_csv(tables / f"{domain}_identity_per_name.csv",
                      [{"name": n, **row} for n, row in identity["per_name"].items()])
            write_csv(tables / f"{domain}_identity_per_class.csv",
                      [{"class": n, **row} for n, row in identity["per_class"].items()])
            write_csv(tables / f"{domain}_identity_confusion.csv", identity["class_confusion"])
        distance = result.get("distance_visible_pairs_diagnostic")
        if distance:
            write_csv(tables / f"{domain}_distance_visible_pairs_diagnostic.csv",
                      [{"head": head, **row} for head, row in distance.items()])
        if result.get("embedding"):
            write_csv(tables / f"{domain}_embedding.csv", [result["embedding"]])
    write_csv(tables / "losses_per_domain.csv",
              [{"domain": d, **row} for d, row in metrics["losses"]["per_domain_batch_means"].items()])
    for name, snapshot in (("protein", metrics["protein_per_atom"]), ("rna", metrics["rna_per_atom"])):
        if snapshot:
            write_csv(tables / f"{name}_per_atom.csv",
                      [{"atom": atom, **row} for atom, row in snapshot["per_atom"].items()])
    for domain in ("protein", "rna"):
        rows = (metrics["virtual_slot"] or {}).get(domain, {}).get("rows")
        if rows:
            write_csv(tables / f"{domain}_virtual_slot_prior.csv", rows)
    write_csv(out / "mask_manifest.csv", run_cb.mask_rows)


def counterfactual_summary(args, settings, domains, metrics_domain_mlm: dict, losses: dict) -> dict:
    """The ``counterfactual`` block of metrics.json (also written for mode ``none``)."""
    block = {
        "mode": args.counterfactual,
        "sigma": None if settings is None else settings.sigma,
        "domains": [] if settings is None else list(settings.domains),
        "applied": {} if settings is None else dict(sorted(settings.stats.items())),
        "note": "inputs of masked UniMol atoms are corrupted for the encoder/ESA only; targets, mask and "
                "distance-loss targets are clean, so the distance losses below see corrupted inputs",
    }
    if args.counterfactual in MODE_REVISIONS:
        block["mode_revision"] = MODE_REVISIONS[args.counterfactual]
    unimol = (metrics_domain_mlm or {}).get("unimol")
    if unimol is not None:
        summary = unimol["summary"]
        block["unimol"] = {
            "n_masked": summary["n_masked"],
            "top1_acc_real": summary["top1_acc_real"],
            "top3_acc_real": summary["top3_acc_real"],
            "ce_real": summary["ce_real"],
            "ece_real": summary["ece_real"],
            "element_recall": {row["element"]: row["recall"] for row in unimol["elements"] if row["support"] > 0},
            "element_support": {row["element"]: row["support"] for row in unimol["elements"] if row["support"] > 0},
            "losses": {k: v for k, v in losses.get("unimol", {}).items() if k != "n_batches"},
        }
    return block


def format_loss_table(losses: dict) -> str:
    lines = ["PER-DOMAIN LOSSES (mean over the domain's batches; batches are domain-pure)"]
    for domain, row in losses.items():
        items = " | ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "n_batches")
        lines.append(f"  {domain:<8} batches={row['n_batches']:>5}  {items}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def _counterfactual_mode(value: str) -> str:
    try:
        return validate_mode(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Offline evaluation of a frozen MIND pretraining checkpoint")
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint (read only)")
    parser.add_argument("--config-yaml-path", default=None,
                        help="YAML for the data selection/masking protocol (default: the checkpoint's config)")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--seed", type=int, required=True, help="evaluation mask/RNG seed")
    parser.add_argument("--domains", default=None, help="comma list (protein,rna,unimol,coconut); default: all enabled")
    parser.add_argument("--max-batches", type=int, default=None, help="cap on batches per domain")
    parser.add_argument("--max-atoms-per-batch", type=int, default=None, help="override the config atom budget")
    parser.add_argument("--num-workers", type=int, default=None, help="loader workers (default: config num_workers)")
    parser.add_argument("--output-dir", required=True, help="new directory (must not exist)")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:<i> or cpu (cpu only for tests)")
    parser.add_argument("--counterfactual", type=_counterfactual_mode, default="none", metavar="MODE",
                        help="evaluation-only corruption of masked UniMol atoms: "
                             + ", ".join(COUNTERFACTUAL_MODES) + " (default none = unchanged evaluation)")
    parser.add_argument("--counterfactual-sigma", type=float, default=None,
                        help=f"Angstrom, only for masked_pos_noise (default {DEFAULT_NOISE_SIGMA} for that mode)")
    parser.add_argument("--unimol-geometry-hiding", choices=HIDING_MODES, default="checkpoint",
                        help="UniMol masked-geometry hiding for this evaluation: checkpoint (default, the trained "
                             "setting; absent = off), on (needs --unimol-bond-threshold) or off")
    parser.add_argument("--unimol-bond-threshold", type=float, default=None,
                        help="Angstrom, only with --unimol-geometry-hiding on (no default)")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for torch; the machine is shared with training runs")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be at least 1")
    if args.counterfactual == "masked_pos_noise" and args.counterfactual_sigma is None:
        args.counterfactual_sigma = DEFAULT_NOISE_SIGMA
    try:
        build_counterfactual_settings(args.counterfactual, args.counterfactual_sigma)
    except ValueError as error:
        parser.error(str(error))
    if args.unimol_geometry_hiding == "on" and args.unimol_bond_threshold is None:
        parser.error("--unimol-geometry-hiding on needs an explicit --unimol-bond-threshold")
    if args.unimol_geometry_hiding != "on" and args.unimol_bond_threshold is not None:
        parser.error("--unimol-bond-threshold applies to --unimol-geometry-hiding on only")
    if args.unimol_bond_threshold is not None:
        try:
            validate_unimol_bond_threshold(args.unimol_bond_threshold, "--unimol-bond-threshold")
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    return args


def _prepare_paths(args) -> tuple:
    ckpt = Path(args.ckpt).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint does not exist or is not a file: {ckpt}")
    out = Path(args.output_dir).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to write into an existing output directory: {out}")
    if out == ckpt.parent or ckpt.parent in out.parents:
        raise ValueError(f"refusing to write next to the checkpoint: {out} is inside {ckpt.parent}")
    return ckpt, out


def _trainer_device(device: str) -> tuple:
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device {device} requested but CUDA is not available")
        return "gpu", [device.index if device.index is not None else 0]
    if device.type == "cpu":
        return "cpu", 1
    raise ValueError(f"unsupported device: {device}")


def evaluate(args) -> dict:
    started = time.time()
    ckpt, out = _prepare_paths(args)
    torch.set_num_threads(args.threads)
    accelerator, devices = _trainer_device(args.device)
    if accelerator == "gpu":
        enable_deterministic_algorithms()
    identity_before = file_identity(ckpt)
    ckpt_sha256 = sha256_file(ckpt)
    out.mkdir(parents=True)
    log_handle = open(out / "eval_log.txt", "w")
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log_handle)), warnings.catch_warnings():
            # The trainer has no logger on purpose (results go to files), so
            # Lightning's per-key "no logger configured" warning is expected.
            warnings.filterwarnings("ignore", message=r".*but have no logger configured.*")
            metrics = _evaluate(args, ckpt, out, accelerator, devices, ckpt_sha256)
    finally:
        log_handle.close()
    if file_identity(ckpt) != identity_before:
        raise RuntimeError(f"checkpoint changed during evaluation: {ckpt}")
    metrics["provenance"]["checkpoint"]["unchanged_after_run"] = True
    metrics["provenance"]["wall_seconds"] = time.time() - started
    with open(out / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2, default=_json_default)
    with open(out / "provenance.json", "w") as handle:
        json.dump(metrics["provenance"], handle, indent=2, default=_json_default)
    return metrics


def _evaluate(args, ckpt: Path, out: Path, accelerator, devices, ckpt_sha256: str) -> dict:
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_config = checkpoint_config_dict(checkpoint)
    yaml_config = load_config_yaml(args.config_yaml_path) if args.config_yaml_path else None
    eval_dict, ignored = merge_eval_config(model_config, yaml_config)
    if args.max_atoms_per_batch is not None:
        eval_dict["max_atoms_per_batch"] = int(args.max_atoms_per_batch)
    if args.num_workers is not None:
        eval_dict["num_workers"] = int(args.num_workers)
    eval_dict["use_wandb"] = False
    eval_config = create_pretraining_config(**eval_dict)
    if ignored:
        print(f"ℹ️  YAML keys that differ from the checkpoint and are NOT applied: {sorted(ignored)}")
    masking_changes = {k: (model_config.get(k), eval_dict.get(k)) for k in MASKING_KEYS
                       if model_config.get(k) != eval_dict.get(k)}
    if masking_changes:
        print(f"ℹ️  Masking protocol differs from the checkpoint's training config: {masking_changes}")

    model_eval_config, hiding_record = resolve_geometry_hiding(
        model_config, args.unimol_geometry_hiding, args.unimol_bond_threshold)
    print(describe_geometry_hiding(hiding_record))
    model = PretrainingESAModel(config=model_eval_config)
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    for key in MASKING_KEYS:
        if key in eval_dict:
            setattr(model.config, key, eval_dict[key])
    model.config.use_wandb = False
    model.config.seed = int(args.seed)
    model.eval()
    if "mlm" not in model.config.pretraining_tasks:
        raise ValueError("checkpoint was not trained with the MLM task; nothing to evaluate")

    domains = resolve_domains(eval_config, args.domains)
    counterfactual = build_counterfactual_settings(args.counterfactual, args.counterfactual_sigma)
    if counterfactual is not None:
        missing = [d for d in counterfactual.domains if d not in domains]
        if missing:
            raise ValueError(
                f"--counterfactual {args.counterfactual} corrupts {list(counterfactual.domains)} graphs but "
                f"the evaluated domains are {domains}; it would change nothing"
            )
        model.counterfactual_eval = counterfactual
    max_atoms = int(eval_dict.get("max_atoms_per_batch") or 0)
    dataset, indices, source = load_split_dataset(eval_config, args.split, domains)
    entries = sample_entries(dataset, indices, domains)
    empty = [d for d in domains if not entries.get(d)]
    if empty:
        raise ValueError(f"split '{args.split}' has no sample for domains {empty}")
    plan, plan_stats, positions = plan_batches(entries, max_atoms, args.max_batches)
    position_domains = {p: entry["domain"] for entry in plan for p in entry["positions"]}
    eval_dataset = EvalSampleDataset(dataset, positions, args.seed, position_domains)
    num_workers = int(getattr(eval_config, "num_workers", 0) or 0)
    loader = GeometricDataLoader(
        eval_dataset,
        batch_sampler=[entry["positions"] for entry in plan],
        num_workers=num_workers,
        pin_memory=accelerator == "gpu",
    )
    print(f"📋 Evaluation plan ({args.split}, max_atoms_per_batch={max_atoms}): "
          + ", ".join(f"{d}={s['evaluated_batches']} batches/{s['evaluated_graphs']} graphs" for d, s in plan_stats.items()))

    virtual_slot_cb = VirtualSlotDiagnosticsCallback(enabled=bool(
        (getattr(model.config, "geometry_neutral_virtual_slots", None) or {}).get("enabled", False)))
    run_cb = EvalRunCallback(plan, args.seed, virtual_slot_cb)
    domain_cb = DomainMLMMetricsCallback(
        enabled=True, log_every_n_epochs=1, print_every_n_epochs=1, distance_diagnostics=True,
        embedding_sample_nodes_per_batch=EMBEDDING_NODES_PER_BATCH,
        embedding_max_nodes_per_domain=EMBEDDING_MAX_NODES_PER_DOMAIN,
    )
    protein_cb = ProteinPerAtomMetricsCallback(enabled=True, only_protein=True, compute_groups=True,
                                               compute_per_aa=True, compute_per_aa_block=True, log_every_n_epochs=1)
    rna_cb = PerAtomMetricsCallback(enabled=True, only_rna=True, compute_groups=True, log_every_n_epochs=1)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision="bf16-mixed" if model.config.use_bfloat16 else "32-true",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[run_cb, domain_cb, protein_cb, rna_cb, virtual_slot_cb],
        default_root_dir=str(out),
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
    )
    logged = trainer.validate(model, dataloaders=loader, verbose=False)[0]
    if run_cb.batches_seen != len(plan):
        raise RuntimeError(f"evaluated {run_cb.batches_seen} batches, planned {len(plan)}")
    if counterfactual is not None:
        expected = sum(1 for entry in plan if entry["domain"] in counterfactual.domains)
        if counterfactual.stats["batches"] != expected or counterfactual.stats["masked_atoms"] == 0:
            raise RuntimeError(
                f"counterfactual {args.counterfactual} was applied to {counterfactual.stats['batches']} batches "
                f"and {counterfactual.stats['masked_atoms']} masked atoms; expected {expected} batches with masks"
            )

    per_domain_losses = run_cb.per_domain_losses()
    counterfactual_block = counterfactual_summary(
        args, counterfactual, domains, metrics_domain_mlm=domain_cb.last_results, losses=per_domain_losses)
    metrics = {
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "arguments": vars(args),
            "checkpoint": {
                "path": str(ckpt),
                "sha256": ckpt_sha256,
                "epoch": checkpoint.get("epoch"),
                "global_step": checkpoint.get("global_step"),
                "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
            },
            "git": git_provenance(REPO_ROOT),
            "split": args.split,
            "seed": int(args.seed),
            "domains": domains,
            "data_source": source,
            "config_yaml_path": str(Path(args.config_yaml_path).resolve()) if args.config_yaml_path else None,
            "yaml_keys_not_applied": ignored,
            "counterfactual": {k: counterfactual_block[k] for k in ("mode", "sigma", "domains", "mode_revision")
                               if k in counterfactual_block},
            HIDING_KEY: hiding_record,
            "masking_protocol": {k: eval_dict.get(k) for k in MASKING_KEYS},
            "masking_differs_from_checkpoint": masking_changes,
            "max_atoms_per_batch": max_atoms,
            "num_workers": num_workers,
            "device": args.device,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "precision": trainer.precision,
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "pytorch_lightning": pl.__version__},
            "model_config": model_config,
            "eval_config": eval_dict,
        },
        "plan": plan_stats,
        "counterfactual": counterfactual_block,
        "losses": {
            "per_domain_batch_means": per_domain_losses,
            "lightning_logged": {k: float(v) for k, v in logged.items()},
        },
        "domain_mlm": domain_cb.last_results,
        "protein_per_atom": protein_cb._last_snapshot,
        "rna_per_atom": rna_cb._last_val_metrics_snapshot,
        "virtual_slot": run_cb.virtual_slot_snapshot,
        "mask_digest": run_cb.mask_digests(),
    }
    write_tables(out, metrics, run_cb)
    text = "\n\n".join([domain_cb.last_text or "", format_loss_table(per_domain_losses)])
    (out / "tables.txt").write_text(text + "\n")
    print(text)
    return metrics


def main(argv=None):
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    args = parse_args(argv)
    metrics = evaluate(args)
    print(f"✅ Evaluation written to {Path(args.output_dir).resolve()} "
          f"(mask digests: {metrics['mask_digest']})")


if __name__ == "__main__":
    main()
