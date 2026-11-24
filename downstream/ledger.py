"""Append-only results ledger: every downstream / evaluation number, one JSON line per seed and metric.

    python -m downstream.ledger add --outputs-root DIR [--eval-dir DIR ...]
    python -m downstream.ledger annotate --select "task=ec data_revision=<rev>" --status superseded --reason "..."
    python -m downstream.ledger verify
    python -m downstream.ledger recompute-macro-aupr --task ec --model <m> --control <m2> --data-revision <rev> [--dry-run]

The ledger lives outside Git under ``$MIND_RESULTS_ROOT/ledger/`` (no default: an unset variable is an
error, so a ledger is never opened in a surprising place; the Makefile passes the project's root):

``measurements.jsonl``  one row = one model x one run seed x one (eval split, subset, metric)
``models.jsonl``        one row per snapshot card (checkpoint sha256, epoch, selection rule, training run)
``annotations.jsonl``   status notes (superseded / invalid / caveat / smoke / valid) selected by row fields
``ingests.jsonl``       one record per ``add`` run (files seen, rows added, embedding-only sets)
``MANIFEST.json``       line count and sha256 of every file; any edit outside this module stops all writes

Rows are never changed or deleted. A changed status is a new annotation; a recomputed number is a new row
(reports show the newest and footnote the older one). Writes take an exclusive ``flock`` and append with
one ``write`` call; the manifest is checked before and updated after each append.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import yaml

from downstream.common import split_replicates

SCHEMA_VERSION = 1
ENV_VAR = "MIND_RESULTS_ROOT"
CATALOG_PATH = Path(__file__).with_name("ledger_catalog.yaml")
LEDGER_FILES = ("measurements", "models", "annotations", "ingests")
STATUSES = ("valid", "superseded", "invalid", "caveat", "smoke")
DOWNSTREAM_KINDS = ("multilabel", "multiclass", "regression", "multilabel_masked", "multitarget_regression",
                    "ranking_regression")
PROBE_HEADS = {"linear_probe": "linear", "mlp_probe": "mlp", "sklearn_tuned": "sklearn_tuned"}
# Models without a MIND checkpoint card: a Hugging Face baseline and the fingerprint baseline (downstream.baselines.ecfp).
UNTRAINED_KINDS = ("hf_baseline", "fingerprint_baseline")
EVAL_METRICS = ("top1_acc_real", "top3_acc_real", "ce_real", "ece_real")
_TOP_KEYS = {"task", "kind", "model_id", "protocol", "pooling", "seed", "best_epoch", "split_tag", "representation",
             "val", "test", "target_standardization"}
# Count / operating-point fields: kept as row context, never as metric rows.
_CONTEXT_KEYS = {"num_proteins", "num_samples", "molecules", "num_classes_present", "macro_aupr_num_terms",
                 "macro_aupr_num_terms_skipped", "target_mean", "target_std", "num_tasks", "num_tasks_evaluated",
                 "num_tasks_skipped", "num_targets", "fmax_threshold", "num_puzzles", "num_residues", "literature_residues",
                 "literature_chains", "evaluated_chains"}
_SAMPLE_KEYS = ("num_proteins", "num_samples", "molecules", "num_residues")
_LIST_KEYS = {"per_task", "per_task_prc_auc", "skipped_task_indices", "rmse_per_target", "mae_per_target",
              "num_labeled_per_target", "puzzles", "per_puzzle_num_samples", "per_puzzle_spearman_rho",
              "per_puzzle_pearson_r", "per_puzzle_top1_loss"}
# Metrics a block of each kind always has; an undefined block (empty subset, single-class subset) gets one
# row per metric with the reason, so the report can say UNDEFINED instead of leaving a hole.
_KIND_METRICS = {"multilabel": ("fmax", "pair_aupr", "top1_accuracy", "macro_aupr"),
                 "multiclass": ("accuracy", "balanced_accuracy"),
                 "regression": ("rmse", "mae", "pearson_r", "spearman_rho"),
                 "multilabel_masked": ("roc_auc", "prc_auc"), "multitarget_regression": ("rmse", "mae"),
                 "ranking_regression": ("spearman_rho", "pearson_r")}


class LedgerError(RuntimeError):
    """The ledger is missing, inconsistent or was edited outside this module."""


def results_root(root: Optional[Path] = None) -> Path:
    if root is not None:
        return Path(root)
    value = os.environ.get(ENV_VAR)
    if not value:
        raise LedgerError(f"{ENV_VAR} is not set; pass the results root explicitly (the Makefile sets it)")
    return Path(value)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True).encode())


# ---------------------------------------------------------------------------------------------------------
# Catalog (required metrics per task; checked into the repository, holds no numbers)
# ---------------------------------------------------------------------------------------------------------

def load_catalog(path: Path = CATALOG_PATH) -> dict:
    path = Path(path)
    catalog = yaml.safe_load(path.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported catalog schema_version {catalog.get('schema_version')!r}")
    metrics, tasks = catalog["metrics"], catalog["tasks"]
    families = {spec.get("family") for spec in tasks.values()}
    for model, entry in (catalog.get("fingerprint_baselines") or {}).items():
        unknown = sorted(set(entry.get("families") or ()) - families)
        if entry.get("protocol") is not None and entry["protocol"] not in PROBE_HEADS:
            raise ValueError(f"{path}: fingerprint_baselines.{model}.protocol must be one of {sorted(PROBE_HEADS)}")
        if not entry.get("families") or unknown or not entry.get("table_pooling") or "{task}" not in str(entry.get("command")):
            raise ValueError(f"{path}: fingerprint_baselines.{model} needs families (task families of the catalog"
                             f"{', unknown: ' + str(unknown) if unknown else ''}), table_pooling and a command "
                             "containing {task}")
    for name, info in metrics.items():
        if info.get("direction") not in ("higher", "lower") or not info.get("unit"):
            raise ValueError(f"{path}: metric {name!r} needs direction higher|lower and a unit")
    for task, spec in tasks.items():
        for key in ("family", "kind", "required_protocols", "required"):
            if key not in spec:
                raise ValueError(f"{path}: task {task!r} lacks {key!r}")
        if not isinstance(spec.get("split_replicates", False), bool):
            raise ValueError(f"{path}: task {task!r}: split_replicates must be true or false")
        if spec.get("split_replicates") and any("." in protocol for protocol in spec["required_protocols"]):
            raise ValueError(f"{path}: task {task!r} has per-seed splits; its protocols carry no split tag "
                             "(the tag follows from the seed)")
        cells = list(spec["required"]) + ([spec["primary"]] if spec.get("primary") else [])
        for cell in cells:
            if cell["metric"] not in metrics:
                raise ValueError(f"{path}: task {task!r} names unknown metric {cell['metric']!r}")
            if metrics[cell["metric"]]["unit"] == "task" and not spec.get("target_unit"):
                raise ValueError(f"{path}: task {task!r} uses {cell['metric']!r} (unit: task) without target_unit")
            if cell.get("code", "present") not in ("present", "missing"):
                raise ValueError(f"{path}: task {task!r} cell {cell} has code other than present|missing")
    catalog["sha256"] = sha256_path(path)
    return catalog


def metric_info(catalog: dict, task: str, metric: str) -> dict:
    """Direction and unit of ``metric`` for ``task``; metrics outside the catalog have neither."""
    info = catalog["metrics"].get(metric)
    if info is None:
        return {"direction": None, "unit": None, "is_catalog_metric": False}
    unit = info["unit"]
    if unit == "task":
        unit = catalog["tasks"].get(task, {}).get("target_unit")
        if unit is None:
            raise ValueError(f"catalog: metric {metric!r} takes the task unit but task {task!r} has no target_unit")
    return {"direction": info["direction"], "unit": unit, "is_catalog_metric": True}


def task_spec(catalog: dict, task: str) -> dict:
    if task not in catalog["tasks"]:
        raise ValueError(f"task {task!r} is not in the ledger catalog {CATALOG_PATH}")
    return catalog["tasks"][task]


# ---------------------------------------------------------------------------------------------------------
# Append-only storage
# ---------------------------------------------------------------------------------------------------------

def ledger_dir(root: Path) -> Path:
    return Path(root) / "ledger"


@contextmanager
def _locked(root: Path, *, exclusive: bool = True):
    directory = ledger_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _manifest(root: Path) -> dict:
    path = ledger_dir(root) / "MANIFEST.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _checked_bytes(root: Path, name: str, manifest: dict) -> bytes:
    """Content of ``<name>.jsonl`` after checking it against the manifest (line count and sha256)."""
    if name not in LEDGER_FILES:
        raise LedgerError(f"unknown ledger file {name!r}; choose from {LEDGER_FILES}")
    path = ledger_dir(root) / f"{name}.jsonl"
    data = path.read_bytes() if path.exists() else b""
    entry = manifest.get(name, {"n_lines": 0, "sha256": sha256_bytes(b"")})
    if data and not data.endswith(b"\n"):
        raise LedgerError(f"{path}: last line is incomplete (interrupted write?); inspect the file by hand")
    n_lines = data.count(b"\n")
    if n_lines != entry["n_lines"]:
        raise LedgerError(f"{path}: has {n_lines} lines but MANIFEST.json records {entry['n_lines']}; lines were "
                          "added or removed outside downstream.ledger (or a write was interrupted)")
    if sha256_bytes(data) != entry["sha256"]:
        raise LedgerError(f"{path}: content differs from MANIFEST.json; recorded history was edited")
    return data


def _parse(root: Path, name: str, data: bytes) -> list:
    records = []
    for number, line in enumerate(data.splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise LedgerError(f"{ledger_dir(root) / (name + '.jsonl')} line {number}: not JSON ({exc})") from exc
    return records


def _read_locked(root: Path, name: str) -> list:
    return _parse(root, name, _checked_bytes(root, name, _manifest(root)))


def _append_locked(root: Path, name: str, records: list) -> None:
    if not records:
        return
    manifest = _manifest(root)
    data = _checked_bytes(root, name, manifest)
    payload = b"".join(json.dumps(record, sort_keys=True, allow_nan=False).encode() + b"\n" for record in records)
    path = ledger_dir(root) / f"{name}.jsonl"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    manifest[name] = {"n_lines": data.count(b"\n") + len(records), "sha256": sha256_bytes(data + payload)}
    partial = ledger_dir(root) / "MANIFEST.json.partial"
    partial.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(partial, ledger_dir(root) / "MANIFEST.json")


def append_records(root: Path, name: str, records: list) -> None:
    with _locked(root):
        _append_locked(root, name, records)


def read_records(root: Path, name: str) -> list:
    """Records of one ledger file, checked against the manifest; an absent ledger reads as empty."""
    if not ledger_dir(root).exists():
        return []
    with _locked(root, exclusive=False):
        return _read_locked(root, name)


def _check_identities(measurements: Iterable[dict], models: Iterable[dict]) -> None:
    """One model key names one checkpoint (or one HF revision) everywhere in the ledger."""
    seen = {}
    for row in measurements:
        identity = (row["model_kind"] in ("mind", "mind_random_init"), row["model_sha256"], row.get("hf_revision"))
        previous = seen.setdefault(row["model_key"], (identity, row.get("source_metrics_path")))
        if previous[0] != identity:
            raise LedgerError(
                f"model {row['model_key']!r}: sha256/revision {previous[0][1:]} from {previous[1]} but "
                f"{identity[1:]} from {row.get('source_metrics_path')}; one model name must name one checkpoint")
    cards = {}
    for model in models:
        if cards.setdefault(model["model_key"], model["sha256"]) != model["sha256"]:
            raise LedgerError(f"models.jsonl: model {model['model_key']!r} has two sha256 values")
        if model["model_key"] in seen and seen[model["model_key"]][0][1] != model["sha256"]:
            raise LedgerError(f"model {model['model_key']!r}: card sha256 {model['sha256']} differs from the "
                              f"measurement sha256 {seen[model['model_key']][0][1]}")


def verify(root: Optional[Path] = None) -> dict:
    """Check every file against the manifest, the row schema, unique row ids and model identities."""
    root = results_root(root)
    if not (ledger_dir(root) / "MANIFEST.json").exists():
        raise LedgerError(f"no ledger at {ledger_dir(root)} (MANIFEST.json missing)")
    with _locked(root, exclusive=False):
        records = {name: _read_locked(root, name) for name in LEDGER_FILES}
    ids = [row.get("row_id") for row in records["measurements"]]
    if len(set(ids)) != len(ids) or None in ids:
        raise LedgerError("measurements.jsonl: row_id missing or repeated")
    for row in records["measurements"]:
        if row.get("schema_version") != SCHEMA_VERSION:
            raise LedgerError(f"measurements.jsonl row {row.get('row_id')}: schema_version {row.get('schema_version')}")
        if row["value"] is None and not row.get("undefined_reason"):
            raise LedgerError(f"measurements.jsonl row {row['row_id']}: null value without undefined_reason")
    for note in records["annotations"]:
        if note.get("status") not in STATUSES:
            raise LedgerError(f"annotations.jsonl: status {note.get('status')!r} is not one of {STATUSES}")
    _check_identities(records["measurements"], records["models"])
    return {"root": str(root), "lines": {name: len(rows) for name, rows in records.items()},
            "measurements_sha256": _manifest(root).get("measurements", {}).get("sha256")}


# ---------------------------------------------------------------------------------------------------------
# Flattening of one result file into rows
# ---------------------------------------------------------------------------------------------------------

def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _block_rows(block, kind: str, where: str) -> tuple[list, Optional[int], dict]:
    """``[(metric, value, undefined_reason)]``, sample count and context of one metric block."""
    if block is None:
        return [(metric, None, "empty subset (no evaluated sample)") for metric in _KIND_METRICS[kind]], 0, {}
    if not isinstance(block, dict):
        raise ValueError(f"{where}: expected a metric dict, got {type(block).__name__}")
    whole_reason = block.get("undefined")
    correlation_reason = block.get("correlation_undefined_reason")
    for key in ("undefined", "correlation_undefined_reason"):
        if block.get(key) is not None and not isinstance(block[key], str):
            raise ValueError(f"{where}: {key!r} must be a string reason")
    rows, context = [], {}
    for key, value in block.items():
        if key in ("undefined", "correlation_undefined_reason"):
            continue
        if key in _LIST_KEYS:
            if not isinstance(value, list):
                raise ValueError(f"{where}: field {key!r} should be a list")
            continue
        if key in _CONTEXT_KEYS or (key.startswith("ic_binned_macro_aupr") and key.endswith("_num_terms")):
            context[key] = value
            continue
        if value is not None and not _is_number(value):
            raise ValueError(f"{where}: field {key!r} has unexpected type {type(value).__name__}; "
                             "extend downstream.ledger before ingesting this format")
        if value is None or math.isnan(value):
            reason = correlation_reason if key in ("pearson_r", "spearman_rho") else None
            if reason is None:
                raise ValueError(f"{key} is {value} without an undefined reason in {where}")
            rows.append((key, None, reason))
        else:
            rows.append((key, float(value), None))
    if whole_reason:
        present = {metric for metric, _, _ in rows}
        rows += [(metric, None, whole_reason) for metric in _KIND_METRICS[kind] if metric not in present]
    samples = next((context[key] for key in _SAMPLE_KEYS if key in context), None)
    return rows, samples, context


def _eval_blocks(metrics: dict, kind: str, where: str) -> list:
    test = metrics["test"]
    if not isinstance(test, dict):
        raise ValueError(f"{where}: 'test' must be a dict")
    if kind == "ranking_regression":  # both splits hold one block per subset: mean_per_puzzle and global
        if not isinstance(metrics["val"], dict):
            raise ValueError(f"{where}: 'val' must be a dict")
        return ([("val", name, block) for name, block in metrics["val"].items()]
                + [("test", name, block) for name, block in test.items()])
    blocks = [("val", "all", metrics["val"])]
    if kind == "multiclass":
        blocks += [(name, "all", block) for name, block in test.items()]
    elif kind in ("multilabel_masked", "multitarget_regression"):
        unknown = set(test) - {"all", "subsets"}
        if unknown:
            raise ValueError(f"{where}: unexpected test keys {sorted(unknown)}")
        blocks.append(("test", "all", test["all"]))
        blocks += [("test", name, block) for name, block in sorted(test.get("subsets", {}).items())]
    else:
        blocks += [("test", name, block) for name, block in test.items()]
    return blocks


def _split_name(key: str) -> str:
    """Embedding records of LBA are keyed ``identity30.joint.test``; the split is the last part."""
    return key.rsplit(".", 1)[-1]


def selection_rule(card: dict) -> Optional[str]:
    """How the snapshot's checkpoint was chosen, read from its card (never assumed)."""
    monitor = (card.get("val_metrics") or {}).get("monitor")
    best = (card.get("val_metrics") or {}).get("best_model_path")
    if not monitor:
        return None
    score = (card.get("val_metrics") or {}).get("best_model_score")
    if best and best == card.get("source_path"):
        return f"best-checkpoint by {monitor} (ModelCheckpoint best_model_path, score {score})"
    return (f"not the ModelCheckpoint best: source {Path(str(card.get('source_path'))).name}, "
            f"best by {monitor} was {Path(str(best)).name}")


def _train_seed(card: dict) -> Optional[int]:
    for run in (card.get("wandb_runs") or {}).values():
        args = run.get("args") or []
        if "--seed" in args and args.index("--seed") + 1 < len(args):
            return int(args[args.index("--seed") + 1])
    return None


def _model_record(card_path: Path, card: dict) -> dict:
    return {
        "model_key": card["name"], "sha256": card["sha256"], "epoch": card.get("epoch"),
        "global_step": card.get("global_step"), "source_path": card.get("source_path"),
        "source_mtime_ns": card.get("source_mtime_ns"), "train_git_commits": card.get("training_git_commits"),
        "train_args": next(iter((card.get("wandb_runs") or {}).values()), {}).get("args"),
        "train_seed": _train_seed(card), "train_config_path": card.get("config_path"),
        "train_config_sha256": card.get("config_sha256"), "domains": card.get("domains"),
        "created_utc": card.get("created_utc"), "snapshot_repo_commit": card.get("snapshot_repo_commit"),
        "selection_rule": selection_rule(card), "card_path": str(card_path), "card_sha256": sha256_path(card_path),
    }


def _load_card(outputs_root: Optional[Path], snapshot: str, sha: str) -> tuple[Optional[dict], Optional[Path]]:
    if outputs_root is None:
        return None, None
    path = Path(outputs_root) / "models" / snapshot / "card.json"
    if not path.is_file():
        return None, None
    card = json.loads(path.read_text())
    if card.get("sha256") != sha:
        raise LedgerError(f"{path}: card sha256 {card.get('sha256')} differs from the result's checkpoint sha256 {sha}")
    return card, path


def _card_by_sha(outputs_root: Optional[Path], sha: str, hint: Path) -> tuple[Optional[dict], Optional[Path]]:
    candidates = [hint.parent / "card.json"]
    if outputs_root is not None:
        candidates += sorted((Path(outputs_root) / "models").glob("*/card.json"))
    for path in candidates:
        if path.is_file():
            card = json.loads(path.read_text())
            if card.get("sha256") == sha:
                return card, path
    return None, None


def _model_fields(card: Optional[dict], card_path: Optional[Path], *, untrained: bool, missing: list) -> dict:
    """Training fields copied from the card so a row is understandable alone (none for a random init or HF model)."""
    fields = {"train_run_dir": None, "train_git_commits": None, "train_config_path": None,
              "train_config_sha256": None, "train_seed": None, "train_epoch": None, "train_global_step": None,
              "selection_rule": None}
    if untrained:
        return fields
    if card is None:
        missing.append("model_card")
        return fields
    record = _model_record(card_path, card)
    fields.update(train_run_dir=str(Path(str(card.get("source_path"))).parent),
                  train_git_commits=record["train_git_commits"], train_config_path=record["train_config_path"],
                  train_config_sha256=record["train_config_sha256"], train_seed=record["train_seed"],
                  train_epoch=record["epoch"], train_global_step=record["global_step"],
                  selection_rule=record["selection_rule"])
    for key in ("train_config_sha256", "train_seed", "selection_rule"):
        if fields[key] is None:
            missing.append(key)
    return fields


def _probe_provenance(config: dict, metrics: dict, where: str, missing: list) -> dict:
    records = config["embeddings"]
    if not isinstance(records, dict) or not records:
        raise ValueError(f"{where}: config.json 'embeddings' must be a non-empty dict")
    model_ids = {record.get("model_id") for record in records.values()}
    if model_ids != {metrics["model_id"]}:
        raise ValueError(f"{where}: embedding records name models {sorted(map(str, model_ids))}, "
                         f"metrics.json names {metrics['model_id']!r}")
    hf = {record.get("hf_revision") for record in records.values()}
    shas = {record.get("sha256") for record in records.values()}
    if len(hf) != 1 or len(shas) != 1:
        raise ValueError(f"{where}: embedding splits come from different checkpoints {sorted(map(str, shas | hf))}")
    first = next(iter(records.values()))
    scaling = config.get("args", {}).get("feature_scaling", "standardize")  # older probes had no such option
    fingerprints = {json.dumps(record.get("fingerprint"), sort_keys=True) for record in records.values()}
    if len(fingerprints) != 1:
        raise ValueError(f"{where}: embedding splits disagree on the fingerprint definition")
    settings = config.get("probe")
    run = config.get("run") or {}
    if not run.get("git"):
        missing.append("run_git")
    head = metrics["protocol"].split(".")[0]
    seeds = config.get("args", {}).get("seeds") or (settings or {}).get("seeds")
    fields = {"protocol_family": "frozen_probe", "head": PROBE_HEADS.get(head), "settings": settings,
              "code_git_embed": sorted({str(record.get("git")) for record in records.values()}),
              "code_git_run": run.get("git"), "run_started_utc": run.get("started_utc"), "expected_seeds": seeds,
              "processed": {_split_name(key): (record.get("processed"), record.get("processed_sha256"))
                            for key, record in records.items()}}
    if "fingerprint" in first:  # downstream.baselines.ecfp: the feature is a fingerprint, there is no checkpoint
        if "hf_revision" in first or "sha256" in first:
            raise ValueError(f"{where}: an embedding record cannot be both a fingerprint and a model embedding")
        if scaling != "none":
            raise ValueError(f"{where}: a fingerprint probe must run with feature_scaling 'none', got {scaling!r}")
        fields.update(model_kind="fingerprint_baseline", model_sha256=None, hf_revision=None, hf_model=None,
                      snapshot=None, init_seed=None, input_feature={**first["fingerprint"], "feature_scaling": scaling})
    elif scaling != "standardize":
        raise ValueError(f"{where}: feature_scaling {scaling!r} is only valid for a fingerprint baseline")
    elif "hf_revision" in first:
        fields.update(model_kind="hf_baseline", model_sha256=None, hf_revision=first["hf_revision"],
                      hf_model=first.get("model"), snapshot=None, init_seed=None)
    else:
        random_init = {bool(record.get("random_init")) for record in records.values()}
        if len(random_init) != 1:
            raise ValueError(f"{where}: embedding splits disagree on random_init")
        random_init = random_init.pop()
        fields.update(model_kind="mind_random_init" if random_init else "mind", model_sha256=first.get("sha256"),
                      hf_revision=None, hf_model=None, snapshot=first.get("snapshot"),
                      init_seed=first.get("init_seed") if random_init else None)
    return fields


def _finetune_provenance(config: dict, metrics: dict, where: str, missing: list) -> dict:
    settings = config.get("finetune")
    random_init = str(config.get("args", {}).get("random_init")) == "True"
    seeds = config.get("args", {}).get("seeds")
    if isinstance(seeds, str) and seeds != "None":
        seeds = json.loads(seeds)
    elif seeds in (None, "None"):
        seeds = (settings or {}).get("seeds")
    processed = config.get("processed")
    if processed is None:
        missing.append("data_revision")
        missing.append("eval_processed_sha256")
    return {"protocol_family": "finetune", "head": (settings or {}).get("head"), "settings": settings,
            "code_git_embed": None, "code_git_run": config.get("git"),
            "run_started_utc": (config.get("run") or {}).get("started_utc"), "expected_seeds": seeds,
            "processed": {_split_name(name): (entry["path"], entry["sha256"]) for name, entry in (processed or {}).items()},
            "model_kind": "mind_random_init" if random_init else "mind", "model_sha256": config["sha256"],
            "hf_revision": None, "hf_model": None, "snapshot": config["snapshot"],
            # finetune.py initializes a random encoder with init_seed=seed
            "init_seed": int(metrics["seed"]) if random_init else None}


def _processed_for(processed: dict, eval_split: str, where: str):
    if not processed:
        return None, None
    names = ("valid", "val") if eval_split == "val" else (eval_split,)
    for name in names:
        if name in processed:
            return processed[name]
    raise ValueError(f"{where}: no processed-data record for eval split {eval_split!r} (has {sorted(processed)})")


def flatten_run(metrics_path: Path, catalog: dict, outputs_root: Optional[Path]) -> tuple[list, Optional[dict]]:
    """Rows of one ``seed<k>/metrics.json`` (+ its ``config.json``) and the model card record, if any."""
    metrics_path = Path(metrics_path)
    where = str(metrics_path)
    metrics = json.loads(metrics_path.read_text())
    config_path = metrics_path.with_name("config.json")
    config = json.loads(config_path.read_text())
    unknown = set(metrics) - _TOP_KEYS
    if unknown:
        raise ValueError(f"{where}: unexpected top-level keys {sorted(unknown)}")
    task = metrics.get("task")
    spec = task_spec(catalog, task)
    missing: list = []
    if "kind" not in metrics:  # EC probes written before the kind field existed; listed, not hidden
        missing.append("kind")
    kind = metrics.get("kind", spec["kind"])
    if kind not in DOWNSTREAM_KINDS:
        raise ValueError(f"{where}: kind {kind!r} is not one of {DOWNSTREAM_KINDS}")
    if spec["kind"] != kind:
        raise ValueError(f"{where}: task {task!r} has kind {kind!r}, the catalog says {spec['kind']!r}")
    head, *qualifiers = metrics["protocol"].split(".")
    if "embeddings" in config:
        if head not in PROBE_HEADS:
            raise ValueError(f"{where}: probe config with protocol {metrics['protocol']!r}")
        prov = _probe_provenance(config, metrics, where, missing)
    elif "snapshot" in config and "git" in config:
        if head != "finetune":
            raise ValueError(f"{where}: fine-tune config with protocol {metrics['protocol']!r}")
        prov = _finetune_provenance(config, metrics, where, missing)
    else:
        raise ValueError(f"{where}: config.json is neither a probe config (embeddings) nor a fine-tune config")
    if prov["settings"] is None:
        missing.append("protocol_settings")
    random_init = prov["model_kind"] == "mind_random_init"
    card = card_path = None
    if prov["model_kind"] not in UNTRAINED_KINDS:
        card, card_path = _load_card(outputs_root, prov["snapshot"], prov["model_sha256"])
    train = _model_fields(card, card_path, untrained=random_init or prov["model_kind"] in UNTRAINED_KINDS,
                          missing=missing)
    revisions = {Path(path).parent.name for path, _ in prov["processed"].values() if path}
    if len(revisions) > 1:
        raise ValueError(f"{where}: splits come from different processed directories {sorted(revisions)}")
    split_tag = metrics.get("split_tag") or (qualifiers[0] if qualifiers else None)
    representation = metrics.get("representation") or (qualifiers[1] if len(qualifiers) > 1 else None)
    if spec.get("split_replicates"):
        # One split per run seed: the tag is a function of the seed, the protocol carries no tag, and the repeats of a
        # cell are the runs of the different tags, so the expected seeds are those of the protocol settings.
        wanted = split_replicates.tag_for_seed(int(metrics["seed"]))
        if split_tag != wanted or qualifiers or representation is not None:
            raise ValueError(f"{where}: task {task!r} draws one split per seed; seed {metrics['seed']} must carry "
                             f"split_tag {wanted!r} and a protocol without qualifiers, got split_tag {split_tag!r}, "
                             f"protocol {metrics['protocol']!r}")
        settings_seeds = (prov["settings"] or {}).get("seeds")
        if not settings_seeds or int(metrics["seed"]) not in settings_seeds:
            raise ValueError(f"{where}: the protocol settings list seeds {settings_seeds!r}, which must contain seed "
                             f"{metrics['seed']} of this per-seed-split task")
        prov = {**prov, "expected_seeds": list(settings_seeds)}
    base = {
        "schema_version": SCHEMA_VERSION, "model_key": metrics["model_id"], "model_kind": prov["model_kind"],
        "model_sha256": prov["model_sha256"], "hf_revision": prov["hf_revision"], "hf_model": prov["hf_model"],
        "template_model_key": prov["snapshot"] if random_init else None, "init_seed": prov["init_seed"], **train,
        "input_feature": prov.get("input_feature"), "task": task, "task_family": spec["family"], "kind": kind,
        "data_revision": revisions.pop() if revisions else None, "split_tag": split_tag,
        "representation": representation, "protocol_family": prov["protocol_family"], "head": prov["head"],
        "protocol_id": metrics["protocol"], "pooling": metrics["pooling"], "run_seed": int(metrics["seed"]),
        "best_epoch": metrics.get("best_epoch"), "expected_seeds": prov["expected_seeds"],
        "protocol_settings_sha256": sha256_json(prov["settings"]) if prov["settings"] is not None else None,
        "code_git_embed": prov["code_git_embed"], "code_git_run": prov["code_git_run"],
        "run_started_utc": prov["run_started_utc"], "source_metrics_path": str(metrics_path.resolve()),
        "source_metrics_sha256": sha256_path(metrics_path), "source_config_sha256": sha256_path(config_path),
        "outputs_root": str(Path(outputs_root).resolve()) if outputs_root else None, "derived_from": None,
    }
    rows = []
    primary = spec.get("primary") or {}
    for eval_split, subset, block in _eval_blocks(metrics, kind, where):
        values, samples, context = _block_rows(block, kind, f"{where} [{eval_split}/{subset}]")
        path, data_sha = _processed_for(prov["processed"], eval_split, where)
        for metric, value, reason in values:
            rows.append({**base, "eval_split": eval_split, "subset": subset, "metric": metric, "value": value,
                         "undefined_reason": reason, "n_samples": samples, "context": context,
                         "eval_processed_path": path, "eval_processed_sha256": data_sha,
                         **metric_info(catalog, task, metric)})
    for spec_avg in spec.get("derived_averages") or []:
        for metric in spec_avg["metrics"]:
            parts = [next((r for r in rows if r["eval_split"] == name and r["subset"] == "all"
                           and r["metric"] == metric), None) for name in spec_avg["of"]]
            if any(part is None or part["value"] is None for part in parts):
                continue
            rows.append({**parts[0], "eval_split": spec_avg["eval_split"],
                         "value": float(sum(p["value"] for p in parts) / len(parts)), "n_samples": None,
                         "context": {}, "eval_processed_path": None,
                         "eval_processed_sha256": sha256_json([p["eval_processed_sha256"] for p in parts]),
                         "derived_from": f"mean of {', '.join(spec_avg['of'])} (same seed)"})
    for row in rows:
        row["is_primary"] = (row["eval_split"], row["subset"], row["metric"]) == (
            primary.get("eval_split"), primary.get("subset"), primary.get("metric"))
        row["missing_provenance"] = sorted(set(missing))
    model = _model_record(card_path, card) if card is not None and not random_init else None
    return rows, model


def flatten_eval(eval_dir: Path, catalog: dict, outputs_root: Optional[Path]) -> tuple[list, Optional[dict]]:
    """Rows of one ``core/evaluate_pretrain.py`` output directory (plain or counterfactual)."""
    metrics_path = Path(eval_dir) / "metrics.json"
    where = str(metrics_path)
    metrics = json.loads(metrics_path.read_text())
    prov = metrics.get("provenance")
    if not isinstance(prov, dict) or "checkpoint" not in prov or "domain_mlm" not in metrics:
        raise ValueError(f"{where}: not an evaluate_pretrain metrics.json (provenance/checkpoint/domain_mlm missing)")
    counterfactual = prov.get("counterfactual")
    task = "counterfactual" if counterfactual else "pretrain_eval"
    spec = task_spec(catalog, task)
    sha = prov["checkpoint"]["sha256"]
    missing: list = []
    card, card_path = _card_by_sha(outputs_root, sha, Path(prov["checkpoint"]["path"]))
    train = _model_fields(card, card_path, untrained=False, missing=missing)
    git = prov.get("git") or {}
    settings = {"masking_protocol": prov.get("masking_protocol"), "counterfactual": counterfactual,
                "domains": prov.get("domains")}
    hiding = prov.get("unimol_masked_geometry_hiding")
    if hiding is not None:  # recorded since the eval-time override; older files keep their settings hash
        if not isinstance(hiding, dict) or not isinstance(hiding.get("effective"), dict):
            raise ValueError(f"{where}: provenance.unimol_masked_geometry_hiding has no 'effective' mapping")
        settings["unimol_masked_geometry_hiding"] = hiding["effective"]
    base = {
        "schema_version": SCHEMA_VERSION, "model_key": card["name"] if card else f"ckpt_sha256:{sha[:12]}",
        "model_kind": "mind", "model_sha256": sha, "hf_revision": None, "hf_model": None,
        "template_model_key": None, "init_seed": None, **train, "task": task, "task_family": spec["family"],
        "kind": spec["kind"], "data_revision": prov.get("data_source"), "split_tag": None, "representation": None,
        "protocol_family": "evaluation", "head": None,
        "protocol_id": counterfactual["mode"] if counterfactual else "mlm_eval", "pooling": None,
        "run_seed": int(prov["seed"]), "best_epoch": None, "expected_seeds": [int(prov["seed"])],
        "protocol_settings_sha256": sha256_json(settings), "code_git_embed": None,
        "code_git_run": (git.get("commit") + ("+dirty" if git.get("dirty") else "")) if git.get("commit") else None,
        "run_started_utc": None, "source_metrics_path": str(metrics_path.resolve()),
        "source_metrics_sha256": sha256_path(metrics_path), "source_config_sha256": None,
        "outputs_root": str(Path(outputs_root).resolve()) if outputs_root else None, "derived_from": None,
        "eval_split": prov["split"], "eval_processed_path": None, "eval_processed_sha256": None,
    }
    if base["code_git_run"] is None:
        missing.append("run_git")
    rows = []
    for domain, entry in sorted(metrics["domain_mlm"].items()):
        summary = entry.get("summary") if isinstance(entry, dict) else None
        if not isinstance(summary, dict):
            raise ValueError(f"{where}: domain_mlm[{domain!r}] has no summary dict")
        for metric in EVAL_METRICS:
            value = summary.get(metric)
            if not _is_number(value) or math.isnan(value):
                raise ValueError(f"{where}: domain_mlm[{domain!r}].{metric} is {value!r} without an undefined reason")
            rows.append({**base, "subset": domain, "metric": metric, "value": float(value), "undefined_reason": None,
                         "n_samples": summary.get("n_masked_real", summary.get("n_masked")), "context": {},
                         **metric_info(catalog, task, metric)})
    primary = spec.get("primary") or {}
    for row in rows:
        row["is_primary"] = (row["eval_split"], row["subset"], row["metric"]) == (
            primary.get("eval_split"), primary.get("subset"), primary.get("metric"))
        row["missing_provenance"] = sorted(set(missing))
    return rows, (_model_record(card_path, card) if card is not None else None)


def _embedding_sets(outputs_root: Path) -> list:
    sets = {}
    for path in sorted(Path(outputs_root).glob("downstream/*/*/embeddings/*.json")):
        sets.setdefault((path.parents[2].name, path.parents[1].name), set()).add(path.stem)
    return [{"task": task, "model_key": model, "splits": sorted(splits)} for (task, model), splits in sorted(sets.items())]


def ingest(root: Optional[Path] = None, *, outputs_root: Optional[Path] = None, eval_dirs: Iterable[Path] = (),
           command: Optional[str] = None, catalog: Optional[dict] = None) -> dict:
    """Add every new result file under ``outputs_root`` and each evaluation directory; re-running is safe."""
    root = results_root(root)
    catalog = catalog or load_catalog()
    started = now_utc()
    eval_dirs = [Path(d) for d in eval_dirs]
    metric_files = sorted(Path(outputs_root).glob("downstream/*/*/*/seed*/metrics.json")) if outputs_root else []
    if outputs_root is not None and not Path(outputs_root).is_dir():
        raise LedgerError(f"outputs root {outputs_root} does not exist")
    with _locked(root):
        existing = _read_locked(root, "measurements")
        models = _read_locked(root, "models")
        known = {(row["source_metrics_sha256"], row["source_config_sha256"]) for row in existing}
        new_rows, new_models, skipped = [], {}, 0
        sources = [(path, sha256_path(path), sha256_path(path.with_name("config.json")), flatten_run)
                   for path in metric_files]
        sources += [(d / "metrics.json", sha256_path(d / "metrics.json"), None, flatten_eval) for d in eval_dirs]
        for path, metrics_sha, config_sha, flatten in sources:
            if (metrics_sha, config_sha) in known:
                skipped += 1
                continue
            rows, model = flatten(path if flatten is flatten_run else path.parent, catalog, outputs_root)
            new_rows += rows
            known.add((metrics_sha, config_sha))
            if model is not None and model["model_key"] not in {m["model_key"] for m in models}:
                if new_models.setdefault(model["model_key"], model)["sha256"] != model["sha256"]:
                    raise LedgerError(f"model {model['model_key']!r}: two cards with different sha256 in one ingest")
        _check_identities(existing + new_rows, models + list(new_models.values()))
        ingest_id, stamp = str(uuid.uuid4()), now_utc()
        for row in new_rows:
            row.update(row_id=str(uuid.uuid4()), ingest_id=ingest_id, ingested_utc=stamp)
        for model in new_models.values():
            model.update(ingest_id=ingest_id, ingested_utc=stamp)
        record = {"ingest_id": ingest_id, "started_utc": started, "finished_utc": now_utc(),
                  "command": command or " ".join(sys.argv), "code_git": _code_git(),
                  "outputs_root": str(Path(outputs_root).resolve()) if outputs_root else None,
                  "eval_dirs": [str(d.resolve()) for d in eval_dirs], "files_seen": len(sources),
                  "rows_added": len(new_rows), "models_added": sorted(new_models),
                  "files_skipped_same_sha": skipped,
                  "embedding_sets": _embedding_sets(outputs_root) if outputs_root else [],
                  "catalog_sha256": catalog["sha256"]}
        _append_locked(root, "measurements", new_rows)
        _append_locked(root, "models", list(new_models.values()))
        _append_locked(root, "ingests", [record])
    return record


# ---------------------------------------------------------------------------------------------------------
# Recomputation of a missing metric from saved predictions
# ---------------------------------------------------------------------------------------------------------

RECOMPUTE_TOLERANCE = 1e-9


def _recompute_cells(rows: list, setting: dict, model: str, metric: str) -> dict:
    """``{run_seed: row}`` of one model's test/all cell in one frozen-probe setting; two rows per seed is an error."""
    found = {}
    for row in rows:
        if (row["model_key"] == model and row["eval_split"] == "test" and row["subset"] == "all"
                and row["metric"] == metric and row["protocol_family"] == "frozen_probe"
                and all(row[key] == value for key, value in setting.items())):
            if row["run_seed"] in found:
                raise LedgerError(f"{model} seed {row['run_seed']} {metric}: more than one row in setting {setting}; "
                                  "the setting is ambiguous")
            found[row["run_seed"]] = row
    return found


def _saved_test_arrays(row: dict):
    """(predictions.npz path, labels, scores) next to a row's metrics.json, which must be the ingested file."""
    import numpy as np

    path = Path(row["source_metrics_path"])
    if sha256_path(path) != row["source_metrics_sha256"]:
        raise LedgerError(f"{path} changed since it was ingested (sha256 differs); refusing to recompute from it")
    npz = path.with_name("predictions.npz")
    with np.load(npz) as data:
        return npz, data["test_labels"], data["test_scores"]


def recompute_macro_aupr(root: Optional[Path], *, task: str, model: str, control: str, data_revision: str,
                         protocol_id: str = "mlp_probe", pooling: str = "mean_atoms", dry_run: bool = False,
                         command: Optional[str] = None, out=print) -> dict:
    """Add the macro AUPR a frozen-probe run did not record, computed from its saved ``predictions.npz``.

    The probe is not re-run: the value is ``downstream.common.metrics.label_aupr`` (the function probe.py uses)
    on the stored test labels and scores. Only the test/all cell (the catalog's required one); subset masks and
    validation labels are not in predictions.npz. Nothing is written unless all four checks pass:

    1. each metrics.json still has the sha256 recorded in the ledger;
    2. fmax and pair_aupr recomputed from the predictions equal the ledger values (``RECOMPUTE_TOLERANCE``);
    3. the CONTROL model, whose probe did record macro AUPR, gets its stored value back from the same formula;
    4. the target has no macro_aupr row in this cell yet (a second run is an error).

    New rows copy the fmax row of the same run and name their source in ``recomputed_from``.
    """
    import numpy as np

    from downstream.common import metrics as M

    root = results_root(root)
    catalog = load_catalog()
    if task_spec(catalog, task)["kind"] != "multilabel":
        raise LedgerError(f"recompute-macro-aupr applies to multilabel tasks (ec, go_*); {task!r} is "
                          f"{task_spec(catalog, task)['kind']}")
    setting = {"task": task, "data_revision": data_revision, "protocol_id": protocol_id, "pooling": pooling}
    verify(root)
    with _locked(root):
        rows = _read_locked(root, "measurements")
        control_fmax = _recompute_cells(rows, setting, control, "fmax")
        control_macro = _recompute_cells(rows, setting, control, "macro_aupr")
        if not control_fmax:
            raise LedgerError(f"control {control}: no {task} test/all fmax row in setting {setting}")
        for seed, row in sorted(control_fmax.items()):  # check 3
            stored = control_macro.get(seed)
            if stored is None or stored["value"] is None:
                raise LedgerError(f"control {control} seed {seed}: no recorded macro_aupr to check the formula against")
            _, labels, scores = _saved_test_arrays(row)
            value = M.label_aupr(labels, scores)["macro_aupr"]
            if abs(value - stored["value"]) > RECOMPUTE_TOLERANCE:
                raise LedgerError(f"control {control} seed {seed}: recomputed {value} != recorded {stored['value']}")
            out(f"control {control} seed {seed}: recomputed {value:.10f} == recorded {stored['value']:.10f}")
        targets = _recompute_cells(rows, setting, model, "fmax")
        if not targets:
            raise LedgerError(f"{model}: no {task} test/all fmax row in setting {setting}")
        existing = _recompute_cells(rows, setting, model, "macro_aupr")
        pair = _recompute_cells(rows, setting, model, "pair_aupr")
        ingest_id, stamp = str(uuid.uuid4()), now_utc()
        new = []
        for seed, template in sorted(targets.items()):
            if seed in existing:  # check 4
                raise LedgerError(f"{model} seed {seed} already has a macro_aupr row in setting {setting}")
            npz, labels, scores = _saved_test_arrays(template)  # check 1
            full = M.evaluate(labels, scores)
            for name, reference in (("fmax", template["value"]), ("pair_aupr", pair.get(seed, {}).get("value"))):
                if reference is None or abs(full[name] - reference) > RECOMPUTE_TOLERANCE:  # check 2
                    raise LedgerError(f"{npz} does not reproduce the ledger {name} of {model} seed {seed}: "
                                      f"{full[name]} vs {reference}")
            macro = M.label_aupr(labels, scores)
            row = dict(template)
            row.update(metric="macro_aupr", value=float(macro["macro_aupr"]), undefined_reason=None, is_primary=False,
                       context={"macro_aupr_num_terms": macro["num_terms_evaluated"],
                                "macro_aupr_num_terms_skipped": macro["num_terms_skipped_no_positive"]},
                       row_id=str(uuid.uuid4()), ingest_id=ingest_id, ingested_utc=stamp,
                       recomputed_from=(f"{npz} predictions_sha256={sha256_path(npz)} via "
                                        "downstream.common.metrics.label_aupr (probe not re-run)"),
                       **metric_info(catalog, task, "macro_aupr"))
            new.append(row)
            out(f"{model} seed {seed}: macro_aupr {row['value']:.10f} (terms {macro['num_terms_evaluated']}, skipped "
                f"{macro['num_terms_skipped_no_positive']}; recomputed fmax/pair_aupr equal the ledger values)")
        values = [row["value"] for row in new]
        spread = f" std(ddof=1) {float(np.std(values, ddof=1)):.6f}" if len(values) > 1 else ""
        out(f"mean {float(np.mean(values)):.6f}{spread} n={len(values)}")
        record = {"ingest_id": ingest_id, "started_utc": stamp, "finished_utc": now_utc(),
                  "command": command or " ".join(sys.argv), "code_git": _code_git(), "outputs_root": None,
                  "eval_dirs": [], "files_seen": len(new), "rows_added": len(new), "models_added": [],
                  "files_skipped_same_sha": 0, "embedding_sets": [], "catalog_sha256": catalog["sha256"],
                  "note": f"macro_aupr of {model} recomputed from predictions.npz; control={control}; setting={setting}"}
        if dry_run:
            out("DRY-RUN: nothing written")
            return dict(record, rows_added=0, dry_run=True)
        _append_locked(root, "measurements", new)
        _append_locked(root, "ingests", [record])
    out(f"written: {len(new)} rows, ingest {ingest_id}")
    return record


def _code_git() -> Optional[str]:
    """Commit of this checkout (``+dirty`` when it has local changes); None outside a git checkout."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}+dirty" if dirty else sha


# ---------------------------------------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------------------------------------

def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def matches(row: dict, selector: dict) -> bool:
    return all(_text(row.get(key)) == _text(value) for key, value in selector.items())


def annotate(root: Optional[Path], selector: dict, status: str, reason: str, *, replaced_by: Optional[str] = None,
             author: Optional[str] = None) -> dict:
    root = results_root(root)
    if status not in STATUSES:
        raise LedgerError(f"status {status!r} is not one of {STATUSES}")
    if not selector:
        raise LedgerError("empty selector: an annotation must name at least one row field")
    if not reason:
        raise LedgerError("an annotation needs a reason")
    with _locked(root):
        rows = _read_locked(root, "measurements")
        fields = set().union(*(row.keys() for row in rows)) if rows else set()
        unknown = sorted(set(selector) - fields)
        if unknown:
            raise LedgerError(f"unknown selector field(s) {unknown}; measurement fields are {sorted(fields)}")
        matched = sum(matches(row, selector) for row in rows)
        if not matched:
            raise LedgerError(f"selector {selector} matches no measurement")
        record = {"annotation_id": str(uuid.uuid4()), "created_utc": now_utc(), "selector": selector,
                  "status": status, "reason": reason, "replaced_by": replaced_by, "author": author,
                  "rows_matched_at_creation": matched}
        _append_locked(root, "annotations", [record])
    return record


def apply_annotations(rows: list, annotations: list) -> list:
    """Copies of ``rows`` with ``status`` (latest non-caveat note wins; default valid) and ``caveats``."""
    out = []
    for row in rows:
        row = dict(row, status="valid", status_reason=None, replaced_by=None, caveats=[])
        for note in annotations:
            if not matches(row, note["selector"]):
                continue
            if note["status"] == "caveat":
                row["caveats"].append(note["reason"])
            else:
                row.update(status=note["status"], status_reason=note["reason"], replaced_by=note.get("replaced_by"))
        out.append(row)
    return out


def parse_selector(text: str) -> dict:
    selector = {}
    for part in text.split():
        if "=" not in part:
            raise LedgerError(f"selector part {part!r} is not key=value")
        key, value = part.split("=", 1)
        selector[key] = value
    return selector


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-root", type=Path, default=None, help=f"default: ${ENV_VAR}")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("--outputs-root", type=Path, default=None)
    add.add_argument("--eval-dir", type=Path, action="append", default=[])
    note = commands.add_parser("annotate")
    note.add_argument("--select", required=True, help='"field=value field=value"')
    note.add_argument("--status", required=True, choices=STATUSES)
    note.add_argument("--reason", required=True)
    note.add_argument("--replaced-by", default=None)
    note.add_argument("--author", default=None)
    commands.add_parser("verify")
    recompute = commands.add_parser("recompute-macro-aupr",
                                    help="add a missing macro AUPR from saved predictions.npz (see recompute_macro_aupr)")
    recompute.add_argument("--task", required=True)
    recompute.add_argument("--model", required=True)
    recompute.add_argument("--control", required=True, help="model whose probe recorded macro AUPR (formula check)")
    recompute.add_argument("--data-revision", required=True)
    recompute.add_argument("--protocol-id", default="mlp_probe")
    recompute.add_argument("--pooling", default="mean_atoms")
    recompute.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "recompute-macro-aupr":
        result = recompute_macro_aupr(args.results_root, task=args.task, model=args.model, control=args.control,
                                      data_revision=args.data_revision, protocol_id=args.protocol_id,
                                      pooling=args.pooling, dry_run=args.dry_run, command=" ".join(sys.argv))
        print(json.dumps({k: v for k, v in result.items() if k != "embedding_sets"}, indent=1))
        return 0
    if args.command == "add":
        if args.outputs_root is None and not args.eval_dir:
            parser.error("add needs --outputs-root and/or --eval-dir")
        result = ingest(args.results_root, outputs_root=args.outputs_root, eval_dirs=args.eval_dir,
                        command=" ".join(sys.argv))
        result = {k: v for k, v in result.items() if k != "embedding_sets"} | {
            "embedding_sets": len(result["embedding_sets"])}
    elif args.command == "annotate":
        result = annotate(args.results_root, parse_selector(args.select), args.status, args.reason,
                          replaced_by=args.replaced_by, author=args.author)
    else:
        result = verify(args.results_root)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
