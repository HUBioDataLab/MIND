"""Fine-tune settings of one run, resolved from the task.yaml ``finetune`` block (no torch import: the ds-run plan
step compares finished runs with the current settings without loading torch)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

HEADS = ("linear", "mlp")  # downstream.common.heads.HEADS
REQUIRED_SETTINGS = ("pooling", "head", "seeds", "epochs", "patience", "max_atoms_per_batch", "max_samples_per_batch",
                     "lr_encoder", "lr_encoder_random_init", "lr_head", "weight_decay", "grad_clip_norm", "mlp_hidden",
                     "mlp_dropout")
POOLINGS = ("mean_atoms", "ca", "mean_residues", "mean_pocket", "mean_ligand", "concat_pocket_ligand")  # downstream.common.encoder.ALL_POOLINGS
OVERRIDES = ("lr_encoder", "lr_head", "weight_decay", "mlp_dropout", "pooling", "epochs", "patience",
             "lr_schedule", "warmup_ratio")
LR_SCHEDULES = ("constant", "cosine", "plateau")  # semantics: downstream.lr_schedule
DEFAULT_SCHEDULE = ("constant", 0.0)  # (lr_schedule, warmup_ratio); recorded in the settings only when changed


def _positive_or_null(settings: dict, name: str) -> None:
    value = settings[name]
    if value is not None and not (isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0):
        raise ValueError(f"finetune.{name} must be null or positive, got {value!r}")


def _check_ranges(settings: dict, allowed_poolings: Optional[Sequence[str]]) -> None:
    for name in ("lr_encoder", "lr_head"):
        if not (isinstance(settings[name], (int, float)) and not isinstance(settings[name], bool) and settings[name] > 0):
            raise ValueError(f"finetune.{name} must be positive, got {settings[name]!r}")
    decay, dropout = settings["weight_decay"], settings["mlp_dropout"]
    if not (isinstance(decay, (int, float)) and not isinstance(decay, bool) and decay >= 0):
        raise ValueError(f"finetune.weight_decay must be >= 0, got {decay!r}")
    if not (isinstance(dropout, (int, float)) and not isinstance(dropout, bool) and 0 <= dropout < 1):
        raise ValueError(f"finetune.mlp_dropout must be in [0, 1), got {dropout!r}")
    if settings["pooling"] not in POOLINGS:
        raise ValueError(f"finetune.pooling must be one of {POOLINGS}, got {settings['pooling']!r}")
    if allowed_poolings is not None and settings["pooling"] not in allowed_poolings:
        raise ValueError(f"finetune.pooling {settings['pooling']!r} is not available for this task; "
                         f"its embedding poolings are {tuple(allowed_poolings)}")


def schedule_of(settings: dict) -> tuple:
    """``(lr_schedule, warmup_ratio)`` of resolved settings; the defaults when the settings do not record them."""
    return settings.get("lr_schedule", DEFAULT_SCHEDULE[0]), float(settings.get("warmup_ratio", DEFAULT_SCHEDULE[1]))


def _check_schedule(settings: dict) -> None:
    """Validate the schedule keys; keep them in the settings only when they differ from the default, so runs without
    a schedule record exactly the settings they always did (``same_settings`` compares them)."""
    schedule = settings.pop("lr_schedule", DEFAULT_SCHEDULE[0])
    ratio = settings.pop("warmup_ratio", DEFAULT_SCHEDULE[1])
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"finetune.lr_schedule must be one of {LR_SCHEDULES}, got {schedule!r}")
    if not (isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and 0 <= ratio < 1):
        raise ValueError(f"finetune.warmup_ratio must be in [0, 1), got {ratio!r}")
    if schedule == "plateau" and ratio > 0:
        raise ValueError("finetune.warmup_ratio > 0 cannot be combined with lr_schedule 'plateau' "
                         "(warmup and the plateau scheduler would both write the learning rate)")
    if (schedule, float(ratio)) != DEFAULT_SCHEDULE:
        settings["lr_schedule"], settings["warmup_ratio"] = schedule, float(ratio)


def resolve_settings(block: dict, *, random_init: bool, overrides: Optional[Dict[str, Any]] = None,
                     allowed_poolings: Optional[Sequence[str]] = None) -> dict:
    """The settings a run actually uses: the task.yaml block, the encoder LR of the init arm, then CLI overrides.

    ``allowed_poolings`` (the task's ``embedding.poolings``) restricts the pooling; None checks the known names only."""
    missing = [key for key in REQUIRED_SETTINGS if key not in block]
    if missing:
        raise ValueError(f"task.yaml finetune block lacks {missing}")
    if block["head"] not in HEADS:
        raise ValueError(f"finetune.head must be one of {HEADS}, got {block['head']!r}")
    settings = dict(block)
    settings["encoder_init"] = "random" if random_init else "pretrained"
    if random_init:
        settings["lr_encoder"] = settings["lr_encoder_random_init"]
    for name, value in (overrides or {}).items():
        if name not in OVERRIDES:
            raise ValueError(f"finetune override {name!r} is not one of {OVERRIDES}")
        if value is not None:
            settings[name] = value
    if int(settings["epochs"]) < 1:
        raise ValueError(f"finetune.epochs must be positive, got {settings['epochs']}")
    for name in ("patience", "max_samples_per_batch", "grad_clip_norm"):
        _positive_or_null(settings, name)
    _check_ranges(settings, allowed_poolings)
    _check_schedule(settings)
    return settings


def same_settings(recorded: Optional[dict], expected: dict) -> bool:
    """Whether a finished run's recorded settings are the ones a new run would use (key order ignored)."""
    return recorded is not None and json.dumps(recorded, sort_keys=True) == json.dumps(expected, sort_keys=True)
