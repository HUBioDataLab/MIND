"""Fine-tune settings of one run, resolved from the task.yaml ``finetune`` block (no torch import: the ds-run plan
step compares finished runs with the current settings without loading torch)."""

from __future__ import annotations

import json
from typing import Dict, Optional

HEADS = ("linear", "mlp")  # downstream.common.heads.HEADS
REQUIRED_SETTINGS = ("pooling", "head", "seeds", "epochs", "patience", "max_atoms_per_batch", "max_samples_per_batch",
                     "lr_encoder", "lr_encoder_random_init", "lr_head", "weight_decay", "grad_clip_norm", "mlp_hidden",
                     "mlp_dropout")
OVERRIDES = ("lr_encoder", "epochs", "patience")


def _positive_or_null(settings: dict, name: str) -> None:
    value = settings[name]
    if value is not None and not (isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0):
        raise ValueError(f"finetune.{name} must be null or positive, got {value!r}")


def resolve_settings(block: dict, *, random_init: bool, overrides: Optional[Dict[str, Optional[float]]] = None) -> dict:
    """The settings a run actually uses: the task.yaml block, the encoder LR of the init arm, then CLI overrides."""
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
    return settings


def same_settings(recorded: Optional[dict], expected: dict) -> bool:
    """Whether a finished run's recorded settings are the ones a new run would use (key order ignored)."""
    return recorded is not None and json.dumps(recorded, sort_keys=True) == json.dumps(expected, sort_keys=True)
