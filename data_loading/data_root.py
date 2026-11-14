"""Expand ``${MIND_DATA_ROOT}`` in config values.

Pretraining configs, downstream ``task.yaml`` files and ``downstream/leakage_references.yaml`` name dataset
locations as ``${MIND_DATA_ROOT}/<relative path>``. The root comes from the ``MIND_DATA_ROOT`` environment
variable (``make`` exports it; set it in ``local.mk``); when it is unset or empty the default is ``<repo>/datasets``.
Any other ``${...}`` in a value raises, so a misspelled variable cannot turn into a literal path.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "MIND_DATA_ROOT"
PLACEHOLDER = "${" + ENV_VAR + "}"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "datasets"


def data_root() -> Path:
    """The data root: ``$MIND_DATA_ROOT`` if set and non-empty, else ``<repo>/datasets``."""
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value) if value else DEFAULT_DATA_ROOT


def expand_data_root(value):
    """Return ``value`` with every ``${MIND_DATA_ROOT}`` replaced; mappings and lists are walked recursively."""
    if isinstance(value, str):
        expanded = value.replace(PLACEHOLDER, str(data_root()))
        if "${" in expanded:
            raise ValueError(f"config value {value!r}: only {PLACEHOLDER} is expanded")
        return expanded
    if isinstance(value, dict):
        return {key: expand_data_root(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_data_root(item) for item in value]
    return value
