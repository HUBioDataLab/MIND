"""Downstream task registry.

Every task module exposes the same surface used by the ``downstream.*`` CLIs:

``TASK_YAML``      path of the checked-in task config
``KIND``           ``multilabel`` (EC, GO-MF/BP/CC), ``multiclass`` (Fold, Reaction) or ``regression`` (LBA)
``load_task_config(path, *, data_root)``
``processed_dir(config, ...)``
``ProcessedSplit``  processed split loader
"""

from __future__ import annotations

import importlib
from types import ModuleType

TASKS = ("ec", "lba", "go_mf", "go_bp", "go_cc", "fold", "reaction")
# Tasks whose module is not ``downstream.tasks.<name>.dataset``: the three GO branches share one family.
TASK_MODULES = {
    "go_mf": "downstream.tasks.go.mf",
    "go_bp": "downstream.tasks.go.bp",
    "go_cc": "downstream.tasks.go.cc",
}
KINDS = ("multilabel", "multiclass", "regression")


def load_task(name: str) -> ModuleType:
    """Import one task's dataset module; unknown names fail loudly."""
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {TASKS}")
    module = importlib.import_module(TASK_MODULES.get(name, f"downstream.tasks.{name}.dataset"))
    for attribute in ("TASK_YAML", "KIND", "load_task_config", "processed_dir", "ProcessedSplit"):
        if not hasattr(module, attribute):
            raise AttributeError(f"task {name!r} does not expose {attribute!r}")
    if module.KIND not in KINDS:
        raise ValueError(f"task {name!r} has unknown kind {module.KIND!r}")
    return module
