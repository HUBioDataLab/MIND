"""Reaction task surface (IEConv ProtFunct enzyme-reaction classification) for the generic downstream CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from downstream.tasks import multiclass
from downstream.tasks.multiclass import KIND, MulticlassProcessedSplit, file_index_path, processed_dir, task_paths  # noqa: F401

TASK_YAML = Path(__file__).with_name("task.yaml")
TASK = "reaction"


def load_task_config(path: Path = TASK_YAML, *, data_root: Optional[Path] = None) -> dict:
    return multiclass.load_config(path, expected_task=TASK, data_root=data_root)


class ProcessedSplit(MulticlassProcessedSplit):
    pass


def lists_dir(config: dict) -> Path:
    return task_paths(config)["raw"] / "lists"


def read_unique_functions(path: Path, expected: int) -> List[str]:
    """``unique_functions.txt``: one EC number per line, class id = line number (IEConv ``ProtFunctDataSet``)."""
    with open(path) as handle:
        classes = [line.rstrip("\n") for line in handle]
    if classes and classes[-1] == "":
        classes.pop()  # a single trailing newline is not a class
    if len(classes) != expected or len(set(classes)) != expected or any(not c.strip() for c in classes):
        raise ValueError(f"{path}: {len(classes)} classes ({len(set(classes))} unique), expected {expected} non-empty")
    return classes


def read_chain_functions(path: Path, class_count: int) -> Dict[str, int]:
    """``chain_functions.txt``: ``<protein>,<class index>``."""
    labels: Dict[str, int] = {}
    with open(path) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            cells = line.rstrip("\n").split(",")
            if len(cells) != 2 or not cells[1].isdigit():
                raise ValueError(f"{path}:{number}: expected '<protein>,<class index>', got {line[:60]!r}")
            name, index = cells[0], int(cells[1])
            if name in labels:
                raise ValueError(f"{path}:{number}: duplicate protein {name}")
            if index >= class_count:
                raise ValueError(f"{path}:{number}: class index {index} >= {class_count}")
            labels[name] = index
    if not labels:
        raise ValueError(f"{path}: no rows")
    return labels


def read_labels(config: dict) -> Tuple[List[str], Dict[str, int], Dict[str, List[str]]]:
    """``(classes, label of every protein, protein names per split)`` from the pinned repository lists."""
    source = config["sources"][config["source"]]
    base = lists_dir(config)
    classes = read_unique_functions(base / "unique_functions.txt", int(config["label_count"]))
    labels = read_chain_functions(base / "chain_functions.txt", len(classes))
    split_names: Dict[str, List[str]] = {}
    for split, filename in source["split_lists"].items():
        with open(base / filename) as handle:
            names = [line.strip() for line in handle if line.strip()]
        if len(set(names)) != len(names):
            raise ValueError(f"{base / filename}: duplicate proteins")
        missing = [n for n in names if n not in labels]
        if missing:
            raise ValueError(f"{base / filename}: {len(missing)} proteins have no chain_functions row, first {missing[:5]}")
        split_names[split] = names
    return classes, labels, split_names
