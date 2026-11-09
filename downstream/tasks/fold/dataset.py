"""Fold task surface (SCOPe 1.75 fold classification, GearNet Fold3D) for the generic downstream CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from downstream.tasks import multiclass
from downstream.tasks.multiclass import KIND, MulticlassProcessedSplit, file_index_path, processed_dir, task_paths  # noqa: F401

TASK_YAML = Path(__file__).with_name("task.yaml")
TASK = "fold"
_SID_COLUMNS = (3, 4)  # id, [aux integer], fold, fold  (observed in the IEConv-repo copies of the lists)


def load_task_config(path: Path = TASK_YAML, *, data_root: Optional[Path] = None) -> dict:
    return multiclass.load_config(path, expected_task=TASK, data_root=data_root)


class ProcessedSplit(MulticlassProcessedSplit):
    pass


def read_class_map(path: Path, expected: int) -> List[str]:
    """``class_map.txt``: ``<fold id>\\t<index>`` per line; indices must be exactly 0..expected-1 once each."""
    names: Dict[int, str] = {}
    with open(path) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            cells = line.rstrip("\n").split("\t")
            if len(cells) != 2 or not cells[1].isdigit():
                raise ValueError(f"{path}:{number}: expected '<fold id><TAB><index>', got {line[:60]!r}")
            index = int(cells[1])
            if index in names:
                raise ValueError(f"{path}:{number}: class index {index} appears twice")
            names[index] = cells[0]
    if sorted(names) != list(range(expected)):
        raise ValueError(f"{path}: {len(names)} classes with indices {min(names, default=None)}..{max(names, default=None)}, "
                         f"expected exactly 0..{expected - 1}")
    ordered = [names[i] for i in range(expected)]
    if len(set(ordered)) != expected:
        raise ValueError(f"{path}: duplicate class names")
    return ordered


def read_label_file(path: Path, class_index: Dict[str, int]) -> List[Tuple[str, int]]:
    """Rows ``<domain id>[\\t<aux>]\\t<fold>\\t<fold>``; the label is the LAST column (GearNet ``get_label_list``)."""
    rows: List[Tuple[str, int]] = []
    seen = set()
    with open(path) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            cells = line.rstrip("\n").split("\t")
            if len(cells) not in _SID_COLUMNS:
                raise ValueError(f"{path}:{number}: {len(cells)} columns, expected one of {_SID_COLUMNS}")
            name, label = cells[0], cells[-1]
            if label not in class_index:
                raise ValueError(f"{path}:{number}: label {label!r} of {name} is not in class_map")
            if name in seen:
                raise ValueError(f"{path}:{number}: duplicate protein {name}")
            seen.add(name)
            rows.append((name, class_index[label]))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def locate_root(config: dict) -> Path:
    """Folder holding ``class_map.txt`` below ``raw/<extract_dir>`` (the archive's inner layout is not fixed)."""
    source = config["sources"][config["source"]]
    base = task_paths(config)["raw"] / source["extract_dir"]
    hits = sorted(base.rglob(source["class_map_file"])) if base.is_dir() else []
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {source['class_map_file']} below {base}, found {[str(h) for h in hits]}")
    return hits[0].parent


def read_labels(config: dict) -> Tuple[List[str], Dict[str, int], Dict[str, List[str]]]:
    """``(classes, label of every protein, protein names per split)`` from the extracted archive's own files."""
    source = config["sources"][config["source"]]
    root = locate_root(config)
    classes = read_class_map(root / source["class_map_file"], int(config["label_count"]))
    index = {name: i for i, name in enumerate(classes)}
    labels: Dict[str, int] = {}
    split_names: Dict[str, List[str]] = {}
    for split in config["splits"]:
        rows = read_label_file(root / source["label_files"][split], index)
        split_names[split] = [name for name, _ in rows]
        for name, label in rows:
            if name in labels:
                raise ValueError(f"protein {name} is listed in more than one split")
            labels[name] = label
    return classes, labels, split_names
