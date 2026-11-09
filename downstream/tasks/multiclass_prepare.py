"""Shared command-line driver of the Fold and Reaction prepare modules (data-processing CLI; run only on explicit request).

    python -m downstream.tasks.<fold|reaction>.prepare download [--data-root DIR]
    python -m downstream.tasks.<fold|reaction>.prepare validate [--data-root DIR]      # read-only
    python -m downstream.tasks.<fold|reaction>.prepare process  [--data-root DIR] [--splits ...] [--workers N]

``download`` is task specific (passed in as ``download_fn``), never overwrites existing raw files, and ends with the
same validation as ``validate``. Each download/validate run appends a record to ``<data_root>/provenance.json`` (other
keys such as leakage records are preserved). ``process`` builds one processed file per split and exits 1 when a split
fails QC and appends a ``process`` record (command, config and input/output sha256, QC summaries) to
``provenance.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from downstream.common.provenance import append_run as _append_run, record_process_run
from downstream.common.registry import git_commit, sha256_file
from downstream.tasks import multiclass
from downstream.tasks.ec.dataset import read_split


def write_split_lists(config: dict, split_names: dict) -> None:
    """Write ``splits/<split>.txt``; an existing list with different content raises."""
    directory = multiclass.task_paths(config)["splits"]
    directory.mkdir(parents=True, exist_ok=True)
    for split, names in split_names.items():
        target = directory / f"{split}.txt"
        content = "\n".join(names) + "\n"
        if target.exists() and target.read_text() != content:
            raise FileExistsError(f"{target} exists with different content")
        target.write_text(content)


def validate_with(config: dict, read_labels: Callable) -> dict:
    """Read-only: labels, class list, split lists and the structure index must agree with ``task.yaml``."""
    paths = multiclass.task_paths(config)
    classes, labels, split_names = read_labels(config)
    splits = {s: read_split(paths["splits"] / f"{s}.txt") for s in config["splits"]}
    for split, names in split_names.items():
        if splits[split] != names:
            raise ValueError(f"splits/{split}.txt differs from the archive/list order of {split}")
    result = multiclass.validate_inputs(config, splits, labels, classes)
    index = json.loads(multiclass.file_index_path(config).read_text())
    missing = [c for ids in splits.values() for c in ids if c not in index]
    if missing:
        raise ValueError(f"{len(missing)} listed proteins are not in the structure index, first {missing[:5]}")
    result["structure_files_indexed"] = len(index)
    return result


def main(argv: Optional[list], *, doc: str, load_config: Callable, task_yaml: Path, read_labels: Callable,
         download_fn: Callable, builder: str) -> int:
    parser = argparse.ArgumentParser(description=doc.split("\n\n")[0])
    parser.add_argument("command", choices=("download", "validate", "process"))
    parser.add_argument("--config", type=Path, default=task_yaml)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    config = load_config(args.config, data_root=args.data_root)
    paths = multiclass.task_paths(config)
    if args.command == "process":
        splits = args.splits or config["splits"]
        started = datetime.now(timezone.utc).isoformat()
        classes, labels, _ = read_labels(config)
        exit_code = multiclass.build_processed(config, labels, classes, splits=splits, workers=args.workers,
                                               builder=builder)
        record_process_run(
            paths["provenance"], started_utc=started, config_path=args.config,
            processed_dir=multiclass.processed_dir(config), splits=splits, exit_code=exit_code,
            inputs=[multiclass.file_index_path(config), *(paths["splits"] / f"{s}.txt" for s in splits)])
        return exit_code
    started = datetime.now(timezone.utc).isoformat()
    result = download_fn(config) if args.command == "download" else {}
    result["validation"] = validate_with(config, read_labels)
    record = {
        "command": " ".join(sys.argv), "kind": args.command, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "source": config["source"],
        "config": str(args.config), "config_sha256": sha256_file(args.config), "git": git_commit(), **result,
    }
    paths["root"].mkdir(parents=True, exist_ok=True)
    _append_run(paths["provenance"], record)
    print(json.dumps(record, indent=1))
    return 0
