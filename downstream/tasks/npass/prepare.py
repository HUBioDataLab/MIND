"""Download and process the NPASS tasks (NaFM data release; data-processing CLI, run only on explicit request).

    python -m downstream.tasks.npass.prepare download --task npass_ptp1b [--data-root DIR]
    python -m downstream.tasks.npass.prepare process  --task npass_ptp1b|npass [--data-root DIR] [--tags seed0 ...]
                                                      [--splits train valid test]
    python -m downstream.tasks.npass.prepare validate --task npass_ptp1b|npass [--data-root DIR] [--tags ...] [--splits ...]

``download`` fetches ``regression_data.csv`` of the NaFM Figshare release (url, size, md5 and sha256 pinned under
``source_file`` in task.yaml) into the SHARED ``<source_root>/raw`` (one copy for all eight tasks; a file already there
is checked and never fetched again), refuses any size or checksum difference and appends a ``download`` record to
``<source_root>/provenance.json``. ``--task`` only selects the config; the download is the same for every task.

``process`` reads that CSV (read-only), builds the conformers once (RDKit, CPU), draws the split of every seed of
``split.seeds`` (tag ``seed<k>``, ``--tags`` selects some) and writes
``<data_root>/processed/<tag>/<revision>/{split}.pt`` plus ``qc_<split>.json/.tsv`` and one ``process`` record per tag in
``<data_root>/provenance.json``. ``--task npass`` processes the eight tasks one after the other. It never overwrites
an existing split file (the call raises before writing anything) and raises on any corrupt row, count mismatch or
split whose size differs from the rule's; molecules whose embedding fails are kept (flat 2-D coordinates), flagged and
counted in the QC files.

``validate`` runs the same reading, conformer generation and checks and prints the QC summaries without writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from downstream.common.prepare_spec import Archive, DownloadSpec, disk_estimate
from downstream.common.provenance import append_run
from downstream.common.registry import git_commit, sha256_file
from downstream.tasks import multiclass
from downstream.tasks.ec.prepare import _fetch
from downstream.tasks.npass import common

ALL = "npass"


def download(config: dict) -> dict:
    """Fetch (unless present) and check the pinned CSV; return the fields of the provenance record."""
    spec = config["source_file"]
    raw_dir = common.task_paths(config)["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / spec["name"]
    fetched = _fetch(spec["url"], target)
    size = target.stat().st_size
    if size != int(spec["bytes"]):
        raise RuntimeError(f"{target}: {size} bytes, pinned {spec['bytes']}")
    if fetched["sha256"] != spec["sha256"]:
        raise RuntimeError(f"{target}: sha256 {fetched['sha256']} != pinned {spec['sha256']}")
    md5 = multiclass.md5_file(target)
    if md5 != spec["md5"]:
        raise RuntimeError(f"{target}: md5 {md5} != pinned {spec['md5']}")
    with open(target, encoding="utf-8") as handle:
        rows = sum(1 for _ in handle) - 1
    if rows != int(spec["rows"]):
        raise RuntimeError(f"{target}: {rows} data rows, pinned {spec['rows']}")
    return {"file": fetched, "bytes": size, "md5": md5, "rows": rows, "record": spec["record"],
            "tasks": sorted(common.DATASET_OF_TASK)}


def download_spec(config: dict) -> DownloadSpec:
    """The shared CSV (``ds-prepare``); every ``npass_*`` task has the same spec."""
    spec, paths = config["source_file"], common.task_paths(config)
    target = paths["raw"] / spec["name"]
    archive = Archive(target, spec["url"], int(spec["bytes"]), md5=spec["md5"], sha256=spec["sha256"])
    return DownloadSpec((archive,), (target,), extracted_gib=disk_estimate(config, "extracted"))


def cli_args(step: str, *, task: str, workers: int, threads: int) -> list:
    return ["--task", task]


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("download", "validate", "process"))
    parser.add_argument("--task", required=True, choices=sorted(common.DATASET_OF_TASK) + [ALL])
    parser.add_argument("--config", type=Path, default=common.TASK_YAML)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--tags", nargs="+", default=None, help="split tags to process (default: every tag of the task)")
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args(argv)
    if args.task == ALL and (args.data_root is not None or args.command == "download"):
        parser.error("--task npass takes neither --data-root nor `download` (one shared download: use any npass_* task)")
    tasks = sorted(common.DATASET_OF_TASK) if args.task == ALL else [args.task]
    for task in tasks:
        config = common.load_task_config(args.config, data_root=args.data_root, dataset=common.DATASET_OF_TASK[task])
        if args.command == "download":
            started = datetime.now(timezone.utc).isoformat()
            result = download(config)
            record = {
                "command": " ".join(sys.argv), "kind": "download", "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(), "config": str(args.config),
                "config_sha256": sha256_file(args.config), "git": git_commit(), **result,
            }
            append_run(common.task_paths(config)["source_root"] / "provenance.json", record)
            print(json.dumps(record, indent=1))
            return 0
        splits = args.splits or config["splits"]
        unknown = [s for s in splits if s not in config["splits"]]
        if unknown:
            parser.error(f"unknown splits {unknown}; choose from {config['splits']}")
        tags = args.tags or config["split_tags"]
        unknown = [tag for tag in tags if tag not in config["split_tags"]]
        if unknown:
            parser.error(f"unknown split tags {unknown}; choose from {config['split_tags']}")
        print(f"== {task}", flush=True)
        common.build_processed(config, splits=splits, tags=tags, config_path=args.config,
                               dry_run=args.command == "validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
