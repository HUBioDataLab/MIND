"""Download and process the MoleculeNet tasks (Uni-Mol LMDB files; data-processing CLI, run only on explicit request).

    python -m downstream.tasks.molnet.prepare download --task mn_esol [--data-root DIR]
    python -m downstream.tasks.molnet.prepare process  --task mn_esol [--data-root DIR] [--splits train valid test]
    python -m downstream.tasks.molnet.prepare validate --task mn_esol [--data-root DIR] [--splits ...]

``download`` fetches the Uni-Mol archive (url, size, md5 and sha256 pinned under ``archive`` in task.yaml) into the
SHARED ``<source_root>/raw`` (one copy for all 14 tasks; an archive already there is checked and never fetched again),
checks size and both checksums, extracts it once and checks that every data set has its three LMDB files. Each run
appends a ``download`` record to ``<source_root>/provenance.json`` (other keys are preserved). ``--task`` only selects
the config; the download is the same for every ``mn_*`` task.

``process`` reads ``<source_root>/raw/extracted/molecular_property_prediction/<dataset>/{train,valid,test}.lmdb``
(read-only), writes ``<data_root>/processed/<revision>/{split}.pt`` plus ``qc_<split>.json/.tsv`` and appends a
``process`` record (command, config and input/output sha256, QC summaries) to ``<data_root>/provenance.json``.
It never overwrites an existing split file and raises on any corrupt record, count mismatch or duplicate molecule.

``validate`` runs the same reading and checks and prints the QC summaries without writing anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from downstream.common.provenance import append_run
from downstream.common.prepare_spec import Archive, DownloadSpec, disk_estimate
from downstream.common.registry import git_commit, sha256_file
from downstream.tasks import multiclass
from downstream.tasks.ec.prepare import _fetch
from downstream.tasks.molnet import common


def download(config: dict) -> dict:
    """Fetch (unless present), check and extract the pinned archive; return the fields of the provenance record."""
    spec = config["archive"]
    paths = common.task_paths(config)
    raw_dir = paths["source_root"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / spec["name"]
    fetched = _fetch(spec["url"], archive)
    size = archive.stat().st_size
    if size != int(spec["bytes"]):
        raise RuntimeError(f"{archive}: {size} bytes, pinned {spec['bytes']}")
    if fetched["sha256"] != spec["sha256"]:
        raise RuntimeError(f"{archive}: sha256 {fetched['sha256']} != pinned {spec['sha256']}")
    md5 = multiclass.md5_file(archive)
    if md5 != spec["md5"]:
        raise RuntimeError(f"{archive}: md5 {md5} != pinned {spec['md5']}")

    extract_to = paths["source_root"] / spec["extract_subdir"]
    extracted = False
    if not paths["raw"].exists():
        if extract_to.exists():
            raise RuntimeError(f"{extract_to} exists but holds no {paths['raw'].name}; inspect it before extracting")
        partial = extract_to.with_name(extract_to.name + ".partial")
        if partial.exists():
            raise RuntimeError(f"{partial} is left from an interrupted extraction; remove it and rerun")
        partial.mkdir(parents=True)
        multiclass.safe_extract_archive(archive, partial)
        os.replace(partial, extract_to)
        extracted = True
    missing = [str(paths["raw"] / dataset / f"{split}.lmdb")
               for dataset in sorted(common.TASK_OF_DATASET) for split in config["splits"]
               if not (paths["raw"] / dataset / f"{split}.lmdb").is_file()]
    if missing:
        raise FileNotFoundError(f"the extracted archive lacks {len(missing)} LMDB file(s): {missing}")
    return {"archive": fetched, "bytes": size, "md5": md5, "extracted": extracted, "lmdb_dir": str(paths["raw"]),
            "datasets": sorted(common.TASK_OF_DATASET)}


def download_spec(config: dict) -> DownloadSpec:
    """The shared archive and its extracted LMDB tree (``ds-prepare``); every ``mn_*`` task has the same spec."""
    spec, paths = config["archive"], common.task_paths(config)
    archive = Archive(paths["source_root"] / "raw" / spec["name"], spec["url"], int(spec["bytes"]),
                      md5=spec["md5"], sha256=spec["sha256"])
    return DownloadSpec((archive,), (paths["raw"],), extracted_gib=disk_estimate(config, "extracted"))


def cli_args(step: str, *, task: str, workers: int, threads: int) -> list:
    return ["--task", task]


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("download", "validate", "process"))
    parser.add_argument("--task", required=True, choices=sorted(common.DATASET_OF_TASK))
    parser.add_argument("--config", type=Path, default=common.TASK_YAML)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args(argv)
    config = common.load_task_config(args.config, data_root=args.data_root, dataset=common.DATASET_OF_TASK[args.task])
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
    return common.build_processed(config, splits=splits, config_path=args.config, dry_run=args.command == "validate")


if __name__ == "__main__":
    raise SystemExit(main())
