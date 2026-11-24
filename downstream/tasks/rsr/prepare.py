"""Download, validate and process ATOM3D RSR (data-processing CLI; run only on explicit request).

    python -m downstream.tasks.rsr.prepare download [--data-root DIR]
    python -m downstream.tasks.rsr.prepare validate [--data-root DIR] [--sample 20]
    python -m downstream.tasks.rsr.prepare process  [--data-root DIR] [--splits train val test] [--workers N]

``download`` is idempotent: a file that is already present is kept (never overwritten), re-hashed and checked against
the md5 and size pinned in ``task.yaml``; a new download goes to ``*.partial`` and is renamed only when complete.
Zenodo publishes md5 only, so the sha256 of every file is written to ``provenance.json`` on the first run and a later
run whose file hashes differently is refused. Extraction refuses members that escape the destination and never
overwrites an existing tree. The three LMDB directories are located by name (the archive layout is not documented), the
split sizes are compared with the line counts of the record's index files and with ``expected_counts``, and every run
appends a record to ``provenance.json``.

``validate`` reads without writing: it locates the LMDBs, checks the split sizes and converts the first ``--sample``
models of every split through the production path, printing what the conversion dropped or counted.

``process`` converts every model (QC per split, see ``dataset.py``); it exits 1 when a split fails its QC and then does
not write that split's ``.pt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import requests

from downstream.common.prepare_spec import Archive, DownloadSpec, disk_estimate
from downstream.common.registry import git_commit, sha256_file
from downstream.tasks.lba.prepare import _append_provenance, _safe_extract
from downstream.tasks.rsr.dataset import (
    TASK_YAML, build_processed, iter_lmdb, lmdb_dirs, lmdb_length, load_task_config, processing_settings,
    process_item, task_paths,
)

TIMEOUT = 300


def hash_file(path: Path) -> dict:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            md5.update(block)
            sha.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def previous_sha256(provenance: Path, name: str) -> Optional[str]:
    """The sha256 an earlier ``download`` recorded for ``name`` (None on the first run); records must agree."""
    if not provenance.exists():
        return None
    seen = {entry["sha256"] for run in json.loads(provenance.read_text())["runs"]
            if run.get("step") == "download" for key, entry in run.get("files", {}).items()
            if entry.get("name") == name and "sha256" in entry}
    if len(seen) > 1:
        raise RuntimeError(f"{provenance}: earlier runs recorded different sha256 values for {name}: {sorted(seen)}")
    return next(iter(seen), None)


def fetch(url: str, dest: Path) -> bool:
    """Download ``url`` to ``dest`` unless it exists; returns whether a download happened."""
    if dest.exists():
        return False
    partial = dest.with_name(dest.name + ".partial")
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        if response.status_code != 200:
            raise RuntimeError(f"{url}: HTTP {response.status_code}")
        with open(partial, "wb") as handle:
            for block in response.iter_content(1 << 20):
                handle.write(block)
    os.replace(partial, dest)
    return True


def index_counts(config: dict, indices_root: Path) -> dict:
    """Line counts of the record's split-index files (the independent count of each split)."""
    files = config["sources"][config["source"]]["index_files"]
    counts = {}
    for split in config["splits"]:
        path = indices_root / files[split]
        if not path.is_file():
            raise FileNotFoundError(f"{path}: index file of the {split} split not found after extraction")
        lines = [line for line in path.read_text().split("\n") if line.strip()]
        if len(set(lines)) != len(lines):
            raise ValueError(f"{path}: duplicated indices")
        counts[split] = len(lines)
    return counts


def check_counts(config: dict, *, lmdb_counts: dict, indices: Optional[dict]) -> None:
    expected = {split: int(config["expected_counts"][split]) for split in config["splits"]}
    if lmdb_counts != expected:
        raise RuntimeError(f"LMDB counts {lmdb_counts} != expected_counts {expected}")
    if indices is not None and indices != expected:
        raise RuntimeError(f"index-file counts {indices} != expected_counts {expected}")


def download(config: dict) -> dict:
    source = config["sources"][config["source"]]
    paths = task_paths(config)
    paths["raw"].mkdir(parents=True, exist_ok=True)
    paths["extracted"].mkdir(parents=True, exist_ok=True)
    records = {}
    for key, entry in source["files"].items():
        dest = paths["raw"] / entry["name"]
        downloaded = fetch(source["base_url"].format(name=entry["name"]), dest)
        found = hash_file(dest)
        if found["md5"] != entry["md5"]:
            raise RuntimeError(f"{entry['name']}: md5 {found['md5']} != pinned {entry['md5']}")
        if found["bytes"] != int(entry["bytes"]):
            raise RuntimeError(f"{entry['name']}: {found['bytes']} bytes != pinned {entry['bytes']}")
        earlier = previous_sha256(paths["provenance"], entry["name"])
        if earlier is not None and earlier != found["sha256"]:
            raise RuntimeError(f"{entry['name']}: sha256 {found['sha256']} != {earlier} recorded by an earlier download")
        records[key] = {"name": entry["name"], "path": str(dest), "downloaded": downloaded,
                        "sha256_first_recorded": earlier is None, **found}
    trees = {}
    for key in ("indices", "split"):
        tree = paths["extracted"] / key
        if not tree.exists():
            partial = paths["extracted"] / f"{key}.partial"
            if partial.exists():
                raise RuntimeError(f"{partial} exists from an interrupted extraction; remove it and run again")
            partial.mkdir()
            _safe_extract(paths["raw"] / source["files"][key]["name"], partial)
            os.replace(partial, tree)
        trees[key] = str(tree)
    lmdb_counts = {split: lmdb_length(path) for split, path in lmdb_dirs(config).items()}
    indices = index_counts(config, paths["extracted"] / "indices")
    check_counts(config, lmdb_counts=lmdb_counts, indices=indices)
    return {"files": records, "extracted": trees, "lmdb_dirs": {s: str(p) for s, p in lmdb_dirs(config).items()},
            "lmdb_counts": lmdb_counts, "index_counts": indices}


def validate(config: dict, sample: int) -> int:
    """Read-only check of the extracted data; exit code 1 when any sampled model cannot be converted."""
    paths = task_paths(config)
    directories = lmdb_dirs(config)
    lmdb_counts = {split: lmdb_length(path) for split, path in directories.items()}
    indices_root = paths["extracted"] / "indices"
    indices = index_counts(config, indices_root) if indices_root.is_dir() else None
    check_counts(config, lmdb_counts=lmdb_counts, indices=indices)
    print(json.dumps({"lmdb_dirs": {s: str(p) for s, p in directories.items()}, "lmdb_counts": lmdb_counts,
                      "index_counts": indices}, indent=1))
    settings, bad = processing_settings(config), 0
    for split, path in directories.items():
        stop = min(sample, lmdb_counts[split])
        for index, item in iter_lmdb(path, 0, stop):
            result = process_item(item, index, settings)
            bad += result.status != "ok"
            print(f"[{split}] {index}: {result.status} {result.reason} puzzle={result.puzzle!r} source_id={result.source_id!r} "
                  f"rmsd={result.target:.3f} report={result.report}", flush=True)
    return 1 if bad else 0


def download_spec(config: dict) -> DownloadSpec:
    """The two pinned archives (split LMDBs, index files) and their extracted trees (``ds-prepare``)."""
    source = config["sources"][config["source"]]
    paths = task_paths(config)
    archives = tuple(
        Archive(paths["raw"] / source["files"][key]["name"], source["base_url"].format(name=source["files"][key]["name"]),
                int(source["files"][key]["bytes"]), md5=source["files"][key]["md5"])
        for key in ("split", "indices"))
    outputs = tuple(paths["extracted"] / key for key in ("split", "indices"))
    return DownloadSpec(archives, outputs, extracted_gib=disk_estimate(config, "extracted"))


def cli_args(step: str, *, task: str, workers: int, threads: int) -> list:
    return ["--workers", str(workers)] if step == "process" else []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("download", "validate", "process"))
    parser.add_argument("--config", type=Path, default=TASK_YAML)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8, help="process: processes converting models in parallel")
    parser.add_argument("--sample", type=int, default=20, help="validate: models converted per split")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.sample < 1:
        parser.error("--workers and --sample must be at least 1")
    config = load_task_config(args.config, data_root=args.data_root)
    splits = args.splits or list(config["splits"])
    unknown = [split for split in splits if split not in config["splits"]]
    if unknown:
        parser.error(f"unknown splits {unknown}; expected {config['splits']}")
    started = datetime.now(timezone.utc).isoformat()
    if args.command == "validate":
        return validate(config, args.sample)
    provenance = task_paths(config)["provenance"]
    record = {"command": " ".join(sys.argv), "started_utc": started, "step": args.command,
              "config": str(args.config), "config_sha256": sha256_file(args.config), "git": git_commit()}
    if args.command == "process":
        code = build_processed(config, splits=splits, workers=args.workers)
        record.update(splits=splits, workers=args.workers, exit_code=code,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        _append_provenance(provenance, record)
        return code
    result = download(config)
    record.update(source=config["source"], record=config["sources"][config["source"]]["record"],
                  doi=config["sources"][config["source"]]["doi"], finished_utc=datetime.now(timezone.utc).isoformat(),
                  **result)
    _append_provenance(provenance, record)
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
