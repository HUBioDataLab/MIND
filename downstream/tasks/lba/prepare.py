"""Download and process ATOM3D LBA (data-processing CLI; run only on explicit request).

    python -m downstream.tasks.lba.prepare download [--data-root DIR] [--tags identity30 identity60]
    python -m downstream.tasks.lba.prepare process  [--data-root DIR] [--tags ...] [--splits train val test]

``download`` is idempotent: a tarball that is already present is kept (never
overwritten), re-hashed and checked against the md5 pinned in ``task.yaml``;
a new download goes to ``*.partial`` and is renamed only when complete.
Extraction refuses members that escape the destination and never overwrites an
existing extracted tree. Every run appends a record (URLs, md5/sha256, byte
counts, LMDB entry counts) to ``provenance.json``.

``process`` exits 1 when a split's failed/dropped fraction exceeds
``qc.max_failed_complex_fraction``; that split's ``.pt`` is then not written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from downstream.common.registry import git_commit, sha256_file
from downstream.tasks.lba.dataset import (
    TASK_YAML, build_processed, lmdb_dir, lmdb_length, load_task_config, task_paths,
)

TIMEOUT = 300


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, dest: Path) -> dict:
    """Download ``url`` to ``dest`` unless it exists; return a provenance record."""
    if dest.exists():
        return {"url": url, "path": str(dest), "bytes": dest.stat().st_size,
                "md5": md5_file(dest), "sha256": sha256_file(dest), "downloaded": False}
    partial = dest.with_name(dest.name + ".partial")
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        if response.status_code != 200:
            raise RuntimeError(f"{url}: HTTP {response.status_code}")
        with open(partial, "wb") as handle:
            for block in response.iter_content(1 << 20):
                handle.write(block)
    os.replace(partial, dest)
    return {"url": url, "path": str(dest), "bytes": dest.stat().st_size,
            "md5": md5_file(dest), "sha256": sha256_file(dest), "downloaded": True}


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract without allowing a member to escape ``dest`` or to be a link."""
    root = dest.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"{archive}: member {member.name!r} is a link")
            target = (dest / member.name).resolve()
            if target != root and not str(target).startswith(str(root) + os.sep):
                raise RuntimeError(f"{archive}: member {member.name!r} escapes {dest}")
        handle.extractall(dest)


def download(config: dict, tags) -> dict:
    source = config["sources"][config["source"]]
    paths = task_paths(config)
    paths["raw"].mkdir(parents=True, exist_ok=True)
    paths["extracted"].mkdir(parents=True, exist_ok=True)
    records, trees = {}, {}
    for key in list(tags) + ["readme"]:
        entry = source["files"][key]
        record = _fetch(source["base_url"].format(name=entry["name"]), paths["raw"] / entry["name"])
        if record["md5"] != entry["md5"]:
            raise RuntimeError(f"{entry['name']}: md5 {record['md5']} != pinned {entry['md5']}")
        if int(record["bytes"]) != int(entry["bytes"]):
            raise RuntimeError(f"{entry['name']}: {record['bytes']} bytes != pinned {entry['bytes']}")
        records[key] = record
        if key == "readme":
            continue
        stem = entry["name"][len("LBA-"):-len(".tar.gz")]
        tree = paths["extracted"] / stem
        if not tree.is_dir():
            _safe_extract(paths["raw"] / entry["name"], paths["extracted"])
        if not tree.is_dir():
            raise RuntimeError(f"{entry['name']}: expected {tree} after extraction")
        trees[key] = str(tree)
    counts = {
        tag: {split: lmdb_length(lmdb_dir(config, tag, split)) for split in config["splits"]}
        for tag in tags
    }
    for tag, split_counts in counts.items():
        expected = {k: int(v) for k, v in config["expected_counts"][tag].items()}
        if split_counts != expected:
            raise RuntimeError(f"{tag}: LMDB counts {split_counts} != expected {expected}")
    return {"files": records, "extracted": trees, "lmdb_counts": counts}


def _append_provenance(path: Path, record: dict) -> None:
    runs = json.loads(path.read_text())["runs"] if path.exists() else []
    runs.append(record)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps({"runs": runs}, indent=1))
    os.replace(partial, path)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("download", "process"))
    parser.add_argument("--config", type=Path, default=TASK_YAML)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--tags", nargs="+", default=None, help="default: every split tag in task.yaml")
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    import torch

    if args.threads < 1:
        parser.error("--threads must be at least 1")
    torch.set_num_threads(args.threads)
    config = load_task_config(args.config, data_root=args.data_root)
    tags = args.tags or list(config["split_tags"])
    unknown = [tag for tag in tags if tag not in config["split_tags"]]
    if unknown:
        parser.error(f"unknown split tags {unknown}; expected {config['split_tags']}")
    paths = task_paths(config)
    started = datetime.now(timezone.utc).isoformat()

    if args.command == "process":
        exit_code = 0
        for tag in tags:
            exit_code |= build_processed(config, tag=tag, splits=args.splits or config["splits"])
        _append_provenance(paths["provenance"], {
            "command": " ".join(sys.argv), "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(), "step": "process",
            "tags": tags, "splits": args.splits or config["splits"],
            "config": str(args.config), "config_sha256": sha256_file(args.config),
            "git": git_commit(), "exit_code": exit_code,
        })
        return exit_code

    result = download(config, tags)
    record = {
        "command": " ".join(sys.argv), "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "step": "download",
        "source": config["source"], "record": config["sources"][config["source"]]["record"],
        "doi": config["sources"][config["source"]]["doi"], "tags": tags,
        "config": str(args.config), "config_sha256": sha256_file(args.config),
        "git": git_commit(), **result,
    }
    _append_provenance(paths["provenance"], record)
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
