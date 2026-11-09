"""Download and process the GO benchmark (data-processing CLI; run only on explicit user request).

    python -m downstream.tasks.go.prepare download [--data-root DIR]
    python -m downstream.tasks.go.prepare validate [--data-root DIR]
    python -m downstream.tasks.go.prepare process  [--data-root DIR] [--splits train valid test] [--workers N]

``download`` fetches GeneOntology.zip (Zenodo 6622158, md5 and size pinned in task.yaml) into the SHARED
``<source_root>/raw`` (never overwritten; one copy for every task of the DeepFRI source), extracts it,
writes the three split id lists into ``<data_root>/splits`` and validates annotations against the splits.
``validate`` repeats that validation without touching the network. ``process`` builds the processed
files (one per split, three label matrices) and exits 1 when a split fails QC.

Every download/validate run appends a record to ``<data_root>/provenance.json`` (other keys such as the
leakage records are preserved).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from downstream.common.registry import git_commit, sha256_file
from downstream.tasks.ec.dataset import read_split, read_test_cutoffs, split_chain_id
from downstream.tasks.ec.prepare import _copy_once, _fetch, _safe_extract
from downstream.tasks.go import common
from downstream.tasks.go.common import BRANCH_ORDER, TASK_YAML, task_paths


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_run(path: Path, record: dict) -> None:
    existing = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(existing, dict):
        raise ValueError(f"{path}: expected a JSON object")
    existing.setdefault("runs", []).append(record)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(existing, indent=1))
    os.replace(partial, path)


def _list_dir(path: Path) -> list:
    return sorted(p.name for p in path.iterdir())[:30] if path.exists() else []


def download(config: dict) -> dict:
    paths = task_paths(config)
    source = config["sources"][config["source"]]
    paths["raw"].mkdir(parents=True, exist_ok=True)
    archive = paths["raw"] / Path(source["url"]).name
    record = _fetch(source["url"], archive)
    if archive.stat().st_size != int(source["bytes"]):
        raise RuntimeError(f"{archive}: {archive.stat().st_size} bytes, pinned {source['bytes']}")
    md5 = _md5(archive)
    if md5 != source["md5"]:
        raise RuntimeError(f"{archive}: md5 {md5} != pinned {source['md5']}")
    root = paths["raw"] / source["extract_dir"]
    if not root.exists():
        _safe_extract(archive, paths["raw"])
    if not root.is_dir():
        raise RuntimeError(f"archive did not produce {root}; raw dir holds {_list_dir(paths['raw'])}")
    for key in ("annotation_file", "test_cutoff_file"):
        if not (root / source[key]).is_file():
            raise RuntimeError(f"{root / source[key]} not found ({key} is UNVERIFIED in task.yaml); "
                               f"extracted dir holds {_list_dir(root)}")
    index, split_ids = {}, {}
    for split in config["splits"]:
        split_dir = root / split
        if not split_dir.exists() and (root / f"{split}.zip").is_file():
            _safe_extract(root / f"{split}.zip", root)
        if not split_dir.is_dir():
            raise RuntimeError(f"no {split} directory or {split}.zip under {root}; it holds {_list_dir(root)}")
        ids = []
        for pdb_file in sorted(split_dir.glob("*.pdb")):
            chain = pdb_file.name.split("_")[0]  # TorchDrug's own id rule
            split_chain_id(chain)
            if chain in index:
                raise RuntimeError(f"chain {chain} appears twice in the archive")
            index[chain] = str(pdb_file)
            ids.append(chain)
        split_ids[split] = ids
    paths["splits"].mkdir(parents=True, exist_ok=True)
    for split, ids in split_ids.items():
        target = paths["splits"] / f"{split}.txt"
        content = "\n".join(ids) + "\n"
        if target.exists() and target.read_text() != content:
            raise FileExistsError(f"{target} exists with different content")
        target.write_text(content)
    _copy_once(root / source["test_cutoff_file"], paths["splits"] / "test_cutoffs.csv")
    index_path = common.file_index_path(config)
    if not index_path.exists():
        index_path.write_text(json.dumps(index, indent=0))
    return {"archive": record, "md5": md5, "chains": {s: len(i) for s, i in split_ids.items()}}


def validate(config: dict) -> dict:
    paths = task_paths(config)
    source = config["sources"][config["source"]]
    expected_terms = {b: config["branches"][b]["label_count"] for b in BRANCH_ORDER}
    terms, labels = common.read_go_annotations(paths["raw"] / source["extract_dir"] / source["annotation_file"], expected_terms)
    splits = {s: read_split(paths["splits"] / f"{s}.txt") for s in config["splits"]}
    result = common.validate_inputs(config, splits, labels)
    subsets = read_test_cutoffs(paths["splits"] / "test_cutoffs.csv", config["test_cutoffs"])
    result["test_cutoff_counts"] = {c: len(s) for c, s in subsets.items()}
    result["terms"] = {b: len(terms[b]) for b in BRANCH_ORDER}
    return result


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("download", "validate", "process"))
    parser.add_argument("--config", type=Path, default=TASK_YAML)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    config = common.load_task_config(args.config, data_root=args.data_root, branch="mf")
    paths = task_paths(config)
    if args.command == "process":
        return common.build_processed(config, splits=args.splits or config["splits"], workers=args.workers)
    started = datetime.now(timezone.utc).isoformat()
    result = download(config) if args.command == "download" else {}
    result["validation"] = validate(config)
    record = {
        "command": " ".join(sys.argv), "kind": args.command, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "source": config["source"],
        "config": str(args.config), "config_sha256": sha256_file(args.config), "git": git_commit(), **result,
    }
    paths["root"].mkdir(parents=True, exist_ok=True)
    _append_run(paths["provenance"], record)
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
