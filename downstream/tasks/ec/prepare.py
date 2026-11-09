"""Download and process the EC benchmark (data-processing CLI; run only on explicit request).

    python -m downstream.tasks.ec.prepare download [--data-root DIR]
    python -m downstream.tasks.ec.prepare process  [--data-root DIR] [--splits train valid test] [--workers N]

``download`` is idempotent: files already present are kept (never overwritten)
and re-hashed; new files are written to ``*.partial`` and renamed when
complete. Every run appends a record (URLs, sha256/md5, counts, failures with
reasons) to ``provenance.json`` and exits 1 when the structure failure
fraction exceeds ``qc.max_download_failure_fraction``. ``process`` appends its own record (command, config
and input/output sha256, QC summaries) to ``provenance.json``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from downstream.common.provenance import append_run as _append_provenance, record_process_run
from downstream.common.registry import git_commit, sha256_file
from downstream.tasks.ec.dataset import (
    TASK_YAML, build_processed, load_task_config, processed_dir, read_annotations, read_split, read_test_cutoffs,
    split_chain_id, task_paths,
)

TIMEOUT = 120


def _fetch(url: str, dest: Path) -> dict:
    """Download ``url`` to ``dest`` unless it exists; return a provenance record."""
    if dest.exists():
        return {"url": url, "path": str(dest), "sha256": sha256_file(dest), "downloaded": False}
    partial = dest.with_name(dest.name + ".partial")
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        with open(partial, "wb") as handle:
            for block in response.iter_content(1 << 20):
                handle.write(block)
    os.replace(partial, dest)
    return {"url": url, "path": str(dest), "sha256": sha256_file(dest), "downloaded": True}


def _copy_once(src: Path, dest: Path) -> None:
    if dest.exists():
        if sha256_file(dest) != sha256_file(src):
            raise FileExistsError(f"{dest} exists with different content than {src}")
        return
    shutil.copyfile(src, dest)


def _fetch_structure(url: str, dest: Path) -> dict:
    try:
        record = _fetch(url, dest)
        with gzip.open(dest, "rb") as handle:
            if not handle.read(64).startswith(b"data_"):
                raise RuntimeError("not an mmCIF file")
        return record
    except Exception as exc:  # recorded and bounded by qc.max_download_failure_fraction
        return {"url": url, "path": str(dest), "error": f"{type(exc).__name__}: {exc}"}


def download_deepfri_rcsb(config: dict, paths: dict) -> dict:
    source = config["sources"]["deepfri_rcsb"]
    meta_dir = paths["raw"] / "deepfri"
    meta_dir.mkdir(parents=True, exist_ok=True)
    base = source["base_url"].format(commit=source["commit"])
    files = {}
    for key, entry in source["files"].items():
        record = _fetch(f"{base}/{entry['name']}", meta_dir / entry["name"])
        if record["sha256"] != entry["sha256"]:
            raise RuntimeError(f"{entry['name']}: sha256 {record['sha256']} != pinned {entry['sha256']}")
        files[key] = record
    paths["splits"].mkdir(parents=True, exist_ok=True)
    for split in config["splits"]:
        _copy_once(Path(files[split]["path"]), paths["splits"] / f"{split}.txt")
    _copy_once(Path(files["test_cutoffs"]["path"]), paths["splits"] / "test_cutoffs.csv")
    _copy_once(Path(files["annot"]["path"]), paths["raw"] / "annotations.tsv")

    chains = [c for s in config["splits"] for c in read_split(paths["splits"] / f"{s}.txt")]
    pdb_ids = sorted({split_chain_id(c)[0] for c in chains})
    structure_dir = paths["raw"] / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(source["structure_url"].format(pdb_id=p), structure_dir / f"{p}.cif.gz") for p in pdb_ids]
    with ThreadPoolExecutor(max_workers=int(source["download_workers"])) as pool:
        records = list(pool.map(lambda job: _fetch_structure(*job), jobs))
    failures = [r for r in records if "error" in r]
    with open(paths["raw"] / "structures_sha256.tsv", "w") as handle:
        for r in records:
            handle.write(f"{Path(r['path']).name}\t{r.get('sha256', 'ERROR ' + r.get('error', ''))}\n")
    return {
        "metadata_files": files, "pdb_entries": len(pdb_ids), "chains": len(chains),
        "structures_downloaded": sum(bool(r.get("downloaded")) for r in records),
        "structures_present": len(records) - len(failures), "structure_failures": failures,
        "structure_hashes": str(paths["raw"] / "structures_sha256.tsv"),
    }


def _safe_extract(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        for member in handle.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve()) + os.sep) and target != dest.resolve():
                raise RuntimeError(f"{archive}: member {member!r} escapes {dest}")
        handle.extractall(dest)


def download_torchdrug_zip(config: dict, paths: dict) -> dict:
    """TorchDrug/GearNet archive: EnzymeCommission/{nrPDB-EC_annot.tsv, nrPDB-EC_test.csv, <split>.zip}."""
    source = config["sources"]["torchdrug_zip"]
    td_dir = paths["raw"] / "torchdrug"
    td_dir.mkdir(parents=True, exist_ok=True)
    archive = td_dir / "EnzymeCommission.zip"
    record = _fetch(source["url"], archive)
    md5 = hashlib.md5()
    with open(archive, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            md5.update(block)
    if md5.hexdigest() != source["md5"]:
        raise RuntimeError(f"{archive}: md5 {md5.hexdigest()} != {source['md5']}")
    root = td_dir / "EnzymeCommission"
    if not root.exists():
        _safe_extract(archive, td_dir)
    index, split_ids = {}, {}
    for split in config["splits"]:
        split_dir = root / split
        if not split_dir.exists():
            _safe_extract(root / f"{split}.zip", root)
        ids = []
        for pdb_file in sorted(split_dir.glob("*.pdb")):
            chain = pdb_file.name.split("_")[0]  # TorchDrug's own id rule
            split_chain_id(chain)
            if chain in index:
                raise RuntimeError(f"chain {chain} appears twice in the TorchDrug archive")
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
    _copy_once(root / "nrPDB-EC_test.csv", paths["splits"] / "test_cutoffs.csv")
    _copy_once(root / "nrPDB-EC_annot.tsv", paths["raw"] / "annotations.tsv")
    index_path = td_dir / "file_index.json"
    if not index_path.exists():
        index_path.write_text(json.dumps(index, indent=0))
    return {"archive": record, "md5": md5.hexdigest(), "chains": {s: len(i) for s, i in split_ids.items()},
            "structure_failures": []}


def validate_metadata(config: dict, paths: dict) -> dict:
    terms, labels = read_annotations(paths["raw"] / "annotations.tsv", config["label_count"])
    counts, unlabeled = {}, []
    for split in config["splits"]:
        ids = read_split(paths["splits"] / f"{split}.txt")
        counts[split] = len(ids)
        unlabeled += [c for c in ids if c not in labels]
    if unlabeled:
        raise RuntimeError(f"{len(unlabeled)} split chains lack annotations, first {unlabeled[:5]}")
    subsets = read_test_cutoffs(paths["splits"] / "test_cutoffs.csv", config["test_cutoffs"])
    return {"terms": len(terms), "split_counts": counts, "test_cutoff_counts": {c: len(s) for c, s in subsets.items()}}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("download", "process"))
    parser.add_argument("--config", type=Path, default=TASK_YAML)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    config = load_task_config(args.config, data_root=args.data_root)
    paths = task_paths(config)
    if args.command == "process":
        splits = args.splits or config["splits"]
        started = datetime.now(timezone.utc).isoformat()
        exit_code = build_processed(config, splits=splits, workers=args.workers)
        record_process_run(
            paths["provenance"], started_utc=started, config_path=args.config, processed_dir=processed_dir(config),
            splits=splits, exit_code=exit_code,
            inputs=[paths["raw"] / "annotations.tsv", *(paths["splits"] / f"{s}.txt" for s in splits)])
        return exit_code

    paths["raw"].mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    fetch = download_deepfri_rcsb if config["source"] == "deepfri_rcsb" else download_torchdrug_zip
    result = fetch(config, paths)
    result["validation"] = validate_metadata(config, paths)
    failures = result["structure_failures"]
    total = result.get("pdb_entries") or 1
    fraction = len(failures) / total
    record = {
        "command": " ".join(sys.argv), "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "source": config["source"],
        "config": str(args.config), "config_sha256": sha256_file(args.config), "git": git_commit(),
        "structure_failure_fraction": fraction, **result,
    }
    _append_provenance(paths["provenance"], record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("structure_failures", "metadata_files")}, indent=1))
    print(f"{len(failures)} structure failures ({fraction:.2%}); first: {failures[:5]}")
    if fraction > config["qc"]["max_download_failure_fraction"]:
        print("FAILED: structure failure fraction above qc.max_download_failure_fraction")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
