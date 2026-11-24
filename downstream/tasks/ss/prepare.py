"""Download and process the secondary-structure benchmark (data-processing CLI; run only on explicit user request).

    python -m downstream.tasks.ss.prepare download [--data-root DIR] [--records-only]
    python -m downstream.tasks.ss.prepare validate [--data-root DIR]        # read-only
    python -m downstream.tasks.ss.prepare process  [--data-root DIR] [--splits ...] [--workers N] [--max-failed-fraction F]

``download`` fetches the TAPE / PEER secondary-structure archive (size and md5 pinned in task.yaml), extracts it, converts the
five LMDB files into ``raw/records/<split>.jsonl.gz`` plus ``splits/<split>.txt`` (counts checked against TAPE Table S1, labels
checked: ss3 must equal the standard collapse of ss8), prints a census of the chain-id formats, then fetches the RCSB mmCIF file of
every PDB id into ``raw/structures`` (resumable, rate limited, one provenance line per file in ``fetch_manifest.jsonl``).
``--records-only`` stops before the structure fetch (inspect the id census and the entry count first). HTTP 404 entries are
recorded and counted (they become chains that cannot be mapped); any other fetch error makes the command exit 1, and a rerun
fetches only what is missing. ``process`` maps and builds one processed file per split (``dataset.build_processed``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from downstream.common.prepare_spec import Archive, DownloadSpec, disk_estimate
from downstream.common.provenance import append_run as _append_run, record_process_run
from downstream.common.registry import git_commit, sha256_file
from downstream.tasks import multiclass
from downstream.tasks.ec.prepare import _fetch
from downstream.tasks.ss import dataset, structures, tape_lmdb

COMPLETE_MARKER = ".complete"


def _archive(config: dict) -> Archive:
    source = config["sources"][config["source"]]
    return Archive(dataset.task_paths(config)["raw"] / source["archive_name"], source["url"], int(source["bytes"]), md5=source["md5"])


def download_spec(config: dict) -> DownloadSpec:
    """Archive and outputs of ``download`` (``ds-prepare``); the structure fetch is complete when its marker exists."""
    paths, source = dataset.task_paths(config), config["sources"][config["source"]]
    outputs = (paths["raw"] / source["extract_dir"], *(paths["records"] / f"{s}.jsonl.gz" for s in config["splits"]),
               *(paths["splits"] / f"{s}.txt" for s in config["splits"]), paths["structures"] / COMPLETE_MARKER)
    return DownloadSpec((_archive(config),), outputs, extracted_gib=disk_estimate(config, "extracted"))


def cli_args(step: str, *, task: str, workers: int, threads: int) -> list:
    return ["--workers", str(workers)] if step == "process" else []


def _id_census(records_by_split: dict) -> dict:
    """Chain-id shapes per split, so an unexpected id format is visible before any structure is fetched."""
    census = {}
    for split, records in records_by_split.items():
        parsed = [dataset.parse_chain_id(r["id"]) for r in records]
        census[split] = {"chains": len(records), "parseable": sum(p is not None for p in parsed),
                         "unparseable_examples": [r["id"] for r, p in zip(records, parsed) if p is None][:10],
                         "pdb_ids": len({p[0] for p in parsed if p is not None})}
    return census


def convert_archive(config: dict) -> dict:
    """LMDB -> records and split lists (idempotent; differing existing content raises)."""
    paths, source = dataset.task_paths(config), config["sources"][config["source"]]
    paths["records"].mkdir(parents=True, exist_ok=True)
    paths["splits"].mkdir(parents=True, exist_ok=True)
    records_by_split, duplicates = {}, {}
    for split in config["splits"]:
        lmdb_path = paths["raw"] / source["lmdb_pattern"].format(split=split)
        records = list(tape_lmdb.read_lmdb(lmdb_path))
        expected = int(config["expected_chain_counts"][split])
        if len(records) != expected:
            raise ValueError(f"{split}: {len(records)} records, expected {expected} (expected_chain_counts, TAPE Table S1)")
        seen = Counter()
        for r in records:  # the same chain twice in a split (CB513 regions): the k-th occurrence becomes <id>#k, counted below
            seen[r["id"]] += 1
            if seen[r["id"]] > 1:
                r["id"] = f"{r['id']}#{seen[r['id']]}"
        duplicates[split] = int(sum(c - 1 for c in seen.values()))
        ids = [r["id"] for r in records]
        for r in records:
            dataset.M.check_label_arrays(r["id"], r["primary"], r["ss3"], r["ss8"], r["observed"])
        dataset.write_records(paths["records"] / f"{split}.jsonl.gz", records)
        content = "\n".join(ids) + "\n"
        target = paths["splits"] / f"{split}.txt"
        if target.exists() and target.read_text() != content:
            raise FileExistsError(f"{target} exists with different content")
        target.write_text(content)
        records_by_split[split] = records
    owner = {}
    for split, records in records_by_split.items():
        for r in records:
            if r["id"] in owner:
                raise ValueError(f"chain {r['id']} is in both {owner[r['id']]} and {split}")
            owner[r["id"]] = split
    return {"chains": {s: len(r) for s, r in records_by_split.items()}, "duplicate_chain_regions_renamed": duplicates,
            "id_census": _id_census(records_by_split),
            "residues": {s: int(sum(len(x["primary"]) for x in r)) for s, r in records_by_split.items()},
            "unresolved_flagged_residues": {s: int(sum((x["observed"] == 0).sum() for x in r))
                                            for s, r in records_by_split.items()}}


def needed_pdb_ids(config: dict) -> list:
    paths = dataset.task_paths(config)
    ids = set()
    for split in config["splits"]:
        for record in dataset.read_records(paths["records"] / f"{split}.jsonl.gz"):
            parsed = dataset.parse_chain_id(record["id"])
            if parsed is not None:
                ids.add(parsed[0])
    return sorted(ids)


def download(config: dict, *, records_only: bool = False) -> dict:
    paths, source = dataset.task_paths(config), config["sources"][config["source"]]
    paths["raw"].mkdir(parents=True, exist_ok=True)
    archive = _archive(config)
    record = _fetch(archive.url, archive.path)
    if archive.path.stat().st_size != archive.bytes:
        raise RuntimeError(f"{archive.path}: {archive.path.stat().st_size} bytes, pinned {archive.bytes}")
    md5 = multiclass.md5_file(archive.path)
    if md5 != archive.md5:
        raise RuntimeError(f"{archive.path}: md5 {md5} != pinned {archive.md5}")
    extract_to = paths["raw"]
    if not (extract_to / source["extract_dir"]).exists():
        multiclass.safe_extract_archive(archive.path, extract_to)
    result = {"archive": record, "md5": md5, **convert_archive(config)}
    if records_only:
        result["structures"] = "skipped (--records-only)"
        return result
    ids = needed_pdb_ids(config)
    fetch = config["sources"]["structures"]
    counts = structures.fetch_structures(ids, paths["structures"], url_template=fetch["url_template"],
                                         rate=float(fetch["rate_per_second"]), workers=int(fetch["workers"]),
                                         retries=int(fetch["retries"]))
    manifest = structures.read_manifest(paths["structures"])
    result["structures"] = {"entries": len(ids), **counts,
                            "not_found_ids": sorted(i for i in ids if manifest[i]["status"] == "not_found"),
                            "bytes": sum(m.get("bytes", 0) for m in manifest.values())}
    if counts["error"]:
        raise RuntimeError(f"{counts['error']} structure fetches failed (see fetch_manifest.jsonl); rerun to fetch them again")
    (paths["structures"] / COMPLETE_MARKER).write_text(f"{len(ids)} entries accounted for\n")
    return result


def validate(config: dict) -> dict:
    """Read-only: records, split lists and (when present) the structure manifest must agree with task.yaml."""
    paths = dataset.task_paths(config)
    result = {"splits": {}}
    for split in config["splits"]:
        records = dataset.read_records(paths["records"] / f"{split}.jsonl.gz")
        ids = dataset.read_split_ids(config, split)
        if [r["id"] for r in records] != ids:
            raise ValueError(f"{split}: splits/{split}.txt differs from records/{split}.jsonl.gz")
        if len(ids) != int(config["expected_chain_counts"][split]):
            raise ValueError(f"{split}: {len(ids)} chains, expected {config['expected_chain_counts'][split]}")
        result["splits"][split] = len(ids)
    result["id_census"] = _id_census({s: dataset.read_records(paths["records"] / f"{s}.jsonl.gz") for s in config["splits"]})
    ids = needed_pdb_ids(config)
    manifest = structures.read_manifest(paths["structures"]) if paths["structures"].is_dir() else {}
    status = Counter(manifest[i]["status"] if i in manifest else "not_fetched" for i in ids)
    result["structures"] = {"entries": len(ids), **status}
    return result


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("download", "validate", "process"))
    parser.add_argument("--config", type=Path, default=dataset.TASK_YAML)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--records-only", action="store_true", help="download: stop before fetching the RCSB structures")
    parser.add_argument("--max-failed-fraction", type=float, default=None, help="process: override qc.max_failed_chain_fraction")
    args = parser.parse_args(argv)
    config = dataset.load_task_config(args.config, data_root=args.data_root)
    paths = dataset.task_paths(config)
    started = datetime.now(timezone.utc).isoformat()
    if args.command == "process":
        splits = args.splits or config["splits"]
        exit_code = dataset.build_processed(config, splits=splits, workers=args.workers, max_failed_fraction=args.max_failed_fraction)
        record_process_run(paths["provenance"], started_utc=started, config_path=args.config,
                           processed_dir=dataset.processed_dir(config), splits=splits, exit_code=exit_code,
                           inputs=[paths["records"] / f"{s}.jsonl.gz" for s in splits] + [paths["structures"] / structures.MANIFEST],
                           extra={"max_failed_fraction": args.max_failed_fraction})
        return exit_code
    result = download(config, records_only=args.records_only) if args.command == "download" else validate(config)
    record = {"command": " ".join(sys.argv), "kind": args.command, "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(), "source": config["source"], "config": str(args.config),
              "config_sha256": sha256_file(args.config), "git": git_commit(), **result}
    paths["root"].mkdir(parents=True, exist_ok=True)
    _append_run(paths["provenance"], record)
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
