"""Download and process the Reaction benchmark (data-processing CLI; run only on explicit user request).

    python -m downstream.tasks.reaction.prepare download [--data-root DIR]
    python -m downstream.tasks.reaction.prepare validate [--data-root DIR]
    python -m downstream.tasks.reaction.prepare process  [--data-root DIR] [--splits ...] [--workers N]

``download`` (1) fetches the five label lists from the pinned IEConv_proteins commit and checks their sha256 pins,
(2) fetches the Google Drive archive of the README unless ``raw/<archive_name>`` already exists (a manual copy is
accepted), records its size and sha256, and REFUSES to continue while ``archive_sha256``/``archive_bytes`` are not
pinned in task.yaml (trust on first use: review provenance.json, pin, rerun), (3) extracts it, maps every listed
protein to its hdf5 file (any missing or unlisted file raises) and validates labels against task.yaml.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import requests

from downstream.common.registry import sha256_file
from downstream.tasks import multiclass
from downstream.tasks.ec.prepare import _fetch
from downstream.tasks.multiclass_prepare import main as run_main, write_split_lists
from downstream.tasks.reaction import dataset

TIMEOUT = 120


def _fetch_drive(url: str, dest) -> dict:
    """Stream a Google Drive file; an HTML answer (quota / confirmation page) raises and leaves nothing behind."""
    if dest.exists():
        return {"url": url, "path": str(dest), "downloaded": False}
    partial = dest.with_name(dest.name + ".partial")
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        if response.status_code != 200:
            raise RuntimeError(f"Google Drive answered HTTP {response.status_code}")
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError("Google Drive returned an HTML page instead of the archive (quota or confirmation page); "
                               f"download the archive manually and place it at {dest}, then rerun")
        with open(partial, "wb") as handle:
            for block in response.iter_content(1 << 20):
                handle.write(block)
    os.replace(partial, dest)
    return {"url": url, "path": str(dest), "downloaded": True}


def download(config: dict) -> dict:
    paths = multiclass.task_paths(config)
    source = config["sources"][config["source"]]
    lists = dataset.lists_dir(config)
    lists.mkdir(parents=True, exist_ok=True)
    list_records = {}
    for name, pinned in source["lists"].items():
        url = source["lists_url"].format(commit=source["commit"], name=name)
        record = _fetch(url, lists / name)
        if record["sha256"] != pinned:
            raise RuntimeError(f"{lists / name}: sha256 {record['sha256']} != pinned {pinned}")
        list_records[name] = record
    archive = paths["raw"] / source["archive_name"]
    record = _fetch_drive(source["archive_url"], archive)
    record.update(bytes=archive.stat().st_size, sha256=sha256_file(archive))
    if source["archive_sha256"] is None or source["archive_bytes"] is None:
        raise RuntimeError(f"UNPINNED archive {archive}: {record['bytes']} bytes, sha256 {record['sha256']}. Review it, "
                           "set archive_bytes and archive_sha256 in downstream/tasks/reaction/task.yaml, then rerun download")
    if record["bytes"] != int(source["archive_bytes"]) or record["sha256"] != source["archive_sha256"]:
        raise RuntimeError(f"{archive}: {record['bytes']} bytes / sha256 {record['sha256']} differ from the pins")
    extract_to = paths["raw"] / source["extract_dir"]
    if not extract_to.exists():
        extract_to.mkdir()
        multiclass.safe_extract_archive(archive, extract_to)
    classes, labels, split_names = dataset.read_labels(config)
    index = multiclass.index_hdf5(extract_to, [n for names in split_names.values() for n in names], what="protfunct")
    write_split_lists(config, split_names)
    index_path = multiclass.file_index_path(config)
    if not index_path.exists():
        index_path.write_text(json.dumps(index, indent=0))
    return {"lists": list_records, "archive": record,
            "proteins": {s: len(n) for s, n in split_names.items()}, "classes": len(classes)}


def main(argv: Optional[list] = None) -> int:
    return run_main(argv, doc=__doc__, load_config=dataset.load_task_config, task_yaml=dataset.TASK_YAML,
                    read_labels=dataset.read_labels, download_fn=download, builder="downstream.tasks.reaction.dataset")


if __name__ == "__main__":
    sys.exit(main())
