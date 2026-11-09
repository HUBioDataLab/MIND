"""Download and process the Fold benchmark (data-processing CLI; run only on explicit user request).

    python -m downstream.tasks.fold.prepare download [--data-root DIR]
    python -m downstream.tasks.fold.prepare validate [--data-root DIR]
    python -m downstream.tasks.fold.prepare process  [--data-root DIR] [--splits ...] [--workers N]

``download`` fetches fold3d.zip (Zenodo 7593591; size and md5 pinned in task.yaml) into the shared
``<source_root>/raw``, extracts it and the five per-split archives, maps every listed domain to its hdf5 file
(any missing or unlisted file raises), writes ``splits/<split>.txt`` and validates labels against task.yaml.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from downstream.tasks import multiclass
from downstream.tasks.ec.prepare import _fetch
from downstream.tasks.fold import dataset
from downstream.tasks.multiclass_prepare import main as run_main, write_split_lists


def download(config: dict) -> dict:
    paths = multiclass.task_paths(config)
    source = config["sources"][config["source"]]
    paths["raw"].mkdir(parents=True, exist_ok=True)
    archive = paths["raw"] / source["archive_name"]
    record = _fetch(source["url"], archive)
    if archive.stat().st_size != int(source["bytes"]):
        raise RuntimeError(f"{archive}: {archive.stat().st_size} bytes, pinned {source['bytes']}")
    md5 = multiclass.md5_file(archive)
    if md5 != source["md5"]:
        raise RuntimeError(f"{archive}: md5 {md5} != pinned {source['md5']}")
    extract_to = paths["raw"] / source["extract_dir"]
    if not extract_to.exists():
        extract_to.mkdir()
        multiclass.safe_extract_archive(archive, extract_to)
    root = dataset.locate_root(config)
    for split, inner in source["inner_archives"].items():
        marker = root / f".{split}.extracted"
        if marker.exists():
            continue
        inner_path = root / inner
        if inner_path.is_file():
            multiclass.safe_extract_archive(inner_path, root)
        marker.write_text("ok\n")
    classes, labels, split_names = dataset.read_labels(config)
    index = multiclass.index_hdf5(root, [n for names in split_names.values() for n in names], what="fold3d")
    write_split_lists(config, split_names)
    index_path = multiclass.file_index_path(config)
    if not index_path.exists():
        index_path.write_text(json.dumps(index, indent=0))
    return {"archive": record, "md5": md5, "archive_root": str(root),
            "proteins": {s: len(n) for s, n in split_names.items()}, "classes": len(classes)}


def main(argv: Optional[list] = None) -> int:
    return run_main(argv, doc=__doc__, load_config=dataset.load_task_config, task_yaml=dataset.TASK_YAML,
                    read_labels=dataset.read_labels, download_fn=download, builder="downstream.tasks.fold.dataset")


if __name__ == "__main__":
    sys.exit(main())
