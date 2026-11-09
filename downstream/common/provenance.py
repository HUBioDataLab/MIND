"""Shared ``provenance.json`` helpers of the downstream prepare CLIs (EC, GO, Fold, Reaction).

``provenance.json`` is a JSON object whose ``runs`` list holds one record per download/validate/process
run; other keys (for example the leakage records) are preserved. ``record_process_run`` is the single
place that records a ``process`` run: command, config hash, git state, input and output sha256 and the
per-split QC summaries, so a processed directory can be traced back without hand-written entries.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

from downstream.common.registry import git_commit, sha256_file


def append_run(path: Path, record: dict) -> None:
    """Append ``record`` to ``path``'s ``runs`` list (atomic replace; other top-level keys are kept)."""
    path = Path(path)
    existing = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(existing, dict):
        raise ValueError(f"{path}: expected a JSON object")
    existing.setdefault("runs", []).append(record)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(existing, indent=1))
    os.replace(partial, path)


def process_run_record(
    *, started_utc: str, config_path: Path, processed_dir: Path, splits: Sequence[str], exit_code: int,
    inputs: Sequence[Path], command: Optional[str] = None, extra: Optional[Mapping] = None,
) -> dict:
    """Build the ``process`` record of one run.

    ``outputs_sha256`` covers the ``<split>.pt`` files that exist (a split that failed QC has none, which the
    ``exit_code`` and the ``qc`` summary show) and each split's ``qc_<split>.tsv``. A missing ``qc_<split>.json``
    raises: every processed split must have written one before the run ends.
    """
    processed_dir = Path(processed_dir)
    outputs, qc = {}, {}
    for split in splits:
        qc_json = processed_dir / f"qc_{split}.json"
        if not qc_json.is_file():
            raise FileNotFoundError(f"{qc_json} was not written; the process run left no QC report for {split!r}")
        qc[split] = json.loads(qc_json.read_text())
        for name in (f"{split}.pt", f"qc_{split}.tsv"):
            if (processed_dir / name).is_file():
                outputs[name] = sha256_file(processed_dir / name)
    missing = [str(p) for p in inputs if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"process inputs not found: {missing}")
    return {
        "kind": "process", "command": command or " ".join(sys.argv), "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "config": str(config_path), "config_sha256": sha256_file(Path(config_path)),
        "processed_dir": str(processed_dir), "splits": list(splits), "exit_code": int(exit_code),
        "inputs_sha256": {str(p): sha256_file(Path(p)) for p in inputs},
        "outputs_sha256": outputs, "qc": qc, **(dict(extra) if extra else {}),
    }


def record_process_run(provenance_path: Path, **kwargs) -> dict:
    """Write the process record to ``provenance_path`` and return it (see ``process_run_record``)."""
    provenance_path = Path(provenance_path)
    record = process_run_record(**kwargs)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    append_run(provenance_path, record)
    return record
