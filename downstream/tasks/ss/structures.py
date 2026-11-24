"""Resumable, rate-limited download of RCSB mmCIF files with a per-file provenance manifest.

``fetch_structures`` writes ``<dir>/<PDBID>.cif.gz`` and appends one JSON line per entry to ``fetch_manifest.jsonl``
(url, status, bytes, sha256, HTTP ``Last-Modified`` / ``ETag``, UTC time). A file that exists and is listed in the
manifest with the same size is not fetched again (resume); a file that exists without a manifest line is an error
(never silently trusted); a ``.partial`` file left by an interrupted run is deleted and refetched. HTTP 404 is recorded
as ``not_found`` (obsolete or withdrawn entry) and counted by the caller; other failures are retried with backoff and
then recorded as ``error``. Requests are limited to ``rate`` per second over all workers.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import requests

MANIFEST = "fetch_manifest.jsonl"
USER_AGENT = "mind-downstream-structure-fetch/1.0"


class RateLimiter:
    """At most ``rate`` acquisitions per second across threads."""

    def __init__(self, rate: float, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.interval, self.clock, self.sleep = 1.0 / rate, clock, sleep
        self.lock, self.next_time = threading.Lock(), 0.0

    def acquire(self) -> None:
        with self.lock:
            now = self.clock()
            wait = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval
        if wait:
            self.sleep(wait)


def read_manifest(directory: Path) -> Dict[str, dict]:
    path = Path(directory) / MANIFEST
    entries: Dict[str, dict] = {}
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.strip():
                try:
                    record = json.loads(line)
                    entries[record["pdb_id"]] = record  # a later line (retry) replaces an earlier one
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"{path}:{number}: unreadable manifest line: {exc}") from exc
    return entries


def _fetch_one(pdb_id: str, directory: Path, url_template: str, limiter: RateLimiter, session, retries: int,
               backoff: float, sleep) -> dict:
    url = url_template.format(pdb_id=pdb_id)
    target = directory / f"{pdb_id}.cif.gz"
    partial = target.with_name(target.name + ".partial")
    record = {"pdb_id": pdb_id, "url": url}
    for attempt in range(retries + 1):
        limiter.acquire()
        try:
            response = session.get(url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT})
            if response.status_code == 404:
                return {**record, "status": "not_found", "http_status": 404,
                        "fetched_utc": datetime.now(timezone.utc).isoformat()}
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            digest = hashlib.sha256()
            with open(partial, "wb") as handle:
                for block in response.iter_content(1 << 16):
                    handle.write(block)
                    digest.update(block)
            with gzip.open(partial, "rb") as handle:
                if not handle.read(64).startswith(b"data_"):
                    raise RuntimeError("not an mmCIF file")
            os.replace(partial, target)
            return {**record, "status": "ok", "bytes": target.stat().st_size, "sha256": digest.hexdigest(),
                    "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag"),
                    "fetched_utc": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:  # retried, then recorded: the caller counts every error
            error = f"{type(exc).__name__}: {exc}"
            if partial.exists():
                partial.unlink()
            if attempt < retries:
                sleep(backoff * 2 ** attempt)
    return {**record, "status": "error", "error": error, "fetched_utc": datetime.now(timezone.utc).isoformat()}


def fetch_structures(pdb_ids: Sequence[str], directory: Path, *, url_template: str, rate: float = 4.0, workers: int = 4,
                     retries: int = 3, backoff: float = 2.0, session=None, sleep: Callable[[float], None] = time.sleep,
                     clock: Callable[[], float] = time.monotonic) -> Dict[str, int]:
    """Fetch what is missing; returns counts ``{ok, not_found, error, already_present}`` over ``pdb_ids``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ids = sorted({i.upper() for i in pdb_ids})
    manifest = read_manifest(directory)
    for stale in directory.glob("*.partial"):
        stale.unlink()
    todo, counts = [], {"ok": 0, "not_found": 0, "error": 0, "already_present": 0}
    for pdb_id in ids:
        target = directory / f"{pdb_id}.cif.gz"
        entry = manifest.get(pdb_id)
        if target.exists():
            if entry is None or entry.get("status") != "ok" or entry.get("bytes") != target.stat().st_size:
                raise RuntimeError(f"{target} exists but the manifest has no matching ok line; move it aside and rerun")
            counts["already_present"] += 1
        elif entry is not None and entry["status"] == "not_found":
            counts["not_found"] += 1
        else:
            todo.append(pdb_id)
    session = session or requests.Session()
    limiter = RateLimiter(rate, clock=clock, sleep=sleep)
    manifest_path = directory / MANIFEST
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, open(manifest_path, "a") as manifest_handle:
        futures = [pool.submit(_fetch_one, i, directory, url_template, limiter, session, retries, backoff, sleep)
                   for i in todo]
        for future in futures:
            record = future.result()
            manifest_handle.write(json.dumps(record) + "\n")
            manifest_handle.flush()
            counts[record["status"]] += 1
            done += 1
            if done % 500 == 0:
                print(f"fetched {done}/{len(todo)} structure files", flush=True)
    return counts
