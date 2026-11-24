"""Declarative description of what a downstream task's ``download`` step produces (read by ``downstream.prepare_all``).

A task's ``prepare`` module (``downstream/tasks/<family>/prepare.py``) may define two hooks next to its CLI:

``download_spec(config) -> DownloadSpec``
    The pinned archives the ``download`` command fetches and the paths it leaves behind. ``prepare_all`` uses it to
    share one download between tasks, to skip a finished download, to recognise a partial one, and to size the run.
``cli_args(step, *, task, workers, threads) -> list``
    The extra command-line arguments of ``download`` / ``process`` for this task (for example ``--task`` or
    ``--workers``); without the hook a step is run with none.

A prepare module without these hooks is still run (its CLI is idempotent on present files) but its sizes are unknown
and a finished download is not recognised.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

GIB = 1024 ** 3
_BLOCK = 1 << 24


@dataclass(frozen=True)
class Archive:
    """One pinned source file. ``bytes`` and at least one of ``md5`` / ``sha256`` are expected for a pinned archive."""

    path: Path
    url: str
    bytes: Optional[int] = None
    md5: Optional[str] = None
    sha256: Optional[str] = None

    @property
    def partial(self) -> Path:
        """The temporary name the download CLIs write to before the final rename."""
        return self.path.with_name(self.path.name + ".partial")


@dataclass(frozen=True)
class DownloadSpec:
    """Archives plus the outputs (extracted trees, split lists, indexes) a finished ``download`` leaves behind.

    ``unavailable`` is a reason string when the source cannot be fetched (the step is then not run, and a request
    that names the task fails). ``extracted_gib`` is the measured size of the extracted tree, for the disk estimate.
    """

    archives: Tuple[Archive, ...]
    outputs: Tuple[Path, ...]
    unavailable: Optional[str] = None
    extracted_gib: Optional[float] = None


def spec_key(spec: DownloadSpec) -> Tuple[str, ...]:
    """Identity of a download: tasks whose archives are the same files share one download step."""
    return tuple(sorted(str(archive.path) for archive in spec.archives))


def disk_estimate(config: dict, key: str) -> Optional[float]:
    """``disk_estimate_gib.<key>`` of a task config (measured with ``du`` on a finished build), else None."""
    value = (config.get("disk_estimate_gib") or {}).get(key)
    return None if value is None else float(value)


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(archive: Archive) -> str:
    """Check size and every pinned checksum of a present archive; raises ``RuntimeError`` on any difference.

    Returns a one-line description of what was verified. An archive with no pin at all raises: it cannot be verified.
    """
    if not archive.path.is_file():
        raise RuntimeError(f"{archive.path} is missing")
    size = archive.path.stat().st_size
    if archive.bytes is not None and size != archive.bytes:
        raise RuntimeError(f"{archive.path}: {size} bytes, pinned {archive.bytes}")
    checked = []
    for algorithm in ("md5", "sha256"):
        pinned = getattr(archive, algorithm)
        if pinned is None:
            continue
        found = _digest(archive.path, algorithm)
        if found != pinned:
            raise RuntimeError(f"{archive.path}: {algorithm} {found} != pinned {pinned}")
        checked.append(algorithm)
    if not checked:
        raise RuntimeError(f"{archive.path} has no pinned checksum; it cannot be verified")
    return f"{archive.path.name}: {size} bytes, {'+'.join(checked)} match"
