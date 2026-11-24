#!/usr/bin/env python3
"""Build the COCONUT pretraining chunks from the raw 3D SDF in one recorded command.

Raw SDF (or the official zip) -> record filters -> molecule-level train/val/test
assignment -> chunk plan -> adapter conversion -> stored-edge chunks, followed by a
completeness check that fails the run when the books do not close.

Records and molecules. The COCONUT 3D SDF holds one record per conformer, several
conformers per molecule, ordered by molecule size. A split is assigned per molecule
(SHA-256 of ``salt:key``, ``--molecule-key``), so all conformers of a molecule land in
one split; the assignment never depends on the record position. Inside each split the
records are shuffled with ``--shuffle-seed`` before chunking, so a chunk mixes sizes, and
are cut into ``ceil(n / --chunk-size)`` balanced chunks.

Filters (``FILTER_VERSION``, evaluated in this order, each rejected record is counted
under its first reason): fewer than ``--min-heavy-atoms`` heavy atoms; an element
outside ``--allowed-elements`` (hydrogen, deuterium and tritium count as H); all heavy
atoms with z = 0 (a flat 2D depiction inside a "3D" file). A record that cannot be
parsed (not V2000, missing counts, missing molecule key) aborts the build with its
byte offset; it is never filtered silently.

Conversion failures. A record that passes the filters but fails the adapter (PS_300
tokenizer, no conformer, graph conversion) is dropped from the chunk and recorded with
its id and reason in the chunk metadata and in ``<dest>_failures.csv``. A chunk aborts
the run when its failure fraction exceeds ``--max-failure-fraction``.

Books. ``raw records = rejected + failed + built`` must hold exactly, the number of
chunk directories must equal the plan, and the built plus failed records of every chunk
must be exactly its planned records. ``--verify-only`` re-runs that check from the files.

Layout. Chunks are written as ``<dest>_<split>_chunk_NNN/processed/`` in the stored-edge
layout of the UniMol/COCONUT route (``cache_to_pyg``: radius edges, ``bond_edge_index``),
plus the metadata sidecar. Writes are atomic (``.partial`` then rename), existing chunks
are never rewritten, and a restart resumes only when the recorded plan and parameters
are identical.

Example (detached, survives logout):
    setsid nohup python -u -m data_loading.build_coconut --input <coconut.sdf|coconut.zip> \\
        --dest-prefix <MIND_DATA_ROOT>/metabolites/coconut_full --workers 32 > build.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import random
import shutil
import sys
import time
import zipfile
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

# CPU-only tool: hide GPUs so forked workers never initialize CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data.collate import collate

from data_loading.build_edge_free_chunks import (
    SPLITS,
    _format_seconds,
    _git_sha,
    _sha256,
    assign_split,
    chunk_directory,
)
from data_loading.create_chunk_metadata import assign_bucket

BUILDER = "data_loading.build_coconut"
FILTER_VERSION = "coconut-filter-v1"
LAYOUT = "stored_edge_legacy_v1"
DEFAULT_ALLOWED_ELEMENTS = ("H", "C", "O", "N", "S", "Cl", "Br", "P", "I", "Na", "F")
DEFAULT_MIN_HEAVY_ATOMS = 10
HYDROGEN_SYMBOLS = (b"H", b"D", b"T")
EDGE_CUTOFF = 5.0           # Angstrom, radius graph stored in the chunk
EDGE_MAX_NEIGHBORS = 64
MAX_CHUNKS_PER_SPLIT = 1000  # chunk names are zero-padded to three digits

KEPT = 0
REASON_NAMES = {1: "too_few_heavy_atoms", 2: "disallowed_element", 3: "flat_2d_coordinates"}
REJECTED_SPLIT = 255

PLAN_COLUMNS = ["record", "split", "chunk", "position", "molecule", "conformer"]


class BuildVerificationError(RuntimeError):
    """The finished build does not match its plan or its record accounting."""


# ---------------------------------------------------------------------------
# Record parsing and filters (pure functions on one SDF record)
# ---------------------------------------------------------------------------

def _tag_value(record: bytes, name: bytes) -> Optional[bytes]:
    """Value line of the SD data item ``<name>`` (searched after ``M  END``), or None."""
    start = record.find(b"M  END")
    marker = b"<" + name + b">"
    at = record.find(marker, max(start, 0))
    if at < 0:
        return None
    line_end = record.find(b"\n", at)
    if line_end < 0:
        return None
    value_end = record.find(b"\n", line_end + 1)
    value = record[line_end + 1:value_end if value_end >= 0 else len(record)].strip()
    return value or None


def classify_record(
    record: bytes,
    *,
    min_heavy_atoms: int,
    allowed_elements: Sequence[str],
    molecule_key: str,
) -> tuple[int, int, Optional[bytes], int]:
    """Return ``(heavy_atoms, reason, key, conformer_id)`` for one V2000 record.

    ``reason`` is ``KEPT`` (0) or a key of ``REASON_NAMES``. ``key`` is the molecule key
    (bytes) for kept records and None for rejected ones; ``conformer_id`` is -1 when the
    record has no ``CONFORMER_ID`` item. Raises ``ValueError`` on a malformed record.
    """
    head = record.split(b"\n", 4)
    if len(head) < 5:
        raise ValueError("record has no atom block")
    counts = head[3]
    if b"V2000" not in counts:
        raise ValueError(f"only V2000 mol blocks are supported, counts line: {counts[:40]!r}")
    try:
        num_atoms = int(counts[0:3])
    except ValueError:
        raise ValueError(f"unreadable counts line: {counts[:40]!r}") from None
    atom_lines = head[4].split(b"\n", num_atoms)[:num_atoms]
    if len(atom_lines) != num_atoms:
        raise ValueError(f"atom block shorter than the {num_atoms} atoms of the counts line")
    tokens = [line[31:34].strip() for line in atom_lines]
    if b"" in tokens:
        raise ValueError("atom line without an element symbol")
    heavy = num_atoms - sum(tokens.count(symbol) for symbol in HYDROGEN_SYMBOLS)
    if heavy < min_heavy_atoms:
        return heavy, 1, None, -1
    allowed = {element.encode() for element in allowed_elements}
    symbols = {token for token in tokens if token not in HYDROGEN_SYMBOLS}
    if not symbols <= allowed:
        return heavy, 2, None, -1
    if heavy > 0 and all(
        float(line[20:30]) == 0.0
        for line, token in zip(atom_lines, tokens) if token not in HYDROGEN_SYMBOLS
    ):
        return heavy, 3, None, -1
    key_tag = b"SMILES" if molecule_key == "smiles" else b"IDENTIFIER"
    key = _tag_value(record, key_tag)
    if key is None:
        raise ValueError(f"record has no <{key_tag.decode()}> item (--molecule-key {molecule_key})")
    conformer = _tag_value(record, b"CONFORMER_ID")
    return heavy, KEPT, key, int(conformer) if conformer is not None else -1


def molecule_hash(key: bytes) -> int:
    """64-bit identity of a molecule key (used for counting and the leak check)."""
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def mol_block_text(record: bytes) -> str:
    """The mol block of a record, up to and including ``M  END``."""
    end = record.find(b"M  END")
    if end < 0:
        raise ValueError("record has no 'M  END' line")
    return record[:end + 6].decode("ascii")


# ---------------------------------------------------------------------------
# Scan: one streaming pass over the SDF, parallel over byte ranges
# ---------------------------------------------------------------------------

def _next_record_start(handle, position: int, size: int) -> int:
    """Offset of the first record that starts after ``position`` (file size when none)."""
    handle.seek(max(position - 1, 0))
    offset = max(position - 1, 0)
    carry = b""
    while True:
        block = handle.read(4 * 1024 * 1024)
        if not block:
            return size
        data = carry + block
        found = data.find(b"\n$$$$")
        if found >= 0:
            newline = data.find(b"\n", found + 1)
            if newline >= 0:
                return offset - len(carry) + newline + 1
            rest = handle.read(1024)
            line_end = rest.find(b"\n")
            if line_end < 0:
                return size
            return offset + len(block) + line_end + 1
        carry = data[-8:]
        offset += len(block)


def plan_byte_ranges(path: Path, range_bytes: int) -> list[tuple[int, int]]:
    """Split the file into ranges that start and end on record boundaries."""
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"{path} is empty")
    starts = [0]
    with open(path, "rb") as handle:
        guess = range_bytes
        while guess < size:
            boundary = _next_record_start(handle, guess, size)
            if boundary >= size:
                break
            if boundary > starts[-1]:
                starts.append(boundary)
            guess = max(guess + range_bytes, boundary + 1)
    return list(zip(starts, starts[1:] + [size]))


def scan_range(task: tuple) -> dict:
    """Classify every record of one byte range; returns column arrays in file order."""
    (path, start, end, salt, fractions, molecule_key, min_heavy_atoms, allowed_elements) = task
    with open(path, "rb") as handle:
        handle.seek(start)
        data = handle.read(end - start)
    offsets, lengths, heavies, reasons, molecules, conformers, splits = [], [], [], [], [], [], []
    split_of_key: dict[bytes, tuple[int, int]] = {}
    position = 0
    while True:
        found = data.find(b"\n$$$$", position)
        if found < 0:
            break
        record = data[position:found + 1]
        newline = data.find(b"\n", found + 1)
        following = len(data) if newline < 0 else newline + 1
        try:
            heavy, reason, key, conformer = classify_record(
                record, min_heavy_atoms=min_heavy_atoms,
                allowed_elements=allowed_elements, molecule_key=molecule_key,
            )
        except ValueError as exc:
            raise ValueError(f"malformed SDF record at byte {start + position}: {exc}") from None
        if key is None:
            split_code, molecule = REJECTED_SPLIT, 0
        else:
            if key not in split_of_key:
                label = assign_split(key.decode("utf-8"), salt, fractions)
                split_of_key[key] = (SPLITS.index(label), molecule_hash(key))
            split_code, molecule = split_of_key[key]
        offsets.append(start + position)
        lengths.append(following - position)
        heavies.append(heavy)
        reasons.append(reason)
        molecules.append(molecule)
        conformers.append(conformer)
        splits.append(split_code)
        position = following
    if data[position:].strip():
        raise ValueError(f"unterminated SDF record after byte {start + position}")
    return {
        "offset": np.array(offsets, dtype=np.int64), "length": np.array(lengths, dtype=np.int32),
        "heavy": np.array(heavies, dtype=np.int16), "reason": np.array(reasons, dtype=np.uint8),
        "molecule": np.array(molecules, dtype=np.uint64), "conformer": np.array(conformers, dtype=np.int32),
        "split": np.array(splits, dtype=np.uint8),
    }


@dataclass
class ScanResult:
    offset: np.ndarray
    length: np.ndarray
    heavy: np.ndarray
    reason: np.ndarray
    molecule: np.ndarray
    conformer: np.ndarray
    split: np.ndarray
    scanned_bytes: int

    @property
    def num_records(self) -> int:
        return int(self.offset.size)


def ordered_bounded_map(executor, function, items: Iterable, window: int) -> Iterator:
    """Yield ``function(item)`` in input order with at most ``window`` tasks in flight."""
    pending: deque = deque()
    try:
        for item in items:
            pending.append(executor.submit(function, item))
            if len(pending) >= window:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        for future in pending:
            future.cancel()


def scan_sdf(
    path: Path, *, salt: str, fractions: Sequence[float], molecule_key: str,
    min_heavy_atoms: int, allowed_elements: Sequence[str], workers: int,
    range_bytes: int, max_records: Optional[int],
) -> ScanResult:
    ranges = plan_byte_ranges(path, range_bytes)
    tasks = [
        (str(path), start, end, salt, tuple(fractions), molecule_key, min_heavy_atoms, tuple(allowed_elements))
        for start, end in ranges
    ]
    parts: list[dict] = []
    count = 0
    scanned_bytes = 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = ordered_bounded_map(executor, scan_range, tasks, window=2 * workers)
        for (start, end), part in zip(ranges, results):
            parts.append(part)
            count += part["offset"].size
            scanned_bytes = end
            print(
                f"  scanned {scanned_bytes / 1e9:6.2f}/{path.stat().st_size / 1e9:.2f} GB, "
                f"{count:,} records, {_format_seconds(time.monotonic() - started)}",
                flush=True,
            )
            if max_records is not None and count >= max_records:
                results.close()
                break
    columns = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    if max_records is not None and count > max_records:
        columns = {name: values[:max_records] for name, values in columns.items()}
        scanned_bytes = int(columns["offset"][-1] + columns["length"][-1])
    return ScanResult(scanned_bytes=scanned_bytes, **columns)


def reason_counts(scan: ScanResult) -> dict[str, int]:
    counts = Counter(scan.reason.tolist())
    return {name: int(counts.get(code, 0)) for code, name in REASON_NAMES.items()}


# ---------------------------------------------------------------------------
# Plan: molecule-level splits, shuffled chunks
# ---------------------------------------------------------------------------

def build_plan(scan: ScanResult, *, chunk_size: int, shuffle_seed: int) -> pd.DataFrame:
    """Rows ``record, split, chunk, position, molecule, conformer`` for every kept record.

    Each split is cut into ``ceil(n / chunk_size)`` balanced chunks (``chunk_size`` is an
    upper bound).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    frames = []
    for split_index, split in enumerate(SPLITS):
        members = np.flatnonzero(scan.split == split_index).tolist()
        random.Random(f"{shuffle_seed}:{split}").shuffle(members)
        if (len(members) + chunk_size - 1) // chunk_size > MAX_CHUNKS_PER_SPLIT:
            raise ValueError(
                f"{split}: {len(members)} records need more than {MAX_CHUNKS_PER_SPLIT} chunks of "
                f"{chunk_size}; chunk names are zero-padded to three digits, raise --chunk-size"
            )
        order = np.array(members, dtype=np.int64)
        rank = np.arange(order.size, dtype=np.int64)
        # Balanced chunks: ceil(n / chunk_size) chunks whose sizes differ by at most one,
        # so there is no small tail chunk whose failure fraction is mere noise.
        num_chunks = max(-(-order.size // chunk_size), 1)
        chunk = rank * num_chunks // max(order.size, 1)
        position = rank - np.searchsorted(chunk, chunk, side="left")
        frames.append(pd.DataFrame({
            "record": order,
            "split": split,
            "chunk": chunk,
            "position": position,
            "molecule": [f"{value:016x}" for value in scan.molecule[order]],
            "conformer": scan.conformer[order],
        }))
    plan = pd.concat(frames, ignore_index=True)
    check_no_molecule_in_two_splits(plan)
    return plan[PLAN_COLUMNS]


def check_no_molecule_in_two_splits(plan: pd.DataFrame) -> None:
    pairs = plan[["molecule", "split"]].drop_duplicates()
    leaked = pairs[pairs["molecule"].duplicated(keep=False)]
    if len(leaked):
        raise BuildVerificationError(
            f"{leaked['molecule'].nunique()} molecules have conformers in more than one split; "
            f"first: {leaked['molecule'].iloc[0]}"
        )


def plan_csv_gz_bytes(plan: pd.DataFrame) -> bytes:
    text = plan.to_csv(index=False).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(text)
    return buffer.getvalue()


def write_once_or_match(path: Path, content: bytes) -> None:
    """Write ``path`` atomically; if it exists it must hold exactly ``content``."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"{path} differs from the recomputed content; refusing to mix builds")
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)


def rejected_csv_bytes(scan: ScanResult) -> bytes:
    rejected = np.flatnonzero(scan.reason != KEPT)
    frame = pd.DataFrame({
        "record": rejected,
        "reason": [REASON_NAMES[code] for code in scan.reason[rejected].tolist()],
        "heavy_atoms": scan.heavy[rejected],
    })
    return frame.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Chunk worker: SDF records -> adapter -> stored-edge chunk
# ---------------------------------------------------------------------------

_WORKER: Optional[tuple] = None


def _init_chunk_worker(sdf_path: str) -> None:
    global _WORKER
    torch.set_num_threads(1)
    from data_loading.adapters.coconut_adapter import COCONUTAdapter
    from data_loading.cache_to_pyg import OptimizedUniversalDataset

    # Config-only converter, as in cache_to_pyg._init_conversion_worker.
    converter = object.__new__(OptimizedUniversalDataset)
    converter.molecule_max_atoms = None
    converter.cutoff_distance = EDGE_CUTOFF
    converter.max_neighbors = EDGE_MAX_NEIGHBORS
    converter.pre_filter = None
    converter.pre_transform = None
    _WORKER = (COCONUTAdapter(), converter, sdf_path)


def convert_record(record_index: int, record: bytes, adapter, converter):
    """Return ``(Data, None)`` or ``(None, reason)`` for one raw record."""
    from rdkit import Chem

    try:
        mol = Chem.MolFromMolBlock(mol_block_text(record), sanitize=False, removeHs=False)
    except ValueError as exc:
        return None, f"mol block: {exc}"
    if mol is None:
        return None, "RDKit could not parse the mol block"
    molecule = adapter.convert_to_universal(
        {"mol": mol, "id": f"coconut_{record_index}", "properties": {}}
    )
    if not molecule.blocks:
        return None, adapter.last_failure_reason or "adapter returned no blocks"
    try:
        return converter._create_pyg_data_object(molecule), None
    except Exception as exc:  # recorded per record and bounded by --max-failure-fraction
        return None, f"graph conversion: {type(exc).__name__}: {exc}"[:200]


def write_stored_edge_chunk(samples: list, chunk_dir: Path, *, record_ids: list[int],
                            extra_metadata: dict) -> dict:
    """Collate samples and write ``chunk_dir/processed/{pt,metadata}`` atomically."""
    if chunk_dir.exists():
        raise FileExistsError(f"refusing to overwrite {chunk_dir}")
    if not samples:
        raise ValueError(f"{chunk_dir}: no samples to write")
    partial = chunk_dir.with_name(chunk_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    processed = partial / "processed"
    processed.mkdir(parents=True)
    stem = chunk_dir.name
    data, slices, _ = collate(samples[0].__class__, data_list=samples, increment=False, add_batch=False)
    pt_path = processed / f"{stem}.pt"
    torch.save((data, slices), pt_path)

    records = []
    for local_idx, (sample, record) in enumerate(zip(samples, record_ids)):
        num_atoms = int(sample.z.numel())
        records.append({
            "idx": local_idx, "num_atoms": num_atoms, "num_edges": int(sample.edge_index.size(1)),
            "bucket": assign_bucket(num_atoms), "mol_id": str(sample.mol_id), "record": int(record),
        })
    metadata = {
        "chunk_file": pt_path.name,
        "num_samples": len(samples),
        "dataset_type": str(samples[0].dataset_type),
        "total_atoms": sum(entry["num_atoms"] for entry in records),
        "total_edges": sum(entry["num_edges"] for entry in records),
        "pt_bytes": pt_path.stat().st_size,
        "samples": records,
        **extra_metadata,
    }
    with open(processed / f"{stem}_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=1)

    reloaded_data, reloaded_slices = torch.load(pt_path, weights_only=False, mmap=True)
    if len(reloaded_slices["pos"]) - 1 != len(samples) or len(reloaded_data.mol_id) != len(samples):
        raise ValueError(f"{partial}: written chunk failed reload verification")
    os.replace(partial, chunk_dir)
    return {
        "samples": len(samples),
        "bytes": sum(path.stat().st_size for path in (chunk_dir / "processed").iterdir()),
    }


def build_chunk(task: tuple) -> dict:
    """Convert and write one planned chunk; returns its counts and failures."""
    chunk_dir_text, split, chunk, rows, max_failure_fraction, source = task
    adapter, converter, sdf_path = _WORKER
    chunk_dir = Path(chunk_dir_text)
    started = time.monotonic()
    samples, built_records, failures = [], [], []
    with open(sdf_path, "rb") as handle:
        # Plan order is shuffled; read forward through the file, convert in plan order.
        raw = {}
        for position, record_index, offset, length in sorted(rows, key=lambda row: row[2]):
            handle.seek(offset)
            raw[position] = (record_index, handle.read(length))
    for position in sorted(raw):
        record_index, record = raw[position]
        sample, reason = convert_record(record_index, record, adapter, converter)
        if sample is None:
            failures.append({"record": record_index, "id": f"coconut_{record_index}", "reason": reason})
        else:
            samples.append(sample)
            built_records.append(record_index)
    if len(failures) > max_failure_fraction * len(rows):
        raise RuntimeError(
            f"{chunk_dir.name}: {len(failures)}/{len(rows)} records failed conversion "
            f"(> {max_failure_fraction:.4%}); first: {failures[:3]}"
        )
    written = write_stored_edge_chunk(
        samples, chunk_dir, record_ids=built_records,
        extra_metadata={
            "split": split, "chunk": chunk, "planned_records": len(rows), "failures": failures,
            "coconut_build": dict(source, layout=LAYOUT),
        },
    )
    return {
        "split": split, "chunk": chunk, "planned": len(rows), "built": len(samples),
        "failed": len(failures), "bytes": written["bytes"], "seconds": time.monotonic() - started,
    }


def chunk_tasks(plan: pd.DataFrame, scan: ScanResult, dest_prefix: Path, *,
                max_failure_fraction: float, source: dict, skip_existing: bool = True) -> list[tuple]:
    tasks = []
    for (split, chunk), group in plan.groupby(["split", "chunk"], sort=True):
        chunk_dir = chunk_directory(dest_prefix, split, int(chunk))
        if skip_existing and chunk_dir.exists():
            continue
        group = group.sort_values("position")
        rows = [
            (int(position), int(record), int(scan.offset[record]), int(scan.length[record]))
            for position, record in zip(group["position"], group["record"])
        ]
        tasks.append((str(chunk_dir), split, int(chunk), rows, max_failure_fraction, source))
    return tasks


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------

def _chunk_files(chunk_dir: Path) -> tuple[Path, Path]:
    processed = chunk_dir / "processed"
    pts = [path for path in processed.glob("*.pt") if not path.name.startswith("pre_")]
    if len(pts) != 1:
        raise BuildVerificationError(f"{chunk_dir}: expected one .pt file, found {len(pts)}")
    sidecar = processed / f"{pts[0].stem}_metadata.json"
    if not sidecar.exists():
        raise BuildVerificationError(f"{chunk_dir}: missing metadata sidecar {sidecar.name}")
    return pts[0], sidecar


def verify_build(dest_prefix: Path, *, max_failure_fraction: Optional[float] = None) -> dict:
    """Check the finished build against its plan and record books; raise on any gap.

    Reads only files: the manifest (raw count, parameters), the plan, the rejected list and
    every chunk directory with its sidecar. Returns the reconciliation report.
    """
    manifest_path = Path(f"{dest_prefix}_build_manifest.json")
    if not manifest_path.exists():
        raise BuildVerificationError(f"{manifest_path} not found")
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []

    plan = pd.read_csv(Path(f"{dest_prefix}_plan.csv.gz"), dtype={"molecule": str})
    rejected = pd.read_csv(Path(f"{dest_prefix}_rejected.csv"))
    raw_records = int(manifest["counts"]["raw_records"])
    check_no_molecule_in_two_splits(plan)
    if len(plan) + len(rejected) != raw_records:
        errors.append(
            f"plan ({len(plan)}) + rejected ({len(rejected)}) != raw records ({raw_records})"
        )
    if plan["record"].duplicated().any() or set(plan["record"]) & set(rejected["record"]):
        errors.append("a raw record appears more than once in the plan and the rejected list")
    rejected_by_reason = {
        reason: int(count) for reason, count in rejected["reason"].value_counts().items()
    }
    for reason in REASON_NAMES.values():
        if rejected_by_reason.get(reason, 0) != manifest["counts"]["rejected_by_reason"][reason]:
            errors.append(f"rejected count for {reason} differs from the manifest")

    planned = {
        (split, int(chunk)): set(map(int, group["record"]))
        for (split, chunk), group in plan.groupby(["split", "chunk"])
    }
    prefix_name = Path(dest_prefix).name
    present = {
        path.name for split in SPLITS
        for path in Path(dest_prefix).parent.glob(f"{prefix_name}_{split}_chunk_*")
    }
    expected_dirs = {chunk_directory(dest_prefix, split, chunk).name for split, chunk in planned}
    for name in sorted(expected_dirs - present):
        errors.append(f"planned chunk directory missing: {name}")
    for name in sorted(present - expected_dirs):
        errors.append(f"unplanned or unfinished chunk directory: {name}")

    per_split = {split: {"planned_records": 0, "chunks_planned": 0, "chunks_built": 0,
                         "built": 0, "failed": 0} for split in SPLITS}
    failures_rows = []
    pt_bytes_total = 0
    for (split, chunk), records in sorted(planned.items()):
        per_split[split]["planned_records"] += len(records)
        per_split[split]["chunks_planned"] += 1
        chunk_dir = chunk_directory(dest_prefix, split, chunk)
        if not chunk_dir.exists():
            continue
        try:
            pt_path, sidecar = _chunk_files(chunk_dir)
        except BuildVerificationError as exc:
            errors.append(str(exc))
            continue
        metadata = json.loads(sidecar.read_text())
        per_split[split]["chunks_built"] += 1
        built = {int(entry["record"]) for entry in metadata["samples"]}
        failed = {int(entry["record"]) for entry in metadata["failures"]}
        if metadata["split"] != split or metadata["chunk"] != chunk:
            errors.append(f"{chunk_dir.name}: sidecar split/chunk does not match its directory")
        if built & failed or (built | failed) != records:
            errors.append(
                f"{chunk_dir.name}: built {len(built)} + failed {len(failed)} records do not equal "
                f"the {len(records)} planned records"
            )
        if metadata["num_samples"] != len(metadata["samples"]) or metadata["num_samples"] != len(built):
            errors.append(f"{chunk_dir.name}: sidecar sample count is inconsistent")
        if pt_path.stat().st_size != metadata["pt_bytes"]:
            errors.append(f"{chunk_dir.name}: .pt size differs from the size recorded at write time")
        pt_bytes_total += pt_path.stat().st_size
        per_split[split]["built"] += len(built)
        per_split[split]["failed"] += len(failed)
        failures_rows.extend(
            {"split": split, "chunk": chunk, **entry} for entry in metadata["failures"]
        )

    built_total = sum(entry["built"] for entry in per_split.values())
    failed_total = sum(entry["failed"] for entry in per_split.values())
    chunks_planned = sum(entry["chunks_planned"] for entry in per_split.values())
    chunks_built = sum(entry["chunks_built"] for entry in per_split.values())
    if chunks_built != chunks_planned:
        errors.append(f"built chunks ({chunks_built}) != planned chunks ({chunks_planned})")
    if raw_records != len(rejected) + failed_total + built_total:
        errors.append(
            f"raw ({raw_records}) != rejected ({len(rejected)}) + failed ({failed_total}) "
            f"+ built ({built_total})"
        )
    limit = max_failure_fraction if max_failure_fraction is not None else \
        manifest["parameters"]["max_failure_fraction"]
    if len(plan) and failed_total > limit * len(plan):
        errors.append(f"{failed_total} failures exceed {limit:.4%} of the {len(plan)} planned records")
    if errors:
        raise BuildVerificationError(
            f"{len(errors)} problems in the COCONUT build {dest_prefix}:\n  " + "\n  ".join(errors[:50])
        )

    molecules = plan.groupby("split")["molecule"].nunique()
    for split in SPLITS:
        per_split[split]["unique_molecules"] = int(molecules.get(split, 0))
    return {
        "raw_records": raw_records, "rejected": int(len(rejected)),
        "rejected_by_reason": {reason: rejected_by_reason.get(reason, 0) for reason in REASON_NAMES.values()},
        "failed": failed_total, "built": built_total, "kept_records": int(len(plan)),
        "chunks_planned": chunks_planned, "chunks_built": chunks_built,
        "unique_molecules": int(plan["molecule"].nunique()),
        "pt_bytes": pt_bytes_total, "per_split": per_split, "failures": failures_rows,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    os.replace(tmp, path)


def resolve_sdf(input_path: Path, work_dir: Path) -> tuple[Path, Optional[str]]:
    """Return the SDF to read and, for a zip, the sha256 of the archive."""
    if input_path.suffix.lower() != ".zip":
        return input_path, None
    with zipfile.ZipFile(input_path) as archive:
        members = [info for info in archive.infolist() if info.filename.lower().endswith(".sdf")]
        if len(members) != 1:
            raise ValueError(f"{input_path}: expected exactly one .sdf member, found {len(members)}")
        member = members[0]
        target = work_dir / Path(member.filename).name
        if target.exists() and target.stat().st_size == member.file_size:
            print(f"Reusing extracted {target}", flush=True)
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            print(f"Extracting {member.filename} ({member.file_size / 1e9:.1f} GB) to {target}", flush=True)
            with archive.open(member) as source, open(partial, "wb") as sink:
                shutil.copyfileobj(source, sink, 16 * 1024 * 1024)
            os.replace(partial, target)
    return target, _sha256(input_path)


def _sha256_prefix(path: Path, num_bytes: int) -> str:
    digest = hashlib.sha256()
    remaining = num_bytes
    with open(path, "rb") as handle:
        while remaining > 0:
            block = handle.read(min(16 * 1024 * 1024, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _tokenizer_sha256() -> str:
    vocab = Path(__file__).resolve().parent / "ps_tokenizer" / "vocabs" / "ps_vocab_300.txt"
    return hashlib.sha256(vocab.read_bytes()).hexdigest()


def _versions() -> dict:
    import rdkit
    import torch_geometric

    return {
        "python": platform.python_version(), "rdkit": rdkit.__version__,
        "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
    }


def _parameters(args: argparse.Namespace) -> dict:
    return {
        "split_salt": args.split_salt, "split_fractions": list(args.split_fractions),
        "shuffle_seed": args.shuffle_seed, "molecule_key": args.molecule_key,
        "chunk_size": args.chunk_size, "min_heavy_atoms": args.min_heavy_atoms,
        "allowed_elements": sorted(args.allowed_elements), "max_failure_fraction": args.max_failure_fraction,
        "max_records": args.max_records, "edge_cutoff": EDGE_CUTOFF, "edge_max_neighbors": EDGE_MAX_NEIGHBORS,
    }


def run_build(args: argparse.Namespace) -> int:
    dest_prefix: Path = args.dest_prefix
    manifest_path = Path(f"{dest_prefix}_build_manifest.json")
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "complete":
        recorded, requested = json.loads(manifest_path.read_text())["parameters"], _parameters(args)
        differing = sorted(key for key in requested if recorded.get(key) != requested[key])
        if differing:
            raise ValueError(
                f"{manifest_path} records a complete build with different {differing}; "
                "choose a new --dest-prefix instead of rebuilding in place"
            )
        report = verify_build(dest_prefix)
        print(f"Build already complete; verification passed: {report['built']:,} records built.", flush=True)
        return 0

    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"COCONUT build started {started_utc}", flush=True)
    print(f"  command: {' '.join(sys.argv)}", flush=True)
    dest_prefix.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(f"{dest_prefix}_work")
    sdf_path, archive_sha256 = resolve_sdf(args.input, work_dir)

    print("Stage 1/4: scanning records and applying filters", flush=True)
    scan = scan_sdf(
        sdf_path, salt=args.split_salt, fractions=args.split_fractions,
        molecule_key=args.molecule_key, min_heavy_atoms=args.min_heavy_atoms,
        allowed_elements=args.allowed_elements, workers=args.workers,
        range_bytes=args.scan_range_mb * 1024 * 1024, max_records=args.max_records,
    )
    reasons = reason_counts(scan)
    kept = int((scan.reason == KEPT).sum())
    print(f"  raw {scan.num_records:,}, rejected {scan.num_records - kept:,} {reasons}, kept {kept:,}", flush=True)
    if kept == 0:
        raise ValueError("no record passed the filters")
    sdf_sha256 = _sha256_prefix(sdf_path, scan.scanned_bytes)

    print("Stage 2/4: planning molecule-level splits and chunks", flush=True)
    plan = build_plan(scan, chunk_size=args.chunk_size, shuffle_seed=args.shuffle_seed)
    plan_bytes = plan_csv_gz_bytes(plan)
    write_once_or_match(Path(f"{dest_prefix}_plan.csv.gz"), plan_bytes)
    write_once_or_match(Path(f"{dest_prefix}_rejected.csv"), rejected_csv_bytes(scan))

    parameters = _parameters(args)
    identity = {
        "builder": BUILDER, "filter_version": FILTER_VERSION, "layout": LAYOUT,
        "inputs": {
            "file": args.input.name, "archive_sha256": archive_sha256,
            "sdf_sha256": sdf_sha256, "scanned_bytes": scan.scanned_bytes,
        },
        "parameters": parameters, "tokenizer": {"name": "PS_300", "vocab_sha256": _tokenizer_sha256()},
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "counts": {
            "raw_records": scan.num_records, "rejected_by_reason": reasons,
            "rejected_records": scan.num_records - kept, "kept_records": kept,
        },
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if {key: previous.get(key) for key in identity} != identity:
            raise ValueError(f"{manifest_path} was written by a different build; refusing to mix builds")
        print("  resuming a started build with an identical plan", flush=True)
    else:
        _atomic_json(manifest_path, dict(
            identity, status="started", started_utc=started_utc, git_sha=_git_sha(),
            command=" ".join(sys.argv), versions=_versions(),
        ))

    source = {
        "filter_version": FILTER_VERSION, "sdf_sha256": sdf_sha256,
        "split_salt": args.split_salt, "molecule_key": args.molecule_key,
        "builder": BUILDER, "git_sha": _git_sha(),
    }
    tasks = chunk_tasks(plan, scan, dest_prefix, max_failure_fraction=args.max_failure_fraction, source=source)
    total_chunks = plan.groupby(["split", "chunk"]).ngroups
    print(
        f"Stage 3/4: converting {len(tasks)} chunks ({total_chunks - len(tasks)} already complete) "
        f"with {args.workers} workers",
        flush=True,
    )
    started = time.monotonic()
    done_records = written = 0
    total_records = sum(len(task[3]) for task in tasks)
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_chunk_worker, initargs=(str(sdf_path),)
    ) as executor:
        for number, result in enumerate(
            ordered_bounded_map(executor, build_chunk, tasks, window=args.workers), start=1
        ):
            done_records += result["planned"]
            written += result["bytes"]
            elapsed = time.monotonic() - started
            eta = (total_records - done_records) / max(done_records / max(elapsed, 1e-9), 1e-9)
            print(
                f"  [{number}/{len(tasks)}] {result['split']} chunk {result['chunk']:03d}: "
                f"{result['built']:,} built, {result['failed']} failed, {result['seconds']:.0f}s | "
                f"{100.0 * done_records / max(total_records, 1):5.1f}% | elapsed {_format_seconds(elapsed)} | "
                f"ETA {_format_seconds(eta)} | {written / 1e9:.1f} GB",
                flush=True,
            )

    print("Stage 4/4: completeness check", flush=True)
    previous = json.loads(manifest_path.read_text())
    try:
        report = verify_build(dest_prefix)
    except BuildVerificationError as exc:
        _atomic_json(manifest_path, dict(previous, status="failed_verification", error=str(exc)))
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    failures = pd.DataFrame(report.pop("failures"), columns=["split", "chunk", "record", "id", "reason"])
    write_once_or_match(Path(f"{dest_prefix}_failures.csv"), failures.to_csv(index=False).encode("utf-8"))
    _atomic_json(manifest_path, dict(
        previous, status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
        build=report, failures_file=Path(f"{dest_prefix}_failures.csv").name,
        failed_ids_by_reason=dict(Counter(failures["reason"].str.split(":").str[0])),
    ))
    if archive_sha256 is not None and not args.keep_extracted:
        shutil.rmtree(work_dir)
    print(
        f"Done: raw {report['raw_records']:,} = rejected {report['rejected']:,} + failed "
        f"{report['failed']:,} + built {report['built']:,}; {report['chunks_built']} chunks, "
        f"{report['pt_bytes'] / 1e9:.1f} GB, {_format_seconds(time.monotonic() - started)} conversion.",
        flush=True,
    )
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, help="raw COCONUT 3D SDF or the official .zip")
    parser.add_argument("--dest-prefix", required=True, type=Path,
                        help="new prefix; writes <prefix>_<split>_chunk_NNN and <prefix>_* records")
    parser.add_argument("--chunk-size", type=int, default=5000, help="upper bound of records per chunk")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--split-salt", default="mind-coconut-split-v1")
    parser.add_argument("--split-fractions", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--molecule-key", choices=("smiles", "identifier"), default="smiles",
                        help="SD item identifying a molecule; all its conformers share one split")
    parser.add_argument("--min-heavy-atoms", type=int, default=DEFAULT_MIN_HEAVY_ATOMS)
    parser.add_argument("--allowed-elements", nargs="+", default=list(DEFAULT_ALLOWED_ELEMENTS))
    parser.add_argument("--max-failure-fraction", type=float, default=0.02,
                        help="abort when a chunk, or the whole build, exceeds this failure fraction")
    parser.add_argument("--max-records", type=int, default=None,
                        help="probe only: use the first N raw records (recorded in the manifest)")
    parser.add_argument("--scan-range-mb", type=int, default=128)
    parser.add_argument("--keep-extracted", action="store_true", help="keep the SDF extracted from a zip")
    parser.add_argument("--verify-only", action="store_true",
                        help="re-run the completeness check of an existing build")
    args = parser.parse_args(argv)
    if abs(sum(args.split_fractions) - 1.0) > 1e-9 or min(args.split_fractions) < 0:
        parser.error(f"--split-fractions must be non-negative and sum to 1: {args.split_fractions}")
    if args.workers < 1 or args.chunk_size < 1 or args.scan_range_mb < 1:
        parser.error("--workers, --chunk-size and --scan-range-mb must be positive")
    if not 0.0 <= args.max_failure_fraction < 1.0:
        parser.error("--max-failure-fraction must be in [0, 1)")
    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be positive")
    if not args.verify_only:
        if args.input is None:
            parser.error("--input is required unless --verify-only is given")
        if not args.input.is_file():
            parser.error(f"input not found: {args.input}")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        report = verify_build(args.dest_prefix)
        print(f"Verification passed: raw {report['raw_records']:,} = rejected {report['rejected']:,} + "
              f"failed {report['failed']:,} + built {report['built']:,}; {report['chunks_built']} chunks.")
        return 0
    return run_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
