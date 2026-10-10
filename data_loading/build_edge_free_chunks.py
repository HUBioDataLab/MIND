#!/usr/bin/env python3
"""Build edge-free protein/RNA chunks directly from raw structure files.

Per structure this runs the production node pipeline without materializing
cached edges: adapter (14/23-slot blocks) -> ``_universal_to_pyg`` node
fields -> visible-V prepositioning (``preposition_visible_virtuals``, the
node step of the template stage) -> compact edge-free encoding
(``data_loading.pt_schema``).

Splits are assigned per structure ID by SHA-256 (``--split-salt``), so an ID
always lands in the same split. Within each split the IDs are shuffled with
``--shuffle-seed`` before chunking, because manifests can be sorted (the 1M
protein manifest is ordered by pLDDT). Chunks are written as
``<dest>_<split>_chunk_NNN/processed/``; ``--no-split`` keeps manifest order
and writes ``<dest>_chunk_NNN`` (used to reproduce an existing chunking).

The run is resumable: the ID plan is written once and verified on restart,
each chunk is written to ``<dir>.partial`` and renamed when complete, and
complete chunks are skipped. Structures that fail to parse or template are
recorded per chunk; a chunk aborts the run when its failure fraction exceeds
``--max-failure-fraction``.

Example (detached, survives logout):
    setsid nohup python -m data_loading.build_edge_free_chunks --domain protein \\
        --manifest .../manifest_hq_1m_1024.csv --raw-dir .../raw_structures_1m_1024 \\
        --dest-prefix /home/yusuf/data/proteins/processed_14atom_1m_edgefree \\
        --chunk-size 4000 --workers 32 > build.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

# CPU-only tool: hide GPUs so forked workers never initialize CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate

from data_loading.atom_position_vocab import count_p_only_rna_blocks
from data_loading.create_chunk_metadata import assign_bucket
from data_loading.precompute_visible_virtual_templates import preposition_visible_virtuals
from data_loading.pt_schema import (
    PT_SCHEMA_NAME,
    build_edge_free_schema,
    encode_edge_free_sample,
    normalize_schema_domain,
    regenerate_protein_sequence_edges,
    split_chunk_payload,
)

SPLITS = ("train", "val", "test")
ID_COLUMNS = {"protein": "repId", "rna": "name"}
FILE_SUFFIXES = (".pdb", ".cif")


# ---------------------------------------------------------------------------
# Plan: split assignment and chunking
# ---------------------------------------------------------------------------

def assign_split(structure_id: str, salt: str, fractions: Sequence[float]) -> str:
    """Deterministic split from SHA-256 of ``salt:id`` (process-independent)."""
    digest = hashlib.sha256(f"{salt}:{structure_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for split, fraction in zip(SPLITS, fractions):
        cumulative += fraction
        if value < cumulative:
            return split
    return SPLITS[-1]


def split_group(structure_id: str, group_by: str) -> str:
    """Key that must land in one split: the ID, or its PDB entry (``1asy_R`` -> ``1asy``)."""
    if group_by == "id":
        return structure_id
    if group_by == "pdb-entry":
        return structure_id.rsplit("_", 1)[0]
    raise ValueError(f"unknown split grouping {group_by!r}")


def build_plan(
    ids: Sequence[str],
    *,
    chunk_size: int,
    split_salt: Optional[str],
    fractions: Sequence[float],
    shuffle_seed: int,
    group_by: str = "id",
) -> list[dict]:
    """Return rows ``{id, split, chunk, position}``; ``split`` is "" without a salt."""
    if len(set(ids)) != len(ids):
        raise ValueError("manifest contains duplicate structure IDs")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if split_salt is None:
        groups = {"": list(ids)}
    else:
        if len(fractions) != len(SPLITS) or abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError(f"split fractions must be three values summing to 1: {fractions}")
        groups = {split: [] for split in SPLITS}
        for structure_id in ids:
            key = split_group(structure_id, group_by)
            groups[assign_split(key, split_salt, fractions)].append(structure_id)
        for split, members in groups.items():
            members.sort()
            random.Random(f"{shuffle_seed}:{split}").shuffle(members)
    rows = []
    for split, members in groups.items():
        for offset, structure_id in enumerate(members):
            rows.append({
                "id": structure_id,
                "split": split,
                "chunk": offset // chunk_size,
                "position": offset % chunk_size,
            })
    return rows


def chunk_directory(dest_prefix: Path, split: str, chunk: int) -> Path:
    tag = f"_{split}" if split else ""
    return dest_prefix.parent / f"{dest_prefix.name}{tag}_chunk_{chunk:03d}"


def resolve_raw_file(raw_dir: Path, structure_id: str) -> Path:
    for suffix in FILE_SUFFIXES:
        candidate = raw_dir / f"{structure_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no raw structure for {structure_id} in {raw_dir}")


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "split", "chunk", "position"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_or_write_plan(path: Path, rows: list[dict]) -> None:
    """Write the plan once; a restart must reproduce it exactly."""
    if path.exists():
        with open(path, newline="") as handle:
            existing = [
                {**row, "chunk": int(row["chunk"]), "position": int(row["position"])}
                for row in csv.DictReader(handle)
            ]
        if existing != rows:
            raise ValueError(f"{path} differs from the recomputed plan; refusing to mix plans")
        return
    _write_csv_atomic(path, rows)


# ---------------------------------------------------------------------------
# Worker: raw file -> node-only sample
# ---------------------------------------------------------------------------

_WORKER = None


def _init_worker(domain: str) -> None:
    global _WORKER
    torch.set_num_threads(1)
    from data_loading.cache_to_pyg import OptimizedUniversalDataset

    if domain == "protein":
        from data_loading.adapters.protein_adapter import ProteinAdapter
        adapter = ProteinAdapter()
    else:
        from data_loading.adapters.rna_adapter import RNAAdapter
        adapter = RNAAdapter()
    # Config-only converter, as in cache_to_pyg._init_conversion_worker.
    converter = object.__new__(OptimizedUniversalDataset)
    converter.molecule_max_atoms = None
    converter.cutoff_distance = 5.0
    converter.max_neighbors = 64
    converter.pre_filter = None
    converter.pre_transform = None
    _WORKER = (domain, adapter, converter)


def build_node_sample(path: str) -> tuple[str, Optional[Data], Optional[str]]:
    """Return ``(id, sample, None)`` or ``(id, None, reason)`` for one raw file."""
    domain, adapter, converter = _WORKER
    raw_path = Path(path)
    structure_id = raw_path.stem
    try:
        molecule = adapter.convert_to_universal(raw_path)
        if not molecule.blocks:
            return structure_id, None, "no parsable residues"
        if domain == "rna" and count_p_only_rna_blocks(molecule):
            return structure_id, None, "contains a P-only nucleotide block"
        data = converter._universal_to_pyg(molecule, raise_on_error=True)
        if data is None:
            return structure_id, None, "no atoms"
        sample = Data(
            z=data.z, pos=data.pos, block_idx=data.block_idx, entity_idx=data.entity_idx,
            pos_code=list(data.pos_code), block_symbols=list(data.block_symbols),
            mol_id=data.mol_id, dataset_type=data.dataset_type, num_nodes=int(data.z.numel()),
        )
        sample, result, original_dataset_type = preposition_visible_virtuals(sample, domain)
        return structure_id, Data(
            z=sample.z, pos=result.input_pos, block_idx=sample.block_idx,
            entity_idx=sample.entity_idx, pos_code=list(sample.pos_code),
            block_symbols=list(sample.block_symbols), mol_id=sample.mol_id,
            dataset_type=original_dataset_type, num_nodes=int(sample.z.numel()),
        ), None
    except Exception as exc:  # recorded per structure and bounded by --max-failure-fraction
        return structure_id, None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Chunk writer
# ---------------------------------------------------------------------------

def write_edge_free_chunk(
    samples: list,
    chunk_dir: Path,
    *,
    domain: str,
    sequence_neighbors_k: Optional[int],
    source: dict,
    extra_metadata: Optional[dict] = None,
) -> dict:
    """Encode samples and write ``chunk_dir/processed/{pt,metadata}`` atomically."""
    domain = normalize_schema_domain(domain)
    if chunk_dir.exists():
        raise FileExistsError(f"refusing to overwrite {chunk_dir}")
    if not samples:
        raise ValueError(f"{chunk_dir}: no samples to write")
    partial = chunk_dir.with_name(chunk_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    stem = f"{chunk_dir.name}_edgefree_v1"
    processed = partial / "processed"
    processed.mkdir(parents=True)

    vocabulary = sorted({str(code) for sample in samples for code in sample.pos_code})
    index = {code: position for position, code in enumerate(vocabulary)}
    compact = [encode_edge_free_sample(sample, domain, index) for sample in samples]
    data, slices, _ = collate(
        compact[0].__class__, data_list=compact, increment=False, add_batch=False
    )
    schema = build_edge_free_schema(domain, vocabulary, sequence_neighbors_k, source)
    torch.save((data, slices, schema), processed / f"{stem}.pt")

    sample_records = []
    for local_idx, sample in enumerate(samples):
        num_atoms = int(sample.z.numel())
        num_edges = 0
        if domain == "protein":
            num_edges = int(regenerate_protein_sequence_edges(
                list(sample.pos_code), sample.block_idx, sample.entity_idx, sequence_neighbors_k
            ).size(1))
        sample_records.append({
            "idx": local_idx, "num_atoms": num_atoms, "num_edges": num_edges,
            "bucket": assign_bucket(num_atoms), "mol_id": str(sample.mol_id),
        })
    metadata = {
        "chunk_file": f"{stem}.pt",
        "num_samples": len(samples),
        "dataset_type": str(samples[0].dataset_type),
        "total_atoms": sum(record["num_atoms"] for record in sample_records),
        "total_edges": sum(record["num_edges"] for record in sample_records),
        "samples": sample_records,
        **(extra_metadata or {}),
        PT_SCHEMA_NAME: schema,
    }
    with open(processed / f"{stem}_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=1)

    reloaded = split_chunk_payload(
        torch.load(processed / f"{stem}.pt", weights_only=False, mmap=True), origin=str(partial)
    )
    if reloaded[2] != schema or len(reloaded[1]["pos"]) - 1 != len(samples):
        raise ValueError(f"{partial}: written chunk failed reload verification")
    os.replace(partial, chunk_dir)
    return {
        "chunk_dir": str(chunk_dir),
        "samples": len(samples),
        "atoms": metadata["total_atoms"],
        "bytes": sum(path.stat().st_size for path in (chunk_dir / "processed").iterdir()),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _format_seconds(seconds: float) -> str:
    seconds = int(round(max(seconds, 0.0)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


def _git_sha() -> str:
    repo = Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}+dirty" if dirty.strip() else sha


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_chunk_selection(text: Optional[str]) -> Optional[set]:
    """``train:0,val:0`` (or ``0,1`` with --no-split) -> {(split, chunk)}."""
    if text is None:
        return None
    selected = set()
    for item in text.split(","):
        split, _, chunk = item.strip().rpartition(":")
        selected.add((split, int(chunk)))
    return selected


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domain", required=True, choices=("protein", "rna"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--dest-prefix", required=True, type=Path)
    parser.add_argument("--chunk-size", required=True, type=int)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--sequence-neighbors-k", type=int, default=3)
    parser.add_argument("--split-salt", default="mind-split-v1")
    parser.add_argument("--split-fractions", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--split-group", choices=("id", "pdb-entry"), default="id",
        help="keep all chains of one PDB entry in one split (RNA)",
    )
    parser.add_argument("--no-split", action="store_true", help="keep manifest order, no split")
    parser.add_argument("--chunks", default=None, help="only these chunks, e.g. train:0,val:0")
    parser.add_argument("--max-failure-fraction", type=float, default=0.001)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    args = parser.parse_args(argv)

    k = args.sequence_neighbors_k if args.domain == "protein" else None
    with open(args.manifest, newline="") as handle:
        ids = [row[ID_COLUMNS[args.domain]] for row in csv.DictReader(handle)]
    plan = build_plan(
        ids, chunk_size=args.chunk_size,
        split_salt=None if args.no_split else args.split_salt,
        fractions=args.split_fractions, shuffle_seed=args.shuffle_seed,
        group_by=args.split_group,
    )
    args.dest_prefix.parent.mkdir(parents=True, exist_ok=True)
    load_or_write_plan(args.dest_prefix.parent / f"{args.dest_prefix.name}_plan.csv", plan)

    chunks: dict[tuple[str, int], list[str]] = {}
    for row in plan:
        chunks.setdefault((row["split"], row["chunk"]), []).append(row["id"])
    selected = _parse_chunk_selection(args.chunks)
    if selected is not None:
        unknown = selected - set(chunks)
        if unknown:
            raise ValueError(f"unknown chunks requested: {sorted(unknown)}")
        chunks = {key: value for key, value in chunks.items() if key in selected}
    pending = {
        key: members for key, members in chunks.items()
        if not chunk_directory(args.dest_prefix, *key).exists()
    }
    total = sum(len(members) for members in chunks.values())
    already = total - sum(len(members) for members in pending.values())

    source_common = {
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "raw_dir": str(args.raw_dir),
        "split_salt": None if args.no_split else args.split_salt,
        "split_fractions": None if args.no_split else list(args.split_fractions),
        "shuffle_seed": None if args.no_split else args.shuffle_seed,
        "split_group": None if args.no_split else args.split_group,
        "builder": "data_loading.build_edge_free_chunks",
        "git_sha": _git_sha(),
    }
    print(f"Edge-free build started {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"  command: {' '.join(sys.argv)}", flush=True)
    print(f"  git: {source_common['git_sha']}", flush=True)
    print(
        f"  {len(chunks)} chunks / {total:,} structures selected; "
        f"{len(chunks) - len(pending)} chunks already complete ({already:,} structures)",
        flush=True,
    )

    started = time.monotonic()
    done_now = 0
    written = 0
    failures_total = 0
    last_report = 0.0
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(args.domain,)
    ) as executor:
        for number, (key, members) in enumerate(sorted(pending.items()), start=1):
            split, chunk = key
            chunk_dir = chunk_directory(args.dest_prefix, split, chunk)
            paths = [str(resolve_raw_file(args.raw_dir, member)) for member in members]
            samples, failures = [], []
            for structure_id, sample, reason in executor.map(build_node_sample, paths, chunksize=4):
                if sample is None:
                    failures.append({"id": structure_id, "reason": reason})
                else:
                    samples.append(sample)
                done_now += 1
                now = time.monotonic()
                if now - last_report >= args.progress_interval:
                    last_report = now
                    done = already + done_now
                    rate = done_now / max(now - started, 1e-9)
                    eta = (total - done) / rate if rate > 0 else float("nan")
                    print(
                        f"[chunk {number}/{len(pending)} {chunk_dir.name}] "
                        f"{100.0 * done / total:5.1f}% | {done:,}/{total:,} structures | "
                        f"elapsed {_format_seconds(now - started)} | ETA {_format_seconds(eta)} | "
                        f"written {written / 1e9:.2f} GB | failures {failures_total + len(failures)}",
                        flush=True,
                    )
            if len(failures) > args.max_failure_fraction * len(members):
                raise RuntimeError(
                    f"{chunk_dir.name}: {len(failures)}/{len(members)} structures failed "
                    f"(> {args.max_failure_fraction:.4%}); first: {failures[:3]}"
                )
            record = write_edge_free_chunk(
                samples, chunk_dir, domain=args.domain, sequence_neighbors_k=k,
                source=dict(source_common, split=split, chunk=chunk),
                extra_metadata={"split": split, "planned_ids": len(members), "failures": failures},
            )
            failures_total += len(failures)
            written += record["bytes"]
            print(
                f"  done {chunk_dir.name}: {record['samples']:,} structures, "
                f"{record['bytes'] / 1e6:.1f} MB, failures {len(failures)}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    print(
        f"Done: {done_now:,} structures built in {_format_seconds(elapsed)}, "
        f"{written / 1e9:.2f} GB written, {failures_total} failures recorded in chunk metadata.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
