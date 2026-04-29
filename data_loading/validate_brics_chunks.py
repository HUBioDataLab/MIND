#!/usr/bin/env python3
"""
Validate BRICS-precomputed fields inside processed PyG chunk .pt files.

Checks per sample:
- required fields exist
- brics_frag_ptr starts at 0, is monotonic, and ends at len(brics_frag_atom_index)
- brics_frag_atom_index values are within [0, num_nodes)
- has_brics exists and is readable
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch


REQUIRED_FIELDS = [
    "smiles",
    "brics_frag_atom_index",
    "brics_frag_ptr",
    "has_brics",
    "pos",
]


def _slice_range(slices: dict, key: str, idx: int) -> Tuple[int, int]:
    s = int(slices[key][idx].item())
    e = int(slices[key][idx + 1].item())
    return s, e


def validate_chunk(chunk_file: Path, max_samples: int = -1) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    try:
        data, slices = torch.load(str(chunk_file), map_location="cpu", weights_only=False)
    except Exception as e:
        return False, [f"Failed to load {chunk_file}: {e}"]

    keys = set(data.keys()) if hasattr(data, "keys") else set()
    for field in REQUIRED_FIELDS:
        if field not in keys:
            errors.append(f"Missing field '{field}'")

    if errors:
        return False, errors

    num_samples = int(slices["pos"].numel() - 1)
    if num_samples <= 0:
        errors.append("No samples found in chunk")
        return False, errors

    n_check = num_samples if max_samples is None or max_samples < 0 else min(num_samples, max_samples)
    for i in range(n_check):
        pos_s, pos_e = _slice_range(slices, "pos", i)
        num_nodes = pos_e - pos_s

        ai_s, ai_e = _slice_range(slices, "brics_frag_atom_index", i)
        ptr_s, ptr_e = _slice_range(slices, "brics_frag_ptr", i)
        hb_s, hb_e = _slice_range(slices, "has_brics", i)

        atom_index = data.brics_frag_atom_index[ai_s:ai_e]
        ptr = data.brics_frag_ptr[ptr_s:ptr_e]
        has_brics = data.has_brics[hb_s:hb_e]

        if ptr.numel() == 0:
            errors.append(f"sample {i}: empty brics_frag_ptr")
            continue
        if int(ptr[0].item()) != 0:
            errors.append(f"sample {i}: brics_frag_ptr[0] != 0")
        if not bool(torch.all(ptr[1:] >= ptr[:-1])):
            errors.append(f"sample {i}: brics_frag_ptr is not monotonic")
        if int(ptr[-1].item()) != int(atom_index.numel()):
            errors.append(
                f"sample {i}: brics_frag_ptr[-1]={int(ptr[-1].item())} "
                f"!= len(brics_frag_atom_index)={int(atom_index.numel())}"
            )

        if atom_index.numel() > 0:
            in_bounds = bool(torch.all((atom_index >= 0) & (atom_index < num_nodes)))
            if not in_bounds:
                errors.append(f"sample {i}: brics_frag_atom_index out of [0, {num_nodes}) range")

        if has_brics.numel() == 0:
            errors.append(f"sample {i}: has_brics missing value")

    return len(errors) == 0, errors


def collect_chunk_files(chunk_dir: Path) -> List[Path]:
    return sorted(
        p for p in chunk_dir.rglob("*.pt")
        if "processed" in p.parts and not p.name.startswith("pre_")
    )


def main():
    parser = argparse.ArgumentParser(description="Validate BRICS fields in chunk .pt files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chunk-file", type=str, help="Single .pt chunk file")
    group.add_argument("--chunk-dir", type=str, help="Directory containing chunk folders/files")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples to check per chunk (-1 for all)")
    args = parser.parse_args()

    if args.chunk_file:
        files = [Path(args.chunk_file)]
    else:
        files = collect_chunk_files(Path(args.chunk_dir))
        if not files:
            print(f"No .pt files found under: {args.chunk_dir}")
            raise SystemExit(1)

    failed = 0
    for f in files:
        ok, errs = validate_chunk(f, args.max_samples)
        if ok:
            print(f"[OK] {f}")
        else:
            failed += 1
            print(f"[FAIL] {f}")
            for e in errs[:20]:
                print(f"  - {e}")
            if len(errs) > 20:
                print(f"  - ... {len(errs) - 20} more")

    print(f"\nChecked {len(files)} chunk file(s). Failed: {failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

