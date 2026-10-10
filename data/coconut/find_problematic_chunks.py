#!/usr/bin/env python3
"""
Find which chunk indices contain problematic 3D coordinates in an SDF file.

This script scans an SDF file and reports which chunk(s) problematic molecules
would fall into, using the same chunking logic as:
`data_loading/adapters/coconut_adapter.py`.

Default problematic criteria:
1) Molecule cannot be parsed by RDKit (invalid block)
2) Molecule has no conformer (no 3D coordinates)
3) All atoms have z ~= 0 (flat-z structure)

Usage:
    python data/COCONUT/find_problematic_chunks.py --num-chunks 1000

    python data/COCONUT/find_problematic_chunks.py \
      --sdf-path data/COCONUT/coconut.sdf \
      --num-chunks 1000 \
      --z-zero-eps 1e-8 \
      --top-chunks 20
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit is required. Install with: pip install rdkit", file=sys.stderr)
    sys.exit(1)


DEFAULT_SDF = "coconut.sdf"
PROGRESS_INTERVAL = 10000


@dataclass
class IssueRecord:
    mol_index: int
    chunk_index: int
    reason: str
    molecule_id: str


def _get_molecule_id(mol, idx: int) -> str:
    """Best-effort molecule identifier for reporting."""
    if mol is None:
        return f"invalid_{idx}"
    if mol.HasProp("IDENTIFIER"):
        ident = mol.GetProp("IDENTIFIER").strip()
        if ident:
            return ident
    if mol.HasProp("_Name"):
        name = mol.GetProp("_Name").strip()
        if name:
            return name
    return f"mol_{idx}"


def _compute_chunk_bounds(total_molecules: int, num_chunks: int) -> int:
    """Same chunk-size formula used in coconut adapter."""
    return (total_molecules + num_chunks - 1) // num_chunks


def _detect_issue(mol, z_zero_eps: float) -> Optional[str]:
    """Return issue reason if problematic, else None."""
    if mol is None:
        return "invalid_rdkit_block"
    if mol.GetNumConformers() == 0:
        return "no_conformer"

    conf = mol.GetConformer()
    atom_count = mol.GetNumAtoms()
    if atom_count == 0:
        return "empty_molecule"

    # Flag molecules whose every atom has near-zero z coordinate.
    all_z_zero = True
    for atom_idx in range(atom_count):
        z = conf.GetAtomPosition(atom_idx).z
        if abs(float(z)) > z_zero_eps:
            all_z_zero = False
            break
    if all_z_zero:
        return "all_z_zero"

    return None


def analyze_problematic_chunks(
    sdf_path: str,
    num_chunks: int,
    z_zero_eps: float,
    keep_examples_per_chunk: int,
    verbose: bool,
) -> Dict:
    """Scan SDF and aggregate problematic molecules by chunk."""
    if not os.path.isfile(sdf_path):
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")
    if num_chunks <= 0:
        raise ValueError("--num-chunks must be > 0")

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    total_molecules = len(suppl) if hasattr(suppl, "__len__") else None
    if total_molecules is None:
        raise RuntimeError("Could not determine total molecule count from SDF supplier.")
    if total_molecules == 0:
        raise RuntimeError("SDF appears empty.")

    chunk_size = _compute_chunk_bounds(total_molecules, num_chunks)

    issue_counts = Counter()
    issue_counts_by_chunk = defaultdict(Counter)
    total_by_chunk = Counter()
    examples_by_chunk: Dict[int, List[IssueRecord]] = defaultdict(list)

    for idx, mol in enumerate(suppl):
        chunk_index = idx // chunk_size
        if chunk_index >= num_chunks:
            chunk_index = num_chunks - 1

        total_by_chunk[chunk_index] += 1
        reason = _detect_issue(mol, z_zero_eps=z_zero_eps)
        if reason is not None:
            issue_counts[reason] += 1
            issue_counts_by_chunk[chunk_index][reason] += 1
            if len(examples_by_chunk[chunk_index]) < keep_examples_per_chunk:
                examples_by_chunk[chunk_index].append(
                    IssueRecord(
                        mol_index=idx,
                        chunk_index=chunk_index,
                        reason=reason,
                        molecule_id=_get_molecule_id(mol, idx),
                    )
                )

        if verbose and (idx + 1) % PROGRESS_INTERVAL == 0:
            print(f"\rProcessed: {idx + 1:,}/{total_molecules:,}", end="", flush=True)

    if verbose:
        print()

    problematic_counts_by_chunk = {
        c: sum(reason_counts.values())
        for c, reason_counts in issue_counts_by_chunk.items()
    }

    return {
        "sdf_path": sdf_path,
        "total_molecules": total_molecules,
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "issue_counts": dict(issue_counts),
        "issue_counts_by_chunk": {k: dict(v) for k, v in issue_counts_by_chunk.items()},
        "problematic_counts_by_chunk": problematic_counts_by_chunk,
        "total_by_chunk": dict(total_by_chunk),
        "examples_by_chunk": examples_by_chunk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find chunk indices that contain problematic 3D coordinates."
    )
    parser.add_argument(
        "--sdf-path",
        type=str,
        default=None,
        help="Path to SDF file (default: data/COCONUT/coconut.sdf).",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        required=True,
        help="Chunk count used by process_chunked_dataset.py",
    )
    parser.add_argument(
        "--z-zero-eps",
        type=float,
        default=1e-8,
        help="Tolerance for treating z as zero (default: 1e-8).",
    )
    parser.add_argument(
        "--top-chunks",
        type=int,
        default=30,
        help="How many chunks with most issues to print (default: 30).",
    )
    parser.add_argument(
        "--examples-per-chunk",
        type=int,
        default=3,
        help="How many example molecules to print per problematic chunk (default: 3).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while scanning the SDF.",
    )
    args = parser.parse_args()

    if args.sdf_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sdf_path = os.path.join(script_dir, DEFAULT_SDF)
    else:
        sdf_path = os.path.abspath(args.sdf_path)

    result = analyze_problematic_chunks(
        sdf_path=sdf_path,
        num_chunks=args.num_chunks,
        z_zero_eps=args.z_zero_eps,
        keep_examples_per_chunk=args.examples_per_chunk,
        verbose=args.verbose,
    )

    total = result["total_molecules"]
    num_chunks = result["num_chunks"]
    chunk_size = result["chunk_size"]
    issue_counts = result["issue_counts"]
    issue_counts_by_chunk = result["issue_counts_by_chunk"]
    problematic_counts_by_chunk = result["problematic_counts_by_chunk"]
    examples_by_chunk = result["examples_by_chunk"]

    print(f"SDF file: {result['sdf_path']}")
    print(f"Total molecules: {total:,}")
    print(f"Num chunks: {num_chunks:,}")
    print(f"Computed chunk size: {chunk_size:,}")
    print()

    print("=== Problem Summary ===")
    total_issues = sum(issue_counts.values())
    print(f"Total problematic molecules: {total_issues:,}")
    if not issue_counts:
        print("No problematic molecules found by the selected criteria.")
        return

    for reason, count in sorted(issue_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"- {reason}: {count:,}")
    print()

    affected_chunks = sorted(problematic_counts_by_chunk.keys())
    print(f"Affected chunk count: {len(affected_chunks):,}")
    print(
        "Affected chunk indices (first 100): "
        + ", ".join(str(c) for c in affected_chunks[:100])
        + (" ..." if len(affected_chunks) > 100 else "")
    )
    print()

    print(f"=== Top {args.top_chunks} Chunks by Problem Count ===")
    ranked_chunks = sorted(
        problematic_counts_by_chunk.items(),
        key=lambda x: (-x[1], x[0]),
    )[: args.top_chunks]
    for chunk_idx, bad_count in ranked_chunks:
        reasons = issue_counts_by_chunk.get(chunk_idx, {})
        reason_str = ", ".join(
            f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda x: (-x[1], x[0]))
        )
        print(f"- chunk {chunk_idx}: problematic={bad_count:,} ({reason_str})")

    print()
    print("=== Examples ===")
    for chunk_idx, _ in ranked_chunks[: min(10, len(ranked_chunks))]:
        examples = examples_by_chunk.get(chunk_idx, [])
        if not examples:
            continue
        print(f"chunk {chunk_idx}:")
        for rec in examples:
            print(
                f"  - mol_index={rec.mol_index}, id={rec.molecule_id}, reason={rec.reason}"
            )


if __name__ == "__main__":
    main()
