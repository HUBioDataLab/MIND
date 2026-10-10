#!/usr/bin/env python3

"""
Filter an SDF file by atom count in a streaming, low‑memory way.

This version is optimized for very large SDFs:
- Uses a forward (streaming) supplier
- Avoids sanitization
- Minimizes Python attribute lookups inside the tight loop
- Provides optional, infrequent progress logging
"""

import argparse
import os
import sys
from typing import Optional

from rdkit import Chem


DEFAULT_INPUT_SDF = "coconut.sdf"
DEFAULT_OUTPUT_SDF = "coconut_min10atoms.sdf"


def filter_sdf_by_atom_count(
    input_sdf: str,
    output_sdf: str,
    min_atoms: int = 10,
    heavy_only: bool = True,
    verbose: bool = True,
    progress_interval: int = 10_000,
) -> tuple[int, int]:
    """
    Stream molecules from an SDF and write only those with >= min_atoms.

    Returns:
        (total_molecules, kept_molecules)
    """
    if not os.path.isfile(input_sdf):
        raise FileNotFoundError(f"SDF file not found: {input_sdf}")

    # ForwardSDMolSupplier is more streaming‑oriented and can be a bit lighter
    # than SDMolSupplier for huge files.
    suppl = Chem.ForwardSDMolSupplier(
        open(input_sdf, "rb"),
        sanitize=False,
        removeHs=False,
    )

    writer = Chem.SDWriter(output_sdf)

    kept = 0
    total = 0

    # Bind frequently used attributes locally to reduce attribute lookups
    get_num_heavy = Chem.Mol.GetNumHeavyAtoms
    get_num_atoms = Chem.Mol.GetNumAtoms
    write = writer.write

    for mol in suppl:
        total += 1
        if mol is None:
            # Skip unreadable records quickly
            continue

        if heavy_only:
            atom_count = get_num_heavy(mol)
        else:
            atom_count = get_num_atoms(mol)

        if atom_count >= min_atoms:
            write(mol)
            kept += 1

        if verbose and progress_interval > 0 and total % progress_interval == 0:
            # Simple, infrequent progress so "server not responding" UIs
            # show that work is ongoing.
            print(f"\rProcessed {total:,} molecules, kept {kept:,}", end="", flush=True)

    writer.close()

    if verbose:
        print()  # newline after progress
        print(f"Done. Processed {total:,} molecules, kept {kept:,}.")

    return total, kept


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter an SDF file by atom count (streaming, low‑memory)."
    )
    parser.add_argument(
        "input_sdf",
        nargs="?",
        help=f"Input SDF file (default: {DEFAULT_INPUT_SDF} in this directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_sdf",
        help=f"Output SDF file (default: {DEFAULT_OUTPUT_SDF} in this directory)",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=10,
        help="Minimum number of atoms required to keep a molecule (default: 10).",
    )
    parser.add_argument(
        "--all-atoms",
        action="store_true",
        help="Count all atoms including hydrogens (default: heavy atoms only).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Run quietly (no progress output).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100_000,
        help="Print progress every N molecules (default: 100000, 0 to disable).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_sdf = (
        os.path.abspath(args.input_sdf)
        if args.input_sdf is not None
        else os.path.join(script_dir, DEFAULT_INPUT_SDF)
    )
    output_sdf = (
        os.path.abspath(args.output_sdf)
        if args.output_sdf is not None
        else os.path.join(script_dir, DEFAULT_OUTPUT_SDF)
    )

    try:
        filter_sdf_by_atom_count(
            input_sdf=input_sdf,
            output_sdf=output_sdf,
            min_atoms=args.min_atoms,
            heavy_only=not args.all_atoms,
            verbose=not args.quiet,
            progress_interval=args.progress_interval,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()