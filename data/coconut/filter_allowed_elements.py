#!/usr/bin/env python3

"""
Filter an SDF file to keep only molecules composed of a given set of elements.

By default, it keeps molecules that contain ONLY these elements:
    H, C, O, N, S, Cl, Br, P, I, Na, F
Any molecule that has at least one atom outside this set is discarded.

The script is streaming and low‑memory; it is suitable for very large SDF files.
"""

import argparse
import os
import sys
from typing import Iterable, Optional, Set

from rdkit import Chem


# Default paths when running from data/COCONUT
DEFAULT_INPUT_SDF = "coconut_min10atoms.sdf"
DEFAULT_OUTPUT_SDF = "coconut_min10atoms_allowed_elements.sdf"

# Default allowed elements (includes hydrogen)
DEFAULT_ALLOWED = {"H", "C", "O", "N", "S", "Cl", "Br", "P", "I", "Na", "F"}


def parse_allowed_elements(raw: Optional[str]) -> Set[str]:
    """
    Parse a comma‑separated list of element symbols into a set.

    Example: "H,C,O,N,S,Cl,Br,P,I,Na,F"
    """
    if raw is None:
        return set(DEFAULT_ALLOWED)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return set(parts) if parts else set(DEFAULT_ALLOWED)


def molecule_uses_only_allowed_elements(mol: Chem.Mol, allowed: Set[str]) -> bool:
    """Return True if all atom symbols in mol are in the allowed set."""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed:
            return False
    return True


def filter_sdf_by_elements(
    input_sdf: str,
    output_sdf: str,
    allowed_elements: Iterable[str],
    verbose: bool = True,
    progress_interval: int = 100_000,
) -> tuple[int, int]:
    """
    Stream molecules from an SDF and write only those whose atoms are all allowed.

    Args:
        input_sdf: path to input SDF file
        output_sdf: path to output SDF file
        allowed_elements: iterable of allowed element symbols
        verbose: whether to print progress and summary
        progress_interval: print progress every N molecules (0 to disable)

    Returns:
        (total_molecules, kept_molecules)
    """
    if not os.path.isfile(input_sdf):
        raise FileNotFoundError(f"SDF file not found: {input_sdf}")

    allowed = set(allowed_elements)

    suppl = Chem.ForwardSDMolSupplier(
        open(input_sdf, "rb"),
        sanitize=False,
        removeHs=False,
    )
    writer = Chem.SDWriter(output_sdf)

    total = 0
    kept = 0

    uses_only_allowed = molecule_uses_only_allowed_elements
    write = writer.write

    for mol in suppl:
        total += 1
        if mol is None:
            continue

        if uses_only_allowed(mol, allowed):
            write(mol)
            kept += 1

        if verbose and progress_interval > 0 and total % progress_interval == 0:
            print(f"\rProcessed {total:,} molecules, kept {kept:,}", end="", flush=True)

    writer.close()

    if verbose:
        print()
        print(f"Done. Processed {total:,} molecules, kept {kept:,}.")

    return total, kept


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an SDF file to keep only molecules composed of a given set of elements.\n"
            "Any molecule containing at least one atom outside the allowed set is removed."
        )
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
        "--allowed",
        type=str,
        default=None,
        help=(
            "Comma‑separated list of allowed element symbols "
            f'(default: {",".join(sorted(DEFAULT_ALLOWED))})'
        ),
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
    allowed_set = parse_allowed_elements(args.allowed)

    try:
        filter_sdf_by_elements(
            input_sdf=input_sdf,
            output_sdf=output_sdf,
            allowed_elements=allowed_set,
            verbose=not args.quiet,
            progress_interval=args.progress_interval,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

