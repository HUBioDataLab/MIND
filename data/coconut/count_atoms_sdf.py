#!/usr/bin/env python3
"""
Count atom types in an SDF file.

Reads an SDF file (e.g. COCONUT), counts how many of each atom type occur,
and reports atom names with their counts and prevalence (percentage of all atoms).

Usage:
    python count_atoms_sdf.py [path/to/file.sdf]
    python count_atoms_sdf.py   # uses default SDF in this directory
"""

import argparse
import os
import sys
from collections import Counter

try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit is required. Install with: pip install rdkit", file=sys.stderr)
    sys.exit(1)


# Default SDF filename when running from data/COCONUT
DEFAULT_SDF = "coconut_min10atoms_allowed_elements.sdf"


# Progress report interval when verbose (molecules)
VERBOSE_INTERVAL = 50_000


def count_atoms_in_sdf(sdf_path: str, verbose: bool = False) -> tuple[Counter[str], int, int]:
    """
    Count each atom type (symbol) across all molecules in an SDF file.

    Args:
        sdf_path: Path to the SDF file.
        verbose: If True, print progress to the terminal.

    Returns:
        (atom_counter, total_atoms, num_molecules)
        - atom_counter: Counter mapping atom symbol -> count
        - total_atoms: total number of atoms
        - num_molecules: number of molecules processed
    """
    if not os.path.isfile(sdf_path):
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    counter: Counter[str] = Counter()
    num_molecules = 0

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

    for mol in suppl:
        if mol is None:
            continue
        num_molecules += 1
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            counter[symbol] += 1
        if verbose and num_molecules % VERBOSE_INTERVAL == 0:
            total_so_far = sum(counter.values())
            print(f"\r  Molecules: {num_molecules:,}  |  Atoms: {total_so_far:,}", end="", flush=True)

    if verbose:
        print()  # newline after progress line

    total_atoms = sum(counter.values())
    return counter, total_atoms, num_molecules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count atom types in an SDF file and report names and prevalence."
    )
    parser.add_argument(
        "sdf_file",
        nargs="?",
        default=None,
        help=f"SDF file path (default: {DEFAULT_SDF} in this directory)",
    )
    parser.add_argument(
        "--sort",
        choices=("count", "symbol", "prevalence"),
        default="count",
        help="Sort output by count (desc), symbol, or prevalence (desc). Default: count",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress (molecules and atoms) while processing.",
    )
    args = parser.parse_args()

    if args.sdf_file is not None:
        sdf_path = os.path.abspath(args.sdf_file)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sdf_path = os.path.join(script_dir, DEFAULT_SDF)

    if args.verbose:
        print(f"Reading SDF: {sdf_path}")
        print("Progress:")
    try:
        counter, total_atoms, num_molecules = count_atoms_in_sdf(sdf_path, verbose=args.verbose)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if total_atoms == 0:
        print("No atoms found in the SDF file.")
        return

    # Build rows: (symbol, count, prevalence)
    rows = [
        (symbol, count, 100.0 * count / total_atoms)
        for symbol, count in counter.items()
    ]

    if args.sort == "count":
        rows.sort(key=lambda x: (-x[1], x[0]))
    elif args.sort == "symbol":
        rows.sort(key=lambda x: x[0])
    else:  # prevalence
        rows.sort(key=lambda x: (-x[2], x[0]))

    # Report
    print(f"SDF file: {sdf_path}")
    print(f"Molecules: {num_molecules:,}")
    print(f"Total atoms: {total_atoms:,}")
    print(f"Unique atom types: {len(rows)}")
    print()
    print(f"{'Atom':<6} {'Count':>12} {'Prevalence':>12}")
    print("-" * 32)
    for symbol, count, pct in rows:
        print(f"{symbol:<6} {count:>12,} {pct:>11.2f}%")


if __name__ == "__main__":
    main()
