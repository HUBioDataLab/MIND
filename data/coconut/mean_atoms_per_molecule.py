#!/usr/bin/env python3
"""
Analyze mean atom number per molecule in an SDF file.

Reads an SDF file and reports:
- Mean, min, max, and standard deviation of atoms per molecule
- Optional histogram of atom counts per molecule

Usage:
    python mean_atoms_per_molecule.py [path/to/file.sdf]
    python mean_atoms_per_molecule.py   # uses default SDF in this directory
"""

import argparse
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt

try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit is required. Install with: pip install rdkit", file=sys.stderr)
    sys.exit(1)


# Default SDF filename when running from data/COCONUT
DEFAULT_SDF = "coconut_min10atoms_allowed_elements.sdf"

# Progress report interval when verbose
VERBOSE_INTERVAL = 10_000


def analyze_atoms_per_molecule(sdf_path: str, verbose: bool = True) -> dict:
    """
    Read an SDF file and compute statistics of atom count per molecule.

    Args:
        sdf_path: Path to the SDF file.
        verbose: If True, print progress while processing.

    Returns:
        dict with keys: num_molecules, atom_counts, mean, min, max, std, distribution
    """
    if not os.path.isfile(sdf_path):
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    atom_counts: list[int] = []
    invalid_blocks = 0

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

    for i, mol in enumerate(suppl):
        if mol is None:
            invalid_blocks += 1
            continue
        n_atoms = mol.GetNumHeavyAtoms()
        atom_counts.append(n_atoms)
        if verbose and len(atom_counts) % VERBOSE_INTERVAL == 0:
            print(f"\r  Molecules: {len(atom_counts):,}", end="", flush=True)

    if verbose:
        print()

    num_molecules = len(atom_counts)
    if num_molecules == 0:
        return {
            "sdf_path": sdf_path,
            "num_molecules": 0,
            "invalid_blocks": invalid_blocks,
            "atom_counts": [],
            "mean": 0.0,
            "min": None,
            "max": None,
            "std": 0.0,
            "distribution": {},
        }

    total_atoms = sum(atom_counts)
    mean = total_atoms / num_molecules
    min_atoms = min(atom_counts)
    max_atoms = max(atom_counts)
    variance = sum((x - mean) ** 2 for x in atom_counts) / num_molecules
    std = variance ** 0.5
    distribution = dict(Counter(atom_counts))

    return {
        "sdf_path": sdf_path,
        "num_molecules": num_molecules,
        "invalid_blocks": invalid_blocks,
        "atom_counts": atom_counts,
        "mean": mean,
        "min": min_atoms,
        "max": max_atoms,
        "std": std,
        "distribution": distribution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze mean atom number per molecule in an SDF file."
    )
    parser.add_argument(
        "sdf_file",
        nargs="?",
        default=None,
        help=f"SDF file path (default: {DEFAULT_SDF} in this directory)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress while processing.",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Plot histogram of atom counts per molecule.",
    )
    args = parser.parse_args()

    if args.sdf_file is not None:
        sdf_path = os.path.abspath(args.sdf_file)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sdf_path = os.path.join(script_dir, DEFAULT_SDF)

    if args.verbose:
        print(f"Reading SDF: {sdf_path}")
    try:
        result = analyze_atoms_per_molecule(sdf_path, verbose=args.verbose)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    n = result["num_molecules"]
    if n == 0:
        print(f"SDF file: {result['sdf_path']}")
        print("No valid molecules found.")
        if result["invalid_blocks"]:
            print(f"Invalid/unreadable blocks: {result['invalid_blocks']:,}")
        return

    print(f"SDF file: {result['sdf_path']}")
    print()
    print("--- Atoms per molecule ---")
    print(f"  Molecules (valid):     {n:,}")
    if result["invalid_blocks"]:
        print(f"  Invalid blocks:        {result['invalid_blocks']:,}")
    print(f"  Mean atoms/molecule:  {result['mean']:.2f}")
    print(f"  Min atoms:            {result['min']:,}")
    print(f"  Max atoms:            {result['max']:,}")
    print(f"  Std deviation:        {result['std']:.2f}")
    print()

    if args.histogram and result["atom_counts"]:
        plt.figure(figsize=(8, 5))
        plt.hist(result["atom_counts"], bins="auto", edgecolor="black")
        plt.xlabel("Atoms per molecule")
        plt.ylabel("Number of molecules")
        plt.title("Histogram of atoms per molecule")
        plt.tight_layout()

        base, _ = os.path.splitext(result["sdf_path"])
        out_path = f"{base}_atoms_per_molecule_hist.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Histogram saved to: {out_path}")


if __name__ == "__main__":
    main()
