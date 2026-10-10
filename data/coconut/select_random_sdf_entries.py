#!/usr/bin/env python3
"""
Randomly select molecule entries from the first K records of an SDF file.

By default this script reads only the first 100 mol blocks from the SDF,
then randomly selects a user-defined number of them.

Selected entries are:
- written to an output SDF file
- logged (indices and molecule identifiers) to a log file and stdout

Usage examples (run from data/COCONUT):
    python select_random_sdf_entries.py -n 100
    python select_random_sdf_entries.py coconut_filtered.sdf -n 50 --seed 42
    python select_random_sdf_entries.py pre/coconut_fixed.sdf -n 10 \\
        --output-sdf selected_fixed_10.sdf --log-file selection_fixed_10.log
"""

import argparse
import os
import random
import sys
from datetime import datetime
from typing import List, Tuple

try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit is required. Install with: pip install rdkit", file=sys.stderr)
    sys.exit(1)


# Default SDF filename when running from data/COCONUT
DEFAULT_SDF = "coconut_min10atoms_allowed_elements.sdf"

# How many leading records from the SDF to consider
MAX_CANDIDATES = 2000000


def get_molecule_id(mol, index: int) -> str:
    """Get a stable molecule identifier for logging."""
    if mol is None:
        return f"__invalid_{index}"
    if mol.HasProp("IDENTIFIER"):
        return mol.GetProp("IDENTIFIER").strip()
    if mol.HasProp("_Name"):
        name = mol.GetProp("_Name").strip()
        if name:
            return name
    return f"__block_{index}"


def take_first_k_and_sample(
    sdf_path: str, n_select: int, k: int, rng: random.Random
) -> Tuple[List[Tuple[int, object]], int]:
    """
    Take the first k records from the SDF, then randomly select n_select of them.

    Returns:
        (selected, total_considered)
        - selected: list of (index, mol) pairs (indices are original 0-based file indices)
        - total_considered: min(k, total_records_in_file)
    """
    if not os.path.isfile(sdf_path):
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    if n_select <= 0:
        raise ValueError("Number of entries to select must be positive.")

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

    candidates: List[Tuple[int, object]] = []
    for idx, mol in enumerate(suppl):
        if len(candidates) >= k:
            break
        candidates.append((idx, mol))

    total_considered = len(candidates)

    if total_considered == 0:
        raise ValueError("SDF appears to contain 0 records.")
    if n_select > total_considered:
        raise ValueError(
            f"Requested {n_select} entries, but only {total_considered} records "
            f"are available in the first {k} entries."
        )

    chosen_indices = sorted(rng.sample(range(total_considered), n_select))
    selected = [candidates[i] for i in chosen_indices]
    return selected, total_considered


def write_selected_sdf(
    selected: List[Tuple[int, object]],
    output_sdf: str,
) -> None:
    """Write selected molecules to an output SDF file."""
    writer = Chem.SDWriter(output_sdf)
    try:
        for idx, mol in selected:
            if mol is None:
                continue
            writer.write(mol)
    finally:
        writer.close()


def log_selection(
    sdf_path: str,
    selected: List[Tuple[int, object]],
    total_considered: int,
    output_sdf: str,
    log_file: str,
) -> None:
    """Log selected entries to a logfile and stdout in a readable table."""
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"[{timestamp}] Random SDF selection")
    lines.append(f"Source SDF          : {sdf_path}")
    lines.append(f"Output SDF          : {output_sdf}")
    lines.append(f"Records considered  : {total_considered}")
    lines.append(f"Records selected    : {len(selected)}")
    lines.append("-" * 72)
    lines.append(f"{'Sel#':>4} | {'FileIdx':>7} | Molecule ID")
    lines.append("-" * 72)

    for i, (idx, mol) in enumerate(selected, start=1):
        mid = get_molecule_id(mol, idx)
        lines.append(f"{i:4d} | {idx:7d} | {mid}")

    lines.append("=" * 72)
    lines.append("")  # trailing blank line

    text = "\n".join(lines)

    # Print to stdout
    print(text)

    # Append to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly select molecule entries from an SDF file."
    )
    parser.add_argument(
        "sdf_file",
        nargs="?",
        default=None,
        help=f"SDF file path (default: {DEFAULT_SDF} in this directory)",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        required=True,
        help="Number of entries (mol blocks) to select randomly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-sdf",
        type=str,
        default=None,
        help="Output SDF path for selected entries (default: <input_basename>_selected_<n>.sdf).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="selection.log",
        help="Log file to append selection info to (default: selection.log in this directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve SDF path
    if args.sdf_file is not None:
        sdf_path = os.path.abspath(args.sdf_file)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sdf_path = os.path.join(script_dir, DEFAULT_SDF)

    # Determine output SDF path
    if args.output_sdf is not None:
        output_sdf = os.path.abspath(args.output_sdf)
    else:
        base, ext = os.path.splitext(os.path.basename(sdf_path))
        if not ext:
            ext = ".sdf"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_sdf = os.path.join(
            script_dir,
            f"{base}_selected_{args.num}{ext}",
        )

    # Prepare RNG
    rng = random.Random(args.seed)

    try:
        selected, total_considered = take_first_k_and_sample(
            sdf_path, args.num, MAX_CANDIDATES, rng
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_sdf), exist_ok=True)

    write_selected_sdf(selected, output_sdf)

    # Log selection
    log_path = os.path.abspath(args.log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_selection(sdf_path, selected, total_considered, output_sdf, log_path)


if __name__ == "__main__":
    main()

