#!/usr/bin/env python3
"""
Fragment molecules given as SMILES strings using RDKit BRICS.

CLI modes:
  1) Single SMILES:
         python brics_fragment_smiles.py "CCOC1=CC=CC=C1"

  2) File with one SMILES per line:
         python brics_fragment_smiles.py --smiles-file input.smi --output fragments.tsv

  3) CSV with SMILES column (distribution of fragment counts):
         python brics_fragment_smiles.py \\
             --smiles-csv coconut_unique_smiles_from_field.csv \\
             --smiles-column smiles \\
             --hist-prefix coconut_fragments

     This will generate a matplotlib histogram PNG showing the distribution of
     fragment counts across the dataset.
"""

import argparse
import os
import sys
from typing import Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from rdkit import Chem
    from rdkit.Chem import BRICS
except ImportError:
    print("Error: RDKit is required. Install with: pip install rdkit", file=sys.stderr)
    sys.exit(1)


def _mol_from_smiles(smiles: str):
    """Parse SMILES; returns (mol, error_string). error_string is non-empty on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Failed to parse SMILES"
    return mol, ""


def fragment_smiles_brics(smiles: str) -> Tuple[List[str], str]:
    """
    Fragment a SMILES string with BRICS and return fragment SMILES and an error (if any).
    """
    mol, err = _mol_from_smiles(smiles)
    if mol is None:
        return [], err
    try:
        frag_set = BRICS.BRICSDecompose(mol, singlePass=False)
    except Exception as e:
        return [], f"BRICS decomposition failed: {e}"
    return sorted(frag_set), ""


    return sorted(frag_set), ""


def iter_smiles_from_file(path: str) -> Iterable[str]:
    """Yield SMILES strings (one per non-empty line) from a file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line.split()[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fragment SMILES using RDKit BRICS."
    )
    parser.add_argument(
        "smiles",
        nargs="?",
        default=None,
        help="Single SMILES string to fragment (if not using --smiles-file / --smiles-csv).",
    )
    parser.add_argument(
        "--smiles-file",
        type=str,
        default=None,
        help="Text file with one SMILES per line.",
    )
    parser.add_argument(
        "--smiles-csv",
        type=str,
        default=None,
        help="CSV file containing a SMILES column (use --smiles-column to select the column).",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="smiles",
        help="Name of the SMILES column in the CSV file (default: 'smiles').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output TSV file of fragments. If omitted, a human-readable report is printed "
            "to stdout instead of TSV. Ignored when --smiles-csv is used."
        ),
    )
    parser.add_argument(
        "--hist-prefix",
        type=str,
        default="fragments_hist",
        help=(
            "Prefix for output histogram PNG files when using --smiles-csv "
            "(one file per method). Default: 'fragments_hist'."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display matplotlib figures (useful in batch mode).",
    )
    return parser.parse_args()


def _run_csv_histograms(
    csv_path: str,
    smiles_column: str,
    hist_prefix: str,
    no_show: bool,
) -> None:
    """
    Read SMILES from a CSV file and generate matplotlib histograms of
    fragment-count distributions for each selected method.
    """
    if plt is None:
        print(
            "Error: matplotlib is required for histogram generation. "
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    counts: List[int] = []
    total_rows = 0
    skipped_rows = 0

    # Delayed import to avoid requiring csv module when not needed
    import csv  # type: ignore

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if smiles_column not in (reader.fieldnames or []):
            print(
                f"Error: SMILES column '{smiles_column}' not found in CSV. "
                f"Available columns: {', '.join(reader.fieldnames or [])}",
                file=sys.stderr,
            )
            sys.exit(1)

        for row in reader:
            total_rows += 1
            s = (row.get(smiles_column) or "").strip()
            if not s:
                skipped_rows += 1
                continue

            fragments, err = fragment_smiles_brics(s)
            if err and not fragments:
                # Skip molecules that could not be processed
                continue
            counts.append(len(fragments))

    if total_rows == 0:
        print("Warning: CSV file appears to be empty.", file=sys.stderr)
        return

    print(
        f"Processed {total_rows} rows from CSV, skipped {skipped_rows} with empty SMILES.",
        file=sys.stderr,
    )

    # Create a single histogram for BRICS
    max_count = max(counts)
    bins = range(0, max_count + 2)  # integer bins

    plt.figure()
    plt.hist(counts, bins=bins, edgecolor="black", align="left")
    plt.xlabel("Number of BRICS fragments")
    plt.ylabel("Molecule count")
    plt.title("Fragment count distribution - BRICS")
    plt.tight_layout()

    out_path = f"{hist_prefix}_brics.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved histogram for BRICS to: {out_path}", file=sys.stderr)

    if not no_show:
        plt.show()

    plt.close()


def main() -> None:
    args = parse_args()

    # Determine which input mode is used
    input_modes = [
        bool(args.smiles),
        bool(args.smiles_file),
        bool(args.smiles_csv),
    ]
    if sum(input_modes) == 0:
        print(
            "Error: Provide one of: a SMILES string, --smiles-file, or --smiles-csv.",
            file=sys.stderr,
        )
        sys.exit(1)
    if sum(input_modes) > 1:
        print(
            "Error: Use only one of: single SMILES, --smiles-file, or --smiles-csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    # CSV mode: generate histograms of fragment-count distributions
    if args.smiles_csv:
        _run_csv_histograms(
            csv_path=args.smiles_csv,
            smiles_column=args.smiles_column,
            hist_prefix=args.hist_prefix,
            no_show=args.no_show,
        )
        return

    # Line-based SMILES input modes
    if args.smiles_file is not None:
        if not os.path.isfile(args.smiles_file):
            print(f"Error: SMILES file not found: {args.smiles_file}", file=sys.stderr)
            sys.exit(1)
        smiles_iter = iter_smiles_from_file(args.smiles_file)
    else:
        smiles_iter = [args.smiles]

    # If an output file is given, write machine-friendly TSV there.
    if args.output:
        out_path = os.path.abspath(args.output)
        if os.path.dirname(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as out_f:
            print("input_smiles\tfragment_smiles\terror", file=out_f)
            for s in smiles_iter:
                fragments, err = fragment_smiles_brics(s)
                if fragments:
                    for frag in fragments:
                        print(f"{s}\t{frag}\t{err}", file=out_f)
                else:
                    print(f"{s}\t\t{err or 'no_fragments'}", file=out_f)
        return

    # Human-readable report to stdout.
    first = True
    for s in smiles_iter:
        if not first:
            print()
        first = False

        print("=" * 72)
        print(f"Input SMILES : {s}")
        fragments, err = fragment_smiles_brics(s)
        if err:
            print(f"Status       : ERROR - {err}")
            print("=" * 72)
            continue
        if not fragments:
            print("Status       : OK (no BRICS fragments found)")
            print("=" * 72)
            continue
        print("Status       : OK")
        print(f"Fragments    : {len(fragments)}")
        print("-" * 72)
        print(f"{'Frag#':>5} | Fragment SMILES")
        print("-" * 72)
        for i, frag in enumerate(fragments, start=1):
            print(f"{i:5d} | {frag}")
        print("=" * 72)


if __name__ == "__main__":
    main()

