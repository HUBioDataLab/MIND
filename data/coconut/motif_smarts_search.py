#!/usr/bin/env python3
"""
Find common structural motifs in molecules using RDKit SMARTS.

Two main modes:
  1) Single SMILES passed via CLI:
         python motif_smarts_search.py "CCOC1=CC=CC=C1"
  2) File with one SMILES per line:
         python motif_smarts_search.py --smiles-file input.smi

For each input SMILES, the script:
  - parses the molecule with RDKit
  - searches for a set of predefined SMARTS motifs
  - prints a human-readable report with match counts and atom indices.

You can customize / extend the built-in motif set by editing MOTIF_SMARTS.
"""

import argparse
import os
import sys
from typing import Dict, Iterable, List, Tuple

try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit is required. Install with: pip install rdkit", file=sys.stderr)
    sys.exit(1)


# Motif set tuned for small molecules / metabolites.
# The keys are human-readable names; values are SMARTS patterns.
MOTIF_SMARTS: Dict[str, str] = {
    # Aromatics / rings
    "aromatic_6_ring": "a1aaaaa1",                 # generic 6-member aromatic ring
    "benzene_core": "c1ccccc1",                    # benzene-like core
    "heteroaromatic_5_ring": "[n,o,s]1cccc1",      # simple 5-member hetero aromatic

    # Carbonyls and related
    "ketone": "[CX3](=O)[#6]",                     # R-CO-R'
    "aldehyde": "[CX3H1](=O)[#6]",                 # R-CHO
    "amide": "[NX3;H1,H2][CX3](=[OX1])",           # -NH-C(=O)-
    "ester": "[CX3](=O)[OX2H0][#6]",               # -C(=O)-O-R
    "carboxylic_acid": "C(=O)[O;H1,-1]",          # -COOH / -COO-
    "carboxylate": "[CX3](=O)[O-]",                # deprotonated acid
    "urea_like": "N=C(N)N",                        # simple urea core

    # Amines / cations
    "primary_amine": "[NX3;H2][#6]",               # -NH2-R
    "secondary_amine": "[NX3;H1]([#6])[#6]",       # -NHR2
    "tertiary_amine": "[NX3]([#6])([#6])[#6]",     # -NR3
    "quaternary_ammonium": "[NX4+]",               # positively charged N

    # Alcohols, phenols, ethers
    "alcohol": "[OX2H][CX4]",                      # aliphatic -CH-OH
    "phenol": "c[OX2H]",                           # aromatic -OH
    "ether": "[OD2]([#6])[#6]",                    # -O- between carbons

    # Sulfur-containing
    "thiol": "[SX2H]",                             # -SH
    "thioether": "[#16X2]([#6])[#6]",              # R-S-R'
    "sulfone": "S(=O)(=O)[#6]",                    # sulfone
    "sulfonamide": "S(=O)(=O)N",                   # sulfonamide

    # Phosphate / phosphonate (common in metabolites)
    "phosphate_mono": "P(=O)(O)(O)[O;H1,-1]",      # PO4H / PO4-
    "phosphonate": "P(=O)(O)(O)[#6]",              # C-PO3

    # Heterocycles / heteroatoms
    "sp2_nitrogen": "[nX2]",                       # aromatic / sp2 N
    "pyridine_like_N": "n1ccccc1",                 # pyridine N
    "imidazole_like": "n1cc[nH]c1",                # imidazole-ish motif

    # Carbohydrate-like
    "sugar_like_oh_chain": "[CX4H]([OX2H])[CX4H]([OX2H])[CX4H]",  # rough polyol chain

    # Miscellaneous functional groups
    "nitrile": "C#N",                              # -C≡N
    "alkyne": "C#C",                               # C≡C
    "alkene": "C=C",                               # simple C=C
    "halogen": "[F,Cl,Br,I]",                      # any halogen
}


def build_motif_queries() -> Dict[str, Chem.Mol]:
    """Compile SMARTS patterns to RDKit Mol objects, skipping invalid ones."""
    queries: Dict[str, Chem.Mol] = {}
    for name, smarts in MOTIF_SMARTS.items():
        q = Chem.MolFromSmarts(smarts)
        if q is None:
            print(
                f"Warning: invalid SMARTS for motif '{name}': {smarts}",
                file=sys.stderr,
            )
            continue
        queries[name] = q
    return queries


def iter_smiles_from_file(path: str) -> Iterable[str]:
    """Yield SMILES strings (one per non-empty, non-comment line) from a file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line.split()[0]


def find_motifs_in_smiles(
    smiles: str, motif_queries: Dict[str, Chem.Mol]
) -> Tuple[Chem.Mol, Dict[str, List[Tuple[int, ...]]], str]:
    """
    For a SMILES string, return:
      - parsed RDKit Mol (or None)
      - dict: motif_name -> list of atom index tuples (matches)
      - error message if parsing failed, else "".
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {}, "Failed to parse SMILES"

    results: Dict[str, List[Tuple[int, ...]]] = {}
    for name, q in motif_queries.items():
        matches = mol.GetSubstructMatches(q)
        if matches:
            results[name] = list(matches)

    return mol, results, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find structural motifs in molecules using RDKit SMARTS."
    )
    parser.add_argument(
        "smiles",
        nargs="?",
        default=None,
        help="Single SMILES string to analyze (if not using --smiles-file).",
    )
    parser.add_argument(
        "--smiles-file",
        type=str,
        default=None,
        help="File with one SMILES per line.",
    )
    parser.add_argument(
        "--max-matches-per-motif",
        type=int,
        default=50,
        help="Maximum number of matches to print per motif (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.smiles is None and args.smiles_file is None:
        print(
            "Error: Provide either a SMILES string or --smiles-file.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.smiles is not None and args.smiles_file is not None:
        print(
            "Error: Use either a single SMILES or --smiles-file, not both.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Prepare input iterable
    if args.smiles_file is not None:
        if not os.path.isfile(args.smiles_file):
            print(f"Error: SMILES file not found: {args.smiles_file}", file=sys.stderr)
            sys.exit(1)
        smiles_iter = iter_smiles_from_file(args.smiles_file)
    else:
        smiles_iter = [args.smiles]

    motif_queries = build_motif_queries()
    if not motif_queries:
        print("Error: no valid SMARTS motifs defined.", file=sys.stderr)
        sys.exit(1)

    first = True
    for s in smiles_iter:
        mol, motif_hits, err = find_motifs_in_smiles(s, motif_queries)

        if not first:
            print()
        first = False

        print("=" * 80)
        print(f"Input SMILES : {s}")

        if err:
            print(f"Status       : ERROR - {err}")
            print("=" * 80)
            continue

        if not motif_hits:
            print("Status       : OK (no motifs found)")
            print("=" * 80)
            continue

        print("Status       : OK")
        print(f"Motifs found : {len(motif_hits)} motif types")
        print("-" * 80)

        for motif_name, matches in motif_hits.items():
            count = len(matches)
            print(f"Motif: {motif_name}")
            print(f"  SMARTS     : {MOTIF_SMARTS.get(motif_name, '<unknown>')}")
            print(f"  Occurrences: {count}")
            if count > args.max_matches_per_motif:
                shown = matches[: args.max_matches_per_motif]
                print(f"  Showing first {args.max_matches_per_motif} matches:")
            else:
                shown = matches
            for i, idx_tuple in enumerate(shown, start=1):
                idx_str = ", ".join(str(idx) for idx in idx_tuple)
                print(f"    #{i:3d}: atoms [{idx_str}]")
            if count > args.max_matches_per_motif:
                print(f"  ... {count - args.max_matches_per_motif} more matches omitted")
            print("-" * 80)

        print("=" * 80)


if __name__ == "__main__":
    main()

