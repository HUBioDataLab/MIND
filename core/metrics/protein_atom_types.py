#!/usr/bin/env python3
"""
Protein Atom Type Definitions and Groupings

Defines the 4+30+1 = 35 atom types used in the MIND 14-atom uniform representation
for proteins, along with groupings and helper functions for per-atom metric
computation.

Atom layout (matches data_loading/adapters/protein_adapter.py):
- 4 backbone atoms: N, CA, C, O (always present)
- Up to 10 sidechain slots: filled by real atoms from SIDECHAIN_ATOM_PRIORITY,
  then virtual atoms (V) for the remaining slots
- Virtual atoms have element 'V' (atomic_num 119); their pos_code is "V1".."V10"
  but they collapse to a single 'V' class in the metric vocabulary.

Parallels core/rna_atom_types.py; intentionally keeps the same API so that
PerAtomMetricsCalculator and the per-atom metric callback can be reused.
"""

from typing import Dict, List


# =============================================================================
# PROTEIN BACKBONE ATOMS (4 types) — always present in every residue
# =============================================================================
PROTEIN_BACKBONE_ATOMS: List[str] = ['N', 'CA', 'C', 'O']


# =============================================================================
# PROTEIN SIDECHAIN ATOMS (30 types)
# =============================================================================
# Priority-ordered list from data_loading/adapters/protein_adapter.py:33-39.
# Kept in the same order so that atom_idx_in_block in the standardized 14-atom
# representation maps deterministically to a position in this list.
PROTEIN_SIDECHAIN_ATOMS: List[str] = [
    'CB',
    'CG', 'CG1', 'CG2',
    'CD', 'CD1', 'CD2',
    'CE', 'CE1', 'CE2', 'CE3',
    'CZ', 'CZ2', 'CZ3',
    'CH2',
    'ND1', 'ND2',
    'NE', 'NE1', 'NE2',
    'NZ',
    'OD1', 'OD2',
    'OE1', 'OE2',
    'OG', 'OG1',
    'OH',
    'SG', 'SD',
]


# =============================================================================
# VIRTUAL ATOM — single class collapsing V1..V10
# =============================================================================
PROTEIN_VIRTUAL_ATOM = 'V'


# =============================================================================
# ALL PROTEIN ATOMS (vocabulary for metric computation)
# =============================================================================
ALL_PROTEIN_ATOMS: List[str] = (
    PROTEIN_BACKBONE_ATOMS + PROTEIN_SIDECHAIN_ATOMS + [PROTEIN_VIRTUAL_ATOM]
)  # 4 + 30 + 1 = 35


# =============================================================================
# ELEMENT LOOKUP — atomic number → element symbol
# =============================================================================
# Matches core/rna_atom_types.ATOMIC_NUM_TO_ELEMENT with 'S' added for proteins
# (cysteine SG, methionine SD).
ATOMIC_NUM_TO_ELEMENT: Dict[int, str] = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    15: 'P',
    16: 'S',
    17: 'Cl',
    35: 'Br',
    53: 'I',
    119: 'V',
}


# =============================================================================
# GROUP DEFINITIONS — for macro-level analysis in W&B
# =============================================================================
# These names become W&B keys like "val_protein_atom_metrics/group_Backbone/F1".
PROTEIN_ATOM_GROUPS: Dict[str, Dict[str, List[str]]] = {
    'backbone_vs_sidechain': {
        'Backbone': PROTEIN_BACKBONE_ATOMS,
        'SideChain': PROTEIN_SIDECHAIN_ATOMS,
        'Virtual': [PROTEIN_VIRTUAL_ATOM],
    },
    'sidechain_by_shell': {
        # Grouped by distance from CA (Greek letters): β, γ, δ, ε, ζ, η
        'Beta': ['CB'],
        'Gamma': ['CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG'],
        'Delta': ['CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD'],
        'Epsilon': ['CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2'],
        'Zeta': ['CZ', 'CZ2', 'CZ3', 'NZ'],
        'Eta': ['CH2', 'OH'],
    },
    'by_element': {
        # Useful when comparing "how well does the model predict each element class"
        'Carbon': ['CA', 'C', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2',
                   'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2'],
        'Nitrogen': ['N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NZ'],
        'Oxygen': ['O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH'],
        'Sulfur': ['SG', 'SD'],
        'Virtual': [PROTEIN_VIRTUAL_ATOM],
    },
}


# =============================================================================
# AMINO ACID LIST (20 standard) — used for per-AA metrics namespace
# =============================================================================
STANDARD_AMINO_ACIDS: List[str] = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def pos_code_to_element(pos_code: str) -> str:
    """
    Protein pos_code → element symbol.

    First character of the pos_code is the element for all 30+4 standard
    atoms (N, CA, C, O, CB, CG*, CD*, CE*, CZ*, CH2, ND*, NE*, NZ, OD*,
    OE*, OG*, OH, SG, SD). Virtual atoms ('V', 'V1'..'V10') collapse to 'V'.
    """
    if not pos_code:
        return 'V'
    if pos_code == 'V' or pos_code.startswith('V'):
        return 'V'
    return pos_code[0]


def pos_code_to_atom_class(pos_code: str) -> str:
    """
    Map a raw pos_code to its class in ALL_PROTEIN_ATOMS.

    Real atoms return their own pos_code (e.g., 'CA' → 'CA', 'OD1' → 'OD1').
    Virtual atoms ('V1', 'V2', ..., 'V10') collapse to the single 'V' class so
    the metric vocabulary stays at 35 entries regardless of slot index.
    Unknown pos_codes fall back to 'V' (treated as virtual/unknown).
    """
    if not pos_code:
        return PROTEIN_VIRTUAL_ATOM
    if pos_code == PROTEIN_VIRTUAL_ATOM or pos_code.startswith('V'):
        return PROTEIN_VIRTUAL_ATOM
    if pos_code in ALL_PROTEIN_ATOMS:
        return pos_code
    return PROTEIN_VIRTUAL_ATOM


def is_backbone_atom(pos_code: str) -> bool:
    return pos_code in PROTEIN_BACKBONE_ATOMS


def is_sidechain_atom(pos_code: str) -> bool:
    return pos_code in PROTEIN_SIDECHAIN_ATOMS


def is_virtual_atom(pos_code: str) -> bool:
    return pos_code == PROTEIN_VIRTUAL_ATOM or pos_code.startswith('V')


def is_sidechain_slot(pos_code: str) -> bool:
    """True if pos_code occupies a sidechain slot (real sidechain atom OR virtual padding).

    In the 14-atom layout (4 backbone + 10 sidechain slots), every non-backbone
    position is a "sidechain slot" — used for the block-level AA metric so
    sidechain accuracy is computed over the full 10-slot window regardless of
    how many slots are real vs virtual for a given residue.
    """
    return not is_backbone_atom(pos_code)


def get_atom_group(pos_code: str, group_type: str = 'backbone_vs_sidechain') -> str:
    atom_class = pos_code_to_atom_class(pos_code)
    groups = PROTEIN_ATOM_GROUPS.get(group_type, {})
    for group_name, atom_list in groups.items():
        if atom_class in atom_list:
            return group_name
    return 'Unknown'


def get_atom_type_vocabulary() -> Dict[str, int]:
    return {atom: i for i, atom in enumerate(ALL_PROTEIN_ATOMS)}


# =============================================================================
# CLI DIAGNOSTIC
# =============================================================================
def print_protein_atom_info() -> None:
    """Print vocabulary summary — run `python core/protein_atom_types.py`."""
    print("\n" + "=" * 80)
    print("PROTEIN ATOM TYPES VOCABULARY")
    print("=" * 80)
    print(f"\nTotal atom types: {len(ALL_PROTEIN_ATOMS)}")
    print(f"  - Backbone:  {len(PROTEIN_BACKBONE_ATOMS)}")
    print(f"  - Sidechain: {len(PROTEIN_SIDECHAIN_ATOMS)}")
    print(f"  - Virtual:   1 (collapses V1..V10)")

    print(f"\nBACKBONE ATOMS ({len(PROTEIN_BACKBONE_ATOMS)}):")
    for i, atom in enumerate(PROTEIN_BACKBONE_ATOMS):
        print(f"  [{i:2d}] {atom:5s} → element {pos_code_to_element(atom)}")

    print(f"\nSIDECHAIN ATOMS ({len(PROTEIN_SIDECHAIN_ATOMS)}):")
    for i, atom in enumerate(PROTEIN_SIDECHAIN_ATOMS, start=len(PROTEIN_BACKBONE_ATOMS)):
        print(f"  [{i:2d}] {atom:5s} → element {pos_code_to_element(atom)}")

    print(f"\nVIRTUAL: [34] V (collapses V1..V10)")

    print(f"\nGroupings defined:")
    for gname, subs in PROTEIN_ATOM_GROUPS.items():
        print(f"  {gname}: {list(subs.keys())}")

    print(f"\nAmino acids tracked for per-AA metrics: {len(STANDARD_AMINO_ACIDS)}")
    print(f"  {STANDARD_AMINO_ACIDS}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print_protein_atom_info()
    print("Sanity checks:")
    print(f"  pos_code_to_element('CA')    = {pos_code_to_element('CA')}")
    print(f"  pos_code_to_element('OD1')   = {pos_code_to_element('OD1')}")
    print(f"  pos_code_to_element('SG')    = {pos_code_to_element('SG')}")
    print(f"  pos_code_to_element('V3')    = {pos_code_to_element('V3')}")
    print(f"  pos_code_to_atom_class('V5') = {pos_code_to_atom_class('V5')}")
    print(f"  is_backbone_atom('CA')       = {is_backbone_atom('CA')}")
    print(f"  is_sidechain_atom('OD1')     = {is_sidechain_atom('OD1')}")
    print(f"  is_virtual_atom('V7')        = {is_virtual_atom('V7')}")
    print(f"  get_atom_group('CG1')        = {get_atom_group('CG1')}")
    print(f"  get_atom_group('CG1', 'sidechain_by_shell') = {get_atom_group('CG1', 'sidechain_by_shell')}")
