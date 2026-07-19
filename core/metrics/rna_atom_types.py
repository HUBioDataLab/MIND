#!/usr/bin/env python3
"""
RNA Atom Type Definitions and Groupings

Defines positional atom types in RNA molecules and groups them into:
- Backbone: Phosphate and sugar atoms (12 types)
- SideChain: Nitrogenous-base position labels (18 recognized labels)

This module provides vocabulary for per-atom metrics calculation.
"""

from typing import Dict, List

from data_loading.atom_position_vocab import (
    RNA_BACKBONE_ATOMS as SHARED_RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOMS_BY_NUCLEOTIDE,
    RNA_BASE_ATOM_PRIORITY,
    RNA_TERMINAL_ATOMS as SHARED_RNA_TERMINAL_ATOMS,
    is_virtual_position_code,
    normalize_rna_position_code,
)

# =============================================================================
# RNA BACKBONE ATOMS (12 types)
# =============================================================================
# These are the standard atoms in the RNA phosphate-sugar backbone
RNA_BACKBONE_ATOMS: List[str] = list(SHARED_RNA_BACKBONE_ATOMS)

# =============================================================================
# RNA SIDECHAIN POSITION LABELS
# =============================================================================
# A block has at most 11 base slots, while A/G/C/U collectively use 15
# distinct labels. Modified-base labels recognized by the adapter are included
# too so they cannot silently enter the virtual row.
RNA_SIDECHAIN_ATOMS: List[str] = list(RNA_BASE_ATOM_PRIORITY)
RNA_TERMINAL_ATOMS: List[str] = list(SHARED_RNA_TERMINAL_ATOMS)

# Alternative: Explicitly list by base for reference
RNA_PURINE_ATOMS = list(dict.fromkeys(
    RNA_BASE_ATOMS_BY_NUCLEOTIDE['A'] + RNA_BASE_ATOMS_BY_NUCLEOTIDE['G']
))
RNA_PYRIMIDINE_ATOMS = list(dict.fromkeys(
    RNA_BASE_ATOMS_BY_NUCLEOTIDE['C'] + RNA_BASE_ATOMS_BY_NUCLEOTIDE['U']
))

# =============================================================================
# VIRTUAL ATOM (used in 23-atom uniform representation)
# =============================================================================
RNA_VIRTUAL_ATOM = 'V'
RNA_UNKNOWN_ATOM = 'UNK'

# =============================================================================
# ALL RNA ATOMS
# =============================================================================
ALL_RNA_ATOMS: List[str] = (
    RNA_BACKBONE_ATOMS
    + RNA_SIDECHAIN_ATOMS
    + RNA_TERMINAL_ATOMS
    + [RNA_VIRTUAL_ATOM, RNA_UNKNOWN_ATOM]
)

# =============================================================================
# GROUP DEFINITIONS (for granular analysis)
# =============================================================================
RNA_ATOM_GROUPS: Dict[str, Dict[str, List[str]]] = {
    'backbone_vs_sidechain': {
        'Backbone': RNA_BACKBONE_ATOMS,
        'SideChain': RNA_SIDECHAIN_ATOMS,
        'Terminal': RNA_TERMINAL_ATOMS,
        'Virtual': [RNA_VIRTUAL_ATOM],
        'Unknown': [RNA_UNKNOWN_ATOM],
    },
    'backbone_by_function': {
        'Phosphate': ['P', 'OP1', 'OP2'],
        'Sugar': ["O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"],
    },
    'base_by_type': {
        'Purine': RNA_PURINE_ATOMS,
        'Pyrimidine': RNA_PYRIMIDINE_ATOMS,
    }
}

# =============================================================================
# POSITION CODE MAPPINGS (for GET vocabulary)
# =============================================================================
# Maps RNA atom names to GET position codes
RNA_ATOM_TO_GET_CODE: Dict[str, str] = {
    # Phosphate backbone
    'P': 'A',       # Phosphorus -> A (analogous to N backbone)
    'OP1': 'A',     # Phosphate oxygen 1
    'OP2': 'A',     # Phosphate oxygen 2
    
    # Sugar ring
    "O5'": 'B',     # 5' oxygen -> B (analogous to CA)
    "O3'": 'B',     # 3' oxygen -> B
    "C5'": 'E',     # 5' carbon -> E (analogous to CB)
    "C4'": 'G',     # 4' carbon -> G (analogous to C)
    "C3'": 'G',     # 3' carbon
    "C2'": 'G',     # 2' carbon
    "C1'": 'G',     # 1' carbon
    "O4'": 'D',     # Sugar oxygen
    "O2'": 'D',     # 2' hydroxyl -> D
    
    # Base atoms - general mapping
    'N9': 'Z',      # Purine attachment
    'C8': 'Z',
    'N7': 'Z',
    'C6': 'H',      # Ring carbons
    'C5': 'H',
    'C4': 'H',
    'C2': 'H',
    'N3': 'H',      # Ring nitrogens
    'N1': 'H',      # Pyrimidine attachment
    'N2': 'P',      # Amino groups
    'N6': 'P',
    'O6': 'P',      # Carbonyl
    'O4': 'D',
    'O2': 'D',
    'N4': 'P',
    'N5': 'P',
    'N8': 'P',
    'C7': 'H',
    
    # Virtual atom
    'V': 'm',       # Virtual -> mask
}

# =============================================================================
# ATOM TYPE VOCABULARY (indices for one-hot or embedding lookups)
# =============================================================================
def get_atom_type_vocabulary() -> Dict[str, int]:
    """Get atom type to index mapping."""
    vocab = {}
    for i, atom in enumerate(ALL_RNA_ATOMS):
        vocab[atom] = i
    return vocab

def get_inverse_atom_type_vocabulary() -> Dict[int, str]:
    """Get index to atom type mapping."""
    vocab = get_atom_type_vocabulary()
    return {v: k for k, v in vocab.items()}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_backbone_atom(atom_name: str) -> bool:
    """Check if atom is part of backbone."""
    return normalize_rna_position_code(atom_name) in RNA_BACKBONE_ATOMS

def is_sidechain_atom(atom_name: str) -> bool:
    """Check if atom is part of sidechain."""
    return normalize_rna_position_code(atom_name) in RNA_SIDECHAIN_ATOMS

def is_terminal_atom(atom_name: str) -> bool:
    """Check if atom is an optional terminal phosphate position."""
    return normalize_rna_position_code(atom_name) in RNA_TERMINAL_ATOMS

def is_virtual_atom(atom_name: str) -> bool:
    """Check if atom is virtual."""
    return is_virtual_position_code(atom_name)

def pos_code_to_element(pos_code: str) -> str:
    """Map an RNA position label to its chemical element or metric sentinel."""
    code = normalize_rna_position_code(pos_code)
    if not code:
        return RNA_UNKNOWN_ATOM
    if is_virtual_position_code(code):
        return RNA_VIRTUAL_ATOM
    return code[0]

def pos_code_to_atom_class(pos_code: str) -> str:
    """Map only real virtual padding to V; unexpected real labels become UNK."""
    code = normalize_rna_position_code(pos_code)
    if not code:
        return RNA_UNKNOWN_ATOM
    if is_virtual_position_code(code):
        return RNA_VIRTUAL_ATOM
    if code in ALL_RNA_ATOMS:
        return code
    return RNA_UNKNOWN_ATOM

def get_atom_group(atom_name: str, group_type: str = 'backbone_vs_sidechain') -> str:
    """Get the group name for an atom."""
    atom_name = pos_code_to_atom_class(atom_name)
    groups = RNA_ATOM_GROUPS.get(group_type, {})
    for group_name, atom_list in groups.items():
        if atom_name in atom_list:
            return group_name
    return 'Unknown'

def get_get_code(atom_name: str) -> str:
    """Get GET position code for an RNA atom."""
    return RNA_ATOM_TO_GET_CODE.get(normalize_rna_position_code(atom_name), "'")

# =============================================================================
# SUMMARY AND PRINTING
# =============================================================================

def print_rna_atom_info():
    """Print comprehensive information about RNA atom types."""
    print("\n" + "="*80)
    print("RNA ATOM TYPES VOCABULARY")
    print("="*80)
    
    print(f"\n📍 Total Atom Types: {len(ALL_RNA_ATOMS)}")
    print(f"   - Backbone: {len(RNA_BACKBONE_ATOMS)}")
    print(f"   - SideChain: {len(RNA_SIDECHAIN_ATOMS)}")
    print(f"   - Terminal: {len(RNA_TERMINAL_ATOMS)}")
    print(f"   - Virtual: 1")
    print(f"   - Unknown: 1")
    
    print(f"\n🔗 BACKBONE ATOMS ({len(RNA_BACKBONE_ATOMS)}):")
    for i, atom in enumerate(RNA_BACKBONE_ATOMS, 1):
        code = RNA_ATOM_TO_GET_CODE.get(atom, '?')
        print(f"   {i:2d}. {atom:5s} → GET code: {code}")
    
    print(f"\n🧬 SIDECHAIN ATOMS ({len(RNA_SIDECHAIN_ATOMS)}):")
    for i, atom in enumerate(RNA_SIDECHAIN_ATOMS, 1):
        code = RNA_ATOM_TO_GET_CODE.get(atom, '?')
        print(f"   {i:2d}. {atom:5s} → GET code: {code}")
    
    print(f"\n🔷 VIRTUAL ATOM:")
    print(f"    V → GET code: m")
    
    print("\n" + "="*80 + "\n")

# ... (mevcut kodun devamına ekle)

# Modelin tahmin ettiği atom numaralarını genel element isimlerine çevirir
ATOMIC_NUM_TO_ELEMENT = {
    15: 'P',
    8: 'O',
    6: 'C',
    7: 'N',
    1: 'H',
    9: 'F',
    16: 'S',
    17: 'Cl',
    35: 'Br',
    53: 'I',
    119: 'V'
}

def map_atomic_num_to_rna_idx(atomic_num: int) -> int:
    """
    Modelin bastığı atom numarasını (örn: 15), 
    ALL_RNA_ATOMS listesindeki en mantıklı indekse (örn: 0) çevirir.
    """
    element = ATOMIC_NUM_TO_ELEMENT.get(atomic_num, RNA_UNKNOWN_ATOM)
    
    # Element ismine göre ALL_RNA_ATOMS içindeki İLK eşleşen indeksi döndür
    for i, atom_name in enumerate(ALL_RNA_ATOMS):
        if atom_name.startswith(element):
            return i
    return ALL_RNA_ATOMS.index(RNA_UNKNOWN_ATOM)


# Run if executed directly
if __name__ == "__main__":
    print_rna_atom_info()
    
    # Test utility functions
    print("Testing utility functions:")
    print(f"  is_backbone_atom('P'): {is_backbone_atom('P')}")
    print(f"  is_sidechain_atom('N9'): {is_sidechain_atom('N9')}")
    print(f"  is_virtual_atom('V'): {is_virtual_atom('V')}")
    print(f"  get_atom_group('P'): {get_atom_group('P')}")
    print(f"  get_atom_group('N9'): {get_atom_group('N9')}")
