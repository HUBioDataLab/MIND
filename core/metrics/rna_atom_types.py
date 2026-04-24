#!/usr/bin/env python3
"""
RNA Atom Type Definitions and Groupings

Defines the 11+12 atom types in RNA molecules and groups them into:
- Backbone: Phosphate and sugar atoms (12 types)
- SideChain: Nitrogenous base atoms (11 types)

This module provides vocabulary for per-atom metrics calculation.
"""

from typing import Dict, List, Set, Tuple

# =============================================================================
# RNA BACKBONE ATOMS (12 types)
# =============================================================================
# These are the standard atoms in the RNA phosphate-sugar backbone
RNA_BACKBONE_ATOMS: List[str] = [
    'P',      # Phosphorus
    'OP1',    # Oxygen phosphate 1
    'OP2',    # Oxygen phosphate 2
    "O5'",    # 5' oxygen (bridging)
    "C5'",    # 5' carbon (sugar)
    "C4'",    # 4' carbon (sugar)
    "O4'",    # 4' oxygen (sugar ring)
    "C3'",    # 3' carbon (sugar)
    "O3'",    # 3' oxygen (bridging)
    "C2'",    # 2' carbon (sugar)
    "O2'",    # 2' hydroxyl (RNA-specific)
    "C1'",    # 1' carbon (anomeric - connects to base)
]

# =============================================================================
# RNA SIDECHAIN ATOMS (11 types - base atoms)
# =============================================================================
# These are atoms in the nitrogenous bases (A, G, C, U)
RNA_SIDECHAIN_ATOMS: List[str] = [
    'N9',     # Purine attachment (A, G)
    'C8',     # Purine ring
    'N7',     # Purine ring
    'C6',     # Ring carbon
    'C5',     # Ring carbon
    'C4',     # Ring carbon
    'N3',     # Ring nitrogen
    'C2',     # Ring carbon
    'N1',     # Ring nitrogen / Pyrimidine attachment (C, U)
    'N2',     # Amino group (G, modified C, A)
    'O6',     # Carbonyl (G) / alternate
]

# Alternative: Explicitly list by base for reference
RNA_PURINE_ATOMS = ['N9', 'C8', 'N7', 'C6', 'C5', 'C4', 'N3', 'C2', 'N1', 'N6', 'O6']
RNA_PYRIMIDINE_ATOMS = ['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N4', 'O2', 'O4', 'N2', 'O6']

# =============================================================================
# VIRTUAL ATOM (used in 23-atom uniform representation)
# =============================================================================
RNA_VIRTUAL_ATOM = 'V'

# =============================================================================
# ALL RNA ATOMS
# =============================================================================
ALL_RNA_ATOMS: List[str] = RNA_BACKBONE_ATOMS + RNA_SIDECHAIN_ATOMS + [RNA_VIRTUAL_ATOM]

# =============================================================================
# GROUP DEFINITIONS (for granular analysis)
# =============================================================================
RNA_ATOM_GROUPS: Dict[str, Dict[str, List[str]]] = {
    'backbone_vs_sidechain': {
        'Backbone': RNA_BACKBONE_ATOMS,
        'SideChain': RNA_SIDECHAIN_ATOMS,
        'Virtual': [RNA_VIRTUAL_ATOM],
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
    return atom_name in RNA_BACKBONE_ATOMS

def is_sidechain_atom(atom_name: str) -> bool:
    """Check if atom is part of sidechain."""
    return atom_name in RNA_SIDECHAIN_ATOMS

def is_virtual_atom(atom_name: str) -> bool:
    """Check if atom is virtual."""
    return atom_name == RNA_VIRTUAL_ATOM

def get_atom_group(atom_name: str, group_type: str = 'backbone_vs_sidechain') -> str:
    """Get the group name for an atom."""
    groups = RNA_ATOM_GROUPS.get(group_type, {})
    for group_name, atom_list in groups.items():
        if atom_name in atom_list:
            return group_name
    return 'Unknown'

def get_get_code(atom_name: str) -> str:
    """Get GET position code for an RNA atom."""
    return RNA_ATOM_TO_GET_CODE.get(atom_name, "'")

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
    print(f"   - Virtual: 1")
    
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
    element = ATOMIC_NUM_TO_ELEMENT.get(atomic_num, 'V')
    
    # Element ismine göre ALL_RNA_ATOMS içindeki İLK eşleşen indeksi döndür
    for i, atom_name in enumerate(ALL_RNA_ATOMS):
        if atom_name.startswith(element):
            return i
    return ALL_RNA_ATOMS.index('V')


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
