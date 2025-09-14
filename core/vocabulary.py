#!/usr/bin/env python3
"""
Universal Vocabulary Mappings

Exact GET vocabulary mappings for universal molecular representation.
This ensures compatibility with GET's proven tokenization system.
"""

from typing import Dict, Any

# =============================================================================
# BLOCK SYMBOL MAPPINGS (EXACT GET VOCABULARY)
# =============================================================================

def get_block_symbol_mappings() -> Dict[str, int]:
    """
    Get block symbol mappings - EXACT GET VOCABULARY
    
    Returns:
        Dict mapping block symbols to indices
    """
    return {
        # Special tokens (GET order)
        '#': 0,   # PAD
        '*': 1,   # MASK  
        '?': 2,   # UNK
        '&': 3,   # GLB (global node)
        
        # Amino acids (GET single letter codes)
        'G': 4,   # GLY
        'A': 5,   # ALA
        'V': 6,   # VAL
        'L': 7,   # LEU
        'I': 8,   # ILE
        'F': 9,   # PHE
        'W': 10,  # TRP
        'Y': 11,  # TYR
        'D': 12,  # ASP
        'H': 13,  # HIS
        'N': 14,  # ASN
        'E': 15,  # GLU
        'K': 16,  # LYS
        'Q': 17,  # GLN
        'M': 18,  # MET
        'R': 19,  # ARG
        'S': 20,  # SER
        'T': 21,  # THR
        'C': 22,  # CYS
        'P': 23,  # PRO
        
        # DNA/RNA bases
        'DA': 24, 'DG': 25, 'DC': 26, 'DT': 27,  # DNA
        'RA': 28, 'RG': 29, 'RC': 30, 'RU': 31,  # RNA
        
        # Small molecule atoms (GET lowercase)
        'h': 32, 'he': 33, 'li': 34, 'be': 35, 'b': 36, 'c': 37, 'n': 38, 'o': 39, 'f': 40,
        'ne': 41, 'na': 42, 'mg': 43, 'al': 44, 'si': 45, 'p': 46, 's': 47, 'cl': 48, 'ar': 49,
        'k': 50, 'ca': 51, 'sc': 52, 'ti': 53, 'v': 54, 'cr': 55, 'mn': 56, 'fe': 57, 'co': 58,
        'ni': 59, 'cu': 60, 'zn': 61, 'ga': 62, 'ge': 63, 'as': 64, 'se': 65, 'br': 66, 'kr': 67,
        'rb': 68, 'sr': 69, 'y': 70, 'zr': 71, 'nb': 72, 'mo': 73, 'tc': 74, 'ru': 75, 'rh': 76,
        'pd': 77, 'ag': 78, 'cd': 79, 'in': 80, 'sn': 81, 'sb': 82, 'te': 83, 'i': 84, 'xe': 85,
        'cs': 86, 'ba': 87, 'la': 88, 'ce': 89, 'pr': 90, 'nd': 91, 'pm': 92, 'sm': 93, 'eu': 94,
        'gd': 95, 'tb': 96, 'dy': 97, 'ho': 98, 'er': 99, 'tm': 100, 'yb': 101, 'lu': 102,
        'hf': 103, 'ta': 104, 'w': 105, 're': 106, 'os': 107, 'ir': 108, 'pt': 109, 'au': 110,
        'hg': 111, 'tl': 112, 'pb': 113, 'bi': 114, 'po': 115, 'at': 116, 'rn': 117,
        'fr': 118, 'ra': 119, 'ac': 120, 'th': 121, 'pa': 122, 'u': 123, 'np': 124, 'pu': 125,
        'am': 126, 'cm': 127, 'bk': 128, 'cf': 129, 'es': 130, 'fm': 131, 'md': 132, 'no': 133,
        'lr': 134, 'rf': 135, 'db': 136, 'sg': 137, 'bh': 138, 'hs': 139, 'mt': 140, 'ds': 141,
        'rg': 142, 'cn': 143, 'nh': 144, 'fl': 145, 'mc': 146, 'lv': 147, 'ts': 148, 'og': 149,
    }

# =============================================================================
# POSITION CODE MAPPINGS (EXACT GET VOCABULARY)
# =============================================================================

def get_position_code_mappings() -> Dict[str, int]:
    """
    Get position code mappings - EXACT GET VOCABULARY
    
    Returns:
        Dict mapping position codes to indices
    """
    return {
        # GET position codes
        'p': 0,    # pad
        'm': 1,    # mask
        'g': 2,    # global
        '': 3,     # empty
        'A': 4,    # A
        'B': 5,    # B
        'G': 6,    # G
        'D': 7,    # D
        'E': 8,    # E
        'Z': 9,    # Z
        'H': 10,   # H
        'XT': 11,  # XT
        'P': 12,   # P
        'sm': 13,  # small molecule
        "'": 14,   # base
    }

def map_lba_position_codes_to_get(pos_code: str) -> str:
    """
    Map LBA position codes to GET vocabulary codes
    
    Args:
        pos_code: LBA position code (e.g., 'N', 'CA', 'C', etc.)
        
    Returns:
        GET vocabulary code
    """
    # Mapping from LBA position codes to GET vocabulary
    lba_to_get_mapping = {
        # Protein backbone atoms
        'N': 'A',      # Nitrogen backbone -> A
        'CA': 'B',     # Alpha carbon -> B  
        'C': 'G',      # Carbonyl carbon -> G
        'O': 'D',      # Carbonyl oxygen -> D
        
        # Protein side chain atoms - Beta carbons
        'CB': 'E',     # Beta carbon -> E
        
        # Protein side chain atoms - Gamma carbons
        'CG': 'Z',     # Gamma carbon -> Z
        'CG1': 'Z',    # Gamma carbon 1 -> Z
        'CG2': 'Z',    # Gamma carbon 2 -> Z
        
        # Protein side chain atoms - Delta carbons
        'CD': 'H',     # Delta carbon -> H
        'CD1': 'H',    # Delta carbon 1 -> H
        'CD2': 'H',    # Delta carbon 2 -> H
        
        # Protein side chain atoms - Epsilon carbons
        'CE': 'XT',    # Epsilon carbon -> XT
        'CE1': 'XT',   # Epsilon carbon 1 -> XT
        'CE2': 'XT',   # Epsilon carbon 2 -> XT
        'CE3': 'XT',   # Epsilon carbon 3 -> XT
        
        # Protein side chain atoms - Zeta carbons
        'CZ': 'P',     # Zeta carbon -> P
        'CZ2': 'P',    # Zeta carbon 2 -> P
        'CZ3': 'P',    # Zeta carbon 3 -> P
        
        # Protein side chain atoms - Nitrogen
        'NZ': 'P',     # Nitrogen side chain -> P
        'NE': 'P',     # Nitrogen epsilon -> P
        'NE1': 'P',    # Nitrogen epsilon 1 -> P
        'NE2': 'P',    # Nitrogen epsilon 2 -> P
        'ND1': 'P',    # Nitrogen delta 1 -> P
        'ND2': 'P',    # Nitrogen delta 2 -> P
        'NH1': 'P',    # Nitrogen eta 1 -> P
        'NH2': 'P',    # Nitrogen eta 2 -> P
        
        # Protein side chain atoms - Oxygen
        'OG': 'D',     # Oxygen gamma -> D
        'OG1': 'D',    # Oxygen gamma 1 -> D
        'OG2': 'D',    # Oxygen gamma 2 -> D
        'OE1': 'D',    # Oxygen epsilon 1 -> D
        'OE2': 'D',    # Oxygen epsilon 2 -> D
        'OD1': 'D',    # Oxygen delta 1 -> D
        'OD2': 'D',    # Oxygen delta 2 -> D
        'OH': 'D',     # Oxygen eta -> D
        
       
        # Default fallback
        '': 'm',       # Empty -> mask
    }
    
    # Return mapped code or default to 'sm' for unknown codes
    return lba_to_get_mapping.get(pos_code, 'sm')

def map_lba_block_symbols_to_get(block_symbol: str) -> str:
    """
    Map LBA block symbols to GET vocabulary codes
    
    Args:
        block_symbol: LBA block symbol (e.g., 'GLY', 'ALA', 'VAL', etc.)
        
    Returns:
        GET vocabulary code
    """
    # Mapping from LBA block symbols to GET vocabulary
    lba_to_get_mapping = {
        # Amino acids (3-letter to 1-letter mapping)
        'GLY': 'G',    # Glycine
        'ALA': 'A',    # Alanine
        'VAL': 'V',    # Valine
        'LEU': 'L',    # Leucine
        'ILE': 'I',    # Isoleucine
        'PHE': 'F',    # Phenylalanine
        'TRP': 'W',    # Tryptophan
        'TYR': 'Y',    # Tyrosine
        'ASP': 'D',    # Aspartic acid
        'HIS': 'H',    # Histidine
        'ASN': 'N',    # Asparagine
        'GLU': 'E',    # Glutamic acid
        'LYS': 'K',    # Lysine
        'GLN': 'Q',    # Glutamine
        'MET': 'M',    # Methionine
        'ARG': 'R',    # Arginine
        'SER': 'S',    # Serine
        'THR': 'T',    # Threonine
        'CYS': 'C',    # Cysteine
        'PRO': 'P',    # Proline       

    }
    
    # Return mapped symbol or default to '?' for unknown symbols
    return lba_to_get_mapping.get(block_symbol, '?')

# =============================================================================
# ENTITY TYPE MAPPINGS
# =============================================================================

def get_entity_type_mappings() -> Dict[int, int]:
    """
    Get entity type mappings
    
    Returns:
        Dict mapping entity IDs to indices
    """
    return {
        0: 0,  # protein
        1: 1,  # ligand
        2: 2,  # DNA
        3: 3,  # RNA
        4: 4,  # small molecule
        5: 5,  # other
    }

# =============================================================================
# VOCABULARY CONSTANTS
# =============================================================================

# Vocabulary sizes
BLOCK_SYMBOL_VOCAB_SIZE = 150  # 4 special + 20 AA + 8 bases + 118 atoms
POSITION_CODE_VOCAB_SIZE = 15  # 15 position codes
ENTITY_TYPE_VOCAB_SIZE = 6     # 6 entity types

# Entity type names for debugging
ENTITY_TYPE_NAMES = {
    0: "protein",
    1: "ligand", 
    2: "DNA",
    3: "RNA",
    4: "small molecule",
    5: "other"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_vocabulary_info() -> Dict[str, Any]:
    """
    Get comprehensive vocabulary information
    
    Returns:
        Dict with vocabulary statistics and mappings
    """
    return {
        "block_symbols": {
            "mapping": get_block_symbol_mappings(),
            "size": BLOCK_SYMBOL_VOCAB_SIZE,
            "description": "GET block symbol vocabulary (special tokens, amino acids, bases, atoms)"
        },
        "position_codes": {
            "mapping": get_position_code_mappings(),
            "size": POSITION_CODE_VOCAB_SIZE,
            "description": "GET position code vocabulary (protein positions, small molecules, bases)"
        },
        "entity_types": {
            "mapping": get_entity_type_mappings(),
            "size": ENTITY_TYPE_VOCAB_SIZE,
            "names": ENTITY_TYPE_NAMES,
            "description": "Entity type vocabulary (protein, ligand, DNA, RNA, small molecule, other)"
        }
    }

def print_vocabulary_summary():
    """Print a summary of all vocabulary mappings"""
    info = get_vocabulary_info()
    
    print("=== UNIVERSAL VOCABULARY SUMMARY ===")
    print(f"Block symbols: {info['block_symbols']['size']} tokens")
    print(f"Position codes: {info['position_codes']['size']} tokens") 
    print(f"Entity types: {info['entity_types']['size']} types")
    print()
    
    print("Entity type mappings:")
    for entity_id, name in ENTITY_TYPE_NAMES.items():
        print(f"  {entity_id}: {name}")
    print()
    
    print("Sample block symbol mappings:")
    block_mapping = info['block_symbols']['mapping']
    for symbol in ['#', 'G', 'c', 'DA', 'sm']:
        if symbol in block_mapping:
            print(f"  {symbol}: {block_mapping[symbol]}")
    print()
    
    print("Sample position code mappings:")
    pos_mapping = info['position_codes']['mapping']
    for code in ['sm', 'P', '']:
        if code in pos_mapping:
            print(f"  '{code}': {pos_mapping[code]}")

if __name__ == "__main__":
    print_vocabulary_summary()
