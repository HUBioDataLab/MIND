#!/usr/bin/env python3
"""
Universal Data Types

Minimal universal representation for molecular systems across all domains.
Inspired by GET's approach with entity indexing for molecular interactions.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class UniversalAtom:
    """Minimal universal atom representation - only what the model needs"""
    element: str                    # Atom type: 'C', 'N', 'O', etc.
    position: Tuple[float, float, float]  # 3D coordinates: (x, y, z)
    pos_code: str                   # Position within block: 'CA', 'CB', 'sm', etc.
    block_idx: int                  # Which block this atom belongs to
    atom_idx_in_block: int          # Index within the block
    entity_idx: int                 # Which molecular entity (0=protein, 1=ligand, etc.)

@dataclass
class UniversalBlock:
    """Hierarchical block containing atoms"""
    symbol: str                     # Block token: 'ALA', 'BENZENE', 'PS_300_fragment'
    atoms: List[UniversalAtom]      # Atoms in this block
    
    def __len__(self):
        return len(self.atoms)

@dataclass
class UniversalMolecule:
    """Universal representation of any molecular system - minimal cross-domain info"""
    id: str                         # Unique identifier
    dataset_type: str               # 'qm9', 'pdb', 'lba', 'cath', etc.
    blocks: List[UniversalBlock]    # Hierarchical blocks
    properties: Dict[str, Any] = field(default_factory=dict)  # Optional: for evaluation only

    def get_all_atoms(self) -> List[UniversalAtom]:
        """Get all atoms from all blocks"""
        atoms = []
        for block in self.blocks:
            atoms.extend(block.atoms)
        return atoms
    
    def get_atoms_by_entity(self, entity_idx: int) -> List[UniversalAtom]:
        """Get all atoms belonging to a specific entity"""
        atoms = []
        for block in self.blocks:
            for atom in block.atoms:
                if atom.entity_idx == entity_idx:
                    atoms.append(atom)
        return atoms
    
    def get_blocks_by_entity(self, entity_idx: int) -> List[UniversalBlock]:
        """Get all blocks belonging to a specific entity"""
        entity_blocks = []
        for block in self.blocks:
            if block.atoms and block.atoms[0].entity_idx == entity_idx:
                entity_blocks.append(block)
        return entity_blocks
