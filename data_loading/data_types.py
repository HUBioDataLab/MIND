#!/usr/rbin/env python3
"""
Universal Molecular Data Types for the MIND Project.

This module defines the core data structures for the "Universal Molecular Representation".
These dataclasses create a standardized, hierarchical format (Molecule -> Block -> Atom)
for any molecular system, regardless of its origin (protein, ligand, DNA, etc.).
This abstraction is fundamental to the project's goal of building a generalist,
multi-modal foundation model.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

# Suggestion: For enhanced data integrity, you can make these dataclasses immutable
# by adding '(frozen=True)'. This prevents accidental modification after creation.
# e.g., @dataclass(frozen=True)
@dataclass
class UniversalAtom:
    """
    Represents a single atom, the most fundamental unit in the hierarchy.
    It holds essential chemical, geometric, and contextual information.
    """
    element: str                    # Chemical element symbol (e.g., 'C', 'N', 'O').
    position: Tuple[float, float, float]  # Tuple of (x, y, z) coordinates in 3D space.
    pos_code: str                   # A semantic code for the atom's role within its block (e.g., 'CA'
                                    # for alpha-carbon, 'CB' for beta-carbon, or 'sm' for a generic
                                    # small molecule atom). Provides chemical context beyond a simple index.
    block_idx: int                  # The global index of the `UniversalBlock` this atom belongs to
                                    # within the parent `UniversalMolecule`'s list of blocks.
    atom_idx_in_block: int          # The local, zero-based index of this atom within its parent
                                    # `UniversalBlock`'s list of atoms. Establishes a canonical
                                    # ordering of atoms inside the block.
    entity_idx: int                 # Index identifying the molecular entity this atom belongs to
                                    # (e.g., 0 for protein, 1 for ligand). Crucial for modeling
                                    # interactions in a multi-component complex.

@dataclass
class UniversalBlock:
    """
    Represents a chemically meaningful, hierarchical subunit of a molecule.
    For a protein, this is typically an amino acid (e.g., Alanine).
    For a small molecule, it could be a functional group or a fragment.
    This serves as the "token" for the higher-level representation of the model.
    """
    symbol: str                     # The symbol representing the block's identity (e.g., 'ALA', 'BENZENE',
                                    # or a fragment ID like 'PS_300_x').
    atoms: List[UniversalAtom]      # An ordered list of `UniversalAtom` objects that constitute this block.
    
    def __len__(self):
        return len(self.atoms)

@dataclass
class UniversalMolecule:
    """
    The top-level container for an entire molecular system.
    This can represent a single molecule or a multi-component complex like protein-ligand.
    """
    id: str                         # A unique identifier for the system, such as a PDB ID or a sample name.
    dataset_type: str               # The source dataset of the molecule (e.g., 'pdb', 'qm9', 'lba'),
                                    # used for metadata and tracking.
    blocks: List[UniversalBlock]    # An ordered list of `UniversalBlock` objects that make up the molecular system.
    properties: Dict[str, Any] = field(default_factory=dict)  # A dictionary to store target values (e.g., binding affinity)
                                                              # or other metadata for downstream tasks and evaluation.

    def get_all_atoms(self) -> List[UniversalAtom]:
        """Flattens the hierarchy and returns a single list of all atoms in the molecule."""
        atoms = []
        for block in self.blocks:
            atoms.extend(block.atoms)
        return atoms
    
    def get_atoms_by_entity(self, entity_idx: int) -> List[UniversalAtom]:
        """Returns all atoms belonging to a specific entity index."""
        atoms = []
        for block in self.blocks:
            for atom in block.atoms:
                if atom.entity_idx == entity_idx:
                    atoms.append(atom)
        return atoms
    
    def get_blocks_by_entity(self, entity_idx: int) -> List[UniversalBlock]:
        """Returns all blocks where the atoms belong to a specific entity index."""
        entity_blocks = []
        for block in self.blocks:
            # A block is considered part of an entity if its atoms belong to that entity.
            # Checking the first atom is sufficient as all atoms in a block share the same entity.
            if block.atoms and block.atoms[0].entity_idx == entity_idx:
                entity_blocks.append(block)
        return entity_blocks