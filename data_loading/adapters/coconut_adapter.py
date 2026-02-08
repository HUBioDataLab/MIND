#!/usr/bin/env python3
"""
COCONUT Adapter

Professional adapter for COCONUT (Collection of Open Natural Products) dataset 
using GET's PS_300 fragment tokenization system for sophisticated molecular representation.

Features:
- SDF file parsing for natural product structures
- GET's PS_300 vocabulary for molecular fragmentation
- Universal format conversion with single entity indexing
- Hydrogen filtering for consistency with protein representations
- Chunked processing support for large-scale datasets
- Robust error handling with statistics tracking
"""

import os
import sys
import logging
from typing import List, Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Add GET's project directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'GET'))

from adapters.base_adapter import BaseAdapter
from data_types import UniversalAtom, UniversalBlock, UniversalMolecule

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStatistics:
    """Track processing statistics for debugging and monitoring"""
    total_molecules: int = 0
    successful_molecules: int = 0
    failed_molecules: int = 0
    skipped_no_conformer: int = 0
    skipped_tokenization_error: int = 0
    skipped_empty_after_h_filter: int = 0
    skipped_invalid_load: int = 0
    
    def summary(self) -> str:
        """Generate a summary string"""
        if self.total_molecules == 0:
            return "No molecules processed yet"
        
        success_rate = self.successful_molecules / self.total_molecules * 100
        return (
            f"Processing Statistics:\n"
            f"  Total: {self.total_molecules:,}\n"
            f"  Successful: {self.successful_molecules:,} ({success_rate:.1f}%)\n"
            f"  Failed: {self.failed_molecules:,}\n"
            f"  - No conformer: {self.skipped_no_conformer:,}\n"
            f"  - Tokenization error: {self.skipped_tokenization_error:,}\n"
            f"  - Empty after H filter: {self.skipped_empty_after_h_filter:,}\n"
            f"  - Invalid load: {self.skipped_invalid_load:,}"
        )


class COCONUTAdapter(BaseAdapter):
    """
    Professional COCONUT adapter using GET's fragment tokenization.
    
    Key features:
    - Hydrogen filtering for consistency with protein representations
    - Chunked processing support for large datasets (696K+ molecules)
    - No premature sanitization (let tokenizer handle it)
    - Statistics tracking for monitoring processing quality
    """

    def __init__(self, remove_hydrogens: bool = True):
        """
        Initialize COCONUT adapter.
        
        Args:
            remove_hydrogens: If True, filter out hydrogen atoms for consistency
                            with protein representations (default: True)
        """
        super().__init__("coconut")
        self.remove_hydrogens = remove_hydrogens
        self.stats = ProcessingStatistics()
        self._initialize_get_components()

    def _initialize_get_components(self):
        """Initialize GET's tokenization components"""
        try:
            from data.pdb_utils import VOCAB
            from data.tokenizer.tokenize_3d import TOKENIZER, tokenize_3d
            from rdkit import RDLogger
            
            # Suppress RDKit warnings for cleaner output
            RDLogger.DisableLog('rdApp.*')
            
            self.get_vocab = VOCAB
            self.tokenize_3d_func = tokenize_3d
            self.tokenize_3d = TOKENIZER
            
            # Initialize the tokenizer
            self.tokenize_3d.load('PS_300')
            logger.info("Successfully initialized GET tokenization components")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GET components: {e}")

    def load_raw_data(
        self, 
        data_path: str, 
        max_samples: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """
        Load COCONUT data from SDF file with chunking support.
        
        Args:
            data_path: Path to data directory containing COCONUT folder
            max_samples: Maximum number of samples to load (None for all)
            **kwargs: Additional arguments:
                - num_chunks: Total number of chunks to split data into
                - chunk_index: Index of chunk to load (0-based)
                - sdf_filename: Custom SDF filename (default: coconut_sdf_3d-10-2025.sdf)
        
        Returns:
            List of sample dictionaries with 'mol', 'id', and 'properties' keys
        """
        from rdkit import Chem
        
        # Extract chunking parameters
        num_chunks = kwargs.get('num_chunks', None)
        chunk_index = kwargs.get('chunk_index', None)
        sdf_filename = kwargs.get('sdf_filename', 'coconut.sdf')
        
        # Construct SDF path
        sdf_path = os.path.join(data_path, "COCONUT", sdf_filename)
        
        if not os.path.exists(sdf_path):
            raise FileNotFoundError(f"COCONUT SDF file not found: {sdf_path}")

        logger.info(f"Loading molecules from: {sdf_path}")
        
        # Use RDKit's SDMolSupplier to parse SDF file
        # NOTE: removeHs=False and sanitize=False - let tokenizer handle sanitization
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        
        # Get total count for chunking (this is fast for SDF files)
        # We estimate based on known COCONUT size or iterate once
        total_molecules = len(suppl) if hasattr(suppl, '__len__') else 696466
        
        # Calculate chunk boundaries
        if num_chunks is not None and chunk_index is not None:
            chunk_size = (total_molecules + num_chunks - 1) // num_chunks
            start_idx = chunk_index * chunk_size
            end_idx = min(start_idx + chunk_size, total_molecules)
            
            if max_samples:
                end_idx = min(end_idx, start_idx + max_samples)
            
            logger.info(
                f"📦 Chunking: Loading samples {start_idx:,} to {end_idx:,} "
                f"(chunk {chunk_index + 1}/{num_chunks})"
            )
        else:
            start_idx = 0
            end_idx = max_samples if max_samples else total_molecules
            logger.info(f"Loading up to {end_idx:,} molecules...")
        
        samples = []
        skipped_count = 0
        
        for i, mol in enumerate(suppl):
            # Skip molecules before start index
            if i < start_idx:
                continue
            
            # Stop at end index
            if i >= end_idx:
                break
            
            if mol is None:
                skipped_count += 1
                self.stats.skipped_invalid_load += 1
                continue
            
            # Create a sample dictionary with molecule and properties
            sample = {
                'mol': mol,
                'id': f"coconut_{i}",
                'properties': {}
            }
            
            # Extract properties from SDF data
            if mol.HasProp('IDENTIFIER'):
                sample['id'] = mol.GetProp('IDENTIFIER')
                sample['properties']['identifier'] = mol.GetProp('IDENTIFIER')
            
            # Add any other properties from SDF
            for prop_name in mol.GetPropNames():
                if prop_name != 'IDENTIFIER':
                    sample['properties'][prop_name.lower()] = mol.GetProp(prop_name)
            
            samples.append(sample)
            
            # Progress indicator for large dataset
            if (i - start_idx + 1) % 10000 == 0:
                logger.info(f"Loaded {i - start_idx + 1:,} molecules...")
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count:,} invalid molecules during loading")

        logger.info(f"Successfully loaded {len(samples):,} COCONUT samples")
        
        return samples

    def create_blocks(self, raw_item: Dict[str, Any]) -> List[UniversalBlock]:
        """
        Convert COCONUT raw data to universal blocks using GET tokenization.
        
        Args:
            raw_item: Dictionary with 'mol' (RDKit Mol), 'id', and 'properties'
        
        Returns:
            List of UniversalBlock objects, empty list if processing fails
        """
        self.stats.total_molecules += 1
        
        mol = raw_item['mol']
        mol_id = raw_item.get('id', 'unknown')
        
        # Extract molecule information from RDKit Mol object
        # NOTE: No sanitization here - let the tokenizer handle it
        try:
            atoms, positions, bonds, atom_idx_map = self._extract_molecule_info_from_rdkit(mol)
        except ValueError as e:
            # No conformer or other extraction error
            logger.debug(f"Skipping molecule {mol_id}: {e}")
            self.stats.skipped_no_conformer += 1
            self.stats.failed_molecules += 1
            return []
        
        # Check if we have any atoms after hydrogen filtering
        if len(atoms) == 0:
            logger.debug(f"Skipping molecule {mol_id}: No atoms after hydrogen filtering")
            self.stats.skipped_empty_after_h_filter += 1
            self.stats.failed_molecules += 1
            return []
        
        # Use GET's tokenization with coordinates and bonds
        try:
            fragments, atom_indices = self.tokenize_3d_func(
                atoms=atoms,
                coords=positions,
                smiles=None,  # Don't use SMILES to avoid misalignment
                bonds=bonds
            )
        except Exception as e:
            # Skip molecules with invalid valences or other RDKit issues
            # This is common in natural products with complex ring systems
            logger.debug(f"Tokenization failed for {mol_id}: {e}")
            self.stats.skipped_tokenization_error += 1
            self.stats.failed_molecules += 1
            return []

        # Create blocks from fragments
        blocks = []
        for frag_idx, (fragment, atom_group) in enumerate(zip(fragments, atom_indices)):
            fragment_atoms = []
            
            for atom_idx in atom_group:
                # Bounds check
                if atom_idx < 0 or atom_idx >= len(atoms):
                    logger.warning(
                        f"Invalid atom index {atom_idx} in molecule {mol_id} "
                        f"(valid range: 0-{len(atoms)-1})"
                    )
                    continue
                
                atom = UniversalAtom(
                    element=atoms[atom_idx],
                    position=positions[atom_idx],
                    pos_code='sm',  # Small molecule position code (vocabulary index 13)
                    block_idx=0,  # Will be updated later
                    atom_idx_in_block=len(fragment_atoms),
                    entity_idx=0  # Single entity for COCONUT natural products
                )
                fragment_atoms.append(atom)

            # Create block with fragment as symbol (only if it has atoms)
            if fragment_atoms:
                block = UniversalBlock(
                    symbol=fragment,  # Use GET fragment as symbol (e.g., PS_300_x)
                    atoms=fragment_atoms
                )
                blocks.append(block)

        # Fix block indices to match actual positions
        for block_idx, block in enumerate(blocks):
            for atom in block.atoms:
                atom.block_idx = block_idx

        if blocks:
            self.stats.successful_molecules += 1
        else:
            self.stats.failed_molecules += 1
            
        return blocks

    def _extract_molecule_info_from_rdkit(
        self, 
        mol
    ) -> Tuple[List[str], List[Tuple[float, float, float]], List[Tuple[int, int, int]], Dict[int, int]]:
        """
        Extract atoms, positions, and bonds from RDKit Mol object.
        
        NOTE: Does NOT sanitize the molecule - let the tokenizer handle sanitization.
        The tokenize_3d function rebuilds the molecule and sanitizes it properly.
        
        Args:
            mol: RDKit Mol object
        
        Returns:
            Tuple of (atom_types, positions, bonds, atom_idx_map)
            - atom_types: List of element symbols (H filtered if remove_hydrogens=True)
            - positions: List of (x, y, z) coordinate tuples
            - bonds: List of (src_idx, dst_idx, bond_type) tuples with remapped indices
            - atom_idx_map: Dictionary mapping original atom indices to filtered indices
        
        Raises:
            ValueError: If molecule has no conformers (no 3D coordinates)
        """
        from rdkit import Chem
        
        # Check for 3D coordinates
        if mol.GetNumConformers() == 0:
            raise ValueError("Molecule has no conformers - 3D coordinates not available")
        
        conformer = mol.GetConformer()
        
        # Extract atom types and positions, optionally filtering hydrogens
        atom_types = []
        positions = []
        atom_idx_map = {}  # Maps original index -> filtered index
        
        filtered_idx = 0
        for orig_idx, atom in enumerate(mol.GetAtoms()):
            element = atom.GetSymbol()
            
            # Filter hydrogen atoms if requested (for consistency with protein adapter)
            if self.remove_hydrogens and element == 'H':
                continue
            
            atom_types.append(element)
            
            pos = conformer.GetAtomPosition(orig_idx)
            positions.append((float(pos.x), float(pos.y), float(pos.z)))
            
            atom_idx_map[orig_idx] = filtered_idx
            filtered_idx += 1
        
        # Extract bonds with remapped indices
        bonds = []
        bond_type_map = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 4
        }
        
        for bond in mol.GetBonds():
            orig_atom1_idx = bond.GetBeginAtomIdx()
            orig_atom2_idx = bond.GetEndAtomIdx()
            
            # Skip bonds involving filtered atoms (hydrogens)
            if orig_atom1_idx not in atom_idx_map or orig_atom2_idx not in atom_idx_map:
                continue
            
            # Remap indices to filtered atom list
            atom1_idx = atom_idx_map[orig_atom1_idx]
            atom2_idx = atom_idx_map[orig_atom2_idx]
            
            bond_type = bond_type_map.get(bond.GetBondType(), 1)  # Default to single
            bonds.append((atom1_idx, atom2_idx, bond_type))
        
        return atom_types, positions, bonds, atom_idx_map

    def convert_to_universal(self, raw_item: Dict[str, Any]) -> UniversalMolecule:
        """
        Convert COCONUT raw data to universal format with blocks.
        
        Args:
            raw_item: Dictionary with 'mol', 'id', and 'properties'
        
        Returns:
            UniversalMolecule object
        """
        blocks = self.create_blocks(raw_item)
        
        # Use the ID from the sample
        mol_id = raw_item.get('id', 'unknown')
        
        # Extract properties
        properties = raw_item.get('properties', {})
        
        return UniversalMolecule(
            id=mol_id,
            dataset_type=self.dataset_type,
            blocks=blocks,
            properties=properties
        )

    def get_statistics(self) -> ProcessingStatistics:
        """Get current processing statistics"""
        return self.stats
    
    def print_statistics(self):
        """Print processing statistics summary"""
        print(self.stats.summary())
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = ProcessingStatistics()
