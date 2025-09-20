#!/usr/bin/env python3
"""
COCONUT Adapter

Professional adapter for COCONUT (Collection of Open Natural Products) dataset 
using GET's PS_300 fragment tokenization system for sophisticated molecular representation.

Features:
- SDF file parsing for natural product structures
- GET's PS_300 vocabulary for molecular fragmentation
- Universal format conversion with single entity indexing
- Robust error handling for large-scale natural product processing
"""

import os
import sys
import pickle
from typing import List, Any, Dict, Iterator
import numpy as np

# Add GET's project directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'GET'))

from adapters.base_adapter import BaseAdapter
from data_types import UniversalAtom, UniversalBlock, UniversalMolecule

class COCONUTAdapter(BaseAdapter):
    """Professional COCONUT adapter using GET's fragment tokenization"""

    def __init__(self):
        super().__init__("coconut")
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
            print("Successfully initialized GET tokenization components")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GET components: {e}")

    def load_raw_data(self, data_path: str, max_samples: int = None) -> List[Any]:
        """Load COCONUT data from SDF file"""
        
        # Check for cached data
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"coconut_samples_{max_samples or 'all'}.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached COCONUT data from: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"Loaded {len(cached_data)} cached samples")
                return cached_data
            except Exception as e:
                print(f"Error loading cache: {e}. Re-loading raw data.")

        # Load from SDF file
        print("Loading COCONUT data from SDF file...")
        try:
            from rdkit import Chem
            
            sdf_path = os.path.join(data_path, "COCONUT", "coconut_sdf_3d-09-2025.sdf")
            
            if not os.path.exists(sdf_path):
                raise FileNotFoundError(f"COCONUT SDF file not found: {sdf_path}")

            print(f"Loading molecules from: {sdf_path}")
            
            # Use RDKit's SDMolSupplier to parse SDF file
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
            
            samples = []
            total_molecules = 695112  # From our previous count
            skipped_count = 0
            
            print(f"Processing COCONUT molecules (total: ~{total_molecules})...")
            
            for i, mol in enumerate(suppl):
                if mol is None:
                    skipped_count += 1
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
                
                # Limit samples if requested
                if max_samples and len(samples) >= max_samples:
                    break
                
                # Progress indicator for large dataset
                if (i + 1) % 10000 == 0:
                    print(f"Processed {i + 1} molecules (skipped {skipped_count})...")
            
            if skipped_count > 0:
                print(f"Note: Skipped {skipped_count} invalid molecules during loading")

            print(f"Successfully loaded {len(samples)} COCONUT samples")
            
            # Cache the loaded samples
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)
            print(f"Cached {len(samples)} samples to {cache_file}")
            
            return samples

        except Exception as e:
            raise RuntimeError(f"Failed to load COCONUT data: {e}")

    def create_blocks(self, raw_item: Dict[str, Any]) -> List[UniversalBlock]:
        """Convert COCONUT raw data to universal blocks using GET tokenization"""
        
        mol = raw_item['mol']
        
        # Extract molecule information from RDKit Mol object
        atoms, positions, bonds = self._extract_molecule_info_from_rdkit(mol)
        
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
            mol_id = raw_item.get('id', 'unknown')
            print(f"\n🔍 PROBLEMATIC MOLECULE DETAILS:")
            print(f"   ID: {mol_id}")
            print(f"   Error: {e}")
            print(f"   Atoms: {len(atoms)} atoms")
            print(f"   Elements: {atoms}")
            print(f"   Bonds: {len(bonds)} bonds")
            if len(bonds) <= 20:  # Only show bonds for smaller molecules
                print(f"   Bond details: {bonds}")
            else:
                print(f"   Bond details: First 10 bonds: {bonds[:10]}...")
            
            # Try to get SMILES if available
            try:
                from rdkit import Chem
                mol_copy = raw_item['mol']
                smiles = Chem.MolToSmiles(mol_copy)
                print(f"   SMILES: {smiles}")
            except:
                print(f"   SMILES: Could not generate")
            
            print(f"   Reason: {'Aromaticity/valence issue' if 'kekulize' in str(e).lower() or 'valence' in str(e).lower() else 'Other tokenization error'}")
            print("   " + "="*50)
            return []

        # Create blocks from fragments
        blocks = []
        for frag_idx, (fragment, atom_group) in enumerate(zip(fragments, atom_indices)):
            fragment_atoms = []
            
            for atom_idx in atom_group:
                if atom_idx < len(atoms):
                    atom = UniversalAtom(
                        element=atoms[atom_idx],
                        position=positions[atom_idx],
                        pos_code='np',  # Natural product position code
                        block_idx=0,  # Will be updated later
                        atom_idx_in_block=len(fragment_atoms),
                        entity_idx=0  # Single entity for COCONUT natural products
                    )
                    fragment_atoms.append(atom)

            # Create block with fragment as symbol
            if fragment_atoms:  # Only create block if it has atoms
                block = UniversalBlock(
                    symbol=fragment,  # Use GET fragment as symbol
                    atoms=fragment_atoms
                )
                blocks.append(block)

        # Fix block indices to match actual positions
        for block_idx, block in enumerate(blocks):
            for atom in block.atoms:
                atom.block_idx = block_idx

        return blocks

    def _extract_molecule_info_from_rdkit(self, mol) -> tuple:
        """Extract atoms, positions, and bonds from RDKit Mol object"""
        from rdkit import Chem
        
        # Sanitize the molecule to ensure proper valences and aromaticity
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            # If sanitization fails, try to work with the molecule as-is
            print(f"⚠️ Molecule sanitization failed: {e}")
        
        # Extract atom types
        atom_types = []
        for atom in mol.GetAtoms():
            atom_types.append(atom.GetSymbol())
        
        # Skip molecules that are too large for our conservative limits
        num_atoms = mol.GetNumAtoms()
        # if num_atoms > 32:
        #     print(f"⚠️ Skipping molecule {raw_item.get('id', 'unknown')} - too many atoms ({num_atoms} > 32)")
        #     return []
        
        # Extract positions from conformer
        if mol.GetNumConformers() == 0:
            raise ValueError("Molecule has no conformers - 3D coordinates not available")
        
        conformer = mol.GetConformer()
        positions = []
        for i in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            positions.append((float(pos.x), float(pos.y), float(pos.z)))
        
        # Extract bonds
        bonds = []
        for bond in mol.GetBonds():
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            
            # Map RDKit bond types to integer codes
            bond_type_map = {
                Chem.BondType.SINGLE: 1,
                Chem.BondType.DOUBLE: 2,
                Chem.BondType.TRIPLE: 3,
                Chem.BondType.AROMATIC: 4
            }
            
            bond_type = bond_type_map.get(bond.GetBondType(), 1)  # Default to single
            bonds.append((atom1_idx, atom2_idx, bond_type))
        
        return atom_types, positions, bonds

    def convert_to_universal(self, raw_item: Dict[str, Any]) -> UniversalMolecule:
        """Convert COCONUT raw data to universal format with blocks"""
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

