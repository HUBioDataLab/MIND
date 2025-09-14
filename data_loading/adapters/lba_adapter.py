#!/usr/bin/env python3
"""
LBA Adapter

Professional adapter for LBA (Ligand Binding Affinity) dataset using GET's PS_300
fragment tokenization system for sophisticated molecular representation.

Features:
- Real LMDB data loading using ATOM3D
- GET's PS_300 vocabulary for molecular fragmentation
- Universal format conversion with entity indexing
- No fallbacks or simplifications - robust error handling
"""

import os
import sys
import pickle
from typing import List, Any, Dict
import pandas as pd

# Add GET's project directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'GET'))

from adapters.base_adapter import BaseAdapter
from data_types import UniversalAtom, UniversalBlock, UniversalMolecule

class LBAAdapter(BaseAdapter):
    """Professional LBA adapter using GET's fragment tokenization"""

    def __init__(self):
        super().__init__("lba")
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
        """Load LBA data using ATOM3D's LMDBDataset"""
        
        # Check for cached data
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"lba_samples_{max_samples or 'all'}.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached LBA data from: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"Loaded {len(cached_data)} cached samples")
                return cached_data
            except Exception as e:
                print(f"Error loading cache: {e}. Re-loading raw data.")

        # Load using ATOM3D
        print("Loading LBA data using ATOM3D...")
        try:
            from atom3d.datasets import LMDBDataset

            lba_data_path = os.path.join(data_path, "split-by-sequence-identity-30", "data", "train")

            if not os.path.exists(lba_data_path):
                raise FileNotFoundError(f"LBA LMDB path not found: {lba_data_path}")

            print(f"Loading LMDB dataset from: {lba_data_path}")

            dataset = LMDBDataset(lba_data_path)
            print(f"Found {len(dataset)} LBA samples")

            if max_samples:
                sample_count = min(max_samples, len(dataset))
                print(f"Loading {sample_count} samples...")
                samples = [dataset[i] for i in range(sample_count)]
            else:
                print(f"Loading all {len(dataset)} samples...")
                samples = [dataset[i] for i in range(len(dataset))]

            print(f"Successfully loaded {len(samples)} LBA samples")
            
            # Cache the loaded samples
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)
            print(f"Cached {len(samples)} samples to {cache_file}")
            
            return samples

        except Exception as e:
            raise RuntimeError(f"Failed to load LBA data: {e}")

    def create_blocks(self, raw_item: Dict[str, Any]) -> List[UniversalBlock]:
        """Convert LBA raw data to universal blocks using GET tokenization"""
        blocks = []

        # Process receptor (pocket) - use residue-based blocks
        receptor_blocks = self._df_to_blocks(raw_item['atoms_pocket'], key_atom_name='name')
        
        # Process ligand - use GET fragment tokenization
        bonds_df = raw_item.get('bonds', None)
        if bonds_df is None:
            raise ValueError("LBA data must contain bonds information for GET tokenization")
        
        ligand_blocks = self._create_ligand_blocks_with_get_tokenization(
            raw_item['atoms_ligand'], 
            bonds_df
        )

        # Update entity indices
        for block in receptor_blocks:
            for atom in block.atoms:
                atom.entity_idx = 0  # Receptor entity

        for block in ligand_blocks:
            for atom in block.atoms:
                atom.entity_idx = 1  # Ligand entity

        # Combine blocks
        blocks.extend(receptor_blocks)
        blocks.extend(ligand_blocks)

        # Fix block indices to match actual positions
        for block_idx, block in enumerate(blocks):
            for atom in block.atoms:
                atom.block_idx = block_idx

        return blocks

    def _create_ligand_blocks_with_get_tokenization(self, ligand_df: pd.DataFrame, bonds_df: pd.DataFrame) -> List[UniversalBlock]:
        """Create ligand blocks using GET's PS_300 tokenization"""
        
        # Extract ligand information
        atoms = []
        positions = []
        
        for row in ligand_df.itertuples():
            atoms.append(getattr(row, 'element'))
            positions.append((getattr(row, 'x'), getattr(row, 'y'), getattr(row, 'z')))

        # Convert bonds to GET format
        bonds = []
        for row in bonds_df.itertuples():
            bond_type = int(getattr(row, 'type'))
            if bond_type == 1.5:
                bond_type = 4  # aromatic
            bonds.append((getattr(row, 'atom1'), getattr(row, 'atom2'), bond_type))
        
        # Use GET's tokenization with bonds
        fragments, atom_indices = self.tokenize_3d_func(
            atoms=atoms,
            coords=positions,
            bonds=bonds
        )

        # Create blocks from fragments
        blocks = []
        for frag_idx, (fragment, atom_group) in enumerate(zip(fragments, atom_indices)):
            fragment_atoms = []
            
            for atom_idx in atom_group:
                if atom_idx < len(ligand_df):
                    row = ligand_df.iloc[atom_idx]
                    atom = UniversalAtom(
                        element=row['element'],
                        position=(row['x'], row['y'], row['z']),
                        pos_code=row['name'],
                        block_idx=0,  # Will be updated later
                        atom_idx_in_block=len(fragment_atoms),
                        entity_idx=1  # Ligand entity
                    )
                    fragment_atoms.append(atom)

            # Create block with fragment as symbol
            block = UniversalBlock(
                symbol=fragment,  # Use GET fragment as symbol
                atoms=fragment_atoms
            )
            blocks.append(block)

        return blocks

    def _df_to_blocks(self, df: pd.DataFrame, key_atom_name: str = 'name') -> List[UniversalBlock]:
        """Convert DataFrame to blocks following GET's df_to_blocks function"""
        blocks = []
        units = []
        last_res_id = None
        last_res_symbol = None

        for row in df.itertuples():
            # Group by residue
            residue = getattr(row, 'residue')
            insert_code = getattr(row, 'insertion_code', ' ')
            res_id = f'{residue}{insert_code}'.rstrip()

            if res_id != last_res_id and last_res_id is not None:
                # One block ended, create it
                block = UniversalBlock(
                    symbol=last_res_symbol,
                    atoms=units
                )
                blocks.append(block)
                units = []

            last_res_id = res_id
            last_res_symbol = getattr(row, 'resname')

            # Skip hydrogen atoms
            element = getattr(row, 'element')
            if element == 'H':
                continue

            # Create atom
            atom = UniversalAtom(
                element=element,
                position=(getattr(row, 'x'), getattr(row, 'y'), getattr(row, 'z')),
                pos_code=getattr(row, key_atom_name),
                block_idx=len(blocks),
                atom_idx_in_block=len(units),
                entity_idx=0  # Will be updated later for receptor
            )
            units.append(atom)

        # Add the last block
        if units:
            block = UniversalBlock(
                symbol=last_res_symbol,
                atoms=units
            )
            blocks.append(block)

        return blocks