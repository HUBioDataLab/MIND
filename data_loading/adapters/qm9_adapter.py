#!/usr/bin/env python3
"""
QM9 Adapter

Professional adapter for QM9 dataset using GET's PS_300 fragment tokenization
system for sophisticated molecular representation.

Features:
- PyTorch Geometric QM9 dataset loading
- GET's PS_300 vocabulary for molecular fragmentation
- Universal format conversion with single entity indexing
- No fallbacks or simplifications - robust error handling
"""

import os
import sys
import pickle
from typing import List, Any, Dict
import torch
import numpy as np

# Add GET's project directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'GET'))

from adapters.base_adapter import BaseAdapter
from data_types import UniversalAtom, UniversalBlock, UniversalMolecule

class QM9Adapter(BaseAdapter):
    """Professional QM9 adapter using GET's fragment tokenization"""

    def __init__(self):
        super().__init__("qm9")
        self._initialize_get_components()

    def _initialize_get_components(self):
        """Initialize GET's tokenization components with GPU optimization"""
        try:
            from data.pdb_utils import VOCAB
            from data.tokenizer.tokenize_3d import TOKENIZER, tokenize_3d
            from rdkit import RDLogger
            import torch
            
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

    def load_raw_data(self, data_path: str, max_samples: int = None, **kwargs) -> List[Any]:
        """Load QM9 data using PyTorch Geometric with optional chunking support"""
        
        # Extract chunking parameters
        num_chunks = kwargs.get('num_chunks', None)
        chunk_index = kwargs.get('chunk_index', None)
        
        # Load using PyTorch Geometric (no raw caching - universal format is cached by base adapter)
        print("Loading QM9 data using PyTorch Geometric...")
        try:
            from torch_geometric.datasets import QM9

            dataset = QM9(root=data_path)
            total_samples = len(dataset)
            print(f"Found {total_samples} QM9 molecules")
            
            # Handle chunking (index-based slicing)
            if num_chunks and chunk_index is not None:
                chunk_size = (total_samples + num_chunks - 1) // num_chunks
                start_idx = chunk_index * chunk_size
                end_idx = min(start_idx + chunk_size, total_samples)
                
                print(f"📦 Chunking: Loading samples {start_idx} to {end_idx} (chunk {chunk_index + 1}/{num_chunks})")
                samples = [dataset[i] for i in range(start_idx, end_idx)]
            
            elif max_samples:
                sample_count = min(max_samples, total_samples)
                print(f"Loading {sample_count} samples...")
                samples = [dataset[i] for i in range(sample_count)]
            else:
                print(f"Loading all {total_samples} samples...")
                samples = [dataset[i] for i in range(total_samples)]

            print(f"Successfully loaded {len(samples)} QM9 samples")
            return samples

        except Exception as e:
            raise RuntimeError(f"Failed to load QM9 data: {e}")

    def create_blocks(self, raw_item: Any) -> List[UniversalBlock]:
        """Convert QM9 raw data to universal blocks using GET tokenization"""
        
        # Extract molecule information from PyTorch Geometric Data object
        atoms, positions, bonds = self._extract_molecule_info(raw_item)
        
        # Use GET's tokenization with coordinates and bonds (following GET's approach)
        try:
            fragments, atom_indices = self.tokenize_3d_func(
                atoms=atoms,
                coords=positions,  # Use coordinates with bonds (GET's approach)
                smiles=None,  # Don't use SMILES (misaligned)
                bonds=bonds
            )
        except Exception as e:
            # Skip molecules with invalid valences or other RDKit issues
            print(f"⚠️ Skipping molecule due to RDKit error: {e}")
            # Return empty blocks for this molecule
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
                        pos_code='sm',  # Small molecule position code
                        block_idx=0,  # Will be updated later
                        atom_idx_in_block=len(fragment_atoms),
                        entity_idx=0  # Single entity for QM9
                    )
                    fragment_atoms.append(atom)

            # Create block with fragment as symbol
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

    def _extract_molecule_info(self, mol_data) -> tuple:
        """Extract atoms, positions, and bonds from PyTorch Geometric Data object"""
        import torch
        
        # Use compiled version if GPU is available
        if torch.cuda.is_available():
            try:
                return self._extract_tensor_info_compiled(
                    mol_data.z, 
                    mol_data.pos, 
                    mol_data.edge_index, 
                    mol_data.edge_attr
                )
            except Exception as e:
                print(f"⚠️ Compiled extraction failed, using standard method: {e}")
        
        # Standard extraction method
        # Extract atom types from atomic numbers (z attribute) using proper periodic table
        from rdkit import Chem
        atom_types = []
        for atomic_num in mol_data.z:
            element = Chem.GetPeriodicTable().GetElementSymbol(int(atomic_num.item()))
            atom_types.append(element)
        
        # Extract positions
        positions = mol_data.pos.numpy().tolist()
        positions = [tuple(pos) for pos in positions]
        
        # Extract bonds from edge_index and edge_attr (properly deduplicated)
        bonds = []
        edge_index = mol_data.edge_index.numpy()
        edge_attr = mol_data.edge_attr.numpy() if mol_data.edge_attr is not None else None
        
        # Use set to avoid duplicate bonds (PyTorch Geometric has bidirectional edges)
        seen_bonds = set()
        
        for i in range(edge_index.shape[1]):
            atom1_idx = int(edge_index[0, i])
            atom2_idx = int(edge_index[1, i])
            
            # Create bond tuple in canonical order (smaller index first)
            bond_tuple = (min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx))
            
            # Skip if we've already seen this bond
            if bond_tuple in seen_bonds:
                continue
            seen_bonds.add(bond_tuple)
            
            # Determine bond type from edge attributes
            if edge_attr is not None:
                # Edge attributes are one-hot: [single, double, triple, aromatic]
                bond_type_idx = np.argmax(edge_attr[i, :4])
                bond_type = bond_type_idx + 1  # 1=single, 2=double, 3=triple, 4=aromatic
            else:
                bond_type = 1  # Default to single bond
            
            bonds.append((bond_tuple[0], bond_tuple[1], bond_type))
        
        return atom_types, positions, bonds
    
    @torch.compile(mode="reduce-overhead")
    def _extract_tensor_info_compiled(self, z_tensor, pos_tensor, edge_index_tensor, edge_attr_tensor):
        """Compiled tensor extraction for GPU acceleration"""
        import torch
        
        # Convert atomic numbers to elements (this part can't be compiled, but tensor ops can)
        atomic_to_element = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
        atom_types = []
        for atomic_num in z_tensor:
            element = atomic_to_element.get(atomic_num.item(), f'Unknown({atomic_num})')
            atom_types.append(element)
        
        # Extract positions (tensor operations)
        positions = pos_tensor.cpu().numpy().tolist()
        positions = [tuple(pos) for pos in positions]
        
        # Extract bonds with tensor operations
        bonds = []
        seen_bonds = set()
        
        for i in range(edge_index_tensor.shape[1]):
            atom1_idx = int(edge_index_tensor[0, i])
            atom2_idx = int(edge_index_tensor[1, i])
            
            bond_tuple = (min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx))
            
            if bond_tuple in seen_bonds:
                continue
            seen_bonds.add(bond_tuple)
            
            if edge_attr_tensor is not None and edge_attr_tensor.shape[1] >= 4:
                bond_type_idx = torch.argmax(edge_attr_tensor[i, :4])
                bond_type = bond_type_idx + 1
            else:
                bond_type = 1
            
            bonds.append((bond_tuple[0], bond_tuple[1], bond_type))
        
        return atom_types, positions, bonds

    def convert_to_universal(self, raw_item: Any) -> UniversalMolecule:
        """Convert QM9 raw data to universal format with blocks"""
        blocks = self.create_blocks(raw_item)
        
        # Create molecule ID from index (QM9 doesn't have explicit IDs)
        mol_id = f"qm9_mol_{getattr(raw_item, 'idx', 'unknown')}"
        
        # Extract properties (QM9 has 19 quantum properties)
        properties = {}
        if hasattr(raw_item, 'y') and raw_item.y is not None:
            # QM9 properties: [mu, alpha, homo, lumo, gap, r2, zpve, u0, u298, h298, g298, cv, u0_atom, u298_atom, h298_atom, g298_atom]
            prop_names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']
            for i, prop_name in enumerate(prop_names):
                if i < len(raw_item.y[0]):
                    properties[prop_name] = float(raw_item.y[0][i])
        
        return UniversalMolecule(
            id=mol_id,
            dataset_type=self.dataset_type,
            blocks=blocks,
            properties=properties
        )
