#!/usr/bin/env python3
"""
Universal Representation Datasets using InMemoryDataset

This module provides dataset classes that cache PyTorch Geometric
tensors for instant loading.

Key Features:
- Uses InMemoryDataset for instant tensor loading
- Pre-processes universal → PyTorch Geometric conversion ONCE
- Caches optimized tensors to disk for instant subsequent loads
- Compatible with existing training infrastructure
"""

import os
import sys
import torch
import pickle
from typing import List, Dict, Any, Optional, Tuple
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import warnings

# Add universal representation imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loading.data_types import UniversalMolecule, UniversalBlock, UniversalAtom

from torch_cluster import radius_graph  # preferred
warnings.filterwarnings('ignore')

class OptimizedUniversalDataset(InMemoryDataset):
    """
    Universal Dataset using InMemoryDataset for tensor caching
    
    This class loads cached universal molecular representations, converts them 
    to PyTorch Geometric format ONCE, and caches the tensors for instant loading.
    """
    
    def __init__(self, 
                 root: str,
                 universal_cache_path: str,
                 max_samples: Optional[int] = None,
                 molecule_max_atoms: Optional[int] = None,
                 cutoff_distance: float = 5.0,
                 max_neighbors: int = 32,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """
        Initialize Optimized Universal Dataset
        
        Args:
            root: Root directory for processed tensor cache
            universal_cache_path: Path to cached universal representations (.pkl file)
            max_samples: Maximum number of samples to load (None for all)
            molecule_max_atoms: Maximum number of atoms per molecule (None for no limit)
            cutoff_distance: Distance cutoff for edge construction
            max_neighbors: Maximum number of neighbors per atom
            transform: Transform to apply to each sample
            pre_transform: Pre-transform to apply during processing
            pre_filter: Pre-filter to apply during processing
        """
        self.universal_cache_path = universal_cache_path
        self.max_samples = max_samples
        self.molecule_max_atoms = molecule_max_atoms
        self.cutoff_distance = cutoff_distance
        self.max_neighbors = max_neighbors
        
        # Ensure root directory exists
        os.makedirs(root, exist_ok=True)
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Check if processed file exists for current configuration
        processed_path = self.processed_paths[0]
        if os.path.exists(processed_path):
            print(f"✅ Loading cached processed dataset: {os.path.basename(processed_path)}")
            # Robust load that works with PyTorch >=2.6 (weights_only default) and older versions
            try:
                self.data, self.slices = torch.load(processed_path, map_location='cpu', weights_only=False)
            except TypeError:
                # Older torch without weights_only
                self.data, self.slices = torch.load(processed_path, map_location='cpu')
            print(f"✅ Loaded {len(self)} samples from processed cache")
        else:
            print(f"🔄 No processed cache found for current configuration. Will process from universal cache...")
            # This will trigger process() in the parent class
            pass

    @property
    def raw_file_names(self) -> List[str]:
        """Raw files - just our universal cache"""
        return [os.path.basename(self.universal_cache_path)]

    @property
    def processed_file_names(self) -> List[str]:
        """Processed tensor cache file with config signature"""
        cache_name = os.path.basename(self.universal_cache_path).replace('.pkl', '')
        if self.max_samples:
            cache_name += f"_samples_{self.max_samples}"
        if self.molecule_max_atoms:
            cache_name += f"_maxatoms_{self.molecule_max_atoms}"
        config_sig = f"cutoff_{self.cutoff_distance}_neighbors_{self.max_neighbors}"
        return [f"optimized_{cache_name}_{config_sig}.pt"]

    def download(self):
        """Copy universal cache to raw directory if needed"""
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(raw_path):
            print(f"📋 Copying universal cache to raw directory...")
            import shutil
            shutil.copy2(self.universal_cache_path, raw_path)

    def process(self):
        """
        Convert universal representations to PyTorch Geometric tensors and cache them
        
        This is the expensive operation that happens ONCE, then tensors are cached!
        """
        print(f"🔄 Processing universal representations to PyTorch Geometric tensors...")
        print(f"📂 Loading from pickle cache: {self.universal_cache_path}")
        print(f"💾 Will save processed tensors to: {self.processed_paths[0]}")
        
        # Load universal representations with backward compatibility
        import types
        
        # Create compatibility module for pickle loading
        from data_loading.data_types import UniversalMolecule, UniversalBlock, UniversalAtom
        data_types_module = types.ModuleType('data_types')
        data_types_module.UniversalMolecule = UniversalMolecule
        data_types_module.UniversalBlock = UniversalBlock
        data_types_module.UniversalAtom = UniversalAtom
        sys.modules['data_types'] = data_types_module
        
        try:
            with open(self.universal_cache_path, 'rb') as f:
                universal_molecules = pickle.load(f)
        finally:
            # Clean up the temporary module
            if 'data_types' in sys.modules:
                del sys.modules['data_types']
        
        # Limit samples if requested
        if self.max_samples is not None:
            universal_molecules = universal_molecules[:self.max_samples]
            print(f"🔬 Using {self.max_samples} samples for processing")
        
        print(f"✅ Loaded {len(universal_molecules)} universal molecules")
        
        # Convert to PyTorch Geometric format
        data_list = []
        filtered_count = 0
        print("🔄 Converting to PyTorch Geometric format...")
        
        for i, mol in enumerate(tqdm(universal_molecules, desc="Converting molecules")):
            try:
                # Check filtering before expensive conversion
                atoms = mol.get_all_atoms()
                if self.molecule_max_atoms is not None and len(atoms) > self.molecule_max_atoms:
                    filtered_count += 1
                    continue
                    
                pyg_data = self._universal_to_pyg(mol)
                if pyg_data is None:
                    continue
                    
                # Apply pre_filter if provided
                if self.pre_filter is not None and not self.pre_filter(pyg_data):
                    continue
                
                # Apply pre_transform if provided  
                if self.pre_transform is not None:
                    pyg_data = self.pre_transform(pyg_data)
                
                data_list.append(pyg_data)
            except Exception as e:
                print(f"⚠️ Skipping molecule {i} due to conversion error: {e}")
                continue
        
        print(f"✅ Converted {len(data_list)} molecules to PyTorch Geometric format")
        if self.molecule_max_atoms is not None and filtered_count > 0:
            print(f"🔍 Filtered out {filtered_count} molecules with more than {self.molecule_max_atoms} atoms")
        
        # Efficiently collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        print(f"✅ Processing complete! Saved {len(data_list)} molecules to: {os.path.basename(self.processed_paths[0])}")

    def _universal_to_pyg(self, mol: UniversalMolecule) -> Optional[Data]:
        """
        Convert a UniversalMolecule to PyTorch Geometric Data object
        
        Args:
            mol: UniversalMolecule object
            
        Returns:
            PyTorch Geometric Data object or None if conversion fails
        """
        try:
            atoms = mol.get_all_atoms()
            if len(atoms) == 0:
                return None
            
            # Extract positions
            positions = torch.tensor([atom.position for atom in atoms], dtype=torch.float32)
            
            # Extract atomic numbers
            atomic_numbers = []
            for atom in atoms:
                atomic_num = self._element_to_atomic_number(atom.element)
                atomic_numbers.append(atomic_num)
            
            atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
            
            # Note: Removed wasteful one-hot encoding (was 118*N bytes per molecule)
            # We only need atomic numbers (z) which our model actually uses
            
            # Extract block information for hierarchical structure
            block_indices = torch.tensor([atom.block_idx for atom in atoms], dtype=torch.long)
            entity_indices = torch.tensor([atom.entity_idx for atom in atoms], dtype=torch.long)
            pos_codes = [atom.pos_code for atom in atoms]  # Position codes for each atom
            
            # Create block symbols tensor (for GET compatibility)
            block_symbols = [block.symbol for block in mol.blocks]
            
            # Create edge index using radius graph (more efficient than fully connected)
            num_atoms = len(atoms)
            if num_atoms > 1 and radius_graph is not None:
                try:
                    # Use configured geometric neighborhood parameters
                    cutoff = float(self.cutoff_distance)
                    max_neighbors = int(self.max_neighbors)
                    edge_index = radius_graph(
                        positions, r=cutoff, batch=None, loop=False, max_num_neighbors=max_neighbors
                    )
                    # Calculate edge distances
                    edge_attr = self._calculate_edge_features(positions, edge_index)
                except Exception as e:
                    # Fallback to fully connected if radius_graph fails
                    edge_index = torch.combinations(torch.arange(num_atoms), r=2).t().contiguous()
                    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                    edge_attr = self._calculate_edge_features(positions, edge_index)
            else:
                # Single atom molecule or no radius_graph available
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float32)
            
            # Create PyTorch Geometric Data object
            data = Data(
                # Note: Removed x=node_features (wasteful one-hot encoding)
                # Only keeping z=atomic_numbers which is what the model actually uses
                pos=positions,    # 3D coordinates
                z=atomic_numbers, # Atomic numbers (this is all we need!)
                edge_index=edge_index,
                edge_attr=edge_attr,
                
                # Universal representation specific attributes
                block_idx=block_indices,
                entity_idx=entity_indices,
                pos_code=pos_codes,  # Position codes for hierarchical featurization
                block_symbols=block_symbols,
                mol_id=str(mol.id),
                dataset_type=mol.dataset_type,
                
                # For compatibility with existing training code
                num_nodes=len(atoms),
                num_edges=edge_index.size(1),
            )
            
            return data
            
        except Exception as e:
            print(f"Error converting molecule {mol.id}: {e}")
            return None
    
    def _element_to_atomic_number(self, element: str) -> int:
        """Convert element symbol to atomic number using RDKit only"""
        try:
            from rdkit import Chem
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(element)
            return atomic_num
        except Exception:
            # Default to Carbon for any unknown/invalid elements
            return 6
    
    def _calculate_edge_features(self, positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Calculate edge features (distances) between connected atoms"""
        row, col = edge_index
        diff = positions[row] - positions[col]
        distances = torch.norm(diff, dim=1, keepdim=True)
        return distances


class OptimizedUniversalQM9Dataset(OptimizedUniversalDataset):
    """
    QM9 Dataset using cached PyTorch Geometric tensors
    
    This specialized dataset class for QM9 provides instant loading by caching
    converted PyTorch Geometric tensors, eliminating the conversion bottleneck.
    """
    
    def __init__(self, 
                 root: str = None,
                 universal_cache_path: str = None,
                 max_samples: Optional[int] = None,
                 molecule_max_atoms: Optional[int] = None,
                 cutoff_distance: float = 5.0,
                 max_neighbors: int = 32,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        
        # Default root path
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'qm9_optimized')
        
        # Default cache path for QM9
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_qm9_all.pkl')
        
        super().__init__(root, universal_cache_path, max_samples, molecule_max_atoms, cutoff_distance, max_neighbors, 
                        transform, pre_transform, pre_filter)


class OptimizedUniversalLBADataset(OptimizedUniversalDataset):
    """
    LBA Dataset using cached PyTorch Geometric tensors
    
    This specialized dataset class for LBA provides instant loading by caching
    converted PyTorch Geometric tensors, eliminating the conversion bottleneck.
    """
    
    def __init__(self, 
                 root: str = None,
                 universal_cache_path: str = None,
                 max_samples: Optional[int] = None,
                 molecule_max_atoms: Optional[int] = None,
                 cutoff_distance: float = 5.0,
                 max_neighbors: int = 32,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        
        # Default root path
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'lba_optimized')
        
        # Default cache path for LBA
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_lba_all.pkl')
        
        super().__init__(root, universal_cache_path, max_samples, molecule_max_atoms, cutoff_distance, max_neighbors, 
                        transform, pre_transform, pre_filter)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create universal datasets")
    parser.add_argument("--dataset", choices=["qm9", "lba"], required=True, help="Dataset type")
    parser.add_argument("--root", default=None, help="Root directory for dataset")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--cutoff-distance", type=float, default=5.0, help="Distance cutoff for edges")
    parser.add_argument("--max-neighbors", type=int, default=32, help="Maximum neighbors per atom")
    
    args = parser.parse_args()
    
    if args.dataset == "qm9":
        root_dir = args.root or "./data/optimized_universal_qm9"
        dataset = OptimizedUniversalQM9Dataset(root=root_dir, max_samples=args.max_samples)
        print(f"✅ Created QM9 dataset: {len(dataset)} samples")
    elif args.dataset == "lba":
        root_dir = args.root or "./data/optimized_universal_lba"
        dataset = OptimizedUniversalLBADataset(root=root_dir, max_samples=args.max_samples)
        print(f"✅ Created LBA dataset: {len(dataset)} samples")