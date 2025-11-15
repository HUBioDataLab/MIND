#!/usr/bin/env python3

# data_loading/cache_to_pyg.py

"""
Universal Representation Datasets using InMemoryDataset

This module provides dataset classes that cache PyTorch Geometric
tensors for instant loading.
"""

import os
import sys
import torch
import pickle
import shutil
from typing import List, Optional, Iterator
from itertools import islice
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import warnings

# Add universal representation imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loading.data_types import UniversalMolecule

from torch_cluster import radius_graph # preferred
warnings.filterwarnings('ignore')

# ADDED: Helper generator function to read a stream of pickled objects
def load_molecules_iteratively(file_path: str) -> Iterator[UniversalMolecule]:
    """A generator to load molecules one by one from a pickle stream."""
    with open(file_path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

class OptimizedUniversalDataset(InMemoryDataset):
    """
    Universal Dataset using InMemoryDataset for tensor caching.
    It processes data in chunks to handle large datasets that don't fit in RAM.
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
                 pre_filter=None
                ):
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

        # The parent constructor handles loading if the processed file exists,
        # or triggers self.process() if it doesn't.
        super().__init__(root, transform, pre_transform, pre_filter)

        # Load the processed data
        self.data, self.slices = torch.load(self.processed_paths[0], map_location='cpu')

    @property
    def raw_file_names(self) -> List[str]:
        return [os.path.basename(self.universal_cache_path)]

    @property
    def processed_file_names(self) -> List[str]:
        """Generates a unique filename for the cache based on processing parameters."""
        cache_name = os.path.basename(self.universal_cache_path).replace('.pkl', '')
        if self.max_samples:
            cache_name += f"_samples_{self.max_samples}"
        if self.molecule_max_atoms:
            cache_name += f"_maxatoms_{self.molecule_max_atoms}"
        config_sig = f"cutoff_{self.cutoff_distance}_neighbors_{self.max_neighbors}"
        return [f"optimized_{cache_name}_{config_sig}.pt"]

    def download(self):
        """Copies the universal .pkl cache to the raw_dir for PyG to find."""
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(raw_path):
            print(f"📋 Copying universal cache to raw directory...")
            shutil.copy2(self.universal_cache_path, raw_path)

    def process(self):
        """
        Process molecules from .pkl cache and convert to PyG format.
        
        MEMORY NOTE: This processes all molecules in one pass.
        For large datasets (>50K molecules), use external chunking:
        - Split dataset at manifest level (e.g., 1M → 50 chunks of 20K)
        - Process each chunk separately using this method
        - Use LazyUniversalDataset for training (loads chunks on-demand)
        
        Recommended chunk sizes for processing:
        - 10K molecules: ~200MB RAM
        - 20K molecules: ~400MB RAM  
        - 50K molecules: ~1GB RAM
        """
        print(f"🔄 Processing universal representations...")

        # 1. Load molecules from .pkl file
        molecule_iterator = load_molecules_iteratively(self.raw_paths[0])

        # 2. Apply max_samples limit if specified
        if self.max_samples is not None:
            molecule_iterator = islice(molecule_iterator, self.max_samples)

        # 3. Convert all molecules to PyG Data objects
        data_list = []
        for mol in tqdm(molecule_iterator, desc="Converting molecules"):
            pyg_data = self._create_pyg_data_object(mol)
            if pyg_data is not None:
                data_list.append(pyg_data)

        if not data_list:
            raise RuntimeError("No molecules were processed successfully. Check data and filters.")

        # Memory usage info
        ram_mb = len(data_list) * 20 / 1024
        print(f"✅ Converted {len(data_list):,} molecules to PyG format")
        print(f"💾 Estimated RAM usage: ~{ram_mb:.0f}MB")
        
        if ram_mb > 2000:
            print(f"⚠️  Large dataset! Consider using smaller chunks for 1M+ proteins.")

        # 4. Collate and save the final dataset
        print(f"🔄 Collating dataset...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"✅ Processing complete! Saved to: {self.processed_paths[0]}")

    def _create_pyg_data_object(self, mol: UniversalMolecule) -> Optional[Data]:
        """Helper function to filter and convert a single UniversalMolecule."""
        if self.molecule_max_atoms is not None and len(mol.get_all_atoms()) > self.molecule_max_atoms:
            return None

        pyg_data = self._universal_to_pyg(mol)

        if pyg_data is None: return None
        if self.pre_filter is not None and not self.pre_filter(pyg_data): return None
        if self.pre_transform is not None: pyg_data = self.pre_transform(pyg_data)

        return pyg_data

    def _universal_to_pyg(self, mol: UniversalMolecule) -> Optional[Data]:
        try:
            atoms = mol.get_all_atoms()
            if not atoms: return None

            positions = torch.tensor([atom.position for atom in atoms], dtype=torch.float32)
            atomic_numbers = torch.tensor([self._element_to_atomic_number(atom.element) for atom in atoms], dtype=torch.long)

            block_indices = torch.tensor([atom.block_idx for atom in atoms], dtype=torch.long)
            entity_indices = torch.tensor([atom.entity_idx for atom in atoms], dtype=torch.long)
            pos_codes = [atom.pos_code for atom in atoms]
            block_symbols = [block.symbol for block in mol.blocks]

            num_atoms = len(atoms)
            if num_atoms > 1:
                edge_index = radius_graph(
                    positions, r=float(self.cutoff_distance), batch=None, loop=False,
                    max_num_neighbors=int(self.max_neighbors)
                )
                edge_attr = self._calculate_edge_features(positions, edge_index)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float32)

            data = Data(
                pos=positions,
                z=atomic_numbers,
                edge_index=edge_index,
                edge_attr=edge_attr,
                block_idx=block_indices,
                entity_idx=entity_indices,
                pos_code=pos_codes,
                block_symbols=block_symbols,
                mol_id=str(mol.id),
                dataset_type=mol.dataset_type,
                num_nodes=len(atoms),
                num_edges=edge_index.size(1),
            )
            return data
        except Exception:
            return None

    """
    Suggestion: For performance and to remove the RDKit dependency from this script,
    a simple dictionary lookup can be used instead.
    This is a non-critical optimization for later.
    """
    def _element_to_atomic_number(self, element: str) -> int:
        """Convert element symbol to atomic number."""
        try:
            from rdkit import Chem
            return Chem.GetPeriodicTable().GetAtomicNumber(element.capitalize())
        except Exception:
            return 6 # Default to Carbon if unknown

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


class OptimizedUniversalCOCONUTDataset(OptimizedUniversalDataset):
    """
    COCONUT Dataset using cached PyTorch Geometric tensors
    
    This specialized dataset class for COCONUT provides instant loading by caching
    converted PyTorch Geometric tensors, eliminating the conversion bottleneck.
    """
    
    def __init__(self, 
                 root: str = None,
                 universal_cache_path: str = None,
                 max_samples: Optional[int] = None,
                 cutoff_distance: float = 5.0,
                 max_neighbors: int = 32,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        
        # Default root path
        if root is None:
            root = os.path.join(
                os.path.dirname(__file__), 
                'processed', 'coconut_optimized'
            )
        
        # Default cache path for COCONUT
        if universal_cache_path is None:
            universal_cache_path = os.path.join(
                os.path.dirname(__file__), 
                'cache', 'universal_coconut_all.pkl'
            )
        
        super().__init__(root, universal_cache_path, max_samples, cutoff_distance, max_neighbors, 
                        transform, pre_transform, pre_filter)

class OptimizedUniversalRNADataset(OptimizedUniversalDataset):
    """
    RNA Dataset using cached PyTorch Geometric tensors

    This specialized dataset class for RNA provides instant loading by caching
    converted PyTorch Geometric tensors from filtered CIF files.
    """

    def __init__(self,
                 root: str = None,
                 universal_cache_path: str = None,
                 max_samples: Optional[int] = None,
                 molecule_max_atoms: Optional[int] = None,
                 cutoff_distance: float = 6.0,  # RNA-specific: slightly larger
                 max_neighbors: int = 20,  # RNA-specific: fewer neighbors
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        # Default root path for RNA
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'rna_optimized')

        # Default cache path for RNA
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_rna.pkl')

        super().__init__(root, universal_cache_path, max_samples, molecule_max_atoms, cutoff_distance, max_neighbors,
                        transform, pre_transform, pre_filter)



if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Convert universal .pkl cache to PyG .pt format")
    parser.add_argument("--input-pkl", required=True, help="Path to input .pkl file")
    parser.add_argument("--output-dir", required=True, help="Output directory for .pt file")
    parser.add_argument("--dataset-type", choices=["qm9", "lba", "pdb", "coconut", "rna"], required=True, help="Dataset type")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Distance cutoff for edges (Å)")
    parser.add_argument("--max-neighbors", type=int, default=64, help="Maximum neighbors per atom")
    parser.add_argument("--force", action="store_true", help="Force rebuild if output exists")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if output already exists (look for any .pt file in processed subdir)
    processed_dir = os.path.join(args.output_dir, "processed")
    if os.path.exists(processed_dir) and not args.force:
        pt_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
        if pt_files:
            print(f"✅ Output already exists: {os.path.join(processed_dir, pt_files[0])}")
            print(f"💡 Use --force to rebuild")
            sys.exit(0)

    # Create dataset based on type
    if args.dataset_type == "qm9":
        dataset = OptimizedUniversalQM9Dataset(
            root=args.output_dir,
            universal_cache_path=args.input_pkl,
            max_samples=args.max_samples,
            cutoff_distance=args.cutoff,
            max_neighbors=args.max_neighbors
        )
        print(f"✅ Created QM9 dataset: {len(dataset)} samples")
    elif args.dataset_type == "lba":
        dataset = OptimizedUniversalLBADataset(
            root=args.output_dir,
            universal_cache_path=args.input_pkl,
            max_samples=args.max_samples,
            cutoff_distance=args.cutoff,
            max_neighbors=args.max_neighbors
        )
        print(f"✅ Created LBA dataset: {len(dataset)} samples")
    elif args.dataset_type == "pdb":
        dataset = OptimizedUniversalDataset(
            root=args.output_dir,
            universal_cache_path=args.input_pkl,
            max_samples=args.max_samples,
            cutoff_distance=args.cutoff,
            max_neighbors=args.max_neighbors
        )
        print(f"✅ Created PDB dataset: {len(dataset)} samples")
    elif args.dataset_type == "coconut":
        dataset = OptimizedUniversalCOCONUTDataset(
            root=args.output_dir,
            universal_cache_path=args.input_pkl,
            max_samples=args.max_samples,
            cutoff_distance=args.cutoff,
            max_neighbors=args.max_neighbors
        )
        print(f"✅ Created COCONUT dataset: {len(dataset)} samples")
    elif args.dataset_type == "rna":
        dataset = OptimizedUniversalRNADataset(
            root=args.output_dir,
            universal_cache_path=args.input_pkl,
            max_samples=args.max_samples,
            cutoff_distance=args.cutoff,
            max_neighbors=args.max_neighbors
        )
        print(f"✅ Created RNA dataset: {len(dataset)} samples")

