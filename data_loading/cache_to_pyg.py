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
from typing import List, Optional, Iterator, Callable
from itertools import islice
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import warnings

# Add universal representation imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loading.data_types import UniversalMolecule

from torch_cluster import radius_graph
warnings.filterwarnings('ignore')

# Periodic table: element symbol -> atomic number
ELEMENT_TO_ATOMIC_NUMBER = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
    'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
    'Rn': 86,
    # Virtual atom for 14-atom uniform representation
    # All virtual atoms share the same type to prevent sequence leakage
    'V': 120,       # Generic virtual atom (used for padding in 14-atom representation)
}
DEFAULT_ATOMIC_NUMBER = 6  # Carbon


def load_molecules_iteratively(file_path: str) -> Iterator[UniversalMolecule]:
    """Generator to load molecules one by one from a pickle stream."""
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

    def __init__(
        self,
        root: str,
        universal_cache_path: str,
        max_samples: Optional[int] = None,
        molecule_max_atoms: Optional[int] = None,
        cutoff_distance: float = 5.0,
        max_neighbors: int = 32,
        use_hybrid_edges: bool = False,
        sequence_neighbors_k: int = 3,
        max_spatial_neighbors: int = 48,
        num_random_edges: int = 8,
        random_edge_min_distance: float = 10.0,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
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
            use_hybrid_edges: Enable Salad-inspired 3-tier hybrid edge construction
            sequence_neighbors_k: Tier 1: Connect residue i to i±k neighbors
            max_spatial_neighbors: Tier 2: Max spatial neighbors (filtered)
            num_random_edges: Tier 3: Number of random long-range edges
            random_edge_min_distance: Tier 3: Minimum distance for random edges (Å)
            transform: Transform to apply to each sample
            pre_transform: Pre-transform to apply during processing
            pre_filter: Pre-filter to apply during processing
        """
        self.universal_cache_path = universal_cache_path
        self.max_samples = max_samples
        self.molecule_max_atoms = molecule_max_atoms
        self.cutoff_distance = cutoff_distance
        self.max_neighbors = max_neighbors
        self.use_hybrid_edges = use_hybrid_edges
        self.sequence_neighbors_k = sequence_neighbors_k
        self.max_spatial_neighbors = max_spatial_neighbors
        self.num_random_edges = num_random_edges
        self.random_edge_min_distance = random_edge_min_distance

        os.makedirs(root, exist_ok=True)
        super().__init__(root, transform, pre_transform, pre_filter)
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
        
        # Include hybrid edge parameters in cache signature
        if self.use_hybrid_edges:
            config_sig = (f"hybrid_seq{self.sequence_neighbors_k}_"
                         f"spatial{self.max_spatial_neighbors}_"
                         f"rand{self.num_random_edges}_"
                         f"mindist{self.random_edge_min_distance}")
        else:
            config_sig = f"cutoff_{self.cutoff_distance}_neighbors_{self.max_neighbors}"
        
        return [f"optimized_{cache_name}_{config_sig}.pt"]

    def download(self) -> None:
        """Copies the universal .pkl cache to the raw_dir for PyG to find."""
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(raw_path):
            print(f"📋 Copying universal cache to raw directory...")
            shutil.copy2(self.universal_cache_path, raw_path)

    def process(self) -> None:
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
        data_list: List[Data] = []
        skipped_count = 0
        for mol in tqdm(molecule_iterator, desc="Converting molecules"):
            try:
                pyg_data = self._create_pyg_data_object(mol)
                if pyg_data is not None:
                    data_list.append(pyg_data)
                else:
                    skipped_count += 1
            except Exception as e:
                warnings.warn(f"Skipping molecule {mol.id}: {e}", UserWarning)
                skipped_count += 1

        if not data_list:
            raise RuntimeError(
                f"No molecules were processed successfully. Skipped {skipped_count} molecules."
            )

        # Memory usage info
        ram_mb = len(data_list) * 20 / 1024
        print(f"✅ Converted {len(data_list):,} molecules to PyG format")
        if skipped_count > 0:
            print(f"⚠️  Skipped {skipped_count} molecules")
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
        if pyg_data is None:
            return None
        if self.pre_filter is not None and not self.pre_filter(pyg_data):
            return None
        if self.pre_transform is not None:
            pyg_data = self.pre_transform(pyg_data)
        return pyg_data

    def _build_sequence_edges(
        self, 
        block_indices: torch.Tensor, 
        entity_indices: torch.Tensor, 
        pos_codes: List[str], 
        k: int
    ) -> torch.Tensor:
        """
        Build Tier 1: Sequence-based edges (backbone connectivity).
        Connects residue i to i±1, i±2, ..., i±k neighbors via CA atoms.
        
        Args:
            block_indices: Block (residue) index for each atom
            entity_indices: Entity (chain) index for each atom
            pos_codes: Position codes (e.g., 'CA', 'N', 'C', etc.)
            k: Number of sequence neighbors to connect (±k)
            
        Returns:
            edge_index: [2, num_edges] tensor of sequence-based edges
        """
        # Find CA atoms only
        ca_mask = torch.tensor([code == 'CA' for code in pos_codes], dtype=torch.bool)
        ca_indices = torch.where(ca_mask)[0]
        
        if len(ca_indices) < 2:
            return torch.empty((2, 0), dtype=torch.long)
        
        ca_block_indices = block_indices[ca_indices]
        ca_entity_indices = entity_indices[ca_indices]
        
        edges = []
        for i, ca_idx in enumerate(ca_indices):
            current_block = ca_block_indices[i]
            current_entity = ca_entity_indices[i]
            
            # Connect to next k neighbors in sequence
            for offset in range(1, k + 1):
                if i + offset < len(ca_indices):
                    neighbor_block = ca_block_indices[i + offset]
                    neighbor_entity = ca_entity_indices[i + offset]
                    
                    # Only connect within same entity and sequential blocks
                    if (neighbor_entity == current_entity and 
                        neighbor_block == current_block + offset):
                        neighbor_idx = ca_indices[i + offset]
                        edges.append([ca_idx.item(), neighbor_idx.item()])
                        edges.append([neighbor_idx.item(), ca_idx.item()])  # Bidirectional
        
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def _build_random_edges(
        self,
        positions: torch.Tensor,
        block_indices: torch.Tensor,
        entity_indices: torch.Tensor,
        pos_codes: List[str],
        num_random: int,
        min_distance: float
    ) -> torch.Tensor:
        """
        Build Tier 3: Random long-range edges with inverse distance³ weighting.
        
        Args:
            positions: Atom positions [N, 3]
            block_indices: Block index for each atom
            entity_indices: Entity index for each atom
            pos_codes: Position codes
            num_random: Number of random edges to sample per CA atom
            min_distance: Minimum distance threshold (Å)
            
        Returns:
            edge_index: [2, num_edges] tensor of random long-range edges
        """
        # Find CA atoms
        ca_mask = torch.tensor([code == 'CA' for code in pos_codes], dtype=torch.bool)
        ca_indices = torch.where(ca_mask)[0]
        
        if len(ca_indices) < 2 or num_random == 0:
            return torch.empty((2, 0), dtype=torch.long)
        
        ca_positions = positions[ca_indices]
        ca_block_indices = block_indices[ca_indices]
        
        edges = []
        for i, ca_idx in enumerate(ca_indices):
            current_pos = ca_positions[i:i+1]
            current_block = ca_block_indices[i]
            
            # Calculate distances to all other CA atoms
            distances = torch.norm(ca_positions - current_pos, dim=1)
            
            # Filter: exclude self and close neighbors (below min_distance)
            valid_mask = (distances >= min_distance) & (torch.arange(len(ca_indices)) != i)
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                continue
            
            valid_distances = distances[valid_indices]
            
            # Inverse distance³ weighting (prevents division by zero with min_distance filter)
            weights = 1.0 / (valid_distances ** 3 + 1e-6)
            weights = weights / weights.sum()  # Normalize
            
            # Sample random neighbors
            num_to_sample = min(num_random, len(valid_indices))
            sampled_idx = torch.multinomial(weights, num_to_sample, replacement=False)
            sampled_neighbors = valid_indices[sampled_idx]
            
            for neighbor_local_idx in sampled_neighbors:
                neighbor_global_idx = ca_indices[neighbor_local_idx]
                edges.append([ca_idx.item(), neighbor_global_idx.item()])
                edges.append([neighbor_global_idx.item(), ca_idx.item()])  # Bidirectional
        
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def _filter_duplicate_edges(
        self, 
        edge_index_main: torch.Tensor, 
        edge_index_to_remove: torch.Tensor
    ) -> torch.Tensor:
        """
        Remove edges from edge_index_main that exist in edge_index_to_remove.
        
        Args:
            edge_index_main: Main edge set [2, E1]
            edge_index_to_remove: Edges to exclude [2, E2]
            
        Returns:
            Filtered edge_index [2, E_filtered]
        """
        if edge_index_main.size(1) == 0:
            return edge_index_main
        if edge_index_to_remove.size(1) == 0:
            return edge_index_main
        
        # Convert to set of tuples for fast lookup
        edges_to_remove_set = set(
            zip(edge_index_to_remove[0].tolist(), edge_index_to_remove[1].tolist())
        )
        
        # Filter main edges
        filtered_edges = [
            [src, dst] for src, dst in zip(edge_index_main[0].tolist(), edge_index_main[1].tolist())
            if (src, dst) not in edges_to_remove_set
        ]
        
        if not filtered_edges:
            return torch.empty((2, 0), dtype=torch.long)
        
        return torch.tensor(filtered_edges, dtype=torch.long).t()
    
    def _build_hybrid_edges(
        self,
        positions: torch.Tensor,
        block_indices: torch.Tensor,
        entity_indices: torch.Tensor,
        pos_codes: List[str]
    ) -> torch.Tensor:
        """
        Build 3-tier hybrid edge construction (Salad-inspired).
        
        Returns:
            edge_index: [2, num_edges] tensor combining all 3 tiers
        """
        # Tier 1: Sequence-based edges (backbone)
        tier1_edges = self._build_sequence_edges(
            block_indices, entity_indices, pos_codes, self.sequence_neighbors_k
        )
        
        # Tier 2: Spatial edges (radius graph with duplicate filtering)
        tier2_edges = radius_graph(
            positions, r=float(self.cutoff_distance), batch=None, loop=False,
            max_num_neighbors=int(self.max_spatial_neighbors)
        )
        # Filter out Tier 1 edges from Tier 2 to maximize information diversity
        tier2_edges = self._filter_duplicate_edges(tier2_edges, tier1_edges)
        
        # Tier 3: Random long-range edges
        tier3_edges = self._build_random_edges(
            positions, block_indices, entity_indices, pos_codes,
            self.num_random_edges, self.random_edge_min_distance
        )
        
        # Combine all tiers
        all_edges = torch.cat([tier1_edges, tier2_edges, tier3_edges], dim=1)
        
        return all_edges

    def _universal_to_pyg(self, mol: UniversalMolecule) -> Optional[Data]:
        """Convert a UniversalMolecule to PyTorch Geometric Data object."""
        try:
            atoms = mol.get_all_atoms()
            if not atoms:
                return None

            positions = torch.tensor([atom.position for atom in atoms], dtype=torch.float32)
            atomic_numbers = torch.tensor(
                [self._element_to_atomic_number(atom.element) for atom in atoms],
                dtype=torch.long
            )
            block_indices = torch.tensor([atom.block_idx for atom in atoms], dtype=torch.long)
            entity_indices = torch.tensor([atom.entity_idx for atom in atoms], dtype=torch.long)
            pos_codes = [atom.pos_code for atom in atoms]
            block_symbols = [block.symbol for block in mol.blocks]

            num_atoms = len(atoms)
            if num_atoms > 1:
                # Choose edge construction method
                if self.use_hybrid_edges:
                    edge_index = self._build_hybrid_edges(
                        positions, block_indices, entity_indices, pos_codes
                    )
                else:
                    # Legacy: radius graph
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
        except (ValueError, IndexError, TypeError) as e:
            warnings.warn(f"Error converting molecule {mol.id}: {e}", UserWarning)
            return None
        except Exception as e:
            warnings.warn(f"Unexpected error converting molecule {mol.id}: {e}", UserWarning)
            return None

    def _element_to_atomic_number(self, element: str) -> int:
        """Convert element symbol to atomic number using dictionary lookup."""
        element_normalized = element.capitalize() if len(element) <= 2 else element
        return ELEMENT_TO_ATOMIC_NUMBER.get(element_normalized, DEFAULT_ATOMIC_NUMBER)

    def _calculate_edge_features(self, positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Calculate edge features (distances) between connected atoms"""
        row, col = edge_index
        diff = positions[row] - positions[col]
        distances = torch.norm(diff, dim=1, keepdim=True)
        return distances


class OptimizedUniversalQM9Dataset(OptimizedUniversalDataset):
    """QM9 Dataset using cached PyTorch Geometric tensors."""

    def __init__(
        self,
        root: Optional[str] = None,
        universal_cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        molecule_max_atoms: Optional[int] = None,
        cutoff_distance: float = 5.0,
        max_neighbors: int = 32,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'qm9_optimized')

        # Default cache path for QM9
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_qm9_all.pkl')
        super().__init__(root, universal_cache_path, max_samples, molecule_max_atoms, cutoff_distance, max_neighbors,
                        transform, pre_transform, pre_filter)


class OptimizedUniversalLBADataset(OptimizedUniversalDataset):
    """LBA Dataset using cached PyTorch Geometric tensors."""

    def __init__(
        self,
        root: Optional[str] = None,
        universal_cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        molecule_max_atoms: Optional[int] = None,
        cutoff_distance: float = 5.0,
        max_neighbors: int = 32,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'lba_optimized')
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_lba_all.pkl')
        super().__init__(root, universal_cache_path, max_samples, molecule_max_atoms, cutoff_distance, max_neighbors,
                        transform, pre_transform, pre_filter)


class OptimizedUniversalCOCONUTDataset(OptimizedUniversalDataset):
    """COCONUT Dataset using cached PyTorch Geometric tensors."""
    
    def __init__(
        self,
        root: Optional[str] = None,
        universal_cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        cutoff_distance: float = 5.0,
        max_neighbors: int = 32,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'coconut_optimized')
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_coconut_all.pkl')
        super().__init__(root, universal_cache_path, max_samples, None, cutoff_distance, max_neighbors,
                        transform, pre_transform, pre_filter)

class OptimizedUniversalRNADataset(OptimizedUniversalDataset):
    """
    RNA Dataset using cached PyTorch Geometric tensors.
    
    RNA-specific optimizations:
    - Larger cutoff distance (8.0Å) for base pairing interactions
    - Fewer max neighbors (32) as RNA has sparser structure than proteins
    """

    def __init__(
        self,
        root: Optional[str] = None,
        universal_cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        molecule_max_atoms: Optional[int] = None,
        cutoff_distance: float = 8.0,  # Changed from 6.0 to 8.0 for base pairing
        max_neighbors: int = 32,       # Changed from 20 to 32 for consistency
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'processed', 'rna_optimized')
        if universal_cache_path is None:
            universal_cache_path = os.path.join(os.path.dirname(__file__), 'cache', 'universal_rna_all.pkl')
        super().__init__(root, universal_cache_path, max_samples, molecule_max_atoms, 
                        cutoff_distance, max_neighbors, transform, pre_transform, pre_filter)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert universal .pkl cache to PyG .pt format")
    parser.add_argument("--input-pkl", required=True, help="Path to input .pkl file")
    parser.add_argument("--output-dir", required=True, help="Output directory for .pt file")
    parser.add_argument("--dataset-type", choices=["qm9", "lba", "pdb", "coconut", "rna"], required=True, help="Dataset type")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Distance cutoff for edges (Å)")
    parser.add_argument("--max-neighbors", type=int, default=64, help="Maximum neighbors per atom")
    parser.add_argument("--force", action="store_true", help="Force rebuild if output exists")
    
    # Hybrid edge construction (Salad-inspired)
    parser.add_argument("--use-hybrid-edges", action="store_true", help="Enable 3-tier hybrid edge construction")
    parser.add_argument("--sequence-neighbors-k", type=int, default=3, help="Tier 1: Sequence neighbors (±k)")
    parser.add_argument("--max-spatial-neighbors", type=int, default=48, help="Tier 2: Max spatial neighbors")
    parser.add_argument("--num-random-edges", type=int, default=8, help="Tier 3: Random long-range edges")
    parser.add_argument("--random-edge-min-distance", type=float, default=10.0, help="Tier 3: Min distance (Å)")

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
    dataset_classes = {
        "qm9": OptimizedUniversalQM9Dataset,
        "lba": OptimizedUniversalLBADataset,
        "pdb": OptimizedUniversalDataset,
        "coconut": OptimizedUniversalCOCONUTDataset,
        "rna": OptimizedUniversalRNADataset,
    }
    
    dataset_class = dataset_classes[args.dataset_type]
    
    # Hybrid edges only for pdb dataset type
    if args.dataset_type == "pdb" and args.use_hybrid_edges:
        dataset = dataset_class(
            root=args.output_dir,
            universal_cache_path=args.input_pkl,
            max_samples=args.max_samples,
            cutoff_distance=args.cutoff,
            max_neighbors=args.max_neighbors,
            use_hybrid_edges=True,
            sequence_neighbors_k=args.sequence_neighbors_k,
            max_spatial_neighbors=args.max_spatial_neighbors,
            num_random_edges=args.num_random_edges,
            random_edge_min_distance=args.random_edge_min_distance
        )
        print(f"✅ Created {args.dataset_type.upper()} dataset with HYBRID EDGES: {len(dataset)} samples")
    else:
        # Legacy: radius graph
        dataset = dataset_class(
            root=args.output_dir,
            universal_cache_path=args.input_pkl,
            max_samples=args.max_samples,
            cutoff_distance=args.cutoff,
            max_neighbors=args.max_neighbors
        )
        print(f"✅ Created {args.dataset_type.upper()} dataset: {len(dataset)} samples")
