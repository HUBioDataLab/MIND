#!/usr/bin/env python3
"""
Rotational Cache Sampler for Multi-Domain Training

This sampler implements atom-aware rotational sampling for mixed-domain batching.
It solves the challenge of training on heterogeneous molecular datasets (proteins,
small molecules, metabolites, RNA, DNA) while maintaining:
- Cross-modal batches (proteins + small molecules in the same batch)
- Efficient disk I/O (chunk-aware sequential access)
- Memory efficiency (respects LazyDataset's LRU cache limit)
- Balanced atom distribution (prevents gradient dominance)

Key Innovation:
Instead of loading all chunks, the sampler creates "rotations" where each rotation
contains a subset of chunks from different dataset types (e.g., 3 protein chunks +
1 molecule chunk). Within each rotation, samples are shuffled to create mixed batches,
then chunks are processed sequentially to maintain high cache hit rates.

Example Rotation Schedule:
    Rotation 0: [pdb_chunk_0, pdb_chunk_1, qm9_chunk_0, metabolite_chunk_0]
                → Samples shuffled → Mixed batches
                → Sequential chunk access → LRU cache efficient
    
    Rotation 1: [pdb_chunk_2, pdb_chunk_3, qm9_chunk_1, metabolite_chunk_1]
                → Repeat
    
    ...

Atom-Aware Balancing:
Rotations are balanced by total atom count, not sample count. This ensures:
- Consistent GPU memory usage across rotations
- Balanced gradient contribution from each dataset type
- No dominance by large proteins over small molecules

Usage:
    from rotational_cache_sampler import RotationalCacheSampler
    
    dataset = LazyUniversalDataset(..., load_metadata=True)
    
    sampler = RotationalCacheSampler(
        dataset,
        config={
            'datasets': {
                'pdb': {'enabled': True, 'atom_ratio': 70},
                'qm9': {'enabled': True, 'atom_ratio': 20},
                'metabolite': {'enabled': True, 'atom_ratio': 10}
            },
            'target_atoms_per_rotation': 50000,
            'max_cache_chunks': 10
        },
        shuffle=True
    )
    
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            train_step(batch)  # Batch contains mixed molecule types!
"""

import torch
import random
import warnings
from torch.utils.data import Sampler
from typing import Iterator, List, Dict, Any
from collections import defaultdict


class RotationalCacheSampler(Sampler):
    """
    Atom-aware rotational sampler for multi-domain molecular training.
    
    This sampler creates rotations of chunks from different dataset types,
    ensuring mixed batches while maintaining efficient disk I/O and memory usage.
    """
    
    def __init__(
        self,
        dataset,
        config: Dict[str, Any],
        shuffle: bool = True
    ):
        """
        Initialize RotationalCacheSampler
        
        Args:
            dataset: LazyUniversalDataset instance (or Subset wrapping one)
                     Must have metadata loaded (load_metadata=True)
            config: Multi-domain configuration dict with:
                - datasets: Dict of dataset configs (enabled, atom_ratio)
                - target_atoms_per_rotation: Target atom count per rotation
                - max_cache_chunks: Maximum chunks in cache (for validation)
            shuffle: Whether to shuffle rotation order and samples within rotations
        
        Raises:
            ValueError: If metadata is not loaded or config is invalid
        """
        # Handle Subset from random_split
        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            self.subset_indices = dataset.indices
            self.dataset = dataset.dataset
        else:
            self.subset_indices = None
            self.dataset = dataset
        
        self.config = config
        self.shuffle = shuffle
        self.epoch = 0
        
        # Validate dataset
        if not hasattr(self.dataset, 'metadata_loaded'):
            raise ValueError("Dataset must be LazyUniversalDataset with metadata support")
        
        if not self.dataset.metadata_loaded:
            raise ValueError(
                "Metadata must be loaded! Initialize dataset with load_metadata=True. "
                "Run create_chunk_metadata.py to generate metadata files."
            )
        
        # Extract enabled dataset types and their ratios
        self.enabled_types = {}
        total_ratio = 0
        
        for dtype, dtype_config in config['datasets'].items():
            if dtype_config.get('enabled', False):
                atom_ratio = dtype_config.get('atom_ratio', 0)
                if atom_ratio > 0:
                    self.enabled_types[dtype] = atom_ratio
                    total_ratio += atom_ratio
        
        if not self.enabled_types:
            raise ValueError("At least one dataset type must be enabled with atom_ratio > 0")
        
        # Normalize ratios to sum to 1.0
        self.normalized_ratios = {
            dtype: ratio / total_ratio 
            for dtype, ratio in self.enabled_types.items()
        }
        
        # Get rotation parameters
        self.target_atoms_per_rotation = config.get('target_atoms_per_rotation', 50000)
        self.max_cache_chunks = config.get('max_cache_chunks', 10)
        
        # Build chunk groups by dataset type
        self._build_chunk_groups()
        
        # Create atom-balanced rotations
        self._create_atom_based_rotations()
        
        # Print summary
        if self.dataset.verbose:
            self._print_summary()
    
    def _build_chunk_groups(self):
        """
        Group chunks by dataset type and collect metadata.
        
        Creates:
            self.chunks_by_type: Dict[str, List[Dict]] containing chunk info
        """
        self.chunks_by_type = defaultdict(list)
        
        for chunk_idx, chunk_meta in enumerate(self.dataset.chunk_metadata):
            dtype = chunk_meta['dataset_type']
            
            # Only include enabled types
            if dtype not in self.enabled_types:
                continue
            
            # Get global sample indices for this chunk
            start_idx, end_idx, _ = self.dataset.file_ranges[chunk_idx]
            chunk_indices = list(range(start_idx, end_idx))
            
            # Filter by subset if needed (for train/val/test splits)
            if self.subset_indices is not None:
                subset_set = set(self.subset_indices)
                chunk_indices = [idx for idx in chunk_indices if idx in subset_set]
            
            # Skip chunks with no samples after subset filtering
            if len(chunk_indices) == 0:
                continue
            
            self.chunks_by_type[dtype].append({
                'chunk_idx': chunk_idx,
                'total_atoms': chunk_meta['total_atoms'],
                'num_samples': len(chunk_indices),
                'indices': chunk_indices
            })
    
    def _create_atom_based_rotations(self):
        """
        Create rotations balanced by atom count.
        
        Each rotation contains chunks from different dataset types, where the
        total atom count from each type matches the target ratio.
        
        Algorithm:
        1. Calculate target atoms per type based on ratios
        2. For each rotation, greedily select chunks until atom budget is met
        3. Continue until all chunks are assigned
        4. Handle edge cases (one type runs out before others)
        """
        self.rotations = []
        
        # Track remaining chunks for each type
        remaining_chunks = {
            dtype: list(chunks)  # Create a copy
            for dtype, chunks in self.chunks_by_type.items()
        }
        
        # Create rotations until all chunks are consumed
        while any(remaining_chunks.values()):
            rotation = {}
            
            # Calculate active types (types that still have chunks)
            active_types = {
                dtype: chunks 
                for dtype, chunks in remaining_chunks.items() 
                if len(chunks) > 0
            }
            
            if not active_types:
                break
            
            # Recalculate ratios for active types (dynamic ratio adjustment)
            total_active_ratio = sum(
                self.normalized_ratios[dtype] 
                for dtype in active_types.keys()
            )
            
            active_ratios = {
                dtype: self.normalized_ratios[dtype] / total_active_ratio
                for dtype in active_types.keys()
            }
            
            # Allocate chunks to this rotation
            for dtype, chunks in active_types.items():
                target_atoms = self.target_atoms_per_rotation * active_ratios[dtype]
                
                # Greedily select chunks until target is met
                selected_chunks = []
                total_atoms = 0
                chunk_ids = []
                
                for chunk in chunks:
                    if total_atoms >= target_atoms:
                        break
                    
                    selected_chunks.append(chunk)
                    chunk_ids.append(chunk['chunk_idx'])
                    total_atoms += chunk['total_atoms']
                
                # Must have at least one chunk per active type
                if not selected_chunks and chunks:
                    selected_chunks = [chunks[0]]
                    chunk_ids = [chunks[0]['chunk_idx']]
                    total_atoms = chunks[0]['total_atoms']
                
                # Remove selected chunks from remaining
                remaining_chunks[dtype] = chunks[len(selected_chunks):]
                
                # Store in rotation
                rotation[dtype] = {
                    'chunk_ids': chunk_ids,
                    'chunks': selected_chunks,
                    'total_atoms': total_atoms
                }
            
            self.rotations.append(rotation)
        
        # Validate rotations
        if not self.rotations:
            raise RuntimeError("Failed to create any rotations! Check your data and configuration.")
        
        # Validate cache capacity
        max_chunks_in_rotation = max(
            sum(len(data['chunk_ids']) for data in rotation.values())
            for rotation in self.rotations
        )
        
        if max_chunks_in_rotation > self.max_cache_chunks:
            warnings.warn(
                f"Rotation requires {max_chunks_in_rotation} chunks but cache limit is "
                f"{self.max_cache_chunks}. This may cause cache thrashing. "
                f"Consider increasing max_cache_chunks or target_atoms_per_rotation."
            )
    
    def __iter__(self) -> Iterator[int]:
        """
        Generate sample indices for one epoch.
        
        Returns:
            Iterator of sample indices with mixed molecule types
        """
        # Shuffle rotation order
        rotation_order = list(range(len(self.rotations)))
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(rotation_order)
        
        # Process each rotation
        for rot_idx in rotation_order:
            rotation = self.rotations[rot_idx]
            
            # Collect all sample indices from this rotation
            rotation_samples = []
            
            for dtype, data in rotation.items():
                for chunk_info in data['chunks']:
                    rotation_samples.extend(chunk_info['indices'])
            
            # CRITICAL: Shuffle samples within rotation for cross-modal mixing!
            # This ensures proteins and small molecules are interleaved in batches
            if self.shuffle:
                random.seed(self.epoch * 10000 + rot_idx)
                random.shuffle(rotation_samples)
            
            # Yield samples in shuffled order
            # LazyDataset's LRU cache will automatically load/evict chunks as needed
            for idx in rotation_samples:
                yield idx
    
    def __len__(self) -> int:
        """Return total number of samples"""
        if self.subset_indices is not None:
            return len(self.subset_indices)
        return len(self.dataset)
    
    def set_epoch(self, epoch: int):
        """
        Set the epoch number for shuffling.
        
        This should be called at the beginning of each epoch to ensure
        different rotation orders and sample orders across epochs.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch
    
    def _print_summary(self):
        """Print summary of rotation configuration"""
        print("\n" + "=" * 60)
        print("🔄 RotationalCacheSampler Configuration")
        print("=" * 60)
        
        # Dataset types
        print(f"\n📊 Enabled Dataset Types:")
        for dtype, ratio in self.normalized_ratios.items():
            num_chunks = len(self.chunks_by_type[dtype])
            total_atoms = sum(c['total_atoms'] for c in self.chunks_by_type[dtype])
            print(f"   {dtype:12s}: {ratio*100:5.1f}% | {num_chunks:3d} chunks | {total_atoms:,} atoms")
        
        # Rotation stats
        print(f"\n🔄 Rotation Statistics:")
        print(f"   Total rotations: {len(self.rotations)}")
        print(f"   Target atoms/rotation: {self.target_atoms_per_rotation:,}")
        print(f"   Max cache chunks: {self.max_cache_chunks}")
        
        # Rotation breakdown
        print(f"\n📦 First 3 Rotations (example):")
        for i, rotation in enumerate(self.rotations[:3]):
            total_atoms = sum(data['total_atoms'] for data in rotation.values())
            total_chunks = sum(len(data['chunk_ids']) for data in rotation.values())
            
            print(f"   Rotation {i}: {total_chunks} chunks | {total_atoms:,} atoms")
            for dtype, data in rotation.items():
                print(f"      {dtype}: {len(data['chunk_ids'])} chunks, {data['total_atoms']:,} atoms")
        
        if len(self.rotations) > 3:
            print(f"   ... and {len(self.rotations) - 3} more rotations")
        
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Simple test/demo
    print("RotationalCacheSampler - Atom-Aware Multi-Domain Sampling")
    print("\nThis sampler enables training on mixed molecular datasets while")
    print("maintaining efficient disk I/O and balanced atom distribution.")
    print("\nFor usage examples, see the docstring at the top of this file.")

