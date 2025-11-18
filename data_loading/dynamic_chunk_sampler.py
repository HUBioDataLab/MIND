#!/usr/bin/env python3
"""
Dynamic Chunk-Aware Batch Sampler for Lazy Loading Datasets

This batch sampler creates batches based on total atoms (not fixed number of samples),
while maintaining chunk-aware sequential reading for optimal disk I/O.

Key Features:
- Dynamic batching: Variable samples per batch based on atom count
- Cross-modal support: Mix different dataset types (PDB, QM9, etc.) in same batch
- Proportional sampling: Automatic ratio calculation from dataset sizes
- Cache-optimized: Pre-computes and reorders batches to minimize chunk switching
- Chunk-aware reading: Sequential access for optimal disk I/O performance

Usage:
    # Single-domain training
    batch_sampler = DynamicChunkAwareBatchSampler(
        dataset,
        max_atoms_per_batch=60000,
        shuffle_chunks=True,
        shuffle_within_chunk=True
    )
    
    # Multi-domain training with cross-modal batching
    batch_sampler = DynamicChunkAwareBatchSampler(
        dataset,
        max_atoms_per_batch=40000,
        enable_cross_modal_batches=True
    )
    
    loader = DataLoader(dataset, batch_sampler=batch_sampler)
    
    for epoch in range(num_epochs):
        batch_sampler.set_epoch(epoch)
        for batch in loader:
            train_step(batch)
"""

import torch
import random
from torch.utils.data import BatchSampler, Sampler
from typing import Iterator, List
from collections import defaultdict
from data_loading.lazy_universal_dataset import LazyUniversalDataset


class DynamicChunkAwareBatchSampler(BatchSampler):
    """
    Batch sampler that creates batches based on total atoms while maintaining
    chunk-aware sequential reading.
    
    Features:
    1. Dynamic batching (variable samples per batch based on atom count)
    2. Atom-based batching (respects max_atoms_per_batch limit)
    3. Cross-modal mixing (optional, for multi-domain training)
    4. Cache-optimized batch ordering (minimizes chunk switching)
    5. Proportional sampling (automatic from dataset sizes)
    """
    
    def __init__(
        self,
        dataset,
        max_atoms_per_batch: int,
        shuffle_chunks: bool = True,
        shuffle_within_chunk: bool = True,
        subset_indices: List[int] = None,
        seed: int = 42,
        enable_cross_modal_batches: bool = False
    ):
        """
        Initialize DynamicChunkAwareBatchSampler
        
        Args:
            dataset: LazyUniversalDataset instance (or Subset wrapping one)
            max_atoms_per_batch: Maximum number of atoms per batch
            shuffle_chunks: Whether to shuffle chunk order each epoch
            shuffle_within_chunk: Whether to shuffle samples within each chunk
            subset_indices: Optional list of indices (for train/val/test splits)
            seed: Random seed for reproducibility
            enable_cross_modal_batches: If True, mix different dataset types in same batch
        """
        # Handle Subset from random_split
        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            self.subset_indices = dataset.indices
            self.original_dataset = dataset
            self.dataset = dataset.dataset
            if not isinstance(self.dataset, LazyUniversalDataset):
                raise ValueError(f"Expected LazyUniversalDataset, got {type(self.dataset)}")
        else:
            self.subset_indices = subset_indices
            self.original_dataset = dataset
            self.dataset = dataset
            if not isinstance(self.dataset, LazyUniversalDataset):
                raise ValueError(f"Expected LazyUniversalDataset, got {type(self.dataset)}")
        
        self.max_atoms_per_batch = max_atoms_per_batch
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.enable_cross_modal_batches = enable_cross_modal_batches
        self.epoch = 0
        self.seed = seed
        
        # Validate dataset attributes
        if not hasattr(self.dataset, 'chunk_pt_files'):
            raise ValueError("Dataset must be LazyUniversalDataset with chunk_pt_files")
        if not hasattr(self.dataset, 'file_ranges'):
            raise ValueError("Dataset must have file_ranges attribute")
        
        # Cache for atom counts
        self._atom_count_cache = {}
        
        # Initialize base BatchSampler
        dummy_sampler = Sampler(range(len(self)))
        super().__init__(dummy_sampler, batch_size=1, drop_last=False)
    
    def _get_atom_count(self, idx: int) -> int:
        """Get atom count for a sample (uses metadata if available, otherwise accesses dataset)"""
        if idx in self._atom_count_cache:
            return self._atom_count_cache[idx]
        
        # Try metadata first (fast, no disk access)
        if hasattr(self.dataset, 'get_sample_metadata') and hasattr(self.dataset, 'metadata_loaded') and self.dataset.metadata_loaded:
            try:
                global_idx = self.subset_indices[idx] if self.subset_indices is not None else idx
                metadata = self.dataset.get_sample_metadata(global_idx)
                if metadata and 'num_nodes' in metadata:
                    atom_count = metadata['num_nodes']
                    self._atom_count_cache[idx] = atom_count
                    return atom_count
            except:
                pass
        
        # Fallback: access dataset (slower)
        try:
            sample = self.original_dataset[idx]
            atom_count = sample.num_nodes if hasattr(sample, 'num_nodes') else sample.x.size(0) if hasattr(sample, 'x') else 500
        except:
            atom_count = 500  # Conservative estimate
        
        self._atom_count_cache[idx] = atom_count
        return atom_count
    
    def _get_chunk_for_sample(self, global_idx: int) -> int:
        """Get which chunk a sample belongs to"""
        for chunk_idx, (start_idx, end_idx, _) in enumerate(self.dataset.file_ranges):
            if start_idx <= global_idx < end_idx:
                return chunk_idx
        return -1
    
    def _get_chunk_indices(self) -> List[int]:
        """
        Get all sample indices organized by chunk.
        For cross-modal mode, collects all indices and lets proportional sampling handle mixing.
        """
        num_chunks = len(self.dataset.chunk_pt_files)
        
        # Setup subset mapping if needed
        if self.subset_indices is not None:
            subset_set = set(self.subset_indices)
            global_to_subset_pos = {global_idx: pos for pos, global_idx in enumerate(self.subset_indices)}
        else:
            subset_set = None
            global_to_subset_pos = None
        
        # CROSS-MODAL MODE: Mix dataset types
        if self.enable_cross_modal_batches:
            # Group chunks by dataset type
            chunks_by_type = {}
            for chunk_idx in range(num_chunks):
                _, _, pt_file = self.dataset.file_ranges[chunk_idx]
                
                # Get dataset type from metadata or filename
                dataset_type = 'unknown'
                if hasattr(self.dataset, 'chunk_metadata') and chunk_idx < len(self.dataset.chunk_metadata):
                    metadata = self.dataset.chunk_metadata[chunk_idx]
                    if isinstance(metadata, dict) and 'dataset_type' in metadata:
                        dataset_type = metadata['dataset_type']
                
                # Fallback: infer from filename
                if dataset_type == 'unknown':
                    filename = pt_file.lower()
                    if 'pdb' in filename or 'protein' in filename:
                        dataset_type = 'pdb'
                    elif 'qm9' in filename:
                        dataset_type = 'qm9'
                    elif 'metabolite' in filename:
                        dataset_type = 'metabolite'
                
                chunks_by_type.setdefault(dataset_type, []).append(chunk_idx)
            
            # Simplified: Collect all indices from all chunks (no windows needed)
            # Proportional sampling in __iter__() will handle mixing efficiently
            all_indices = []
            
            # Process chunks in order (shuffled if requested)
            all_chunk_indices = []
            for dataset_type in chunks_by_type:
                chunk_list = chunks_by_type[dataset_type]
                if self.shuffle_chunks:
                    g = torch.Generator()
                    g.manual_seed(self.seed + self.epoch)
                    indices_tensor = torch.tensor(chunk_list)
                    shuffled = indices_tensor[torch.randperm(len(indices_tensor), generator=g)]
                    chunk_list = shuffled.tolist()
                all_chunk_indices.extend(chunk_list)
            
            # Shuffle chunk order across types for better mixing
            if self.shuffle_chunks:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch + 1000)
                all_chunk_indices = torch.tensor(all_chunk_indices)[torch.randperm(len(all_chunk_indices), generator=g)].tolist()
            
            # Collect all samples from all chunks
            for chunk_idx in all_chunk_indices:
                start_idx, end_idx, _ = self.dataset.file_ranges[chunk_idx]
                chunk_indices = list(range(start_idx, end_idx))
                
                if subset_set is not None:
                    chunk_indices = [idx for idx in chunk_indices if idx in subset_set]
                
                if self.shuffle_within_chunk and chunk_indices:
                    random.seed(self.seed + self.epoch * 10000 + chunk_idx)
                    random.shuffle(chunk_indices)
                
                if global_to_subset_pos is not None:
                    chunk_indices = [global_to_subset_pos[idx] for idx in chunk_indices if idx in global_to_subset_pos]
                
                all_indices.extend(chunk_indices)
            
            return all_indices
        
        # STANDARD MODE: Sequential chunk processing
        else:
            if self.shuffle_chunks:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                chunk_order = torch.randperm(num_chunks, generator=g).tolist()
            else:
                chunk_order = list(range(num_chunks))
            
            indices = []
            for chunk_idx in chunk_order:
                start_idx, end_idx, _ = self.dataset.file_ranges[chunk_idx]
                chunk_indices = list(range(start_idx, end_idx))
                
                if subset_set is not None:
                    chunk_indices = [idx for idx in chunk_indices if idx in subset_set]
                
                if self.shuffle_within_chunk and chunk_indices:
                    random.seed(self.seed + self.epoch * 10000 + chunk_idx)
                    random.shuffle(chunk_indices)
                
                if global_to_subset_pos is not None:
                    chunk_indices = [global_to_subset_pos[idx] for idx in chunk_indices]
                
                indices.extend(chunk_indices)
            
            return indices
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches of sample indices for one epoch.
        
        For cross-modal mode: Uses proportional sampling and cache-optimized batch ordering.
        For standard mode: Simple sequential batching.
        """
        all_indices = self._get_chunk_indices()
        
        # STEP 1: Pre-compute all batches
        if self.enable_cross_modal_batches:
            # Proportional sampling with credit system
            indices_by_type = {}
            for idx in all_indices:
                global_idx = self.subset_indices[idx] if self.subset_indices is not None else idx
                chunk_idx = self._get_chunk_for_sample(global_idx)
                
                dataset_type = 'unknown'
                if chunk_idx >= 0 and hasattr(self.dataset, 'chunk_metadata') and chunk_idx < len(self.dataset.chunk_metadata):
                    metadata = self.dataset.chunk_metadata[chunk_idx]
                    if isinstance(metadata, dict) and 'dataset_type' in metadata:
                        dataset_type = metadata['dataset_type']
                
                indices_by_type.setdefault(dataset_type, []).append(idx)
            
            # Calculate proportions
            total_samples = sum(len(indices_by_type[dt]) for dt in indices_by_type)
            type_proportions = {dt: len(indices_by_type[dt]) / total_samples for dt in indices_by_type}
            
            # Create batches with proportional sampling (credit system)
            all_batches = []
            type_positions = {dt: 0 for dt in indices_by_type}
            type_credits = type_proportions.copy()
            dataset_types = list(indices_by_type.keys())
            
            while any(type_positions[dt] < len(indices_by_type[dt]) for dt in dataset_types):
                current_batch = []
                current_atoms = 0
                
                while True:
                    available_types = [dt for dt in dataset_types if type_positions[dt] < len(indices_by_type[dt])]
                    if not available_types:
                        break
                    
                    # Pick type with highest credit
                    selected_type = max(available_types, key=lambda dt: type_credits[dt])
                    idx = indices_by_type[selected_type][type_positions[selected_type]]
                    atom_count = self._get_atom_count(idx)
                    
                    if current_atoms + atom_count <= self.max_atoms_per_batch:
                        current_batch.append(idx)
                        current_atoms += atom_count
                        type_positions[selected_type] += 1
                        
                        # Update credits
                        type_credits[selected_type] -= 1.0
                        for dt in dataset_types:
                            if dt != selected_type:
                                type_credits[dt] += type_proportions[dt]
                    else:
                        if current_batch:
                            break
                        else:
                            # Single sample exceeds limit - add anyway
                            current_batch.append(idx)
                            type_positions[selected_type] += 1
                            break
                
                if current_batch:
                    all_batches.append(current_batch)
        
        else:
            # Standard sequential batching
            all_batches = []
            current_batch = []
            current_atoms = 0
            
            for idx in all_indices:
                atom_count = self._get_atom_count(idx)
                
                if current_atoms + atom_count > self.max_atoms_per_batch and current_batch:
                    all_batches.append(current_batch)
                    current_batch = [idx]
                    current_atoms = atom_count
                else:
                    current_batch.append(idx)
                    current_atoms += atom_count
            
            if current_batch:
                all_batches.append(current_batch)
        
        # STEP 2: Optimize batch order for cache efficiency (cross-modal only)
        if self.enable_cross_modal_batches and len(all_batches) > 1:
            # Group batches by chunk requirements
            batch_chunk_sets = []
            for batch in all_batches:
                global_indices = [self.subset_indices[idx] for idx in batch] if self.subset_indices is not None else batch
                chunks_needed = frozenset(self._get_chunk_for_sample(global_idx) for global_idx in global_indices if self._get_chunk_for_sample(global_idx) >= 0)
                batch_chunk_sets.append(chunks_needed)
            
            # Group batches by chunk sets
            chunk_set_to_batches = defaultdict(list)
            for batch_idx, chunk_set in enumerate(batch_chunk_sets):
                chunk_set_to_batches[chunk_set].append(batch_idx)
            
            # Reorder: process all batches with same chunks together
            optimized_order = []
            for chunk_set in sorted(chunk_set_to_batches.keys(), key=lambda x: min(x) if x else 0):
                optimized_order.extend(chunk_set_to_batches[chunk_set])
            
            all_batches = [all_batches[i] for i in optimized_order]
        
        # STEP 3: Yield batches
        for batch in all_batches:
            yield batch
    
    def __len__(self) -> int:
        """Estimate number of batches"""
        total_samples = len(self.subset_indices) if self.subset_indices is not None else len(self.dataset)
        avg_atoms_per_sample = 500
        return max(1, (total_samples * avg_atoms_per_sample) // self.max_atoms_per_batch)
    
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling"""
        self.epoch = epoch
        self._atom_count_cache.clear()


