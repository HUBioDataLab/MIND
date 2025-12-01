#!/usr/bin/env python3
"""
Chunk-Aware Sampler for Lazy Loading Datasets

This sampler optimizes disk I/O by reading chunks sequentially while maintaining
sufficient randomness for training.

Key Features:
- Shuffles chunk order at each epoch (epoch-level randomness)
- Optionally shuffles samples within each chunk (sample-level randomness)
- Sequential chunk reading → optimal disk I/O performance
- Predictable access pattern → enables prefetching

Usage:
    from chunk_sampler import ChunkAwareSampler
    
    dataset = LazyUniversalDataset(...)
    sampler = ChunkAwareSampler(
        dataset,
        shuffle_chunks=True,       # Shuffle chunk order each epoch
        shuffle_within_chunk=True  # Also shuffle within chunks
    )
    
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Update shuffle seed
        for batch in loader:
            train_step(batch)
"""

import torch
import random
from torch.utils.data import Sampler
from typing import Iterator


class ChunkAwareSampler(Sampler):
    """
    Sampler that reads chunks sequentially but shuffles chunk order.
    
    How it works:
    - At each epoch, shuffle the order of chunks
    - Within each chunk, optionally shuffle samples
    - Read chunks one by one in the shuffled order
    - This ensures each chunk is loaded exactly once per epoch
    
    Works with both:
    - Full LazyUniversalDataset
    - Subset from random_split (extracts underlying dataset automatically)
    
    Example:
        Epoch 0: [chunk_3, chunk_7, chunk_0, ...]
        Epoch 1: [chunk_1, chunk_4, chunk_9, ...]
        
        Within chunk_3: [sample_12000, sample_12543, ..., sample_15999]
                        (shuffled if shuffle_within_chunk=True)
    """
    
    def __init__(
        self,
        dataset,
        shuffle_chunks: bool = True,
        shuffle_within_chunk: bool = True,
        subset_indices: list = None
    ):
        """
        Initialize ChunkAwareSampler
        
        Args:
            dataset: LazyUniversalDataset instance (or Subset wrapping one)
            shuffle_chunks: Whether to shuffle chunk order each epoch
            shuffle_within_chunk: Whether to shuffle samples within each chunk
            subset_indices: Optional list of indices to sample from (for train/val/test splits)
        """
        # Handle Subset from random_split
        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            self.subset_indices = dataset.indices
            self.dataset = dataset.dataset  # Get underlying LazyUniversalDataset
        else:
            self.subset_indices = subset_indices
            self.dataset = dataset
        
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.epoch = 0
        
        # Validate dataset has required attributes
        if not hasattr(self.dataset, 'chunk_pt_files'):
            raise ValueError("Dataset must be a LazyUniversalDataset with chunk_pt_files")
        if not hasattr(self.dataset, 'file_ranges'):
            raise ValueError("Dataset must have file_ranges attribute")
    
    def __iter__(self) -> Iterator[int]:
        """
        Generate sample indices for one epoch.
        
        Returns:
            Iterator of sample indices (subset-aware if using train/val/test split)
        """
        # Create chunk order
        num_chunks = len(self.dataset.chunk_pt_files)
        
        if self.shuffle_chunks:
            # Shuffle chunk order using epoch-based seed
            g = torch.Generator()
            g.manual_seed(self.epoch)
            chunk_order = torch.randperm(num_chunks, generator=g).tolist()
        else:
            # Sequential chunk order
            chunk_order = list(range(num_chunks))
        
        # Collect indices chunk by chunk
        indices = []
        
        # If using Subset, create reverse mapping: global_idx -> subset_position
        if self.subset_indices is not None:
            subset_set = set(self.subset_indices)
            # Map global index to position in subset.indices list
            global_to_subset_pos = {global_idx: pos for pos, global_idx in enumerate(self.subset_indices)}
        
        for chunk_idx in chunk_order:
            # Get the file range for this chunk
            start_idx, end_idx, pt_file = self.dataset.file_ranges[chunk_idx]
            
            # Get all sample indices in this chunk
            chunk_indices = list(range(start_idx, end_idx))
            
            # Filter by subset if needed (for train/val/test splits)
            if self.subset_indices is not None:
                # Keep only indices that are in the subset
                chunk_indices = [idx for idx in chunk_indices if idx in subset_set]
            
            # Optionally shuffle within chunk
            if self.shuffle_within_chunk and len(chunk_indices) > 0:
                # Use different seed for each chunk to ensure diversity
                random.seed(self.epoch * 10000 + chunk_idx)
                random.shuffle(chunk_indices)
            
            # Convert to subset positions if needed
            if self.subset_indices is not None:
                # Convert global indices to subset positions
                chunk_indices = [global_to_subset_pos[idx] for idx in chunk_indices]
            
            # Add to total indices
            indices.extend(chunk_indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        """Return total number of samples"""
        if self.subset_indices is not None:
            return len(self.subset_indices)
        return len(self.dataset)
    
    def set_epoch(self, epoch: int):
        """
        Set the epoch number for shuffling.
        
        This should be called at the beginning of each epoch to ensure
        different chunk orders and sample orders across epochs.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch
