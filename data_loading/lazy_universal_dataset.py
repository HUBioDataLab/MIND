#!/usr/bin/env python3
"""
Lazy-Loading Universal Dataset

Memory-efficient dataset that loads PyG data on-demand from disk.
Ideal for large-scale datasets (1M+ samples) that don't fit in RAM.

Usage:
    from lazy_universal_dataset import LazyUniversalDataset
    
    dataset = LazyUniversalDataset(
        chunk_pt_files=[
            'processed/chunk_0/processed/data.pt',
            'processed/chunk_1/processed/data.pt',
            ...
        ],
        transform=my_transform
    )
    
    # Works with PyTorch DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
"""

import torch
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from tqdm import tqdm


class LazyUniversalDataset(Dataset):
    """
    Lazy-loading dataset for PyG data stored in multiple .pt files.
    
    Key Features:
    - Memory-efficient: Only loads chunks on-demand
    - LRU caching: Keeps most recently used chunks in memory
    - Compatible with PyTorch DataLoader and random_split
    - Supports transforms
    
    RAM Usage:
    - Metadata: ~10KB per chunk
    - Cache: max_cache_chunks × chunk_size × 20KB (default: ~1.5GB)
    - Total: ~2GB for 1M proteins (vs ~50GB with InMemoryDataset)
    """
    
    def __init__(
        self,
        chunk_pt_files: List[str],
        transform: Optional[Callable] = None,
        max_cache_chunks: int = 3,
        verbose: bool = True,
        load_metadata: bool = False,
        metadata_search_paths: Optional[List[str]] = None
    ):
        """
        Initialize LazyUniversalDataset
        
        Args:
            chunk_pt_files: List of paths to processed .pt files
            transform: Optional transform to apply to each sample
            max_cache_chunks: Maximum number of chunks to keep in RAM (default: 3)
            verbose: Print loading information
            load_metadata: If True, load metadata for atom-aware sampling (default: False)
        """
        # Filter out PyG metadata files (pre_transform.pt, pre_filter.pt)
        self.chunk_pt_files = [
            str(Path(f).resolve()) for f in chunk_pt_files 
            if not Path(f).name.startswith('pre_')
        ]
        self.transform = transform
        self.max_cache_chunks = max_cache_chunks
        self.verbose = verbose
        self.load_metadata = load_metadata
        self.metadata_search_paths = metadata_search_paths or []
        
        # LRU cache for loaded chunks
        self._cache = OrderedDict()
        
        # Metadata storage
        self.metadata_loaded = False
        self.chunk_metadata = []  # List of metadata dicts (one per chunk)
        self.sample_metadata_index = []  # Global index -> metadata mapping
        
        # Build index map: which sample is in which file?
        self._build_index_map()
        
        # Load metadata if requested
        if self.load_metadata:
            self._load_metadata()
        
        if self.verbose:
            print(f"✅ LazyUniversalDataset initialized:")
            print(f"   Total chunks: {len(self.chunk_pt_files)}")
            print(f"   Total samples: {self.total_samples:,}")
            print(f"   Cache size: {self.max_cache_chunks} chunks")
            print(f"   Estimated cache RAM: ~{self._estimate_cache_ram():.0f}MB")
            if self.load_metadata:
                print(f"   Metadata loaded: {'✅ Yes' if self.metadata_loaded else '⚠️ No (fallback mode)'}")
    
    def _build_index_map(self):
        """Build index map: (global_idx -> (file_idx, local_idx, pt_file))"""
        self.file_ranges = []
        self.samples_per_file = []
        cumsum = 0
        
        desc = "Building index map" if self.verbose else None
        for pt_file in tqdm(self.chunk_pt_files, desc=desc, disable=not self.verbose):
            if not Path(pt_file).exists():
                raise FileNotFoundError(f"Chunk file not found: {pt_file}")
            
            # Load only slices to count samples (minimal RAM)
            try:
                _, slices = torch.load(pt_file, map_location='cpu', weights_only=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load {pt_file}: {e}")
            
            # Determine number of samples
            # Use 'pos' as reference (every sample must have positions)
            if 'pos' not in slices:
                raise ValueError(f"Invalid .pt file (no 'pos' in slices): {pt_file}")
            
            n_samples = len(slices['pos']) - 1
            self.samples_per_file.append(n_samples)
            self.file_ranges.append((cumsum, cumsum + n_samples, pt_file))
            cumsum += n_samples
            
            # Free memory immediately
            del slices
        
        self.total_samples = cumsum
        
        if self.total_samples == 0:
            raise ValueError("No samples found in any chunk files!")
    
    def _estimate_cache_ram(self):
        """Estimate RAM usage for cache (in MB)"""
        # Rough estimate: 20KB per protein sample
        avg_samples_per_chunk = sum(self.samples_per_file) / len(self.samples_per_file)
        ram_per_chunk_mb = avg_samples_per_chunk * 20 / 1024  # KB to MB
        total_cache_ram = ram_per_chunk_mb * self.max_cache_chunks
        return total_cache_ram
    
    def __len__(self):
        """Return total number of samples"""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a single sample by index.
        
        This method:
        1. Finds which chunk file contains the sample
        2. Loads the chunk (or retrieves from cache)
        3. Extracts the specific sample using PyG's separate() function
        4. Applies transform if specified
        
        Args:
            idx: Global sample index
            
        Returns:
            PyG Data object
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples})")
        
        # Find which file contains this sample
        for chunk_idx, (start_idx, end_idx, pt_file) in enumerate(self.file_ranges):
            if start_idx <= idx < end_idx:
                local_idx = idx - start_idx
                
                # Load chunk if not in cache
                if pt_file not in self._cache:
                    self._load_chunk(pt_file)
                else:
                    # Move to end (most recently used)
                    self._cache.move_to_end(pt_file)
                
                # Extract sample
                data, slices = self._cache[pt_file]
                sample = self._extract_sample(data, slices, local_idx)
                
                # Add dataset_type for batch statistics
                if self.metadata_loaded and chunk_idx < len(self.chunk_metadata):
                    sample.dataset_type = self.chunk_metadata[chunk_idx].get('dataset_type', 'unknown')
                else:
                    # Fallback: infer from filename
                    filename = pt_file.lower()
                    if 'pdb' in filename or 'protein' in filename:
                        sample.dataset_type = 'protein'
                    elif 'qm9' in filename:
                        sample.dataset_type = 'qm9'
                    elif 'metabolite' in filename:
                        sample.dataset_type = 'metabolite'
                    elif 'rna' in filename:
                        sample.dataset_type = 'rna'
                    else:
                        sample.dataset_type = 'unknown'
                
                # Apply transform
                if self.transform is not None:
                    sample = self.transform(sample)
                
                return sample
        
        # Should never reach here
        raise IndexError(f"Could not locate index {idx}")
    
    def _load_chunk(self, pt_file: str):
        """Load a chunk file into cache"""
        # Evict oldest chunk if cache is full
        if len(self._cache) >= self.max_cache_chunks:
            oldest_file = next(iter(self._cache))
            del self._cache[oldest_file]
        
        # Load new chunk
        data, slices = torch.load(pt_file, map_location='cpu', weights_only=False)
        self._cache[pt_file] = (data, slices)
    
    def _extract_sample(self, data: Data, slices: dict, idx: int) -> Data:
        """
        Extract a single sample from collated data.
        
        This uses PyG's separate() function which handles:
        - Tensor slicing based on slices dictionary
        - Edge index offsetting
        - Heterogeneous data structures
        
        Args:
            data: Collated PyG Data object
            slices: Slices dictionary
            idx: Local index within this chunk
            
        Returns:
            Individual PyG Data object
        """
        # Use PyG's built-in separate function
        sample = separate(
            cls=data.__class__,
            batch=data,
            idx=idx,
            slice_dict=slices,
            inc_dict=None,
            decrement=False
        )
        
        return sample
    
    def get_chunk_stats(self):
        """Get statistics about chunks"""
        return {
            'num_chunks': len(self.chunk_pt_files),
            'total_samples': self.total_samples,
            'samples_per_chunk': self.samples_per_file,
            'avg_samples_per_chunk': sum(self.samples_per_file) / len(self.samples_per_file),
            'cache_size': self.max_cache_chunks,
            'cached_chunks': list(self._cache.keys()),
        }
    
    def clear_cache(self):
        """Clear the chunk cache to free memory"""
        self._cache.clear()
    
    def _load_metadata(self):
        """
        Load metadata from JSON files for all chunks.
        
        Metadata files are expected to be in the same directory as .pt files
        with naming convention: <chunk_name>_metadata.json
        
        If metadata is missing for any chunk, gracefully falls back to no metadata mode.
        """
        if self.verbose:
            print("🔄 Loading chunk metadata...")
        
        missing_metadata = []
        
        for chunk_idx, pt_file in enumerate(self.chunk_pt_files):
            pt_path = Path(pt_file)
            metadata_path = pt_path.parent / f"{pt_path.stem}_metadata.json"
            
            # If not found in same directory, check alternative search paths
            if not metadata_path.exists():
                found = False
                for search_path in self.metadata_search_paths:
                    alt_path = Path(search_path) / f"{pt_path.stem}_metadata.json"
                    if alt_path.exists():
                        metadata_path = alt_path
                        found = True
                        break
                
                if not found:
                    missing_metadata.append(pt_path.name)
                    continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.chunk_metadata.append(metadata)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to load metadata for {pt_path.name}: {e}")
                missing_metadata.append(pt_path.name)
        
        # Check if all metadata loaded successfully
        if len(self.chunk_metadata) == len(self.chunk_pt_files):
            self.metadata_loaded = True
            self._build_metadata_index()
            
            if self.verbose:
                total_atoms = sum(m['total_atoms'] for m in self.chunk_metadata)
                dataset_types = set(m['dataset_type'] for m in self.chunk_metadata)
                print(f"✅ Metadata loaded for {len(self.chunk_metadata)} chunks")
                print(f"   Total atoms: {total_atoms:,}")
                print(f"   Dataset types: {', '.join(sorted(dataset_types))}")
        else:
            # Graceful fallback
            self.metadata_loaded = False
            self.chunk_metadata = []
            self.sample_metadata_index = []
            
            if self.verbose:
                print(f"⚠️  Metadata missing for {len(missing_metadata)} chunks:")
                for name in missing_metadata[:5]:
                    print(f"     - {name}")
                if len(missing_metadata) > 5:
                    print(f"     ... and {len(missing_metadata) - 5} more")
                print(f"💡 Tip: Run create_chunk_metadata.py to generate metadata")
                print(f"🔄 Falling back to standard mode (metadata-free)")
    
    def _build_metadata_index(self):
        """
        Build global sample index -> metadata mapping.
        
        This enables O(1) metadata lookup by global sample index.
        """
        self.sample_metadata_index = []
        
        for chunk_idx, metadata in enumerate(self.chunk_metadata):
            for sample_meta in metadata['samples']:
                self.sample_metadata_index.append({
                    'chunk_idx': chunk_idx,
                    'local_idx': sample_meta['idx'],
                    'num_atoms': sample_meta['num_atoms'],
                    'num_edges': sample_meta['num_edges'],
                    'bucket': sample_meta['bucket'],
                    'dataset_type': metadata['dataset_type']
                })
    
    def get_sample_metadata(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a sample without loading the chunk.
        
        Args:
            idx: Global sample index
            
        Returns:
            Dictionary with metadata or None if metadata not loaded
        """
        if not self.metadata_loaded:
            return None
        
        if idx < 0 or idx >= len(self.sample_metadata_index):
            raise IndexError(f"Index {idx} out of range [0, {len(self.sample_metadata_index)})")
        
        return self.sample_metadata_index[idx]
    
    def get_chunk_metadata(self, chunk_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an entire chunk.
        
        Args:
            chunk_idx: Chunk index (0 to num_chunks-1)
            
        Returns:
            Dictionary with chunk metadata or None if metadata not loaded
        """
        if not self.metadata_loaded:
            return None
        
        if chunk_idx < 0 or chunk_idx >= len(self.chunk_metadata):
            raise IndexError(f"Chunk index {chunk_idx} out of range [0, {len(self.chunk_metadata)})")
        
        return self.chunk_metadata[chunk_idx]
    
    def get_chunks_by_type(self, dataset_type: str) -> List[int]:
        """
        Get all chunk indices for a specific dataset type.
        
        Args:
            dataset_type: Dataset type (e.g., 'pdb', 'qm9', 'metabolite')
            
        Returns:
            List of chunk indices
        """
        if not self.metadata_loaded:
            return []
        
        return [
            i for i, meta in enumerate(self.chunk_metadata)
            if meta['dataset_type'] == dataset_type
        ]
    
    def get_chunk_indices(self, chunk_idx: int) -> List[int]:
        """
        Get all global sample indices for a specific chunk.
        
        Used by ChunkAwareSampler to enable sequential chunk reading.
        
        Args:
            chunk_idx: Index of the chunk (0 to num_chunks-1)
            
        Returns:
            List of global sample indices in this chunk
        """
        if chunk_idx < 0 or chunk_idx >= len(self.chunk_pt_files):
            raise IndexError(f"Chunk index {chunk_idx} out of range [0, {len(self.chunk_pt_files)})")
        
        start_idx, end_idx, _ = self.file_ranges[chunk_idx]
        return list(range(start_idx, end_idx))
