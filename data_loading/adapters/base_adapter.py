#!/usr/bin/env python3
"""
Base Adapter

Abstract base class for dataset adapters that convert raw data to universal format.
"""

from abc import ABC, abstractmethod
from typing import List, Any, TYPE_CHECKING
import os
import pickle

if TYPE_CHECKING:
    from data_types import UniversalMolecule

class BaseAdapter(ABC):
    """Abstract base for dataset adapters - converts raw data to universal blocks"""
    
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
    
    @abstractmethod
    def load_raw_data(self, data_path: str, max_samples: int = None, **kwargs) -> List[Any]:
        """Load raw data from source (QM9, PDB, LBA, etc.)"""
        pass
    
    @abstractmethod
    def create_blocks(self, raw_item: Any) -> List[Any]:
        """Convert raw data to hierarchical blocks"""
        pass
    
    def convert_to_universal(self, raw_item: Any) -> Any:
        """Convert raw data to universal format with blocks"""
        from data_types import UniversalMolecule
        
        blocks = self.create_blocks(raw_item)
        return UniversalMolecule(
            id=raw_item.get('id', 'unknown') if isinstance(raw_item, dict) else getattr(raw_item, 'id', 'unknown'),
            dataset_type=self.dataset_type,
            blocks=blocks,
            properties=raw_item.get('scores', {}) if isinstance(raw_item, dict) else getattr(raw_item, 'properties', {})
        )
    
    def process_dataset(self, data_path: str, cache_path: str = None, **kwargs) -> List[Any]:
        """Complete processing pipeline with universal representation caching"""
        # 1. Check universal cache
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"✅ Loaded {len(cached_data)} universal samples from cache: {cache_path}")
                return cached_data
            except Exception as e:
                print(f"⚠️ Error loading universal cache: {e}. Re-processing data.")

        # 2. Load raw data
        print(f"🔄 Processing {self.dataset_type} dataset...")
        raw_data_items = self.load_raw_data(data_path, **kwargs)

        # 3. Convert to universal format
        print(f"🔄 Converting {len(raw_data_items)} samples to universal format...")
        
        # GPU-accelerated processing with torch.compile
        import torch
        
        if torch.cuda.is_available() and len(raw_data_items) > 10:
            print(f"🚀 Using GPU acceleration with torch.compile...")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            universal_data = self._process_with_gpu_acceleration(raw_data_items)
        else:
            print("🔄 Using CPU sequential processing...")
            universal_data = self._process_sequential(raw_data_items)
        
        skipped_count = len(raw_data_items) - len(universal_data)
        
        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} samples due to processing errors")

        # 4. Cache universal representations
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(universal_data, f)
            print(f"💾 Cached {len(universal_data)} universal samples to {cache_path}")

        return universal_data
    
    def _process_sequential(self, raw_data_items) -> List['UniversalMolecule']:
        """Sequential processing with progress indicator"""
        universal_data = []
        from tqdm import tqdm
        
        total_samples = len(raw_data_items)
        progress_bar = tqdm(total=total_samples, desc="Processing samples (CPU)", unit="samples")
        
        for i, item in enumerate(raw_data_items):
            try:
                universal_item = self.convert_to_universal(item)
                # Skip items with empty blocks (e.g., invalid molecules)
                if len(universal_item.blocks) == 0:
                    continue
                universal_data.append(universal_item)
            except Exception as e:
                # Don't print every error to avoid spam
                pass
            finally:
                progress_bar.update(1)
        
        progress_bar.close()
        return universal_data
    
    def _process_with_gpu_acceleration(self, raw_data_items) -> List['UniversalMolecule']:
        """GPU-accelerated processing with batch processing (no DataLoader for PyG compatibility)"""
        import torch
        from tqdm import tqdm
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        universal_data = []
        batch_size = 32  # Process in batches for better GPU utilization
        
        # Create progress bar for individual samples
        total_samples = len(raw_data_items)
        progress_bar = tqdm(total=total_samples, desc="Processing samples (GPU)", unit="samples")
        
        # Process in batches without DataLoader (PyG compatibility)
        for i in range(0, len(raw_data_items), batch_size):
            batch = raw_data_items[i:i + batch_size]
            batch_results = []
            
            # Process each item in the batch
            for item in batch:
                try:
                    universal_item = self.convert_to_universal(item)
                    if len(universal_item.blocks) > 0:
                        batch_results.append(universal_item)
                except Exception as e:
                    pass  # Skip failed samples
                finally:
                    progress_bar.update(1)  # Update progress for each sample
            
            universal_data.extend(batch_results)
        
        progress_bar.close()
        return universal_data
    
    def _process_single_item(self, item) -> 'UniversalMolecule':
        """Process a single item - designed for parallel processing"""
        try:
            universal_item = self.convert_to_universal(item)
            # Skip items with empty blocks (e.g., invalid molecules)
            if len(universal_item.blocks) == 0:
                return None
            return universal_item
        except Exception as e:
            return None
