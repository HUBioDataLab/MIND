#!/usr/bin/env python3
"""
Base Adapter

Abstract base class for dataset adapters that convert raw data to universal format.
"""

from abc import ABC, abstractmethod
from typing import List, Any, TYPE_CHECKING
import os
import pickle
from tqdm import tqdm

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

    def _data_generator(self, raw_data_items: List[Any]):
        """
        A memory-efficient generator that processes raw data items one by one.
        It yields processed UniversalMolecule objects without storing them all in a list.
        """
        # This loop processes one raw item at a time (e.g., one PDB file path)
        for item in tqdm(raw_data_items, desc="Processing raw samples"):
            try:
                # Convert the raw item to the universal format
                universal_item = self.convert_to_universal(item)
                # Ensure the processed item is valid (contains atoms) before yielding
                if len(universal_item.blocks) > 0:
                    yield universal_item
            except Exception as e:
                # Silently skip any file that fails to process and continue with the next one.
                print(f"⚠️ Skipping item due to error: {e}")
                pass
    
    def process_dataset(self, data_path: str, cache_path: str = None, **kwargs) -> int:
        """
        Complete and memory-efficient processing pipeline with universal representation caching.
        This version processes data as a stream and writes to the cache file iteratively.
        It returns the count of successfully processed items.
        """
        # 1. Load raw data references (e.g., file paths). This is memory-efficient.
        print(f"🔄 Staging {self.dataset_type} dataset for processing...")
        raw_data_items = self.load_raw_data(data_path, **kwargs)
        print(f"Found {len(raw_data_items)} raw data items.")

        # 2. Create a memory-efficient generator for processing.
        # No large lists are created in memory here.
        processed_item_generator = self._data_generator(raw_data_items)
        
        # 3. Cache universal representations iteratively (memory-safe).
        count = 0
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            print(f"💾 Caching processed samples to {cache_path}...")
            
            with open(cache_path, 'wb') as f:
                # This loop pulls one processed item at a time from the generator
                # and writes it directly to the file.
                for universal_item in processed_item_generator:
                    pickle.dump(universal_item, f)
                    count += 1
            
            print(f"✅ Successfully cached {count} universal samples.")
        else:
            # If no cache path is provided, we can't save. We'll just count.
            # This case is less common.
            print("No cache path provided. Counting processed items without saving.")
            for _ in processed_item_generator:
                count += 1

        return count
