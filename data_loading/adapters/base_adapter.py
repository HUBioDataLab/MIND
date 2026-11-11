#!/usr/bin/env python3
"""
Base Adapter

Abstract base class for dataset adapters that convert raw data to universal format.
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Any, Optional, Sequence, Tuple, TYPE_CHECKING
import json
import os
import pickle
from tqdm import tqdm

if TYPE_CHECKING:
    from data_types import UniversalMolecule

MAX_REPORTED_SKIPS = 10000


def report_skipped_items(
    adapter_name: str,
    seen: int,
    skipped: Sequence[Tuple[str, str]],
    cache_path: Optional[str],
) -> None:
    """Make dropped items visible: print counts, write a report next to the cache, then fail.

    ``skipped`` holds ``(item label, reason)``. Nothing is raised when it is empty. A data
    tool may drop a bad item, but never silently: the run ends with a non-zero exit.
    """
    if not skipped:
        return
    by_reason = dict(Counter(reason for _, reason in skipped).most_common())
    report_path = None
    if cache_path:
        report_path = f"{cache_path}.skipped.json"
        with open(report_path, "w") as handle:
            json.dump(
                {
                    "adapter": adapter_name,
                    "seen": seen,
                    "skipped": len(skipped),
                    "by_reason": by_reason,
                    "items": [{"item": item, "reason": reason} for item, reason in skipped[:MAX_REPORTED_SKIPS]],
                },
                handle,
                indent=2,
            )
    shown = "; ".join(f"{count} x {reason}" for reason, count in list(by_reason.items())[:5])
    first_item, first_reason = skipped[0]
    raise RuntimeError(
        f"{adapter_name}: skipped {len(skipped)} of {seen} items ({shown}). "
        f"First: {first_item}: {first_reason}. "
        + (f"Report: {report_path}. " if report_path else "")
        + "Fix the inputs and rebuild; the cache written so far lacks these items."
    )


def _item_label(raw_item: Any) -> str:
    if isinstance(raw_item, dict):
        return str(raw_item.get("id") or raw_item.get("smi") or "<dict item>")[:200]
    return str(raw_item)[:200]


class BaseAdapter(ABC):
    """Abstract base for dataset adapters - converts raw data to universal blocks"""
    
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
    
    @abstractmethod
    def load_raw_data(self, data_path: str, max_samples: int = None, **kwargs) -> List[Any]:
        """Load raw data from source (QM9, PDB, LBA, RNA etc.)"""
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
        # This loop processes one raw item at a time (e.g., one PDB file path).
        # Dropped items are recorded in self.skipped_items (label, reason) and reported by
        # process_dataset; nothing is dropped without a record.
        self.skipped_items = []
        for item in tqdm(raw_data_items, desc="Processing raw samples"):
            try:
                universal_item = self.convert_to_universal(item)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"[:200]
                print(f"⚠️ Skipping item {_item_label(item)}: {reason}")
                self.skipped_items.append((_item_label(item), reason))
                continue
            # An item without blocks is unreadable or empty (the protein/RNA adapters
            # return [] for a file that cannot be parsed).
            if len(universal_item.blocks) == 0:
                self.skipped_items.append((_item_label(item), "no blocks (unreadable or empty structure)"))
                continue
            yield universal_item
    
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
            report_skipped_items(
                self.dataset_type, len(raw_data_items), self.skipped_items, cache_path
            )
        else:
            # If no cache path is provided, we can't save. We'll just count.
            # This case is less common.
            print("No cache path provided. Counting processed items without saving.")
            for _ in processed_item_generator:
                count += 1
            report_skipped_items(self.dataset_type, len(raw_data_items), self.skipped_items, None)

        return count
