#!/usr/bin/env python3
"""
Universal Dataset Caching System

Caches universal representations for any dataset using the adapter system.
This script can cache QM9, LBA, PDB, or any future dataset adapters.

Usage:
    # Process the first 1000 samples of the QM9 dataset using default paths
    python data_loading/cache_universal_datasets.py --dataset qm9 --max_samples 1000

    # Process the custom PDB dataset from a specific folder of raw structures
    *CUDA_VISIBLE_DEVICES=0* python data_loading/cache_universal_datasets.py --dataset pdb --data-path ../data/proteins/raw_structures_hq_40k

    # Process the PDB dataset and save the output cache to a custom location
    python data_loading/cache_universal_datasets.py --dataset pdb \
      --data-path ../data/proteins/raw_structures_hq_40k \
      --cache-dir ../data/proteins/cache

    # List all caches in a specific directory
    python data_loading/cache_universal_datasets.py --list --cache-dir ../data/proteins/cache
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add project root to path to allow imports like 'data_loading.adapters'
sys.path.append('.')

def get_adapter(dataset_name: str):
    """Get adapter instance by name"""
    if dataset_name.lower() == 'qm9':
        from data_loading.adapters.qm9_adapter import QM9Adapter
        return QM9Adapter(), './data/qm9'
    elif dataset_name.lower() == 'lba':
        from data_loading.adapters.lba_adapter import LBAAdapter
        return LBAAdapter(), './data/LBA'
    elif dataset_name.lower() == 'pdb':
        from data_loading.adapters.protein_adapter import ProteinAdapter
        # This is just a default path if the user does not provide one.
        # It's better to specify the path via the command line.
        return ProteinAdapter(), '../data/proteins/raw_structures_hq' 
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_cache_path(dataset_name: str, cache_dir: Path, max_samples: int = None) -> Path:
    """Generate cache path for dataset."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename based on sample count
    if max_samples:
        cache_file = f"universal_{dataset_name}_{max_samples}.pkl"
    else:
        cache_file = f"universal_{dataset_name}_all.pkl"
    
    return cache_dir / cache_file

def cache_dataset(dataset_name: str, data_path: Path, cache_dir: Path, max_samples: int = None, force_rebuild: bool = False):
    """Cache universal representations for a dataset."""
    print(f"🚀 Caching {dataset_name.upper()} dataset...")
    print("=" * 60)
    
    adapter, default_data_path = get_adapter(dataset_name)
    
    # If a data_path isn't provided via CLI, use the default from the adapter
    if data_path is None:
        data_path = Path(default_data_path)

    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        print(f"💡 Please provide the correct path using the --data-path argument.")
        return False
    
    cache_path = get_cache_path(dataset_name, cache_dir, max_samples)
    
    if cache_path.exists() and not force_rebuild:
        print(f"✅ Cache already exists: {cache_path}")
        print(f"💡 Use --force to rebuild the cache.")
        return True
    
    start_time = time.time()
    try:
        # The BaseAdapter's process_dataset method is called here
        universal_data = adapter.process_dataset(
            data_path=str(data_path),
            cache_path=str(cache_path),
            max_samples=max_samples
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Successfully cached {len(universal_data)} universal samples.")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds.")
        print(f"💾 Cache file: {cache_path}")
        
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"📊 Cache size: {cache_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def list_cached_datasets(cache_dir: Path):
    """List all cached datasets in the specified directory."""
    if not cache_dir.is_dir():
        print(f"📁 Cache directory not found: {cache_dir}")
        return
    
    print(f"📁 Cached Datasets in: {cache_dir}")
    print("=" * 60)
    
    cache_files = [f for f in cache_dir.glob('universal_*.pkl')]
    
    if not cache_files:
        print("  No cached datasets found.")
        return
    
    for cache_path in sorted(cache_files):
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        parts = cache_path.stem.replace('universal_', '').split('_')
        dataset_name = parts[0]
        sample_count = parts[1] if len(parts) > 1 else 'unknown'
        print(f"  - {dataset_name.upper()}: {sample_count} samples ({cache_size_mb:.2f} MB)")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Cache universal representations for datasets.')
    
    # --- MODIFICATION 1: Add 'pdb' to the list of choices ---
    # --- This fixes the 'invalid choice' error. ---
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        choices=['qm9', 'lba', 'pdb', 'all'],
        help='Dataset to cache.'
    )
    
    # --- MODIFICATION 2: Add arguments to specify data and cache paths ---
    # --- This makes the script flexible and reusable. ---
    parser.add_argument(
        '--data-path', 
        type=Path, 
        default=None,
        help='Path to the directory containing raw data files (e.g., PDBs).'
    )
    parser.add_argument(
        '--cache-dir', 
        type=Path, 
        default=Path('./data_loading/cache'),
        help='Directory where the output .pkl cache file will be saved.'
    )
    
    # --- Existing arguments remain the same ---
    parser.add_argument(
        '--max-samples', 
        type=int, 
        default=None,
        help='Maximum number of samples to cache (default: all).'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force rebuild even if the cache file exists.'
    )
    parser.add_argument(
        '--list', 
        action='store_true',
        help='List cached datasets in the specified cache directory.'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_cached_datasets(args.cache_dir)
        return 0
    
    # The 'all' option is simplified for now.
    datasets_to_process = ['qm9', 'lba', 'pdb'] if args.dataset == 'all' else [args.dataset]
    
    success_all = True
    for ds_name in datasets_to_process:
        # --- MODIFICATION 3: Pass the new path arguments to the caching function ---
        success = cache_dataset(
            dataset_name=ds_name,
            data_path=args.data_path,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
            force_rebuild=args.force
        )
        if not success:
            success_all = False
            
    return 0 if success_all else 1

if __name__ == "__main__":
    exit(main())