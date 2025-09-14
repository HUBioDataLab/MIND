#!/usr/bin/env python3
"""
Universal Dataset Caching System

Caches universal representations for any dataset using the adapter system.
This script can cache QM9, LBA, PDB, or any future dataset adapters.

Usage:
    python cache_universal_datasets.py --dataset qm9 --max_samples 1000
    python cache_universal_datasets.py --dataset lba --max_samples 500
    python cache_universal_datasets.py --dataset all --max_samples 100
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add data_loading to path
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
        # TODO: Add PDB adapter when available
        raise NotImplementedError("PDB adapter not yet implemented")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_cache_path(dataset_name: str, max_samples: int = None) -> str:
    """Generate cache path for dataset"""
    cache_dir = './data_loading/cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    if max_samples:
        cache_file = f"universal_{dataset_name}_{max_samples}.pkl"
    else:
        cache_file = f"universal_{dataset_name}_all.pkl"
    
    return os.path.join(cache_dir, cache_file)

def cache_dataset(dataset_name: str, max_samples: int = None, force_rebuild: bool = False):
    """Cache universal representations for a dataset"""
    print(f"🚀 Caching {dataset_name.upper()} dataset...")
    print("=" * 60)
    
    # Get adapter and data path
    try:
        adapter, data_path = get_adapter(dataset_name)
    except Exception as e:
        print(f"❌ Error getting adapter: {e}")
        return False
    
    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        print(f"💡 Please download the dataset first using the download scripts")
        return False
    
    # Get cache path
    cache_path = get_cache_path(dataset_name, max_samples)
    
    # Check if cache already exists
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"✅ Cache already exists: {cache_path}")
        print(f"💡 Use --force to rebuild cache")
        return True
    
    # Process dataset with caching
    start_time = time.time()
    try:
        universal_data = adapter.process_dataset(
            data_path=data_path,
            cache_path=cache_path,
            max_samples=max_samples
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Successfully cached {len(universal_data)} universal samples")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"💾 Cache file: {cache_path}")
        
        # Show cache file size
        cache_size = os.path.getsize(cache_path)
        cache_size_mb = cache_size / (1024 * 1024)
        print(f"📊 Cache size: {cache_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def cache_all_datasets(max_samples: int = None, force_rebuild: bool = False):
    """Cache all available datasets"""
    datasets = ['qm9', 'lba']  # Add more as adapters become available
    
    print(f"🚀 Caching ALL datasets (max_samples={max_samples or 'all'})...")
    print("=" * 60)
    
    results = {}
    for dataset in datasets:
        print(f"\n📁 Processing {dataset.upper()}...")
        success = cache_dataset(dataset, max_samples, force_rebuild)
        results[dataset] = success
    
    # Summary
    print(f"\n📊 Caching Summary:")
    print("=" * 60)
    for dataset, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {dataset.upper()}: {status}")
    
    return all(results.values())

def list_cached_datasets():
    """List all cached datasets"""
    cache_dir = './data_loading/cache'
    
    if not os.path.exists(cache_dir):
        print("📁 No cache directory found")
        return
    
    print("📁 Cached Datasets:")
    print("=" * 60)
    
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith('universal_') and f.endswith('.pkl')]
    
    if not cache_files:
        print("  No cached datasets found")
        return
    
    for cache_file in sorted(cache_files):
        cache_path = os.path.join(cache_dir, cache_file)
        cache_size = os.path.getsize(cache_path)
        cache_size_mb = cache_size / (1024 * 1024)
        
        # Parse dataset name and sample count
        parts = cache_file.replace('universal_', '').replace('.pkl', '').split('_')
        dataset_name = parts[0]
        sample_count = parts[1] if len(parts) > 1 else 'unknown'
        
        print(f"  {dataset_name.upper()}: {sample_count} samples ({cache_size_mb:.2f} MB)")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Cache universal representations for datasets')
    parser.add_argument('--dataset', type=str, required=False, 
                       choices=['qm9', 'lba', 'pdb', 'all'],
                       help='Dataset to cache')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to cache (default: all)')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild even if cache exists')
    parser.add_argument('--list', action='store_true',
                       help='List cached datasets')
    
    args = parser.parse_args()
    
    if args.list:
        list_cached_datasets()
        return 0
    
    if not args.dataset:
        print("Error: --dataset is required unless using --list")
        return 1
    
    if args.dataset == 'all':
        success = cache_all_datasets(args.max_samples, args.force)
    else:
        success = cache_dataset(args.dataset, args.max_samples, args.force)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
