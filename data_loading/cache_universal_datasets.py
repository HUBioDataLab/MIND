#!/usr/bin/env python3
"""
Universal Dataset Caching System

Caches universal representations for any dataset using the adapter system.
This script can cache QM9, LBA, COCONUT, PDB, RNA or any future dataset adapters.

Usage:
    # Process the first 1000 samples of the QM9 dataset using default paths
    python data_loading/cache_universal_datasets.py --dataset qm9 --max_samples 1000

    # Process the custom PDB dataset from a specific folder of raw structures
    python data_loading/cache_universal_datasets.py \
    --dataset pdb \
    --data-path ../data/proteins/raw_structures_hq_40k \
    --cache-dir data_loading/cache

    # Process the PDB dataset and save the output cache to a custom location
    python data_loading/cache_universal_datasets.py --dataset pdb \
      --data-path ../data/proteins/raw_structures_hq_40k \
      --cache-dir ../data/proteins/cache

    # Process COCONUT dataset
    python data_loading/cache_universal_datasets.py --dataset coconut --max_samples 1000

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
    elif dataset_name.lower() == 'coconut':
        from data_loading.adapters.coconut_adapter import COCONUTAdapter
        return COCONUTAdapter(), './data'
    elif dataset_name.lower() == 'rna':
        from data_loading.adapters.rna_adapter import RNAAdapter
        return RNAAdapter(), './data/filtered_rna_cifs'
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

def cache_dataset(dataset_name: str, data_path: Path, cache_dir: Path, max_samples: int = None, 
                  force_rebuild: bool = False, manifest_file: Path = None,
                  num_chunks: int = None, chunk_index: int = None):
    """
    Cache universal representations for a dataset with optional chunking support.
    
    Args:
        dataset_name: Dataset type (qm9, lba, pdb, rna)
        data_path: Path to raw data directory
        cache_dir: Directory to save cache files
        max_samples: Maximum samples to process (optional)
        force_rebuild: Force rebuild even if cache exists
        manifest_file: Manifest CSV file (for proteins)
        num_chunks: Split dataset into N chunks
        chunk_index: Process only this chunk (0 to num_chunks-1)
    """
    print(f"🚀 Caching {dataset_name.upper()} dataset...")
    if num_chunks and chunk_index is not None:
        print(f"📦 Chunking: Processing chunk {chunk_index + 1}/{num_chunks}")
    print("=" * 60)
    
    adapter, default_data_path = get_adapter(dataset_name)
    
    # If a data_path isn't provided via CLI, use the default from the adapter
    if data_path is None:
        data_path = Path(default_data_path)

    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        print(f"💡 Please provide the correct path using the --data-path argument.")
        return False
    
    # Handle chunking for manifest-based datasets
    chunk_manifest_file = None
    if manifest_file and num_chunks and chunk_index is not None:
        import pandas as pd
        
        print(f"📋 Loading manifest for chunking: {manifest_file}")
        manifest_df = pd.read_csv(manifest_file)
        total_samples = len(manifest_df)
        
        # Calculate chunk boundaries
        chunk_size = (total_samples + num_chunks - 1) // num_chunks  # Ceiling division
        start_idx = chunk_index * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        
        # Extract this chunk's rows
        chunk_df = manifest_df.iloc[start_idx:end_idx]
        
        # Save temporary chunk manifest
        chunk_manifest_file = cache_dir / f"temp_manifest_chunk_{chunk_index}.csv"
        chunk_manifest_file.parent.mkdir(parents=True, exist_ok=True)
        chunk_df.to_csv(chunk_manifest_file, index=False)
        
        print(f"📋 Chunk manifest: {len(chunk_df):,} samples (rows {start_idx}-{end_idx})")
        max_samples = len(chunk_df)  # Override max_samples for this chunk
    
    # Generate cache path with chunk info
    if num_chunks and chunk_index is not None:
        cache_file = f"universal_{dataset_name}_chunk_{chunk_index}.pkl"
    elif max_samples:
        cache_file = f"universal_{dataset_name}_{max_samples}.pkl"
    else:
        cache_file = f"universal_{dataset_name}_all.pkl"
    
    cache_path = cache_dir / cache_file
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists() and not force_rebuild:
        print(f"✅ Cache already exists: {cache_path}")
        print(f"💡 Use --force to rebuild the cache.")
        # Cleanup temp manifest
        if chunk_manifest_file and chunk_manifest_file.exists():
            chunk_manifest_file.unlink()
        return True
    
    start_time = time.time()
    try:
        # The BaseAdapter's process_dataset method is called here
        processed_count = adapter.process_dataset(
            data_path=str(data_path),
            cache_path=str(cache_path),
            max_samples=max_samples,
            manifest_file=str(chunk_manifest_file) if chunk_manifest_file else (str(manifest_file) if manifest_file else None)
        )
        
        # Cleanup temporary chunk manifest
        if chunk_manifest_file and chunk_manifest_file.exists():
            chunk_manifest_file.unlink()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Successfully cached {processed_count} universal samples.")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds.")
        print(f"💾 Cache file: {cache_path}")
        
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"📊 Cache size: {cache_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup temp manifest on error
        if chunk_manifest_file and chunk_manifest_file.exists():
            chunk_manifest_file.unlink()
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
    
    # Add coconut to the list of choices from the old branch
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        choices=['qm9', 'lba', 'coconut', 'pdb', 'rna', 'all'],
        help='Dataset to cache.'
    )
    
    # Keep the flexible path arguments from main
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
    parser.add_argument(
        '--manifest-file',
        type=Path,
        default=None,
        help='Path to manifest CSV file (for proteins, contains repId column)'
    )
    
    # NEW: Chunking parameters
    parser.add_argument(
        '--num-chunks',
        type=int,
        default=None,
        help='Split dataset into N chunks for parallel processing'
    )
    parser.add_argument(
        '--chunk-index',
        type=int,
        default=None,
        help='Process only this chunk (0 to num-chunks-1)'
    )
    
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
    
    # Validate chunking args
    if (args.num_chunks is None) != (args.chunk_index is None):
        print("❌ Error: --num-chunks and --chunk-index must be used together")
        return 1
    
    if args.chunk_index is not None and args.chunk_index >= args.num_chunks:
        print(f"❌ Error: --chunk-index must be < --num-chunks (got {args.chunk_index} >= {args.num_chunks})")
        return 1
    
    # The 'all' option includes coconut now.
    datasets_to_process = ['qm9', 'lba', 'coconut', 'pdb', 'rna'] if args.dataset == 'all' else [args.dataset]
    
    success_all = True
    for ds_name in datasets_to_process:
        success = cache_dataset(
            dataset_name=ds_name,
            data_path=args.data_path,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
            force_rebuild=args.force,
            manifest_file=args.manifest_file,
            num_chunks=args.num_chunks,
            chunk_index=args.chunk_index
        )
        if not success:
            success_all = False
            
    return 0 if success_all else 1

if __name__ == "__main__":
    exit(main())