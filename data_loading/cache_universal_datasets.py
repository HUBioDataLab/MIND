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

import sys
import argparse
import time
import traceback
from pathlib import Path
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from data_loading.adapters.base_adapter import BaseAdapter

sys.path.append('.')

def _get_adapter_factory(dataset_name: str) -> Tuple[type, str]:
    """Get adapter class and default data path by dataset name."""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'qm9':
        from data_loading.adapters.qm9_adapter import QM9Adapter
        return QM9Adapter, './data/qm9'
    elif dataset_name == 'lba':
        from data_loading.adapters.lba_adapter import LBAAdapter
        return LBAAdapter, './data/LBA'
    elif dataset_name == 'coconut':
        from data_loading.adapters.coconut_adapter import COCONUTAdapter
        return COCONUTAdapter, './data'
    elif dataset_name == 'rna':
        from data_loading.adapters.rna_adapter import RNAAdapter
        return RNAAdapter, './data/rna/raw_structures'
    elif dataset_name == 'pdb':
        from data_loading.adapters.protein_adapter import ProteinAdapter
        return ProteinAdapter, '../data/proteins/raw_structures_hq'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_adapter(dataset_name: str) -> Tuple['BaseAdapter', str]:
    """Get adapter instance and default data path by dataset name."""
    adapter_class, default_path = _get_adapter_factory(dataset_name)
    return adapter_class(), default_path

def _generate_cache_filename(dataset_name: str, max_samples: Optional[int] = None,
                             num_chunks: Optional[int] = None, chunk_index: Optional[int] = None) -> str:
    """Generate cache filename based on dataset and chunking parameters."""
    if num_chunks is not None and chunk_index is not None:
        return f"universal_{dataset_name}_chunk_{chunk_index}.pkl"
    elif max_samples:
        return f"universal_{dataset_name}_{max_samples}.pkl"
    else:
        return f"universal_{dataset_name}_all.pkl"

def _create_chunk_manifest(manifest_file: Path, num_chunks: int, chunk_index: int,
                          cache_dir: Path) -> Optional[Path]:
    """Create temporary manifest file for a specific chunk. Returns path or None on error."""
    import pandas as pd
    
    manifest_df = pd.read_csv(manifest_file)
    total_samples = len(manifest_df)
    chunk_size = (total_samples + num_chunks - 1) // num_chunks
    start_idx = chunk_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_samples)
    
    chunk_df = manifest_df.iloc[start_idx:end_idx]
    chunk_manifest_file = cache_dir / f"temp_manifest_chunk_{chunk_index}.csv"
    chunk_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    chunk_df.to_csv(chunk_manifest_file, index=False)
    
    print(f"📋 Chunk manifest: {len(chunk_df):,} samples (rows {start_idx}-{end_idx})")
    return chunk_manifest_file

def cache_dataset(dataset_name: str, data_path: Optional[Path], cache_dir: Path,
                  max_samples: Optional[int] = None, force_rebuild: bool = False,
                  manifest_file: Optional[Path] = None, num_chunks: Optional[int] = None,
                  chunk_index: Optional[int] = None) -> bool:
    """Cache universal representations for a dataset with optional chunking support."""
    print(f"🚀 Caching {dataset_name.upper()} dataset...")
    if num_chunks is not None and chunk_index is not None:
        print(f"📦 Chunking: Processing chunk {chunk_index + 1}/{num_chunks}")
    print("=" * 60)
    
    adapter, default_data_path = get_adapter(dataset_name)
    if data_path is None:
        data_path = Path(default_data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data path not found: {data_path}\n"
            f"Please provide the correct path using the --data-path argument."
        )
    
    chunk_manifest_file = None
    if manifest_file and num_chunks is not None and chunk_index is not None:
        print(f"📋 Loading manifest for chunking: {manifest_file}")
        chunk_manifest_file = _create_chunk_manifest(manifest_file, num_chunks, chunk_index, cache_dir)
        max_samples = None
    
    cache_file = _generate_cache_filename(dataset_name, max_samples, num_chunks, chunk_index)
    cache_path = cache_dir / cache_file
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists() and not force_rebuild:
        print(f"✅ Cache already exists: {cache_path}")
        print(f"💡 Use --force to rebuild the cache.")
        if chunk_manifest_file and chunk_manifest_file.exists():
            chunk_manifest_file.unlink()
        return True
    
    start_time = time.time()
    try:
        manifest_path = str(chunk_manifest_file) if chunk_manifest_file else (str(manifest_file) if manifest_file else None)
        processed_count = adapter.process_dataset(
            data_path=str(data_path),
            cache_path=str(cache_path),
            max_samples=max_samples,
            manifest_file=manifest_path,
            num_chunks=num_chunks,
            chunk_index=chunk_index
        )
        
        if chunk_manifest_file and chunk_manifest_file.exists():
            chunk_manifest_file.unlink()
        
        processing_time = time.time() - start_time
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        
        print(f"✅ Successfully cached {processed_count} universal samples.")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds.")
        print(f"💾 Cache file: {cache_path}")
        print(f"📊 Cache size: {cache_size_mb:.2f} MB")
        
        return True
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        traceback.print_exc()
        if chunk_manifest_file and chunk_manifest_file.exists():
            chunk_manifest_file.unlink()
        return False

def list_cached_datasets(cache_dir: Path) -> None:
    """List all cached datasets in the specified directory."""
    if not cache_dir.is_dir():
        print(f"📁 Cache directory not found: {cache_dir}")
        return
    
    print(f"📁 Cached Datasets in: {cache_dir}")
    print("=" * 60)
    
    cache_files = list(cache_dir.glob('universal_*.pkl'))
    if not cache_files:
        print("  No cached datasets found.")
        return
    
    for cache_path in sorted(cache_files):
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        parts = cache_path.stem.replace('universal_', '').split('_')
        dataset_name = parts[0]
        sample_count = parts[1] if len(parts) > 1 else 'unknown'
        print(f"  - {dataset_name.upper()}: {sample_count} samples ({cache_size_mb:.2f} MB)")

def _validate_chunking_args(num_chunks: Optional[int], chunk_index: Optional[int]) -> None:
    """Validate chunking arguments."""
    if (num_chunks is None) != (chunk_index is None):
        raise ValueError("--num-chunks and --chunk-index must be used together")
    if chunk_index is not None and chunk_index >= num_chunks:
        raise ValueError(f"--chunk-index must be < --num-chunks (got {chunk_index} >= {num_chunks})")

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Cache universal representations for datasets.')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['qm9', 'lba', 'coconut', 'pdb', 'rna', 'all'],
                       help='Dataset to cache.')
    parser.add_argument('--data-path', type=Path, default=None,
                       help='Path to directory containing raw data files (e.g., PDBs).')
    parser.add_argument('--cache-dir', type=Path, default=Path('./data_loading/cache'),
                       help='Directory where output .pkl cache file will be saved.')
    parser.add_argument('--manifest-file', type=Path, default=None,
                       help='Path to manifest CSV file (for proteins, contains repId column).')
    parser.add_argument('--num-chunks', type=int, default=None,
                       help='Split dataset into N chunks for parallel processing.')
    parser.add_argument('--chunk-index', type=int, default=None,
                       help='Process only this chunk (0 to num-chunks-1).')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to cache (default: all).')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild even if cache file exists.')
    parser.add_argument('--list', action='store_true',
                       help='List cached datasets in specified cache directory.')
    
    args = parser.parse_args()
    
    if args.list:
        list_cached_datasets(args.cache_dir)
        return 0
    
    try:
        _validate_chunking_args(args.num_chunks, args.chunk_index)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    
    datasets_to_process = ['qm9', 'lba', 'coconut', 'pdb', 'rna'] if args.dataset == 'all' else [args.dataset]
    
    success_all = True
    for ds_name in datasets_to_process:
        try:
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
        except FileNotFoundError as e:
            print(f"❌ {e}")
            success_all = False
    
    return 0 if success_all else 1

if __name__ == "__main__":
    exit(main())