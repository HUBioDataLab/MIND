#!/usr/bin/env python3
"""
Prepare RNA cache from filtered CIF files.

Usage:
    python data_loading/prepare_rna_cache.py \
        --cif-dir ./filtered_rna_cifs \
        --cache-path data_loading/cache/universal_rna.pkl \
        --max-samples 1000
"""

import sys
import argparse
from pathlib import Path

sys.path.append('.')

def prepare_rna_cache(cif_dir, cache_path, max_samples=None):
    """Create RNA universal cache from CIF directory."""
    from data_loading.adapters.rna_adapter import RNAAdapter
    
    print(f"Preparing RNA cache from: {cif_dir}")
    
    cif_dir = Path(cif_dir)
    if not cif_dir.exists():
        print(f"ERROR: CIF directory not found: {cif_dir}")
        return False
    
    # Create cache directory
    cache_dir = Path(cache_path).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create adapter and process
    adapter = RNAAdapter()
    
    try:
        count = adapter.process_dataset(
            data_path=str(cif_dir),
            cache_path=cache_path,
            max_samples=max_samples
        )
        
        cache_size_mb = Path(cache_path).stat().st_size / (1024 * 1024)
        print(f"Successfully created RNA cache:")
        print(f"  Samples: {count}")
        print(f"  Size: {cache_size_mb:.2f} MB")
        print(f"  Path: {cache_path}")
        return True
        
    except Exception as e:
        print(f"ERROR creating cache: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Prepare RNA cache from CIF files')
    parser.add_argument('--cif-dir', type=str, default='./data/filtered_rna_cifs',
                       help='Directory containing filtered CIF files')
    parser.add_argument('--cache-path', type=str, 
                       default='data_loading/cache/universal_rna.pkl',
                       help='Output cache file path')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to process (default: all)')
    
    args = parser.parse_args()
    
    success = prepare_rna_cache(
        args.cif_dir,
        args.cache_path,
        args.max_samples
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())