#!/usr/bin/env python3
"""
Automated chunked dataset processing pipeline.

Orchestrates the full pipeline for processing large datasets:
1. Split manifest into chunks
2. For each chunk:
   a. Create universal .pkl cache (via cache_universal_datasets.py)
   b. Convert to PyG .pt format (via cache_to_pyg.py)
3. All chunks are automatically detected and used by train_pretrain.py via LazyUniversalDataset

Usage:
    python data_loading/process_chunked_dataset.py \
        --config-yaml-path core/pretraining_config_protein.yaml \
        --data-path ../data/proteins/raw_structures_hq_40k \
        --manifest-file ../data/proteins/afdb_clusters/manifest_hq_40k_len512.csv \
        --num-chunks 10
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_chunk_range(chunk_range: str, num_chunks: int) -> List[int]:
    """
    Parse chunk range string into list of chunk indices.
    
    Args:
        chunk_range: Range string (e.g., '0-3' or '0,2,4')
        num_chunks: Total number of chunks for validation
        
    Returns:
        List of chunk indices to process
        
    Raises:
        ValueError: If chunk indices are invalid
    """
    if '-' in chunk_range:
        start, end = map(int, chunk_range.split('-'))
        chunks = list(range(start, end + 1))
    else:
        chunks = [int(x) for x in chunk_range.split(',')]
    
    # Validate indices
    for chunk_idx in chunks:
        if chunk_idx >= num_chunks or chunk_idx < 0:
            raise ValueError(
                f"Invalid chunk index: {chunk_idx} "
                f"(must be 0-{num_chunks-1})"
            )
    
    return chunks


def apply_config_defaults(args: argparse.Namespace, config: Dict) -> None:
    """Apply configuration defaults to args (CLI args override config)."""
    if args.dataset is None and 'dataset' in config:
        args.dataset = config['dataset'].lower()
        print(f"  ✓ dataset from config: {args.dataset}")
    
    if args.output_base is None and 'dataset_download_dir' in config:
        args.output_base = Path(config['dataset_download_dir'])
        print(f"  ✓ output_base from config: {args.output_base}")
    
    if args.cutoff is None and 'cutoff_distance' in config:
        args.cutoff = config['cutoff_distance']
        print(f"  ✓ cutoff from config: {args.cutoff}")
    
    if args.max_neighbors is None and 'max_neighbors' in config:
        args.max_neighbors = config['max_neighbors']
        print(f"  ✓ max_neighbors from config: {args.max_neighbors}")


def validate_arguments(args: argparse.Namespace) -> Tuple[bool, str]:
    """
    Validate all required arguments are present.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if args.dataset is None:
        return False, "--dataset is required (or must be in config)"
    if args.data_path is None:
        return False, "--data-path is required"
    if args.manifest_file is None:
        return False, "--manifest-file is required"
    if args.output_base is None:
        return False, "--output-base is required (or 'dataset_download_dir' must be in config)"
    
    if not args.data_path.exists():
        return False, f"Data path not found: {args.data_path}"
    
    if not args.manifest_file.exists():
        return False, f"Manifest file not found: {args.manifest_file}"
    
    return True, ""


def run_command(cmd: List[str], description: str) -> bool:
    """Run a subprocess command and handle errors."""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        return False


def process_chunk(chunk_idx: int, args: argparse.Namespace) -> bool:
    """
    Process a single chunk: .pkl creation → .pt conversion.
    
    Args:
        chunk_idx: Index of chunk to process
        args: Parsed command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n\n{'#'*80}")
    print(f"# PROCESSING CHUNK {chunk_idx + 1}/{args.num_chunks}")
    print(f"{'#'*80}\n")
    
    script_dir = Path(__file__).parent
    
    # Step 1: Create universal .pkl cache for this chunk
    cache_cmd = [
        sys.executable,
        str(script_dir / "cache_universal_datasets.py"),
        "--dataset", args.dataset,
        "--data-path", str(args.data_path),
        "--manifest-file", str(args.manifest_file),
        "--cache-dir", str(args.cache_dir),
        "--num-chunks", str(args.num_chunks),
        "--chunk-index", str(chunk_idx),
    ]
    
    if args.force:
        cache_cmd.append("--force")
    
    if not run_command(cache_cmd, f"Creating .pkl cache for chunk {chunk_idx}"):
        return False
    
    # Step 2: Convert .pkl to .pt for this chunk
    pkl_file = args.cache_dir / f"universal_{args.dataset}_chunk_{chunk_idx}.pkl"
    output_dir = Path(f"{args.output_base}_chunk_{chunk_idx}")
    
    pyg_cmd = [
        sys.executable,
        str(script_dir / "cache_to_pyg.py"),
        "--input-pkl", str(pkl_file),
        "--output-dir", str(output_dir),
        "--dataset-type", args.dataset,
    ]
    
    if args.cutoff:
        pyg_cmd.extend(["--cutoff", str(args.cutoff)])
    if args.max_neighbors:
        pyg_cmd.extend(["--max-neighbors", str(args.max_neighbors)])
    if args.force:
        pyg_cmd.append("--force")
    
    if not run_command(pyg_cmd, f"Converting .pkl to .pt for chunk {chunk_idx}"):
        return False
    
    return True


def print_summary(
    chunks_to_process: List[int],
    success_count: int,
    elapsed_time: float,
    output_base: Path
) -> None:
    """Print final pipeline summary."""
    print("\n\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"✅ Successful chunks: {success_count}/{len(chunks_to_process)}")
    print(f"⏱️  Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
    
    print(f"\n📦 Output directories created:")
    for chunk_idx in chunks_to_process:
        output_dir = Path(f"{output_base}_chunk_{chunk_idx}")
        if output_dir.exists():
            pt_files = list(output_dir.glob("processed/*.pt"))
            if pt_files:
                size_mb = pt_files[0].stat().st_size / (1024 * 1024)
                print(f"   - {output_dir.name}: {size_mb:.2f} MB")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Run training with automatic chunked dataset detection:")
    print(f"      python -m core.train_pretrain --config-yaml-path core/pretraining_config_protein.yaml")
    print(f"   2. The training will automatically detect and use LazyUniversalDataset")
    print("="*80)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated chunked dataset processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Config file
    parser.add_argument(
        "--config-yaml-path",
        type=Path,
        default=None,
        help="Path to YAML config file (parameters can be overridden by CLI args)"
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["qm9", "lba", "pdb", "rna"],
        help="Dataset type"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to raw data directory"
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help="Path to manifest CSV file"
    )
    
    # Chunking parameters
    parser.add_argument(
        "--num-chunks",
        type=int,
        required=True,
        help="Number of chunks to split dataset into"
    )
    parser.add_argument(
        "--chunk-range",
        type=str,
        default=None,
        help="Process only specific chunks (e.g., '0-3' or '0,2,4')"
    )
    
    # Directory parameters
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for .pkl cache files (default: 'data_loading/cache')"
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help="Base path for output .pt directories (default: from config 'dataset_download_dir')"
    )
    
    # PyG conversion parameters
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Edge cutoff distance in Å (default: 5.0)"
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=None,
        help="Maximum neighbors per node (default: 64)"
    )
    
    # Control parameters
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild existing caches"
    )
    
    args = parser.parse_args()
    
    # Load and apply config
    if args.config_yaml_path:
        print(f"📋 Loading config from: {args.config_yaml_path}")
        try:
            config = load_config(args.config_yaml_path)
            apply_config_defaults(args, config)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return 1
    
    # Apply final defaults
    if args.cache_dir is None:
        args.cache_dir = Path("data_loading/cache")
    if args.cutoff is None:
        args.cutoff = 5.0
    if args.max_neighbors is None:
        args.max_neighbors = 64
    
    # Validate arguments
    is_valid, error_msg = validate_arguments(args)
    if not is_valid:
        print(f"❌ Error: {error_msg}")
        return 1
    
    # Create output directories
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.output_base.parent != Path('.'):
        args.output_base.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine chunks to process
    if args.chunk_range:
        try:
            chunks_to_process = parse_chunk_range(args.chunk_range, args.num_chunks)
        except ValueError as e:
            print(f"❌ Error: {e}")
            return 1
    else:
        chunks_to_process = list(range(args.num_chunks))
    
    # Print pipeline info
    print("\n" + "="*80)
    print("CHUNKED DATASET PROCESSING PIPELINE")
    print("="*80)
    print(f"Dataset:        {args.dataset}")
    print(f"Data path:      {args.data_path}")
    print(f"Manifest:       {args.manifest_file}")
    print(f"Total chunks:   {args.num_chunks}")
    print(f"Processing:     {chunks_to_process}")
    print(f"Cache dir:      {args.cache_dir}")
    print(f"Output base:    {args.output_base}")
    print(f"Cutoff:         {args.cutoff} Å")
    print(f"Max neighbors:  {args.max_neighbors}")
    print("="*80)
    
    # Process chunks
    total_start = time.time()
    success_count = 0
    
    for chunk_idx in chunks_to_process:
        if process_chunk(chunk_idx, args):
            success_count += 1
        else:
            print(f"\n⚠️  Chunk {chunk_idx} failed. Continue? (y/n): ", end='')
            try:
                response = input().strip().lower()
                if response != 'y':
                    print("❌ Aborting pipeline.")
                    return 1
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Aborting pipeline.")
                return 1
    
    total_elapsed = time.time() - total_start
    print_summary(chunks_to_process, success_count, total_elapsed, args.output_base)
    
    return 0 if success_count == len(chunks_to_process) else 1


if __name__ == "__main__":
    sys.exit(main())
