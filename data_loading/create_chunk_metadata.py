#!/usr/bin/env python3
"""
Chunk Metadata Generator for Multi-Domain Training

This script generates lightweight JSON metadata files for each processed .pt chunk.
Metadata enables atom-aware sampling without loading full chunks into memory.

Generated metadata includes:
- chunk_file: Name of the .pt file
- num_samples: Total number of samples in the chunk
- dataset_type: Type of dataset (pdb, qm9, metabolite, rna, dna)
- total_atoms: Total atoms across all samples
- total_edges: Total edges across all samples
- samples: List of per-sample metadata
  - idx: Local index within chunk
  - num_atoms: Number of atoms in this sample
  - num_edges: Number of edges in this sample
  - bucket: Size bucket (0-4 based on atom count)

Usage:
    # Generate metadata for all chunks in a directory
    python create_chunk_metadata.py --chunk-dir /path/to/chunks
    
    # Force regeneration of existing metadata
    python create_chunk_metadata.py --chunk-dir /path/to/chunks --overwrite
    
    # Process specific chunk file
    python create_chunk_metadata.py --chunk-file /path/to/chunk_0.pt
"""

import os
import sys
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List


def assign_bucket(num_atoms: int) -> int:
    """
    Assign size bucket based on atom count.
    
    Buckets:
    - 0: Tiny (1-50 atoms) - small molecules
    - 1: Small (51-200 atoms) - peptides, large molecules
    - 2: Medium (201-1000 atoms) - small proteins
    - 3: Large (1001-5000 atoms) - medium proteins
    - 4: Huge (5000+ atoms) - large proteins/complexes
    
    Args:
        num_atoms: Number of atoms in the sample
        
    Returns:
        Bucket index (0-4)
    """
    if num_atoms <= 50:
        return 0
    elif num_atoms <= 200:
        return 1
    elif num_atoms <= 1000:
        return 2
    elif num_atoms <= 5000:
        return 3
    else:
        return 4


def create_chunk_metadata(chunk_pt_file: str, overwrite: bool = False) -> Dict[str, Any]:
    """
    Create metadata JSON file for a single chunk.
    
    Args:
        chunk_pt_file: Path to the .pt chunk file
        overwrite: If True, regenerate metadata even if it exists
        
    Returns:
        Dictionary containing metadata
        
    Raises:
        FileNotFoundError: If chunk file doesn't exist
        RuntimeError: If chunk file is corrupted or invalid
    """
    chunk_path = Path(chunk_pt_file)
    
    # Validate input
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_pt_file}")
    
    # Determine metadata path
    metadata_path = chunk_path.parent / f"{chunk_path.stem}_metadata.json"
    
    # Skip if metadata already exists and overwrite is False
    if metadata_path.exists() and not overwrite:
        print(f"⏭️  Metadata already exists: {metadata_path.name}")
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    print(f"🔄 Processing: {chunk_path.name}")
    
    try:
        # Load chunk (only metadata, not full data)
        data, slices = torch.load(chunk_pt_file, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load chunk {chunk_pt_file}: {e}")
    
    # Validate chunk structure
    if not hasattr(data, 'pos') or 'pos' not in slices:
        raise RuntimeError(f"Invalid chunk structure (missing 'pos'): {chunk_pt_file}")
    
    # Extract basic info
    num_samples = len(slices['pos']) - 1
    
    # Extract dataset type (all samples in a chunk should have the same type)
    if hasattr(data, 'dataset_type') and data.dataset_type is not None:
        if isinstance(data.dataset_type, (list, tuple)):
            dataset_type = data.dataset_type[0] if len(data.dataset_type) > 0 else 'unknown'
        else:
            dataset_type = str(data.dataset_type)
    else:
        dataset_type = 'unknown'
    
    # Initialize metadata
    metadata = {
        'chunk_file': chunk_path.name,
        'num_samples': num_samples,
        'dataset_type': dataset_type,
        'total_atoms': 0,
        'total_edges': 0,
        'samples': []
    }
    
    # Extract per-sample metadata
    for i in tqdm(range(num_samples), desc="Extracting sample metadata", leave=False):
        start_idx = slices['pos'][i].item()
        end_idx = slices['pos'][i + 1].item()
        num_atoms = int(end_idx - start_idx)
        
        # Calculate edge count
        if 'edge_index' in slices:
            edge_start = slices['edge_index'][i].item()
            edge_end = slices['edge_index'][i + 1].item()
            num_edges = int(edge_end - edge_start)
        else:
            num_edges = 0
        
        # Assign bucket
        bucket = assign_bucket(num_atoms)
        
        # Append to metadata
        metadata['samples'].append({
            'idx': i,
            'num_atoms': num_atoms,
            'num_edges': num_edges,
            'bucket': bucket
        })
        
        # Update totals
        metadata['total_atoms'] += num_atoms
        metadata['total_edges'] += num_edges
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {metadata_path.name}")
    print(f"   Samples: {num_samples:,} | Total atoms: {metadata['total_atoms']:,} | Type: {dataset_type}")
    
    # Free memory
    del data, slices
    
    return metadata


def process_chunk_directory(chunk_dir: str, overwrite: bool = False) -> List[Dict[str, Any]]:
    """
    Process all .pt files in a directory and generate metadata.
    
    Args:
        chunk_dir: Directory containing .pt chunk files
        overwrite: If True, regenerate existing metadata
        
    Returns:
        List of metadata dictionaries for all processed chunks
    """
    chunk_path = Path(chunk_dir)
    
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")
    
    # Find all .pt files (excluding metadata files)
    pt_files = sorted([
        f for f in chunk_path.glob("*.pt") 
        if not f.name.startswith('pre_')  # Skip PyG metadata files
    ])
    
    if not pt_files:
        print(f"⚠️  No .pt chunk files found in: {chunk_dir}")
        return []
    
    print(f"🔍 Found {len(pt_files)} chunk files in: {chunk_dir}")
    print(f"{'Overwrite mode: ON' if overwrite else 'Skipping existing metadata'}")
    print()
    
    # Process each chunk
    all_metadata = []
    failed_chunks = []
    
    for pt_file in pt_files:
        try:
            metadata = create_chunk_metadata(str(pt_file), overwrite=overwrite)
            all_metadata.append(metadata)
        except Exception as e:
            print(f"❌ Failed to process {pt_file.name}: {e}")
            failed_chunks.append(pt_file.name)
            continue
    
    # Summary
    print()
    print("=" * 60)
    print(f"✅ Successfully processed: {len(all_metadata)}/{len(pt_files)} chunks")
    
    if failed_chunks:
        print(f"❌ Failed chunks: {len(failed_chunks)}")
        for failed in failed_chunks:
            print(f"   - {failed}")
    
    # Aggregate statistics
    if all_metadata:
        total_samples = sum(m['num_samples'] for m in all_metadata)
        total_atoms = sum(m['total_atoms'] for m in all_metadata)
        dataset_types = set(m['dataset_type'] for m in all_metadata)
        
        print()
        print("📊 Aggregate Statistics:")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Total atoms: {total_atoms:,}")
        print(f"   Dataset types: {', '.join(dataset_types)}")
        print(f"   Avg atoms/sample: {total_atoms/total_samples:.1f}")
    
    print("=" * 60)
    
    return all_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate metadata for PyG chunk files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all chunks in a directory
  python create_chunk_metadata.py --chunk-dir /path/to/chunks
  
  # Force regeneration of all metadata
  python create_chunk_metadata.py --chunk-dir /path/to/chunks --overwrite
  
  # Process a single chunk file
  python create_chunk_metadata.py --chunk-file /path/to/chunk_0.pt
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--chunk-dir', type=str, help='Directory containing chunk .pt files')
    group.add_argument('--chunk-file', type=str, help='Single chunk .pt file to process')
    
    parser.add_argument('--overwrite', action='store_true', 
                       help='Regenerate metadata even if it already exists')
    
    args = parser.parse_args()
    
    try:
        if args.chunk_dir:
            # Process entire directory
            process_chunk_directory(args.chunk_dir, overwrite=args.overwrite)
        else:
            # Process single file
            create_chunk_metadata(args.chunk_file, overwrite=args.overwrite)
            
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

