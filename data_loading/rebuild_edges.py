#!/usr/bin/env python3
"""
Rebuild PyG Data with New Edge Construction

This script rebuilds PyTorch Geometric datasets from existing Universal format cache (.pkl),
but with NEW edge construction parameters (e.g., hybrid edges, different cutoffs).

Benefits:
- 3-7x faster than full PDB → PyG pipeline
- Reuses existing Universal format (atoms, positions, blocks)
- Only recomputes edges and saves to new .pt files

Usage:
    python data_loading/rebuild_edges.py \
        --universal-cache cache/universal_protein_chunk_0.pkl \
        --output-dir processed_graphs_hybrid \
        --cutoff 5.0 \
        --max-neighbors 64 \
        --use-hybrid-edges \
        --sequence-neighbors-k 3 \
        --max-spatial-neighbors 48 \
        --num-random-edges 8
"""

import os
import sys
import torch
import pickle
import argparse
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_loading.data_types import UniversalMolecule
from data_loading.cache_to_pyg import OptimizedUniversalDataset


def rebuild_edges_from_cache(
    universal_cache_path: str,
    output_dir: str,
    cutoff_distance: float = 5.0,
    max_neighbors: int = 32,
    use_hybrid_edges: bool = False,
    sequence_neighbors_k: int = 3,
    max_spatial_neighbors: int = 48,
    num_random_edges: int = 8,
    random_edge_min_distance: float = 10.0,
    max_samples: Optional[int] = None,
    force: bool = False
):
    """
    Rebuild PyG dataset from Universal cache with new edge construction.
    
    This is 3-7x faster than full PDB → PyG pipeline because:
    1. BioPython parsing: SKIPPED (already have Universal format)
    2. Block creation: SKIPPED (already have blocks)
    3. Atom extraction: SKIPPED (already have atoms)
    4. Edge construction: ONLY THIS (fast!)
    
    Args:
        universal_cache_path: Path to .pkl cache file
        output_dir: Output directory for new .pt files
        cutoff_distance: Edge cutoff distance (Å)
        max_neighbors: Max neighbors per atom
        use_hybrid_edges: Enable hybrid edge construction
        sequence_neighbors_k: Tier 1 sequence neighbors
        max_spatial_neighbors: Tier 2 spatial neighbors
        num_random_edges: Tier 3 random edges
        random_edge_min_distance: Tier 3 min distance
        max_samples: Limit samples (for testing)
        force: Force rebuild if output exists
    """
    print("=" * 80)
    print("🔄 REBUILD EDGES FROM UNIVERSAL CACHE")
    print("=" * 80)
    print(f"📂 Input:  {universal_cache_path}")
    print(f"📁 Output: {output_dir}")
    print(f"⚙️  Edge params:")
    print(f"   - Cutoff: {cutoff_distance} Å")
    print(f"   - Max neighbors: {max_neighbors}")
    print(f"   - Hybrid edges: {use_hybrid_edges}")
    if use_hybrid_edges:
        print(f"     * Tier 1 (sequence): ±{sequence_neighbors_k} neighbors")
        print(f"     * Tier 2 (spatial): {max_spatial_neighbors} max neighbors")
        print(f"     * Tier 3 (random): {num_random_edges} edges (min {random_edge_min_distance} Å)")
    print("=" * 80)
    
    # Check if output exists
    processed_dir = Path(output_dir) / "processed"
    if processed_dir.exists() and not force:
        pt_files = list(processed_dir.glob("*.pt"))
        if pt_files:
            print(f"⚠️  Output already exists: {pt_files[0]}")
            print(f"💡 Use --force to rebuild")
            return
    
    # Create dataset (this will rebuild edges)
    print("\n🔄 Starting edge reconstruction...")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    
    dataset = OptimizedUniversalDataset(
        root=output_dir,
        universal_cache_path=universal_cache_path,
        max_samples=max_samples,
        cutoff_distance=cutoff_distance,
        max_neighbors=max_neighbors,
        use_hybrid_edges=use_hybrid_edges,
        sequence_neighbors_k=sequence_neighbors_k,
        max_spatial_neighbors=max_spatial_neighbors,
        num_random_edges=num_random_edges,
        random_edge_min_distance=random_edge_min_distance
    )
    
    end_time.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
    
    print(f"\n✅ Edge reconstruction complete!")
    print(f"   Total samples: {len(dataset):,}")
    print(f"   Time: {elapsed_time:.2f}s")
    print(f"   Speed: {len(dataset) / elapsed_time:.2f} samples/s")
    print(f"   Output: {processed_dir}")
    
    # Test loading a sample
    print(f"\n🔍 Testing sample loading...")
    sample = dataset[0]
    print(f"   Sample 0:")
    print(f"   - Nodes: {sample.num_nodes}")
    print(f"   - Edges: {sample.num_edges}")
    print(f"   - Edge density: {sample.num_edges / (sample.num_nodes * (sample.num_nodes - 1)):.4f}")
    
    print("\n🎉 Done!")


def rebuild_chunked_dataset(
    universal_cache_dir: str,
    output_base_dir: str,
    num_chunks: int,
    cutoff_distance: float = 5.0,
    max_neighbors: int = 32,
    use_hybrid_edges: bool = False,
    sequence_neighbors_k: int = 3,
    max_spatial_neighbors: int = 48,
    num_random_edges: int = 8,
    random_edge_min_distance: float = 10.0,
    chunk_range: Optional[str] = None,
    force: bool = False
):
    """
    Rebuild chunked dataset from Universal cache chunks.
    
    Args:
        universal_cache_dir: Directory containing universal_*_chunk_*.pkl files
        output_base_dir: Base output directory (chunks will be created as output_base_dir_chunk_N)
        num_chunks: Number of chunks
        chunk_range: Chunk range to process (e.g., "0-10" or "5")
        force: Force rebuild
    """
    print("=" * 80)
    print("🔄 REBUILD CHUNKED DATASET FROM UNIVERSAL CACHE")
    print("=" * 80)
    
    # Parse chunk range
    if chunk_range:
        if "-" in chunk_range:
            start, end = map(int, chunk_range.split("-"))
            chunks_to_process = list(range(start, end + 1))
        else:
            chunks_to_process = [int(chunk_range)]
    else:
        chunks_to_process = list(range(num_chunks))
    
    print(f"📂 Input dir: {universal_cache_dir}")
    print(f"📁 Output base: {output_base_dir}")
    print(f"📦 Chunks to process: {chunks_to_process}")
    print("=" * 80)
    
    # Process each chunk
    for chunk_idx in chunks_to_process:
        print(f"\n{'=' * 80}")
        print(f"📦 CHUNK {chunk_idx}/{num_chunks - 1}")
        print(f"{'=' * 80}")
        
        # Find universal cache file
        cache_pattern = f"universal_*_chunk_{chunk_idx}.pkl"
        cache_files = list(Path(universal_cache_dir).glob(cache_pattern))
        
        if not cache_files:
            print(f"⚠️  No cache file found matching: {cache_pattern}")
            continue
        
        cache_file = cache_files[0]
        output_dir = f"{output_base_dir}_chunk_{chunk_idx}"
        
        # Rebuild edges for this chunk
        rebuild_edges_from_cache(
            universal_cache_path=str(cache_file),
            output_dir=output_dir,
            cutoff_distance=cutoff_distance,
            max_neighbors=max_neighbors,
            use_hybrid_edges=use_hybrid_edges,
            sequence_neighbors_k=sequence_neighbors_k,
            max_spatial_neighbors=max_spatial_neighbors,
            num_random_edges=num_random_edges,
            random_edge_min_distance=random_edge_min_distance,
            force=force
        )
    
    print(f"\n{'=' * 80}")
    print(f"✅ ALL CHUNKS PROCESSED")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild PyG datasets with new edge construction (3-7x faster than full pipeline)"
    )
    
    # Input/Output
    parser.add_argument("--universal-cache", type=str, help="Path to universal .pkl cache file")
    parser.add_argument("--universal-cache-dir", type=str, help="Directory with chunked universal cache")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-chunks", type=int, help="Number of chunks (for chunked mode)")
    parser.add_argument("--chunk-range", type=str, help="Chunk range to process (e.g., '0-10' or '5')")
    
    # Edge construction params
    parser.add_argument("--cutoff", type=float, default=5.0, help="Edge cutoff distance (Å)")
    parser.add_argument("--max-neighbors", type=int, default=32, help="Max neighbors per atom")
    
    # Hybrid edges (Salad-inspired)
    parser.add_argument("--use-hybrid-edges", action="store_true", help="Enable hybrid edge construction")
    parser.add_argument("--sequence-neighbors-k", type=int, default=3, help="Tier 1: Sequence neighbors (±k)")
    parser.add_argument("--max-spatial-neighbors", type=int, default=48, help="Tier 2: Max spatial neighbors")
    parser.add_argument("--num-random-edges", type=int, default=8, help="Tier 3: Random edges per node")
    parser.add_argument("--random-edge-min-distance", type=float, default=10.0, help="Tier 3: Min distance (Å)")
    
    # Other
    parser.add_argument("--max-samples", type=int, help="Limit samples (for testing)")
    parser.add_argument("--force", action="store_true", help="Force rebuild if output exists")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.universal_cache and args.universal_cache_dir:
        raise ValueError("Specify either --universal-cache OR --universal-cache-dir, not both")
    
    if not args.universal_cache and not args.universal_cache_dir:
        raise ValueError("Must specify --universal-cache or --universal-cache-dir")
    
    # Single file mode
    if args.universal_cache:
        rebuild_edges_from_cache(
            universal_cache_path=args.universal_cache,
            output_dir=args.output_dir,
            cutoff_distance=args.cutoff,
            max_neighbors=args.max_neighbors,
            use_hybrid_edges=args.use_hybrid_edges,
            sequence_neighbors_k=args.sequence_neighbors_k,
            max_spatial_neighbors=args.max_spatial_neighbors,
            num_random_edges=args.num_random_edges,
            random_edge_min_distance=args.random_edge_min_distance,
            max_samples=args.max_samples,
            force=args.force
        )
    
    # Chunked mode
    else:
        if not args.num_chunks:
            raise ValueError("--num-chunks required for chunked mode")
        
        rebuild_chunked_dataset(
            universal_cache_dir=args.universal_cache_dir,
            output_base_dir=args.output_dir,
            num_chunks=args.num_chunks,
            cutoff_distance=args.cutoff,
            max_neighbors=args.max_neighbors,
            use_hybrid_edges=args.use_hybrid_edges,
            sequence_neighbors_k=args.sequence_neighbors_k,
            max_spatial_neighbors=args.max_spatial_neighbors,
            num_random_edges=args.num_random_edges,
            random_edge_min_distance=args.random_edge_min_distance,
            chunk_range=args.chunk_range,
            force=args.force
        )


if __name__ == "__main__":
    main()
