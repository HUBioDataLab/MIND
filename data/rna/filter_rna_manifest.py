#!/usr/bin/env python3
"""
Filter RNA structures by size before processing.

This script analyzes your RNA manifest and filters out extremely large structures
that will cause OOM errors during training.
"""

import os
import pandas as pd
from Bio.PDB import MMCIFParser
from pathlib import Path
from tqdm import tqdm

def count_atoms_in_cif(cif_path: str) -> int:
    """Count non-hydrogen atoms in a CIF file"""
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("rna", cif_path)
        atom_count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        # Skip hydrogen atoms
                        element = (atom.element or "C").strip().upper()
                        if element == 'H':
                            continue
                        # Skip alternate locations
                        altloc = atom.get_altloc()
                        if altloc not in (" ", "A"):
                            continue
                        atom_count += 1
        return atom_count
    except Exception as e:
        print(f"Error reading {cif_path}: {e}")
        return -1

def filter_rna_manifest_by_size(
    input_manifest: str,
    raw_dir: str,
    output_manifest: str,
    max_atoms: int = 5000,
    dry_run: bool = False
):
    """
    Filter RNA manifest to remove structures larger than max_atoms.
    
    Args:
        input_manifest: Path to input manifest CSV
        raw_dir: Directory containing CIF files
        output_manifest: Path to save filtered manifest
        max_atoms: Maximum number of atoms per structure
        dry_run: If True, only print statistics without saving
    """
    print(f"📋 Loading manifest: {input_manifest}")
    df = pd.read_csv(input_manifest)
    print(f"📊 Total structures in manifest: {len(df)}")
    
    # Count atoms for each structure
    print(f"🔍 Analyzing structure sizes...")
    atom_counts = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        cif_path = os.path.join(raw_dir, f"{row['name']}.cif")
        if not os.path.exists(cif_path):
            print(f"⚠️  File not found: {cif_path}")
            atom_counts.append(-1)
            continue
        
        atom_count = count_atoms_in_cif(cif_path)
        atom_counts.append(atom_count)
    
    df['atom_count'] = atom_counts
    
    # Statistics
    print(f"\n📊 RNA Structure Size Statistics:")
    print(f"   Total structures: {len(df)}")
    
    # Filter by size CORRECTLY
    valid_mask = (df['atom_count'] > 0) & (df['atom_count'] <= max_atoms)
    too_large_mask = df['atom_count'] > max_atoms
    failed_mask = df['atom_count'] == -1
    
    print(f"   Valid structures (1-{max_atoms} atoms): {valid_mask.sum()}")
    print(f"   Too large (>{max_atoms} atoms): {too_large_mask.sum()}")
    print(f"   Failed to read: {failed_mask.sum()}")
    
    # Get ONLY valid structures for statistics
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) > 0:
        print(f"\n📈 Size Distribution (VALID structures only):")
        print(f"   Min atoms: {valid_df['atom_count'].min()}")
        print(f"   Max atoms: {valid_df['atom_count'].max()}")
        print(f"   Mean atoms: {valid_df['atom_count'].mean():.0f}")
        print(f"   Median atoms: {valid_df['atom_count'].median():.0f}")
        
        # Show largest VALID structures
        largest = valid_df.nlargest(5, 'atom_count')
        print(f"\n🔬 5 Largest VALID Structures:")
        for _, row in largest.iterrows():
            print(f"   {row['name']}: {row['atom_count']} atoms")
    
    # Show structures that were FILTERED OUT
    too_large_df = df[too_large_mask]
    if len(too_large_df) > 0:
        print(f"\n❌ Filtered Out (>{max_atoms} atoms):")
        largest_removed = too_large_df.nlargest(min(10, len(too_large_df)), 'atom_count')
        for _, row in largest_removed.iterrows():
            print(f"   {row['name']}: {row['atom_count']} atoms")
    
    if not dry_run:
        # Save ONLY valid structures
        valid_df.to_csv(output_manifest, index=False)
        print(f"\n✅ Filtered manifest saved to: {output_manifest}")
        print(f"   Kept {len(valid_df)} structures (removed {len(df) - len(valid_df)})")
    else:
        print(f"\n💡 Dry run mode - no files saved")
        print(f"   Would keep {len(valid_df)}/{len(df)} structures")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter RNA manifest by structure size")
    parser.add_argument("--input-manifest", required=True, help="Input manifest CSV")
    parser.add_argument("--raw-dir", required=True, help="Directory with CIF files")
    parser.add_argument("--output-manifest", required=True, help="Output filtered manifest CSV")
    parser.add_argument("--max-atoms", type=int, default=5000, help="Max atoms per structure")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just show stats")
    
    args = parser.parse_args()
    
    filter_rna_manifest_by_size(
        args.input_manifest,
        args.raw_dir,
        args.output_manifest,
        args.max_atoms,
        args.dry_run
    )