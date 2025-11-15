# data/protein_pipeline/1_filter_and_create_manifest.py

"""
Filter AlphaFold DB metadata and create download manifest.

This script provides two modes:
1. 'analyze': Analyze metadata distribution (pLDDT scores, sequence lengths)
2. 'manifest': Create a filtered manifest CSV for downloading protein structures

Usage:
    # Analyze metadata
    python data/protein_pipeline/1_filter_and_create_manifest.py \
        --mode analyze \
        --metadata-file ../data/proteins/afdb_clusters/representatives_metadata.tsv.gz

    # Create manifest
    python data/protein_pipeline/1_filter_and_create_manifest.py \
        --mode manifest \
        --metadata-file ../data/proteins/afdb_clusters/representatives_metadata.tsv.gz \
        --target-count 40000 \
        --output ../data/proteins/afdb_clusters/manifest_hq_40k.csv \
        --existing-structures-dir ../data/proteins/raw_structures_hq_40k \
        --plddt 70 \
        --max-len 512

    Note: --existing-structures-dir prevents re-downloading proteins that have
    already been downloaded.
"""

import argparse
import time
from pathlib import Path
from typing import Set, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# AlphaFold DB API endpoint
API_BASE_URL = "https://alphafold.ebi.ac.uk/api/prediction"

# Metadata column names
METADATA_COLUMNS = [
    'repId', 'isDark', 'nMem', 'repLen', 'avgLen', 
    'repPlddt', 'avgPlddt', 'LCAtaxid'
]


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load AlphaFold DB metadata from compressed TSV file.
    
    Args:
        metadata_path: Path to the gzipped TSV metadata file
        
    Returns:
        DataFrame with metadata columns
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If file format is invalid
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    try:
        df = pd.read_csv(
            metadata_path, 
            sep='\t', 
            header=None, 
            compression='gzip'
        )
        df.columns = METADATA_COLUMNS
        return df
    except Exception as e:
        raise ValueError(f"Failed to read metadata file: {e}")


def get_download_urls_from_api(uniprot_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Get PDB and CIF download URLs from AlphaFold DB API."""
    api_url = f"{API_BASE_URL}/{uniprot_id}"
    try:
        r = requests.get(api_url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                entry = data[0]
                return entry.get('pdbUrl'), entry.get('cifUrl')
    except Exception:
        pass
    return None, None


def get_existing_structure_ids(structures_dir: Path) -> Set[str]:
    """
    Extract UniProt IDs from existing PDB/CIF files in the directory.
    
    Args:
        structures_dir: Directory containing existing structure files
        
    Returns:
        Set of UniProt IDs (file stems without extension)
    """
    if not structures_dir.is_dir():
        return set()
    
    existing_ids = set() # Used set to avoid duplicates and improve performance
    print(f"Scanning existing structures in: {structures_dir}")
    
    # Collect IDs from both PDB and CIF files
    for ext in ['*.pdb', '*.cif']:
        for file_path in structures_dir.glob(ext):
            existing_ids.add(file_path.stem)
    
    print(f"Found {len(existing_ids):,} existing structures")
    return existing_ids


def analyze_metadata(metadata_path: Path) -> None:
    """
    Analyze metadata distribution: pLDDT scores, sequence lengths, and combined filters.
    
    Prints statistics about:
    - pLDDT score distribution at various thresholds
    - Sequence length distribution
    - Combined filter statistics (pLDDT + length)
    
    Args:
        metadata_path: Path to the metadata file
    """
    print(f"Loading and analyzing metadata: {metadata_path}")
    
    try:
        df = load_metadata(metadata_path)
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    total_representatives = len(df)
    print(f"\nTotal representative proteins: {total_representatives:,}")
    
    # pLDDT score distribution analysis
    print("\n--- pLDDT Score Distribution ---")
    plddt_thresholds = [90, 80, 70, 60, 50]
    for threshold in plddt_thresholds:
        count = np.sum(df['repPlddt'] >= threshold)
        percentage = (count / total_representatives) * 100
        print(f"pLDDT >= {threshold:2d}: {count:10,} proteins ({percentage:5.2f}%)")
    print("-" * 40)
    
    # Sequence length distribution analysis
    print("\n--- Sequence Length Distribution ---")
    len_thresholds = [100, 200, 300, 350, 400, 500, 750, 1024]
    for threshold in len_thresholds:
        count = np.sum(df['repLen'] <= threshold)
        percentage = (count / total_representatives) * 100
        print(f"Length <= {threshold:4d}: {count:10,} proteins ({percentage:5.2f}%)")
    print("-" * 40)
    
    # Combined filter analysis (pLDDT + length)
    print("\n--- Combined Filter Analysis (pLDDT >= 70) ---")
    plddt_filter = 70
    combined_len_thresholds = [256, 300, 350, 384, 512, 768, 1024]
    
    base_count = np.sum(df['repPlddt'] >= plddt_filter)
    
    for threshold in combined_len_thresholds:
        count = np.sum(
            (df['repPlddt'] >= plddt_filter) & (df['repLen'] <= threshold)
        )
        percentage_total = (count / total_representatives) * 100
        percentage_base = (count / base_count) * 100 if base_count > 0 else 0
        
        print(
            f"pLDDT >= {plddt_filter} AND Length <= {threshold:4d}: "
            f"{count:10,} proteins "
            f"({percentage_total:5.2f}% of total, "
            f"{percentage_base:5.2f}% of pLDDT>70)"
        )
    print("-" * 40)


def create_manifest(
    metadata_path: Path,
    target_count: int,
    output_path: Path,
    existing_structures_dir: Path,
    plddt_threshold: int = 70,
    max_len: int = 512
) -> None:
    """
    Create a download manifest CSV file with filtered protein structures.
    
    The manifest excludes proteins that already exist in the structures directory
    and filters by pLDDT score and sequence length. Proteins are sorted by
    pLDDT score (descending) to prioritize higher quality structures.
    
    Args:
        metadata_path: Path to the metadata file
        target_count: Target number of proteins to include in manifest
        output_path: Path where manifest CSV will be saved
        existing_structures_dir: Directory containing already downloaded structures
        plddt_threshold: Minimum pLDDT score (default: 70)
        max_len: Maximum sequence length in amino acids (default: 512)
    """
    # Get existing structure IDs to avoid re-downloading
    existing_ids = get_existing_structure_ids(existing_structures_dir)
    
    # Load metadata
    print(f"Loading metadata from: {metadata_path}")
    try:
        df = load_metadata(metadata_path)
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Apply filters: pLDDT score and sequence length
    print(
        f"Filtering: pLDDT >= {plddt_threshold} AND "
        f"Sequence Length <= {max_len}..."
    )
    
    filtered_df = df[
        (df['repPlddt'] >= plddt_threshold) & 
        (df['repLen'] <= max_len)
    ].sort_values(by='repPlddt', ascending=False)
    
    # Exclude already downloaded structures
    if existing_ids:
        print(f"Excluding {len(existing_ids):,} existing IDs from candidates...")
        filtered_df = filtered_df[~filtered_df['repId'].isin(existing_ids)]
    
    total_candidates = len(filtered_df)
    print(f"Found {total_candidates:,} new candidates matching criteria")
    
    # Select target count (or all if fewer available)
    if total_candidates < target_count:
        print(
            f"WARNING: Only {total_candidates:,} candidates available "
            f"(target: {target_count:,}). Using all available."
        )
        final_df = filtered_df
    else:
        final_df = filtered_df.head(target_count)
    
    # Generate download URLs from API (parallelized)
    print(f"Fetching URLs from AlphaFold DB API for {len(final_df):,} proteins...")
    final_df = final_df.copy()
    
    uniprot_ids = final_df['repId'].tolist()
    pdb_urls = [None] * len(uniprot_ids)
    cif_urls = [None] * len(uniprot_ids)
    failed_count = 0
    
    def fetch_urls_with_index(idx_uniprot_id):
        idx, uniprot_id = idx_uniprot_id
        pdb_url, cif_url = get_download_urls_from_api(uniprot_id)
        return idx, pdb_url, cif_url
    
    # Use parallel workers to fetch URLs
    max_workers = 100  # Adjust based on API rate limits
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_urls_with_index, (idx, uid)): idx
            for idx, uid in enumerate(uniprot_ids)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching URLs"):
            try:
                idx, pdb_url, cif_url = future.result()
                if pdb_url and cif_url:
                    pdb_urls[idx] = pdb_url
                    cif_urls[idx] = cif_url
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1
    
    final_df['pdb_url'] = pdb_urls
    final_df['cif_url'] = cif_urls
    
    if failed_count > 0:
        print(f"Warning: Failed to get URLs for {failed_count} proteins")
        # Remove entries without URLs
        final_df = final_df[final_df['pdb_url'].notna() | final_df['cif_url'].notna()]
        print(f"Manifest will contain {len(final_df):,} proteins with valid URLs")
    
    # Save manifest
    manifest_columns = ['repId', 'repPlddt', 'repLen', 'pdb_url', 'cif_url']
    final_df[manifest_columns].to_csv(output_path, index=False)
    
    print(f"\n✓ Manifest created successfully: {output_path}")
    print(f"  Contains {len(final_df):,} NEW proteins ready for download")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Filter AlphaFold DB metadata and create download manifest. "
            "Supports analysis mode and manifest creation mode."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--metadata-file",
        type=Path,
        required=True,
        help="Path to AlphaFold DB metadata TSV.gz file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=['analyze', 'manifest'],
        required=True,
        help="Operation mode: 'analyze' for statistics, 'manifest' to create CSV"
    )
    
    parser.add_argument(
        "--target-count",
        type=int,
        help="Target number of proteins for manifest mode"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file path for manifest mode"
    )
    
    parser.add_argument(
        "--plddt",
        type=int,
        default=70,
        help="Minimum pLDDT score threshold (default: 70)"
    )
    
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Maximum sequence length in amino acids (default: 512)"
    )
    
    parser.add_argument(
        "--existing-structures-dir",
        type=Path,
        help=(
            "Directory containing existing PDB/CIF files "
            "(to exclude from manifest)"
        )
    )
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        analyze_metadata(args.metadata_file)
    
    elif args.mode == 'manifest':
        required_args = [args.target_count, args.output, args.existing_structures_dir]
        if not all(required_args):
            parser.error(
                "manifest mode requires --target-count, --output, "
                "and --existing-structures-dir"
            )
        
        create_manifest(
            metadata_path=args.metadata_file,
            target_count=args.target_count,
            output_path=args.output,
            existing_structures_dir=args.existing_structures_dir,
            plddt_threshold=args.plddt,
            max_len=args.max_len
        )


if __name__ == "__main__":
    main()
