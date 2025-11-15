#!/usr/bin/env python3
"""
Download protein structure files (PDB/CIF) from URLs in a manifest CSV.

Usage:
    python data/protein_pipeline/2_download_pdbs_from_manifest.py \
        --manifest-file ../data/proteins/afdb_clusters/manifest_hq_40k.csv \
        --structures-outdir ../data/proteins/raw_structures_hq_40k \
        --workers 16
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
}


def is_html_error(content: str) -> bool:
    """Check if content is an HTML error page."""
    lower = content.strip().lower()
    return lower.startswith("<!doctype html") or lower.startswith("<html")


def download_text(url: str, timeout: int = 60, max_retries: int = 3) -> Optional[str]:
    """Download text from URL with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            if r.status_code == 200:
                text = r.text
                if text and not is_html_error(text):
                    return text
        except requests.exceptions.RequestException:
            pass
        if attempt < max_retries:
            time.sleep(1.25 * attempt)
    return None


def download_one_structure(
    uniprot_id: str,
    pdb_url: str,
    cif_url: str,
    out_dir: Path
) -> Tuple[str, str]:
    """Download one protein structure (PDB first, then CIF fallback)."""
    pdb_path = out_dir / f"{uniprot_id}.pdb"
    cif_path = out_dir / f"{uniprot_id}.cif"
    
    if pdb_path.exists() or cif_path.exists():
        return uniprot_id, "SKIPPED_EXIST"
    
    # Try PDB first
    text = download_text(pdb_url)
    if text:
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_path.write_text(text, encoding='utf-8')
        return uniprot_id, "PDB_OK"
    
    # Fallback to CIF
    text = download_text(cif_url)
    if text:
        cif_path.parent.mkdir(parents=True, exist_ok=True)
        cif_path.write_text(text, encoding='utf-8')
        return uniprot_id, "CIF_OK"
    
    return uniprot_id, "FAILED"


def parallel_download_from_manifest(
    manifest_csv: Path,
    out_dir: Path,
    max_workers: int = 8
) -> Dict[str, int]:
    """Download structures from manifest CSV in parallel."""
    try:
        df = pd.read_csv(manifest_csv)
    except Exception as e:
        print(f"ERROR: Failed to read manifest file: {manifest_csv}\n{e}")
        return {}
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(df):,} structures using {max_workers} workers...")
    
    stats = {"PDB_OK": 0, "CIF_OK": 0, "FAILED": 0, "SKIPPED_EXIST": 0, "TOTAL": len(df)}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_one_structure, row.repId, row.pdb_url, row.cif_url, out_dir): row.repId
            for row in df.itertuples(index=False)
        }
        
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Downloading structures")
        for future in pbar:
            uid, status = future.result()
            stats[status] = stats.get(status, 0) + 1
            pbar.set_postfix(
                PDB=stats['PDB_OK'],
                CIF=stats['CIF_OK'],
                FAIL=stats['FAILED'],
                SKIP=stats['SKIPPED_EXIST']
            )
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download PDB/CIF structures from manifest CSV in parallel."
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        required=True,
        help="Path to manifest CSV file"
    )
    parser.add_argument(
        "--structures-outdir",
        type=Path,
        required=True,
        help="Directory to save downloaded files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Starting parallel download...")
    start_time = time.time()
    
    stats = parallel_download_from_manifest(
        manifest_csv=args.manifest_file,
        out_dir=args.structures_outdir,
        max_workers=args.workers
    )
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Download completed")
    print("=" * 60)
    print(f"Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("Results:")
    print(json.dumps(stats, indent=2))
    print(f"\nFiles saved to: {args.structures_outdir}")


if __name__ == "__main__":
    main()
