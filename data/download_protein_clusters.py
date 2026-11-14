# data/download_protein_clusters.py

import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import os

# URL and default file name of the file to download
METADATA_URL = "https://afdb-cluster.steineggerlab.workers.dev/2-repId_isDark_nMem_repLen_avgLen_repPlddt_avgPlddt_LCAtaxId.tsv.gz"
DEFAULT_FILENAME = "representatives_metadata.tsv.gz"

def download_file(url: str, out_path: Path):
    """
    Stream a file from the given URL to ``out_path`` with a progress bar.
    Skip the download if the file already exists with the expected size.
    """
    print(f"Download target: {out_path}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # raises an exception on HTTP 4xx/5xx
        
        total_size = int(response.headers.get('content-length', 0))

        # Skip the download if the file exists and its size matches
        if out_path.exists() and out_path.stat().st_size == total_size:
            print(f"File '{out_path.name}' already exists with the expected size. Download skipped.")
            return

        print(f"Downloading: {url}")
        with open(out_path, 'wb') as f, tqdm(
            desc=out_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024*8):
                size = f.write(chunk)
                pbar.update(size)
        
        print(f"Download complete: {out_path}")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: could not download the file. Check your network connection or the URL.\n{e}")
        # Remove the partial file if the download failed
        if out_path.exists():
            out_path.unlink()
    except Exception as e:
        print(f"Unexpected error: {e}")
        if out_path.exists():
            out_path.unlink()


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Download the precomputed cluster-representative metadata from Foldseek."
    )
    # Default output directory: <parent of the repository>/data/proteins/afdb_clusters
    default_path = Path(__file__).resolve().parents[2] / "data" / "proteins" / "afdb_clusters"
    parser.add_argument(
        "--outdir",
        type=Path,
        default=default_path,
        help=f"Directory for the downloaded file. Default: {default_path}"
    )
    args = parser.parse_args()

    # Create the output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Full output file path
    output_file_path = args.outdir / DEFAULT_FILENAME
    
    # Start the download
    download_file(METADATA_URL, output_file_path)


if __name__ == "__main__":
    main()