#!/usr/bin/env python3
"""
LBA Dataset Downloader

Downloads the raw LBA dataset with sequence identity split.

Reference: https://zenodo.org/record/4914718
"""

import os
import subprocess
import urllib.request
from pathlib import Path

DOWNLOAD_DIR = "./data/LBA"
DOWNLOAD_URL = "https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-30.tar.gz?download=1"

def download_lba():
    """Download LBA dataset."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    tar_file = os.path.join(DOWNLOAD_DIR, "LBA-split-by-sequence-identity-30.tar.gz")
    
    print(f"Downloading LBA dataset to {DOWNLOAD_DIR}...")
    
    # Download with progress
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) / total_size)
            bar_length = 50
            filled_length = int(bar_length * downloaded // total_size)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # Format file sizes
            def format_size(size):
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size < 1024.0:
                        return f"{size:.1f} {unit}"
                    size /= 1024.0
                return f"{size:.1f} TB"
            
            downloaded_str = format_size(downloaded)
            total_str = format_size(total_size)
            
            print(f'\rDownloading: |{bar}| {percent:.1f}% ({downloaded_str}/{total_str})', end='')
    
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, tar_file, reporthook=show_progress)
        print("\nExtracting dataset...")
    except Exception as e:
        print(f"Download failed: {e}")
        return None
    
    # Extract
    subprocess.run(f'tar zxvf {tar_file} -C {DOWNLOAD_DIR}', shell=True, check=True)
    
    # Clean up
    os.remove(tar_file)
    
    print("LBA dataset downloaded successfully")
    return DOWNLOAD_DIR

def load_lba():
    """Load LBA dataset from LMDB files."""
    data_path = os.path.join(DOWNLOAD_DIR, "split-by-sequence-identity-30", "data", "train")
    
    if not os.path.exists(data_path):
        print(f"LBA data path not found: {data_path}")
        return None
    
    print(f"LBA dataset available at: {data_path}")
    print("Note: Use ATOM3D's LMDBDataset to load the data")
    print("Example: from atom3d.datasets import LMDBDataset; dataset = LMDBDataset(data_path)")
    
    return data_path

def main():
    download_path = download_lba()
    if download_path:
        print("Download complete")
    
    # Uncomment to test loading the dataset
    # load_lba()

if __name__ == "__main__":
    main()


