#!/usr/bin/env python3
"""
COCONUT Dataset Downloader

Downloads the COCONUT (COlleCtion of Open Natural ProdUcTs) dataset.
COCONUT is a comprehensive database of natural products containing over 400,000
compounds from various sources.

Reference: https://coconut.naturalproducts.net/
Dataset URL: https://coconut.s3.uni-jena.de/prod/downloads/2025-09/coconut_sdf_3d-09-2025.zip
"""

import os
import urllib.request
import zipfile
from pathlib import Path

DOWNLOAD_DIR = "./data/COCONUT"
DOWNLOAD_URL = "https://coconut.s3.uni-jena.de/prod/downloads/2025-09/coconut_sdf_3d-09-2025.zip"

def download_coconut():
    """Download COCONUT dataset."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    zip_file = os.path.join(DOWNLOAD_DIR, "coconut_sdf_3d-09-2025.zip")
    
    print(f"Downloading COCONUT dataset to {DOWNLOAD_DIR}...")
    print(f"Source: {DOWNLOAD_URL}")
    
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
        # Check if file already exists
        if os.path.exists(zip_file):
            print(f"\nFile already exists: {zip_file}")
            overwrite = input("Do you want to overwrite? (y/N): ").strip().lower()
            if overwrite != 'y':
                print("Skipping download...")
                return extract_coconut(zip_file)
        
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_file, reporthook=show_progress)
        print("\nDownload completed successfully!")
        
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return None
    
    # Extract the ZIP file
    return extract_coconut(zip_file)

def extract_coconut(zip_file):
    """Extract COCONUT ZIP file."""
    print(f"\nExtracting dataset from {zip_file}...")
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get list of files in the ZIP
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            print(f"Found {total_files} files in the archive")
            
            # Extract with progress
            for i, file_name in enumerate(file_list, 1):
                zip_ref.extract(file_name, DOWNLOAD_DIR)
                
                # Show progress every 100 files or for last file
                if i % 100 == 0 or i == total_files:
                    percent = (i * 100) / total_files
                    bar_length = 50
                    filled_length = int(bar_length * i // total_files)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    print(f'\rExtracting: |{bar}| {percent:.1f}% ({i}/{total_files} files)', end='')
            
            print(f"\nExtraction completed successfully!")
            
            # Show extracted contents
            extracted_path = DOWNLOAD_DIR
            print(f"\nDataset extracted to: {extracted_path}")
            
            # List main contents
            contents = os.listdir(extracted_path)
            print("Contents:")
            for item in contents:
                item_path = os.path.join(extracted_path, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    size_str = format_file_size(size)
                    print(f"  📄 {item} ({size_str})")
                elif os.path.isdir(item_path):
                    file_count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                    print(f"  📁 {item}/ ({file_count} files)")
            
            return extracted_path
            
    except Exception as e:
        print(f"\nExtraction failed: {e}")
        return None

def format_file_size(size):
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def cleanup_download(zip_file):
    """Remove the downloaded ZIP file to save space."""
    if os.path.exists(zip_file):
        remove_zip = input(f"\nRemove {os.path.basename(zip_file)} to save space? (y/N): ").strip().lower()
        if remove_zip == 'y':
            os.remove(zip_file)
            print(f"🗑️  Removed: {zip_file}")

def load_coconut():
    """Load and inspect COCONUT dataset."""
    dataset_path = DOWNLOAD_DIR
    
    if not os.path.exists(dataset_path):
        print(f"COCONUT data path not found: {dataset_path}")
        print("Please run download_coconut() first.")
        return None
    
    print(f"\nCOCONUT dataset available at: {dataset_path}")
    
    # Look for SDF files
    sdf_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.sdf') or file.endswith('.sdf.gz'):
                sdf_files.append(os.path.join(root, file))
    
    if sdf_files:
        print(f"\nFound {len(sdf_files)} SDF file(s):")
        for sdf_file in sdf_files:
            size = os.path.getsize(sdf_file)
            size_str = format_file_size(size)
            print(f"  📄 {os.path.relpath(sdf_file, dataset_path)} ({size_str})")
        
        print("\nTo load the dataset, you can use RDKit or other chemistry libraries:")
        print("Example:")
        print("  from rdkit import Chem")
        print("  supplier = Chem.SDMolSupplier('path/to/coconut.sdf')")
        print("  for mol in supplier:")
        print("      if mol is not None:")
        print("          # Process molecule")
        print("          pass")
    else:
        print("No SDF files found in the dataset.")
    
    return dataset_path

def main():
    """Main function to download COCONUT dataset."""
    print("🥥 COCONUT Dataset Downloader")
    print("=" * 50)
    print("COlleCtion of Open Natural ProdUcTs")
    print("A comprehensive database of natural products")
    print()
    
    # Download and extract dataset
    extract_path = download_coconut()
    
    if extract_path:
        print("\n✅ Dataset download and extraction completed successfully!")
        
        # Cleanup option
        zip_file = os.path.join(DOWNLOAD_DIR, "coconut_sdf_3d-09-2025.zip")
        cleanup_download(zip_file)
        
        # Show dataset info
        load_coconut()
    else:
        print("\n❌ Dataset download failed!")

if __name__ == "__main__":
    main()
