#!/usr/bin/env python3
"""
Uni-Mol 209M Molecular Conformation Dataset Downloader

This script downloads the Uni-Mol molecular pretraining dataset which contains
209M molecular 3D conformations (114.76GB).

Dataset Details:
- Size: 114.76GB
- Content: 209M molecular 3D conformations
- Format: LMDB (Lightning Memory-Mapped Database)
- Use: Molecular pretraining for Uni-Mol framework

Download Links from Uni-Mol paper:
- Molecular pretrain: https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/pretrain/ligands.tar.gz
"""

import os
import requests
import tarfile
from tqdm import tqdm
import time

# Dataset URLs
DATASET_URLS = {
    "molecular_pretrain": {
        "url": "https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/pretrain/ligands.tar.gz",
        "size_gb": 114.76,
        "description": "209M molecular 3D conformations for pretraining"
    },
    "pocket_pretrain": {
        "url": "https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/pretrain/pockets.tar.gz", 
        "size_gb": 8.585,
        "description": "3M protein pocket data for pretraining"
    },
    "molecular_property": {
        "url": "https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/molecular_property_prediction.tar.gz",
        "size_gb": 3.506,
        "description": "Molecular property prediction dataset"
    },
    "molecular_conformation": {
        "url": "https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/conformation_generation.tar.gz",
        "size_gb": 8.331,
        "description": "Molecular conformation generation dataset"
    },
    "pocket_property": {
        "url": "https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/pocket_property_prediction.tar.gz",
        "size_gb": 0.455,
        "description": "Pocket property prediction dataset"
    },
    "protein_ligand_binding": {
        "url": "https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/protein_ligand_binding_pose_prediction.tar.gz",
        "size_gb": 0.263,
        "description": "Protein-ligand binding pose prediction dataset"
    }
}

# Download settings
DOWNLOAD_DIR = "./unimol_data"
CHUNK_SIZE = 8192  # 8KB chunks for download progress

def get_file_size(url):
    """Get file size from URL headers."""
    try:
        response = requests.head(url, allow_redirects=True)
        response.raise_for_status()
        return int(response.headers.get('content-length', 0))
    except:
        return 0

def download_file(url, filepath, description=""):
    """Download file with progress bar."""
    print(f"\n📥 Downloading: {description}")
    print(f"🔗 URL: {url}")
    print(f"💾 Save to: {filepath}")
    
    # Get file size for progress bar
    file_size = get_file_size(url)
    if file_size > 0:
        print(f"📊 File size: {file_size / (1024**3):.2f} GB")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Download with progress bar
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ Download completed: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def extract_tar_gz(filepath, extract_dir):
    """Extract tar.gz file."""
    print(f"\n📦 Extracting: {filepath}")
    print(f"📁 Extract to: {extract_dir}")
    
    try:
        with tarfile.open(filepath, 'r:gz') as tar:
            # Get total number of files for progress
            members = tar.getmembers()
            
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    tar.extract(member, extract_dir)
                    pbar.update(1)
        
        print(f"✅ Extraction completed: {extract_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def main():
    """Main function to download Uni-Mol datasets."""
    print("🚀 Uni-Mol Dataset Downloader")
    print("=" * 50)
    
    # Show available datasets
    print("\n📋 Available datasets:")
    for i, (name, info) in enumerate(DATASET_URLS.items(), 1):
        print(f"  {i}. {name.replace('_', ' ').title()}")
        print(f"     Size: {info['size_gb']:.2f} GB")
        print(f"     Description: {info['description']}")
        print()
    
    # Ask user which dataset to download
    print("Which dataset would you like to download?")
    print("Options:")
    print("  1. molecular_pretrain (209M conformations, 114.76GB) - RECOMMENDED")
    print("  2. pocket_pretrain (3M pockets, 8.59GB)")
    print("  3. molecular_property (3.51GB)")
    print("  4. molecular_conformation (8.33GB)")
    print("  5. pocket_property (0.46GB)")
    print("  6. protein_ligand_binding (0.26GB)")
    print("  7. all (all datasets)")
    
    choice = input("\nEnter your choice (1-7): ").strip()
    
    # Map choice to dataset
    choice_map = {
        "1": ["molecular_pretrain"],
        "2": ["pocket_pretrain"], 
        "3": ["molecular_property"],
        "4": ["molecular_conformation"],
        "5": ["pocket_property"],
        "6": ["protein_ligand_binding"],
        "7": list(DATASET_URLS.keys())
    }
    
    if choice not in choice_map:
        print("❌ Invalid choice. Exiting.")
        return
    
    selected_datasets = choice_map[choice]
    
    # Download selected datasets
    for dataset_name in selected_datasets:
        dataset_info = DATASET_URLS[dataset_name]
        
        # Create filepath
        filename = os.path.basename(dataset_info["url"])
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            print(f"\n⚠️  File already exists: {filepath}")
            overwrite = input("Do you want to overwrite? (y/N): ").strip().lower()
            if overwrite != 'y':
                print(f"⏩ Skipping {dataset_name}")
                continue
        
        # Download file
        success = download_file(
            dataset_info["url"], 
            filepath, 
            dataset_info["description"]
        )
        
        if success:
            # Extract file
            extract_dir = os.path.join(DOWNLOAD_DIR, dataset_name)
            extract_success = extract_tar_gz(filepath, extract_dir)
            
            if extract_success:
                print(f"🎉 Dataset '{dataset_name}' ready at: {extract_dir}")
                
                # Optionally remove the tar.gz file to save space
                remove_tar = input(f"\nRemove {filename} to save space? (y/N): ").strip().lower()
                if remove_tar == 'y':
                    os.remove(filepath)
                    print(f"🗑️  Removed: {filepath}")
            else:
                print(f"❌ Failed to extract {dataset_name}")
        else:
            print(f"❌ Failed to download {dataset_name}")
    
    print("\n🎉 Download process completed!")
    print(f"📁 All datasets are in: {DOWNLOAD_DIR}")

if __name__ == "__main__":
    main() 