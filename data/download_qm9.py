#!/usr/bin/env python3
"""
QM9 Dataset Downloader

Downloads the raw QM9 dataset using PyTorch Geometric.

Reference: https://pytorch-geometric.readthedocs.io/en/2.5.0/generated/torch_geometric.datasets.QM9.html
"""

import os
from torch_geometric.datasets import QM9

DOWNLOAD_DIR = "./data/qm9"

def download_qm9():
    """Download QM9 dataset."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    print(f"Downloading QM9 dataset to {DOWNLOAD_DIR}...")
    dataset = QM9(root=DOWNLOAD_DIR)
    
    print(f"Downloaded {len(dataset)} molecules")
    return dataset

def load_qm9():
    """Load QM9 dataset from .pt files."""
    dataset = QM9(root=DOWNLOAD_DIR)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"First molecule:")
    
    # Get first molecule
    mol = dataset[0]
    print(f"  Nodes: {mol.num_nodes}")
    print(f"  Edges: {mol.num_edges}")
    print(f"  Features: {mol.x.shape}")
    print(f"  Positions: {mol.pos.shape}")
    print(f"  Properties: {mol.y.shape}")
    
    return dataset

def main():
    dataset = download_qm9()
    print("Download complete")
    # load_qm9()

if __name__ == "__main__":
    main()