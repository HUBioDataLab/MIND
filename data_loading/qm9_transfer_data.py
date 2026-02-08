"""
Data loading utilities for QM9 transfer learning.

Loads QM9 dataset and prepares it for property prediction tasks.
"""

import os
import torch
import numpy as np
import math
from torch_geometric.datasets import QM9
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


# QM9 property names and indices
QM9_PROPERTIES = {
    'mu': 0,           # Dipole moment
    'alpha': 1,        # Isotropic polarizability
    'homo': 2,         # Highest occupied molecular orbital energy
    'lumo': 3,         # Lowest unoccupied molecular orbital energy
    'gap': 4,          # Gap between HOMO and LUMO
    'r2': 5,           # Electronic spatial extent
    'zpve': 6,         # Zero point vibrational energy
    'u0': 7,           # Internal energy at 0K
    'u298': 8,         # Internal energy at 298.15K
    'h298': 9,         # Enthalpy at 298.15K
    'g298': 10,        # Free energy at 298.15K
    'cv': 11,          # Heat capacity at 298.15K
    'u0_atom': 12,     # Atomization energy at 0K
    'u_atom': 13,      # Atomization energy at 298.15K
    'h_atom': 14,      # Atomization enthalpy at 298.15K
    'g_atom': 15,      # Atomization free energy at 298.15K
    'A': 16,           # Rotational constant A
    'B': 17,           # Rotational constant B
    'C': 18,           # Rotational constant C
}


def load_qm9_transfer_data(
    dataset_dir: str = './data/qm9',
    target_name: str = 'homo',
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_samples: Optional[int] = None
) -> Tuple[list, list, list, Optional[StandardScaler]]:
    """
    Load QM9 dataset for transfer learning.
    
    Args:
        dataset_dir: Directory where QM9 dataset is stored
        target_name: Name of the target property (e.g., 'homo', 'lumo', 'gap')
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        train_dataset: List of Data objects for training
        val_dataset: List of Data objects for validation
        test_dataset: List of Data objects for testing
        scaler: StandardScaler fitted on training data (or None)
    """
    print(f"Loading QM9 dataset from {dataset_dir}...")
    
    # Load QM9 dataset
    dataset = QM9(root=dataset_dir)
    
    if max_samples:
        dataset = dataset[:max_samples]
    
    print(f"Loaded {len(dataset)} molecules")
    
    # Get target property index
    if target_name not in QM9_PROPERTIES:
        raise ValueError(
            f"Unknown target property: {target_name}. "
            f"Available: {list(QM9_PROPERTIES.keys())}"
        )
    
    target_idx = QM9_PROPERTIES[target_name]
    print(f"Using target property: {target_name} (index {target_idx})")
    
    # Extract target values and prepare data
    processed_data = []
    target_values = []
    
    for i, data in enumerate(dataset):
        # Extract target property
        if data.y is not None and len(data.y) > 0:
            target_value = data.y[0][target_idx].item()
            
            # Skip NaN or infinite values (use math functions for Python float)
            if not (math.isnan(target_value) or math.isinf(target_value)):
                # Create new Data object with target
                new_data = Data(
                    z=data.z.clone(),
                    pos=data.pos.clone(),
                    edge_index=data.edge_index.clone(),
                    edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
                    y=torch.tensor([target_value], dtype=torch.float32),
                    idx=i
                )
                
                processed_data.append(new_data)
                target_values.append(target_value)
    
    print(f"Processed {len(processed_data)} valid molecules")
    
    # Split dataset
    total = len(processed_data)
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    
    # Shuffle indices
    indices = torch.randperm(total).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = [processed_data[i] for i in train_indices]
    val_dataset = [processed_data[i] for i in val_indices]
    test_dataset = [processed_data[i] for i in test_indices]
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Fit scaler on training data
    train_targets = np.array([data.y.item() for data in train_dataset]).reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(train_targets)
    
    # Transform all targets
    for data in train_dataset:
        data.y = torch.tensor(
            scaler.transform([[data.y.item()]])[0],
            dtype=torch.float32
        )
    
    for data in val_dataset:
        data.y = torch.tensor(
            scaler.transform([[data.y.item()]])[0],
            dtype=torch.float32
        )
    
    for data in test_dataset:
        data.y = torch.tensor(
            scaler.transform([[data.y.item()]])[0],
            dtype=torch.float32
        )
    
    print(f"Target statistics (original): mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
    
    # Add max_node_global and max_edge_global attributes (required for ESA)
    train_dataset = add_max_node_edge_global(train_dataset)
    val_dataset = add_max_node_edge_global(val_dataset)
    test_dataset = add_max_node_edge_global(test_dataset)
    
    return train_dataset, val_dataset, test_dataset, scaler


def add_max_node_edge_global(dataset):
    """
    Add max_node_global and max_edge_global attributes to dataset.
    Required for ESA models.
    """
    max_nodes = max([data.z.size(0) for data in dataset])
    max_edges = max([data.edge_index.size(1) for data in dataset])
    
    for data in dataset:
        data.max_node_global = torch.tensor([max_nodes])
        data.max_edge_global = torch.tensor([max_edges])
    
    return dataset
