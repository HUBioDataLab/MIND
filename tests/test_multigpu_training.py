#!/usr/bin/env python3
"""
Multi-GPU Integration Test for MIND Pretraining Model.
Tests DDP training on GPUs 1 and 2 using synthetic molecular graph data.
"""

import os
import sys
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from pytorch_lightning.strategies import DDPStrategy

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.pretraining_model import PretrainingESAModel, PretrainingConfig, create_pretraining_config

class SyntheticMolecularDataset(Dataset):
    """Generates synthetic 3D molecular graph data for fast testing."""
    def __init__(self, num_samples=100, num_nodes=20):
        super().__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes

    def len(self):
        return self.num_samples

    def get(self, idx):
        # Create random 3D positions, atomic numbers, and edge indices
        pos = torch.randn(self.num_nodes, 3)
        z = torch.randint(1, 10, (self.num_nodes,))

        # Simple fully connected or random radius graph edges
        row = torch.randint(0, self.num_nodes, (self.num_nodes * 4,))
        col = torch.randint(0, self.num_nodes, (self.num_nodes * 4,))
        edge_index = torch.stack([row, col], dim=0)

        data = Data(x=z, z=z, pos=pos, edge_index=edge_index)
        data.dataset_type = "qm9"
        return data

def run_multigpu_test():
    print("🚀 Starting Multi-GPU DDP Test on GPUs 1 & 2...")

    # Set up synthetic datasets
    train_dataset = SyntheticMolecularDataset(num_samples=128, num_nodes=16)
    val_dataset = SyntheticMolecularDataset(num_samples=32, num_nodes=16)

    train_loader = GeometricDataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=8, shuffle=False)

    # Pretraining configuration
    config = create_pretraining_config(
        graph_dim=128,
        hidden_dims=[128, 128],
        num_heads=[4, 4],
        layer_types=["S", "S"],
        pretraining_tasks=["long_range_distance", "short_range_distance", "mlm"],
        task_weights={"long_range_distance": 1.0, "short_range_distance": 1.0, "mlm": 1.0},
        max_epochs=2,
        devices=2,
        strategy="ddp",
        use_bfloat16=False,
    )

    # Initialize model
    model = PretrainingESAModel(config)

    # Multi-GPU Trainer setup
    ddp_strategy = DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="gpu",
        devices=2,
        strategy=ddp_strategy,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=1,
    )

    print("⚡ Fitting PretrainingESAModel with DDP across 2 GPUs...")
    trainer.fit(model, train_loader, val_loader)

    print("🎉 Multi-GPU DDP Test Completed Successfully on Ada GPUs 1 & 2!")

if __name__ == "__main__":
    run_multigpu_test()
