import sys
import os
import warnings
import argparse
import copy
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
import yaml

from pathlib import Path
import torch
torch.manual_seed(42)
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

# Imports from this project
sys.path.append(os.path.realpath("."))

# Add parent directory to path for imports (relative to current file)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.pretraining_model import PretrainingESAModel, PretrainingConfig, create_pretraining_config
from data_loading.cache_to_pyg import OptimizedUniversalQM9Dataset
from data_loading.pretraining_transforms import MaskAtomTypes
from torch_geometric.transforms import Compose

warnings.filterwarnings("ignore")

os.environ["WANDB__SERVICE_WAIT"] = "500"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def create_pretraining_data_transforms(config: PretrainingConfig):
    """Create data transforms for pretraining"""
    transforms = []
    
    # Add MLM transform if MLM is in pretraining tasks
    if "mlm" in config.pretraining_tasks:
        transforms.append(MaskAtomTypes(
            mask_ratio=0.15,
            mask_token=0  # Assuming 0 is the mask token
        ))
    
    return Compose(transforms) if len(transforms) > 0 else None


def load_universal_dataset(config: PretrainingConfig, dataset_name: str, dataset_dir: str):
    """Load universal dataset with optimized caching"""
    print(f"🔄 Loading universal representation dataset: {dataset_name}")
    
    # Create transform
    transforms = create_pretraining_data_transforms(config)
    
    # Load optimized universal dataset (with efficient tensor caching!)
    if dataset_name.upper() == 'QM9':
        # Get max_samples from config
        max_samples = getattr(config, 'max_samples', 50000)  # Default to 50K for training
        print(f"📊 Loading {max_samples} samples from universal cache...")
        
        full_dataset = OptimizedUniversalQM9Dataset(
            root=os.path.join(dataset_dir, 'processed'),
            universal_cache_path=getattr(config, 'universal_cache_path', None),
            max_samples=max_samples,
            molecule_max_atoms=getattr(config, 'molecule_max_atoms', None),
            cutoff_distance=getattr(config, 'cutoff_distance', 5.0),
            max_neighbors=getattr(config, 'max_neighbors', 32),
            transform=transforms
        )
        
    elif dataset_name.upper() == 'LBA':
        from data_loading.cache_to_pyg import OptimizedUniversalLBADataset
        
        # Get max_samples from config
        max_samples = getattr(config, 'max_samples', 50000)  # Default to 50K for training
        print(f"📊 Loading {max_samples} samples from universal cache...")
        
        full_dataset = OptimizedUniversalLBADataset(
            root=os.path.join(dataset_dir, 'processed'),
            universal_cache_path=getattr(config, 'universal_cache_path', None),
            max_samples=max_samples,
            molecule_max_atoms=getattr(config, 'molecule_max_atoms', None),
            cutoff_distance=getattr(config, 'cutoff_distance', 5.0),
            max_neighbors=getattr(config, 'max_neighbors', 32),
            transform=transforms
        )
    elif dataset_name.upper() == 'PDB':
        from data_loading.cache_to_pyg import OptimizedUniversalDataset
        max_samples = getattr(config, 'max_samples', 50000)  # Default to 50K for training
        
        full_dataset = OptimizedUniversalDataset(
            root=os.path.join(dataset_dir, 'processed'),
            universal_cache_path=getattr(config, 'universal_cache_path', None),
            max_samples=max_samples,
            molecule_max_atoms=getattr(config, 'molecule_max_atoms', None),
            cutoff_distance=getattr(config, 'cutoff_distance', 5.0),
            max_neighbors=getattr(config, 'max_neighbors', 32),
            transform=transforms
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return full_dataset


def create_data_loaders(dataset, config: PretrainingConfig):
    """Create train/val/test data loaders"""
    print("🔄 Creating data loaders...")
    
    # Calculate splits
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    
    # Create splits
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    
    print(f"📊 Data splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders
    train_loader = GeometricDataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = GeometricDataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    test_loader = GeometricDataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    print(f"📊 Data loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def train_universal_pretraining(
    config: PretrainingConfig,
    dataset_name: str,
    dataset_dir: str,
    output_dir: str,
    wandb_project: str,
    wandb_run_name: str
):
    """Train universal pretraining model with ESA optimizations"""
    
    print("🚀 Starting Universal Representation Pretraining")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"Tasks: {config.pretraining_tasks}")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    
    # Load dataset
    print("🔄 Loading universal dataset...")
    full_dataset = load_universal_dataset(config, dataset_name, dataset_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(full_dataset, config)
    
    # Create model
    print("🔄 Creating model...")
    model = PretrainingESAModel(config)
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-checkpoint-{epoch:02d}-{train_total_loss:.4f}",
            monitor=config.monitor_loss_name,
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor=config.monitor_loss_name,
            patience=config.early_stopping_patience,
            mode="min",
            verbose=True,
        ),
        TQDMProgressBar(refresh_rate=1),
    ]
    
    # Create wandb logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=wandb_run_name,
        save_dir=output_dir,
    )
    
    # Create trainer with ESA optimizations
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=config.gradient_clip_val,
        precision="bf16-mixed" if config.use_bfloat16 else "32",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        num_sanity_val_steps=getattr(config, 'num_sanity_val_steps', 0),
        log_every_n_steps=1,
        val_check_interval=1.0,
    )
    
    print("🚀 Starting training...")
    print(f"📊 Training details:")
    print(f"   • Total batches per epoch: {len(train_loader)}")
    print(f"   • Validation batches: {len(val_loader)}")
    print(f"   • Log every step: Yes")
    print(f"   • Validation every: {trainer.val_check_interval} steps")
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    print("✅ Training completed!")
    
    return model, trainer


def main():
    """Main function"""
    # Single-threaded optimization (same as official ESA)
    torch.set_num_threads(1)
    
    print("🚀 Starting optimized ESA pretraining script...")
    
    parser = argparse.ArgumentParser(description="Optimized ESA Universal Pretraining")
    parser.add_argument("--config-yaml-path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to use")
    parser.add_argument("--max-epochs", type=int, default=None, help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="./outputs/universal_pretrain", help="Output directory")
    
    args = parser.parse_args()
    print(f"📋 Parsed arguments: {args}")
    
    # Load config from YAML
    print(f"📂 Loading config from: {args.config_yaml_path}")
    with open(args.config_yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    print(f"✅ Config loaded: {config_dict['dataset']}")
    
    # Override config with command line arguments
    if args.max_samples is not None:
        config_dict['max_samples'] = args.max_samples
        print(f"🔬 Using max_samples: {args.max_samples} (for testing)")
    
    if args.max_epochs is not None:
        config_dict['max_epochs'] = args.max_epochs
    
    if args.batch_size is not None:
        config_dict['batch_size'] = args.batch_size
    
    # Create config object
    config = create_pretraining_config(**config_dict)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    model, trainer = train_universal_pretraining(
        config=config,
        dataset_name=config.dataset,
        dataset_dir=config.dataset_download_dir,
        output_dir=args.output_dir,
        wandb_project=config.wandb_project_name,
        wandb_run_name=config.wandb_run_name
    )
    
    print("🎉 Training completed successfully!")


if __name__ == "__main__":
    main()
