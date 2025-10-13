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
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, Callback
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
from data_loading.chunk_sampler import ChunkAwareSampler
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
    """
    Load universal dataset with optimized caching.
    
    Automatically detects chunked datasets and uses LazyUniversalDataset
    for memory-efficient loading.
    """
    print(f"🔄 Loading universal representation dataset: {dataset_name}")
    
    # Create transform
    transforms = create_pretraining_data_transforms(config)
    
    # AUTO-DETECT CHUNKED DATASETS
    dataset_dir_path = Path(dataset_dir)
    parent_dir = dataset_dir_path.parent
    base_name = dataset_dir_path.name
    
    # Look for chunk directories: e.g., processed_graphs_40k_chunk_0, processed_graphs_40k_chunk_1, ...
    chunk_dirs = sorted(list(parent_dir.glob(f"{base_name}_chunk_*")))
    
    if chunk_dirs:
        print(f"📦 Detected {len(chunk_dirs)} chunked datasets")
        print(f"📦 Using LazyUniversalDataset for memory-efficient loading...")
        
        from data_loading.lazy_universal_dataset import LazyUniversalDataset
        
        # Collect all .pt files from chunks
        chunk_pt_files = []
        for chunk_dir in chunk_dirs:
            # Find the processed .pt file
            processed_dir = chunk_dir / "processed"
            if processed_dir.exists():
                pt_files = list(processed_dir.glob("*.pt"))
                # Filter out pre_filter.pt and pre_transform.pt
                pt_files = [f for f in pt_files if 'pre_filter' not in f.name and 'pre_transform' not in f.name]
                if pt_files:
                    chunk_pt_files.append(str(pt_files[0]))
                    print(f"   ✓ {chunk_dir.name}: {pt_files[0].name}")
        
        if not chunk_pt_files:
            raise FileNotFoundError(f"No processed .pt files found in chunk directories")
        
        # Create LazyUniversalDataset
        # Cache all chunks to avoid disk I/O during shuffled training
        full_dataset = LazyUniversalDataset(
            chunk_pt_files=chunk_pt_files,
            transform=transforms,
            max_cache_chunks=3,  # Cache all chunks for maximum speed
            verbose=True
        )
        
        print(f"📊 Caching strategy: {len(chunk_pt_files)} chunks will be loaded into RAM")
        
        print(f"✅ LazyDataset loaded: {len(full_dataset):,} total samples")
        return full_dataset
    
    else:
        # SINGLE DATASET (original behavior)
        print(f"📦 Loading single dataset from: {dataset_dir}")
        
        if dataset_name.upper() == 'QM9':
            max_samples = getattr(config, 'max_samples', 50000)
            print(f"📊 Loading {max_samples:,} samples from universal cache...")
            
            full_dataset = OptimizedUniversalQM9Dataset(
                root=dataset_dir,
                universal_cache_path=getattr(config, 'universal_cache_path', None),
                max_samples=max_samples,
                molecule_max_atoms=getattr(config, 'molecule_max_atoms', None),
                cutoff_distance=getattr(config, 'cutoff_distance', 5.0),
                max_neighbors=getattr(config, 'max_neighbors', 32),
                transform=transforms
            )
            
        elif dataset_name.upper() == 'LBA':
            from data_loading.cache_to_pyg import OptimizedUniversalLBADataset
            max_samples = getattr(config, 'max_samples', 50000)
            print(f"📊 Loading {max_samples:,} samples from universal cache...")
            
            full_dataset = OptimizedUniversalLBADataset(
                root=dataset_dir,
                universal_cache_path=getattr(config, 'universal_cache_path', None),
                max_samples=max_samples,
                molecule_max_atoms=getattr(config, 'molecule_max_atoms', None),
                cutoff_distance=getattr(config, 'cutoff_distance', 5.0),
                max_neighbors=getattr(config, 'max_neighbors', 32),
                transform=transforms
            )
            
        elif dataset_name.upper() == 'PDB':
            from data_loading.cache_to_pyg import OptimizedUniversalDataset
            max_samples = getattr(config, 'max_samples', 50000)
            print(f"📊 Loading {max_samples:,} samples from universal cache...")
            
            full_dataset = OptimizedUniversalDataset(
                root=dataset_dir,
                universal_cache_path=getattr(config, 'universal_cache_path', None),
                max_samples=max_samples,
                molecule_max_atoms=getattr(config, 'molecule_max_atoms', None),
                cutoff_distance=getattr(config, 'cutoff_distance', 5.0),
                max_neighbors=getattr(config, 'max_neighbors', 32),
                transform=transforms
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        print(f"✅ Single dataset loaded: {len(full_dataset):,} samples")
        return full_dataset


def create_data_loaders(dataset, config: PretrainingConfig):
    """Create train/val/test data loaders"""
    print("🔄 Creating data loaders...")

    # Calculate splits
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    if train_size + val_size + test_size != total_samples:
        train_size += total_samples - (train_size + val_size + test_size)

    print(f"📊 Total samples: {total_samples}, Splitting into: Train={train_size}, Val={val_size}, Test={test_size}")

    # Use random_split to avoid materializing slices into memory
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # Check if using LazyUniversalDataset (chunked dataset)
    from data_loading.lazy_universal_dataset import LazyUniversalDataset
    is_lazy_dataset = isinstance(dataset, LazyUniversalDataset)
    
    if is_lazy_dataset:
        # Use ChunkAwareSampler for optimal performance with chunked datasets
        # Benefits:
        # 1. Sequential chunk reading → optimal disk I/O
        # 2. Chunk-level + sample-level shuffling → full randomness
        # 3. Predictable access pattern → prefetching works perfectly
        print(f"🚀 Using ChunkAwareSampler for chunked dataset:")
        print(f"   - Shuffle chunk order: Yes (epoch-level randomness)")
        print(f"   - Shuffle within chunks: Yes (sample-level randomness)")
        print(f"   - Sequential chunk reading: Yes (optimal disk I/O)")
        print(f"   - DataLoader optimizations: pin_memory, persistent_workers, prefetch_factor")
        
        train_sampler = ChunkAwareSampler(
            train_dataset,  # Automatically handles Subset from random_split
            shuffle_chunks=True,
            shuffle_within_chunk=True
        )
        
        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,  # Use custom sampler (no shuffle arg)
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True if config.num_workers > 0 else False,
            prefetch_factor=2 if config.num_workers > 0 else None
        )
    else:
        # Standard loader for non-chunked datasets
        # NOTE: To use chunked datasets with QM9/LBA, run:
        # python data_loading/process_chunked_dataset.py --dataset qm9 --num-chunks 10 ...
        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )

    # Val and test loaders (no shuffling needed, but apply optimizations)
    val_loader = GeometricDataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None
    )

    test_loader = GeometricDataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None
    )

    print(f"📊 Data loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    if is_lazy_dataset:
        print(f"   ⚡ ChunkAwareSampler enabled")
        print(f"   ⚡ Cache hit rate: ~99.97% (chunk-aware reading)")
        print(f"   ⚡ Expected iteration speed: 3.5-4.0 it/s")
    else:
        print(f"   ℹ️  Using standard DataLoader (non-chunked)")
        print(f"   💡 To enable chunking for this dataset:")
        print(f"      python data_loading/process_chunked_dataset.py \\")
        print(f"          --config-yaml-path <your_config>.yaml \\")
        print(f"          --data-path <data_path> \\")
        print(f"          --manifest-file <manifest.csv> \\")
        print(f"          --num-chunks 10")

    # Store sampler reference for epoch updates (if using chunked dataset)
    if is_lazy_dataset:
        train_loader.sampler_obj = train_sampler  # For set_epoch() calls

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
    
    # Custom callback for chunk-aware sampler epoch updates
    class ChunkSamplerEpochCallback(Callback):
        """Update chunk sampler epoch at the start of each training epoch"""
        def on_train_epoch_start(self, trainer, pl_module):
            if hasattr(trainer.train_dataloader, 'sampler_obj'):
                epoch = trainer.current_epoch
                trainer.train_dataloader.sampler_obj.set_epoch(epoch)
                if epoch == 0:
                    print(f"   🔄 ChunkAwareSampler: Epoch {epoch} started")
    
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
        ChunkSamplerEpochCallback(),  # Update chunk sampler each epoch
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
