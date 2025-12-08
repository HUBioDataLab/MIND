# Code Documentation: MIND Training Pipeline

## Overview
This document provides detailed descriptions of the training pipeline code, focusing on multi-domain pretraining with atom-aware batching and efficient data loading.

---

## Training Flow Diagram

```
Training Command:
CUDA_VISIBLE_DEVICES=6 python -m core.train_pretrain \
    --config-yaml-path core/pretraining_config_multidomain.yaml

1. MAIN ENTRY POINT
   core/train_pretrain.py::main()
   ├─> Parse command line arguments
   ├─> Load YAML config (pretraining_config_multidomain.yaml)
   └─> Call train_universal_pretraining()
   │
   ▼
2. TRAINING ORCHESTRATION
   core/train_pretrain.py::train_universal_pretraining()
   ├─> Set random seeds
   ├─> Load dataset (load_universal_dataset)
   ├─> Create data loaders (create_data_loaders)
   ├─> Create model (PretrainingESAModel)
   ├─> Setup PyTorch Lightning Trainer
   └─> Start training (trainer.fit)
   │
   ├─────────────────────────────────┐
   ▼                                 ▼
3. DATASET LOADING                    4. DATA LOADER CREATION
   load_universal_dataset()            create_data_loaders()
   
   Multi-Domain Mode:                 Multi-Domain + Dynamic Batching:
   ├─> Detect chunk dirs              ├─> ImprovedDynamicBatchSampler (default)
   ├─> Collect .pt files              │   - Metadata-only (fast)
   ├─> LazyUniversalDataset           │   - Simplified proportional sampling
   │   - load_metadata=True           │   - Cache-optimized ordering
   │   - max_cache_chunks=10          │   - max_atoms_per_batch=25000
   │   - metadata_search_paths        │   OR DynamicChunkAwareBatchSampler
   └─> Return dataset                 │   (if use_improved_sampler=False)
                                      │
                                      Single-Domain + Dynamic Batching:
   Single-Domain Mode:                ├─> ImprovedDynamicBatchSampler
   ├─> Detect chunks                  │   - Same as multi-domain
   ├─> LazyUniversalDataset           │   - enable_cross_modal_batches=False
   │   - load_metadata=False          │
   │   - max_cache_chunks=3           │ Single-Domain (no dynamic batching):
   └─> Return dataset                 ├─> ChunkAwareSampler
   │                                   │   - Sequential chunk reading
   │                                   │   - Chunk-level shuffling
   │                                   │   - Fixed batch_size=32
   │                                   └─> GeometricDataLoader
   │                                   │
   └───────────────────────────────────┘
   │
   ▼
5. MODEL ARCHITECTURE
   core/pretraining_model.py::PretrainingESAModel
   
   ├─> UniversalMolecularEncoder
   │   ├─> Atomic number embedding
   │   ├─> Period embedding
   │   ├─> 3D geometric features (GaussianLayer)
   │   └─> Position encodings (optional)
   
   ├─> ESA Backbone (esa/masked_layers.py::ESA)
   │   ├─> Multi-head attention (flash_varlen)
   │   ├─> Set attention blocks (SAB)
   │   ├─> Multi-head attention blocks (MAB)
   │   ├─> Pooling by multi-head attention (PMA)
   │   └─> MLP layers (optional)
   
   └─> PretrainingTasks
       ├─> Long-range distance prediction
       ├─> Short-range distance prediction
       └─> Masked language modeling (MLM)
   │
   ▼
6. TRAINING LOOP (PyTorch Lightning)
   
   For each epoch:
   ├─> For each batch (from DataLoader):
   │   ├─> training_step()
   │   │   ├─> forward()
   │   │   │   ├─> Encode nodes (UniversalMolecularEncoder)
   │   │   │   ├─> Apply attention (ESA backbone)
   │   │   │   └─> Return graph & node embeddings
   │   │   ├─> Compute losses (_compute_pretraining_losses)
   │   │   │   ├─> Long-range distance loss
   │   │   │   ├─> Short-range distance loss
   │   │   │   └─> MLM loss
   │   │   ├─> Backward pass (automatic)
   │   │   └─> Log metrics (wandb)
   │   │
   │   └─> validation_step() (at val_check_interval)
   │       ├─> Same forward pass
   │       ├─> Compute validation losses
   │       └─> Log validation metrics
   │
   └─> Callbacks:
       ├─> ModelCheckpoint (save best model)
       ├─> EarlyStopping (patience=15)
       ├─> BatchStatisticsCallback (log batch composition)
       └─> ChunkSamplerEpochCallback (update sampler epoch)
```

---

## Key Components Overview

### 1. Configuration System
- **Config File**: `core/pretraining_config_multidomain.yaml`
- **Config Class**: `core/pretraining_model.py::PretrainingConfig`
- **Config Creation**: `core/pretraining_model.py::create_pretraining_config()`

### 2. Data Loading System
- **Dataset**: `data_loading/lazy_universal_dataset.py::LazyUniversalDataset`
- **Single-Domain Sampler**: `data_loading/chunk_sampler.py::ChunkAwareSampler`
- **Multi-Domain Sampler (Default)**: `data_loading/improved_dynamic_sampler.py::ImprovedDynamicBatchSampler`
- **Multi-Domain Sampler (Legacy)**: `data_loading/dynamic_chunk_sampler.py::DynamicChunkAwareBatchSampler`

### 3. Model Architecture
- **Main Model**: `core/pretraining_model.py::PretrainingESAModel`
- **Encoder**: `core/pretraining_model.py::UniversalMolecularEncoder`
- **Backbone**: `esa/masked_layers.py::ESA`
- **Tasks**: `core/pretraining_model.py::PretrainingTasks`

---

## File Structure Reference

```
core/
├── train_pretrain.py              # Main training script
├── pretraining_model.py           # Model architecture & tasks
├── batch_statistics_callback.py   # Batch composition logging
├── pretraining_config_multidomain.yaml  # Multi-domain config
└── __init__.py                    # Module exports

data_loading/
├── lazy_universal_dataset.py      # Lazy loading dataset
├── improved_dynamic_sampler.py    # Improved dynamic batch sampler (default)
├── dynamic_chunk_sampler.py       # Legacy dynamic atom-aware batching
├── chunk_sampler.py               # Single-domain chunk sampler
└── create_chunk_metadata.py       # Generate metadata for atom-aware sampling

esa/
├── masked_layers.py               # ESA backbone (Set Transformer)
├── mha_flash_varlen.py            # Flash attention for variable length
├── mlp_utils.py                   # MLP utilities
└── utils/
    ├── norm_layers.py             # Normalization layers
    └── posenc_encoders/           # Position encoding
```

---

## Training Scripts: Detailed Documentation

---

## `core/train_pretrain.py`

### Purpose
Main training script that orchestrates pretraining pipeline. Handles dataset loading, model creation, and training with PyTorch Lightning. Supports single-domain and multi-domain training with optimized data loading.

### Key Functionality
- Automatic dataset detection (chunked vs non-chunked)
- Multi-domain support (proteins + molecules + metabolites + RNA)
- Optimized sampling: `ImprovedDynamicBatchSampler` (default), `DynamicChunkAwareBatchSampler` (legacy), `ChunkAwareSampler` (fixed batch size)
- Atom-aware batching (variable batch size prevents OOM)
- PyTorch Lightning integration (checkpointing, logging, mixed precision)

### Key Functions

- `create_pretraining_data_transforms()`: Creates MLM masking transform (15% atom masking)
- `load_universal_dataset()`: Loads dataset with auto-detection of chunks, supports single/multi-domain modes
- `create_data_loaders()`: Creates train/val/test loaders with optimal sampler selection
- `train_universal_pretraining()`: Main orchestration function (setup, callbacks, trainer, training)
- `main()`: Entry point (CLI args, config loading)

### Usage Example

```bash
# Multi-domain training
CUDA_VISIBLE_DEVICES=6 nohup python -m core.train_pretrain     --config-yaml-path core/pretraining_config_multidomain.yaml > mind_training.log 2>&1 &
``` 

### Notes
- **Called by**: Users via CLI for training
- **Calls**: `load_universal_dataset()`, `create_data_loaders()`, `PretrainingESAModel`, PyTorch Lightning Trainer
- **Multi-domain**: Automatically detects enabled dataset types from config, loads metadata if multiple types
- **Sampler selection**: Default `ImprovedDynamicBatchSampler` (fast, metadata-only), fallback to `DynamicChunkAwareBatchSampler` or `ChunkAwareSampler`
- **Batch sampler fix**: Re-applies custom batch_sampler before training (PyTorch Lightning sometimes replaces it)
- **Callbacks**: `ChunkSamplerEpochCallback` updates sampler epoch, `BatchStatisticsCallback` logs batch composition (optional)
- **Thread optimization**: Uses `torch.set_num_threads(1)` (optimal for ESA)

---

## `core/pretraining_model.py`

### Purpose
Defines the pretraining model architecture, including universal molecular encoder, ESA backbone, and pretraining tasks. Implements PyTorch Lightning module for training.

### Key Functionality
- Universal molecular encoding (works for proteins, small molecules, metabolites, RNA)
- SE(3) invariant 3D geometric features via GaussianLayer
- ESA backbone (Set Transformer) for attention-based learning
- Multiple pretraining tasks: long-range distance, short-range distance, MLM
- Per-type loss tracking for multi-domain analysis
- PyTorch Lightning integration (training_step, validation_step, test_step, configure_optimizers)
- 8-bit AdamW optimizer for memory efficiency

### Key Classes

- `PretrainingConfig`: Configuration dataclass with all model and training hyperparameters
- `UniversalMolecularEncoder`: Encodes atomic features (atomic number, period, 3D geometry)
- `PretrainingTasks`: Task-specific heads (distance prediction, MLM)
- `PretrainingESAModel`: Main PyTorch Lightning model (encoder + backbone + tasks)
- `create_pretraining_config()`: Factory function to create config from dict

### Key Functions

- `UniversalMolecularEncoder.forward()`: Encodes nodes with atomic features and 3D geometry
- `UniversalMolecularEncoder._compute_geometric_features()`: Computes SE(3) invariant features (coordination, distances)
- `UniversalMolecularEncoder._compute_chemical_coordination()`: Computes true chemical bond coordination numbers
- `PretrainingESAModel.forward()`: Forward pass (encode → ESA backbone → return embeddings)
- `PretrainingESAModel.training_step()`: Computes pretraining losses and logs metrics
- `PretrainingESAModel.validation_step()`: Computes validation losses
- `PretrainingESAModel.test_step()`: Computes test losses
- `PretrainingESAModel._compute_per_type_losses()`: Computes losses separately per dataset type (multi-domain)
- `PretrainingESAModel.configure_optimizers()`: Sets up 8-bit AdamW optimizer with cosine LR schedule
- `PretrainingTasks.long_range_distance_loss()`: Long-range distance prediction loss
- `PretrainingTasks.short_range_distance_loss()`: Short-range distance prediction loss
- `PretrainingTasks.mlm_loss()`: Masked language modeling loss

### SE(3) Invariant Geometric Features

The encoder computes rotation and translation invariant features:
```
1. Chemical coordination: True bond count from molecular graph (sp3≈4, sp2≈3, sp≈2)
2. Close coordination: Spatial neighbors within ~3Å (likely bonded)
3. Distance coordination: Medium range neighbors within cutoff (~6Å)
4. Min distance: Closest neighbor distance
5. Mean distance: Average close neighbor distance
6. Gaussian RBF: Radial basis functions for edge distances
```

### Per-Type Loss Tracking (Multi-Domain)

For multi-domain training, losses can be tracked separately per dataset type:
```python
# Config
compute_per_type_losses: bool = True
log_per_type_frequency: int = 10
```

### Input/Output

**Input** (forward):
- PyG batch with `z` (atomic numbers), `pos` (3D coordinates), `edge_index`, `batch` (graph assignment)

**Output**:
- `graph_embeddings`: Graph-level embeddings [batch_size, graph_dim]
- `node_embeddings`: Node-level embeddings [num_nodes, graph_dim]

### Notes
- **Called by**: `core/train_pretrain.py` for model creation
- **Uses**: `esa/masked_layers.py::ESA` (Set Transformer backbone), `data_loading/gaussian.py::GaussianLayer` (3D features)
- **Pretraining tasks**: Configurable via `config.pretraining_tasks` list
- **3D coordinates**: Required for distance prediction tasks, optional for MLM-only training
- **Universal encoding**: Works across all molecular domains (proteins, small molecules, metabolites, RNA)
- **Task weights**: Configurable via `config.task_weights` dict
- **Optimizer**: Uses `bitsandbytes.optim.AdamW8bit` for 8-bit training (memory efficient)
- **LR Schedule**: Cosine decay with 10% warmup, decays from 1.0 to 0.1

