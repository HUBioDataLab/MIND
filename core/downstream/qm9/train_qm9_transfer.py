"""
Training script for QM9 transfer learning.

Loads a pretrained model and fine-tunes it for QM9 property prediction.
"""

import sys
import os
import warnings
import argparse
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
import yaml
import json
from pathlib import Path
from torch_geometric.loader import DataLoader as GeometricDataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.seed import seed_everything

# Add project to path
sys.path.append(os.path.realpath("."))
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up: qm9 -> downstream -> core -> project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from core.downstream.qm9.qm9_transfer_model import QM9TransferModel
from data_loading.qm9_transfer_data import load_qm9_transfer_data

warnings.filterwarnings("ignore")

os.environ["WANDB__SERVICE_WAIT"] = "500"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    parser = argparse.ArgumentParser(description="Train QM9 transfer learning model")
    
    # Config file
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    
    # Override arguments
    parser.add_argument("--pretrained-ckpt", type=str, help="Path to pretrained checkpoint")
    parser.add_argument("--target-name", type=str, help="QM9 target property name")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--max-epochs", type=int, help="Maximum epochs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    if args.pretrained_ckpt:
        config['pretrained_ckpt_path'] = args.pretrained_ckpt
    if args.target_name:
        config['target_name'] = args.target_name
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['lr'] = args.lr
    if args.max_epochs:
        config['max_epochs'] = args.max_epochs
    
    # Set seed
    seed = config.get('seed', 42)
    seed_everything(seed)
    
    # Set GPU (respect CUDA_VISIBLE_DEVICES if already set, otherwise use args.gpu)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            device = f"cuda:0"  # After setting CUDA_VISIBLE_DEVICES, device becomes cuda:0
        else:
            device = "cpu"
    else:
        # CUDA_VISIBLE_DEVICES already set from command line
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # Load data
    print("Loading QM9 data...")
    dataset_dir = config.get('dataset_dir', './data/qm9')
    # Convert relative path to absolute path based on project root
    if not os.path.isabs(dataset_dir):
        # Get project root (3 levels up from this file: qm9 -> downstream -> core -> project root)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir)))
        dataset_dir = os.path.join(project_root, dataset_dir.lstrip('./'))
    target_name = config.get('target_name', 'homo')
    
    train_dataset, val_dataset, test_dataset, scaler = load_qm9_transfer_data(
        dataset_dir=dataset_dir,
        target_name=target_name
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    batch_size = config.get('batch_size', 128)
    num_workers = config.get('num_workers', 4)
    
    train_loader = GeometricDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = GeometricDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = GeometricDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create output directory
    out_path = config.get('out_path', './outputs/qm9_transfer')
    Path(out_path).mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = os.path.join(out_path, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Setup wandb
    wandb_project = config.get('wandb_project_name', 'qm9-transfer-learning')
    # Always use target_name to generate run name (ensures it matches the actual target being trained)
    wandb_run_name = f'qm9-{target_name}-transfer'
    
    logger = WandbLogger(
        name=wandb_run_name,
        project=wandb_project,
        save_dir=out_path
    )
    
    # Create model
    pretrained_ckpt_path = config.get('pretrained_ckpt_path', None)
    
    if pretrained_ckpt_path:
        print(f"Loading pretrained model from {pretrained_ckpt_path}")
        model_kwargs = dict(
            pretrained_ckpt_path=pretrained_ckpt_path,
        )
    else:
        print("Initializing model from scratch (zero weights)")
        model_kwargs = dict(
            pretrained_ckpt_path=None,
            hidden_dims=config.get('hidden_dims', [512, 512, 512, 512]),
            num_heads=config.get('num_heads', [8, 8, 8, 8]),
            layer_types=config.get('layer_types', ['S', 'S', 'S', 'P']),
            apply_attention_on=config.get('apply_attention_on', 'node'),
            xformers_or_torch_attn=config.get('xformers_or_torch_attn', 'xformers'),
            use_3d_coordinates=config.get('use_3d_coordinates', True),
            atom_types=config.get('atom_types', 121),
            gaussian_kernels=config.get('gaussian_kernels', 128),
            cutoff_distance=config.get('cutoff_distance', 5.0),
            sab_dropout=config.get('sab_dropout', 0.0),
            mab_dropout=config.get('mab_dropout', 0.0),
            pma_dropout=config.get('pma_dropout', 0.0),
            attn_residual_dropout=config.get('attn_residual_dropout', 0.0),
            pma_residual_dropout=config.get('pma_residual_dropout', 0.0),
            use_mlps=config.get('use_mlps', True),
            mlp_hidden_size=config.get('mlp_hidden_size', 512),
            mlp_type=config.get('mlp_type', 'standard'),
            norm_type=config.get('norm_type', 'LN'),
            num_mlp_layers=config.get('num_mlp_layers', 3),
            pre_or_post=config.get('pre_or_post', 'pre'),
            use_mlp_ln=config.get('use_mlp_ln', False),
            mlp_dropout=config.get('mlp_dropout', 0.0),
            set_max_items=config.get('set_max_items', 0),
            use_bfloat16=config.get('use_bfloat16', False),
            use_distance_bias=config.get('use_distance_bias', False),
            distance_bias_scale=config.get('distance_bias_scale', 1.0),
            distance_bias_cutoff=config.get('distance_bias_cutoff', 10.0),
        )
    
    model = QM9TransferModel(
        target_name=target_name,
        graph_dim=config.get('graph_dim', 512),
        batch_size=batch_size,
        lr=config.get('lr', 0.001),
        regression_loss_fn=config.get('regression_loss_fn', 'mae'),
        early_stopping_patience=config.get('early_stopping_patience', 30),
        optimiser_weight_decay=config.get('optimiser_weight_decay', 1e-3),
        gradient_clip_val=config.get('gradient_clip_val', 0.5),
        freeze_encoder=config.get('freeze_encoder', False),
        freeze_esa=config.get('freeze_esa', False),
        scaler=scaler,
        **model_kwargs
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=out_path,
        filename='best-{epoch:03d}-{val_loss:.4f}',
        mode='min',
        save_top_k=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.get('early_stopping_patience', 30),
        mode='min'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 100),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=config.get('gradient_clip_val', 0.5),
        precision='bf16' if config.get('use_bfloat16', False) else '32',
        num_sanity_val_steps=0
    )
    
    # Train
    print("Starting training...")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Test
    print("Running test...")
    trainer.test(model=model, dataloaders=test_loader, ckpt_path='best')
    
    # Save test results
    test_preds_path = os.path.join(out_path, 'test_predictions.npy')
    test_true_path = os.path.join(out_path, 'test_true.npy')
    test_metrics_path = os.path.join(out_path, 'test_metrics.npy')
    
    np.save(test_preds_path, model.test_preds)
    np.save(test_true_path, model.test_true)
    np.save(test_metrics_path, model.test_metrics)
    
    # Save loss history for inspection
    loss_history_path = os.path.join(out_path, 'loss_history.json')
    loss_history = {
        'target_name': target_name,
        'train_loss_history': [
            {
                'epoch': int(epoch), 
                'step': int(step) if step != -1 else None, 
                'normalized_loss': float(norm_loss) if norm_loss is not None else None, 
                'unnormalized_loss_eV': float(unnorm_loss.item() if isinstance(unnorm_loss, torch.Tensor) else unnorm_loss)
            }
            for epoch, step, norm_loss, unnorm_loss in model.train_loss_history
        ],
        'val_loss_history': [
            {
                'epoch': int(epoch), 
                'unnormalized_loss_eV': float(unnorm_loss.item() if isinstance(unnorm_loss, torch.Tensor) else unnorm_loss)
            }
            for epoch, unnorm_loss in model.val_loss_history
        ],
        'train_metrics_by_epoch': {
            int(epoch): {k: float(v.item() if isinstance(v, torch.Tensor) else v) 
                        if isinstance(v, (int, float, np.number, torch.Tensor)) else v 
                        for k, v in metrics.items()}
            for epoch, metrics in model.train_metrics.items()
        },
        'val_metrics_by_epoch': {
            int(epoch): {k: float(v.item() if isinstance(v, torch.Tensor) else v) 
                        if isinstance(v, (int, float, np.number, torch.Tensor)) else v 
                        for k, v in metrics.items()}
            for epoch, metrics in model.val_metrics.items()
        }
    }
    with open(loss_history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"\n📊 Loss History Summary:")
    print(f"   Train losses (last 5 epochs, in eV):")
    train_epoch_losses = [l for l in model.train_loss_history if l[1] == -1][-5:]
    for epoch, _, _, unnorm_loss in train_epoch_losses:
        print(f"      Epoch {epoch}: {unnorm_loss:.4f} eV")
    print(f"   Val losses (last 5 epochs, in eV):")
    for epoch, unnorm_loss in model.val_loss_history[-5:]:
        print(f"      Epoch {epoch}: {unnorm_loss:.4f} eV")
    
    print(f"\nTest results saved to {out_path}")
    print(f"Test metrics: {model.test_metrics}")
    print(f"Loss history saved to {loss_history_path}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
