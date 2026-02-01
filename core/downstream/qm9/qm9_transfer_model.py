"""
QM9 Transfer Learning Model

This model loads a pretrained PretrainingESAModel and fine-tunes it
for QM9 property prediction tasks.
"""

import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from collections import defaultdict
from typing import Optional, Dict, Any, List
import numpy as np
import math

import sys
import os
# Add project to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up: qm9 -> downstream -> core -> project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from core.pretraining_model import PretrainingESAModel, PretrainingConfig
from esa.masked_layers import ESA
from esa.mlp_utils import SmallMLP
from esa.utils.norm_layers import BN, LN
from esa.utils.reporting import get_regr_metrics_pt


def nearest_multiple_of_8(n):
    """Round up to nearest multiple of 8"""
    return math.ceil(n / 8) * 8


class QM9TransferModel(pl.LightningModule):
    """
    Transfer learning model for QM9 property prediction.
    
    Loads a pretrained PretrainingESAModel and replaces pretraining losses
    with QM9 regression losses.
    """
    
    def __init__(
        self,
        pretrained_ckpt_path: Optional[str] = None,
        target_name: str = "homo",
        graph_dim: int = 512,
        batch_size: int = 128,
        lr: float = 0.001,
        regression_loss_fn: str = "mae",
        early_stopping_patience: int = 30,
        optimiser_weight_decay: float = 1e-3,
        gradient_clip_val: float = 0.5,
        freeze_encoder: bool = False,
        freeze_esa: bool = False,
        scaler=None,
        # Config parameters for training from scratch
        config: Optional[PretrainingConfig] = None,
        hidden_dims: Optional[list] = None,
        num_heads: Optional[list] = None,
        layer_types: Optional[list] = None,
        apply_attention_on: str = "node",
        xformers_or_torch_attn: str = "xformers",
        use_3d_coordinates: bool = True,
        atom_types: int = 121,
        gaussian_kernels: int = 128,
        cutoff_distance: float = 5.0,
        sab_dropout: float = 0.0,
        mab_dropout: float = 0.0,
        pma_dropout: float = 0.0,
        attn_residual_dropout: float = 0.0,
        pma_residual_dropout: float = 0.0,
        use_mlps: bool = True,
        mlp_hidden_size: int = 512,
        mlp_type: str = "standard",
        norm_type: str = "LN",
        num_mlp_layers: int = 3,
        pre_or_post: str = "pre",
        use_mlp_ln: bool = False,
        mlp_dropout: float = 0.0,
        set_max_items: int = 0,
        use_bfloat16: bool = False,
        use_distance_bias: bool = False,
        distance_bias_scale: float = 1.0,
        distance_bias_cutoff: float = 10.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['scaler', 'config'])
        
        self.target_name = target_name
        self.graph_dim = graph_dim
        self.batch_size = batch_size
        self.lr = lr
        self.regression_loss_fn = regression_loss_fn
        self.early_stopping_patience = early_stopping_patience
        self.optimiser_weight_decay = optimiser_weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.freeze_encoder = freeze_encoder
        self.freeze_esa = freeze_esa
        self.scaler = scaler
        
        # Load pretrained model OR create from scratch
        if pretrained_ckpt_path is not None:
            print(f"Loading pretrained model from {pretrained_ckpt_path}")
            pretrained_model = PretrainingESAModel.load_from_checkpoint(
                pretrained_ckpt_path,
                strict=False  # Allow missing pretraining task heads
            )
            
            # Extract encoder and ESA backbone
            self.encoder = pretrained_model.encoder
            self.esa_backbone = pretrained_model.esa_backbone
            
            # Get config from pretrained model
            self.config = pretrained_model.config
        else:
            print("Initializing model from scratch (zero weights)")
            
            # Create config from parameters
            if config is not None:
                self.config = config
            else:
                # Create a minimal config with provided parameters
                self.config = PretrainingConfig(
                    graph_dim=graph_dim,
                    hidden_dims=hidden_dims or [graph_dim] * 4,
                    num_heads=num_heads or [8] * 4,
                    layer_types=layer_types or ["S", "S", "S", "P"],
                    apply_attention_on=apply_attention_on,
                    xformers_or_torch_attn=xformers_or_torch_attn,
                    use_3d_coordinates=use_3d_coordinates,
                    atom_types=atom_types,
                    gaussian_kernels=gaussian_kernels,
                    cutoff_distance=cutoff_distance,
                    sab_dropout=sab_dropout,
                    mab_dropout=mab_dropout,
                    pma_dropout=pma_dropout,
                    attn_residual_dropout=attn_residual_dropout,
                    pma_residual_dropout=pma_residual_dropout,
                    use_mlps=use_mlps,
                    mlp_hidden_size=mlp_hidden_size,
                    mlp_type=mlp_type,
                    norm_type=norm_type,
                    num_mlp_layers=num_mlp_layers,
                    pre_or_post=pre_or_post,
                    use_mlp_ln=use_mlp_ln,
                    mlp_dropout=mlp_dropout,
                    set_max_items=set_max_items,
                    use_bfloat16=use_bfloat16,
                )
            
            # Create encoder from scratch
            from core.pretraining_model import UniversalMolecularEncoder
            self.encoder = UniversalMolecularEncoder(self.config)
            
            # Create ESA backbone from scratch
            st_args = dict(
                num_outputs=32,
                dim_output=self.config.graph_dim,
                xformers_or_torch_attn=self.config.xformers_or_torch_attn,
                dim_hidden=self.config.hidden_dims,
                num_heads=self.config.num_heads,
                sab_dropout=self.config.sab_dropout,
                mab_dropout=self.config.mab_dropout,
                pma_dropout=self.config.pma_dropout,
                use_mlps=self.config.use_mlps,
                mlp_hidden_size=self.config.mlp_hidden_size,
                mlp_type=self.config.mlp_type,
                norm_type=self.config.norm_type,
                node_or_edge=self.config.apply_attention_on,
                residual_dropout=self.config.attn_residual_dropout,
                set_max_items=nearest_multiple_of_8(self.config.set_max_items + 1),
                use_bfloat16=self.config.use_bfloat16,
                layer_types=self.config.layer_types,
                num_mlp_layers=self.config.num_mlp_layers,
                pre_or_post=self.config.pre_or_post,
                pma_residual_dropout=self.config.pma_residual_dropout,
                use_mlp_ln=self.config.use_mlp_ln,
                mlp_dropout=self.config.mlp_dropout,
            )
            
            self.esa_backbone = ESA(**st_args)
            
            # Configure distance-based attention bias (optional)
            if use_distance_bias:
                self.esa_backbone.use_distance_bias = True
                self.esa_backbone.distance_bias_scale = distance_bias_scale
                self.esa_backbone.distance_bias_cutoff = distance_bias_cutoff
                print(f"✅ Distance-based attention bias enabled: scale={distance_bias_scale}, cutoff={distance_bias_cutoff}Å")
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            print("Freezing encoder weights")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Freeze ESA if requested
        if self.freeze_esa:
            print("Freezing ESA backbone weights")
            for param in self.esa_backbone.parameters():
                param.requires_grad = False
        
        # Output head for QM9 property prediction
        self.output_head = SmallMLP(
            in_dim=self.graph_dim,
            inter_dim=128,
            out_dim=1,
            use_ln=False,
            dropout_p=0.0,
            num_layers=2,
        )
        
        # Metrics tracking
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.test_output = defaultdict(list)
        
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
        # Store test predictions and true values
        self.test_preds = []
        self.test_true = []
        
        # Loss history for inspection (keep all losses)
        self.train_loss_history = []  # List of (epoch, step, normalized_loss, unnormalized_loss)
        self.val_loss_history = []   # List of (epoch, normalized_loss, unnormalized_loss)
    
    def forward(self, batch):
        """
        Forward pass for QM9 property prediction.
        
        Args:
            batch: PyTorch Geometric batch with z (atomic numbers), pos (coordinates), edge_index
            
        Returns:
            predictions: [batch_size] property predictions
        """
        edge_index = batch.edge_index
        batch_mapping = batch.batch
        pos = getattr(batch, 'pos', None)
        
        # Get atomic numbers
        assert hasattr(batch, 'z') and batch.z is not None, "Batch must have 'z' attribute"
        x = batch.z
        
        # Encode molecules - pass full batch object for edge_index access
        h = self.encoder(x, pos=pos, batch=batch)
        
        # Determine max items for ESA
        if self.config.apply_attention_on == "edge":
            # For edge attention, need max edges
            if hasattr(batch, 'max_edge_global'):
                num_max_items = nearest_multiple_of_8(batch.max_edge_global.max().item() + 1)
            else:
                # Fallback: estimate from edge_index
                num_max_items = nearest_multiple_of_8(edge_index.size(1) + 1)
        else:
            # For node attention, need max nodes
            if hasattr(batch, 'max_node_global'):
                num_max_items = nearest_multiple_of_8(batch.max_node_global.max().item() + 1)
            else:
                # Fallback: estimate from batch
                counts = torch.bincount(batch_mapping)
                num_max_items = nearest_multiple_of_8(counts.max().item() + 1)
        
        # Convert to dense batch format
        h, dense_batch_index = to_dense_batch(
            h, batch_mapping, fill_value=0, max_num_nodes=num_max_items
        )
        
        # Pass through ESA backbone
        esa_output = self.esa_backbone(
            h, edge_index, batch_mapping, num_max_items=num_max_items
        )
        
        # ESA returns (graph_embeddings, node_embeddings_before_pma) or just graph_embeddings
        if isinstance(esa_output, tuple):
            graph_emb, _ = esa_output
        else:
            graph_emb = esa_output
        
        # Graph embeddings should be [batch_size, graph_dim]
        # Handle different output formats from ESA
        batch_size = batch_mapping.max().item() + 1
        
        if graph_emb.dim() == 3:
            # [batch_size, max_nodes, graph_dim] -> [batch_size, graph_dim]
            # Use mean pooling over nodes
            graph_emb = graph_emb.mean(dim=1)
        elif graph_emb.dim() == 2:
            if graph_emb.size(0) == batch_size:
                # Already graph-level embeddings [batch_size, graph_dim]
                pass
            else:
                # Node-level embeddings, need to pool
                graph_emb, _ = to_dense_batch(graph_emb, batch_mapping, fill_value=0)
                if graph_emb.dim() == 3:
                    graph_emb = graph_emb.mean(dim=1)
        
        # Predict property
        predictions = self.output_head(graph_emb).squeeze(-1)  # [batch_size]
        
        return predictions
    
    def _batch_loss(self, batch, step_type: str = "train"):
        """Compute loss for a batch"""
        predictions = self.forward(batch)
        
        # Get target property
        y = batch.y
        if y.dim() > 1:
            y = y.squeeze()
        
        # Compute loss (normalized space)
        if self.regression_loss_fn == "mse":
            loss = F.mse_loss(predictions, y.float())
        elif self.regression_loss_fn == "mae":
            loss = F.l1_loss(predictions, y.float())
        else:
            raise ValueError(f"Unknown regression loss function: {self.regression_loss_fn}")
        
        # Compute unnormalized loss (in original units, e.g., eV)
        if self.scaler is not None:
            # Convert predictions and targets to original scale
            pred_unnorm = self.scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).flatten()
            y_unnorm = self.scaler.inverse_transform(y.detach().cpu().numpy().reshape(-1, 1)).flatten()
            
            if self.regression_loss_fn == "mse":
                loss_unnorm = np.mean((pred_unnorm - y_unnorm) ** 2)
            else:  # mae
                loss_unnorm = np.mean(np.abs(pred_unnorm - y_unnorm))
        else:
            loss_unnorm = loss.item()
        
        return loss, predictions, y, loss_unnorm
    
    def training_step(self, batch, batch_idx):
        loss, predictions, y, loss_unnorm = self._batch_loss(batch, "train")
        
        # Log normalized loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log unnormalized loss (in original units, e.g., eV for HOMO)
        self.log(f"train_loss_{self.target_name}_eV", loss_unnorm, prog_bar=True, on_step=True, on_epoch=True)
        
        # Store loss history (ensure loss_unnorm is a Python float)
        loss_unnorm_float = float(loss_unnorm) if isinstance(loss_unnorm, (torch.Tensor, np.ndarray, np.number)) else loss_unnorm
        self.train_loss_history.append((self.current_epoch, batch_idx, loss.item(), loss_unnorm_float))
        
        # Store for epoch-end metrics
        self.train_output[self.current_epoch].append((
            predictions.detach().cpu(),
            y.detach().cpu()
        ))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, predictions, y, loss_unnorm = self._batch_loss(batch, "validation")
        
        # Log normalized loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # Log unnormalized loss (in original units, e.g., eV for HOMO)
        self.log(f"val_loss_{self.target_name}_eV", loss_unnorm, prog_bar=True, on_step=False, on_epoch=True)
        
        # Store for epoch-end metrics
        self.val_output[self.current_epoch].append((
            predictions.detach().cpu(),
            y.detach().cpu()
        ))
        
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, predictions, y, loss_unnorm = self._batch_loss(batch, "test")
        
        # Log normalized loss
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # Log unnormalized loss (in original units, e.g., eV for HOMO)
        self.log(f"test_loss_{self.target_name}_eV", loss_unnorm, prog_bar=True, on_step=False, on_epoch=True)
        
        # Store predictions and true values
        self.test_preds.append(predictions.detach().cpu())
        self.test_true.append(y.detach().cpu())
        
        return loss
    
    def _compute_metrics(self, predictions, targets, epoch_type: str):
        """Compute regression metrics"""
        # Flatten predictions and targets
        if isinstance(predictions, list):
            y_pred = torch.cat(predictions, dim=0).numpy()
        else:
            y_pred = predictions.numpy()
        
        if isinstance(targets, list):
            y_true = torch.cat(targets, dim=0).numpy()
        else:
            y_true = targets.numpy()
        
        # Inverse transform if scaler is available
        if self.scaler is not None:
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred = torch.from_numpy(y_pred)
            y_true = torch.from_numpy(y_true)
        
        # Compute metrics
        metrics = get_regr_metrics_pt(y_true, y_pred)
        
        # Log metrics
        self.log(f"{epoch_type} R2", metrics["R2"], batch_size=self.batch_size)
        self.log(f"{epoch_type} MAE", metrics["MAE"], batch_size=self.batch_size)
        self.log(f"{epoch_type} RMSE", metrics["RMSE"], batch_size=self.batch_size)
        self.log(f"{epoch_type} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)
        
        return metrics, y_pred, y_true
    
    def on_train_epoch_end(self):
        if len(self.train_output[self.current_epoch]) > 0:
            preds = [item[0] for item in self.train_output[self.current_epoch]]
            targets = [item[1] for item in self.train_output[self.current_epoch]]
            
            metrics, y_pred, y_true = self._compute_metrics(preds, targets, "Train")
            self.train_metrics[self.current_epoch] = metrics
            
            # Store epoch-level loss history (average unnormalized loss for this epoch)
            epoch_unnorm_losses = [unnorm_loss for epoch, step, norm_loss, unnorm_loss in self.train_loss_history 
                                   if epoch == self.current_epoch and step != -1]
            if epoch_unnorm_losses:
                avg_unnorm_loss = np.mean(epoch_unnorm_losses)
                # Convert to Python float
                avg_unnorm_loss_float = float(avg_unnorm_loss)
                # Store epoch summary (step=-1 indicates epoch-level summary)
                self.train_loss_history.append((self.current_epoch, -1, None, avg_unnorm_loss_float))
            
            # Keep train_output for inspection (don't delete immediately)
            # We'll keep last few epochs to avoid memory issues
            if self.current_epoch > 5:
                # Keep only last 5 epochs
                epochs_to_keep = sorted(self.train_output.keys())[-5:]
                for epoch in list(self.train_output.keys()):
                    if epoch not in epochs_to_keep:
                        del self.train_output[epoch]
    
    def on_validation_epoch_end(self):
        if len(self.val_output[self.current_epoch]) > 0:
            preds = [item[0] for item in self.val_output[self.current_epoch]]
            targets = [item[1] for item in self.val_output[self.current_epoch]]
            
            metrics, y_pred, y_true = self._compute_metrics(preds, targets, "Validation")
            self.val_metrics[self.current_epoch] = metrics
            
            # Store epoch-level validation loss (unnormalized)
            # MAE is already in original units after inverse transform
            avg_unnorm_loss = metrics["MAE"]
            # Convert to Python float if it's a tensor
            avg_unnorm_loss_float = float(avg_unnorm_loss.item() if isinstance(avg_unnorm_loss, torch.Tensor) else avg_unnorm_loss)
            self.val_loss_history.append((self.current_epoch, avg_unnorm_loss_float))
            
            # Keep val_output for inspection (don't delete)
            # del self.val_output[self.current_epoch]
    
    def on_test_epoch_end(self):
        if len(self.test_preds) > 0:
            metrics, y_pred, y_true = self._compute_metrics(self.test_preds, self.test_true, "Test")
            self.test_metrics = metrics
            
            # Store for saving
            self.test_preds = y_pred
            self.test_true = y_true
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Only optimize parameters that require gradients
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.optimiser_weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.early_stopping_patience // 2,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
