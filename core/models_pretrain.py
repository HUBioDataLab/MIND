import torch
import math
import numpy as np
import pytorch_lightning as pl
import bitsandbytes as bnb

from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from esa.utils.norm_layers import BN, LN
from esa.masked_layers import ESA
from esa.mlp_utils import SmallMLP, GatedMLPMulti

from esa.utils.reporting import (
    get_cls_metrics_binary_pt,
    get_cls_metrics_multilabel_pt,
    get_cls_metrics_multiclass_pt,
    get_regr_metrics_pt,
)

# Positional encodings removed for universal pretraining

# Import vocabulary mappings
from .vocabulary import (
    get_block_symbol_mappings, 
    get_position_code_mappings, 
    get_entity_type_mappings,
    BLOCK_SYMBOL_VOCAB_SIZE,
    POSITION_CODE_VOCAB_SIZE,
    ENTITY_TYPE_VOCAB_SIZE
)

# Task names for the LRGB peptides-func benchmark
pept_struct_target_names = ["Inertia_mass_a", "Inertia_mass_b", "Inertia_mass_c",
                        "Inertia_valence_a", "Inertia_valence_b",
                        "Inertia_valence_c", "length_a", "length_b", "length_c",
                        "Spherocity", "Plane_best_fit"]


def nearest_multiple_of_8(n):
    return math.ceil(n / 8) * 8


@dataclass
class PretrainingConfig:
    """Configuration for universal pretraining model - all values from YAML"""
    
    # Basic settings
    seed: int
    dataset: str
    dataset_download_dir: str
    out_path: str
    wandb_project_name: str
    wandb_run_name: str
    max_epochs: int
    gradient_clip_val: float
    monitor_loss_name: str
    
    # Data settings
    num_workers: int
    molecule_max_atoms: int
    max_samples: int
    
    # Universal representation specific settings
    use_universal_cache: bool
    universal_cache_path: str
    
    # Model architecture
    apply_attention_on: str
    graph_dim: int
    edge_dim: int
    hidden_dims: List[int]
    num_heads: List[int]
    layer_types: List[str]
    xformers_or_torch_attn: str
    norm_type: str
    use_3d_coordinates: bool
    
    # Hierarchical featurization settings
    use_hierarchical_features: bool
    block_embedding_dim: int
    position_embedding_dim: int
    entity_embedding_dim: int
    max_block_types: int
    max_position_types: int
    max_entity_types: int
    
    # MLP config
    use_mlps: bool
    mlp_hidden_size: int
    mlp_type: str
    num_mlp_layers: int
    
    # Dropout settings
    sab_dropout: float
    mab_dropout: float
    pma_dropout: float
    attn_residual_dropout: float
    pma_residual_dropout: float
    mlp_dropout: float
    
    # Training settings
    batch_size: int
    lr: float
    early_stopping_patience: int
    optimiser_weight_decay: float
    
    # Pretraining tasks
    pretraining_tasks: List[str]
    task_weights: Dict[str, float]
    
    # ESA optimizations
    use_bfloat16: bool
    pre_or_post: str
    use_mlp_ln: bool
    set_max_items: int
    triu_attn_mask: bool
    
    # Positional encoding
    posenc: Optional[str]
    
    # Sanity check
    num_sanity_val_steps: int
    
    # Additional config parameters
    gaussian_kernels: int
    max_distance: float
    distance_bins: int
    mlm_mask_ratio: float
    temperature: float
    use_memory_efficient_attention: bool
    attention_dropout: float
    preserve_universal_blocks: bool
    fast_dev_run: bool
    accelerator: str
    devices: str
    precision: str
    cutoff_distance: float
    max_neighbors: int
    
    # Configurable architectural parameters
    max_atomic_num: int  # GET atomic elements vocabulary (118 chemical elements)
    rwse_dim: int
    lap_dim: int
    mlp_inter_dim: int
    pma_num_outputs: int
    max_node_items: int
    max_edge_items: int
    output_dim: int


class PretrainingESAModel(pl.LightningModule):
    """
    Optimized ESA Pretraining Model for Universal Representations
    
    This model combines the full ESA optimization from the original implementation
    with pretraining-specific functionality for universal molecular representations.
    """
    
    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Store all parameters (matching original ESA model)
        self.graph_dim = config.graph_dim
        self.edge_dim = config.edge_dim
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.monitor_loss_name = config.monitor_loss_name
        self.mlp_hidden_size = config.mlp_hidden_size
        self.norm_type = config.norm_type
        self.set_max_items = config.set_max_items
        self.use_mlp_ln = config.use_mlp_ln
        self.pre_or_post = config.pre_or_post
        self.mlp_dropout = config.mlp_dropout
        self.use_mlps = config.use_mlps
        self.early_stopping_patience = config.early_stopping_patience
        self.optimiser_weight_decay = config.optimiser_weight_decay
        self.mlp_type = config.mlp_type
        self.attn_residual_dropout = config.attn_residual_dropout
        self.pma_residual_dropout = config.pma_residual_dropout
        self.triu_attn_mask = config.triu_attn_mask
        self.use_bfloat16 = config.use_bfloat16
        self.layer_types = config.layer_types
        self.apply_attention_on = config.apply_attention_on
        self.xformers_or_torch_attn = config.xformers_or_torch_attn
        self.posenc = config.posenc
        
        # Pretraining specific
        self.pretraining_tasks = config.pretraining_tasks
        self.task_weights = config.task_weights
        
        # Initialize metrics tracking
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.train_metrics = {}
        self.val_metrics = {}
        
        # Initialize positional encoders (disabled for universal pretraining)
        self.rwse_encoder = None
        self.lap_encoder = None
 
        # Initialize hierarchical embedding layers
        if config.use_hierarchical_features:
            self.block_embedding = nn.Embedding(config.max_block_types, config.block_embedding_dim)
            self.position_embedding = nn.Embedding(config.max_position_types, config.position_embedding_dim)
            self.entity_embedding = nn.Embedding(config.max_entity_types, config.entity_embedding_dim)
            
            # Vocabulary mappings for string-to-index conversion
            self._init_vocabulary_mappings()
        else:
            self.block_embedding = None
            self.position_embedding = None
            self.entity_embedding = None
 
        # Initialize normalization (matching original)
        if config.norm_type == "BN":
            norm_fn = BN
        elif config.norm_type == "LN":
            norm_fn = LN

        # Create MLPs based on attention mode (matching original ESA/NSA)
        if config.apply_attention_on == "node":
            # NSA (Node Set Attention) - matching original
            in_dim = config.max_atomic_num
            # Positional encodings removed for universal pretraining
            # Always add hierarchical features when enabled (they're padded if missing)
            if config.use_hierarchical_features:
                in_dim += config.block_embedding_dim + config.position_embedding_dim + config.entity_embedding_dim
            
            # Don't create MLP here - let it be created dynamically based on actual data
            # This avoids dimension mismatches with the universal dataset
        
        elif config.apply_attention_on == "edge":
            # ESA (Edge Set Attention) - matching original
            in_dim = config.max_atomic_num
            # Positional encodings removed for universal pretraining
            if config.use_hierarchical_features:
                in_dim += config.block_embedding_dim + config.position_embedding_dim + config.entity_embedding_dim

            in_dim = in_dim * 2  # Concatenate source and target
            if config.edge_dim is not None:
                in_dim += config.edge_dim
            

            # Don't create MLP here - let it be created dynamically based on actual data
            # This avoids dimension mismatches with the universal dataset
        
        # Layer normalization (matching original)
        self.mlp_norm = norm_fn(config.hidden_dims[0])
        
        # Create ESA backbone with exact original parameters
        st_args = dict(
            num_outputs=config.pma_num_outputs,  # k for PMA
            dim_output=config.graph_dim,
            xformers_or_torch_attn=config.xformers_or_torch_attn,
            dim_hidden=config.hidden_dims,
            num_heads=config.num_heads,
            sab_dropout=config.sab_dropout,
            mab_dropout=config.mab_dropout,
            pma_dropout=config.pma_dropout,
            use_mlps=config.use_mlps,
            mlp_hidden_size=config.mlp_hidden_size,
            mlp_type=config.mlp_type,
            norm_type=config.norm_type,
            node_or_edge=config.apply_attention_on,
            use_bfloat16=config.use_bfloat16,
            layer_types=config.layer_types,
            num_mlp_layers=config.num_mlp_layers,
            pre_or_post=config.pre_or_post,
            residual_dropout=config.attn_residual_dropout,
            pma_residual_dropout=config.pma_residual_dropout,
            set_max_items=config.set_max_items,
            use_mlp_ln=config.use_mlp_ln,
            mlp_dropout=config.mlp_dropout,
        )

        self.st_fast = ESA(**st_args)

        # Create pretraining tasks
        self.pretraining_tasks_module = PretrainingTasks(config)
        
        # Create output MLP for final predictions (matching original)
        self.output_mlp = nn.Linear(config.graph_dim, config.output_dim)
        
        # Debug flag for attention logging
        self._attention_debug_printed = False
    
    def forward(self, batch):
        """Forward pass matching original ESA/NSA implementation exactly"""
        # Extract features from batch (universal representations)
        if hasattr(batch, 'x') and batch.x is not None:
            x = batch.x
        else:
            # Efficient one-hot encoding from atomic numbers
            if hasattr(batch, 'z') and batch.z is not None:
                max_atomic_num = self.config.max_atomic_num
                x = torch.zeros(batch.z.shape[0], max_atomic_num, device=batch.z.device, dtype=torch.float16)
                valid_mask = batch.z < max_atomic_num
                x[valid_mask, batch.z[valid_mask]] = 1.0
            else:
                # Default features - create one-hot encoding
                num_nodes = batch.pos.shape[0] if hasattr(batch, 'pos') else 1
                x = torch.zeros(num_nodes, self.config.max_atomic_num, device=batch.pos.device if hasattr(batch, 'pos') else 'cpu', dtype=torch.float16)
        
        # Convert to float (matching original)
        x = x.float()

        # Positional encodings disabled for universal pretraining

        # Get edge information
        edge_index = getattr(batch, 'edge_index', None)
        edge_attr = getattr(batch, 'edge_attr', None)
        batch_mapping = getattr(batch, 'batch', None)
        num_max_items = getattr(batch, 'max_node_global', self.config.max_node_items) if self.config.apply_attention_on == "node" else getattr(batch, 'max_edge_global', self.config.max_edge_items)
        
        # ESA (Edge Set Attention) - matching original exactly
        if self.config.apply_attention_on == "edge":
            if not self._attention_debug_printed:
                print(f"🔍 [ATTENTION DEBUG] ESA (Edge Set Attention) mode:")
                print(f"   • Node features shape: {x.shape}")
                print(f"   • Edge index shape: {edge_index.shape}")
                print(f"   • Number of edges: {edge_index.shape[1]:,}")
                print(f"   • Memory estimate: {(edge_index.shape[1] ** 2 * 4) / 1024**2:.2f} MB (float32)")
                self._attention_debug_printed = True
            
            source = x[edge_index[0, :], :]
            target = x[edge_index[1, :], :]
            h = torch.cat((source, target), dim=1)

            if self.edge_dim is not None and edge_attr is not None:
                h = torch.cat((h, edge_attr.float()), dim=1)

            h = self.node_edge_mlp(h)

            edge_batch_index = batch_mapping.index_select(0, edge_index[0, :])
            h, _ = to_dense_batch(h, edge_batch_index, fill_value=0, max_num_nodes=num_max_items)
            h = self.st_fast(h, edge_index, batch_mapping, num_max_items=num_max_items)

        # NSA (Node Set Attention) - matching original exactly
        else:
            if not self._attention_debug_printed:
                print(f"🔍 [ATTENTION DEBUG] NSA (Node Set Attention) mode:")
                print(f"   • Node features shape: {x.shape}")
                print(f"   • Number of nodes: {x.shape[0]:,}")
                print(f"   • Memory estimate: {(x.shape[0] ** 2 * 4) / 1024**2:.2f} MB (float32)")
                self._attention_debug_printed = True
            
            h = self.mlp_norm(self.node_mlp(x))

            h, dense_batch_index = to_dense_batch(h, batch_mapping, fill_value=0, max_num_nodes=num_max_items)
            h = self.st_fast(h, edge_index, batch_mapping, num_max_items=num_max_items)

            if getattr(self, 'is_node_task', False):
                h = h[dense_batch_index]

        # Final output (matching original)
        predictions = torch.flatten(self.output_mlp(h))

        return predictions
    
    def _edge_to_node_embeddings(self, edge_embeddings, edge_index, num_nodes):
        """Convert edge embeddings back to node embeddings"""
        node_embeddings = torch.zeros(num_nodes, edge_embeddings.shape[1], device=edge_embeddings.device)
        node_counts = torch.zeros(num_nodes, device=edge_embeddings.device)
        
        # Sum edge embeddings for each node
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            node_embeddings[src] += edge_embeddings[i]
            node_embeddings[dst] += edge_embeddings[i]
            node_counts[src] += 1
            node_counts[dst] += 1
        
        # Average by node degree
        node_counts = torch.clamp(node_counts, min=1)
        node_embeddings = node_embeddings / node_counts.unsqueeze(1)
        
        return node_embeddings
    
    def training_step(self, batch, batch_idx):
        """Training step with pretraining tasks"""
        # Get node embeddings for pretraining tasks
        node_embeddings = self._get_node_embeddings(batch)
        
        # Compute pretraining losses
        task_results = self.pretraining_tasks_module(node_embeddings, batch)
        
        # Weighted loss computation with regularization
        total_loss = 0.0
        loss_dict = {}
        
        for task_name, task_loss in task_results.items():
            if task_loss is not None:
                weight = self.task_weights.get(task_name, 1.0)
                weighted_loss = weight * task_loss
                total_loss += weighted_loss
                loss_dict[f"train_{task_name}_loss_step"] = task_loss
        
        # Add regularization to make training more challenging
        # L2 regularization on node embeddings to prevent overfitting
        embedding_reg = 0.01 * torch.norm(node_embeddings, p=2)
        total_loss += embedding_reg
        loss_dict["train_embedding_reg"] = embedding_reg
        
        # Simplified contrastive loss to prevent hanging
        if node_embeddings.shape[0] > 10:  # Only for reasonably sized batches
            pos = batch.pos if hasattr(batch, 'pos') else None
            if pos is not None and pos.shape[0] == node_embeddings.shape[0] and pos.shape[0] < 1000:  # Limit size
                # Sample a subset to avoid memory issues
                num_samples = min(50, node_embeddings.shape[0])
                sample_idx = torch.randperm(node_embeddings.shape[0])[:num_samples]
                
                sample_embeddings = node_embeddings[sample_idx]
                sample_pos = pos[sample_idx]
                
                # Compute pairwise distances for subset
                pairwise_dist = torch.cdist(sample_pos, sample_pos)
                # Create contrastive targets: 1 for close nodes, 0 for distant nodes
                contrastive_targets = (pairwise_dist < 2.0).float()
                # Compute cosine similarity between embeddings
                norm_embeddings = F.normalize(sample_embeddings, p=2, dim=1)
                pairwise_sim = torch.mm(norm_embeddings, norm_embeddings.t())
                # Contrastive loss: push similar nodes together, pull dissimilar apart
                contrastive_loss = F.binary_cross_entropy_with_logits(pairwise_sim, contrastive_targets)
                total_loss += 0.05 * contrastive_loss  # Reduced weight
                loss_dict["train_contrastive_loss"] = contrastive_loss
        
        loss_dict["train_total_loss"] = total_loss
        
        # Log losses
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with pretraining tasks"""
        # Get node embeddings for pretraining tasks
        node_embeddings = self._get_node_embeddings(batch)
        
        # Compute pretraining losses
        task_results = self.pretraining_tasks_module(node_embeddings, batch)
        
        # Weighted loss computation
        total_loss = 0.0
        loss_dict = {}
        
        for task_name, task_loss in task_results.items():
            if task_loss is not None:
                weight = self.task_weights.get(task_name, 1.0)
                weighted_loss = weight * task_loss
                total_loss += weighted_loss
                loss_dict[f"val_{task_name}_loss"] = task_loss
        
        loss_dict["val_total_loss"] = total_loss
        
        # Log losses
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def _get_node_embeddings(self, batch):
        """Extract node embeddings for pretraining tasks"""
        # Extract features from batch (universal representations)
        if hasattr(batch, 'x') and batch.x is not None:
            x = batch.x
        else:
            # Efficient one-hot encoding from atomic numbers
            if hasattr(batch, 'z') and batch.z is not None:
                max_atomic_num = self.config.max_atomic_num
                x = torch.zeros(batch.z.shape[0], max_atomic_num, device=batch.z.device, dtype=torch.float16)
                valid_mask = batch.z < max_atomic_num
                x[valid_mask, batch.z[valid_mask]] = 1.0
            else:
                # Default features - create one-hot encoding
                num_nodes = batch.pos.shape[0] if hasattr(batch, 'pos') else 1
                x = torch.zeros(num_nodes, self.config.max_atomic_num, device=batch.pos.device if hasattr(batch, 'pos') else 'cpu', dtype=torch.float16)
        
        # Convert to float (matching original)
        x = x.float()
        
        # Add hierarchical features (always when enabled)
        if self.config.use_hierarchical_features:
            block_symbol_ids, position_code_ids, entity_ids = self._get_hierarchical_features(batch)
            
            # Get the actual number of nodes in this batch
            num_nodes = x.shape[0]
            
            # Always produce all three components (pad individually if missing)
            block_emb = torch.zeros(num_nodes, self.config.block_embedding_dim, device=x.device, dtype=x.dtype)
            if block_symbol_ids is not None and self.block_embedding is not None and len(block_symbol_ids) > 0:
                # Ensure block_symbol_ids has the right length
                if len(block_symbol_ids) != num_nodes:
                    # Pad or truncate to match num_nodes
                    padded_block_ids = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
                    min_len = min(len(block_symbol_ids), num_nodes)
                    padded_block_ids[:min_len] = block_symbol_ids[:min_len]
                    block_symbol_ids = padded_block_ids
                block_emb = self.block_embedding(block_symbol_ids.long())

            position_emb = torch.zeros(num_nodes, self.config.position_embedding_dim, device=x.device, dtype=x.dtype)
            if position_code_ids is not None and self.position_embedding is not None and len(position_code_ids) > 0:
                # Ensure position_code_ids has the right length
                if len(position_code_ids) != num_nodes:
                    # Pad or truncate to match num_nodes
                    padded_pos_ids = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
                    min_len = min(len(position_code_ids), num_nodes)
                    padded_pos_ids[:min_len] = position_code_ids[:min_len]
                    position_code_ids = padded_pos_ids
                position_emb = self.position_embedding(position_code_ids.long())

            entity_emb = torch.zeros(num_nodes, self.config.entity_embedding_dim, device=x.device, dtype=x.dtype)
            if entity_ids is not None and self.entity_embedding is not None and len(entity_ids) > 0:
                # Ensure entity_ids has the right length
                if len(entity_ids) != num_nodes:
                    # Pad or truncate to match num_nodes
                    padded_entity_ids = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
                    min_len = min(len(entity_ids), num_nodes)
                    padded_entity_ids[:min_len] = entity_ids[:min_len]
                    entity_ids = padded_entity_ids
                entity_emb = self.entity_embedding(entity_ids.long())

            hierarchical_combined = torch.cat([block_emb, position_emb, entity_emb], dim=1)
            x = torch.cat((x, hierarchical_combined), dim=1)
        
        # Create MLP dynamically based on actual input dimensions
        if not hasattr(self, 'node_mlp'):
            # Create the MLP with the correct input dimension
            self.node_mlp = SmallMLP(
                in_dim=x.shape[1],
                inter_dim=self.config.mlp_inter_dim,
                out_dim=self.config.hidden_dims[0],
                use_ln=False,
                dropout_p=0,
                num_layers=self.config.num_mlp_layers if self.config.num_mlp_layers > 1 else self.config.num_mlp_layers + 1,
            ).to(x.device)
            print(f"Created MLP with {x.shape[1]} input features")
        
        # Positional encodings completely removed for universal pretraining
        
        # Get edge information
        edge_index = getattr(batch, 'edge_index', None)
        edge_attr = getattr(batch, 'edge_attr', None)
        batch_mapping = getattr(batch, 'batch', None)
        num_max_items = getattr(batch, 'max_node_global', self.config.max_node_items) if self.config.apply_attention_on == "node" else getattr(batch, 'max_edge_global', self.config.max_edge_items)
        
        # Process through ESA/NSA to get node embeddings
        if self.config.apply_attention_on == "edge":
            # ESA (Edge Set Attention)
            source = x[edge_index[0, :], :]
            target = x[edge_index[1, :], :]
            h = torch.cat((source, target), dim=1)
            
            if self.edge_dim is not None and edge_attr is not None:
                h = torch.cat((h, edge_attr.float()), dim=1)
            
            h = self.node_edge_mlp(h)
            
            edge_batch_index = batch_mapping.index_select(0, edge_index[0, :])
            h, _ = to_dense_batch(h, edge_batch_index, fill_value=0, max_num_nodes=num_max_items)
            h = self.st_fast(h, edge_index, batch_mapping, num_max_items=num_max_items)
            
            # Convert back to node embeddings
            node_embeddings = self._edge_to_node_embeddings(h, edge_index, x.shape[0])
        else:
            # NSA (Node Set Attention)
            h = self.mlp_norm(self.node_mlp(x))
            
            h, dense_batch_index = to_dense_batch(h, batch_mapping, fill_value=0, max_num_nodes=num_max_items)
            h = self.st_fast(h, edge_index, batch_mapping, num_max_items=num_max_items)
            
            if getattr(self, 'is_node_task', False):
                h = h[dense_batch_index]
            
            node_embeddings = h
        
        return node_embeddings
    
    def configure_optimizers(self):
        """Configure optimizers with ESA optimizations"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.optimiser_weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.monitor_loss_name,
            },
        }
    
    def _init_vocabulary_mappings(self):
        """Initialize vocabulary mappings for string-to-index conversion - EXACT GET VOCABULARY"""
        # Load vocabulary mappings from separate file
        self.block_symbol_to_idx = get_block_symbol_mappings()
        self.position_code_to_idx = get_position_code_mappings()
        self.entity_type_to_idx = get_entity_type_mappings()
    
    def _get_hierarchical_features(self, batch):
        """Extract hierarchical features from batch"""
        if not self.config.use_hierarchical_features:
            return None, None, None
        
        # Get hierarchical information from batch
        block_indices = getattr(batch, 'block_idx', None)
        entity_indices = getattr(batch, 'entity_idx', None)
        block_symbols = getattr(batch, 'block_symbols', None)
        pos_codes = getattr(batch, 'pos_code', None)
        
        # Convert block symbols to indices
        block_symbol_ids = None
        if block_symbols is not None and block_indices is not None:
            # Convert list of block symbols to indices
            block_symbol_ids = torch.zeros_like(block_indices)
            for i, symbol in enumerate(block_symbols):
                if i < len(block_symbols):
                    # Handle case where symbol might be a list
                    if isinstance(symbol, list):
                        symbol = symbol[0] if symbol else '?'
                    symbol_idx = self.block_symbol_to_idx.get(symbol, self.block_symbol_to_idx['?'])
                    block_symbol_ids[block_indices == i] = symbol_idx
        
        # Convert position codes to indices
        position_code_ids = None
        if pos_codes is not None:
            # Create tensor with same length as pos_codes
            position_code_ids = torch.zeros(len(pos_codes), dtype=torch.long, device=block_symbol_ids.device)
            for i, code in enumerate(pos_codes):
                if i < len(pos_codes):
                    # Handle case where code might be a list
                    if isinstance(code, list):
                        code = code[0] if code else 'm'
                    code_idx = self.position_code_to_idx.get(code, self.position_code_to_idx['m'])
                    position_code_ids[i] = code_idx
        
        # Get entity indices (already numeric)
        entity_ids = entity_indices
        
        return block_symbol_ids, position_code_ids, entity_ids


class PretrainingTasks(nn.Module):
    """Pretraining tasks for universal molecular representations"""
    
    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        self.task_heads = nn.ModuleDict()
        
        # Initialize task heads
        for task in config.pretraining_tasks:
            if task == "long_range_distance":
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(config.graph_dim * 2, config.mlp_hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.mlp_hidden_size, 1)
                )
            elif task == "short_range_distance":
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(config.graph_dim * 2, config.mlp_hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.mlp_hidden_size, 1)
                )
            elif task == "mlm":
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(config.graph_dim, config.mlp_hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.mlp_hidden_size, config.max_atomic_num)  # Use max_atomic_num for vocabulary size
                )
            elif task == "coordinate_denoising":
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(config.graph_dim, config.mlp_hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.mlp_hidden_size, 1)  # Predict energy (scalar)
                )
    
    def forward(self, node_embeddings, batch):
        """Compute pretraining task losses"""
        results = {}
        
        for task in self.config.pretraining_tasks:
            if task == "long_range_distance":
                results[task] = self.long_range_distance_loss(node_embeddings, batch)
            elif task == "short_range_distance":
                results[task] = self.short_range_distance_loss(node_embeddings, batch)
            elif task == "mlm":
                results[task] = self.mlm_loss(node_embeddings, batch)
            elif task == "coordinate_denoising":
                results[task] = self.coordinate_denoising_loss(node_embeddings, batch)
        
        # Debug logging to understand which tasks are contributing (reduced verbosity)
        active_tasks = [task for task, loss in results.items() if loss is not None]
        if len(active_tasks) != len(self.config.pretraining_tasks):
            missing_tasks = [task for task in self.config.pretraining_tasks if task not in active_tasks]
            print(f"WARNING: Missing tasks: {missing_tasks}")
        
        return results
    
    def long_range_distance_loss(self, node_embeddings, data):
        """Long-range distance prediction task"""
        if not hasattr(data, 'edge_index') or data.edge_index.shape[1] == 0:
            return None
        
        edge_index = data.edge_index
        distances = torch.norm(data.pos[edge_index[0]] - data.pos[edge_index[1]], dim=1)
        
        # Sample long-range pairs (distance > 5.0)
        long_range_mask = distances > 5.0
        if long_range_mask.sum() == 0:
            # If no long-range edges, sample some random pairs for long-range prediction
            num_nodes = node_embeddings.shape[0]
            if num_nodes < 2:
                return None
            
            # Create deterministic random pairs using a fixed seed based on batch info
            # This makes the sampling more stable across epochs
            torch.manual_seed(hash(str(edge_index.shape[1])) % 2**32)
            num_pairs = min(8, num_nodes * (num_nodes - 1) // 2)  # Reduced from 10
            all_pairs = torch.combinations(torch.arange(num_nodes), r=2)
            if all_pairs.shape[0] > num_pairs:
                random_indices = torch.randperm(all_pairs.shape[0])[:num_pairs]
                long_range_edges = all_pairs[random_indices].t()
            else:
                long_range_edges = all_pairs.t()
            
            # Compute distances for these pairs
            long_range_distances = torch.norm(data.pos[long_range_edges[0]] - data.pos[long_range_edges[1]], dim=1)
        else:
            long_range_edges = edge_index[:, long_range_mask]
            long_range_distances = distances[long_range_mask]
        
        # Ensure indices are within bounds
        max_node_idx = node_embeddings.shape[0] - 1
        valid_src_mask = long_range_edges[0] <= max_node_idx
        valid_dst_mask = long_range_edges[1] <= max_node_idx
        valid_mask = valid_src_mask & valid_dst_mask
        
        if valid_mask.sum() == 0:
            return None
        
        # Filter to valid edges only
        long_range_edges = long_range_edges[:, valid_mask]
        long_range_distances = long_range_distances[valid_mask]
        
        # Get node embeddings for long-range pairs
        src_embeddings = node_embeddings[long_range_edges[0]]
        dst_embeddings = node_embeddings[long_range_edges[1]]
        pair_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Predict distances
        distance_logits = self.task_heads["long_range_distance"](pair_embeddings)
        distance_preds = distance_logits.squeeze()
        
        # Stabilize the long-range distance prediction
        # Use consistent normalization and remove random noise
        max_distance = 10.0  # Reasonable upper bound for molecular distances
        normalized_distances = torch.clamp(long_range_distances / max_distance, 0.0, 1.0)
        normalized_preds = torch.clamp(distance_preds / max_distance, 0.0, 1.0)
        
        # Use smooth L1 loss for stability (less sensitive to outliers)
        loss = F.smooth_l1_loss(normalized_preds, normalized_distances, beta=0.1)
        
        # Add small amount of L2 regularization to prevent extreme predictions
        l2_reg = 0.01 * torch.mean(distance_preds ** 2)
        loss = loss + l2_reg
        return loss
    
    def short_range_distance_loss(self, node_embeddings, data):
        """Short-range distance prediction task"""
        if not hasattr(data, 'edge_index') or data.edge_index.shape[1] == 0:
            return None
        
        edge_index = data.edge_index
        distances = torch.norm(data.pos[edge_index[0]] - data.pos[edge_index[1]], dim=1)
        
        # Sample short-range pairs (distance <= 5.0)
        short_range_mask = distances <= 5.0
        if short_range_mask.sum() == 0:
            return None
        
        short_range_edges = edge_index[:, short_range_mask]
        short_range_distances = distances[short_range_mask]
        
        # Ensure indices are within bounds
        max_node_idx = node_embeddings.shape[0] - 1
        valid_src_mask = short_range_edges[0] <= max_node_idx
        valid_dst_mask = short_range_edges[1] <= max_node_idx
        valid_mask = valid_src_mask & valid_dst_mask
        
        if valid_mask.sum() == 0:
            return None
        
        # Filter to valid edges only
        short_range_edges = short_range_edges[:, valid_mask]
        short_range_distances = short_range_distances[valid_mask]
        
        # Get node embeddings for short-range pairs
        src_embeddings = node_embeddings[short_range_edges[0]]
        dst_embeddings = node_embeddings[short_range_edges[1]]
        pair_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Predict distances
        distance_logits = self.task_heads["short_range_distance"](pair_embeddings)
        distance_preds = distance_logits.squeeze()
        
        # Make short-range prediction harder with more challenging targets
        # Use inverse distance as target (harder to predict)
        epsilon = 1e-6
        target_inverse = 1.0 / (short_range_distances + epsilon)
        pred_inverse = 1.0 / (distance_preds + epsilon)
        
        # Use Huber loss for more robust training
        loss = F.huber_loss(pred_inverse, target_inverse, delta=0.1)
        return loss
    
    def mlm_loss(self, node_embeddings, data):
        """Masked Language Modeling task"""
        # Check for MLM mask and original types (created by MaskAtomTypes transform)
        if not hasattr(data, 'mlm_mask') or not hasattr(data, 'original_types'):
            return None
        
        mlm_mask = data.mlm_mask
        original_types = data.original_types
        
        if mlm_mask.sum() == 0:
            return None
        
        # Handle batch structure - the mask and types are per-molecule, but node_embeddings might be batched
        # We need to ensure the mask length matches the node_embeddings length
        if len(mlm_mask) != node_embeddings.shape[0]:
            # If lengths don't match, we need to handle batching properly
            # For now, let's use the minimum length to avoid indexing errors
            min_length = min(len(mlm_mask), node_embeddings.shape[0])
            mlm_mask = mlm_mask[:min_length]
            original_types = original_types[:min_length]
        
        # Ensure indices are within bounds
        max_node_idx = node_embeddings.shape[0] - 1
        valid_mask = mlm_mask & (torch.arange(len(mlm_mask), device=mlm_mask.device) <= max_node_idx)
        
        if valid_mask.sum() == 0:
            return None
        
        # Get masked node embeddings and labels
        masked_embeddings = node_embeddings[valid_mask]
        masked_labels = original_types[valid_mask]
        
        # Predict atom types
        atom_type_logits = self.task_heads["mlm"](masked_embeddings)
        
        # Make MLM harder by using label smoothing and focal loss
        # Add label smoothing to make the task more challenging
        label_smoothing = 0.1
        num_classes = atom_type_logits.shape[1]
        
        # Convert labels to one-hot for label smoothing
        one_hot_labels = torch.zeros_like(atom_type_logits)
        one_hot_labels.scatter_(1, masked_labels.unsqueeze(1), 1)
        
        # Apply label smoothing
        smoothed_labels = one_hot_labels * (1 - label_smoothing) + label_smoothing / num_classes
        
        # Use KL divergence instead of cross-entropy for more challenging optimization
        log_probs = F.log_softmax(atom_type_logits, dim=1)
        loss = F.kl_div(log_probs, smoothed_labels, reduction='batchmean')
        
        return loss
    
    def coordinate_denoising_loss(self, node_embeddings, data):
        """Coordinate denoising task - adapted for molecular graphs"""
        try:
            if not hasattr(data, 'pos') or data.pos is None:
                return None
            
            pos = data.pos
            num_nodes = pos.shape[0]
            
            if num_nodes < 2:
                return None
            
            # Handle batching: ensure node_embeddings match pos
            if node_embeddings.shape[0] != num_nodes:
                if node_embeddings.shape[0] < num_nodes:
                    # Pad embeddings to match positions
                    padding_size = num_nodes - node_embeddings.shape[0]
                    padding = torch.zeros(padding_size, node_embeddings.shape[1], device=node_embeddings.device)
                    node_embeddings = torch.cat([node_embeddings, padding], dim=0)
                else:
                    # Truncate embeddings to match positions
                    node_embeddings = node_embeddings[:num_nodes]
            
            # Make the task much harder - predict energy from coordinates directly
            # Add significant noise to coordinates
            noise_level = 0.5  # Much larger noise for harder task
            noise = torch.randn_like(pos) * noise_level
            noisy_pos = pos + noise
            noisy_pos.requires_grad_(True)
            
            # Predict energy from noisy coordinates (forces learning of coordinate-energy relationships)
            # Use a simple energy function based on pairwise distances (more efficient)
            # Sample subset of nodes for energy computation to avoid memory issues
            if num_nodes > 100:
                sample_indices = torch.randperm(num_nodes)[:100]
                sample_pos = noisy_pos[sample_indices]
                pairwise_dist = torch.cdist(sample_pos, sample_pos)
            else:
                pairwise_dist = torch.cdist(noisy_pos, noisy_pos)
            
            # Create a more stable energy: mean of distances (not inverse)
            energy_from_coords = torch.mean(pairwise_dist)
            
            # Predict energy from node embeddings
            pred_energy = self.task_heads["coordinate_denoising"](node_embeddings).mean()
            
            # The task: predict the energy that would come from the noisy coordinates
            loss = F.mse_loss(pred_energy, energy_from_coords.detach())
            
            # Scale down the loss to reasonable range
            loss = loss * 0.1
            
            # Debug logging
            if torch.rand(1) < 0.1:
                print(f"Debug: coordinate_denoising_loss = {loss.item():.4f}, energy_from_coords = {energy_from_coords.item():.4f}, pred_energy = {pred_energy.item():.4f}")
            
            return loss
            
        except Exception as e:
            print(f"Warning: coordinate_denoising_loss failed: {e}")
            return None


def create_pretraining_config(config_dict: Dict[str, Any]) -> PretrainingConfig:
    """Create pretraining config from dictionary"""
    return PretrainingConfig(**config_dict)