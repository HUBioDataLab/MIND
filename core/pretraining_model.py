import torch
import math
import random
import numpy as np
import pytorch_lightning as pl
import bitsandbytes as bnb
import torch_geometric

from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import radius_graph
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from esa.utils.norm_layers import BN, LN
from esa.masked_layers import ESA
from esa.mlp_utils import SmallMLP, GatedMLPMulti
from data_loading.gaussian import GaussianLayer

from esa.utils.reporting import (
    get_cls_metrics_binary_pt,
    get_cls_metrics_multilabel_pt,
    get_cls_metrics_multiclass_pt,
    get_regr_metrics_pt,
)

# Positional encodings - required imports (no fallbacks)
from esa.utils.posenc_encoders.laplace_pos_encoder import LapPENodeEncoder
from esa.utils.posenc_encoders.kernel_pos_encoder import KernelPENodeEncoder


@dataclass
class PretrainingConfig:
    """Configuration class for pretraining tasks"""
    # General model config
    num_features: int = 128
    graph_dim: int = 512  # Match YAML default
    edge_dim: int = 64
    batch_size: int = 256  # Match YAML default
    lr: float = 0.0005  # Match YAML default
    monitor_loss_name: str = "train_total_loss"  # Match YAML default
    xformers_or_torch_attn: str = "xformers"
    hidden_dims: List[int] = None
    num_heads: List[int] = None
    num_sabs: int = 4
    sab_dropout: float = 0.0
    mab_dropout: float = 0.0
    pma_dropout: float = 0.0
    apply_attention_on: str = "node"  # Match YAML default
    layer_types: List[str] = None
    use_mlps: bool = True
    set_max_items: int = 0
    early_stopping_patience: int = 30
    optimiser_weight_decay: float = 1e-10  # Match YAML default
    num_workers: int = 4
    mlp_hidden_size: int = 512  # Match YAML default
    mlp_type: str = "standard"
    attn_residual_dropout: float = 0.0
    norm_type: str = "LN"
    triu_attn_mask: bool = False
    output_save_dir: str = None
    use_bfloat16: bool = False  # Match YAML default
    is_node_task: bool = False
    use_posenc: bool = False
    num_mlp_layers: int = 3
    pre_or_post: str = "pre"
    pma_residual_dropout: float = 0
    use_mlp_ln: bool = False
    mlp_dropout: float = 0
    
    # Run-level / project config
    seed: int = 42
    dataset: str = "QM9"
    dataset_download_dir: str = "./data"
    out_path: str = "./outputs/qm9_pretrain"
    wandb_project_name: str = "esa-pretraining"
    wandb_run_name: str = ""
    gradient_clip_val: float = 0.5
    max_epochs: int = 100
    
    # Data/loader
    num_workers: int = 4
    
    # Debug controls (optional, can be set via YAML/JSON)
    debug_subset_n: Optional[int] = None
    debug_verbose: bool = False
    
    # Universal molecular configs
    atom_types: int = 119  # Maximum atom types (universal periodic table)
    bond_types: int = 4    # Single, double, triple, aromatic
    molecule_max_atoms: int = 500  # Maximum atoms per molecular system
    
    # 3D geometric configs
    use_3d_coordinates: bool = True
    coordinate_dim: int = 3
    gaussian_kernels: int = 128
    cutoff_distance: float = 5.0
    max_neighbors: int = 16  # Match YAML default
    
    # Pretraining task configs
    pretraining_tasks: List[str] = None  # ["long_range_distance", "short_range_distance", "mlm", "bond_prediction"]
    task_weights: Dict[str, float] = None  
    
    # Distance prediction
    distance_prediction_weight: float = 1.0
    distance_bins: int = 16  # Match YAML default
    max_distance: float = 10.0  
    
    # Masked language modeling
    mlm_weight: float = 1.0
    mlm_mask_ratio: float = 0.15
    
    # Temperature for softmax (used in distance prediction)
    temperature: float = 0.1
    
    # Dataset caching control
    use_dataset_cache: bool = True  # Whether to use cached processed datasets
    
def nearest_multiple_of_8(n):
    return math.ceil(n / 8) * 8


class UniversalMolecularEncoder(nn.Module):
    """Universal molecular encoder for all molecular systems (domain-agnostic)"""
    
    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        
        # Universal molecular encoder (works for all molecular systems)
        self.molecule_encoder = self._create_molecule_encoder()
        
        # 3D geometric features
        if config.use_3d_coordinates:
            self.gaussian_layer = GaussianLayer(
                K=config.gaussian_kernels,
                edge_types=config.atom_types * config.atom_types
            )
            # 5 invariant features + 3 simple edge features
            self.coordinate_projection = nn.Linear(5 + 3, config.hidden_dims[0])
        else:
            self.gaussian_layer = None
            self.coordinate_projection = None
        
        # Position encodings
        self.rwse_encoder = None
        self.lap_encoder = None
        if config.use_posenc:
            if KernelPENodeEncoder is not None:
                self.rwse_encoder = KernelPENodeEncoder()
            if LapPENodeEncoder is not None:
                self.lap_encoder = LapPENodeEncoder()
    
    def _create_molecule_encoder(self):
        """Create encoder for small molecules with reliable atomic features"""
        # Just atomic number embedding + period (most reliable features)
        self.atom_embedding = nn.Embedding(self.config.atom_types, self.config.hidden_dims[0] - 8)
        self.period_embedding = nn.Embedding(8, 8)  # Period is very reliable (1-7)
        
        return nn.Sequential(
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[0])
        )
    
    
    def forward(self, x, pos=None, batch=None, batch_obj=None):
        """
        Forward pass for universal molecular encoding (domain-agnostic)
        
        Args:
            x: Node features (atomic numbers for all molecular systems)
            pos: 3D coordinates (optional)
            batch: Batch indices (tensor)
            batch_obj: Full batch object (for accessing edge_index)
        """
        # Handle different input formats
        if x is None:
            raise ValueError("Input features cannot be None")
        
        # Universal approach: all molecular systems use atomic numbers
        if x.dtype == torch.long:
            # Use atomic numbers with rich chemical features (universal)
            encoded = self._encode_molecule_features(x)
        else:
            # If x is already feature-based, use molecule encoder
            encoded = self.molecule_encoder(x)
        
        # Add 3D geometric features if available
        if pos is not None and self.config.use_3d_coordinates:
            # Expect a 1D batch index vector for to_dense_batch
            batch_vec = getattr(batch, 'batch', batch) if batch is not None else batch
            geometric_features = self._compute_geometric_features(x, pos, batch_vec, batch_obj)
            encoded = encoded + geometric_features
        
        # Add position encodings
        if self.lap_encoder is not None and hasattr(batch_obj, 'EigVals'):
            lap_pos_enc = self.lap_encoder(batch_obj.EigVals, batch_obj.EigVecs)
            encoded = torch.cat((encoded, lap_pos_enc), 1)
        
        if self.rwse_encoder is not None and hasattr(batch_obj, 'pestat_RWSE'):
            rwse_pos_enc = self.rwse_encoder(batch_obj.pestat_RWSE)
            encoded = torch.cat((encoded, rwse_pos_enc), 1)
        
        return encoded
    
    def _encode_molecule_features(self, atomic_numbers):
        """Encode atomic numbers with reliable chemical features"""
        # Simple, reliable period mapping for common atoms
        period_map = {
            1: 1,   # H
            6: 2, 7: 2, 8: 2, 9: 2,    # C, N, O, F (period 2)
            15: 3, 16: 3, 17: 3,       # P, S, Cl (period 3)
        }
        
        device = atomic_numbers.device
        
        # Main atomic embeddings
        atom_emb = self.atom_embedding(atomic_numbers)
        
        # Simple period feature (very reliable) - VECTORIZED for speed
        periods = torch.zeros_like(atomic_numbers)
        for atomic_num, period in period_map.items():
            periods[atomic_numbers == atomic_num] = period
        period_emb = self.period_embedding(periods)
        
        # Concatenate features
        combined_features = torch.cat([
            atom_emb,      # [batch_size, hidden_dim-8]
            period_emb,    # [batch_size, 8]
        ], dim=-1)       # [batch_size, hidden_dim]
        
        # Pass through molecule encoder
        return self.molecule_encoder(combined_features)
    

    
    def _compute_chemical_coordination(self, x_dense, batch):
        """Compute true chemical coordination numbers from molecular graph topology"""
        # Get dimensions first (always needed)
        batch_size = x_dense.size(0)
        max_nodes = x_dense.size(1)
        device = x_dense.device
        
        # Try to get edge_index from batch data
        if hasattr(batch, 'edge_index') and batch.edge_index is not None:
            from torch_scatter import scatter_add
            
            edge_index = batch.edge_index
            batch_idx = getattr(batch, 'batch', None)
            
            if batch_idx is not None:
                # ✅ ROBUST: Handle bidirectional edges correctly
                # Only count unique edges by keeping edges where src < dst
                src, dst = edge_index[0], edge_index[1]
                
                # Create unique edge mask (avoid double counting)
                unique_edge_mask = src < dst  # Only count edge (i,j) if i < j
                if unique_edge_mask.sum() == 0:
                    # If no edges satisfy src < dst, edges might be unidirectional already
                    unique_edge_mask = torch.ones_like(src, dtype=torch.bool)
                
                unique_src = src[unique_edge_mask]
                unique_dst = dst[unique_edge_mask]
                
                # Count coordination for each node
                num_nodes = len(batch_idx)
                coordination_sparse = torch.zeros(num_nodes, device=device, dtype=torch.float)
                
                # Each unique edge contributes 1 to both src and dst coordination
                coordination_sparse.scatter_add_(0, unique_src, torch.ones_like(unique_src, dtype=torch.float))
                coordination_sparse.scatter_add_(0, unique_dst, torch.ones_like(unique_dst, dtype=torch.float))
                
                # Convert to dense format [batch_size, max_nodes]
                coordination_dense = torch.zeros(batch_size, max_nodes, device=device)
                
                # Fill dense tensor using batch indices
                node_idx = 0
                for graph_idx in range(batch_size):
                    nodes_in_graph = (batch_idx == graph_idx).sum().item()
                    if nodes_in_graph > 0:
                        end_idx = node_idx + nodes_in_graph
                        actual_nodes = min(nodes_in_graph, max_nodes)
                        coordination_dense[graph_idx, :actual_nodes] = coordination_sparse[node_idx:node_idx + actual_nodes]
                    node_idx += nodes_in_graph
                
                return coordination_dense
            else:
                # Single graph case - simpler
                src, dst = edge_index[0], edge_index[1]
                unique_edge_mask = src < dst
                if unique_edge_mask.sum() == 0:
                    unique_edge_mask = torch.ones_like(src, dtype=torch.bool)
                
                unique_src = src[unique_edge_mask]
                unique_dst = dst[unique_edge_mask]
                
                coordination = torch.zeros(max_nodes, device=device, dtype=torch.float)
                coordination.scatter_add_(0, unique_src, torch.ones_like(unique_src, dtype=torch.float))
                coordination.scatter_add_(0, unique_dst, torch.ones_like(unique_dst, dtype=torch.float))
                
                return coordination.unsqueeze(0).expand(batch_size, -1)
        else:
            # No edge information available, return zeros
            return torch.zeros(batch_size, max_nodes, device=x_dense.device)
    
    def _compute_geometric_features(self, x, pos, batch, batch_obj=None):
        """Compute SE(3) invariant geometric features - FIXED VERSION"""
        # Convert to dense format (let to_dense_batch pick the correct max per-batch)
        x_dense, batch_mask = to_dense_batch(x, batch, fill_value=0)
        pos_dense, _ = to_dense_batch(pos, batch, fill_value=0)
        
        n_graph, n_node = x_dense.size()
        device = x_dense.device
        
        # ✅ OPTIMIZED: Use pre-computed edge_index from preprocessing (no redundant computation)
        # The edge_index is already computed efficiently in cache_to_pyg.py using radius_graph
        if batch_obj is not None and hasattr(batch_obj, 'edge_index'):
            edge_index = batch_obj.edge_index
        else:
            # Fallback: create empty edge index if no edge information available
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Compute distances only for edges (much more efficient)
        if edge_index.size(1) > 0:
            # Use pre-computed edge_index with current positions
            edge_src = pos[edge_index[0]]
            edge_dst = pos[edge_index[1]]
            edge_distances = torch.norm(edge_src - edge_dst, dim=1)
        else:
            edge_distances = torch.empty(0, device=device)
        
        # ✅ FIXED: Compute coordination features efficiently
        close_cutoff = getattr(self.config, 'close_cutoff', 3.0)
        medium_cutoff = self.config.cutoff_distance if hasattr(self.config, 'cutoff_distance') else 6.0
        
        # Initialize coordination counters
        close_coordination = torch.zeros(n_graph, n_node, device=device)
        medium_coordination = torch.zeros(n_graph, n_node, device=device)
        min_distances = torch.full((n_graph, n_node), float('inf'), device=device)
        mean_distances = torch.zeros(n_graph, n_node, device=device)
        
        # ✅ FIXED: Efficient neighbor counting using pre-computed edge_index
        if edge_index.size(1) > 0:
            # Use batch mapping from the data
            if batch_obj is not None and hasattr(batch_obj, 'batch'):
                batch_mapping = batch_obj.batch
                edge_batch_idx = batch_mapping[edge_index[0]]  # Batch index for each edge
            else:
                # Fallback: assume all nodes belong to batch 0
                edge_batch_idx = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
            edge_node_src = edge_index[0]
            edge_node_dst = edge_index[1]
            
            # Count neighbors in different distance ranges
            close_mask = (edge_distances < close_cutoff) & (edge_distances > 0.5)
            medium_mask = (edge_distances < medium_cutoff) & (edge_distances > 0.5)
            
            # Use scatter operations for efficient counting
            from torch_scatter import scatter_add
            
            # Count close neighbors
            if close_mask.sum() > 0:
                close_src = edge_node_src[close_mask]
                close_dst = edge_node_dst[close_mask]
                close_batch = edge_batch_idx[close_mask]
                
                # Count for source nodes
                close_counts_src = scatter_add(torch.ones_like(close_src, dtype=torch.float), 
                                             close_batch * n_node + (close_src % n_node), 
                                             dim=0, dim_size=n_graph * n_node).view(n_graph, n_node)
                close_coordination += close_counts_src
                
                # Count for destination nodes  
                close_counts_dst = scatter_add(torch.ones_like(close_dst, dtype=torch.float),
                                             close_batch * n_node + (close_dst % n_node),
                                             dim=0, dim_size=n_graph * n_node).view(n_graph, n_node)
                close_coordination += close_counts_dst
            
            # Count medium neighbors
            if medium_mask.sum() > 0:
                medium_src = edge_node_src[medium_mask]
                medium_dst = edge_node_dst[medium_mask]
                medium_batch = edge_batch_idx[medium_mask]
                
                medium_counts_src = scatter_add(torch.ones_like(medium_src, dtype=torch.float),
                                              medium_batch * n_node + (medium_src % n_node),
                                              dim=0, dim_size=n_graph * n_node).view(n_graph, n_node)
                medium_coordination += medium_counts_src
                
                medium_counts_dst = scatter_add(torch.ones_like(medium_dst, dtype=torch.float),
                                              medium_batch * n_node + (medium_dst % n_node),
                                              dim=0, dim_size=n_graph * n_node).view(n_graph, n_node)
                medium_coordination += medium_counts_dst
            
            # ✅ FIXED: Compute distance statistics efficiently
            if close_mask.sum() > 0:
                close_dists = edge_distances[close_mask]
                close_src_nodes = edge_node_src[close_mask]
                close_dst_nodes = edge_node_dst[close_mask]
                close_batch_nodes = edge_batch_idx[close_mask]
                
                # Min distances using scatter_min
                from torch_scatter import scatter_min
                min_vals, _ = scatter_min(close_dists, close_batch_nodes * n_node + (close_src_nodes % n_node),
                                         dim=0, dim_size=n_graph * n_node)
                min_vals = min_vals.view(n_graph, n_node)
                min_distances = torch.minimum(min_distances, min_vals)
                
                # Mean distances using scatter_mean
                from torch_scatter import scatter_mean
                mean_vals = scatter_mean(close_dists, close_batch_nodes * n_node + (close_src_nodes % n_node),
                                       dim=0, dim_size=n_graph * n_node).view(n_graph, n_node)
                mean_distances = torch.where(close_coordination > 0, mean_vals, mean_distances)
        
        # Handle infinite min distances
        min_distances = torch.where(min_distances == float('inf'), torch.zeros_like(min_distances), min_distances)
        
        # ✅ FIXED: Compute chemical coordination (unchanged)
        chemical_coordination = self._compute_chemical_coordination(x_dense, batch)
        
        # ✅ SIMPLIFIED: Use simple distance-based features instead of complex GaussianLayer
        if edge_index.size(1) > 0:
            # Simple distance-based edge features
            # Use scatter operations to aggregate edge distances to nodes
            from torch_scatter import scatter_add
            
            # Aggregate distances to source nodes
            node_dist_src = scatter_add(edge_distances, edge_batch_idx * n_node + (edge_node_src % n_node),
                                      dim=0, dim_size=n_graph * n_node).view(n_graph, n_node)
            
            # Aggregate distances to destination nodes
            node_dist_dst = scatter_add(edge_distances, edge_batch_idx * n_node + (edge_node_dst % n_node),
                                      dim=0, dim_size=n_graph * n_node).view(n_graph, n_node)
            
            # Combine and create simple features
            total_distances = node_dist_src + node_dist_dst
            avg_distances = total_distances / (close_coordination + medium_coordination + 1e-8)
            
            # Create simple edge features: [total_dist, avg_dist, num_edges]
            aggregated_edge_features = torch.stack([
                total_distances,  # Sum of all edge distances to this node
                avg_distances,    # Average edge distance to this node
                close_coordination + medium_coordination  # Total number of edges
            ], dim=-1)  # [n_graph, n_node, 3]
        else:
            # Fallback: zero features when no edges
            aggregated_edge_features = torch.zeros(n_graph, n_node, 3, device=device)
        
        # ✅ FIXED: Combine all features
        invariant_features = torch.stack([
            chemical_coordination,        # True chemical bonds
            close_coordination,          # Close spatial neighbors
            medium_coordination,         # Medium range environment
            min_distances,               # Closest neighbor distance
            mean_distances,              # Average close neighbor distance
        ], dim=-1)  # [n_graph, n_node, 5]
        
        # Combine invariant node features with aggregated edge features
        combined_features = torch.cat([
            invariant_features,  # [n_graph, n_node, 5]
            aggregated_edge_features  # [n_graph, n_node, K or 1]
        ], dim=-1)  # [n_graph, n_node, 5 + K]
        
        # Project to hidden dimension
        geometric_features = self.coordinate_projection(combined_features)
        
        # Convert back to sparse format
        geometric_features = geometric_features[batch_mask]
        
        return geometric_features


class PretrainingTasks(nn.Module):
    """Module containing all pretraining tasks"""
    
    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        
        # Task-specific heads
        self.long_range_distance_head = self._create_long_range_distance_head()
        self.distance_head = self._create_distance_head()  # Used by short_range_distance
        self.mlm_head = self._create_mlm_head()
    
    def _create_long_range_distance_head(self):
        """Head for long-range distance prediction (global 3D structure learning)"""
        # Use classification like the working distance_prediction_loss
        return nn.Sequential(
            nn.Linear(self.config.graph_dim * 2, self.config.graph_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.graph_dim, self.config.graph_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim // 2, self.config.distance_bins)  # Classify into distance bins
        )    


    def _create_distance_head(self):
        """Head for distance prediction"""
        return nn.Sequential(
            nn.Linear(self.config.graph_dim * 2, self.config.graph_dim),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim, self.config.distance_bins)
        )
    

    def _create_mlm_head(self):
        """Head for masked language modeling"""
        max_types = self.config.atom_types  # Universal atomic types only
        return nn.Sequential(
            nn.Linear(self.config.graph_dim, self.config.graph_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim // 2, max_types)
        )
    


    def long_range_distance_loss(self, node_embeddings, data):
        """
        Long-range distance prediction loss for global 3D structure learning.
        
        Predicts distances between all atom pairs (or sampled pairs for large molecular systems)
        to learn overall molecular geometry. This is completely SE(3) invariant.
        """
        if not hasattr(data, 'pos') or data.pos is None:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        pos = data.pos
        num_atoms = pos.size(0)
        
        # For efficiency and memory safety, limit pairs based on batch size
        max_pairs = min(1000, num_atoms * (num_atoms - 1) // 2)
        
        # Note: Removed atom limit since we use efficient random sampling
        
        if num_atoms <= 50:
            # Small molecules: use all pairs
            i_indices, j_indices = torch.triu_indices(
                num_atoms, num_atoms, offset=1, device=node_embeddings.device
            )
        else:
            # Large molecules: use simple random sampling to avoid memory issues
            pairs = set()  # Use set to avoid duplicates
            
            # Sample random pairs for long-range structure learning
            attempts = 0
            while len(pairs) < max_pairs and attempts < max_pairs * 3:
                i, j = torch.randint(0, num_atoms, (2,), device=pos.device).tolist()
                if i != j:
                    # Ensure i < j for consistency
                    pair = (min(i, j), max(i, j))
                    pairs.add(pair)
                attempts += 1
            
            pairs = list(pairs)
            
            # Convert to tensors
            if pairs:
                i_indices, j_indices = zip(*pairs)
                i_indices = torch.tensor(i_indices, device=node_embeddings.device)
                j_indices = torch.tensor(j_indices, device=node_embeddings.device)
            else:
                return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        if len(i_indices) == 0:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        # Get node embeddings for pairs
        emb_i = node_embeddings[i_indices]
        emb_j = node_embeddings[j_indices]
        pair_embeddings = torch.cat([emb_i, emb_j], dim=-1)
        
        # Predict distance bins using long-range distance head
        distance_logits = self.long_range_distance_head(pair_embeddings)
        
        # True distances (SE(3) invariant ground truth)
        pos_i = pos[i_indices]
        pos_j = pos[j_indices]
        true_distances = torch.norm(pos_i - pos_j, dim=1)
        
        # Convert distances to bins (same as distance_prediction_loss)
        distance_bins = torch.clamp(
            (true_distances / self.config.max_distance * self.config.distance_bins).long(),
            0, self.config.distance_bins - 1
        )
        
        # Debug: Check predictions occasionally
        with torch.no_grad():
            pred_bins = distance_logits.argmax(dim=-1)
            true_mean = true_distances.mean().item()
            pred_dist_mean = (pred_bins.float() / self.config.distance_bins * self.config.max_distance).mean().item()
            
            # Log debug info occasionally
            if torch.rand(1).item() < 0.01:  # 1% chance to log
                print(f"SE3 Debug - True dist: {true_mean:.3f}Å, Pred dist: {pred_dist_mean:.3f}Å")
        
        # Use cross-entropy loss (same as distance_prediction_loss)
        loss = F.cross_entropy(distance_logits, distance_bins, reduction='mean')
        
        return loss
    

    

    def short_range_distance_loss(self, node_embeddings, edge_index, distances, mask):
        """Compute short-range distance loss (local chemical bonds)"""
        source_emb = node_embeddings[edge_index[0]]
        target_emb = node_embeddings[edge_index[1]]
        edge_emb = torch.cat([source_emb, target_emb], dim=-1)
        
        logits = self.distance_head(edge_emb)
        
        # Convert distances to bins
        distance_bins = torch.clamp(
            (distances / self.config.max_distance * self.config.distance_bins).long(),
            0, self.config.distance_bins - 1
        )
        
        loss = F.cross_entropy(logits[mask], distance_bins[mask], reduction='mean')
        return loss
    

    def mlm_loss(self, node_embeddings, data):
        """
        Compute masked language modeling loss
        """
        if not (hasattr(data, 'mlm_mask') and hasattr(data, 'original_types') and hasattr(data, 'masked_types')):
            return torch.tensor(0.0, device=node_embeddings.device)

        mask = data.mlm_mask
        original_types = data.original_types
        masked_types = getattr(data, 'masked_types', original_types)

        # Debug logging (remove after confirming)
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        if self._step_count % 100 == 1:  # Log every 100 steps
            mask_ratio = mask.sum().item() / mask.size(0)
            print(f"MLM Debug - Step {self._step_count}: Mask ratio = {mask_ratio:.3f}, Masked nodes = {mask.sum().item()}/{mask.size(0)}")

        logits = self.mlm_head(node_embeddings)
        loss = F.cross_entropy(logits[mask], original_types[mask], reduction='mean')
        return loss
    



class PretrainingESAModel(pl.LightningModule):
    """
    Universal pretraining ESA model for all molecular systems
    """
    
    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(vars(config))
        
        # Universal molecular encoder
        self.encoder = UniversalMolecularEncoder(config)
        
        # ESA backbone
        st_args = dict(
            num_outputs=32,
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
            residual_dropout=config.attn_residual_dropout,
            set_max_items=nearest_multiple_of_8(config.set_max_items + 1),
            use_bfloat16=config.use_bfloat16,
            layer_types=config.layer_types,
            num_mlp_layers=config.num_mlp_layers,
            pre_or_post=config.pre_or_post,
            pma_residual_dropout=config.pma_residual_dropout,
            use_mlp_ln=config.use_mlp_ln,
            mlp_dropout=config.mlp_dropout,
        )
        
        self.esa_backbone = ESA(**st_args)
        
        # Pretraining tasks
        self.pretraining_tasks = PretrainingTasks(config)
        
        # Output projection for node-level tasks
        if config.apply_attention_on == "edge":
            # For edge attention, input is concatenated source and target embeddings
            # We'll create the MLP dynamically in forward pass to handle variable edge dimensions
            self.node_edge_mlp = None  # Will be created dynamically
        else:
            if config.mlp_type in ["standard", "gated_mlp"]:
                # In node attention mode, the encoder outputs node embeddings of size graph_dim
                self.node_mlp = SmallMLP(
                    in_dim=config.graph_dim,
                    inter_dim=128,
                    out_dim=config.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=config.num_mlp_layers if config.num_mlp_layers > 1 else config.num_mlp_layers + 1,
                )
        
        # Normalization
        if config.norm_type == "BN":
            norm_fn = BN
        elif config.norm_type == "LN":
            norm_fn = LN
        
        self.mlp_norm = norm_fn(config.hidden_dims[0])
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)

        # If we are in node-attention mode and need per-node outputs for a node-level task
        # (e.g., coordinate denoising), ensure we align ESA outputs back to node space.
        if self.config.apply_attention_on == "node":
            self.config.is_node_task = True
    
    def forward(self, batch):
        """
        Forward pass for pretraining
        
        Args:
            batch: PyTorch Geometric batch
        """
        edge_index, batch_mapping = batch.edge_index, batch.batch
        pos = getattr(batch, 'pos', None)
        
        # Handle different input formats for QM9
        if hasattr(batch, 'z') and batch.z is not None:
            # QM9 uses atomic numbers (z) instead of features (x)
            x = batch.z
        elif hasattr(batch, 'x') and batch.x is not None:
            x = batch.x
        else:
            raise ValueError("Batch must have either 'x' or 'z' attribute")
        
        # Encode nodes (universal approach)
        node_embeddings = self.encoder(x, pos, batch, batch)
        
        # Debug check
        if node_embeddings is None:
            raise ValueError("Encoder returned None embeddings")
        
        # Apply attention mechanism
        if self.config.apply_attention_on == "edge":
            source = node_embeddings[edge_index[0, :], :]
            target = node_embeddings[edge_index[1, :], :]
            h = torch.cat((source, target), dim=1)
            
            edge_attr = getattr(batch, 'edge_attr', None)
            if edge_attr is not None:
                h = torch.cat((h, edge_attr.float()), dim=1)
            
            # Create MLP dynamically if needed
            if self.node_edge_mlp is None:
                in_dim = h.shape[1]
                self.node_edge_mlp = SmallMLP(
                    in_dim=in_dim,
                    inter_dim=128,
                    out_dim=self.config.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=self.config.num_mlp_layers if self.config.num_mlp_layers > 1 else self.config.num_mlp_layers + 1,
                ).to(h.device)
            
            # Ensure tensors are on the same device
            device = node_embeddings.device
            h = self.node_edge_mlp(h.to(device))
            edge_index = edge_index.to(device)
            batch_mapping = batch_mapping.to(device)
            edge_batch_index = batch_mapping.index_select(0, edge_index[0, :]).to(device)

            # Determine max set size dynamically if not provided
            if self.config.set_max_items and self.config.set_max_items > 0:
                num_max_items = nearest_multiple_of_8(self.config.set_max_items + 1)
            else:
                counts = torch.bincount(edge_batch_index)
                num_max_items = int(counts.max().item()) if counts.numel() > 0 else 1
                num_max_items = max(1, num_max_items)
                num_max_items = nearest_multiple_of_8(num_max_items + 1)

            if getattr(self.config, 'debug_verbose', False):
                uniq, cnt = torch.unique(edge_batch_index, return_counts=True)
                print(f"[DEBUG] Edge mode - h:{tuple(h.shape)} edge_index:{tuple(edge_index.shape)} batches:{int(uniq.numel())} max_edges_per_graph:{int(cnt.max().item() if cnt.numel()>0 else 0)} num_max_items:{num_max_items}")
            
            h, _ = to_dense_batch(h, edge_batch_index, fill_value=0, max_num_nodes=num_max_items)
            h = self.esa_backbone(h, edge_index, batch_mapping, num_max_items=num_max_items)
        else:
            # Ensure tensors are on the same device
            device = node_embeddings.device
            h = self.mlp_norm(self.node_mlp(node_embeddings))
            batch_mapping = batch_mapping.to(device)
            if edge_index is not None:
                edge_index = edge_index.to(device)
            else:
                # Node attention path does not require edges; pass an empty tensor
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            # Determine max set size for nodes
            if self.config.set_max_items and self.config.set_max_items > 0:
                num_max_items = nearest_multiple_of_8(self.config.set_max_items + 1)
            else:
                counts = torch.bincount(batch_mapping)
                num_max_items = int(counts.max().item()) if counts.numel() > 0 else 1
                num_max_items = max(1, num_max_items)
                num_max_items = nearest_multiple_of_8(num_max_items + 1)

            if getattr(self.config, 'debug_verbose', False):
                uniq, cnt = torch.unique(batch_mapping, return_counts=True)
                print(f"[DEBUG] Node mode - h:{tuple(h.shape)} edge_index:{tuple(edge_index.shape)} batches:{int(uniq.numel())} max_nodes_per_graph:{int(cnt.max().item() if cnt.numel()>0 else 0)} num_max_items:{num_max_items}")

            h, dense_batch_index = to_dense_batch(h, batch_mapping, fill_value=0, max_num_nodes=num_max_items)
            h = self.esa_backbone(h, edge_index, batch_mapping, num_max_items=num_max_items)
            
            if self.config.is_node_task and h.dim() == 3:
                # Map dense ESA outputs back to the original node ordering only if ESA kept per-node outputs
                h = h[dense_batch_index]
        
        return h, node_embeddings
    
    def _compute_pretraining_losses(self, batch, graph_embeddings, node_embeddings):
        """Compute all pretraining task losses"""
        losses = {}
        total_loss = 0.0               

        # Long-range distance learning (global 3D structure)
        if "long_range_distance" in self.config.pretraining_tasks:
            losses['long_range_distance'] = self.pretraining_tasks.long_range_distance_loss(node_embeddings, batch)
            total_loss += self.config.task_weights['long_range_distance'] * losses['long_range_distance']
        
        # Short-range distance learning (local chemical bonds)
        if "short_range_distance" in self.config.pretraining_tasks:
            if hasattr(batch, 'edge_index') and hasattr(batch, 'pos'):
                # Compute distances from coordinates on-the-fly
                edge_index = batch.edge_index
                pos = batch.pos
                distances = torch.norm(pos[edge_index[1]] - pos[edge_index[0]], dim=1)
                
                mask = torch.rand(edge_index.size(1)) < 0.15
                dist_loss = self.pretraining_tasks.short_range_distance_loss(
                    node_embeddings, edge_index, distances, mask
                )
                losses['short_range_distance'] = dist_loss
                total_loss += self.config.task_weights['short_range_distance'] * dist_loss
        

        # Masked language modeling
        if "mlm" in self.config.pretraining_tasks:
            losses['mlm'] = self.pretraining_tasks.mlm_loss(node_embeddings, batch)
            total_loss += self.config.task_weights['mlm'] * losses['mlm']
        

        return losses, total_loss
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Debug: Calculate average node size per batch
        if hasattr(batch, 'z'):
            num_atoms = batch.z.size(0)
        elif hasattr(batch, 'x'):
            num_atoms = batch.x.size(0)
        else:
            num_atoms = 0
            
        if hasattr(batch, 'batch'):
            batch_size = batch.batch.max().item() + 1
            avg_nodes_per_molecule = num_atoms / batch_size if batch_size > 0 else 0
        else:
            batch_size = 1
            avg_nodes_per_molecule = num_atoms
            
        # Log debug info every 50 steps
        if batch_idx % 50 == 0:
            print(f"🔍 Batch {batch_idx}: {batch_size} molecules, {num_atoms} total atoms, avg {avg_nodes_per_molecule:.1f} atoms/molecule")
        
        graph_embeddings, node_embeddings = self.forward(batch)
        
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)
        
        # Log individual losses with more detailed info
        for task_name, loss_value in losses.items():
            self.log(f"train_{task_name}_loss", loss_value, prog_bar=True)
            # Also log to console occasionally for debugging
            if batch_idx % 50 == 0:
                print(f"  {task_name}: {loss_value:.4f}")
        
        # Print detailed loss breakdown occasionally
        if batch_idx % 50 == 0:
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
            print(f"Step {batch_idx} - {loss_str} | Total: {total_loss:.4f}")
        
        self.log("train_total_loss", total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        graph_embeddings, node_embeddings = self.forward(batch)
        
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)
        
        # Log individual losses
        for task_name, loss_value in losses.items():
            self.log(f"val_{task_name}_loss", loss_value)
        
        self.log("val_total_loss", total_loss)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        graph_embeddings, node_embeddings = self.forward(batch)
        
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)
        
        # Log individual losses
        for task_name, loss_value in losses.items():
            self.log(f"test_{task_name}_loss", loss_value)
        
        self.log("test_total_loss", total_loss)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Simple optimizer setup for all parameters
        optimizer = bnb.optim.AdamW8bit(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.optimiser_weight_decay,
        )

        # Cosine with warmup
        total_epochs = getattr(self.config, 'max_epochs', 100)
        warmup_frac = 0.1
        warmup_epochs = max(1, int(total_epochs * warmup_frac))

        def lr_lambda(current_epoch: int):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs)
            # Cosine decay from 1.0 to 0.1 over the remaining epochs
            progress = (current_epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return 0.1 + 0.9 * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": self.config.monitor_loss_name,
        }
        ]
    
    def get_embeddings(self, batch):
        """Get embeddings for downstream tasks"""
        with torch.no_grad():
            graph_embeddings, node_embeddings = self.forward(batch)
            return graph_embeddings, node_embeddings


def create_pretraining_config(**kwargs) -> PretrainingConfig:
    """Helper function to create pretraining configuration"""
    config = PretrainingConfig()
    
    # Update with provided kwargs
    for key, value in kwargs.items():
        # Set all keys as attributes - will fail if invalid
        setattr(config, key, value)
    
    return config


# Example usage and configuration
if __name__ == "__main__":
    config = create_pretraining_config()
    
    # Create model
    model = PretrainingESAModel(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Pretraining tasks: {config.pretraining_tasks}")
    print(f"Task weights: {config.task_weights}")