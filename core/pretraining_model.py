import torch
import math
import pytorch_lightning as pl
import bitsandbytes as bnb

from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from esa.utils.norm_layers import BN, LN
from esa.masked_layers import ESA
from esa.mlp_utils import SmallMLP, GatedMLPMulti
from data_loading.gaussian import GaussianLayer
from torch_scatter import scatter_add, scatter_min

from esa.utils.reporting import (
    get_cls_metrics_binary_pt,
    get_cls_metrics_multilabel_pt,
    get_cls_metrics_multiclass_pt,
    get_regr_metrics_pt,
)

from esa.utils.posenc_encoders.laplace_pos_encoder import LapPENodeEncoder
from esa.utils.posenc_encoders.kernel_pos_encoder import KernelPENodeEncoder


@dataclass
class PretrainingConfig:
    """Configuration class for pretraining tasks"""
    # General model config
    num_features: int = 128
    graph_dim: int = 512
    edge_dim: int = 64
    batch_size: int = 256
    lr: float = 0.0005
    monitor_loss_name: str = "val_total_loss"
    xformers_or_torch_attn: str = "xformers"
    hidden_dims: List[int] = None
    num_heads: List[int] = None
    num_sabs: int = 4
    sab_dropout: float = 0.0
    mab_dropout: float = 0.0
    pma_dropout: float = 0.0
    apply_attention_on: str = "node"
    layer_types: List[str] = None
    use_mlps: bool = True
    set_max_items: int = 0
    early_stopping_patience: int = 30
    optimiser_weight_decay: float = 1e-10
    num_workers: int = 4
    mlp_hidden_size: int = 512
    mlp_type: str = "standard"
    attn_residual_dropout: float = 0.0
    norm_type: str = "LN"
    triu_attn_mask: bool = False
    use_bfloat16: bool = False
    is_node_task: bool = False
    use_posenc: bool = False
    num_mlp_layers: int = 3
    pre_or_post: str = "pre"
    pma_residual_dropout: float = 0
    use_mlp_ln: bool = False
    mlp_dropout: float = 0
    
    # Development controls (optional)
    debug_subset_n: Optional[int] = None
    debug_verbose: bool = False
    
    # Universal molecular configs
    atom_types: int = 121  # Maximum atom types (119 real elements + 1 virtual atom + 1 mask token)
    bond_types: int = 4    # Single, double, triple, aromatic
    molecule_max_atoms: int = 500  # Maximum atoms per molecular system
    
    # 3D geometric configs
    use_3d_coordinates: bool = True
    coordinate_dim: int = 3
    gaussian_kernels: int = 128
    cutoff_distance: float = 5.0
    max_neighbors: int = 16
    
    # Pretraining task configs
    pretraining_tasks: List[str] = None  # ["long_range_distance", "short_range_distance", "mlm", "bond_prediction"]
    task_weights: Dict[str, float] = None  
    
    # Distance prediction
    distance_prediction_weight: float = 1.0
    distance_bins: int = 16
    max_distance: float = 10.0
    
    # Long-range distance loss configuration
    long_range_ranking_margin: float = 1.0  
    
    # Masked language modeling
    mlm_weight: float = 1.0
    mlm_mask_ratio: float = 0.15
    
    # Temperature for softmax (used in distance prediction)
    temperature: float = 0.1
    
    # Per-type loss tracking (for multi-domain analysis)
    compute_per_type_losses: bool = False
    log_per_type_frequency: int = 10
    
    
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
            # 5 invariant features + gaussian_kernels edge features
            self.coordinate_projection = nn.Linear(5 + config.gaussian_kernels, config.hidden_dims[0])
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
    
    
    def forward(self, x, pos=None, batch=None):
        """
        Forward pass for universal molecular encoding (domain-agnostic)
        
        Args:
            x: Node features (atomic numbers for all molecular systems)
            pos: 3D coordinates (optional)
            batch: Batch indices
        """
        # Handle different input formats
        assert x is not None, "Input features cannot be None"
        assert x.dtype == torch.long, f"Expected atomic numbers (torch.long), got {x.dtype}"
        
        # Universal approach: all molecular systems use atomic numbers
        # Use atomic numbers with rich chemical features (universal)
        encoded = self._encode_molecule_features(x)
        
        # Add 3D geometric features if available
        if pos is not None and self.config.use_3d_coordinates:
            # Expect a 1D batch index vector for to_dense_batch
            batch_vec = getattr(batch, 'batch', batch)
            geometric_features = self._compute_geometric_features(x, pos, batch_vec, batch)
            encoded = encoded + geometric_features
        
        # Add position encodings
        if self.lap_encoder is not None and hasattr(batch, 'EigVals'):
            lap_pos_enc = self.lap_encoder(batch.EigVals, batch.EigVecs)
            encoded = torch.cat((encoded, lap_pos_enc), 1)
        
        if self.rwse_encoder is not None and hasattr(batch, 'pestat_RWSE'):
            rwse_pos_enc = self.rwse_encoder(batch.pestat_RWSE)
            encoded = torch.cat((encoded, rwse_pos_enc), 1)
        
        return encoded
    
    def _encode_molecule_features(self, atomic_numbers):
        """Encode atomic numbers with reliable chemical features"""
        # Period mapping for common atoms
        period_map = {
            1: 1,   # H
            6: 2, 7: 2, 8: 2, 9: 2,    # C, N, O, F (period 2)
            15: 3, 16: 3, 17: 3,       # P, S, Cl (period 3)
        }
        
        device = atomic_numbers.device
        
        # Main atomic embeddings
        atom_emb = self.atom_embedding(atomic_numbers)
        
        # Period feature computation - vectorized for efficiency
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
        assert hasattr(batch, 'edge_index') and batch.edge_index is not None, "Edge index is required for chemical coordination computation"
        # from torch_scatter import scatter_add
        
        edge_index = batch.edge_index
        batch_idx = getattr(batch, 'batch', None)
        
        if batch_idx is not None:
            # Handle bidirectional edges correctly
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
    
    def _compute_geometric_features(self, x, pos, batch, batch_obj=None):
        """
        Compute SE(3) invariant geometric features using existing radius graph edges
        
        RNA-SPECIFIC OPTIMIZATION:
        For RNA structures (>1000 atoms), uses sparse edge_index to avoid OOM errors.
        For other modalities (proteins, small molecules), uses original dense computation.
        """
        # Convert to dense format for per-node features
        x_dense, batch_mask = to_dense_batch(x, batch, fill_value=0)
        pos_dense, _ = to_dense_batch(pos, batch, fill_value=0)
        
        n_graph, n_node = x_dense.size()
        
        # Use existing radius graph edges for efficient computation
        assert batch_obj is not None, "Batch object is required for geometric features"
        assert hasattr(batch_obj, 'edge_index'), "Edge index is required"
        
        edge_index = batch_obj.edge_index
        
        # =====================================================================
        # RNA-SPECIFIC CHECK: Use sparse computation for large structures
        # =====================================================================
        # Detect RNA by checking dataset_type attribute or molecule size
        is_rna = False
        if hasattr(batch_obj, 'dataset_type'):
            dataset_type = batch_obj.dataset_type
            if isinstance(dataset_type, (list, tuple)):
                is_rna = any(dt.upper() == 'RNA' for dt in dataset_type)
            elif isinstance(dataset_type, str):
                is_rna = dataset_type.upper() == 'RNA'
        
        # Also check molecule size (RNA structures typically > 500 atoms)
        num_atoms = len(x)
        is_large_molecule = num_atoms > 500
        
        # Use sparse computation for RNA or large molecules (>500 atoms)
        use_sparse_computation = is_rna or is_large_molecule
            
        
        if use_sparse_computation and num_atoms > 500:
            print(f"🧬 RNA/Large molecule detected ({num_atoms} atoms) - Using SPARSE geometric features")
            
            # =====================================================================
            # SPARSE COMPUTATION PATH (RNA & Large Molecules)
            # =====================================================================
            # Edge distances (sparse - only for connected nodes)
            edge_src, edge_dst = edge_index[0], edge_index[1]
            edge_distances = torch.norm(pos[edge_src] - pos[edge_dst], dim=1)  # [num_edges]
            
            # SE(3) invariant node-level coordination features
            # 1. Chemical bond-based coordination (true molecular graph neighbors)
            chemical_coordination = self._compute_chemical_coordination(x_dense, batch_obj)
            
            # 2. Distance-based coordination from radius graph
            # Count neighbors in different distance shells using edge_distances
            close_cutoff = getattr(self.config, 'close_cutoff', 3.0)
            
            # Create edge masks for different distance ranges
            close_mask = (edge_distances < close_cutoff) & (edge_distances > 0.5)
            medium_mask = (edge_distances < self.config.cutoff_distance) & (edge_distances > 0.5)
            
            # Count neighbors per node using scatter operations
            # from torch_scatter import scatter_add
            num_nodes = len(x)
            
            # Count close neighbors
            close_coordination_sparse = scatter_add(
                close_mask.float(), edge_src, dim=0, dim_size=num_nodes
            )
            
            # Count medium-range neighbors
            distance_coordination_sparse = scatter_add(
                medium_mask.float(), edge_src, dim=0, dim_size=num_nodes
            )
            
            # Convert to dense format [n_graph, n_node]
            close_coordination, _ = to_dense_batch(close_coordination_sparse, batch, fill_value=0)
            distance_based_coordination, _ = to_dense_batch(distance_coordination_sparse, batch, fill_value=0)
            
            # 3. Distance statistics per node (min and mean of close neighbors)
            # OPTIMIZED: Vectorized computation instead of per-node loop
            min_distances_sparse = torch.full((num_nodes,), float('inf'), device=pos.device)
            mean_distances_sparse = torch.zeros(num_nodes, device=pos.device)
            count_per_node = torch.zeros(num_nodes, device=pos.device)
            
            # Filter to close neighbors only
            close_indices = torch.where(close_mask)[0]
            if len(close_indices) > 0:
                close_src = edge_src[close_indices]
                close_dists = edge_distances[close_indices]
                
                # Min distance per node (scatter_reduce with min operation)
                # from torch_scatter import scatter_min, scatter_add
                min_distances_sparse, _ = scatter_min(close_dists, close_src, dim=0, dim_size=num_nodes)
                
                # Mean distance per node
                sum_per_node = scatter_add(close_dists, close_src, dim=0, dim_size=num_nodes)
                count_per_node = scatter_add(torch.ones_like(close_dists), close_src, dim=0, dim_size=num_nodes)
                mean_distances_sparse = sum_per_node / (count_per_node + 1e-8)
            
            # Replace inf with 0 for nodes with no close neighbors
            min_distances_sparse[torch.isinf(min_distances_sparse)] = 0.0
            
            # Convert to dense format
            min_distances, _ = to_dense_batch(min_distances_sparse, batch, fill_value=0)
            mean_distances, _ = to_dense_batch(mean_distances_sparse, batch, fill_value=0)
            
            # SE(3) invariant feature combination
            # Shape: [n_graph, n_node, feature_dim]
            invariant_features = torch.stack([
                chemical_coordination,        # True chemical bonds
                close_coordination,          # Close spatial neighbors (~3Å)
                distance_based_coordination, # Medium range environment (~cutoff)
                min_distances,               # Closest neighbor distance
                mean_distances,              # Average close neighbor distance
            ], dim=-1)  # [n_graph, n_node, 5]
            
            # =====================================================================
            # Vectorized radius graph RBF computation (GPU-optimized)
            # =====================================================================
            # Compute edge types
            edge_types = x[edge_src] * self.config.atom_types + x[edge_dst]
            
            # Apply GaussianLayer to all edges at once
            edge_dist_3d = edge_distances.unsqueeze(0).unsqueeze(0)  # [1, 1, num_edges]
            edge_type_3d = edge_types.unsqueeze(0).unsqueeze(0)      # [1, 1, num_edges]
            
            edge_rbf_4d = self.gaussian_layer(edge_dist_3d, edge_type_3d)  # [1, 1, num_edges, gaussian_kernels]
            edge_rbf = edge_rbf_4d.squeeze(0).squeeze(0)  # [num_edges, gaussian_kernels]
            
            # Aggregate RBF features to nodes using scatter operations
            gaussian_kernels = self.config.gaussian_kernels
            node_rbf_src = scatter_add(edge_rbf, edge_src, dim=0, dim_size=num_nodes)
            node_rbf_dst = scatter_add(edge_rbf, edge_dst, dim=0, dim_size=num_nodes)
            
            # Combine source and destination features
            node_rbf_combined = node_rbf_src + node_rbf_dst
            
            # Convert to dense format
            gbf_feature, _ = to_dense_batch(node_rbf_combined, batch, fill_value=0)
            
            # Apply padding mask
            padding_mask = x_dense.eq(0)
            aggregated_edge_features = gbf_feature.masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            )
        
        else:
            # =====================================================================
            # ORIGINAL DENSE COMPUTATION PATH (Proteins, Small Molecules, QM9)
            # =====================================================================
            print(f"🔬 Small/medium molecule ({num_atoms} atoms) - Using DENSE geometric features")
            
            # Compute distances and edge types for coordination features
            delta_pos = pos_dense.unsqueeze(1) - pos_dense.unsqueeze(2)
            dist = delta_pos.norm(dim=-1)  # [n_graph, n_node, n_node]
            
            # Compute edge types for Gaussian computation
            edge_type = x_dense.view(n_graph, n_node, 1) * self.config.atom_types + x_dense.view(n_graph, 1, n_node)
            
            # SE(3) invariant node-level coordination features
            # 1. Distance-based neighbors (local environment density)
            # Use chemically meaningful cutoffs
            close_cutoff = getattr(self.config, 'close_cutoff', 3.0)   # First coordination shell (~typical bond + van der Waals)
            medium_cutoff = self.config.cutoff_distance if hasattr(self.config, 'cutoff_distance') else 6.0
            
            # Count neighbors in different shells
            close_mask = (dist < close_cutoff) & (dist > 0.5)     # Close neighbors (likely bonded)
            medium_mask = (dist < medium_cutoff) & (dist > 0.5)   # Medium range (environment)
            
            close_coordination = close_mask.sum(dim=-1).float()    # [n_graph, n_node]
            distance_based_coordination = medium_mask.sum(dim=-1).float()  # [n_graph, n_node]
            
            # 2. Chemical bond-based coordination (true molecular graph neighbors)
            chemical_coordination = self._compute_chemical_coordination(x_dense, batch_obj)
            
            # SE(3) invariant distance statistics per node
            # Use close_mask for meaningful distance statistics
            masked_dist = dist.masked_fill(~close_mask, float('inf'))
            min_distances = masked_dist.min(dim=-1)[0].masked_fill(close_mask.sum(dim=-1) == 0, 0.0)
            
            masked_dist_finite = dist.masked_fill(~close_mask, 0.0)
            mean_distances = masked_dist_finite.sum(dim=-1) / (close_mask.sum(dim=-1) + 1e-8)
            
            # SE(3) invariant feature combination
            # Shape: [n_graph, n_node, feature_dim]
            invariant_features = torch.stack([
                chemical_coordination,        # True chemical bonds (sp3≈4, sp2≈3, sp≈2)
                close_coordination,          # Close spatial neighbors (~3Å, likely bonded)
                distance_based_coordination,  # Medium range environment (~6Å)
                min_distances,               # Closest neighbor distance
                mean_distances,              # Average close neighbor distance
            ], dim=-1)  # [n_graph, n_node, 5]
            
            # Vectorized radius graph RBF computation for GPU optimization
            # Process all edges simultaneously
            edge_src, edge_dst = batch_obj.edge_index  # [num_edges]
            
            # Compute edge distances and types vectorized
            edge_distances = torch.norm(pos[edge_src] - pos[edge_dst], dim=1)  # [num_edges]
            edge_types = x[edge_src] * self.config.atom_types + x[edge_dst]  # [num_edges]
            
            # Apply GaussianLayer to all edges at once
            edge_dist_3d = edge_distances.unsqueeze(0).unsqueeze(0)  # [1, 1, num_edges]
            edge_type_3d = edge_types.unsqueeze(0).unsqueeze(0)  # [1, 1, num_edges]
            
            edge_rbf_4d = self.gaussian_layer(edge_dist_3d, edge_type_3d)  # [1, 1, num_edges, gaussian_kernels]
            edge_rbf = edge_rbf_4d.squeeze(0).squeeze(0)  # [num_edges, gaussian_kernels]
            
            # Vectorized aggregation using scatter operations
            # from torch_scatter import scatter_add
            
            # Initialize RBF features for all nodes
            gaussian_kernels = self.config.gaussian_kernels
            gbf_feature = torch.zeros(n_graph, n_node, gaussian_kernels, device=x.device)
            
            # Aggregate to source and destination nodes
            node_rbf_src = scatter_add(edge_rbf, edge_src, dim=0, dim_size=len(x))  # [total_nodes, gaussian_kernels]
            node_rbf_dst = scatter_add(edge_rbf, edge_dst, dim=0, dim_size=len(x))  # [total_nodes, gaussian_kernels]
            
            # Combine source and destination features
            node_rbf_combined = node_rbf_src + node_rbf_dst  # [total_nodes, gaussian_kernels]
            
            # Reshape to batch format using to_dense_batch
            gbf_feature, _ = to_dense_batch(node_rbf_combined, batch, fill_value=0)  # [n_graph, n_node, gaussian_kernels]
            
            # Handle sparse case: [n_graph, n_node, gaussian_kernels] (already aggregated)
            # Apply padding mask
            padding_mask = x_dense.eq(0)
            aggregated_edge_features = gbf_feature.masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            )
        
            aggregated_edge_features = gbf_feature.masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            )
        
        # =====================================================================
        # COMMON PATH: Combine features and project (both sparse and dense)
        # =====================================================================
        # Combine invariant node features with aggregated edge features
        combined_features = torch.cat([
            invariant_features,  # [n_graph, n_node, 5]
            aggregated_edge_features  # [n_graph, n_node, gaussian_kernels]
        ], dim=-1)  # [n_graph, n_node, 5 + gaussian_kernels]
        
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
        """Head for learning atom ordering relative to seed atom"""
        return nn.Sequential(
            nn.Linear(self.config.graph_dim * 2, self.config.graph_dim),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim, self.config.distance_bins)  # Output proximity score
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
        
        if num_atoms <= 50:
            # Small molecules: use all pairs
            i_indices, j_indices = torch.triu_indices(
                num_atoms, num_atoms, offset=1, device=node_embeddings.device
            )
        else:
            # Large molecules: use simple random sampling to avoid memory issues
            pairs = set()
            
            attempts = 0
            while len(pairs) < max_pairs and attempts < max_pairs * 3:
                i, j = torch.randint(0, num_atoms, (2,), device=pos.device).tolist()
                if i != j:
                    pair = (min(i, j), max(i, j))
                    pairs.add(pair)
                attempts += 1
            
            pairs = list(pairs)
            
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
        
        # Convert distances to bins
        # DÜZELTME: Sınır değer hatasını (t == n_classes) önlemek için distance_bins'ten çok küçük bir sayı çıkarıyoruz.
        distance_bins = (true_distances / self.config.max_distance * (self.config.distance_bins - 1e-6)).long()
        
        # Clamp fonksiyonu hala bir güvenlik katmanı olarak kalmalıdır.
        distance_bins = torch.clamp(distance_bins, 0, self.config.distance_bins - 1)
        
        # Use cross-entropy loss
        loss = F.cross_entropy(distance_logits, distance_bins, reduction='mean')
        
        return loss

    

    def short_range_distance_loss(self, node_embeddings, edge_index, distances, mask):
        """Compute short-range distance loss (local chemical bonds)"""
        source_emb = node_embeddings[edge_index[0]]
        target_emb = node_embeddings[edge_index[1]]
        edge_emb = torch.cat([source_emb, target_emb], dim=-1)
        
        logits = self.distance_head(edge_emb)
        
        # Convert distances to bins
        # DÜZELTME: Sınır değer hatasını (t == n_classes) önlemek için distance_bins'ten çok küçük bir sayı çıkarıyoruz.
        distance_bins = (distances / self.config.max_distance * (self.config.distance_bins - 1e-6)).long()
        
        # Clamp fonksiyonu hala bir güvenlik katmanı olarak kalmalıdır.
        distance_bins = torch.clamp(distance_bins, 0, self.config.distance_bins - 1)
        
        # NaN değerlerine karşı güvenlik kontrolü
        valid_mask = torch.isfinite(distances) & mask
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)

        loss = F.cross_entropy(logits[valid_mask], distance_bins[valid_mask], reduction='mean')
        return loss
    

    def mlm_loss(self, node_embeddings, data):
        """
        Compute masked language modeling loss
        """
        assert hasattr(data, 'mlm_mask') and hasattr(data, 'original_types') and hasattr(data, 'masked_types'), "MLM data attributes are required for MLM loss"

        mask = data.mlm_mask
        original_types = data.original_types
        masked_types = getattr(data, 'masked_types', original_types)


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
            # Create MLP dynamically to handle variable edge dimensions
            self.node_edge_mlp = None  # Will be created dynamically
        else:
            if config.mlp_type in ["standard", "gated_mlp"]:
                # In node attention mode, encoder outputs node embeddings of size graph_dim
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

        # For node-attention mode, ensure ESA outputs align to node space for node-level tasks
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

        # Handle input features
        assert hasattr(batch, 'z') and batch.z is not None, \
            "Batch must have 'z' attribute with atomic numbers"
        x = batch.z

        # Ensure coordinates are available for pretraining tasks
        assert pos is not None, \
            "Coordinates (pos) are required for pretraining tasks"

        # Encode nodes (universal approach)
        node_embeddings = self.encoder(x, pos, batch)
        if node_embeddings is None:
            raise ValueError("Encoder returned None embeddings")

        # ============================================================
        # EDGE ATTENTION PATH
        # ============================================================
        if self.config.apply_attention_on == "edge":

            source = node_embeddings[edge_index[0]]
            target = node_embeddings[edge_index[1]]
            h = torch.cat((source, target), dim=1)

            edge_attr = getattr(batch, 'edge_attr', None)
            if edge_attr is not None:
                h = torch.cat((h, edge_attr.float()), dim=1)

            """
            if self.node_edge_mlp is None:
                self.node_edge_mlp = SmallMLP(
                    in_dim=h.shape[1],
                    inter_dim=128,
                    out_dim=self.config.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=max(2, self.config.num_mlp_layers),
                ).to(h.device)

            device = node_embeddings.device
            h = self.node_edge_mlp(h.to(device)) """

            # ============================================================
            # HYBRID MLP SELECTION (RNA vs PROTEIN)
            # ============================================================
            # 1. RNA/Büyük Molekül Tespiti (Encoder'daki mantığın aynısı)
            is_rna = False
            if hasattr(batch, 'dataset_type'):
                dataset_type = batch.dataset_type
                if isinstance(dataset_type, (list, tuple)):
                    is_rna = any(str(dt).upper() == 'RNA' for dt in dataset_type)
                elif isinstance(dataset_type, str):
                    is_rna = dataset_type.upper() == 'RNA'
            
            # Atom sayısına göre kontrol
            num_atoms = x.size(0)
            use_sparse_computation = is_rna or (num_atoms > 500)

            device = node_embeddings.device
            h = h.to(device)

            # 2. İki Farklı Yol: RNA için Ayrı, Protein için Ayrı MLP
            if use_sparse_computation:
                # --- YOL A: RNA & BÜYÜK MOLEKÜLLER (YENİ) ---
                # RNA için özel "rna_node_edge_mlp" oluşturuyoruz.
                # Bu, eski checkpointleri bozmaz.
                if not hasattr(self, 'rna_node_edge_mlp') or self.rna_node_edge_mlp is None:
                    self.rna_node_edge_mlp = SmallMLP(
                        in_dim=h.shape[1],
                        inter_dim=128,
                        out_dim=self.config.hidden_dims[0],
                        use_ln=False,
                        dropout_p=0,
                        num_layers=max(2, self.config.num_mlp_layers), # RNA için en az 2 katman
                    ).to(device)
                
                h = self.rna_node_edge_mlp(h)
                
            else:
                # --- YOL B: PROTEIN & KÜÇÜK MOLEKÜLLER (ESKİ/LEGACY) ---
                # Orijinal "node_edge_mlp" ismini koruyoruz.
                # Böylece eski eğitilmiş modeller hata vermez.
                if self.node_edge_mlp is None:
                    # Eski mantık: Katman sayısı config'e bağlı
                    target_num_layers = self.config.num_mlp_layers if self.config.num_mlp_layers > 1 else self.config.num_mlp_layers + 1
                    
                    self.node_edge_mlp = SmallMLP(
                        in_dim=h.shape[1],
                        inter_dim=128,
                        out_dim=self.config.hidden_dims[0],
                        use_ln=False,
                        dropout_p=0,
                        num_layers=target_num_layers,
                    ).to(device)

                h = self.node_edge_mlp(h)
            # ============================================================

            edge_index = edge_index.to(device)
            batch_mapping = batch_mapping.to(device)

            edge_batch_index = batch_mapping.index_select(0, edge_index[0])

            if self.config.set_max_items and self.config.set_max_items > 0:
                num_max_items = nearest_multiple_of_8(self.config.set_max_items + 1)
            else:
                counts = torch.bincount(edge_batch_index)
                num_max_items = int(counts.max().item())
                num_max_items = nearest_multiple_of_8(num_max_items + 1)

            h, _ = to_dense_batch(
                h,
                edge_batch_index,
                fill_value=0,
                max_num_nodes=num_max_items
            )

            # --- FIX: convert global edge_index to local indices ---
            unique_nodes = torch.unique(edge_index)
            node_id_map = {int(n.item()): i for i, n in enumerate(unique_nodes)}

            local_edge_index = edge_index.clone()
            local_edge_index[0] = torch.tensor(
                [node_id_map[int(i)] for i in edge_index[0].tolist()],
                device=edge_index.device
            )
            local_edge_index[1] = torch.tensor(
                [node_id_map[int(i)] for i in edge_index[1].tolist()],
                device=edge_index.device
            )

            h = self.esa_backbone(
                h,
                local_edge_index,
                batch_mapping,
                num_max_items=num_max_items
            )

        # ============================================================
        # NODE ATTENTION PATH (DOĞRU & AKADEMİK)
        # ============================================================
        else:
            device = node_embeddings.device
            h = self.mlp_norm(self.node_mlp(node_embeddings))
            batch_mapping = batch_mapping.to(device)

            if self.config.set_max_items and self.config.set_max_items > 0:
                num_max_items = nearest_multiple_of_8(self.config.set_max_items + 1)
            else:
                counts = torch.bincount(batch_mapping)
                num_max_items = nearest_multiple_of_8(int(counts.max().item()) + 1)

            # dense node batch
            h, dense_batch_index = to_dense_batch(
                h,
                batch_mapping,
                fill_value=0,
                max_num_nodes=num_max_items
            )

            # ❗ EDGE YOK
            h = self.esa_backbone(
                h,
                edge_index,
                batch_mapping=batch_mapping,
                num_max_items=num_max_items
            )

            if self.config.is_node_task and h.dim() == 3:
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
                
                mask = (torch.rand(edge_index.size(1)) < 0.15).to(distances.device)
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
    
    def _compute_per_type_losses(self, batch, graph_embeddings, node_embeddings):
        """
        Compute losses separately for each dataset type.
        Uses SAME embeddings from forward pass (no extra computation).
        
        Returns:
            Dict[str, Dict[str, float]]: {dtype: {task: loss_value}}
        """
        if not hasattr(batch, 'dataset_type'):
            return {}
        
        type_losses = {}
        unique_types = set(batch.dataset_type) if isinstance(batch.dataset_type, (list, tuple)) else {batch.dataset_type}
        
        # Get device from embeddings (more reliable than batch.x)
        device = node_embeddings.device
        
        for dtype in unique_types:
            # Create masks for this dataset type
            if isinstance(batch.dataset_type, (list, tuple)):
                graph_mask = torch.tensor([dt == dtype for dt in batch.dataset_type], 
                                         device=device)
            else:
                graph_mask = torch.ones(batch.num_graphs, dtype=torch.bool, device=device)
            
            graph_indices = torch.where(graph_mask)[0]
            node_mask = torch.isin(batch.batch, graph_indices)
            
            # Skip if no samples of this type
            if node_mask.sum() == 0:
                continue
            
            dtype_losses = {}
            dtype_total = 0.0
            
            # Short-range distance loss (masked edges)
            if "short_range_distance" in self.config.pretraining_tasks:
                if hasattr(batch, 'edge_index') and batch.edge_index is not None and hasattr(batch, 'pos') and batch.pos is not None:
                    edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
                    
                    if edge_mask.sum() > 0:
                        masked_edge_index = batch.edge_index[:, edge_mask]
                        masked_pos = batch.pos[node_mask]
                        
                        # Remap node indices for masked graph
                        node_mapping = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
                        node_mapping[node_mask] = torch.arange(node_mask.sum(), device=device)
                        remapped_edge_index = node_mapping[masked_edge_index]
                        
                        # Compute distances
                        distances = torch.norm(
                            masked_pos[remapped_edge_index[1]] - masked_pos[remapped_edge_index[0]], 
                            dim=1
                        )
                        
                        # Random mask (15%)
                        random_mask = (torch.rand(remapped_edge_index.size(1)) < 0.15).to(distances.device)
                        
                        # Compute loss
                        masked_node_emb = node_embeddings[node_mask]
                        dist_loss = self.pretraining_tasks.short_range_distance_loss(
                            masked_node_emb, remapped_edge_index, distances, random_mask
                        )
                        dtype_losses['short_range_distance'] = dist_loss
                        dtype_total += self.config.task_weights['short_range_distance'] * dist_loss
            
            # Long-range distance loss (masked nodes)
            if "long_range_distance" in self.config.pretraining_tasks:
                if hasattr(batch, 'pos') and batch.pos is not None:
                    # Create pseudo-batch for this type
                    masked_batch = batch.__class__()
                    if hasattr(batch, 'x') and batch.x is not None:
                        masked_batch.x = batch.x[node_mask]
                    masked_batch.pos = batch.pos[node_mask]
                    masked_batch.batch = torch.zeros(node_mask.sum(), dtype=torch.long, device=device)
                    masked_batch.num_graphs = 1
                    
                    masked_node_emb = node_embeddings[node_mask]
                    long_range_loss = self.pretraining_tasks.long_range_distance_loss(
                        masked_node_emb, masked_batch
                    )
                    dtype_losses['long_range_distance'] = long_range_loss
                    dtype_total += self.config.task_weights['long_range_distance'] * long_range_loss
            
            # MLM loss (masked nodes)
            if "mlm" in self.config.pretraining_tasks:
                if hasattr(batch, 'mlm_mask') and batch.mlm_mask is not None:
                    masked_mlm = batch.mlm_mask[node_mask]
                    
                    if masked_mlm.sum() > 0:
                        # Create pseudo-batch for this type with all required MLM attributes
                        masked_batch = batch.__class__()
                        # Add x or z if available (for compatibility, though MLM doesn't strictly need it)
                        if hasattr(batch, 'x') and batch.x is not None:
                            masked_batch.x = batch.x[node_mask]
                        if hasattr(batch, 'z') and batch.z is not None:
                            masked_batch.z = batch.z[node_mask]
                        masked_batch.mlm_mask = masked_mlm
                        # MLM loss requires original_types and masked_types (not mlm_labels)
                        if hasattr(batch, 'original_types') and batch.original_types is not None:
                            masked_batch.original_types = batch.original_types[node_mask]
                        if hasattr(batch, 'masked_types') and batch.masked_types is not None:
                            masked_batch.masked_types = batch.masked_types[node_mask]
                        
                        masked_node_emb = node_embeddings[node_mask]
                        try:
                            mlm_loss = self.pretraining_tasks.mlm_loss(masked_node_emb, masked_batch)
                            dtype_losses['mlm'] = mlm_loss
                            dtype_total += self.config.task_weights['mlm'] * mlm_loss
                        except (AssertionError, AttributeError):
                            pass  # Skip if MLM attributes are missing for this type
            
            dtype_losses['total'] = dtype_total
            type_losses[dtype] = dtype_losses
        
        return type_losses
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # DEBUG: Check if dataset_type exists
        if batch_idx % 10 == 0:
            if hasattr(batch, 'dataset_type'):
                print(f"✅ Batch {batch_idx} has dataset_type: {batch.dataset_type}")
            else:
                print(f"⚠️  Batch {batch_idx} MISSING dataset_type attribute!")
        
        # Log batch composition every 10 batches
        if batch_idx % 10 == 0:
            num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            num_atoms = batch.num_nodes if hasattr(batch, 'num_nodes') else batch.x.size(0)
            
            # Calculate atoms per graph
            if hasattr(batch, 'batch'):
                atoms_per_graph = [(batch.batch == i).sum().item() for i in range(num_graphs)]
                avg_atoms = sum(atoms_per_graph) / len(atoms_per_graph)
                min_atoms = min(atoms_per_graph)
                max_atoms = max(atoms_per_graph)
            else:
                avg_atoms = num_atoms / num_graphs
                min_atoms = max_atoms = int(avg_atoms)
            
            print(f"\n{'='*80}")
            print(f"📊 BATCH {batch_idx} | Epoch {self.current_epoch}")
            print(f"{'='*80}")
            print(f"  Total Graphs: {num_graphs:4d} | Total Atoms: {num_atoms:6,d}")
            print(f"  Avg Atoms:    {avg_atoms:6.1f} | Min: {min_atoms:6,d} | Max: {max_atoms:6,d}")
            
            # Show dataset composition if available
            if hasattr(batch, 'dataset_type'):
                from collections import Counter
                if isinstance(batch.dataset_type, (list, tuple)):
                    type_counts = Counter(batch.dataset_type)
                elif hasattr(batch.dataset_type, 'tolist'):
                    type_counts = Counter(batch.dataset_type.tolist() if hasattr(batch.dataset_type, 'tolist') else [batch.dataset_type.item()])
                else:
                    type_counts = {batch.dataset_type: num_graphs}
                
                print(f"  Dataset Mix:")
                for dtype, count in sorted(type_counts.items()):
                    pct = count / num_graphs * 100
                    # Calculate atoms for this dataset type
                    if hasattr(batch, 'batch') and isinstance(batch.dataset_type, (list, tuple)):
                        type_atoms = sum(atoms_per_graph[i] for i, dt in enumerate(batch.dataset_type) if dt == dtype)
                        print(f"    {dtype.upper():12s}: {count:4d} graphs ({pct:5.1f}%) | {type_atoms:6,d} atoms")
                    else:
                        print(f"    {dtype.upper():12s}: {count:4d} graphs ({pct:5.1f}%)")
            
            print(f"{'='*80}")
        
        graph_embeddings, node_embeddings = self.forward(batch)
        
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)

        for task_name, loss_value in losses.items():
            self.log(f"train_{task_name}_loss", loss_value, prog_bar=True, on_step=True, on_epoch=True, logger=True)
            if batch_idx % 50 == 0:
                print(f"  {task_name}: {loss_value:.4f}")

        if batch_idx % 50 == 0:
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
            print(f"Step {batch_idx} - {loss_str} | Total: {total_loss:.4f}")
        
        self.log("train_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        
        # Per-type loss tracking (optional, minimal overhead)
        if self.config.compute_per_type_losses and batch_idx % self.config.log_per_type_frequency == 0:
            type_losses = self._compute_per_type_losses(batch, graph_embeddings, node_embeddings)
            
            # Log per-type losses to WandB
            for dtype, dtype_losses in type_losses.items():
                for task_name, loss_value in dtype_losses.items():
                    self.log(f"train_{dtype}_{task_name}_loss", loss_value, on_step=True, logger=True)
            
            # Print per-type summary (optional)
            if batch_idx % 50 == 0 and type_losses:
                print(f"\n  📊 Per-Type Losses (Batch {batch_idx}):")
                for dtype, dtype_losses in type_losses.items():
                    loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in dtype_losses.items() if k != 'total'])
                    print(f"    {dtype.upper():12s}: {loss_str} | Total: {dtype_losses['total']:.4f}")
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        graph_embeddings, node_embeddings = self.forward(batch)
        
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)
        
        # Log individual losses
        for task_name, loss_value in losses.items():
            self.log(f"val_{task_name}_loss", loss_value, on_epoch=True, logger=True)
        
        self.log("val_total_loss", total_loss, on_epoch=True, logger=True)
        
        # Per-type loss tracking for validation
        if self.config.compute_per_type_losses:
            type_losses = self._compute_per_type_losses(batch, graph_embeddings, node_embeddings)
            for dtype, dtype_losses in type_losses.items():
                for task_name, loss_value in dtype_losses.items():
                    self.log(f"val_{dtype}_{task_name}_loss", loss_value, on_epoch=True, logger=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        graph_embeddings, node_embeddings = self.forward(batch)
        
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)
        
        # Log individual losses
        for task_name, loss_value in losses.items():
            self.log(f"test_{task_name}_loss", loss_value, on_epoch=True, logger=True)
        
        self.log("test_total_loss", total_loss, on_epoch=True, logger=True)
        
        # Per-type loss tracking for test
        if self.config.compute_per_type_losses:
            type_losses = self._compute_per_type_losses(batch, graph_embeddings, node_embeddings)
            for dtype, dtype_losses in type_losses.items():
                for task_name, loss_value in dtype_losses.items():
                    self.log(f"test_{dtype}_{task_name}_loss", loss_value, on_epoch=True, logger=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Optimizer setup for all parameters
        optimizer = bnb.optim.AdamW8bit(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.optimiser_weight_decay,
        )

        # Cosine learning rate schedule with warmup
        total_epochs = getattr(self.config, 'max_epochs', 100)
        warmup_frac = 0.1
        warmup_epochs = max(1, int(total_epochs * warmup_frac))

        def lr_lambda(current_epoch: int):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs)
            # Cosine decay from 1.0 to 0.1 over remaining epochs
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

    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return config


        # Example usage
if __name__ == "__main__":
    config = create_pretraining_config()
    
    # Create model
    model = PretrainingESAModel(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Pretraining tasks: {config.pretraining_tasks}")
    print(f"Task weights: {config.task_weights}")