import math
import torch

from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add, scatter_min

from data_loading.gaussian import GaussianLayer
from esa.utils.posenc_encoders.laplace_pos_encoder import LapPENodeEncoder
from esa.utils.posenc_encoders.kernel_pos_encoder import KernelPENodeEncoder

from core.pretraining_config import PretrainingConfig


def nearest_multiple_of_8(n: int) -> int:
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
        assert x is not None, "Input features cannot be None"
        assert x.dtype == torch.long, f"Expected atomic numbers (torch.long), got {x.dtype}"

        encoded = self._encode_molecule_features(x)

        # Add 3D geometric features if available
        if pos is not None and self.config.use_3d_coordinates:
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
        period_map = {
            1: 1,                           # H
            6: 2, 7: 2, 8: 2, 9: 2,        # C, N, O, F (period 2)
            15: 3, 16: 3, 17: 3,            # P, S, Cl (period 3)
        }

        device = atomic_numbers.device

        atom_emb = self.atom_embedding(atomic_numbers)

        periods = torch.zeros_like(atomic_numbers)
        for atomic_num, period in period_map.items():
            periods[atomic_numbers == atomic_num] = period
        period_emb = self.period_embedding(periods)

        combined_features = torch.cat([
            atom_emb,    # [batch_size, hidden_dim-8]
            period_emb,  # [batch_size, 8]
        ], dim=-1)       # [batch_size, hidden_dim]

        return self.molecule_encoder(combined_features)

    def _compute_chemical_coordination(self, x_dense, batch):
        """Compute true chemical coordination numbers from molecular graph topology"""
        batch_size = x_dense.size(0)
        max_nodes = x_dense.size(1)
        device = x_dense.device

        assert hasattr(batch, 'edge_index') and batch.edge_index is not None, "Edge index is required for chemical coordination computation"
        edge_index = batch.edge_index
        batch_idx = getattr(batch, 'batch', None)

        if batch_idx is not None:
            src, dst = edge_index[0], edge_index[1]

            unique_edge_mask = src < dst
            if unique_edge_mask.sum() == 0:
                unique_edge_mask = torch.ones_like(src, dtype=torch.bool)

            unique_src = src[unique_edge_mask]
            unique_dst = dst[unique_edge_mask]

            num_nodes = len(batch_idx)
            coordination_sparse = torch.zeros(num_nodes, device=device, dtype=torch.float)
            coordination_sparse.scatter_add_(0, unique_src, torch.ones_like(unique_src, dtype=torch.float))
            coordination_sparse.scatter_add_(0, unique_dst, torch.ones_like(unique_dst, dtype=torch.float))

            coordination_dense = torch.zeros(batch_size, max_nodes, device=device)
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
        Compute SE(3) invariant geometric features using sparse edge_index computation.

        All data types use sparse computation for memory efficiency and consistency.
        """
        x_dense, batch_mask = to_dense_batch(x, batch, fill_value=0)
        pos_dense, _ = to_dense_batch(pos, batch, fill_value=0)

        n_graph, n_node = x_dense.size()

        assert batch_obj is not None, "Batch object is required for geometric features"
        assert hasattr(batch_obj, 'edge_index'), "Edge index is required"

        edge_index = batch_obj.edge_index

        edge_src, edge_dst = edge_index[0], edge_index[1]
        edge_distances = torch.norm(pos[edge_src] - pos[edge_dst], dim=1)  # [num_edges]

        chemical_coordination = self._compute_chemical_coordination(x_dense, batch_obj)

        close_cutoff = getattr(self.config, 'close_cutoff', 3.0)
        close_mask = (edge_distances < close_cutoff) & (edge_distances > 0.5)
        medium_mask = (edge_distances < self.config.cutoff_distance) & (edge_distances > 0.5)

        num_nodes = len(x)

        close_coordination_sparse = scatter_add(
            close_mask.float(), edge_src, dim=0, dim_size=num_nodes
        )
        distance_coordination_sparse = scatter_add(
            medium_mask.float(), edge_src, dim=0, dim_size=num_nodes
        )

        close_coordination, _ = to_dense_batch(close_coordination_sparse, batch, fill_value=0)
        distance_based_coordination, _ = to_dense_batch(distance_coordination_sparse, batch, fill_value=0)

        min_distances_sparse = torch.full((num_nodes,), float('inf'), device=pos.device)
        mean_distances_sparse = torch.zeros(num_nodes, device=pos.device)
        count_per_node = torch.zeros(num_nodes, device=pos.device)

        close_indices = torch.where(close_mask)[0]
        if len(close_indices) > 0:
            close_src = edge_src[close_indices]
            close_dists = edge_distances[close_indices]

            min_distances_sparse, _ = scatter_min(close_dists, close_src, dim=0, dim_size=num_nodes)

            sum_per_node = scatter_add(close_dists, close_src, dim=0, dim_size=num_nodes)
            count_per_node = scatter_add(torch.ones_like(close_dists), close_src, dim=0, dim_size=num_nodes)
            mean_distances_sparse = sum_per_node / (count_per_node + 1e-8)

        min_distances_sparse[torch.isinf(min_distances_sparse)] = 0.0

        min_distances, _ = to_dense_batch(min_distances_sparse, batch, fill_value=0)
        mean_distances, _ = to_dense_batch(mean_distances_sparse, batch, fill_value=0)

        invariant_features = torch.stack([
            chemical_coordination,        # True chemical bonds
            close_coordination,           # Close spatial neighbors (~3Å)
            distance_based_coordination,  # Medium range environment (~cutoff)
            min_distances,                # Closest neighbor distance
            mean_distances,               # Average close neighbor distance
        ], dim=-1)  # [n_graph, n_node, 5]

        edge_types = x[edge_src] * self.config.atom_types + x[edge_dst]

        edge_dist_3d = edge_distances.unsqueeze(0).unsqueeze(0)  # [1, 1, num_edges]
        edge_type_3d = edge_types.unsqueeze(0).unsqueeze(0)      # [1, 1, num_edges]

        edge_rbf_4d = self.gaussian_layer(edge_dist_3d, edge_type_3d)  # [1, 1, num_edges, gaussian_kernels]
        edge_rbf = edge_rbf_4d.squeeze(0).squeeze(0)  # [num_edges, gaussian_kernels]

        node_rbf_src = scatter_add(edge_rbf, edge_src, dim=0, dim_size=num_nodes)
        node_rbf_dst = scatter_add(edge_rbf, edge_dst, dim=0, dim_size=num_nodes)
        node_rbf_combined = node_rbf_src + node_rbf_dst

        gbf_feature, _ = to_dense_batch(node_rbf_combined, batch, fill_value=0)

        padding_mask = x_dense.eq(0)
        aggregated_edge_features = gbf_feature.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        combined_features = torch.cat([
            invariant_features,          # [n_graph, n_node, 5]
            aggregated_edge_features     # [n_graph, n_node, gaussian_kernels]
        ], dim=-1)  # [n_graph, n_node, 5 + gaussian_kernels]

        geometric_features = self.coordinate_projection(combined_features)

        # Convert back to sparse format
        geometric_features = geometric_features[batch_mask]

        return geometric_features
