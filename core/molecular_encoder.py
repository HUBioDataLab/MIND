import math
import torch

from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add, scatter_min

from data_loading.gaussian import GaussianLayer
from esa.utils.posenc_encoders.laplace_pos_encoder import LapPENodeEncoder
from esa.utils.posenc_encoders.kernel_pos_encoder import KernelPENodeEncoder

from core.pretraining_config import PretrainingConfig


PROTEIN_LOCAL_SLOT_COUNT = 14
RNA_LOCAL_SLOT_COUNT = 23
PROTEIN_LOCAL_SLOT_OFFSET = 1
RNA_LOCAL_SLOT_OFFSET = PROTEIN_LOCAL_SLOT_OFFSET + PROTEIN_LOCAL_SLOT_COUNT
NUM_DOMAIN_LOCAL_SLOT_IDS = (
    1 + PROTEIN_LOCAL_SLOT_COUNT + RNA_LOCAL_SLOT_COUNT
)


def build_domain_local_slot_ids(batch, num_nodes: int, device) -> torch.Tensor:
    """Return domain-local atom ordinal ids without consulting atom identity.

    ID 0 means "no reliable fixed slot". Protein blocks with exactly 14 nodes
    receive ids 1..14; RNA blocks with exactly 23 nodes receive ids 15..37.
    The calculation uses only graph domain, block membership, and node order,
    so masking atomic numbers cannot reveal the target label through this id.
    """
    slot_ids = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if (
        batch is None
        or not hasattr(batch, "batch")
        or batch.batch is None
        or not hasattr(batch, "block_idx")
        or batch.block_idx is None
        or batch.batch.numel() != num_nodes
        or batch.block_idx.numel() != num_nodes
        or num_nodes == 0
    ):
        return slot_ids

    graph_ids = batch.batch.to(device=device, dtype=torch.long)
    block_ids = batch.block_idx.to(device=device)
    raw_types = getattr(batch, "dataset_type", None)
    if raw_types is None:
        return slot_ids
    if isinstance(raw_types, (list, tuple)):
        graph_types = [str(value).lower() for value in raw_types]
    else:
        num_graphs = int(getattr(batch, "num_graphs", 1) or 1)
        graph_types = [str(raw_types).lower()] * num_graphs

    num_graphs = int(graph_ids.max().item()) + 1
    expected_by_graph = torch.zeros(num_graphs, dtype=torch.long, device=device)
    offset_by_graph = torch.zeros(num_graphs, dtype=torch.long, device=device)
    for graph_index in range(min(num_graphs, len(graph_types))):
        graph_type = graph_types[graph_index]
        if graph_type in {"pdb", "protein"}:
            expected_by_graph[graph_index] = PROTEIN_LOCAL_SLOT_COUNT
            offset_by_graph[graph_index] = PROTEIN_LOCAL_SLOT_OFFSET
        elif graph_type == "rna":
            expected_by_graph[graph_index] = RNA_LOCAL_SLOT_COUNT
            offset_by_graph[graph_index] = RNA_LOCAL_SLOT_OFFSET

    # PyG preserves node order and each residue/nucleotide is contiguous. Build
    # run ids once, then calculate run length and local ordinal vectorially.
    run_start = torch.ones(num_nodes, dtype=torch.bool, device=device)
    if num_nodes > 1:
        run_start[1:] = (
            (graph_ids[1:] != graph_ids[:-1])
            | (block_ids[1:] != block_ids[:-1])
        )
    run_ids = torch.cumsum(run_start.to(torch.long), dim=0) - 1
    run_starts = torch.where(run_start)[0]
    run_lengths = torch.bincount(run_ids)
    local_ordinals = torch.arange(num_nodes, device=device) - run_starts[run_ids]

    expected = expected_by_graph[graph_ids]
    valid = (expected > 0) & (run_lengths[run_ids] == expected)
    slot_ids[valid] = offset_by_graph[graph_ids[valid]] + local_ordinals[valid]
    return slot_ids


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

        self.domain_local_slot_embedding = None
        if getattr(config, "use_domain_local_slot_embedding", False):
            self.domain_local_slot_embedding = nn.Embedding(
                NUM_DOMAIN_LOCAL_SLOT_IDS,
                config.hidden_dims[0],
                padding_idx=0,
            )

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

        if self.domain_local_slot_embedding is not None:
            slot_ids = build_domain_local_slot_ids(batch, x.size(0), x.device)
            encoded = encoded + self.domain_local_slot_embedding(slot_ids)

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

        if not hasattr(self, '_period_lut') or self._period_lut.device != device:
            period_lut = torch.zeros(128, dtype=torch.long, device=device)
            for atomic_num, period in period_map.items():
                period_lut[atomic_num] = period
            self.register_buffer('_period_lut', period_lut, persistent=False)

        atom_emb = self.atom_embedding(atomic_numbers)
        periods = self._period_lut[atomic_numbers]
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

            if getattr(self.config, 'use_vectorized_geometric_features', True):
                coordination_dense, _ = to_dense_batch(coordination_sparse, batch_idx, fill_value=0, max_num_nodes=max_nodes)
                return coordination_dense
            else:
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

        if getattr(self.config, 'use_triton_gather_scatter', False) and x.is_cuda:
            from esa.triton_graph_gather_scatter import triton_fused_gather_scatter
            feat_matrix = torch.stack([close_mask.float(), medium_mask.float()], dim=-1)
            fused_sparse = triton_fused_gather_scatter(feat_matrix, edge_index, num_nodes=num_nodes)
            close_coordination_sparse = fused_sparse[:, 0]
            distance_coordination_sparse = fused_sparse[:, 1]
        else:
            close_coordination_sparse = scatter_add(
                close_mask.float(), edge_src, dim=0, dim_size=num_nodes
            )
            distance_coordination_sparse = scatter_add(
                medium_mask.float(), edge_src, dim=0, dim_size=num_nodes
            )

        close_coordination, _ = to_dense_batch(close_coordination_sparse, batch, fill_value=0)
        distance_based_coordination, _ = to_dense_batch(distance_coordination_sparse, batch, fill_value=0)

        if getattr(self.config, 'use_vectorized_geometric_features', True):
            # Fully Vectorized Zero-Nonzero Distance Features (0 calls to aten::nonzero)
            masked_close_dists = torch.where(close_mask, edge_distances, torch.full_like(edge_distances, float('inf')))
            min_distances_sparse, _ = scatter_min(masked_close_dists, edge_src, dim=0, dim_size=num_nodes)
            min_distances_sparse[torch.isinf(min_distances_sparse)] = 0.0

            sum_per_node = scatter_add(torch.where(close_mask, edge_distances, 0.0), edge_src, dim=0, dim_size=num_nodes)
            count_per_node = scatter_add(close_mask.float(), edge_src, dim=0, dim_size=num_nodes)
            mean_distances_sparse = sum_per_node / (count_per_node + 1e-8)
        else:
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
