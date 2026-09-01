import torch

from torch import nn
from torch.nn import functional as F

from core.pretraining_config import PretrainingConfig
from core.molecular_encoder import (
    PROTEIN_LOCAL_SLOT_COUNT,
    PROTEIN_LOCAL_SLOT_OFFSET,
    RNA_LOCAL_SLOT_COUNT,
    RNA_LOCAL_SLOT_OFFSET,
)
from data_loading.atom_position_vocab import VIRTUAL_ATOMIC_NUMBER


class PretrainingTasks(nn.Module):
    """Module containing all pretraining task heads and their loss functions."""

    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config

        # Long-range bin boundaries (0 – max_distance, typically 0-20Å)
        if config.distance_bin_boundaries is not None:
            boundaries = config.distance_bin_boundaries
            assert len(boundaries) == config.distance_bins - 1, (
                f"distance_bin_boundaries must have distance_bins-1={config.distance_bins - 1} "
                f"entries, got {len(boundaries)}"
            )
            self.register_buffer('bin_boundaries', torch.tensor(boundaries, dtype=torch.float32))
        else:
            self.bin_boundaries = None

        # Short-range bin boundaries (0 – short_range_max_distance, typically 0-5Å)
        if config.short_range_bin_boundaries is not None:
            sr_boundaries = config.short_range_bin_boundaries
            assert len(sr_boundaries) == config.short_range_distance_bins - 1, (
                f"short_range_bin_boundaries must have short_range_distance_bins-1="
                f"{config.short_range_distance_bins - 1} entries, got {len(sr_boundaries)}"
            )
            self.register_buffer('sr_bin_boundaries', torch.tensor(sr_boundaries, dtype=torch.float32))
        else:
            self.sr_bin_boundaries = None

        self.long_range_distance_head = self._create_long_range_distance_head()
        self.distance_head = self._create_distance_head()  # short-range, output = short_range_distance_bins
        self.mlm_head = self._create_mlm_head()
        self.coordinate_denoising_head = self._create_coordinate_denoising_head()

    def _create_coordinate_denoising_head(self):
        """Predict 3D noise vector from node embeddings (invariant head; loss is SE(3)-invariant).
        Literature: equivariant denoising (e.g. E(3)-equivariant noise prediction) would use
        vector (1o) features from EquivariantSAB; currently we use scalar→3D MLP (invariant head)."""
        return nn.Sequential(
            nn.Linear(self.config.graph_dim, self.config.graph_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim // 2, 3)
        )

    def _create_long_range_distance_head(self):
        """Head for learning atom ordering relative to seed atom"""
        return nn.Sequential(
            nn.Linear(self.config.graph_dim * 2, self.config.graph_dim),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim, self.config.distance_bins)
        )

    def _create_distance_head(self):
        """Short-range distance head — output size matches short_range_distance_bins."""
        return nn.Sequential(
            nn.Linear(self.config.graph_dim * 2, self.config.graph_dim),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim, self.config.short_range_distance_bins)
        )

    def _distances_to_bins(self, distances: torch.Tensor) -> torch.Tensor:
        """Long-range binning: non-uniform boundaries if configured, else uniform over [0, max_distance]."""
        if self.bin_boundaries is not None:
            return torch.bucketize(distances, self.bin_boundaries.to(distances.device))
        bins = (distances / self.config.max_distance * (self.config.distance_bins - 1e-6)).long()
        return torch.clamp(bins, 0, self.config.distance_bins - 1)

    def _short_range_distances_to_bins(self, distances: torch.Tensor) -> torch.Tensor:
        """Short-range binning: non-uniform boundaries if configured, else uniform over [0, short_range_max_distance]."""
        if self.sr_bin_boundaries is not None:
            return torch.bucketize(distances, self.sr_bin_boundaries.to(distances.device))
        bins = (distances / self.config.short_range_max_distance * (self.config.short_range_distance_bins - 1e-6)).long()
        return torch.clamp(bins, 0, self.config.short_range_distance_bins - 1)

    def _create_mlm_head(self):
        """Head for masked language modeling"""
        max_types = self.config.atom_types  # Universal atomic types only
        return nn.Sequential(
            nn.Linear(self.config.graph_dim, self.config.graph_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim // 2, max_types)
        )

    @staticmethod
    def _sample_graph_local_pairs(
        batch_mapping: torch.Tensor,
        eligible_indices: torch.Tensor,
        max_pairs: int = 1000,
    ):
        """Return atom pairs sampled only within their original graph.

        All pairs are enumerated when the batch contains at most ``max_pairs``
        eligible intra-graph pairs. Larger batches are sampled uniformly with
        replacement over the union of their intra-graph pairs.
        """
        device = eligible_indices.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        if eligible_indices.numel() < 2 or max_pairs <= 0:
            return empty, empty

        eligible_graphs = batch_mapping[eligible_indices]
        graph_ids, graph_counts = torch.unique(
            eligible_graphs, sorted=True, return_counts=True
        )
        pair_counts = graph_counts * (graph_counts - 1) // 2
        usable = pair_counts > 0
        graph_ids = graph_ids[usable]
        graph_counts = graph_counts[usable]
        pair_counts = pair_counts[usable]
        if graph_ids.numel() == 0:
            return empty, empty

        total_pairs = int(pair_counts.sum().item())
        target_pairs = min(max_pairs, total_pairs)
        if total_pairs <= max_pairs:
            left_parts = []
            right_parts = []
            for graph_id, graph_count in zip(
                graph_ids.tolist(), graph_counts.tolist()
            ):
                graph_nodes = eligible_indices[eligible_graphs == graph_id]
                local_left, local_right = torch.triu_indices(
                    graph_count,
                    graph_count,
                    offset=1,
                    device=device,
                )
                left_parts.append(graph_nodes[local_left])
                right_parts.append(graph_nodes[local_right])
            return torch.cat(left_parts), torch.cat(right_parts)

        sampled_graph_positions = torch.multinomial(
            pair_counts.float(), target_pairs, replacement=True
        )
        sampled_positions, samples_per_graph = torch.unique(
            sampled_graph_positions, sorted=True, return_counts=True
        )
        left_parts = []
        right_parts = []
        for graph_position, sample_count in zip(
            sampled_positions.tolist(), samples_per_graph.tolist()
        ):
            graph_id = graph_ids[graph_position]
            graph_nodes = eligible_indices[eligible_graphs == graph_id]
            graph_count = int(graph_nodes.numel())
            local_left = torch.randint(
                graph_count, (sample_count,), device=device
            )
            local_right = torch.randint(
                graph_count - 1, (sample_count,), device=device
            )
            local_right += (local_right >= local_left).long()
            lower = torch.minimum(local_left, local_right)
            upper = torch.maximum(local_left, local_right)
            left_parts.append(graph_nodes[lower])
            right_parts.append(graph_nodes[upper])
        return torch.cat(left_parts), torch.cat(right_parts)

    def long_range_distance_loss(self, node_embeddings, data):
        """
        Long-range distance prediction loss for global 3D structure learning.

        Predicts distances between all atom pairs (or sampled pairs for large molecular systems)
        to learn overall molecular geometry. This is completely SE(3) invariant.
        """
        if not hasattr(data, 'pos') or data.pos is None:
            return node_embeddings.sum() * 0.0

        pos = data.pos
        num_atoms = pos.size(0)

        geometry_valid_mask = getattr(data, 'geometry_valid_mask', None)
        template_node_mask = getattr(data, 'template_node_mask', None)
        exclusion_mask = None
        if geometry_valid_mask is not None:
            if geometry_valid_mask.numel() != num_atoms:
                raise ValueError(
                    "geometry_valid_mask must contain one value per atom"
                )
            exclusion_mask = ~geometry_valid_mask.to(
                device=pos.device, dtype=torch.bool
            )
        elif template_node_mask is not None:
            if template_node_mask.numel() != num_atoms:
                raise ValueError(
                    "template_node_mask must contain one value per atom"
                )
            exclusion_mask = template_node_mask.to(
                device=pos.device, dtype=torch.bool
            )

        if exclusion_mask is None:
            eligible_indices = torch.arange(num_atoms, device=pos.device)
        else:
            eligible_indices = torch.where(~exclusion_mask)[0]

        batch_mapping = getattr(data, 'batch', None)
        if batch_mapping is None:
            batch_mapping = torch.zeros(
                num_atoms, dtype=torch.long, device=pos.device
            )
        else:
            if batch_mapping.numel() != num_atoms:
                raise ValueError("batch must contain one graph id per atom")
            batch_mapping = batch_mapping.to(device=pos.device, dtype=torch.long)

        i_indices, j_indices = self._sample_graph_local_pairs(
            batch_mapping, eligible_indices, max_pairs=1000
        )
        if i_indices.numel() == 0:
            return node_embeddings.sum() * 0.0

        emb_i = node_embeddings[i_indices]
        emb_j = node_embeddings[j_indices]
        pair_embeddings = torch.cat([emb_i, emb_j], dim=-1)

        distance_logits = self.long_range_distance_head(pair_embeddings)

        pos_i = pos[i_indices]
        pos_j = pos[j_indices]
        true_distances = torch.norm(pos_i - pos_j, dim=1)

        distance_bins = self._distances_to_bins(true_distances)

        loss = F.cross_entropy(distance_logits, distance_bins, reduction='mean')
        return loss

    def short_range_distance_loss(self, node_embeddings, edge_index, distances, mask):
        """Short-range loss over finite, selected edges within its distance range."""
        valid_mask = (
            torch.isfinite(distances)
            & mask.to(device=distances.device, dtype=torch.bool)
            & (distances <= float(self.config.short_range_max_distance))
        )
        if not bool(valid_mask.any()):
            return node_embeddings.sum() * 0.0

        valid_edges = edge_index[:, valid_mask]
        valid_distances = distances[valid_mask]
        source_emb = node_embeddings[valid_edges[0]]
        target_emb = node_embeddings[valid_edges[1]]
        edge_emb = torch.cat([source_emb, target_emb], dim=-1)

        logits = self.distance_head(edge_emb)
        distance_bins = self._short_range_distances_to_bins(valid_distances)
        loss = F.cross_entropy(logits, distance_bins, reduction='mean')
        return loss

    def mlm_loss(self, node_embeddings, data):
        """Compute masked language modeling loss"""
        assert hasattr(data, 'mlm_mask') and hasattr(data, 'original_types') and hasattr(data, 'masked_types'), \
            "MLM data attributes are required for MLM loss"

        mask = data.mlm_mask
        original_types = data.original_types

        if mask is None or mask.sum() == 0:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)

        logits = self.mlm_head(node_embeddings)
        masked_logits = logits[mask]
        masked_targets = original_types[mask]
        virtual_weight = float(
            getattr(self.config, "virtual_atom_loss_weight", 1.0)
        )
        if virtual_weight == 1.0:
            return F.cross_entropy(
                masked_logits, masked_targets, reduction='mean'
            )

        virtual_mask = masked_targets == VIRTUAL_ATOMIC_NUMBER
        real_mask = ~virtual_mask
        per_node_loss = F.cross_entropy(
            masked_logits, masked_targets, reduction='none'
        )
        real_loss = (
            (per_node_loss * real_mask).sum()
            / real_mask.sum().clamp_min(1)
        )
        virtual_loss = (
            (per_node_loss * virtual_mask).sum()
            / virtual_mask.sum().clamp_min(1)
        )
        return real_loss + virtual_weight * virtual_loss

    def coordinate_denoising_loss(self, node_embeddings, data):
        """
        Coordinate denoising: predict noise (pos - clean_pos) from node embeddings.
        Uses invariant head (scalar -> 3D); loss is MSE on masked nodes, SE(3)-invariant.
        """
        if not hasattr(data, 'clean_pos') or data.clean_pos is None:
            return torch.tensor(0.0, device=node_embeddings.device)
        if not hasattr(data, 'pos') or data.pos is None:
            return torch.tensor(0.0, device=node_embeddings.device)
        coord_mask = getattr(data, 'coord_mask', None)
        if coord_mask is None:
            coord_mask = torch.ones(node_embeddings.size(0), dtype=torch.bool, device=node_embeddings.device)
        if coord_mask.sum() == 0:
            return torch.tensor(0.0, device=node_embeddings.device)
        true_noise = data.pos - data.clean_pos
        pred_noise = self.coordinate_denoising_head(node_embeddings)
        return F.mse_loss(pred_noise[coord_mask], true_noise[coord_mask], reduction='mean')
