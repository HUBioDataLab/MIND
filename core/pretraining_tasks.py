import torch

from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean

from core.pretraining_config import PretrainingConfig


RESIDUE_3TO_IDX = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
}
NUM_RESIDUE_TYPES = 20


class PretrainingTasks(nn.Module):
    """Module containing all pretraining task heads and their loss functions."""

    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config

        self.long_range_distance_head = self._create_long_range_distance_head()
        self.distance_head = self._create_distance_head()  # Used by short_range_distance
        self.mlm_head = self._create_mlm_head()
        self.coordinate_denoising_head = self._create_coordinate_denoising_head()
        self.residue_head = self._create_residue_head()

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

    def _create_residue_head(self):
        """Head for residue type prediction (20 standard amino acids)"""
        return nn.Sequential(
            nn.Linear(self.config.graph_dim, self.config.graph_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim // 2, NUM_RESIDUE_TYPES)
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

        max_pairs = min(1000, num_atoms * (num_atoms - 1) // 2)

        if num_atoms <= 50:
            i_indices, j_indices = torch.triu_indices(
                num_atoms, num_atoms, offset=1, device=node_embeddings.device
            )
        else:
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

        emb_i = node_embeddings[i_indices]
        emb_j = node_embeddings[j_indices]
        pair_embeddings = torch.cat([emb_i, emb_j], dim=-1)

        distance_logits = self.long_range_distance_head(pair_embeddings)

        pos_i = pos[i_indices]
        pos_j = pos[j_indices]
        true_distances = torch.norm(pos_i - pos_j, dim=1)

        distance_bins = (true_distances / self.config.max_distance * (self.config.distance_bins - 1e-6)).long()
        distance_bins = torch.clamp(distance_bins, 0, self.config.distance_bins - 1)

        loss = F.cross_entropy(distance_logits, distance_bins, reduction='mean')
        return loss

    def short_range_distance_loss(self, node_embeddings, edge_index, distances, mask):
        """Compute short-range distance loss (local chemical bonds)"""
        source_emb = node_embeddings[edge_index[0]]
        target_emb = node_embeddings[edge_index[1]]
        edge_emb = torch.cat([source_emb, target_emb], dim=-1)

        logits = self.distance_head(edge_emb)

        distance_bins = (distances / self.config.max_distance * (self.config.distance_bins - 1e-6)).long()
        distance_bins = torch.clamp(distance_bins, 0, self.config.distance_bins - 1)

        valid_mask = torch.isfinite(distances) & mask
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)

        loss = F.cross_entropy(logits[valid_mask], distance_bins[valid_mask], reduction='mean')
        return loss

    def mlm_loss(self, node_embeddings, data):
        """Compute masked language modeling loss"""
        assert hasattr(data, 'mlm_mask') and hasattr(data, 'original_types') and hasattr(data, 'masked_types'), \
            "MLM data attributes are required for MLM loss"

        mask = data.mlm_mask
        original_types = data.original_types

        logits = self.mlm_head(node_embeddings)
        loss = F.cross_entropy(logits[mask], original_types[mask], reduction='mean')
        return loss

    def residue_type_prediction_loss(self, node_embeddings, batch):
        """
        Residue type prediction loss for protein pretraining.

        For each masked residue, aggregates its atom embeddings (scatter_mean),
        then predicts the amino acid type from the aggregated embedding.
        Only PDB graphs participate; non-standard residues are ignored (label=-1).
        """
        if not hasattr(batch, 'residue_mlm_mask'):
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)

        pdb_atom_mask = batch.pdb_atom_mask            # [num_atoms] bool
        flat_residue_idx = batch.pdb_flat_residue_idx  # [num_pdb_atoms] long
        residue_mlm_mask = batch.residue_mlm_mask      # [total_pdb_residues] bool
        residue_labels = batch.residue_labels           # [total_pdb_residues] long
        total_pdb_residues = batch._total_pdb_residues

        if pdb_atom_mask.sum() == 0 or residue_mlm_mask.sum() == 0:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)

        pdb_embeddings = node_embeddings[pdb_atom_mask]  # [num_pdb_atoms, graph_dim]
        residue_embeddings = scatter_mean(
            pdb_embeddings, flat_residue_idx, dim=0, dim_size=total_pdb_residues
        )  # [total_pdb_residues, graph_dim]

        masked_embeddings = residue_embeddings[residue_mlm_mask]  # [num_masked, graph_dim]
        true_labels = residue_labels[residue_mlm_mask]            # [num_masked]

        valid_mask = true_labels >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)

        logits = self.residue_head(masked_embeddings[valid_mask])  # [valid, 20]
        loss = F.cross_entropy(logits, true_labels[valid_mask], reduction='mean')
        return loss

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
