import math
import torch
import pytorch_lightning as pl
import bitsandbytes as bnb

from torch import nn
from torch_geometric.utils import to_dense_batch
from collections import defaultdict
from typing import Optional, List, Dict

from esa.utils.norm_layers import BN, LN
from esa.masked_layers import ESA
from esa.mlp_utils import SmallMLP
from torch_scatter import scatter_max

# Sub-module imports
from core.pretraining_config import PretrainingConfig, create_pretraining_config
from core.molecular_encoder import UniversalMolecularEncoder, nearest_multiple_of_8
from core.pretraining_tasks import PretrainingTasks, RESIDUE_3TO_IDX, NUM_RESIDUE_TYPES

# Backward-compatibility re-exports (other files can still import from here)
__all__ = [
    'PretrainingConfig',
    'create_pretraining_config',
    'UniversalMolecularEncoder',
    'nearest_multiple_of_8',
    'PretrainingTasks',
    'RESIDUE_3TO_IDX',
    'NUM_RESIDUE_TYPES',
    'PretrainingESAModel',
]


class PretrainingESAModel(pl.LightningModule):
    """
    Universal pretraining ESA model for all molecular systems.
    Orchestrates the encoder, ESA backbone, and all pretraining task losses.
    """

    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(vars(config))

        # Universal molecular encoder
        self.encoder = UniversalMolecularEncoder(config)

        # ESA backbone (with layer-wise equivariant integration if enabled)
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
            use_equivariant=config.use_equivariant_features,
            equivariant_lmax=config.equivariant_lmax,
            equivariant_num_features=config.equivariant_num_features,
            equivariant_fusion_method=config.equivariant_fusion_method,
            equivariant_cross_connection=config.equivariant_cross_connection,
        )

        self.esa_backbone = ESA(**st_args)

        if hasattr(config, 'use_distance_bias') and config.use_distance_bias:
            self.esa_backbone.use_distance_bias = True
            self.esa_backbone.distance_bias_scale = getattr(config, 'distance_bias_scale', 1.0)
            self.esa_backbone.distance_bias_cutoff = getattr(config, 'distance_bias_cutoff', 10.0)
            print(f"✅ Distance-based attention bias enabled: scale={self.esa_backbone.distance_bias_scale}, cutoff={self.esa_backbone.distance_bias_cutoff}Å")

        # Pretraining tasks (all loss heads)
        self.pretraining_tasks = PretrainingTasks(config)

        if config.use_equivariant_features:
            print("✅ Pretraining equivariant: ENABLED | pos→ESA→EquivariantSAB | gradients flow through equivariant branch")
        else:
            print("⬜ Pretraining equivariant: DISABLED | ESA uses invariant-only SAB/MAB (no EquivariantSAB)")

        # Output projection for node-level tasks
        if config.apply_attention_on == "edge":
            self.node_edge_mlp = None  # Created dynamically on first forward pass
        else:
            if config.mlp_type in ["standard", "gated_mlp"]:
                self.node_mlp = SmallMLP(
                    in_dim=config.graph_dim,
                    inter_dim=128,
                    out_dim=config.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=config.num_mlp_layers if config.num_mlp_layers > 1 else config.num_mlp_layers + 1,
                )

        # Normalization
        norm_fn = BN if config.norm_type == "BN" else LN
        self.mlp_norm = norm_fn(config.hidden_dims[0])

        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)

        if self.config.apply_attention_on == "node":
            self.config.is_node_task = True

    def forward(self, batch):
        """
        Forward pass for pretraining.

        Returns:
            (graph_embeddings, node_embeddings): graph_embeddings for graph-level tasks,
            node_embeddings (pre-PMA) for node-level losses.
        """
        edge_index, batch_mapping = batch.edge_index, batch.batch
        pos = getattr(batch, 'pos', None)

        assert hasattr(batch, 'z') and batch.z is not None, \
            "Batch must have 'z' attribute with atomic numbers"
        x = batch.z

        assert pos is not None, "Coordinates (pos) are required for pretraining tasks"

        # Center positions for translation invariance (per-graph)
        pos_centered = pos.clone()
        for graph_id in torch.unique(batch_mapping):
            mask = batch_mapping == graph_id
            if mask.sum() > 0:
                com = pos[mask].mean(dim=0, keepdim=True)
                pos_centered[mask] = pos[mask] - com
        pos = pos_centered
        batch.pos = pos_centered

        # Encode nodes
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

            # Route large/RNA batches through a separate MLP to preserve checkpoint compat
            is_rna = False
            if hasattr(batch, 'dataset_type'):
                dataset_type = batch.dataset_type
                if isinstance(dataset_type, (list, tuple)):
                    is_rna = any(str(dt).upper() == 'RNA' for dt in dataset_type)
                elif isinstance(dataset_type, str):
                    is_rna = dataset_type.upper() == 'RNA'

            num_atoms = x.size(0)
            use_sparse_computation = is_rna or (num_atoms > 500)

            device = node_embeddings.device
            h = h.to(device)

            if use_sparse_computation:
                if not hasattr(self, 'rna_node_edge_mlp') or self.rna_node_edge_mlp is None:
                    self.rna_node_edge_mlp = SmallMLP(
                        in_dim=h.shape[1],
                        inter_dim=128,
                        out_dim=self.config.hidden_dims[0],
                        use_ln=False,
                        dropout_p=0,
                        num_layers=max(2, self.config.num_mlp_layers),
                    ).to(device)
                h = self.rna_node_edge_mlp(h)
            else:
                if self.node_edge_mlp is None:
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

            edge_index = edge_index.to(device)
            batch_mapping = batch_mapping.to(device)
            edge_batch_index = batch_mapping.index_select(0, edge_index[0])

            if self.config.set_max_items and self.config.set_max_items > 0:
                num_max_items = nearest_multiple_of_8(self.config.set_max_items + 1)
            else:
                counts = torch.bincount(edge_batch_index)
                num_max_items = int(counts.max().item())
                num_max_items = nearest_multiple_of_8(num_max_items + 1)

            h, _ = to_dense_batch(h, edge_batch_index, fill_value=0, max_num_nodes=num_max_items)

            unique_nodes = torch.unique(edge_index)
            node_id_map = {int(n.item()): i for i, n in enumerate(unique_nodes)}
            local_edge_index = edge_index.clone()
            local_edge_index[0] = torch.tensor(
                [node_id_map[int(i)] for i in edge_index[0].tolist()], device=edge_index.device
            )
            local_edge_index[1] = torch.tensor(
                [node_id_map[int(i)] for i in edge_index[1].tolist()], device=edge_index.device
            )

            esa_output = self.esa_backbone(h, local_edge_index, batch_mapping, num_max_items=num_max_items, pos=pos)
            graph_emb, _ = esa_output
            h = graph_emb
            h_node_level = node_embeddings  # pre-attention for node-level tasks in edge mode

        # ============================================================
        # NODE ATTENTION PATH
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

            h, dense_batch_index = to_dense_batch(
                h, batch_mapping, fill_value=0, max_num_nodes=num_max_items
            )

            esa_output = self.esa_backbone(
                h, edge_index, batch_mapping=batch_mapping, num_max_items=num_max_items, pos=pos
            )
            graph_emb, node_emb_before_pma = esa_output

            if node_emb_before_pma.dim() == 3:
                h_node_level = node_emb_before_pma[dense_batch_index]
            else:
                h_node_level = node_emb_before_pma

            h = graph_emb

        return h, h_node_level

    # ------------------------------------------------------------------
    # Pre-encoder masking helpers (called before forward() in each step)
    # ------------------------------------------------------------------

    def _ensure_mlm_attributes(self, batch):
        """Add MLM masking on-the-fly if not already present.
        Protein graphs are skipped — residue type prediction provides a richer signal."""
        if "mlm" not in self.config.pretraining_tasks:
            return
        if hasattr(batch, 'mlm_mask') and batch.mlm_mask is not None:
            return
        if not hasattr(batch, 'z') or batch.z is None:
            return

        mask_ratio = getattr(self.config, 'mlm_mask_ratio', 0.15)
        mask_token = 0
        z = batch.z.long()
        batch.original_types = z.clone()
        num_nodes = z.size(0)
        mlm_mask = torch.zeros(num_nodes, dtype=torch.bool, device=z.device)
        masked_types = z.clone()

        # Identify protein atoms via pdb_atom_mask (set by _ensure_residue_mask_attributes).
        # If not yet available, fall back to dataset_type string matching.
        protein_atom_mask = None
        if hasattr(batch, 'pdb_atom_mask') and batch.pdb_atom_mask is not None:
            protein_atom_mask = batch.pdb_atom_mask
        else:
            dataset_types = getattr(batch, 'dataset_type', None)
            if dataset_types is not None and not isinstance(dataset_types, (list, tuple)):
                dataset_types = dataset_types.tolist() if hasattr(dataset_types, 'tolist') else [dataset_types]
            if dataset_types is not None and hasattr(batch, 'batch') and batch.batch is not None:
                protein_graphs = {i for i, dt in enumerate(dataset_types) if str(dt).lower() in ('pdb', 'protein')}
                if protein_graphs:
                    pg_tensor = torch.tensor(sorted(protein_graphs), dtype=torch.long, device=z.device)
                    protein_atom_mask = torch.isin(batch.batch, pg_tensor)

        if hasattr(batch, 'batch') and batch.batch is not None:
            graph_idx = batch.batch
            for g in graph_idx.unique().tolist():
                if protein_atom_mask is not None and protein_atom_mask[graph_idx == g].any():
                    continue  # residue masking handles proteins
                node_mask = (graph_idx == g)
                indices = torch.where(node_mask)[0]
                n = indices.size(0)
                if n > 0:
                    k = max(1, int(n * mask_ratio))
                    perm = torch.randperm(n, device=z.device)[:k]
                    selected = indices[perm]
                    mlm_mask[selected] = True
                    masked_types[selected] = mask_token
        else:
            k = max(1, int(num_nodes * mask_ratio))
            selected = torch.randperm(num_nodes, device=z.device)[:k]
            mlm_mask[selected] = True
            masked_types[selected] = mask_token

        batch.mlm_mask = mlm_mask
        batch.masked_types = masked_types
        batch.z = masked_types  # encoder sees masked input

    def _ensure_coord_denoising_attributes(self, batch):
        """Add coordinate noise on-the-fly if not already present."""
        if "coordinate_denoising" not in self.config.pretraining_tasks:
            return
        if hasattr(batch, 'clean_pos') and batch.clean_pos is not None:
            return
        if not hasattr(batch, 'pos') or batch.pos is None:
            return

        noise_std = getattr(self.config, 'coordinate_denoising_noise_std', 0.1)
        mask_ratio = getattr(self.config, 'coordinate_denoising_mask_ratio', 0.15)
        batch.clean_pos = batch.pos.clone()
        noise = torch.randn_like(batch.pos, device=batch.pos.device, dtype=batch.pos.dtype) * noise_std
        batch.pos = batch.clean_pos + noise
        num_nodes = batch.pos.size(0)
        coord_mask = torch.zeros(num_nodes, dtype=torch.bool, device=batch.pos.device)
        if hasattr(batch, 'batch') and batch.batch is not None:
            for g in batch.batch.unique().tolist():
                node_mask = (batch.batch == g)
                indices = torch.where(node_mask)[0]
                n = indices.size(0)
                k = max(1, int(n * mask_ratio))
                perm = torch.randperm(n, device=batch.pos.device)[:k]
                coord_mask[indices[perm]] = True
        else:
            k = max(1, int(num_nodes * mask_ratio))
            coord_mask[torch.randperm(num_nodes, device=batch.pos.device)[:k]] = True
        batch.coord_mask = coord_mask

    def _ensure_residue_mask_attributes(self, batch):
        """
        Pre-encoder masking for residue type prediction (PDB graphs only).

        Selects ~15% of residues per protein graph, zeros their atoms' z values so
        the encoder cannot see the true atom types, and stores bookkeeping tensors
        needed by PretrainingTasks.residue_type_prediction_loss.
        """
        if "residue_type_prediction" not in self.config.pretraining_tasks:
            return
        if hasattr(batch, 'residue_mlm_mask') and batch.residue_mlm_mask is not None:
            return
        if not hasattr(batch, 'block_idx') or not hasattr(batch, 'block_symbols'):
            return
        if not hasattr(batch, 'dataset_type'):
            return

        device = batch.z.device
        dataset_types = batch.dataset_type
        if not isinstance(dataset_types, (list, tuple)):
            dataset_types = dataset_types.tolist() if hasattr(dataset_types, 'tolist') else [dataset_types]

        num_graphs = len(dataset_types)
        pdb_graph_set = {i for i, dt in enumerate(dataset_types) if str(dt).lower() in ('pdb', 'protein')}

        if not pdb_graph_set:
            batch.pdb_atom_mask = torch.zeros(batch.z.size(0), dtype=torch.bool, device=device)
            batch.pdb_flat_residue_idx = torch.zeros(0, dtype=torch.long, device=device)
            batch.residue_labels = torch.zeros(0, dtype=torch.long, device=device)
            batch.residue_mlm_mask = torch.zeros(0, dtype=torch.bool, device=device)
            batch._total_pdb_residues = 0
            return

        batch_vec = batch.batch
        block_idx = batch.block_idx.long()
        # PyG collates list attributes as list-of-lists: block_symbols[g] = list of symbols for graph g
        block_symbols = batch.block_symbols

        max_block_per_graph, _ = scatter_max(block_idx, batch_vec, dim=0, dim_size=num_graphs)
        num_blocks_per_graph = max_block_per_graph + 1

        pdb_graph_tensor = torch.tensor(sorted(pdb_graph_set), dtype=torch.long, device=device)
        pdb_atom_mask = torch.isin(batch_vec, pdb_graph_tensor)

        pdb_residues_per_graph = torch.zeros(num_graphs, dtype=torch.long, device=device)
        for g in pdb_graph_set:
            pdb_residues_per_graph[g] = num_blocks_per_graph[g]

        pdb_residue_offset = torch.zeros(num_graphs, dtype=torch.long, device=device)
        if num_graphs > 1:
            pdb_residue_offset[1:] = pdb_residues_per_graph.cumsum(0)[:-1]

        total_pdb_residues = int(pdb_residues_per_graph.sum().item())

        pdb_atoms_graph = batch_vec[pdb_atom_mask]
        pdb_atoms_local_block = block_idx[pdb_atom_mask]
        pdb_flat_residue_idx = pdb_residue_offset[pdb_atoms_graph] + pdb_atoms_local_block

        residue_labels = torch.full((total_pdb_residues,), -1, dtype=torch.long, device=device)
        for g in pdb_graph_set:
            n_res = num_blocks_per_graph[g].item()
            res_offset = pdb_residue_offset[g].item()
            g_symbols = block_symbols[g]
            for local_r in range(n_res):
                sym = g_symbols[local_r] if local_r < len(g_symbols) else ''
                residue_labels[res_offset + local_r] = RESIDUE_3TO_IDX.get(sym, -1)

        mask_ratio = getattr(self.config, 'residue_mask_ratio', 0.15)
        residue_mlm_mask = torch.zeros(total_pdb_residues, dtype=torch.bool, device=device)
        for g in pdb_graph_set:
            res_offset = pdb_residue_offset[g].item()
            n_res = num_blocks_per_graph[g].item()
            if n_res == 0:
                continue
            k = max(1, int(n_res * mask_ratio))
            perm = torch.randperm(n_res, device=device)[:k]
            residue_mlm_mask[res_offset + perm] = True

        atom_residue_masked = residue_mlm_mask[pdb_flat_residue_idx]
        full_atom_residue_masked = torch.zeros(batch.z.size(0), dtype=torch.bool, device=device)
        full_atom_residue_masked[pdb_atom_mask] = atom_residue_masked
        batch.z = batch.z.clone()
        batch.z[full_atom_residue_masked] = 0

        batch.pdb_atom_mask = pdb_atom_mask
        batch.pdb_flat_residue_idx = pdb_flat_residue_idx
        batch.residue_labels = residue_labels
        batch.residue_mlm_mask = residue_mlm_mask
        batch._total_pdb_residues = total_pdb_residues

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_pretraining_losses(self, batch, graph_embeddings, node_embeddings):
        """Dispatch to all active task loss functions and accumulate total loss."""
        losses = {}
        total_loss = 0.0

        if "long_range_distance" in self.config.pretraining_tasks:
            losses['long_range_distance'] = self.pretraining_tasks.long_range_distance_loss(node_embeddings, batch)
            total_loss += self.config.task_weights['long_range_distance'] * losses['long_range_distance']

        if "short_range_distance" in self.config.pretraining_tasks:
            if hasattr(batch, 'edge_index') and hasattr(batch, 'pos'):
                edge_index = batch.edge_index
                pos = batch.pos
                distances = torch.norm(pos[edge_index[1]] - pos[edge_index[0]], dim=1)
                mask = (torch.rand(edge_index.size(1)) < 0.15).to(distances.device)
                dist_loss = self.pretraining_tasks.short_range_distance_loss(
                    node_embeddings, edge_index, distances, mask
                )
                losses['short_range_distance'] = dist_loss
                total_loss += self.config.task_weights['short_range_distance'] * dist_loss

        if "mlm" in self.config.pretraining_tasks:
            losses['mlm'] = self.pretraining_tasks.mlm_loss(node_embeddings, batch)
            total_loss += self.config.task_weights['mlm'] * losses['mlm']

        if "residue_type_prediction" in self.config.pretraining_tasks:
            losses['residue_type_prediction'] = self.pretraining_tasks.residue_type_prediction_loss(node_embeddings, batch)
            w = self.config.task_weights.get('residue_type_prediction', 1.0)
            total_loss += w * losses['residue_type_prediction']

        if "coordinate_denoising" in self.config.pretraining_tasks:
            losses['coordinate_denoising'] = self.pretraining_tasks.coordinate_denoising_loss(node_embeddings, batch)
            w = self.config.task_weights.get('coordinate_denoising', 1.0)
            total_loss += w * losses['coordinate_denoising']

        return losses, total_loss

    def _compute_per_type_losses(self, batch, graph_embeddings, node_embeddings):
        """
        Compute losses separately for each dataset type using the same embeddings.

        Returns:
            Dict[str, Dict[str, float]]: {dtype: {task: loss_value}}
        """
        if not hasattr(batch, 'dataset_type'):
            return {}

        type_losses = {}
        unique_types = set(batch.dataset_type) if isinstance(batch.dataset_type, (list, tuple)) else {batch.dataset_type}
        device = node_embeddings.device

        for dtype in unique_types:
            if isinstance(batch.dataset_type, (list, tuple)):
                graph_mask = torch.tensor([dt == dtype for dt in batch.dataset_type], device=device)
            else:
                graph_mask = torch.ones(batch.num_graphs, dtype=torch.bool, device=device)

            graph_indices = torch.where(graph_mask)[0]
            node_mask = torch.isin(batch.batch, graph_indices)

            if node_mask.sum() == 0:
                continue

            dtype_losses = {}
            dtype_total = 0.0

            if "short_range_distance" in self.config.pretraining_tasks:
                if hasattr(batch, 'edge_index') and batch.edge_index is not None and hasattr(batch, 'pos') and batch.pos is not None:
                    edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
                    if edge_mask.sum() > 0:
                        masked_edge_index = batch.edge_index[:, edge_mask]
                        masked_pos = batch.pos[node_mask]
                        node_mapping = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
                        node_mapping[node_mask] = torch.arange(node_mask.sum(), device=device)
                        remapped_edge_index = node_mapping[masked_edge_index]
                        distances = torch.norm(
                            masked_pos[remapped_edge_index[1]] - masked_pos[remapped_edge_index[0]], dim=1
                        )
                        random_mask = (torch.rand(remapped_edge_index.size(1)) < 0.15).to(distances.device)
                        masked_node_emb = node_embeddings[node_mask]
                        dist_loss = self.pretraining_tasks.short_range_distance_loss(
                            masked_node_emb, remapped_edge_index, distances, random_mask
                        )
                        dtype_losses['short_range_distance'] = dist_loss
                        dtype_total += self.config.task_weights['short_range_distance'] * dist_loss

            if "long_range_distance" in self.config.pretraining_tasks:
                if hasattr(batch, 'pos') and batch.pos is not None:
                    masked_batch = batch.__class__()
                    if hasattr(batch, 'x') and batch.x is not None:
                        masked_batch.x = batch.x[node_mask]
                    masked_batch.pos = batch.pos[node_mask]
                    masked_batch.batch = torch.zeros(node_mask.sum(), dtype=torch.long, device=device)
                    masked_batch.num_graphs = 1
                    masked_node_emb = node_embeddings[node_mask]
                    long_range_loss = self.pretraining_tasks.long_range_distance_loss(masked_node_emb, masked_batch)
                    dtype_losses['long_range_distance'] = long_range_loss
                    dtype_total += self.config.task_weights['long_range_distance'] * long_range_loss

            if "mlm" in self.config.pretraining_tasks and dtype.lower() not in ('protein', 'pdb'):
                if hasattr(batch, 'mlm_mask') and batch.mlm_mask is not None:
                    masked_mlm = batch.mlm_mask[node_mask]
                    if masked_mlm.sum() > 0:
                        masked_batch = batch.__class__()
                        if hasattr(batch, 'x') and batch.x is not None:
                            masked_batch.x = batch.x[node_mask]
                        if hasattr(batch, 'z') and batch.z is not None:
                            masked_batch.z = batch.z[node_mask]
                        masked_batch.mlm_mask = masked_mlm
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
                            pass

            if "coordinate_denoising" in self.config.pretraining_tasks:
                if hasattr(batch, 'clean_pos') and batch.clean_pos is not None and getattr(batch, 'coord_mask', None) is not None:
                    masked_coord = batch.coord_mask[node_mask]
                    if masked_coord.sum() > 0:
                        masked_batch = batch.__class__()
                        masked_batch.clean_pos = batch.clean_pos[node_mask]
                        masked_batch.pos = batch.pos[node_mask]
                        masked_batch.coord_mask = masked_coord
                        masked_node_emb = node_embeddings[node_mask]
                        coord_loss = self.pretraining_tasks.coordinate_denoising_loss(masked_node_emb, masked_batch)
                        dtype_losses['coordinate_denoising'] = coord_loss
                        dtype_total += self.config.task_weights.get('coordinate_denoising', 1.0) * coord_loss

            dtype_losses['total'] = dtype_total
            type_losses[dtype] = dtype_losses

        return type_losses

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        self._ensure_residue_mask_attributes(batch)   # first: sets pdb_atom_mask
        self._ensure_mlm_attributes(batch)             # uses pdb_atom_mask to skip proteins
        self._ensure_coord_denoising_attributes(batch)

        if getattr(self.config, 'log_batch_stats', False) and batch_idx % 10 == 0:
            if hasattr(batch, 'dataset_type'):
                print(f"✅ Batch {batch_idx} has dataset_type: {batch.dataset_type}")
            else:
                print(f"⚠️  Batch {batch_idx} MISSING dataset_type attribute!")
        if getattr(self.config, 'log_batch_stats', False) and batch_idx % 10 == 0:
            num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            num_atoms = batch.num_nodes if hasattr(batch, 'num_nodes') else batch.x.size(0)
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

        if self.config.compute_per_type_losses and batch_idx % self.config.log_per_type_frequency == 0:
            type_losses = self._compute_per_type_losses(batch, graph_embeddings, node_embeddings)
            for dtype, dtype_losses in type_losses.items():
                for task_name, loss_value in dtype_losses.items():
                    self.log(f"train_{dtype}_{task_name}_loss", loss_value, on_step=True, logger=True)
            if batch_idx % 50 == 0 and type_losses:
                print(f"\n  📊 Per-Type Losses (Batch {batch_idx}):")
                for dtype, dtype_losses in type_losses.items():
                    loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in dtype_losses.items() if k != 'total'])
                    print(f"    {dtype.upper():12s}: {loss_str} | Total: {dtype_losses['total']:.4f}")

        return total_loss

    def validation_step(self, batch, batch_idx):
        self._ensure_residue_mask_attributes(batch)
        self._ensure_mlm_attributes(batch)
        self._ensure_coord_denoising_attributes(batch)
        graph_embeddings, node_embeddings = self.forward(batch)
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)

        for task_name, loss_value in losses.items():
            self.log(f"val_{task_name}_loss", loss_value, on_epoch=True, logger=True)
        self.log("val_total_loss", total_loss, on_epoch=True, logger=True)

        if self.config.compute_per_type_losses:
            type_losses = self._compute_per_type_losses(batch, graph_embeddings, node_embeddings)
            for dtype, dtype_losses in type_losses.items():
                for task_name, loss_value in dtype_losses.items():
                    self.log(f"val_{dtype}_{task_name}_loss", loss_value, on_epoch=True, logger=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        self._ensure_residue_mask_attributes(batch)
        self._ensure_mlm_attributes(batch)
        self._ensure_coord_denoising_attributes(batch)
        graph_embeddings, node_embeddings = self.forward(batch)
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)

        for task_name, loss_value in losses.items():
            self.log(f"test_{task_name}_loss", loss_value, on_epoch=True, logger=True)
        self.log("test_total_loss", total_loss, on_epoch=True, logger=True)

        if self.config.compute_per_type_losses:
            type_losses = self._compute_per_type_losses(batch, graph_embeddings, node_embeddings)
            for dtype, dtype_losses in type_losses.items():
                for task_name, loss_value in dtype_losses.items():
                    self.log(f"test_{dtype}_{task_name}_loss", loss_value, on_epoch=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = bnb.optim.AdamW8bit(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.optimiser_weight_decay,
        )

        total_epochs = getattr(self.config, 'max_epochs', 100)
        warmup_epochs = max(1, int(total_epochs * 0.1))

        def lr_lambda(current_epoch: int):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs)
            progress = (current_epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return 0.1 + 0.9 * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": self.config.monitor_loss_name}]

    def get_embeddings(self, batch):
        """Get embeddings for downstream tasks (no gradient)."""
        with torch.no_grad():
            return self.forward(batch)


if __name__ == "__main__":
    config = create_pretraining_config()
    model = PretrainingESAModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Pretraining tasks: {config.pretraining_tasks}")
    print(f"Task weights: {config.task_weights}")
