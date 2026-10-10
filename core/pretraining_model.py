import math
from types import SimpleNamespace
import torch
import pytorch_lightning as pl
import bitsandbytes as bnb

from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from collections import defaultdict
from typing import Optional, List, Dict

from esa.utils.norm_layers import BN, LN
from esa.masked_layers import ESA
from esa.mlp_utils import SmallMLP
# Sub-module imports
from core.pretraining_config import PretrainingConfig, create_pretraining_config
from core.block_masking import legacy_oxt_eligible_node_mask, select_whole_blocks
from core.masked_coordinate_corruption import apply_masked_slot_template_
from core.masked_slot_mapping import flatten_pos_codes
from core.molecular_encoder import (
    UniversalMolecularEncoder,
    build_domain_local_slot_ids,
    nearest_multiple_of_8,
)
from core.pretraining_tasks import PretrainingTasks
from core.virtual_slot_graph import (
    build_geometry_neutral_graph_view,
    build_slot_readout_edges_for_owner_mode,
)
from data_loading.masking_utils import (
    compute_coconut_masking_metrics,
    get_graph_dataset_type,
    is_coconut_dataset_type,
    select_mask_indices,
    slice_bond_edge_index,
)

# Backward-compatibility re-exports (other files can still import from here)
__all__ = [
    'PretrainingConfig',
    'create_pretraining_config',
    'UniversalMolecularEncoder',
    'nearest_multiple_of_8',
    'PretrainingTasks',
    'PretrainingESAModel',
]


_LEGACY_INVARIANT_SAB_TOKEN = ".sab.invariant_sab."
_REAL_EQUIVARIANT_STATE_TOKENS = (
    ".equivariant_interaction.",
    ".equivariant_to_scalar.",
    ".inv_to_equiv.",
    ".equiv_to_inv_gate.",
)


def _migrate_legacy_invariant_sab_state_dict(state_dict, expected_keys):
    """Rename wrapper-only SAB keys for the direct invariant architecture.

    Historical configs requested equivariance even though NequIP/e3nn were
    unavailable. Those checkpoints contain only invariant SAB tensors nested
    under ``sab.invariant_sab``. The tensor values are directly compatible
    with today's invariant SAB; only their names differ.
    """
    if any(
        token in key
        for key in state_dict
        for token in _REAL_EQUIVARIANT_STATE_TOKENS
    ):
        raise RuntimeError(
            "checkpoint contains real equivariant branch weights and cannot "
            "be silently loaded into an invariant-only model"
        )

    legacy_keys = [
        key for key in state_dict if _LEGACY_INVARIANT_SAB_TOKEN in key
    ]
    if not legacy_keys:
        return state_dict, 0

    original_keys = set(state_dict)
    expected_keys = set(expected_keys)
    migrated = state_dict.__class__()
    for key, value in state_dict.items():
        target = key.replace(_LEGACY_INVARIANT_SAB_TOKEN, ".sab.")
        if target != key:
            if target in original_keys:
                raise RuntimeError(
                    f"checkpoint contains both legacy and direct SAB key: {target}"
                )
            if target not in expected_keys:
                raise RuntimeError(
                    f"legacy SAB key has no invariant-model target: {target}"
                )
        if target in migrated:
            raise RuntimeError(f"checkpoint key migration collision: {target}")
        migrated[target] = value

    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        migrated._metadata = metadata.__class__(
            (
                key.replace(".sab.invariant_sab", ".sab"),
                value,
            )
            for key, value in metadata.items()
        )
    return migrated, len(legacy_keys)


class PretrainingESAModel(pl.LightningModule):
    """
    Universal pretraining ESA model for all molecular systems.
    Orchestrates the encoder, ESA backbone, and all pretraining task losses.
    """

    def __init__(
        self,
        config: Optional[PretrainingConfig] = None,
        **legacy_hparams,
    ):
        super().__init__()
        if config is None:
            if not legacy_hparams:
                raise TypeError("PretrainingESAModel requires a config")
            # Lightning checkpoints written before the nested-config format
            # stored every config field as a top-level hyperparameter. Their
            # equivariant=true setting produced no equivariant parameters in
            # the supported training environment, so reconstruct the actual
            # invariant architecture before loading those tensors.
            config = create_pretraining_config(**legacy_hparams)
            config.use_equivariant_features = False
        elif isinstance(config, dict):
            if legacy_hparams:
                raise TypeError(
                    "nested config cannot be combined with legacy hyperparameters"
                )
            config = create_pretraining_config(**config)
        elif legacy_hparams:
            raise TypeError(
                "config object cannot be combined with legacy hyperparameters"
            )

        self.config = config
        self.save_hyperparameters({"config": dict(vars(config))})

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

        self.equivariant_requested = bool(config.use_equivariant_features)
        self.equivariant_effective = bool(
            self.esa_backbone.use_equivariant
        )
        print(
            "Pretraining geometry: "
            f"invariant_3d={'ENABLED' if config.use_3d_coordinates else 'DISABLED'} | "
            f"equivariant_requested={self.equivariant_requested} | "
            f"equivariant_effective={self.equivariant_effective}"
        )

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

    def on_load_checkpoint(self, checkpoint):
        if self.equivariant_effective:
            return
        state_dict, migrated_count = _migrate_legacy_invariant_sab_state_dict(
            checkpoint["state_dict"], self.state_dict().keys()
        )
        checkpoint["state_dict"] = state_dict
        if migrated_count:
            print(
                "Migrated "
                f"{migrated_count} legacy invariant SAB checkpoint keys "
                "to the direct invariant layout."
            )

    def _inject_latent_slot_readouts(
        self,
        batch,
        graph_view,
        full_node_embeddings: torch.Tensor,
        full_slot_ids: torch.Tensor,
        slot_readout_mode: str,
    ) -> torch.Tensor:
        """Create hidden side/base slot states from encoded backbone states.

        The fixed-width slots stay available to MLM and masked-distance heads,
        but they are not ESA nodes. Each state is the mean of its residue or
        nucleotide backbone plus its existing domain-local slot embedding.
        """
        target_nodes = torch.where(graph_view.target_slot_mask)[0]
        if target_nodes.numel() == 0:
            return full_node_embeddings

        slot_embedding = getattr(
            self.encoder, "domain_local_slot_embedding", None
        )
        if slot_embedding is None:
            raise RuntimeError(
                "latent slot readout requires use_domain_local_slot_embedding=true"
            )

        readout_edges = build_slot_readout_edges_for_owner_mode(
            batch, graph_view, slot_readout_mode
        )
        context_sum = full_node_embeddings.new_zeros(
            full_node_embeddings.shape
        )
        context_count = full_node_embeddings.new_zeros(
            (full_node_embeddings.size(0),)
        )
        if readout_edges.numel():
            slots, anchors = readout_edges
            context_sum.index_add_(
                0, slots, full_node_embeddings.index_select(0, anchors)
            )
            context_count.index_add_(
                0,
                slots,
                torch.ones(
                    slots.numel(),
                    dtype=full_node_embeddings.dtype,
                    device=full_node_embeddings.device,
                ),
            )
        if slot_readout_mode == "correct" and bool(
            (context_count[target_nodes] == 0).any()
        ):
            raise RuntimeError("a latent slot has no owner-backbone readout")

        context = context_sum[target_nodes] / context_count[
            target_nodes
        ].clamp_min(1).unsqueeze(-1)
        slot_states = slot_embedding(full_slot_ids[target_nodes])
        if slot_states.size(-1) != context.size(-1):
            raise RuntimeError(
                "latent slot/readout dimensions differ: "
                f"slot={slot_states.size(-1)}, backbone={context.size(-1)}"
            )
        slot_states = F.layer_norm(
            context + slot_states, (context.size(-1),)
        )
        result = full_node_embeddings.clone()
        result.index_copy_(0, target_nodes, slot_states)
        return result

    def forward(self, batch, slot_readout_mode: str = "correct"):
        """
        Forward pass for pretraining.

        Returns:
            (graph_embeddings, node_embeddings): graph_embeddings for graph-level tasks,
            node_embeddings (pre-PMA) for node-level losses.
        """
        edge_index, batch_mapping = batch.edge_index, batch.batch
        attention_edge_index = edge_index
        pos = getattr(batch, 'pos', None)

        assert hasattr(batch, 'z') and batch.z is not None, \
            "Batch must have 'z' attribute with atomic numbers"
        x = batch.z

        assert pos is not None, "Coordinates (pos) are required for pretraining tasks"

        full_num_nodes = int(x.size(0))
        graph_view = build_geometry_neutral_graph_view(batch, self.config)
        if graph_view is not None:
            full_slot_ids = build_domain_local_slot_ids(
                batch, full_num_nodes, x.device
            )
            active = graph_view.compact_to_full
            x = x[active]
            pos = graph_view.compact_pos
            batch_mapping = batch.batch[active]
            edge_index = graph_view.compact_edge_index
            # ESA sees only the physical graph. Owner routing is a post-ESA
            # latent readout and therefore cannot affect attention geometry.
            attention_edge_index = graph_view.compact_attention_edge_index
            encoder_batch = SimpleNamespace(
                batch=batch_mapping,
                block_idx=batch.block_idx[active],
                edge_index=edge_index,
                bond_edge_index=graph_view.compact_bond_edge_index,
                has_explicit_bonds=graph_view.has_explicit_bonds,
                geometry_valid_mask=graph_view.geometry_valid_mask[active],
                domain_local_slot_ids=full_slot_ids[active],
                dataset_type=getattr(batch, 'dataset_type', None),
                num_graphs=getattr(batch, 'num_graphs', 1),
            )

            # Full-index diagnostics and loss routing remain on the original
            # batch. Clean target coordinates are never copied into encoder_batch.
            batch.active_node_mask = graph_view.active_node_mask
            batch.geometry_valid_mask = graph_view.geometry_valid_mask
            batch.target_slot_mask = graph_view.target_slot_mask
            batch.target_real_slot_mask = graph_view.target_real_slot_mask
            batch.domain_local_slot_ids = full_slot_ids
            batch.slot_readout_edge_index = (
                graph_view.slot_readout_edge_index
            )
            batch.physical_edge_index = graph_view.physical_edge_index
            raw_edges = batch.edge_index
            valid_cached_edges = (
                graph_view.geometry_valid_mask[raw_edges[0]]
                & graph_view.geometry_valid_mask[raw_edges[1]]
            )
            batch.virtual_slot_graph_stats = {
                'visible_virtual_nodes_removed': int(
                    (
                        ~graph_view.active_node_mask
                        & ~graph_view.target_slot_mask
                    ).sum().item()
                ),
                'latent_slot_nodes': int(
                    graph_view.target_slot_mask.sum().item()
                ),
                'latent_real_slot_nodes': int(
                    graph_view.target_real_slot_mask.sum().item()
                ),
                'invalid_cached_edges_removed': int(
                    (~valid_cached_edges).sum().item()
                ),
                'physical_edges': int(
                    graph_view.physical_edge_index.size(1)
                ),
                'slot_readout_links': int(
                    graph_view.slot_readout_edge_index.size(1)
                ),
            }
            # Keep existing downstream loss code in the same centered frame,
            # but compute the center from physical atoms only. Invalid slot
            # coordinates remain unavailable to the encoder view above.
            batch.pos = graph_view.centered_full_pos_for_losses
        else:
            # Preserve the legacy path exactly when the feature is disabled.
            pos_centered = pos.clone()
            for graph_id in torch.unique(batch_mapping):
                mask = batch_mapping == graph_id
                if mask.sum() > 0:
                    com = pos[mask].mean(dim=0, keepdim=True)
                    pos_centered[mask] = pos[mask] - com
            pos = pos_centered
            batch.pos = pos_centered
            encoder_batch = batch

        # Encode nodes
        node_embeddings = self.encoder(x, pos, encoder_batch)
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
                h,
                attention_edge_index,
                batch_mapping=batch_mapping,
                num_max_items=num_max_items,
                pos=pos,
            )
            graph_emb, node_emb_before_pma = esa_output

            if node_emb_before_pma.dim() == 3:
                h_node_level = node_emb_before_pma[dense_batch_index]
            else:
                h_node_level = node_emb_before_pma

            h = graph_emb

        if graph_view is not None:
            compact_node_embeddings = h_node_level
            h_node_level = compact_node_embeddings.new_zeros(
                (full_num_nodes, compact_node_embeddings.size(-1))
            )
            h_node_level.index_copy_(
                0, graph_view.compact_to_full, compact_node_embeddings
            )
            h_node_level = PretrainingESAModel._inject_latent_slot_readouts(
                self,
                batch,
                graph_view,
                h_node_level,
                full_slot_ids,
                slot_readout_mode,
            )

        return h, h_node_level

    # ------------------------------------------------------------------
    # Pre-encoder masking helpers (called before forward() in each step)
    # ------------------------------------------------------------------

    def _graph_masking_params(self, dataset_type: Optional[str], for_coord: bool = False):
        """Ratio and bond edges config for a single graph in a batch."""
        if is_coconut_dataset_type(dataset_type):
            ratio = getattr(self.config, 'coconut_mask_ratio', 0.15)
            method = getattr(self.config, 'coconut_masking_method', 'random_atom')
            num_walks = getattr(self.config, 'coconut_num_random_walks', 1)
        else:
            if for_coord:
                ratio = getattr(self.config, 'coordinate_denoising_mask_ratio', 0.15)
            else:
                ratio = getattr(self.config, 'mlm_mask_ratio', 0.15)
            method = 'random_atom'
            num_walks = 1
        return ratio, method, num_walks

    def _select_graph_mask_indices(
        self,
        batch,
        graph_idx: int,
        indices: torch.Tensor,
        for_coord: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        n = indices.size(0)
        device = indices.device
        dtype = get_graph_dataset_type(getattr(batch, 'dataset_type', None), graph_idx)
        ratio, coconut_method, num_walks = self._graph_masking_params(dtype, for_coord=for_coord)
        bond_edge_index = getattr(batch, 'bond_edge_index', None)
        bond_sub = None
        if bond_edge_index is not None and bond_edge_index.numel() > 0:
            node_offset = int(indices[0].item())
            bond_sub = slice_bond_edge_index(bond_edge_index, node_offset, n)
            if bond_sub.numel() > 0:
                bond_sub = bond_sub.to(device)
        local_indices = select_mask_indices(
            n,
            ratio,
            method='random_atom',
            bond_edge_index=bond_sub,
            dataset_type=dtype,
            coconut_method=coconut_method,
            num_walks=num_walks,
            device=device,
            generator=generator,
        )
        return indices[local_indices]

    def _ensure_mlm_attributes(self, batch, batch_idx: int = 0, is_validation: bool = False):
        """Add MLM masking on-the-fly if not already present.

        Default: atom-level random masking for all graph types — applied
        identically to proteins, metabolites, QM9, RNA, etc.

        `config.protein_block_mlm=True` switches protein/pdb graphs to whole
        residue masking. `config.rna_block_mlm=True` independently switches
        RNA graphs to whole nucleotide masking. Other domains remain
        atom-level. Both block paths use one shared selection object so future
        coordinate templating can consume the MLM selection without a second
        random draw.

        Args:
            batch_idx: Batch index used for deterministic val/test masking.
            is_validation: True for validation/test steps; activates the
                deterministic seed path for MLM masks across cycles. Other
                randomized geometry/target paths can still vary the losses.
        """
        if "mlm" not in self.config.pretraining_tasks:
            return
        if hasattr(batch, 'mlm_mask') and batch.mlm_mask is not None:
            return
        if not hasattr(batch, 'z') or batch.z is None:
            return

        use_block_for_protein = getattr(self.config, 'protein_block_mlm', False)
        use_block_for_rna = getattr(self.config, 'rna_block_mlm', False)
        atom_mask_ratio = getattr(self.config, 'mlm_mask_ratio', 0.15)
        protein_block_mask_ratio = getattr(
            self.config, 'protein_block_mlm_ratio', 0.15
        )
        rna_block_mask_ratio = getattr(self.config, 'rna_block_mlm_ratio', 0.15)
        mask_token = 0
        z = batch.z.long()
        batch.original_types = z.clone()
        num_nodes = z.size(0)
        device = z.device
        mlm_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        masked_types = z.clone()

        # Per-graph dtype map. PyG collates dataset_type as a list-of-strings
        # over graphs in the batch; len == batch.num_graphs.
        dtype_list = self._graph_dtype_list(batch)
        protein_keys = {'pdb', 'protein'}
        rna_keys = {'rna'}
        template_settings = getattr(self.config, 'masked_slot_template', None) or {}
        neutral_slot_settings = getattr(
            self.config, 'geometry_neutral_virtual_slots', None
        ) or {}
        exclude_legacy_oxt = bool(
            getattr(
                self.config,
                'protein_block_mlm_exclude_legacy_oxt',
                False,
            )
            or (
                template_settings.get('enabled', False)
                and template_settings.get('protein', False)
            )
        )
        pos_codes = None
        if exclude_legacy_oxt:
            pos_codes = flatten_pos_codes(batch.pos_code, num_nodes)

        # Deterministic generator for val/test so each validation batch keeps
        # exactly the same masked blocks across every validation cycle and
        # epoch. Training stays stochastic.
        gen = None
        if is_validation:
            seed = int(getattr(self.config, 'seed', 0)) + int(batch_idx) * 100003
            gen = torch.Generator(device='cpu')
            gen.manual_seed(seed)

        def _randperm(n: int) -> torch.Tensor:
            if gen is not None:
                return torch.randperm(n, generator=gen).to(device)
            return torch.randperm(n, device=device)

        if hasattr(batch, 'batch') and batch.batch is not None:
            graph_idx = batch.batch
            for g in graph_idx.unique().tolist():
                node_mask = (graph_idx == g)
                indices = torch.where(node_mask)[0]
                n = indices.size(0)
                if n == 0:
                    continue

                g_dtype = dtype_list[g] if (dtype_list is not None and g < len(dtype_list)) else None
                is_protein = (g_dtype is not None
                              and str(g_dtype).lower() in protein_keys)
                is_rna = (g_dtype is not None
                          and str(g_dtype).lower() in rna_keys)
                has_block_idx = (
                    hasattr(batch, 'block_idx')
                    and batch.block_idx is not None
                    and batch.block_idx.numel() == num_nodes
                )

                block_mask_ratio = None
                if is_protein and use_block_for_protein:
                    block_mask_ratio = protein_block_mask_ratio
                elif is_rna and use_block_for_rna:
                    block_mask_ratio = rna_block_mask_ratio

                if block_mask_ratio is not None and has_block_idx:
                    eligible_node_mask = None
                    if is_protein and exclude_legacy_oxt:
                        graph_pos_codes = tuple(
                            pos_codes[int(index)] for index in indices
                        )
                        eligible_node_mask = legacy_oxt_eligible_node_mask(
                            batch.block_idx[indices], graph_pos_codes
                        )
                    needs_exact_slot_block = (
                        neutral_slot_settings.get('enabled', False)
                        and (
                            (is_protein and neutral_slot_settings.get('protein', False))
                            or (is_rna and neutral_slot_settings.get('rna', False))
                        )
                    )
                    if needs_exact_slot_block:
                        reason_bits = getattr(
                            batch, 'template_reason_bits', None
                        )
                        if reason_bits is None or reason_bits.numel() != num_nodes:
                            raise ValueError(
                                "geometry-neutral block masking requires "
                                "template_reason_bits runtime metadata"
                            )
                        exact_block_nodes = reason_bits[indices] == 0
                        eligible_node_mask = (
                            exact_block_nodes
                            if eligible_node_mask is None
                            else eligible_node_mask & exact_block_nodes
                        )
                    selection = select_whole_blocks(
                        batch.block_idx[indices],
                        block_mask_ratio,
                        generator=gen,
                        eligible_node_mask=eligible_node_mask,
                    )
                    selected = indices[selection.selected_node_mask]
                    mlm_mask[selected] = True
                    masked_types[selected] = mask_token
                else:
                    if is_coconut_dataset_type(g_dtype):
                        # COCONUT may opt into bond-aware masking. Keep that
                        # feature isolated, including the legacy validation RNG
                        # stream used by protein/RNA blocks and UniMol atoms.
                        coconut_gen = None
                        if gen is not None:
                            coconut_gen = torch.Generator(device='cpu')
                            coconut_gen.manual_seed(seed + (int(g) + 1) * 1009)
                        selected = PretrainingESAModel._select_graph_mask_indices(
                            self, batch, int(g), indices, for_coord=False,
                            generator=coconut_gen,
                        )
                    else:
                        k = max(1, int(n * atom_mask_ratio))
                        selected = indices[_randperm(n)[:k]]
                    mlm_mask[selected] = True
                    masked_types[selected] = mask_token
        else:
            # No batch index → treat whole tensor as single graph
            dtype = get_graph_dataset_type(getattr(batch, 'dataset_type', None), 0)
            if is_coconut_dataset_type(dtype):
                ratio, coconut_method, num_walks = PretrainingESAModel._graph_masking_params(
                    self, dtype, for_coord=False
                )
                bond_edge_index = getattr(batch, 'bond_edge_index', None)
                local = select_mask_indices(
                    num_nodes,
                    ratio,
                    method='random_atom',
                    bond_edge_index=bond_edge_index,
                    dataset_type=dtype,
                    coconut_method=coconut_method,
                    num_walks=num_walks,
                    device=z.device,
                    generator=gen,
                )
            else:
                k = max(1, int(num_nodes * atom_mask_ratio))
                local = _randperm(num_nodes)[:k]
            mlm_mask[local] = True
            masked_types[local] = mask_token

        batch.mlm_mask = mlm_mask
        batch.masked_types = masked_types
        batch.z = masked_types
        batch.x = masked_types

    @staticmethod
    def _graph_dtype_list(batch):
        """Return per-graph dtype list (len = batch.num_graphs) or None."""
        if not hasattr(batch, 'dataset_type') or batch.dataset_type is None:
            return None
        dt = batch.dataset_type
        if isinstance(dt, (list, tuple)):
            return list(dt)
        # Single string → broadcast to num_graphs entries.
        n = getattr(batch, 'num_graphs', 1) or 1
        return [dt] * n

    def _log_coconut_masking_metrics(self, batch, batch_idx: int) -> None:
        """Log coconut-only masking diagnostics to wandb under the ``masking/`` namespace."""
        if "mlm" not in self.config.pretraining_tasks:
            return
        if not getattr(self.config, "log_coconut_masking_stats", True):
            return
        freq = int(getattr(self.config, "log_coconut_masking_frequency", 50))
        if freq > 0 and batch_idx % freq != 0:
            return

        metrics = compute_coconut_masking_metrics(batch, self.config)
        if metrics is None:
            return

        for key in (
            "mask_frac",
            "mask_target_frac",
            "mask_connected_frac",
            "bond_edges_per_graph",
            "mask_fallback_rate",
        ):
            self.log(
                f"masking/{key}",
                float(metrics[key]),
                on_step=True,
                on_epoch=False,
                logger=True,
                batch_size=1,
            )

    def _ensure_coord_denoising_attributes(self, batch):
        """Add coordinate noise on-the-fly if not already present."""
        if "coordinate_denoising" not in self.config.pretraining_tasks:
            return
        if hasattr(batch, 'clean_pos') and batch.clean_pos is not None:
            return
        if not hasattr(batch, 'pos') or batch.pos is None:
            return

        noise_std = getattr(self.config, 'coordinate_denoising_noise_std', 0.1)
        batch.clean_pos = batch.pos.clone()
        noise = torch.randn_like(batch.pos, device=batch.pos.device, dtype=batch.pos.dtype) * noise_std
        batch.pos = batch.clean_pos + noise
        num_nodes = batch.pos.size(0)

        if hasattr(batch, 'mlm_mask') and batch.mlm_mask is not None:
            batch.coord_mask = batch.mlm_mask.clone()
            return

        coord_mask = torch.zeros(num_nodes, dtype=torch.bool, device=batch.pos.device)
        if hasattr(batch, 'batch') and batch.batch is not None:
            for g in batch.batch.unique().tolist():
                node_mask = (batch.batch == g)
                indices = torch.where(node_mask)[0]
                if indices.size(0) > 0:
                    selected = self._select_graph_mask_indices(batch, int(g), indices, for_coord=True)
                    coord_mask[selected] = True
        else:
            dtype = get_graph_dataset_type(getattr(batch, 'dataset_type', None), 0)
            ratio, coconut_method, num_walks = self._graph_masking_params(dtype, for_coord=True)
            bond_edge_index = getattr(batch, 'bond_edge_index', None)
            local = select_mask_indices(
                num_nodes,
                ratio,
                method='random_atom',
                bond_edge_index=bond_edge_index,
                dataset_type=dtype,
                coconut_method=coconut_method,
                num_walks=num_walks,
                device=batch.pos.device,
            )
            coord_mask[local] = True
        batch.coord_mask = coord_mask

    def _ensure_masked_slot_template(self, batch):
        """Apply the runtime template only when its explicit feature flag is on."""
        return apply_masked_slot_template_(batch, self.config)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_pretraining_losses(self, batch, graph_embeddings, node_embeddings, is_training: bool = True):
        """Dispatch to all active task loss functions and accumulate total loss."""
        losses = {}
        total_loss = 0.0

        if "long_range_distance" in self.config.pretraining_tasks:
            losses['long_range_distance'] = self.pretraining_tasks.long_range_distance_loss(node_embeddings, batch)
            total_loss += self.config.task_weights['long_range_distance'] * losses['long_range_distance']

        if "short_range_distance" in self.config.pretraining_tasks:
            if hasattr(batch, 'edge_index') and hasattr(batch, 'pos'):
                edge_index = getattr(
                    batch, 'physical_edge_index', batch.edge_index
                )
                pos = batch.pos
                distances = torch.norm(pos[edge_index[1]] - pos[edge_index[0]], dim=1)
                if is_training:
                    mask = (torch.rand(edge_index.size(1)) < 0.15).to(distances.device)
                else:
                    mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=distances.device)
                template_nodes = getattr(batch, 'template_node_mask', None)
                if template_nodes is not None:
                    mask = mask & ~(
                        template_nodes[edge_index[0]]
                        | template_nodes[edge_index[1]]
                    )
                dist_loss = self.pretraining_tasks.short_range_distance_loss(
                    node_embeddings, edge_index, distances, mask
                )
                losses['short_range_distance'] = dist_loss
                total_loss += self.config.task_weights['short_range_distance'] * dist_loss

        if "mlm" in self.config.pretraining_tasks:
            losses['mlm'] = self.pretraining_tasks.mlm_loss(node_embeddings, batch)
            total_loss += self.config.task_weights['mlm'] * losses['mlm']

        if "coordinate_denoising" in self.config.pretraining_tasks:
            losses['coordinate_denoising'] = self.pretraining_tasks.coordinate_denoising_loss(node_embeddings, batch)
            w = self.config.task_weights.get('coordinate_denoising', 1.0)
            total_loss += w * losses['coordinate_denoising']

        return losses, total_loss

    def _compute_per_type_losses(self, batch, graph_embeddings, node_embeddings, is_training: bool = True):
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
                    source_edge_index = getattr(
                        batch, 'physical_edge_index', batch.edge_index
                    )
                    edge_mask = node_mask[source_edge_index[0]] & node_mask[source_edge_index[1]]
                    if edge_mask.sum() > 0:
                        masked_edge_index = source_edge_index[:, edge_mask]
                        masked_pos = batch.pos[node_mask]
                        node_mapping = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
                        node_mapping[node_mask] = torch.arange(node_mask.sum(), device=device)
                        remapped_edge_index = node_mapping[masked_edge_index]
                        distances = torch.norm(
                            masked_pos[remapped_edge_index[1]] - masked_pos[remapped_edge_index[0]], dim=1
                        )
                        if is_training:
                            random_mask = (torch.rand(remapped_edge_index.size(1)) < 0.15).to(distances.device)
                        else:
                            random_mask = torch.ones(remapped_edge_index.size(1), dtype=torch.bool, device=distances.device)
                        template_nodes = getattr(batch, 'template_node_mask', None)
                        if template_nodes is not None:
                            global_edges = masked_edge_index
                            random_mask = random_mask & ~(
                                template_nodes[global_edges[0]]
                                | template_nodes[global_edges[1]]
                            )
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
                    _, compact_graph_ids = torch.unique(
                        batch.batch[node_mask], sorted=True, return_inverse=True
                    )
                    masked_batch.batch = compact_graph_ids
                    masked_batch.num_graphs = int(
                        compact_graph_ids.max().item()
                    ) + 1
                    if getattr(batch, 'template_node_mask', None) is not None:
                        masked_batch.template_node_mask = batch.template_node_mask[node_mask]
                    if getattr(batch, 'geometry_valid_mask', None) is not None:
                        masked_batch.geometry_valid_mask = batch.geometry_valid_mask[node_mask]
                    masked_node_emb = node_embeddings[node_mask]
                    long_range_loss = self.pretraining_tasks.long_range_distance_loss(masked_node_emb, masked_batch)
                    dtype_losses['long_range_distance'] = long_range_loss
                    dtype_total += self.config.task_weights['long_range_distance'] * long_range_loss

            # Per-type MLM tracking is enabled for ALL dataset types — including
            # protein/pdb. The previous protein skip was a leftover from older
            # protein-loss experiments. When `protein_block_mlm` is active and
            # the dtype is a protein, the loss is logged under the `block_mlm`
            # task key so W&B shows e.g. `val_pdb_block_mlm_loss` or
            # `val_rna_block_mlm_loss` distinctly from atom-level runs.
            if "mlm" in self.config.pretraining_tasks:
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
                        if (
                            hasattr(batch, 'domain_local_slot_ids')
                            and batch.domain_local_slot_ids is not None
                        ):
                            masked_batch.domain_local_slot_ids = (
                                batch.domain_local_slot_ids[node_mask]
                            )
                        masked_node_emb = node_embeddings[node_mask]
                        try:
                            mlm_loss = self.pretraining_tasks.mlm_loss(masked_node_emb, masked_batch)
                            normalized_dtype = dtype.lower()
                            protein_block_active = (
                                normalized_dtype in ('protein', 'pdb')
                                and getattr(self.config, 'protein_block_mlm', False)
                            )
                            rna_block_active = (
                                normalized_dtype == 'rna'
                                and getattr(self.config, 'rna_block_mlm', False)
                            )
                            key = (
                                'block_mlm'
                                if protein_block_active or rna_block_active
                                else 'mlm'
                            )
                            dtype_losses[key] = mlm_loss
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

    def _audit_representation_gradient_norms(
        self,
        losses: dict,
        node_embeddings: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Measure each weighted task's gradient at the shared node states.

        This is a diagnostic, not a loss term. Measuring at the shared ESA
        output avoids extra backward passes through the FlashAttention kernel
        while still showing the signal each task sends into the backbone.
        """
        settings = getattr(
            self.config, "geometry_neutral_virtual_slots", None
        ) or {}
        max_batches = int(settings.get("gradient_audit_batches", 0))
        if max_batches <= 0 or batch_idx >= max_batches:
            return

        weighted = {}
        for name, loss in losses.items():
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                continue
            if name in self.config.task_weights:
                weight = self.config.task_weights[name]
            elif name == "coordinate_denoising":
                weight = self.config.task_weights.get(name, 1.0)
            else:
                continue
            weighted[name] = float(weight) * loss

        norms = {}
        for name, weighted_loss in weighted.items():
            gradient = torch.autograd.grad(
                weighted_loss,
                node_embeddings,
                retain_graph=True,
                allow_unused=True,
            )[0]
            norm = 0.0 if gradient is None else float(
                torch.linalg.vector_norm(gradient.detach().float()).item()
            )
            norms[name] = norm
            self.log(
                f"gradient_audit_{name}_representation_l2",
                norm,
                on_step=True,
                on_epoch=False,
                logger=True,
            )

        if norms and int(getattr(self, "global_rank", 0)) == 0:
            total = sum(norms.values())
            details = " | ".join(
                f"{name}={value:.6f} ({value / total:.1%})"
                if total > 0.0 else f"{name}={value:.6f}"
                for name, value in norms.items()
            )
            print(
                "GRADIENT AUDIT (weighted task L2 at shared node states; "
                f"ratios are descriptive, not additive shares): {details}"
            )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        self._ensure_mlm_attributes(batch, batch_idx=batch_idx, is_validation=False)
        template_stats = self._ensure_masked_slot_template(batch)
        self._ensure_coord_denoising_attributes(batch)
        self._log_coconut_masking_metrics(batch, batch_idx)

        for name, value in template_stats.items():
            self.log(
                f"virtual_template_{name}",
                float(value),
                on_step=True,
                on_epoch=False,
                logger=True,
            )

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
        for name, value in getattr(
            batch, 'virtual_slot_graph_stats', {}
        ).items():
            self.log(
                f"virtual_slot_graph_{name}",
                float(value),
                on_step=True,
                on_epoch=False,
                logger=True,
            )
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)
        self._audit_representation_gradient_norms(
            losses, node_embeddings, batch_idx
        )

        for task_name, loss_value in losses.items():
            self.log(f"train_{task_name}_loss", loss_value, prog_bar=True, on_step=True, on_epoch=True, logger=True)
            if batch_idx % 50 == 0:
                print(f"  {task_name}: {loss_value:.4f}")

        if batch_idx % 50 == 0:
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
            print(f"Step {batch_idx} - {loss_str} | Total: {total_loss:.4f}")

        self.log("train_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        num_atoms = batch.num_nodes if hasattr(batch, 'num_nodes') else batch.x.size(0)
        self.log("batch_total_atoms", float(num_atoms), on_step=True, logger=True)
        if hasattr(batch, 'dataset_type') and isinstance(batch.dataset_type, (list, tuple)) and hasattr(batch, 'batch'):
            num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            atoms_per_graph = [(batch.batch == i).sum().item() for i in range(num_graphs)]
            from collections import Counter
            type_counts = Counter(batch.dataset_type)
            for dtype, count in type_counts.items():
                dtype_atoms = sum(atoms_per_graph[i] for i, dt in enumerate(batch.dataset_type) if dt == dtype)
                self.log(f"batch_{dtype}_atom_ratio", dtype_atoms / num_atoms, on_step=True, logger=True)

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

    @staticmethod
    def _exact_masked_block_accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        node_mask: torch.Tensor,
        batch: object,
    ) -> tuple[torch.Tensor, int]:
        """All-real-slot accuracy over masked residue/nucleotide blocks."""
        selected = torch.where(node_mask)[0]
        if selected.numel() == 0:
            return predictions.new_tensor(0.0, dtype=torch.float32), 0
        ownership = torch.stack(
            (batch.batch[selected].long(), batch.block_idx[selected].long()),
            dim=1,
        )
        unique_owners = torch.unique(ownership, dim=0, sorted=True)
        exact = []
        for owner in unique_owners:
            owner_nodes = selected[
                (ownership[:, 0] == owner[0])
                & (ownership[:, 1] == owner[1])
            ]
            exact.append(
                (predictions[owner_nodes] == targets[owner_nodes]).all()
            )
        values = torch.stack(exact).float()
        return values.mean(), int(values.numel())

    def _virtual_slot_context_audit(
        self,
        batch: object,
        batch_idx: int,
        correct_node_embeddings: torch.Tensor,
        correct_logits: torch.Tensor,
    ) -> dict:
        """Evaluate owner-context counterfactuals with identical weights/data.

        This is validation-only and intentionally changes only the post-ESA
        backbone readout relation. It answers two distinct questions:

        * nonzero logit delta: does owner routing reach the prediction head?
        * positive shuffled/removed CE increase: did training learn to use it?
        """
        settings = getattr(
            self.config, "geometry_neutral_virtual_slots", None
        ) or {}
        max_batches = int(settings.get("context_audit_batches", 0))
        if (
            max_batches <= 0
            or batch_idx >= max_batches
            or getattr(batch, "target_slot_mask", None) is None
        ):
            return {}

        target_slot_mask = batch.target_slot_mask.bool()
        target_real_mask = batch.target_real_slot_mask.bool()
        targets = batch.original_types.long()
        correct_logits = correct_logits.detach()
        correct_predictions = correct_logits.argmax(dim=-1)

        graph_types = self._graph_dtype_list(batch) or []
        domain_masks = {}
        for domain, accepted in (
            ("protein", {"pdb", "protein"}),
            ("rna", {"rna"}),
        ):
            graph_flags = torch.tensor(
                [str(value).lower() in accepted for value in graph_types],
                dtype=torch.bool,
                device=targets.device,
            )
            if graph_flags.numel():
                domain_masks[domain] = graph_flags[batch.batch.long()]

        audit = {}
        readout_view = SimpleNamespace(
            target_slot_mask=batch.target_slot_mask,
            slot_readout_edge_index=batch.slot_readout_edge_index,
        )
        for mode in ("shuffled", "removed"):
            alternate_nodes = PretrainingESAModel._inject_latent_slot_readouts(
                self,
                batch,
                readout_view,
                correct_node_embeddings,
                batch.domain_local_slot_ids,
                mode,
            )
            alternate_logits = self.pretraining_tasks.mlm_head(
                alternate_nodes
            ).detach()
            alternate_predictions = alternate_logits.argmax(dim=-1)

            all_support = int(target_slot_mask.sum().item())
            if all_support:
                audit[f"{mode}_all_slot_logit_delta"] = (
                    float(
                        (correct_logits[target_slot_mask]
                         - alternate_logits[target_slot_mask])
                        .abs()
                        .mean()
                        .item()
                    ),
                    all_support,
                )

            for domain, domain_mask in domain_masks.items():
                real_mask = target_real_mask & domain_mask
                real_support = int(real_mask.sum().item())
                if real_support == 0:
                    continue
                correct_ce = F.cross_entropy(
                    correct_logits[real_mask], targets[real_mask]
                )
                alternate_ce = F.cross_entropy(
                    alternate_logits[real_mask], targets[real_mask]
                )
                correct_accuracy = (
                    correct_predictions[real_mask] == targets[real_mask]
                ).float().mean()
                alternate_accuracy = (
                    alternate_predictions[real_mask] == targets[real_mask]
                ).float().mean()
                correct_exact, block_support = self._exact_masked_block_accuracy(
                    correct_predictions, targets, real_mask, batch
                )
                alternate_exact, _ = self._exact_masked_block_accuracy(
                    alternate_predictions, targets, real_mask, batch
                )

                audit[f"{domain}_{mode}_real_ce_increase"] = (
                    float((alternate_ce - correct_ce).item()),
                    real_support,
                )
                audit[f"{domain}_{mode}_real_accuracy_drop"] = (
                    float((correct_accuracy - alternate_accuracy).item()),
                    real_support,
                )
                audit[f"{domain}_{mode}_real_exact_block_drop"] = (
                    float((correct_exact - alternate_exact).item()),
                    block_support,
                )

        for name, (value, support) in audit.items():
            self.log(
                f"val_context_audit/{name}",
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=max(support, 1),
            )
        return audit

    def validation_step(self, batch, batch_idx):
        self._ensure_mlm_attributes(batch, batch_idx=batch_idx, is_validation=True)
        self._ensure_masked_slot_template(batch)
        self._ensure_coord_denoising_attributes(batch)
        graph_embeddings, node_embeddings = self.forward(batch)
        losses, total_loss = self._compute_pretraining_losses(
            batch, graph_embeddings, node_embeddings, is_training=False
        )

        for task_name, loss_value in losses.items():
            self.log(f"val_{task_name}_loss", loss_value, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_total_loss", total_loss, on_epoch=True, logger=True, sync_dist=True)

        if self.config.compute_per_type_losses:
            type_losses = self._compute_per_type_losses(
                batch, graph_embeddings, node_embeddings, is_training=False
            )
            for dtype, dtype_losses in type_losses.items():
                for task_name, loss_value in dtype_losses.items():
                    self.log(f"val_{dtype}_{task_name}_loss", loss_value, on_epoch=True, logger=True, sync_dist=True)

        outputs = {'loss': total_loss}
        if "mlm" in self.config.pretraining_tasks:
            # Domain metric callbacks consume these predictions immediately.
            # Reusing this forward avoids running the ESA backbone again in
            # each callback.
            mlm_logits = self.pretraining_tasks.mlm_head(node_embeddings)
            outputs['mlm_preds_atomic_num'] = (
                mlm_logits.argmax(dim=1).detach().cpu()
            )
            outputs['virtual_slot_context_audit'] = (
                self._virtual_slot_context_audit(
                    batch,
                    batch_idx,
                    correct_node_embeddings=node_embeddings,
                    correct_logits=mlm_logits,
                )
            )
        return outputs

    def test_step(self, batch, batch_idx):
        self._ensure_mlm_attributes(batch, batch_idx=batch_idx, is_validation=True)
        self._ensure_masked_slot_template(batch)
        self._ensure_coord_denoising_attributes(batch)
        graph_embeddings, node_embeddings = self.forward(batch)
        losses, total_loss = self._compute_pretraining_losses(
            batch, graph_embeddings, node_embeddings, is_training=False
        )

        for task_name, loss_value in losses.items():
            self.log(f"test_{task_name}_loss", loss_value, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_total_loss", total_loss, on_epoch=True, logger=True, sync_dist=True)

        if self.config.compute_per_type_losses:
            type_losses = self._compute_per_type_losses(
                batch, graph_embeddings, node_embeddings, is_training=False
            )
            for dtype, dtype_losses in type_losses.items():
                for task_name, loss_value in dtype_losses.items():
                    self.log(f"test_{dtype}_{task_name}_loss", loss_value, on_epoch=True, logger=True)

        outputs = {'loss': total_loss}
        if "mlm" in self.config.pretraining_tasks:
            outputs['mlm_preds_atomic_num'] = (
                self.pretraining_tasks.mlm_head(node_embeddings)
                .argmax(dim=1)
                .detach()
                .cpu()
            )
        return outputs

    def on_before_optimizer_step(self, optimizer):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm_before_clip", total_norm, on_step=True, logger=True)

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
