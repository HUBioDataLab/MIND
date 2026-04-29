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
# Sub-module imports
from core.pretraining_config import PretrainingConfig, create_pretraining_config
from core.molecular_encoder import UniversalMolecularEncoder, nearest_multiple_of_8
from core.pretraining_tasks import PretrainingTasks

# Backward-compatibility re-exports (other files can still import from here)
__all__ = [
    'PretrainingConfig',
    'create_pretraining_config',
    'UniversalMolecularEncoder',
    'nearest_multiple_of_8',
    'PretrainingTasks',
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

        # Cumulative masking counters for metabolite-only BRICS budgeting.
        self.register_buffer("metabolite_mlm_atoms_seen", torch.tensor(0.0), persistent=True)
        self.register_buffer("metabolite_mlm_atoms_masked", torch.tensor(0.0), persistent=True)
        self.register_buffer("metabolite_coord_atoms_seen", torch.tensor(0.0), persistent=True)
        self.register_buffer("metabolite_coord_atoms_masked", torch.tensor(0.0), persistent=True)

    @staticmethod
    def _normalize_metabolite_strategy(strategy: str) -> str:
        s = str(strategy).strip().lower()
        if s == "random":
            return "legacy_random"
        if s == "brics":
            return "brics_fragment"
        return s

    @staticmethod
    def _is_metabolite_type(dataset_type_value: str) -> bool:
        s = str(dataset_type_value).strip().lower()
        return s in {"coconut", "metabolite"}

    def _per_graph_dataset_types(self, batch, num_graphs: int) -> List[str]:
        if not hasattr(batch, "dataset_type"):
            return [""] * num_graphs
        dtype = batch.dataset_type
        if isinstance(dtype, (list, tuple)):
            if len(dtype) == num_graphs:
                return [str(x) for x in dtype]
            if len(dtype) == 1:
                return [str(dtype[0])] * num_graphs
        if torch.is_tensor(dtype):
            if dtype.dim() == 0:
                return [str(dtype.item())] * num_graphs
            if dtype.numel() == num_graphs:
                return [str(x.item()) for x in dtype]
        return [str(dtype)] * num_graphs

    def _select_fragment_mask_indices(
        self,
        fragment_groups: List[torch.Tensor],
        k_target: int,
        device: torch.device,
    ) -> torch.Tensor:
        n = int(sum(int(g.numel()) for g in fragment_groups))
        if n == 0 or k_target <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)

        fragment_groups = [g for g in fragment_groups if int(g.numel()) > 0]
        if not fragment_groups:
            return torch.empty((0,), dtype=torch.long, device=device)

        target = int(k_target)
        selected_groups: List[torch.Tensor] = []
        selected_count = 0
        remaining = sorted(fragment_groups, key=lambda g: int(g.numel()), reverse=True)

        # Greedy closest-to-remaining-budget selection.
        while remaining:
            rem_budget = max(0, target - selected_count)
            best_i = 0
            best_score = None
            for i, grp in enumerate(remaining):
                size = int(grp.numel())
                score = abs(size - rem_budget)
                if best_score is None or score < best_score:
                    best_score = score
                    best_i = i
            picked = remaining.pop(best_i)
            selected_groups.append(picked)
            selected_count += int(picked.numel())
            if selected_count >= target:
                break

        selected = (
            torch.cat(selected_groups, dim=0)
            if selected_groups
            else torch.empty((0,), dtype=torch.long, device=device)
        )
        allow_partial = bool(getattr(self.config, "metabolite_allow_partial_fragment_masking", True))
        if int(selected.numel()) > target and allow_partial:
            # Keep full fragments first; trim only the last fragment if required.
            if len(selected_groups) == 1:
                grp = selected_groups[0]
                perm = torch.randperm(int(grp.numel()), device=device)[:target]
                return grp[perm]
            keep = selected_groups[:-1]
            kept = torch.cat(keep, dim=0) if keep else torch.empty((0,), dtype=torch.long, device=device)
            deficit = max(0, target - int(kept.numel()))
            if deficit == 0:
                return kept
            last = selected_groups[-1]
            perm = torch.randperm(int(last.numel()), device=device)[:deficit]
            return torch.cat([kept, last[perm]], dim=0)

        return selected

    def _extract_precomputed_fragment_groups_for_graph(
        self,
        batch,
        graph_indices: torch.Tensor,
    ) -> Optional[List[torch.Tensor]]:
        if not hasattr(batch, "brics_frag_atom_index") or not hasattr(batch, "brics_frag_ptr"):
            return None
        atom_index = batch.brics_frag_atom_index
        ptr = batch.brics_frag_ptr
        if atom_index is None or ptr is None:
            return None
        if ptr.numel() < 2:
            return None
        graph_set = set(int(i) for i in graph_indices.tolist())
        groups = []
        for i in range(int(ptr.numel()) - 1):
            s = int(ptr[i].item())
            e = int(ptr[i + 1].item())
            if e <= s:
                continue
            frag = atom_index[s:e]
            keep = [int(idx.item()) for idx in frag if int(idx.item()) in graph_set]
            if keep:
                groups.append(torch.tensor(keep, dtype=torch.long, device=graph_indices.device))
        if not groups:
            return None
        return groups

    def _select_metabolite_mask_indices_for_task(
        self,
        batch,
        graph_indices: torch.Tensor,
        task_name: str,
        default_ratio: float,
    ) -> torch.Tensor:
        if int(graph_indices.numel()) == 0:
            return graph_indices[:0]

        strategy = self._normalize_metabolite_strategy(
            getattr(self.config, "metabolite_masking_strategy", "legacy_random")
        )
        if strategy != "brics_fragment":
            n = int(graph_indices.numel())
            k = max(1, int(n * float(default_ratio)))
            perm = torch.randperm(n, device=graph_indices.device)[:k]
            return graph_indices[perm]

        brics_source = str(getattr(self.config, "metabolite_brics_source", "precomputed")).lower()
        fragment_groups = None
        if brics_source == "precomputed":
            fragment_groups = self._extract_precomputed_fragment_groups_for_graph(batch, graph_indices)
        # If BRICS cannot be used, fallback to legacy random selection.
        if fragment_groups is None:
            n = int(graph_indices.numel())
            k = max(1, int(n * float(default_ratio)))
            perm = torch.randperm(n, device=graph_indices.device)[:k]
            return graph_indices[perm]

        # Keep eval behavior stable without mutating cumulative training budgets.
        if not self.training:
            n = int(graph_indices.numel())
            k = max(1, int(n * float(default_ratio)))
            return self._select_fragment_mask_indices(
                fragment_groups=fragment_groups,
                k_target=k,
                device=graph_indices.device,
            )

        target_ratio = float(getattr(self.config, "metabolite_mask_target_ratio", 0.15))
        max_graph_ratio = float(getattr(self.config, "metabolite_max_mask_ratio_per_graph", 0.5))
        cap = max(1, int(graph_indices.numel() * max_graph_ratio))

        if task_name == "mlm":
            seen = float(self.metabolite_mlm_atoms_seen.item())
            masked = float(self.metabolite_mlm_atoms_masked.item())
        else:
            seen = float(self.metabolite_coord_atoms_seen.item())
            masked = float(self.metabolite_coord_atoms_masked.item())

        n = int(graph_indices.numel())
        desired_cumulative = target_ratio * (seen + n)
        needed = max(0, int(round(desired_cumulative - masked)))
        k_target = min(cap, n, needed)
        if k_target == 0:
            return graph_indices[:0]

        # If every BRICS fragment is larger than allowed selection target, fallback to legacy random.
        if fragment_groups and min(int(g.numel()) for g in fragment_groups) > k_target:
            perm = torch.randperm(n, device=graph_indices.device)[:k_target]
            return graph_indices[perm]

        return self._select_fragment_mask_indices(fragment_groups, k_target, graph_indices.device)

    def _ensure_shared_metabolite_fragment_mask(self, batch):
        """Create one shared metabolite BRICS mask reused by MLM and coordinate denoising."""
        if hasattr(batch, "shared_metabolite_mask") and batch.shared_metabolite_mask is not None:
            return
        if not hasattr(batch, "batch") or batch.batch is None:
            return
        if not hasattr(batch, "z") or batch.z is None:
            return
        if "mlm" not in self.config.pretraining_tasks and "coordinate_denoising" not in self.config.pretraining_tasks:
            return
        strategy = self._normalize_metabolite_strategy(
            getattr(self.config, "metabolite_masking_strategy", "legacy_random")
        )
        if strategy != "brics_fragment":
            return

        graph_idx = batch.batch
        num_nodes = int(batch.z.size(0))
        shared_mask = torch.zeros(num_nodes, dtype=torch.bool, device=batch.z.device)
        unique_graphs = graph_idx.unique().tolist()
        dtype_by_graph = self._per_graph_dataset_types(batch, len(unique_graphs))
        graph_to_dtype = {int(g): dtype_by_graph[i] for i, g in enumerate(unique_graphs)}
        default_ratio = float(getattr(self.config, "mlm_mask_ratio", 0.15))

        batch_metabolite_seen = 0
        batch_metabolite_masked = 0
        for g in unique_graphs:
            if not self._is_metabolite_type(graph_to_dtype.get(int(g), "")):
                continue
            node_mask = (graph_idx == g)
            indices = torch.where(node_mask)[0]
            n = int(indices.size(0))
            if n == 0:
                continue
            selected = self._select_metabolite_mask_indices_for_task(
                batch=batch,
                graph_indices=indices,
                task_name="mlm",
                default_ratio=default_ratio,
            )
            if selected.numel() > 0:
                shared_mask[selected] = True
            batch_metabolite_seen += n
            batch_metabolite_masked += int(selected.numel())

        batch.shared_metabolite_mask = shared_mask

        if self.training and batch_metabolite_seen > 0:
            # Keep both task-specific counters aligned to the same selected atoms.
            self.metabolite_mlm_atoms_seen += float(batch_metabolite_seen)
            self.metabolite_mlm_atoms_masked += float(batch_metabolite_masked)
            self.metabolite_coord_atoms_seen += float(batch_metabolite_seen)
            self.metabolite_coord_atoms_masked += float(batch_metabolite_masked)

            batch_ratio = batch_metabolite_masked / max(1, batch_metabolite_seen)
            cumulative_ratio = float(self.metabolite_mlm_atoms_masked.item()) / max(
                1.0, float(self.metabolite_mlm_atoms_seen.item())
            )
            self.log("train_mlm_metabolite_mask_ratio_batch", batch_ratio, on_step=True, logger=True)
            self.log("train_mlm_metabolite_mask_ratio_cumulative", cumulative_ratio, on_step=True, logger=True)
            self.log("train_mlm_metabolite_masking_is_brics", 1.0, on_step=True, logger=True)
            self.log("train_coord_metabolite_mask_ratio_batch", batch_ratio, on_step=True, logger=True)
            self.log("train_coord_metabolite_mask_ratio_cumulative", cumulative_ratio, on_step=True, logger=True)
            self.log("train_coord_metabolite_masking_is_brics", 1.0, on_step=True, logger=True)

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
        Applies to all molecule types including proteins."""
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

        if hasattr(batch, 'batch') and batch.batch is not None:
            graph_idx = batch.batch
            unique_graphs = graph_idx.unique().tolist()
            dtype_by_graph = self._per_graph_dataset_types(batch, len(unique_graphs))
            graph_to_dtype = {int(g): dtype_by_graph[i] for i, g in enumerate(unique_graphs)}

            batch_metabolite_seen = 0
            batch_metabolite_masked = 0
            strategy_name = self._normalize_metabolite_strategy(
                getattr(self.config, "metabolite_masking_strategy", "legacy_random")
            )

            for g in unique_graphs:
                node_mask = (graph_idx == g)
                indices = torch.where(node_mask)[0]
                n = indices.size(0)
                if n == 0:
                    continue

                if self._is_metabolite_type(graph_to_dtype.get(int(g), "")):
                    if hasattr(batch, "shared_metabolite_mask") and batch.shared_metabolite_mask is not None:
                        selected = indices[batch.shared_metabolite_mask[indices]]
                    else:
                        selected = self._select_metabolite_mask_indices_for_task(
                            batch=batch,
                            graph_indices=indices,
                            task_name="mlm",
                            default_ratio=mask_ratio,
                        )
                    batch_metabolite_seen += int(n)
                    batch_metabolite_masked += int(selected.numel())
                else:
                    k = max(1, int(n * mask_ratio))
                    perm = torch.randperm(n, device=z.device)[:k]
                    selected = indices[perm]

                if selected.numel() > 0:
                    mlm_mask[selected] = True
                    masked_types[selected] = mask_token

            if self.training and batch_metabolite_seen > 0 and not (
                hasattr(batch, "shared_metabolite_mask") and batch.shared_metabolite_mask is not None
            ):
                self.metabolite_mlm_atoms_seen += float(batch_metabolite_seen)
                self.metabolite_mlm_atoms_masked += float(batch_metabolite_masked)
                batch_ratio = batch_metabolite_masked / max(1, batch_metabolite_seen)
                cumulative_ratio = float(self.metabolite_mlm_atoms_masked.item()) / max(
                    1.0, float(self.metabolite_mlm_atoms_seen.item())
                )
                self.log("train_mlm_metabolite_mask_ratio_batch", batch_ratio, on_step=True, logger=True)
                self.log("train_mlm_metabolite_mask_ratio_cumulative", cumulative_ratio, on_step=True, logger=True)
                self.log("train_mlm_metabolite_masking_is_brics", 1.0 if strategy_name == "brics_fragment" else 0.0, on_step=True, logger=True)
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
            graph_idx = batch.batch
            unique_graphs = graph_idx.unique().tolist()
            dtype_by_graph = self._per_graph_dataset_types(batch, len(unique_graphs))
            graph_to_dtype = {int(g): dtype_by_graph[i] for i, g in enumerate(unique_graphs)}
            strategy_name = self._normalize_metabolite_strategy(
                getattr(self.config, "metabolite_masking_strategy", "legacy_random")
            )

            batch_metabolite_seen = 0
            batch_metabolite_masked = 0
            for g in unique_graphs:
                node_mask = (graph_idx == g)
                indices = torch.where(node_mask)[0]
                n = indices.size(0)
                if n == 0:
                    continue
                if self._is_metabolite_type(graph_to_dtype.get(int(g), "")):
                    if hasattr(batch, "shared_metabolite_mask") and batch.shared_metabolite_mask is not None:
                        selected = indices[batch.shared_metabolite_mask[indices]]
                    else:
                        selected = self._select_metabolite_mask_indices_for_task(
                            batch=batch,
                            graph_indices=indices,
                            task_name="coord",
                            default_ratio=mask_ratio,
                        )
                    batch_metabolite_seen += int(n)
                    batch_metabolite_masked += int(selected.numel())
                else:
                    k = max(1, int(n * mask_ratio))
                    perm = torch.randperm(n, device=batch.pos.device)[:k]
                    selected = indices[perm]
                if selected.numel() > 0:
                    coord_mask[selected] = True

            if self.training and batch_metabolite_seen > 0 and not (
                hasattr(batch, "shared_metabolite_mask") and batch.shared_metabolite_mask is not None
            ):
                self.metabolite_coord_atoms_seen += float(batch_metabolite_seen)
                self.metabolite_coord_atoms_masked += float(batch_metabolite_masked)
                batch_ratio = batch_metabolite_masked / max(1, batch_metabolite_seen)
                cumulative_ratio = float(self.metabolite_coord_atoms_masked.item()) / max(
                    1.0, float(self.metabolite_coord_atoms_seen.item())
                )
                self.log("train_coord_metabolite_mask_ratio_batch", batch_ratio, on_step=True, logger=True)
                self.log("train_coord_metabolite_mask_ratio_cumulative", cumulative_ratio, on_step=True, logger=True)
                self.log("train_coord_metabolite_masking_is_brics", 1.0 if strategy_name == "brics_fragment" else 0.0, on_step=True, logger=True)
        else:
            k = max(1, int(num_nodes * mask_ratio))
            coord_mask[torch.randperm(num_nodes, device=batch.pos.device)[:k]] = True
        batch.coord_mask = coord_mask

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
                edge_index = batch.edge_index
                pos = batch.pos
                distances = torch.norm(pos[edge_index[1]] - pos[edge_index[0]], dim=1)
                if is_training:
                    mask = (torch.rand(edge_index.size(1)) < 0.15).to(distances.device)
                else:
                    mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=distances.device)
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
                        if is_training:
                            random_mask = (torch.rand(remapped_edge_index.size(1)) < 0.15).to(distances.device)
                        else:
                            random_mask = torch.ones(remapped_edge_index.size(1), dtype=torch.bool, device=distances.device)
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
        self._ensure_shared_metabolite_fragment_mask(batch)
        self._ensure_mlm_attributes(batch)
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

    def validation_step(self, batch, batch_idx):
        self._ensure_shared_metabolite_fragment_mask(batch)
        self._ensure_mlm_attributes(batch)
        self._ensure_coord_denoising_attributes(batch)
        graph_embeddings, node_embeddings = self.forward(batch)
        losses, total_loss = self._compute_pretraining_losses(
            batch, graph_embeddings, node_embeddings, is_training=False
        )

        for task_name, loss_value in losses.items():
            self.log(f"val_{task_name}_loss", loss_value, on_epoch=True, logger=True)
        self.log("val_total_loss", total_loss, on_epoch=True, logger=True)

        if self.config.compute_per_type_losses:
            type_losses = self._compute_per_type_losses(
                batch, graph_embeddings, node_embeddings, is_training=False
            )
            for dtype, dtype_losses in type_losses.items():
                for task_name, loss_value in dtype_losses.items():
                    self.log(f"val_{dtype}_{task_name}_loss", loss_value, on_epoch=True, logger=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        self._ensure_shared_metabolite_fragment_mask(batch)
        self._ensure_mlm_attributes(batch)
        self._ensure_coord_denoising_attributes(batch)
        graph_embeddings, node_embeddings = self.forward(batch)
        losses, total_loss = self._compute_pretraining_losses(
            batch, graph_embeddings, node_embeddings, is_training=False
        )

        for task_name, loss_value in losses.items():
            self.log(f"test_{task_name}_loss", loss_value, on_epoch=True, logger=True)
        self.log("test_total_loss", total_loss, on_epoch=True, logger=True)

        if self.config.compute_per_type_losses:
            type_losses = self._compute_per_type_losses(
                batch, graph_embeddings, node_embeddings, is_training=False
            )
            for dtype, dtype_losses in type_losses.items():
                for task_name, loss_value in dtype_losses.items():
                    self.log(f"test_{dtype}_{task_name}_loss", loss_value, on_epoch=True, logger=True)

        return total_loss

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
