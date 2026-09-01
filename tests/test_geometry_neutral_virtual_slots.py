import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from core.masked_slot_mapping import attach_runtime_template_metadata
from core.pretraining_config import create_pretraining_config
from core.pretraining_model import PretrainingESAModel
from core.pretraining_tasks import PretrainingTasks
from core.virtual_slot_graph import (
    build_geometry_neutral_graph_view,
    build_slot_readout_edges_for_owner_mode,
)
from data_loading.atom_position_vocab import (
    PROTEIN_BACKBONE_ATOMS,
    RNA_BACKBONE_ATOMS,
    RNA_BASE_ATOMS_BY_NUCLEOTIDE,
    VIRTUAL_ATOMIC_NUMBER,
)


def _settings():
    return {
        "enabled": True,
        "protein": True,
        "rna": True,
        "masked_real_distance_weight": 1.0,
    }


def _view_config():
    return SimpleNamespace(
        geometry_neutral_virtual_slots=_settings(),
        cutoff_distance=2.1,
        max_spatial_neighbors=48,
    )


def _two_alanine_batch():
    block_codes = list(PROTEIN_BACKBONE_ATOMS) + ["CB"] + [
        f"V{index}" for index in range(2, 11)
    ]
    pos_one = torch.tensor(
        [
            [-1.0, 0.0, 0.0],  # N
            [0.0, 0.0, 0.0],   # CA
            [1.0, 0.0, 0.0],   # C
            [1.8, 0.5, 0.0],   # O
            [0.0, 1.5, 0.0],   # CB
        ]
        + [[0.0, 1.5, 0.0]] * 9,
        dtype=torch.float32,
    )
    pos = torch.cat((pos_one, pos_one + torch.tensor([10.0, 0.0, 0.0])))
    one_types = torch.tensor(
        [7, 6, 6, 8, 6] + [VIRTUAL_ATOMIC_NUMBER] * 9,
        dtype=torch.long,
    )
    original_types = torch.cat((one_types, one_types))
    mlm_mask = torch.zeros(28, dtype=torch.bool)
    mlm_mask[14:] = True
    z = original_types.clone()
    z[mlm_mask] = 0
    batch = SimpleNamespace(
        z=z,
        x=z.clone(),
        original_types=original_types,
        masked_types=z.clone(),
        pos=pos,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        batch=torch.zeros(28, dtype=torch.long),
        block_idx=torch.tensor([0] * 14 + [1] * 14, dtype=torch.long),
        mlm_mask=mlm_mask,
        pos_code=block_codes + block_codes,
        dataset_type="protein",
        num_graphs=1,
        num_nodes=28,
    )
    return attach_runtime_template_metadata(batch)


def _two_cytidine_batch():
    real_base_codes = list(RNA_BASE_ATOMS_BY_NUCLEOTIDE["C"])
    block_codes = list(RNA_BACKBONE_ATOMS) + real_base_codes + [
        f"V{index}" for index in range(len(real_base_codes) + 1, 12)
    ]
    one_types = torch.tensor(
        [15 if code == "P" else 8 if code.startswith("O") else 7
         if code.startswith("N") else 6 for code in block_codes],
        dtype=torch.long,
    )
    one_types[-3:] = VIRTUAL_ATOMIC_NUMBER
    original_types = torch.cat((one_types, one_types))
    z = original_types.clone()
    mlm_mask = torch.zeros(46, dtype=torch.bool)
    mlm_mask[23:] = True
    z[mlm_mask] = 0
    one_pos = torch.arange(69, dtype=torch.float32).reshape(23, 3) / 10
    pos = torch.cat((one_pos, one_pos + torch.tensor([20.0, 0.0, 0.0])))
    batch = SimpleNamespace(
        z=z,
        x=z.clone(),
        original_types=original_types,
        masked_types=z.clone(),
        pos=pos,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        batch=torch.zeros(46, dtype=torch.long),
        block_idx=torch.tensor([0] * 23 + [1] * 23, dtype=torch.long),
        mlm_mask=mlm_mask,
        pos_code=block_codes + block_codes,
        dataset_type="rna",
        num_graphs=1,
        num_nodes=46,
    )
    return attach_runtime_template_metadata(batch)


class GeometryNeutralVirtualSlotTest(unittest.TestCase):
    def test_readout_links_keep_two_masked_residues_separate(self):
        batch = _two_alanine_batch()
        batch.mlm_mask[:] = True
        batch.z[:] = 0
        batch.x[:] = 0
        view = build_geometry_neutral_graph_view(batch, _view_config())

        owner_pairs = set(map(tuple, view.slot_readout_edge_index.t().tolist()))
        first_queries = set(range(4, 14))
        first_backbone = set(range(4))
        second_queries = set(range(18, 28))
        second_backbone = set(range(14, 18))

        self.assertEqual(
            {
                pair for pair in owner_pairs
                if pair[0] in first_queries
            },
            {
                (query, anchor)
                for query in first_queries
                for anchor in first_backbone
            },
        )
        self.assertEqual(
            {
                pair for pair in owner_pairs
                if pair[0] in second_queries
            },
            {
                (query, anchor)
                for query in second_queries
                for anchor in second_backbone
            },
        )
        self.assertFalse(any(
            (source in first_queries and target in second_backbone)
            or (source in second_queries and target in first_backbone)
            for source, target in owner_pairs
        ))

    def test_context_audit_shuffles_owners_only_within_the_same_graph(self):
        batch = _two_alanine_batch()
        batch.mlm_mask[:] = True
        batch.z[:] = 0
        batch.x[:] = 0
        view = build_geometry_neutral_graph_view(batch, _view_config())

        shuffled_edges = build_slot_readout_edges_for_owner_mode(
            batch, view, "shuffled"
        )
        pairs = set(map(tuple, shuffled_edges.t().tolist()))

        first_queries = set(range(4, 14))
        first_backbone = set(range(4))
        second_queries = set(range(18, 28))
        second_backbone = set(range(14, 18))
        self.assertTrue(all(
            (source in first_queries and target in second_backbone)
            or (source in second_queries and target in first_backbone)
            for source, target in pairs
        ))
        self.assertFalse(any(
            (source in first_queries and target in first_backbone)
            or (source in second_queries and target in second_backbone)
            for source, target in pairs
        ))

        removed = build_slot_readout_edges_for_owner_mode(
            batch, view, "removed"
        )
        self.assertEqual(removed.numel(), 0)

    def test_rna_slots_read_only_their_nucleotide_backbone(self):
        batch = _two_cytidine_batch()
        view = build_geometry_neutral_graph_view(batch, _view_config())

        owner_pairs = set(map(tuple, view.slot_readout_edge_index.t().tolist()))
        queries = set(range(35, 46))
        own_backbone = set(range(23, 35))
        other_backbone = set(range(12))

        self.assertEqual(
            owner_pairs,
            {
                (query, anchor)
                for query in queries
                for anchor in own_backbone
            },
        )
        self.assertFalse(any(
            target in other_backbone for _, target in owner_pairs
        ))

    def test_block_masking_skips_quarantined_runtime_metadata(self):
        batch = _two_alanine_batch()
        batch.z = batch.original_types.clone()
        batch.x = batch.z.clone()
        batch.mlm_mask = None
        batch.masked_types = None
        batch.template_reason_bits[14:] = 1

        harness = SimpleNamespace(
            config=SimpleNamespace(
                pretraining_tasks=["mlm"],
                protein_block_mlm=True,
                protein_block_mlm_ratio=1.0,
                rna_block_mlm=False,
                mlm_mask_ratio=0.15,
                protein_block_mlm_exclude_legacy_oxt=False,
                masked_slot_template={"enabled": False},
                geometry_neutral_virtual_slots=_settings(),
            ),
            current_epoch=0,
        )
        harness._graph_dtype_list = PretrainingESAModel._graph_dtype_list
        PretrainingESAModel._ensure_mlm_attributes(
            harness, batch, batch_idx=0, is_validation=True
        )

        self.assertTrue(bool(batch.mlm_mask[:14].all()))
        self.assertFalse(bool(batch.mlm_mask[14:].any()))

    def test_visible_virtuals_and_masked_slots_leave_encoder_graph(self):
        batch = _two_alanine_batch()
        view = build_geometry_neutral_graph_view(batch, _view_config())

        self.assertTrue(bool(view.active_node_mask[:5].all()))
        self.assertFalse(bool(view.active_node_mask[5:14].any()))
        self.assertTrue(bool(view.active_node_mask[14:18].all()))
        self.assertFalse(bool(view.active_node_mask[18:28].any()))
        self.assertEqual(int(view.active_node_mask.sum()), 5 + 4)
        self.assertTrue(bool(view.geometry_valid_mask[:5].all()))
        self.assertTrue(bool(view.geometry_valid_mask[14:18].all()))
        self.assertFalse(bool(view.geometry_valid_mask[18:28].any()))
        self.assertEqual(
            view.target_real_slot_mask.nonzero().flatten().tolist(), [18]
        )

    def test_physical_edges_have_only_valid_endpoints_and_are_restored(self):
        batch = _two_alanine_batch()
        view = build_geometry_neutral_graph_view(batch, _view_config())
        edges = view.physical_edge_index

        self.assertGreater(edges.numel(), 0)
        self.assertTrue(bool(view.geometry_valid_mask[edges].all()))
        pairs = set(map(tuple, edges.t().tolist()))
        self.assertIn((1, 4), pairs)
        self.assertIn((4, 1), pairs)
        self.assertFalse(
            any(source >= 18 or target >= 18 for source, target in pairs)
        )

    def test_stale_spatial_edges_are_rebuilt_but_ca_structural_edges_survive(self):
        batch = _two_alanine_batch()
        batch.edge_index = torch.tensor(
            [
                [1, 15, 0, 15, 4, 15],
                [15, 1, 15, 0, 15, 4],
            ],
            dtype=torch.long,
        )
        view = build_geometry_neutral_graph_view(batch, _view_config())
        pairs = set(map(tuple, view.physical_edge_index.t().tolist()))

        # CA<->CA between adjacent blocks is the protein sequence tier and is
        # retained even though the synthetic blocks are 10 Å apart.
        self.assertIn((1, 15), pairs)
        self.assertIn((15, 1), pairs)
        # Long cached non-CA edges are stale spatial artifacts, not structural
        # tiers, and cannot survive the real-only radius rebuild.
        self.assertNotIn((0, 15), pairs)
        self.assertNotIn((15, 0), pairs)
        self.assertNotIn((4, 15), pairs)
        self.assertNotIn((15, 4), pairs)

    def test_placeholder_coordinates_cannot_change_encoder_view(self):
        baseline_batch = _two_alanine_batch()
        changed_batch = copy.deepcopy(baseline_batch)
        changed_batch.pos[5:14] += 1000.0
        changed_batch.pos[18:28] -= 777.0

        baseline = build_geometry_neutral_graph_view(baseline_batch, _view_config())
        changed = build_geometry_neutral_graph_view(changed_batch, _view_config())

        self.assertTrue(torch.equal(
            baseline.active_node_mask, changed.active_node_mask
        ))
        self.assertTrue(torch.equal(
            baseline.compact_edge_index, changed.compact_edge_index
        ))
        self.assertTrue(torch.equal(
            baseline.compact_attention_edge_index,
            changed.compact_attention_edge_index,
        ))
        self.assertTrue(torch.equal(baseline.compact_pos, changed.compact_pos))

    def test_model_forward_scatter_is_placeholder_invariant(self):
        class RecordingEncoder(nn.Module):
            def forward(self, x, pos, batch):
                # Deterministic stand-in that depends on every encoder input
                # relevant to the geometry-neutral integration.
                coordination = torch.bincount(
                    batch.edge_index[0], minlength=x.numel()
                ).to(pos.dtype)
                return torch.stack(
                    (x.to(pos.dtype), pos[:, 0], pos[:, 1], coordination), dim=1
                )

        class IdentityBackbone(nn.Module):
            def forward(
                self, dense_x, edge_index, batch_mapping, num_max_items, pos=None
            ):
                return dense_x.mean(dim=1), dense_x

        dummy = nn.Module()
        dummy.config = SimpleNamespace(
            geometry_neutral_virtual_slots=_settings(),
            cutoff_distance=2.1,
            max_spatial_neighbors=48,
            apply_attention_on="node",
            set_max_items=0,
        )
        dummy.encoder = RecordingEncoder()
        dummy.encoder.domain_local_slot_embedding = nn.Embedding(38, 4)
        dummy.node_mlp = nn.Identity()
        dummy.mlp_norm = nn.Identity()
        dummy.esa_backbone = IdentityBackbone()

        baseline_batch = _two_alanine_batch()
        changed_batch = copy.deepcopy(baseline_batch)
        changed_batch.pos[5:14] += 1000.0
        changed_batch.pos[18:28] -= 777.0
        baseline_graph, baseline_nodes = PretrainingESAModel.forward(
            dummy, baseline_batch
        )
        changed_graph, changed_nodes = PretrainingESAModel.forward(
            dummy, changed_batch
        )

        self.assertTrue(torch.equal(baseline_graph, changed_graph))
        self.assertTrue(torch.equal(baseline_nodes, changed_nodes))
        self.assertFalse(bool(baseline_nodes[5:14].any()))
        self.assertTrue(bool(baseline_nodes[18:28].any()))

    def test_model_routes_physical_and_attention_edges_separately(self):
        class RecordingEncoder(nn.Module):
            def forward(self, x, pos, batch):
                self.edge_index = batch.edge_index.clone()
                return torch.zeros((x.numel(), 4), dtype=pos.dtype)

        class RecordingBackbone(nn.Module):
            def forward(
                self, dense_x, edge_index, batch_mapping, num_max_items, pos=None
            ):
                self.edge_index = edge_index.clone()
                return dense_x.mean(dim=1), dense_x

        batch = _two_alanine_batch()
        expected = build_geometry_neutral_graph_view(batch, _view_config())
        dummy = nn.Module()
        dummy.config = SimpleNamespace(
            geometry_neutral_virtual_slots=_settings(),
            cutoff_distance=2.1,
            max_spatial_neighbors=48,
            apply_attention_on="node",
            set_max_items=0,
        )
        dummy.encoder = RecordingEncoder()
        dummy.encoder.domain_local_slot_embedding = nn.Embedding(38, 4)
        dummy.node_mlp = nn.Identity()
        dummy.mlp_norm = nn.Identity()
        dummy.esa_backbone = RecordingBackbone()

        PretrainingESAModel.forward(dummy, batch)

        self.assertTrue(torch.equal(
            dummy.encoder.edge_index, expected.compact_edge_index
        ))
        self.assertTrue(torch.equal(
            dummy.esa_backbone.edge_index,
            expected.compact_attention_edge_index,
        ))
        self.assertEqual(
            dummy.esa_backbone.edge_index.size(1),
            dummy.encoder.edge_index.size(1),
        )

    def test_latent_slot_loss_backpropagates_to_its_owner_backbone(self):
        batch = _two_alanine_batch()
        view = build_geometry_neutral_graph_view(batch, _view_config())
        dummy = nn.Module()
        dummy.encoder = nn.Module()
        dummy.encoder.domain_local_slot_embedding = nn.Embedding(38, 4)
        full_slot_ids = torch.tensor(
            list(range(1, 15)) + list(range(1, 15)), dtype=torch.long
        )
        physical_states = torch.randn(
            28, 4, generator=torch.Generator().manual_seed(91),
            requires_grad=True,
        )

        slot_states = PretrainingESAModel._inject_latent_slot_readouts(
            dummy,
            batch,
            view,
            physical_states,
            full_slot_ids,
            "correct",
        )
        slot_states[18:28, 0].sum().backward()

        self.assertEqual(float(physical_states.grad[:14].abs().sum()), 0.0)
        self.assertGreater(
            float(physical_states.grad[14:18].abs().sum()), 0.0
        )
        self.assertEqual(
            float(physical_states.grad[18:28].abs().sum()), 0.0
        )

    def test_gradient_audit_is_diagnostic_and_keeps_main_backward_valid(self):
        node_states = torch.randn(
            7, 4, generator=torch.Generator().manual_seed(811),
            requires_grad=True,
        )
        losses = {
            "short_range_distance": node_states.square().mean(),
            "mlm": node_states[:, 0].abs().mean(),
        }
        logged = {}
        dummy = SimpleNamespace(
            config=SimpleNamespace(
                geometry_neutral_virtual_slots={
                    "gradient_audit_batches": 1,
                    "masked_real_distance_weight": 1.0,
                },
                task_weights={
                    "short_range_distance": 1.0,
                    "mlm": 0.3,
                },
            ),
            global_rank=0,
            log=lambda name, value, **_: logged.setdefault(name, value),
        )

        PretrainingESAModel._audit_representation_gradient_norms(
            dummy, losses, node_states, batch_idx=0
        )
        total = losses["short_range_distance"] + 0.3 * losses["mlm"]
        total.backward()

        self.assertIsNotNone(node_states.grad)
        self.assertTrue(bool(torch.isfinite(node_states.grad).all()))
        self.assertIn(
            "gradient_audit_mlm_representation_l2", logged
        )
        self.assertGreater(
            float(logged["gradient_audit_mlm_representation_l2"]), 0.0
        )

    def test_rna_uses_same_contract_with_twelve_backbone_anchors(self):
        batch = _two_cytidine_batch()
        view = build_geometry_neutral_graph_view(batch, _view_config())

        self.assertEqual(int(view.active_node_mask.sum()), 20 + 12)
        self.assertFalse(bool(view.active_node_mask[20:23].any()))
        self.assertTrue(bool(view.active_node_mask[23:35].all()))
        self.assertFalse(bool(view.active_node_mask[35:46].any()))
        self.assertTrue(bool(view.geometry_valid_mask[23:35].all()))
        self.assertFalse(bool(view.geometry_valid_mask[35:46].any()))

    def test_unimol_graph_keeps_legacy_nodes_edges_and_geometry(self):
        batch = SimpleNamespace(
            z=torch.tensor([0, 6, 8]),
            original_types=torch.tensor([7, 6, 8]),
            pos=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
            ),
            edge_index=torch.tensor([[0, 1, 2], [1, 0, 0]]),
            batch=torch.zeros(3, dtype=torch.long),
            block_idx=torch.arange(3),
            mlm_mask=torch.tensor([True, False, False]),
            pos_code=["N", "C", "O"],
            dataset_type="unimol",
            num_graphs=1,
            num_nodes=3,
        )
        view = build_geometry_neutral_graph_view(batch, _view_config())

        self.assertTrue(bool(view.active_node_mask.all()))
        self.assertTrue(bool(view.geometry_valid_mask.all()))
        self.assertTrue(torch.equal(view.physical_edge_index, batch.edge_index))
        self.assertTrue(torch.equal(view.compact_edge_index, batch.edge_index))
        self.assertEqual(view.slot_readout_edge_index.numel(), 0)
        self.assertTrue(torch.equal(view.attention_edge_index, batch.edge_index))
        self.assertTrue(torch.allclose(
            view.compact_pos, batch.pos - batch.pos.mean(dim=0, keepdim=True)
        ))

    def test_pyg_mixed_batch_preserves_unimol_subgraph(self):
        protein = _two_alanine_batch()
        protein_data = Data(**{
            key: value
            for key, value in vars(protein).items()
            if key not in ("batch", "num_graphs", "num_nodes")
        })
        unimol_data = Data(
            z=torch.tensor([0, 6, 8]),
            x=torch.tensor([0, 6, 8]),
            original_types=torch.tensor([7, 6, 8]),
            masked_types=torch.tensor([0, 6, 8]),
            pos=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
            ),
            edge_index=torch.tensor([[0, 1, 2], [1, 0, 0]]),
            block_idx=torch.arange(3),
            entity_idx=torch.zeros(3, dtype=torch.long),
            mlm_mask=torch.tensor([True, False, False]),
            pos_code=["N", "C", "O"],
            dataset_type="unimol",
        )
        unimol_data = attach_runtime_template_metadata(unimol_data)
        batch = Batch.from_data_list([protein_data, unimol_data])
        view = build_geometry_neutral_graph_view(batch, _view_config())

        offset = protein_data.num_nodes
        expected_unimol_edges = unimol_data.edge_index + offset
        physical_pairs = set(map(tuple, view.physical_edge_index.t().tolist()))
        self.assertTrue(view.active_node_mask[offset:].all())
        self.assertTrue(view.geometry_valid_mask[offset:].all())
        self.assertTrue(all(
            tuple(edge) in physical_pairs
            for edge in expected_unimol_edges.t().tolist()
        ))

    def test_config_rejects_old_and_new_coordinate_schemes_together(self):
        with self.assertRaisesRegex(ValueError, "cannot be enabled together"):
            create_pretraining_config(
                pretraining_tasks=["mlm"],
                protein_block_mlm=True,
                geometry_neutral_virtual_slots={
                    "enabled": True,
                    "protein": True,
                    "rna": False,
                },
                masked_slot_template={
                    "enabled": True,
                    "protein": True,
                    "rna": False,
                    "visible_virtuals_prepositioned": True,
                },
            )


if __name__ == "__main__":
    unittest.main()
