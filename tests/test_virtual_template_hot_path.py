import copy
import unittest
from types import SimpleNamespace

import torch

from core.masked_coordinate_corruption import rebuild_template_incident_edges
from core.masked_slot_mapping import (
    attach_runtime_template_metadata,
    map_masked_template_slots,
)
from data_loading.atom_position_vocab import PROTEIN_BACKBONE_ATOMS


def _protein_batch():
    codes = list(PROTEIN_BACKBONE_ATOMS) + ["CB"] + [
        f"V{index}" for index in range(2, 11)
    ]
    return SimpleNamespace(
        z=torch.full((14,), 6, dtype=torch.long),
        pos=torch.arange(42, dtype=torch.float32).reshape(14, 3) / 10,
        block_idx=torch.zeros(14, dtype=torch.long),
        batch=torch.zeros(14, dtype=torch.long),
        mlm_mask=torch.ones(14, dtype=torch.bool),
        pos_code=codes,
        dataset_type="protein",
        num_graphs=1,
        num_nodes=14,
    )


class VirtualTemplateHotPathTest(unittest.TestCase):
    def test_numeric_runtime_mapping_matches_legacy_mapping(self):
        legacy_batch = _protein_batch()
        numeric_batch = attach_runtime_template_metadata(copy.deepcopy(legacy_batch))
        legacy = map_masked_template_slots(legacy_batch)
        numeric = map_masked_template_slots(numeric_batch)

        self.assertEqual(len(legacy.mapped_blocks), 1)
        self.assertEqual(len(numeric.mapped_blocks), 1)
        legacy_block = legacy.mapped_blocks[0]
        numeric_block = numeric.mapped_blocks[0]
        self.assertTrue(torch.equal(legacy.node_mask, numeric.node_mask))
        self.assertTrue(torch.equal(
            legacy_block.node_indices, numeric_block.node_indices
        ))
        self.assertTrue(torch.equal(
            legacy_block.anchor_node_indices, numeric_block.anchor_node_indices
        ))
        self.assertTrue(torch.equal(
            legacy_block.virtual_slot_mask, numeric_block.virtual_slot_mask
        ))

    def test_targeted_knn_edges_match_dense_nearest_reference(self):
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [13.0, 0.0, 0.0],
            ]
        )
        batch = SimpleNamespace(
            batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            edge_index=torch.tensor([[1, 5], [2, 6]]),
            edge_attr=torch.tensor([[1.0], [1.0]]),
        )
        template_mask = torch.tensor(
            [True, False, False, False, False, False, False, False]
        )
        rebuilt = rebuild_template_incident_edges(
            batch, positions, template_mask, cutoff=2.1, max_neighbors=2
        )

        # Dense reference: node 0's two nearest non-self graph-local nodes.
        expected_new = {(0, 1), (1, 0), (0, 2), (2, 0)}
        kept_count = 2
        actual_new = set(map(tuple, rebuilt.edge_index[:, kept_count:].t().tolist()))
        self.assertEqual(actual_new, expected_new)
        self.assertTrue(torch.equal(rebuilt.edge_index[:, :kept_count], batch.edge_index))


if __name__ == "__main__":
    unittest.main()
