#!/usr/bin/env python3
"""Unit tests for ProteinDynamicRandomEdges transform."""

import pytest
import torch
from torch_geometric.data import Data

from data_loading.protein_dynamic_random_edges import (
    OptimizedProteinDynamicRandomEdges,
    ProteinDynamicRandomEdges,
)


def _make_dummy_protein_data(num_residues: int = 50, use_pos_code: bool = True):
    canonical_14 = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'V', 'V', 'V', 'V', 'V']
    num_atoms = num_residues * 14
    pos = torch.randn(num_atoms, 3) * 20.0
    edge_index = torch.randint(0, num_atoms, (2, 500))

    if use_pos_code:
        pos_code = canonical_14 * num_residues
        return Data(pos=pos, pos_code=pos_code, edge_index=edge_index, dataset_type='protein')
    else:
        position_indices = torch.tensor([i % 14 for i in range(num_atoms)])
        return Data(pos=pos, position_indices=position_indices, edge_index=edge_index, dataset_type='protein')


def test_dynamic_random_edges_with_pos_code():
    data = _make_dummy_protein_data(num_residues=50, use_pos_code=True)
    orig_num_edges = data.edge_index.size(1)

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4, min_distance=5.0)
    out = transform(data)

    assert out.edge_index.size(1) > orig_num_edges
    added_edges = out.edge_index[:, orig_num_edges:]
    
    # Verify all added edges connect CA atoms
    ca_indices = set(i for i, code in enumerate(data.pos_code) if code == 'CA')
    for s, d in zip(added_edges[0].tolist(), added_edges[1].tolist()):
        assert s in ca_indices, f"Source node {s} is not a CA atom"
        assert d in ca_indices, f"Destination node {d} is not a CA atom"


def test_dynamic_random_edges_with_template_anchor_order():
    num_residues = 30
    num_atoms = num_residues * 14
    pos = torch.randn(num_atoms, 3) * 20.0
    edge_index = torch.randint(0, num_atoms, (2, 300))
    template_anchor_order = torch.tensor([1 if i % 14 == 1 else -1 for i in range(num_atoms)], dtype=torch.int8)
    data = Data(pos=pos, template_anchor_order=template_anchor_order, edge_index=edge_index, dataset_type='protein')

    orig_num_edges = data.edge_index.size(1)
    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4, min_distance=5.0)
    out = transform(data)

    assert out.edge_index.size(1) > orig_num_edges


def test_dynamic_random_edges_non_protein_passthrough():
    # RNA / UniMol should pass through without error and without adding CA edges
    pos = torch.randn(20, 3)
    edge_index = torch.randint(0, 20, (2, 40))
    data = Data(pos=pos, edge_index=edge_index, dataset_type='unimol')

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4)
    out = transform(data)

    assert out.edge_index.size(1) == 40


def test_dynamic_random_edges_missing_metadata_fails_loud():
    # Protein with missing pos_code / template metadata must raise ValueError, not silently return
    pos = torch.randn(28, 3)
    edge_index = torch.randint(0, 28, (2, 50))
    data = Data(pos=pos, edge_index=edge_index, dataset_type='protein')

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4)
    with pytest.raises(ValueError, match="missing position metadata"):
        transform(data)
