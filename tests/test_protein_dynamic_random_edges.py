#!/usr/bin/env python3
"""Unit tests for ProteinDynamicRandomEdges transform."""

import pytest
import torch
from torch_geometric.data import Data

from data_loading.protein_dynamic_random_edges import (
    OptimizedProteinDynamicRandomEdges,
    ProteinDynamicRandomEdges,
)


def _make_dummy_protein_data(num_residues: int = 50, use_pos_code: bool = True, with_edge_attr: bool = False):
    canonical_14 = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'V', 'V', 'V', 'V', 'V']
    num_atoms = num_residues * 14
    pos = torch.randn(num_atoms, 3) * 20.0
    edge_index = torch.randint(0, num_atoms, (2, 500))

    edge_attr = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1, keepdim=True) if with_edge_attr else None

    if use_pos_code:
        pos_code = canonical_14 * num_residues
        data = Data(pos=pos, pos_code=pos_code, edge_index=edge_index, dataset_type='protein')
    else:
        position_indices = torch.tensor([i % 14 for i in range(num_atoms)])
        data = Data(pos=pos, position_indices=position_indices, edge_index=edge_index, dataset_type='protein')

    if with_edge_attr:
        data.edge_attr = edge_attr
    return data


def test_dynamic_random_edges_with_pos_code():
    data = _make_dummy_protein_data(num_residues=50, use_pos_code=True)
    orig_num_edges = data.edge_index.size(1)

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4, min_distance=5.0)
    out = transform(data)

    assert out.edge_index.size(1) > 0


def test_dynamic_random_edges_synchronizes_edge_attr_and_deduplicates():
    data = _make_dummy_protein_data(num_residues=40, use_pos_code=True, with_edge_attr=True)
    assert data.edge_index.size(1) == data.edge_attr.size(0)

    transform = OptimizedProteinDynamicRandomEdges(
        num_random_edges=6, min_distance=8.0, replace_existing_random_edges=True
    )
    out = transform(data)

    # 1. edge_index and edge_attr must have identical lengths
    assert out.edge_index.size(1) == out.edge_attr.size(0), (
        f"Length mismatch: edge_index={out.edge_index.size(1)} vs edge_attr={out.edge_attr.size(0)}"
    )

    # 2. No duplicate directed edges
    pairs = [tuple(p) for p in out.edge_index.t().tolist()]
    assert len(pairs) == len(set(pairs)), f"Found {len(pairs) - len(set(pairs))} duplicate edges!"


def test_dynamic_random_edges_with_template_anchor_order():
    num_residues = 30
    num_atoms = num_residues * 14
    pos = torch.randn(num_atoms, 3) * 20.0
    edge_index = torch.randint(0, num_atoms, (2, 300))
    template_anchor_order = torch.tensor([1 if i % 14 == 1 else -1 for i in range(num_atoms)], dtype=torch.int8)
    data = Data(pos=pos, template_anchor_order=template_anchor_order, edge_index=edge_index, dataset_type='protein')

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4, min_distance=5.0)
    out = transform(data)

    assert out.edge_index.size(1) > 0


def test_dynamic_random_edges_non_protein_passthrough():
    pos = torch.randn(20, 3)
    edge_index = torch.randint(0, 20, (2, 40))
    data = Data(pos=pos, edge_index=edge_index, dataset_type='unimol')

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4)
    out = transform(data)

    assert out.edge_index.size(1) == 40


def test_dynamic_random_edges_missing_metadata_fails_loud():
    pos = torch.randn(28, 3)
    edge_index = torch.randint(0, 28, (2, 50))
    data = Data(pos=pos, edge_index=edge_index, dataset_type='protein')

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4)
    with pytest.raises(ValueError, match="missing position metadata"):
        transform(data)
