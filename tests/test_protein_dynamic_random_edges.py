#!/usr/bin/env python3
"""Unit tests for ProteinDynamicRandomEdges transform."""

import pytest
import torch
from torch_geometric.data import Data

from data_loading.protein_dynamic_random_edges import (
    OptimizedProteinDynamicRandomEdges,
    ProteinDynamicRandomEdges,
)


def _make_dummy_protein_data(num_residues: int = 50, with_edge_attr: bool = False, stretch: bool = False):
    canonical_14 = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'V', 'V', 'V', 'V', 'V']
    num_atoms = num_residues * 14
    pos_code = canonical_14 * num_residues

    # Create CA coordinates along a line or random
    if stretch:
        # Fully extended chain: 3.8A per residue along x axis (so i->i+3 distance is 11.4A > 10.0A)
        pos = torch.zeros((num_atoms, 3), dtype=torch.float)
        for res_idx in range(num_residues):
            base_x = res_idx * 3.8
            for atom_slot in range(14):
                pos[res_idx * 14 + atom_slot] = torch.tensor([base_x, atom_slot * 0.5, 0.0])
    else:
        pos = torch.randn(num_atoms, 3) * 20.0

    block_idx = torch.tensor([i // 14 for i in range(num_atoms)], dtype=torch.long)
    entity_idx = torch.zeros(num_atoms, dtype=torch.long)

    # Build Tier 1 sequence edges (CA-CA for i to i+1, i+2, i+3)
    edges = []
    ca_nodes = [i * 14 + 1 for i in range(num_residues)]
    for i, ca_i in enumerate(ca_nodes):
        for k in (1, 2, 3):
            if i + k < len(ca_nodes):
                ca_j = ca_nodes[i + k]
                edges.append([ca_i, ca_j])
                edges.append([ca_j, ca_i])

    # Also add some random Tier 3 edges (> 10A)
    for i in range(len(ca_nodes) - 10):
        ca_i = ca_nodes[i]
        ca_j = ca_nodes[i + 10]
        edges.append([ca_i, ca_j])
        edges.append([ca_j, ca_i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1, keepdim=True) if with_edge_attr else None

    data = Data(
        pos=pos,
        pos_code=pos_code,
        block_idx=block_idx,
        entity_idx=entity_idx,
        edge_index=edge_index,
        dataset_type='protein',
    )
    if with_edge_attr:
        data.edge_attr = edge_attr
    return data


def test_tier1_sequence_edges_preserved_even_when_distance_exceeds_10A():
    # Stretched chain has i->i+3 distance = 11.4A (> 10.0A min_distance threshold)
    data = _make_dummy_protein_data(num_residues=30, with_edge_attr=True, stretch=True)

    # Identify original Tier 1 sequence edges
    ca_nodes = set(i for i, code in enumerate(data.pos_code) if code == 'CA')
    orig_edges = set(tuple(p) for p in data.edge_index.t().tolist())
    orig_tier1 = set()
    for s, d in orig_edges:
        if s in ca_nodes and d in ca_nodes:
            block_diff = abs(data.block_idx[s].item() - data.block_idx[d].item())
            if 0 < block_diff <= 3:
                orig_tier1.add((s, d))

    assert len(orig_tier1) > 0

    transform = OptimizedProteinDynamicRandomEdges(
        num_random_edges=4, min_distance=10.0, sequence_neighbors_k=3, replace_existing_random_edges=True
    )
    out = transform(data)

    out_edges = set(tuple(p) for p in out.edge_index.t().tolist())

    # Every single Tier 1 sequence edge must be preserved (0 lost)
    missing_tier1 = orig_tier1 - out_edges
    assert len(missing_tier1) == 0, f"Lost {len(missing_tier1)} legitimate Tier 1 sequence edges!"


def test_dynamic_random_edges_synchronizes_edge_attr_and_deduplicates():
    data = _make_dummy_protein_data(num_residues=40, with_edge_attr=True)
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


def test_dynamic_random_edges_non_protein_passthrough():
    pos = torch.randn(20, 3)
    edge_index = torch.randint(0, 20, (2, 40))
    data = Data(pos=pos, edge_index=edge_index, dataset_type='unimol')

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4)
    out = transform(data)

    assert out.edge_index.size(1) == 40


def test_dynamic_random_edges_missing_pos_code_fails_loud():
    pos = torch.randn(28, 3)
    edge_index = torch.randint(0, 28, (2, 50))
    data = Data(pos=pos, edge_index=edge_index, dataset_type='protein')

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4)
    with pytest.raises(ValueError, match="missing 'pos_code' contract"):
        transform(data)


def test_dynamic_random_edges_missing_block_idx_fails_loud():
    pos = torch.randn(28, 3)
    edge_index = torch.randint(0, 28, (2, 50))
    pos_code = ['CA'] * 28
    data = Data(pos=pos, pos_code=pos_code, edge_index=edge_index, dataset_type='protein')

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4)
    with pytest.raises(ValueError, match="missing 'block_idx' contract"):
        transform(data)


def test_dynamic_random_edges_missing_entity_idx_fails_loud():
    pos = torch.randn(28, 3)
    edge_index = torch.randint(0, 28, (2, 50))
    pos_code = ['CA'] * 28
    block_idx = torch.tensor([i // 14 for i in range(28)], dtype=torch.long)
    data = Data(pos=pos, pos_code=pos_code, block_idx=block_idx, edge_index=edge_index, dataset_type='protein')

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4)
    with pytest.raises(ValueError, match="missing 'entity_idx' contract"):
        transform(data)


def test_dynamic_random_edges_pos_code_length_mismatch_fails_loud():
    pos = torch.randn(28, 3)
    edge_index = torch.randint(0, 28, (2, 50))
    pos_code = ['CA'] * 14  # Length 14 != 28
    data = Data(pos=pos, pos_code=pos_code, edge_index=edge_index, dataset_type='protein')

    transform = OptimizedProteinDynamicRandomEdges(num_random_edges=4)
    with pytest.raises(ValueError, match="does not match atom count"):
        transform(data)
