#!/usr/bin/env python3
"""Unit tests for chemical coordination calculation in UniversalMolecularEncoder."""

import pytest
import torch
from types import SimpleNamespace

from core.molecular_encoder import UniversalMolecularEncoder
from core.pretraining_config import PretrainingConfig


def test_chemical_coordination_filters_spatial_and_random_edges():
    config = PretrainingConfig(
        hidden_dims=[256],
        atom_types=121,
        cutoff_distance=5.0,
        use_posenc=False,
    )
    encoder = UniversalMolecularEncoder(config)

    # 3 atoms: Atom 0 is at (0,0,0), Atom 1 is at (1.5, 0, 0) [covalent bond], Atom 2 is at (4.0, 0, 0) [spatial 4A contact]
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ], dtype=torch.float)
    x = torch.tensor([6, 6, 6], dtype=torch.long)  # Carbons

    # Edges: (0-1) covalent 1.5A, (1-2) non-covalent 2.5A, (0-2) spatial 4.0A
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 2],
        [1, 0, 2, 1, 2, 0],
    ], dtype=torch.long)

    batch_vec = torch.tensor([0, 0, 0], dtype=torch.long)
    batch_obj = SimpleNamespace(
        edge_index=edge_index,
        batch=batch_vec,
        bond_edge_index=None,
    )

    x_dense = x.unsqueeze(0)
    edge_src, edge_dst = edge_index[0], edge_index[1]
    edge_distances = torch.norm(pos[edge_src] - pos[edge_dst], dim=1)

    coord_dense = encoder._compute_chemical_coordination(
        x_dense, batch_obj, pos=pos, edge_distances=edge_distances
    )

    # Atom 0 is connected covalently only to Atom 1 (dist 1.5A) -> coordination = 1.0 (not 2.0)
    # Atom 1 is connected covalently only to Atom 0 (dist 1.5A) -> coordination = 1.0
    # Atom 2 is at distance 2.5A and 4.0A -> coordination = 0.0 (no covalent bonds <= 1.85A)
    assert coord_dense[0, 0].item() == 1.0, f"Expected Atom 0 coordination = 1.0, got {coord_dense[0, 0].item()}"
    assert coord_dense[0, 1].item() == 1.0, f"Expected Atom 1 coordination = 1.0, got {coord_dense[0, 1].item()}"
    assert coord_dense[0, 2].item() == 0.0, f"Expected Atom 2 coordination = 0.0, got {coord_dense[0, 2].item()}"


def test_chemical_coordination_with_explicit_bond_edge_index():
    config = PretrainingConfig(
        hidden_dims=[256],
        atom_types=121,
        use_posenc=False,
    )
    encoder = UniversalMolecularEncoder(config)

    x = torch.tensor([6, 6, 8], dtype=torch.long)
    pos = torch.zeros((3, 3), dtype=torch.float)
    x_dense = x.unsqueeze(0)
    batch_vec = torch.tensor([0, 0, 0], dtype=torch.long)

    # Full edge_index has all pairwise edges (6 edges)
    edge_index = torch.tensor([
        [0, 1, 1, 0, 1, 2, 2, 1, 0, 2, 2, 0],
        [1, 0, 0, 1, 2, 1, 1, 2, 2, 0, 0, 2],
    ], dtype=torch.long)

    # Explicit bond_edge_index has only single bond 0-1
    bond_edge_index = torch.tensor([
        [0, 1],
        [1, 0],
    ], dtype=torch.long)

    batch_obj = SimpleNamespace(
        edge_index=edge_index,
        batch=batch_vec,
        bond_edge_index=bond_edge_index,
    )

    coord_dense = encoder._compute_chemical_coordination(x_dense, batch_obj)
    assert coord_dense[0, 0].item() == 1.0
    assert coord_dense[0, 1].item() == 1.0
    assert coord_dense[0, 2].item() == 0.0


def test_chemical_coordination_missing_distances_fails_loud():
    config = PretrainingConfig(
        hidden_dims=[256],
        atom_types=121,
        use_posenc=False,
    )
    encoder = UniversalMolecularEncoder(config)

    x_dense = torch.tensor([[6, 6, 6]], dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch_obj = SimpleNamespace(
        edge_index=edge_index,
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
        bond_edge_index=None,
    )

    with pytest.raises(ValueError, match="requires either 'bond_edge_index' or 'edge_distances'"):
        encoder._compute_chemical_coordination(x_dense, batch_obj, pos=None, edge_distances=None)
