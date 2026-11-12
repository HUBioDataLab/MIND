"""random_walk masking on a graph without usable bonds falls back to random_atom, visibly."""

import logging

import pytest
import torch

from data_loading import masking_utils
from data_loading.masking_utils import (
    random_walk_fallback_count,
    reset_random_walk_fallback_count,
    select_mask_indices,
)


@pytest.fixture(autouse=True)
def _reset_counter():
    reset_random_walk_fallback_count()
    yield
    reset_random_walk_fallback_count()


def _walk(num_nodes, bonds):
    return select_mask_indices(
        num_nodes, 0.5, "random_atom", bond_edge_index=bonds,
        dataset_type="coconut", coconut_method="random_walk",
        generator=torch.Generator().manual_seed(0),
    )


@pytest.mark.parametrize("bonds", [None, torch.empty((2, 0), dtype=torch.long),
                                   torch.tensor([[2, 3], [2, 3]])])  # only self-loops
def test_bondless_graph_falls_back_but_is_counted(bonds):
    mask = _walk(6, bonds)
    assert mask.numel() == 3
    assert random_walk_fallback_count() == 1
    _walk(6, bonds)
    assert random_walk_fallback_count() == 2


def test_bonded_graph_and_single_atom_graph_are_not_counted():
    ring = torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]])
    _walk(4, ring)
    assert _walk(1, None).tolist() == [0]
    assert random_walk_fallback_count() == 0


def test_every_fallback_is_logged_until_the_log_is_rate_limited(caplog):
    caplog.set_level(logging.WARNING, logger=masking_utils.logger.name)
    for _ in range(3):
        _walk(6, None)
    assert [r.getMessage() for r in caplog.records if "random_walk" in r.getMessage()]
    assert "1 graph" in caplog.records[0].getMessage()
