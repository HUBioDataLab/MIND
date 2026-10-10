#!/usr/bin/env python3
"""Prepositioned visible virtual (V) coordinates for the edge-free chunk builder.

``preposition_visible_virtuals`` is called per structure by
``data_loading/build_edge_free_chunks.py``. The former stage-2 command that rewrote
edge-stored PT chunks was removed together with that format.
"""

from __future__ import annotations

import torch

from core.masked_coordinate_corruption import build_masked_coordinate_corruption
from core.masked_slot_mapping import flatten_pos_codes
from data_loading.atom_position_vocab import is_virtual_position_code
from data_loading.lazy_universal_dataset import normalize_legacy_atom_ids


def preposition_visible_virtuals(sample, domain: str):
    """Normalize legacy ids and compute prepositioned visible-V coordinates.

    Returns ``(sample, result, original_dataset_type)``. ``sample`` carries the
    temporary single-graph fields ``batch``/``num_graphs``/``mlm_mask`` and
    ``dataset_type=domain``; ``result.input_pos`` holds the new coordinates.
    Only visible V coordinates may differ from the input.
    """
    sample = normalize_legacy_atom_ids(sample)
    original_pos = sample.pos.clone()
    original_dataset_type = getattr(sample, "dataset_type", domain)
    sample.dataset_type = domain
    sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long)
    sample.num_graphs = 1
    sample.mlm_mask = torch.zeros(sample.num_nodes, dtype=torch.bool)

    result = build_masked_coordinate_corruption(
        sample,
        domains=(domain,),
        relocate_context_virtuals=True,
    )
    if result.mapping_quarantines or result.coordinate_quarantines:
        raise AssertionError("an unmasked preprocessing sample produced target quarantine")

    codes = flatten_pos_codes(sample.pos_code, sample.num_nodes)
    original_virtual_mask = torch.tensor(
        [is_virtual_position_code(code) for code in codes], dtype=torch.bool
    )
    if bool((result.context_virtual_node_mask & ~original_virtual_mask).any()):
        raise AssertionError("visible-template preprocessing touched a real atom")
    if not torch.equal(
        result.input_pos[~result.context_virtual_node_mask],
        original_pos[~result.context_virtual_node_mask],
    ):
        raise AssertionError("visible-template preprocessing moved a non-V atom")
    return sample, result, original_dataset_type
