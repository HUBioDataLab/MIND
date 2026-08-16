"""Shared whole-block selection for protein and RNA MLM.

The selector performs the random draw exactly once and returns both views of
that draw: selected block ids and the corresponding node mask.  Runtime
template integration must consume this same selection instead of sampling a
second set of residues/nucleotides.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class BlockMaskSelection:
    """One immutable selection result with block-level and node-level views."""

    selected_block_ids: torch.Tensor
    selected_node_mask: torch.Tensor


def select_whole_blocks(
    block_ids: torch.Tensor,
    mask_ratio: float,
    *,
    generator: Optional[torch.Generator] = None,
    eligible_node_mask: Optional[torch.Tensor] = None,
) -> BlockMaskSelection:
    """Select whole blocks and expand the selection to their nodes.

    ``block_ids`` is local to one graph.  The count follows MIND's existing
    block-MLM rule: ``max(1, floor(num_blocks * mask_ratio))``.  A CPU
    generator is accepted for deterministic validation; training without a
    generator uses the tensor device's global RNG, preserving legacy behavior.
    """
    if block_ids.ndim != 1:
        raise ValueError("block_ids must be a one-dimensional tensor")
    if block_ids.numel() == 0:
        return BlockMaskSelection(
            selected_block_ids=block_ids.clone(),
            selected_node_mask=torch.zeros_like(block_ids, dtype=torch.bool),
        )

    if eligible_node_mask is None:
        unique_blocks = torch.unique(block_ids)
    else:
        if (
            eligible_node_mask.ndim != 1
            or eligible_node_mask.numel() != block_ids.numel()
        ):
            raise ValueError("eligible_node_mask must align with block_ids")
        unique_blocks = torch.unique(block_ids[eligible_node_mask.bool()])
        if unique_blocks.numel() == 0:
            return BlockMaskSelection(
                selected_block_ids=unique_blocks,
                selected_node_mask=torch.zeros_like(block_ids, dtype=torch.bool),
            )
    num_blocks = unique_blocks.numel()
    num_selected = max(1, int(num_blocks * mask_ratio))

    if generator is None:
        permutation = torch.randperm(num_blocks, device=block_ids.device)
    else:
        permutation = torch.randperm(num_blocks, generator=generator).to(
            block_ids.device
        )

    selected_block_ids = unique_blocks[permutation[:num_selected]]
    selected_node_mask = torch.isin(block_ids, selected_block_ids)
    return BlockMaskSelection(
        selected_block_ids=selected_block_ids,
        selected_node_mask=selected_node_mask,
    )


def legacy_oxt_eligible_node_mask(
    block_ids: torch.Tensor,
    pos_codes: Sequence[str],
) -> torch.Tensor:
    """Exclude complete legacy protein blocks containing terminal OXT."""
    if block_ids.ndim != 1 or len(pos_codes) != block_ids.numel():
        raise ValueError("block_ids and pos_codes must contain one value per node")
    oxt_node_mask = torch.tensor(
        [str(code).strip() == "OXT" for code in pos_codes],
        dtype=torch.bool,
        device=block_ids.device,
    )
    oxt_block_ids = torch.unique(block_ids[oxt_node_mask])
    return ~torch.isin(block_ids, oxt_block_ids)
