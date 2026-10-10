"""Geometry-neutral runtime view for fixed protein/RNA virtual slots.

The persisted representation stays fixed-width (14 protein atoms or 23 RNA
atoms).  This module builds the smaller graph that is actually shown to the
encoder:

* visible virtual atoms are absent from attention and message passing;
* sidechain/base slots in an MLM-selected block are latent readout targets,
  not encoder nodes;
* selected sidechain/base slots have no input geometry, even when their target
  is a real atom;
* physical edges are completed from geometry-valid atoms at runtime so virtual
  atoms in an old cache cannot consume a radius-graph neighbour budget.

Clean coordinates of masked real slots are returned only as distance targets.
They are never used to construct the encoder graph.
"""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch_cluster import radius_graph

from data_loading.bond_contract import resolve_bond_sources, validate_graph_edges
from core.masked_slot_mapping import map_masked_template_slots
from data_loading.atom_position_vocab import VIRTUAL_ATOMIC_NUMBER


PROTEIN_GRAPH_TYPES = frozenset(("pdb", "protein"))
RNA_GRAPH_TYPES = frozenset(("rna",))


@dataclass(frozen=True)
class GeometryNeutralGraphView:
    """Masks, edges, and targets derived from one full fixed-slot batch."""

    active_node_mask: torch.Tensor
    geometry_valid_mask: torch.Tensor
    target_slot_mask: torch.Tensor
    target_real_slot_mask: torch.Tensor
    physical_edge_index: torch.Tensor
    slot_readout_edge_index: torch.Tensor
    attention_edge_index: torch.Tensor
    compact_to_full: torch.Tensor
    full_to_compact: torch.Tensor
    compact_edge_index: torch.Tensor
    compact_bond_edge_index: torch.Tensor
    has_explicit_bonds: torch.Tensor
    compact_attention_edge_index: torch.Tensor
    compact_pos: torch.Tensor
    centered_full_pos_for_losses: torch.Tensor


def build_slot_readout_edges_for_owner_mode(
    batch: object,
    view: GeometryNeutralGraphView,
    mode: str,
) -> torch.Tensor:
    """Return full-index slot->backbone links for a context audit mode.

    These links never enter ESA attention or geometry. ``correct`` is the
    training path. ``removed`` gives slots no backbone context. ``shuffled``
    deterministically rotates each masked block's readout anchors to another
    masked block in the same molecular graph. The latter two modes are
    validation-only counterfactuals; they never change coordinates, targets,
    physical edges, or persisted data.
    """
    if mode == "correct":
        return view.slot_readout_edge_index
    if mode == "removed":
        return torch.empty(
            (2, 0),
            dtype=torch.long,
            device=view.slot_readout_edge_index.device,
        )
    if mode != "shuffled":
        raise ValueError(
            "slot readout mode must be one of: correct, removed, shuffled"
        )

    owner_edges = view.slot_readout_edge_index
    if owner_edges.numel() == 0:
        return owner_edges

    source, target = owner_edges
    graph_ids = batch.batch.long()
    block_ids = batch.block_idx.long()
    shuffled_parts = []

    # A molecular graph has a single domain, so keeping rotations within a
    # graph also prevents protein/RNA owner anchors from ever being mixed.
    for graph_id in torch.unique(graph_ids[source], sorted=True).tolist():
        graph_edge_mask = graph_ids[source] == graph_id
        graph_sources = source[graph_edge_mask]
        graph_targets = target[graph_edge_mask]
        masked_blocks = torch.unique(
            block_ids[graph_sources], sorted=True
        )
        if masked_blocks.numel() < 2:
            # There is no valid different owner in this graph. Removing these
            # owner edges is the only non-leaking counterfactual.
            continue

        block_count = int(masked_blocks.numel())
        for index, block_id in enumerate(masked_blocks.tolist()):
            donor_block = masked_blocks[(index + 1) % block_count]
            queries = torch.unique(
                graph_sources[block_ids[graph_sources] == block_id],
                sorted=True,
            )
            donor_targets = torch.unique(
                graph_targets[block_ids[graph_sources] == donor_block],
                sorted=True,
            )
            if queries.numel() and donor_targets.numel():
                shuffled_parts.append(
                    torch.stack(
                        (
                            queries.repeat_interleave(donor_targets.numel()),
                            donor_targets.repeat(queries.numel()),
                        ),
                        dim=0,
                    )
                )

    if shuffled_parts:
        shuffled_owner_edges = torch.cat(shuffled_parts, dim=1)
    else:
        shuffled_owner_edges = torch.empty(
            (2, 0), dtype=torch.long, device=source.device
        )

    if shuffled_owner_edges.numel() and bool(
        (
            batch.batch[shuffled_owner_edges[0]]
            != batch.batch[shuffled_owner_edges[1]]
        ).any()
    ):
        raise RuntimeError("shuffled slot readout created a cross-graph link")
    return shuffled_owner_edges


def _graph_types(batch: object) -> list[str]:
    value = getattr(batch, "dataset_type", None)
    if value is None:
        raise ValueError(
            "geometry-neutral virtual slots require batch.dataset_type"
        )
    if isinstance(value, (list, tuple)):
        return [str(item).lower() for item in value]
    num_graphs = int(getattr(batch, "num_graphs", 1) or 1)
    return [str(value).lower()] * num_graphs


def _enabled_domain_names(settings: dict) -> tuple[str, ...]:
    return tuple(
        domain for domain in ("protein", "rna") if settings.get(domain, False)
    )


def _relevant_node_mask(
    batch: object,
    domains: Iterable[str],
    num_nodes: int,
) -> torch.Tensor:
    domain_set = set(domains)
    graph_types = _graph_types(batch)
    graph_relevant = torch.zeros(
        len(graph_types), dtype=torch.bool, device=batch.batch.device
    )
    for graph_index, graph_type in enumerate(graph_types):
        graph_relevant[graph_index] = (
            ("protein" in domain_set and graph_type in PROTEIN_GRAPH_TYPES)
            or ("rna" in domain_set and graph_type in RNA_GRAPH_TYPES)
        )
    if batch.batch.numel() != num_nodes:
        raise ValueError("batch.batch must contain one graph id per node")
    return graph_relevant[batch.batch.long()]


def _append_missing_edges(
    kept_edges: torch.Tensor,
    candidate_edges: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Preserve cached edge order and append each genuinely missing edge once."""
    if candidate_edges.numel() == 0:
        return kept_edges
    candidate_keys = (
        candidate_edges[0].long() * num_nodes + candidate_edges[1].long()
    )
    candidate_keys = torch.unique(candidate_keys, sorted=True)
    if kept_edges.numel():
        kept_keys = kept_edges[0].long() * num_nodes + kept_edges[1].long()
        candidate_keys = candidate_keys[~torch.isin(candidate_keys, kept_keys)]
    missing_edges = torch.stack(
        (candidate_keys // num_nodes, candidate_keys % num_nodes), dim=0
    )
    return torch.cat((kept_edges, missing_edges), dim=1)


def _complete_physical_edges(
    batch: object,
    pos: torch.Tensor,
    geometry_valid_mask: torch.Tensor,
    spatial_groups: Iterable[tuple[torch.Tensor, int]],
    *,
    cutoff: float,
    preserve_cached_edge_mask: torch.Tensor,
) -> torch.Tensor:
    """Filter invalid endpoints and restore real-only radius neighbours."""
    edge_index = batch.edge_index.long()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("batch.edge_index must have shape [2, E]")

    if preserve_cached_edge_mask.numel() != edge_index.size(1):
        raise ValueError("preserve_cached_edge_mask must align with edge_index")
    keep = (
        geometry_valid_mask[edge_index[0]]
        & geometry_valid_mask[edge_index[1]]
        & preserve_cached_edge_mask
    )
    kept_edges = edge_index[:, keep]

    completed_edges = kept_edges
    for group_mask, max_neighbors in spatial_groups:
        spatial_nodes = torch.where(geometry_valid_mask & group_mask)[0]
        if spatial_nodes.numel() <= 1:
            continue
        local_edges = radius_graph(
            pos[spatial_nodes],
            r=float(cutoff),
            batch=batch.batch[spatial_nodes],
            loop=False,
            max_num_neighbors=int(max_neighbors),
        )
        restored_edges = spatial_nodes[local_edges]
        completed_edges = _append_missing_edges(
            completed_edges, restored_edges, int(pos.shape[0])
        )
    return completed_edges


def _cached_structural_edge_mask(
    batch: object,
    protein_node_mask: torch.Tensor,
    rna_node_mask: torch.Tensor,
    *,
    cutoff: float,
    sequence_neighbors_k: int,
    random_edge_min_distance: float,
) -> torch.Tensor:
    """Keep non-fixed-domain edges plus protein CA sequence/random tiers.

    Protein cached non-structural edges and all RNA cached spatial edges are
    discarded because their max-neighbour selection may already have been
    distorted by V nodes. They are rebuilt from real-only coordinates.
    """
    edge_index = batch.edge_index.long()
    src, dst = edge_index
    touches_fixed_domain = (
        protein_node_mask[src]
        | protein_node_mask[dst]
        | rna_node_mask[src]
        | rna_node_mask[dst]
    )
    preserve = ~touches_fixed_domain

    protein_pair = protein_node_mask[src] & protein_node_mask[dst]
    if not bool(protein_pair.any()):
        return preserve
    anchor_order = getattr(batch, "template_anchor_order", None)
    if anchor_order is None or anchor_order.numel() != batch.z.numel():
        raise ValueError(
            "geometry-neutral protein hybrid edges require "
            "template_anchor_order runtime metadata"
        )
    ca_nodes = anchor_order == 1
    ca_pair = protein_pair & ca_nodes[src] & ca_nodes[dst]
    if not bool(ca_pair.any()):
        return preserve

    same_graph = batch.batch[src] == batch.batch[dst]
    block_gap = torch.abs(batch.block_idx[src] - batch.block_idx[dst])
    sequence_edge = (
        ca_pair
        & same_graph
        & (block_gap > 0)
        & (block_gap <= int(sequence_neighbors_k))
    )
    entity_idx = getattr(batch, "entity_idx", None)
    if entity_idx is not None and entity_idx.numel() == batch.z.numel():
        sequence_edge &= entity_idx[src] == entity_idx[dst]

    distances = torch.linalg.vector_norm(batch.pos[src] - batch.pos[dst], dim=1)
    random_long_edge = ca_pair & same_graph & (
        distances >= float(random_edge_min_distance)
    )
    preserve |= sequence_edge | random_long_edge
    return preserve


def _center_geometry(
    pos: torch.Tensor,
    batch_mapping: torch.Tensor,
    geometry_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center with physical atoms only; geometry-invalid queries get zero."""
    encoder_centered = torch.zeros_like(pos)
    loss_centered = pos.clone()
    for graph_id in torch.unique(batch_mapping):
        graph_nodes = batch_mapping == graph_id
        physical_nodes = graph_nodes & geometry_valid_mask
        if not bool(physical_nodes.any()):
            raise ValueError(
                f"graph {int(graph_id)} has no geometry-valid atoms"
            )
        center = pos[physical_nodes].mean(dim=0, keepdim=True)
        encoder_centered[physical_nodes] = pos[physical_nodes] - center
        loss_centered[graph_nodes] = pos[graph_nodes] - center
    return encoder_centered, loss_centered




def _masked_slot_readout_edges(
    batch: object,
    mapping: object,
    target_slot_mask: torch.Tensor,
    geometry_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Map each latent slot readout only to its own physical backbone.

    These directed slot->anchor links are used only after ESA has encoded the
    physical graph. They never enter attention, geometric/RBF features, or
    distance-prediction targets.
    """
    edge_parts = []
    for mapped_block in mapping.mapped_blocks:
        slot_nodes = mapped_block.node_indices.to(batch.z.device)
        queries = slot_nodes[target_slot_mask[slot_nodes]]
        block_nodes = torch.where(
            (batch.batch == mapped_block.graph_index)
            & (batch.block_idx == mapped_block.block_id)
        )[0]
        anchors = block_nodes[
            geometry_valid_mask[block_nodes] & ~target_slot_mask[block_nodes]
        ]
        expected_queries = 10 if mapped_block.domain == "protein" else 11
        expected_anchors = 4 if mapped_block.domain == "protein" else 12
        if queries.numel() != expected_queries or anchors.numel() != expected_anchors:
            raise RuntimeError(
                "masked slot readout requires a complete fixed-slot block: "
                f"graph={mapped_block.graph_index}, block={mapped_block.block_id}, "
                f"domain={mapped_block.domain}, queries={queries.numel()}/"
                f"{expected_queries}, anchors={anchors.numel()}/{expected_anchors}"
            )
        edge_parts.append(
            torch.stack(
                (
                    queries.repeat_interleave(anchors.numel()),
                    anchors.repeat(queries.numel()),
                ),
                dim=0,
            )
        )

    if not edge_parts:
        return torch.empty((2, 0), dtype=torch.long, device=batch.z.device)
    return torch.cat(edge_parts, dim=1)


def build_geometry_neutral_graph_view(
    batch: object,
    config: object,
) -> GeometryNeutralGraphView | None:
    """Build the encoder-only graph without mutating persisted/raw geometry."""
    settings = getattr(config, "geometry_neutral_virtual_slots", None) or {}
    if not settings.get("enabled", False):
        return None

    required = ("z", "pos", "batch", "block_idx", "edge_index", "mlm_mask")
    missing = [name for name in required if getattr(batch, name, None) is None]
    if missing:
        raise ValueError(
            "geometry-neutral virtual slots require: " + ", ".join(missing)
        )
    original_types = getattr(batch, "original_types", None)
    if original_types is None:
        raise ValueError(
            "geometry-neutral virtual slots require pre-mask original_types"
        )

    num_nodes = int(batch.z.numel())
    validate_graph_edges(batch.edge_index, batch.batch)
    bond_edge_index, has_explicit_bonds = resolve_bond_sources(
        batch, batch.batch, len(_graph_types(batch))
    )
    domains = _enabled_domain_names(settings)
    protein_relevant = _relevant_node_mask(batch, ("protein",), num_nodes)
    rna_relevant = _relevant_node_mask(batch, ("rna",), num_nodes)
    if "protein" not in domains:
        protein_relevant.zero_()
    if "rna" not in domains:
        rna_relevant.zero_()
    relevant = protein_relevant | rna_relevant
    mapping = map_masked_template_slots(batch, domains=domains)
    if mapping.quarantined_blocks:
        details = "; ".join(
            f"g={item.graph_index}/block={item.block_id}:"
            + ",".join(item.reasons)
            for item in mapping.quarantined_blocks
        )
        raise RuntimeError(
            "geometry-neutral virtual slots rejected masked blocks: " + details
        )

    mlm_mask = batch.mlm_mask.bool()
    target_slot_mask = mapping.node_mask.bool()
    visible_virtual = (
        relevant
        & (original_types == VIRTUAL_ATOMIC_NUMBER)
        & ~mlm_mask
    )
    # Visible V padding and every masked sidechain/base target are absent from
    # the encoder. The latter are reconstructed as latent readouts after ESA.
    active_node_mask = ~visible_virtual & ~target_slot_mask
    geometry_valid_mask = active_node_mask
    target_real_slot_mask = (
        target_slot_mask & (original_types != VIRTUAL_ATOMIC_NUMBER)
    )

    cutoff = float(getattr(config, "cutoff_distance", 5.0))
    preserve_cached_edge_mask = _cached_structural_edge_mask(
        batch,
        protein_relevant,
        rna_relevant,
        cutoff=cutoff,
        sequence_neighbors_k=int(getattr(config, "sequence_neighbors_k", 3)),
        random_edge_min_distance=float(
            getattr(config, "random_edge_min_distance", 10.0)
        ),
    )
    physical_edge_index = _complete_physical_edges(
        batch,
        batch.pos,
        geometry_valid_mask,
        (
            (
                protein_relevant,
                int(getattr(config, "max_spatial_neighbors", 48)),
            ),
            (rna_relevant, int(getattr(config, "max_neighbors", 64))),
        ),
        cutoff=cutoff,
        preserve_cached_edge_mask=preserve_cached_edge_mask,
    )
    if physical_edge_index.numel() and not bool(
        (
            geometry_valid_mask[physical_edge_index[0]]
            & geometry_valid_mask[physical_edge_index[1]]
        ).all()
    ):
        raise RuntimeError("physical graph contains a geometry-invalid endpoint")

    slot_readout_edge_index = _masked_slot_readout_edges(
        batch,
        mapping,
        target_slot_mask,
        geometry_valid_mask,
    )
    if slot_readout_edge_index.numel() and not bool(
        (
            target_slot_mask[slot_readout_edge_index[0]]
            & geometry_valid_mask[slot_readout_edge_index[1]]
        ).all()
    ):
        raise RuntimeError("slot readout link violates slot->anchor contract")
    attention_edge_index = physical_edge_index

    compact_to_full = torch.where(active_node_mask)[0]
    full_to_compact = torch.full(
        (num_nodes,), -1, dtype=torch.long, device=batch.z.device
    )
    full_to_compact[compact_to_full] = torch.arange(
        compact_to_full.numel(), device=batch.z.device
    )
    compact_bonds = full_to_compact[bond_edge_index]
    compact_bond_edge_index = compact_bonds[:, (compact_bonds >= 0).all(dim=0)]
    compact_edge_index = full_to_compact[physical_edge_index]
    if compact_edge_index.numel() and bool((compact_edge_index < 0).any()):
        raise RuntimeError("physical edge points to an inactive encoder node")
    compact_attention_edge_index = full_to_compact[attention_edge_index]
    if compact_attention_edge_index.numel() and bool(
        (compact_attention_edge_index < 0).any()
    ):
        raise RuntimeError("attention edge points to an inactive encoder node")

    centered_full_pos, centered_full_pos_for_losses = _center_geometry(
        batch.pos, batch.batch, geometry_valid_mask
    )
    compact_pos = centered_full_pos[compact_to_full]

    return GeometryNeutralGraphView(
        active_node_mask=active_node_mask,
        geometry_valid_mask=geometry_valid_mask,
        target_slot_mask=target_slot_mask,
        target_real_slot_mask=target_real_slot_mask,
        physical_edge_index=physical_edge_index,
        slot_readout_edge_index=slot_readout_edge_index,
        attention_edge_index=attention_edge_index,
        compact_to_full=compact_to_full,
        full_to_compact=full_to_compact,
        compact_edge_index=compact_edge_index,
        compact_bond_edge_index=compact_bond_edge_index,
        has_explicit_bonds=has_explicit_bonds,
        compact_attention_edge_index=compact_attention_edge_index,
        compact_pos=compact_pos,
        centered_full_pos_for_losses=centered_full_pos_for_losses,
    )
