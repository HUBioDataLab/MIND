"""Counterfactual corruption of masked UniMol atoms for offline evaluation.

The UniMol MLM target (element of a masked atom) is solvable from the atom's own
position, its distances and its ESA neighbourhood, none of which the mask hides.
These modes remove or scramble exactly that information for the masked atoms of
a frozen checkpoint, at evaluation time only, to measure where the masked-atom
accuracy falls. They are not training features and are not a masking strategy.

What each mode changes (``masked`` means the atoms selected by the MLM mask of a
UniMol graph; every other atom, every protein/RNA graph and the clean
``batch.pos`` used by the distance losses stay untouched):

``masked_pos_noise``     encoder position of a masked atom += N(0, sigma^2 I) (Angstrom).
                         Edges stay those of the clean geometry, distances are recomputed.
``masked_pos_random``    encoder position of a masked atom := a random point drawn from
                         N(centroid, diag(std^2)) of the graph's visible atoms.
``masked_dist_shuffle``  each masked atom's clean distances to its neighbours are permuted
                         among those neighbours (its distance multiset is kept), which
                         changes RBF and distance-derived features; an edge between two
                         masked atoms carries its source's draw in each direction.
                         Positions and topology are unchanged.
``esa_edges_remove``     the ESA graph-local attention edges at masked atoms are deleted
                         (the encoder still sees their geometry).
``esa_edges_shuffle``    the ESA attention partners of a masked atom are replaced by random
                         atoms of the same graph, keeping the number of partners.
``masked_edges_remove``  edges at masked atoms are deleted for the encoder and ESA: the
                         atom keeps only its mask token and the molecule-global S layers.

All draws come from per-graph generators seeded by the graph's ``eval_mask_seed``
and the mode, so a graph gets the same corruption regardless of its batch, its
neighbours in the batch, the checkpoint or the global RNG.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from core.metrics.domain_mlm_metrics import canonical_domain

COUNTERFACTUAL_MODES = (
    "none",
    "masked_pos_noise",
    "masked_pos_random",
    "masked_dist_shuffle",
    "esa_edges_remove",
    "esa_edges_shuffle",
    "masked_edges_remove",
)
# Revision of a mode whose definition changed. It is written to the provenance so that
# results of different definitions never share a protocol hash. Unlisted modes record none.
# masked_dist_shuffle 2: every masked atom keeps its distance multiset, also next to another
# masked atom (revision 1 kept it only for the lower-indexed atom of a masked-masked pair).
MODE_REVISIONS = {"masked_dist_shuffle": 2}
SUPPORTED_DOMAINS = ("unimol",)
DEFAULT_NOISE_SIGMA = 1.0


def validate_mode(mode) -> str:
    if mode not in COUNTERFACTUAL_MODES:
        raise ValueError(
            f"unknown counterfactual mode {mode!r}; valid modes: {', '.join(COUNTERFACTUAL_MODES)}"
        )
    return mode


@dataclass
class CounterfactualSettings:
    """Validated evaluation-only corruption request; ``stats`` accumulates what was applied."""

    mode: str
    sigma: Optional[float] = None
    domains: Tuple[str, ...] = SUPPORTED_DOMAINS
    stats: Counter = field(default_factory=Counter, compare=False)


def build_counterfactual_settings(
    mode, sigma: Optional[float] = None, domains: Sequence[str] = SUPPORTED_DOMAINS
) -> Optional[CounterfactualSettings]:
    """Return settings for ``mode`` or ``None`` for ``"none"`` (the untouched legacy path)."""
    validate_mode(mode)
    if mode == "none":
        if sigma is not None:
            raise ValueError("counterfactual sigma is only valid with mode 'masked_pos_noise', not 'none'")
        return None
    if mode == "masked_pos_noise":
        if sigma is None:
            raise ValueError("counterfactual mode 'masked_pos_noise' requires a sigma (Angstrom)")
        if not isinstance(sigma, (int, float)) or isinstance(sigma, bool) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"counterfactual sigma must be a finite positive number, got {sigma!r}")
        sigma = float(sigma)
    elif sigma is not None:
        raise ValueError(f"counterfactual sigma is only valid with mode 'masked_pos_noise', not {mode!r}")
    canonical = tuple(canonical_domain(name) for name in domains)
    if not canonical or any(name not in SUPPORTED_DOMAINS for name in canonical):
        raise ValueError(
            f"counterfactual domains must be a non-empty subset of {SUPPORTED_DOMAINS}, got {tuple(domains)!r}"
        )
    return CounterfactualSettings(mode=mode, sigma=sigma, domains=canonical)


@dataclass
class CounterfactualInputs:
    pos: torch.Tensor
    edge_index: torch.Tensor
    attention_edge_index: torch.Tensor
    edge_distances: Optional[torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generator(seed: int, mode: str) -> torch.Generator:
    digest = hashlib.sha256(f"mind_counterfactual\x1f{mode}\x1f{int(seed)}".encode("utf-8")).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
    return generator


def _check_edges(edge_index: torch.Tensor, batch_mapping: torch.Tensor, name: str) -> None:
    if edge_index.ndim != 2 or edge_index.size(0) != 2 or edge_index.dtype != torch.long:
        raise ValueError(f"{name} must be a torch.long tensor of shape [2, E]")
    if edge_index.numel() == 0:
        return
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= batch_mapping.numel():
        raise ValueError(f"{name} has an edge endpoint outside the node range")
    if bool((batch_mapping[edge_index[0]] != batch_mapping[edge_index[1]]).any()):
        raise ValueError(f"{name} contains a cross-graph edge")


def _validate(settings, pos, edge_index, attention_edge_index, batch_mapping, mlm_mask, graph_domains, seeds):
    if pos.ndim != 2 or pos.size(1) != 3:
        raise ValueError("pos must have shape [N, 3]")
    num_nodes = pos.size(0)
    if batch_mapping.ndim != 1 or batch_mapping.numel() != num_nodes or batch_mapping.dtype != torch.long:
        raise ValueError("batch_mapping must be a torch.long tensor with one graph id per node")
    if mlm_mask.dtype != torch.bool or mlm_mask.shape != (num_nodes,):
        raise ValueError("mlm_mask must be a torch.bool tensor with one value per node")
    if num_nodes and (int(batch_mapping.min()) < 0 or bool((batch_mapping[1:] < batch_mapping[:-1]).any())):
        raise ValueError("batch_mapping must be sorted (nodes of a graph contiguous and ascending)")
    num_graphs = int(batch_mapping.max()) + 1 if num_nodes else 0
    if len(graph_domains) != num_graphs:
        raise ValueError(f"graph_domains has {len(graph_domains)} entries for {num_graphs} graphs")
    if len(seeds) != num_graphs:
        raise ValueError(
            f"eval_mask_seeds has {len(seeds)} entries for {num_graphs} graphs; "
            "the counterfactual draws need one seed per graph"
        )
    _check_edges(edge_index, batch_mapping, "edge_index")
    _check_edges(attention_edge_index, batch_mapping, "attention_edge_index")


def _incident_mask(edge_index: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    return nodes[edge_index[0]] | nodes[edge_index[1]]


def _graph_blocks(nodes_sorted: torch.Tensor, batch_mapping: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
    """Split ascending node ids into per-graph chunks: [(graph id, node ids)]."""
    if nodes_sorted.numel() == 0:
        return []
    graphs = batch_mapping[nodes_sorted]
    unique, counts = torch.unique_consecutive(graphs, return_counts=True)
    return list(zip(unique.tolist(), torch.split(nodes_sorted, counts.tolist())))


def _undirected_pairs(edge_index: torch.Tensor, num_nodes: int):
    """Return (pair id per directed edge, number of undirected pairs, (lo, hi) per pair)."""
    lo = torch.minimum(edge_index[0], edge_index[1])
    hi = torch.maximum(edge_index[0], edge_index[1])
    key = lo * num_nodes + hi
    unique_key, inverse = torch.unique(key, return_inverse=True)
    return inverse, unique_key.numel(), (unique_key // num_nodes, unique_key % num_nodes)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _corrupt_positions(settings, pos, batch_mapping, mlm_target, seeds):
    new_pos = pos.clone()
    stats = settings.stats
    visible = ~mlm_target
    masked_ids = torch.where(mlm_target)[0]
    if settings.mode == "masked_pos_random":
        num_graphs = int(batch_mapping.max()) + 1
        vis_count = torch.zeros(num_graphs, device=pos.device).index_add_(
            0, batch_mapping[visible], torch.ones(int(visible.sum()), device=pos.device))
        vis_sum = torch.zeros(num_graphs, 3, device=pos.device, dtype=pos.dtype).index_add_(
            0, batch_mapping[visible], pos[visible])
        centroid = vis_sum / vis_count.clamp(min=1).unsqueeze(1)
        deviation = (pos - centroid[batch_mapping]) ** 2
        var_sum = torch.zeros(num_graphs, 3, device=pos.device, dtype=pos.dtype).index_add_(
            0, batch_mapping[visible], deviation[visible])
        std = (var_sum / vis_count.clamp(min=1).unsqueeze(1)).sqrt()
    for graph, nodes in _graph_blocks(masked_ids, batch_mapping):
        generator = _generator(seeds[graph], settings.mode)
        draw = torch.randn(nodes.numel(), 3, generator=generator).to(device=pos.device, dtype=pos.dtype)
        if settings.mode == "masked_pos_noise":
            new_pos[nodes] = pos[nodes] + settings.sigma * draw
        else:
            if float(vis_count[graph]) == 0:
                if int((batch_mapping == graph).sum()) > 1:
                    raise ValueError(
                        f"graph {graph}: masked_pos_random needs a visible atom to draw from, "
                        "but every atom of this multi-atom graph is masked"
                    )
                stats["graphs_without_visible_atoms"] += 1  # a single atom has no neighbours to disturb
                continue
            new_pos[nodes] = centroid[graph] + std[graph] * draw
        stats["atoms_moved"] += int(nodes.numel())
    return new_pos


def _shuffle_distances(settings, pos, edge_index, batch_mapping, mlm_target, seeds):
    """Permute, for each masked atom, the clean distances to its neighbours among those neighbours.

    Each masked atom draws its own permutation over its incident pairs, so it keeps the
    multiset of its distances. A directed edge takes the value drawn by its source when the
    source is masked, otherwise the value drawn by its masked target (a visible atom sees
    the same value as the masked atom). An edge between two masked atoms therefore carries,
    in each direction, the value drawn by its source.
    """
    num_nodes = pos.size(0)
    distances = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=1)
    if edge_index.size(1) == 0:
        return distances
    pair_of_edge, num_pairs, (lo, hi) = _undirected_pairs(edge_index, num_nodes)
    pair_distance = torch.zeros(num_pairs, device=pos.device, dtype=distances.dtype)
    pair_distance[pair_of_edge] = distances  # both directions carry the same clean distance
    lo_slots = torch.where(mlm_target[lo])[0]
    hi_slots = torch.where(mlm_target[hi] & (hi != lo))[0]
    if lo_slots.numel() + hi_slots.numel() == 0:
        return distances
    # one slot per (masked atom, incident pair): a pair between two masked atoms has two
    slot_pair = torch.cat((lo_slots, hi_slots))
    owner = torch.cat((lo[lo_slots], hi[hi_slots]))
    canonical = torch.argsort(owner * num_pairs + slot_pair)  # by (owner, pair), batch-independent per graph
    slot_pair, owner = slot_pair[canonical], owner[canonical]
    slot_graph = batch_mapping[owner]
    random_key = torch.zeros(owner.numel(), dtype=torch.double)
    for graph in torch.unique(slot_graph).tolist():
        rows = torch.where(slot_graph == graph)[0].cpu()
        generator = _generator(seeds[graph], settings.mode)
        random_key[rows] = torch.rand(rows.numel(), generator=generator, dtype=torch.double)
    random_key = random_key.to(pos.device)
    by_random = torch.argsort(random_key)
    by_random = by_random[torch.sort(owner[by_random], stable=True).indices]
    # slots are grouped by owner and by_random lists the same groups in the same order
    slot_value = pair_distance[slot_pair[by_random]]
    from_lo = owner == lo[slot_pair]
    lo_value, hi_value = pair_distance.clone(), pair_distance.clone()
    lo_value[slot_pair[from_lo]] = slot_value[from_lo]
    hi_value[slot_pair[~from_lo]] = slot_value[~from_lo]
    src, dst = edge_index[0], edge_index[1]
    lo_of_edge = lo[pair_of_edge]
    src_value = torch.where(src == lo_of_edge, lo_value[pair_of_edge], hi_value[pair_of_edge])
    dst_value = torch.where(dst == lo_of_edge, lo_value[pair_of_edge], hi_value[pair_of_edge])
    settings.stats["distances_shuffled"] += int(owner.numel())
    return torch.where(mlm_target[src], src_value, torch.where(mlm_target[dst], dst_value, distances))


def _remove_incident(edge_index, mlm_target):
    keep = ~_incident_mask(edge_index, mlm_target) if edge_index.size(1) else torch.zeros(0, dtype=torch.bool,
                                                                                         device=edge_index.device)
    return edge_index[:, keep], int(edge_index.size(1) - int(keep.sum()))


def _rewire(settings, edge_index, batch_mapping, mlm_target, seeds):
    """Return (edges, removed count). Replace the attention partners of each masked atom by random atoms of its graph."""
    num_nodes = batch_mapping.numel()
    kept, removed = _remove_incident(edge_index, mlm_target)
    if removed == 0:
        return edge_index, 0
    pair_of_edge, num_pairs, (lo, hi) = _undirected_pairs(edge_index, num_nodes)
    touched = torch.where(mlm_target[lo] | mlm_target[hi])[0]
    degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    degree.index_add_(0, lo[touched], mlm_target[lo[touched]].long())
    degree.index_add_(0, hi[touched], mlm_target[hi[touched]].long())
    cpu_degree = degree.cpu()
    graph_start = torch.searchsorted(batch_mapping, torch.arange(int(batch_mapping.max()) + 2,
                                                                 device=batch_mapping.device)).cpu()
    sources, targets = [], []
    for graph, nodes in _graph_blocks(torch.where(mlm_target)[0], batch_mapping):
        generator = _generator(seeds[graph], settings.mode)
        start, end = int(graph_start[graph]), int(graph_start[graph + 1])
        for node in nodes.tolist():
            wanted = min(int(cpu_degree[node]), end - start - 1)
            if wanted <= 0:
                continue
            scores = torch.rand(end - start, generator=generator)
            scores[node - start] = float("inf")
            partners = torch.topk(scores, wanted, largest=False).indices + start
            sources.append(torch.full((wanted,), node, dtype=torch.long))
            targets.append(partners)
    if not sources:
        return kept, removed
    forward = torch.stack((torch.cat(sources), torch.cat(targets))).to(edge_index.device)
    added = torch.unique(torch.cat((forward, forward.flip(0)), dim=1), dim=1)
    settings.stats["attention_edges_added"] += int(added.size(1))
    return torch.cat((kept, added), dim=1), removed


def apply_counterfactual(
    settings: CounterfactualSettings,
    *,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    attention_edge_index: torch.Tensor,
    batch_mapping: torch.Tensor,
    mlm_mask: torch.Tensor,
    graph_domains: Sequence[str],
    eval_mask_seeds: Sequence[int],
) -> CounterfactualInputs:
    """Corrupt the encoder/ESA inputs of masked atoms of the target-domain graphs.

    ``pos``/``edge_index`` are what the encoder receives, ``attention_edge_index``
    what ESA's graph-local layers receive (the same content for UniMol, possibly
    another tensor). Inputs are never modified in place.
    """
    if not isinstance(settings, CounterfactualSettings):
        raise TypeError("apply_counterfactual needs CounterfactualSettings (mode 'none' has no settings)")
    _validate(settings, pos, edge_index, attention_edge_index, batch_mapping, mlm_mask,
              list(graph_domains), list(eval_mask_seeds))
    seeds = [int(s) for s in eval_mask_seeds]
    domains = [canonical_domain(d) for d in graph_domains]
    target_graph = torch.tensor([d in settings.domains for d in domains], dtype=torch.bool, device=pos.device)
    mlm_target = mlm_mask.to(pos.device) & target_graph[batch_mapping]
    stats = settings.stats
    stats["batches_seen"] += 1  # includes batches without any target-domain graph (pass-through)
    if mlm_target.any():
        stats["batches"] += 1
        stats["masked_atoms"] += int(mlm_target.sum())
        stats["graphs"] += int(torch.unique(batch_mapping[mlm_target]).numel())

    new_pos, new_edges, new_attention, distances = pos, edge_index, attention_edge_index, None
    if mlm_target.any():
        mode = settings.mode
        if mode in ("masked_pos_noise", "masked_pos_random"):
            new_pos = _corrupt_positions(settings, pos, batch_mapping, mlm_target, seeds)
        elif mode == "masked_dist_shuffle":
            distances = _shuffle_distances(settings, pos, edge_index, batch_mapping, mlm_target, seeds)
        elif mode == "esa_edges_remove":
            new_attention, removed = _remove_incident(attention_edge_index, mlm_target)
            stats["attention_edges_removed"] += removed
        elif mode == "esa_edges_shuffle":
            new_attention, removed = _rewire(settings, attention_edge_index, batch_mapping, mlm_target, seeds)
            stats["attention_edges_removed"] += removed
        elif mode == "masked_edges_remove":
            new_edges, removed_encoder = _remove_incident(edge_index, mlm_target)
            new_attention, removed = _remove_incident(attention_edge_index, mlm_target)
            stats["encoder_edges_removed"] += removed_encoder
            stats["attention_edges_removed"] += removed
        else:  # pragma: no cover - guarded by build_counterfactual_settings
            raise ValueError(f"unknown counterfactual mode {mode!r}")
    return CounterfactualInputs(new_pos, new_edges, new_attention, distances)
