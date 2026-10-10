"""
Mask index selection utilities for pretraining (random atom vs random walk on bonds).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

_warned_missing_bonds = False

COCONUT_DATASET_TYPES = frozenset({"coconut", "metabolite"})


def is_coconut_dataset_type(dataset_type: Optional[str]) -> bool:
    if dataset_type is None:
        return False
    return str(dataset_type).lower() in COCONUT_DATASET_TYPES


def mask_count(num_nodes: int, ratio: float) -> int:
    if num_nodes <= 0:
        return 0
    return max(1, int(num_nodes * ratio))


def build_adjacency(num_nodes: int, edge_index: torch.Tensor) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index is None or edge_index.numel() == 0:
        return adj
    row, col = edge_index[0], edge_index[1]
    for i, j in zip(row.tolist(), col.tolist()):
        if 0 <= i < num_nodes and 0 <= j < num_nodes and i != j:
            if j not in adj[i]:
                adj[i].append(j)
            if i not in adj[j]:
                adj[j].append(i)
    return adj


def select_random_atom_indices(
    num_nodes: int,
    ratio: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Independent random atom selection (existing behavior)."""
    k = mask_count(num_nodes, ratio)
    if k <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    rng_device = generator.device if generator is not None else device
    return torch.randperm(num_nodes, device=rng_device, generator=generator)[:k].to(device)


def select_random_walk_indices(
    num_nodes: int,
    bond_edge_index: Optional[torch.Tensor],
    ratio: float,
    num_walks: int = 1,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Grow a connected mask via random walk(s) on bond adjacency.

    Target size k = max(1, int(num_nodes * ratio)). Stays connected; does not add
    disconnected atoms if the largest component is smaller than k.
    """
    global _warned_missing_bonds
    if device is None:
        device = bond_edge_index.device if bond_edge_index is not None else torch.device("cpu")

    k = mask_count(num_nodes, ratio)
    if num_nodes <= 0 or k <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    if num_nodes == 1:
        return torch.tensor([0], dtype=torch.long, device=device)

    if bond_edge_index is None or bond_edge_index.numel() == 0:
        if not _warned_missing_bonds:
            logger.warning(
                "random_walk masking: missing bond_edge_index; falling back to random_atom"
            )
            _warned_missing_bonds = True
        return select_random_atom_indices(num_nodes, ratio, device, generator)

    adj = build_adjacency(num_nodes, bond_edge_index.cpu())
    nodes_with_neighbors = [i for i in range(num_nodes) if adj[i]]
    if not nodes_with_neighbors:
        if not _warned_missing_bonds:
            logger.warning(
                "random_walk masking: empty bond graph; falling back to random_atom"
            )
            _warned_missing_bonds = True
        return select_random_atom_indices(num_nodes, ratio, device, generator)

    gen = generator
    # Walk choices index Python adjacency lists. Draw on the generator's device,
    # then return the selected tensor on the requested node/index device.
    rng_device = gen.device if gen is not None else torch.device("cpu")
    start = nodes_with_neighbors[
        int(torch.randint(len(nodes_with_neighbors), (1,), device=rng_device, generator=gen).item())
    ]
    masked = {start}
    current = start

    walks_done = 0
    max_walks = max(1, int(num_walks))

    while len(masked) < k and walks_done < max_walks * num_nodes:
        unvisited_neighbors = [nb for nb in adj[current] if nb not in masked]
        if unvisited_neighbors:
            idx = int(
                torch.randint(len(unvisited_neighbors), (1,), device=rng_device, generator=gen).item()
            )
            current = unvisited_neighbors[idx]
            masked.add(current)
            continue

        frontier = []
        for node in masked:
            for nb in adj[node]:
                if nb not in masked:
                    frontier.append(nb)
        if frontier:
            current = frontier[int(torch.randint(len(frontier), (1,), device=rng_device, generator=gen).item())]
            masked.add(current)
            continue

        walks_done += 1
        if len(masked) >= k:
            break
        start = nodes_with_neighbors[
            int(torch.randint(len(nodes_with_neighbors), (1,), device=rng_device, generator=gen).item())
        ]
        if start in masked:
            for candidate in nodes_with_neighbors:
                if candidate not in masked:
                    start = candidate
                    break
        current = start
        masked.add(current)

    indices = sorted(masked)
    if len(indices) > k:
        perm = torch.randperm(len(indices), device=rng_device, generator=gen)
        indices = [indices[i] for i in perm[:k].tolist()]
        indices.sort()
    return torch.tensor(indices, dtype=torch.long, device=device)


def select_mask_indices(
    num_nodes: int,
    ratio: float,
    method: str,
    bond_edge_index: Optional[torch.Tensor] = None,
    dataset_type: Optional[str] = None,
    coconut_method: str = "random_walk",
    num_walks: int = 1,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Select node indices to mask.

    For coconut/metabolite graphs, uses coconut_method when method is not overridden.
    For other dataset types, always uses random_atom.
    """
    if device is None:
        if bond_edge_index is not None:
            device = bond_edge_index.device
        else:
            device = torch.device("cpu")

    effective_method = method
    if is_coconut_dataset_type(dataset_type):
        effective_method = coconut_method if coconut_method else method
    else:
        effective_method = "random_atom"

    if effective_method == "random_walk":
        return select_random_walk_indices(
            num_nodes,
            bond_edge_index,
            ratio,
            num_walks=num_walks,
            device=device,
            generator=generator,
        )
    return select_random_atom_indices(num_nodes, ratio, device, generator)


def slice_bond_edge_index(
    bond_edge_index: torch.Tensor,
    node_offset: int,
    num_nodes: int,
) -> torch.Tensor:
    """Extract bond edges for a single graph from a batched bond_edge_index."""
    if bond_edge_index is None or bond_edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    lo, hi = node_offset, node_offset + num_nodes
    mask = (bond_edge_index[0] >= lo) & (bond_edge_index[0] < hi) & (bond_edge_index[1] >= lo) & (bond_edge_index[1] < hi)
    sub = bond_edge_index[:, mask] - node_offset
    return sub.contiguous()


def get_graph_dataset_type(
    batch_dataset_type: Union[str, list, tuple, None],
    graph_idx: int,
) -> Optional[str]:
    if batch_dataset_type is None:
        return None
    if isinstance(batch_dataset_type, (list, tuple)):
        if graph_idx < len(batch_dataset_type):
            return batch_dataset_type[graph_idx]
        return None
    return str(batch_dataset_type)


def count_undirected_bond_edges(bond_edge_index: Optional[torch.Tensor]) -> int:
    """Count undirected bonds from bidirectional bond_edge_index."""
    if bond_edge_index is None or bond_edge_index.numel() == 0:
        return 0
    return int(bond_edge_index.size(1) // 2)


def masked_bond_connected_fraction(
    masked_local_indices: List[int],
    bond_edge_index: Optional[torch.Tensor],
) -> float:
    """
    Fraction of masked nodes in the largest bond-connected component.

    Uses bond adjacency restricted to masked nodes. Returns 0 if |mask| > 1 and no
    bond edges among masked nodes; 1.0 for a single masked node.
    """
    if not masked_local_indices:
        return 0.0
    if len(masked_local_indices) == 1:
        return 1.0

    masked_set = set(masked_local_indices)
    if bond_edge_index is None or bond_edge_index.numel() == 0:
        return 0.0

    adj: Dict[int, List[int]] = {i: [] for i in masked_set}
    row, col = bond_edge_index[0], bond_edge_index[1]
    for i, j in zip(row.tolist(), col.tolist()):
        if i in masked_set and j in masked_set and i != j:
            if j not in adj[i]:
                adj[i].append(j)
            if i not in adj[j]:
                adj[j].append(i)

    start = masked_local_indices[0]
    visited = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)
    return len(visited) / len(masked_set)


def compute_coconut_masking_metrics(batch, config) -> Optional[Dict[str, float]]:
    """
    Aggregate masking diagnostics for coconut/metabolite graphs in a training batch.

    Returns None if the batch has no coconut graphs or no MLM mask. Keys match the
    wandb ``masking/`` metric namespace.
    """
    if not hasattr(batch, "mlm_mask") or batch.mlm_mask is None:
        return None

    coconut_method = str(getattr(config, "coconut_masking_method", "random_atom")).lower()
    target_frac = float(getattr(config, "coconut_mask_ratio", 0.15))
    want_random_walk = coconut_method == "random_walk"

    bond_edge_index = getattr(batch, "bond_edge_index", None)
    batch_dataset_type = getattr(batch, "dataset_type", None)

    mask_fracs: List[float] = []
    connected_fracs: List[float] = []
    bond_edges: List[float] = []
    fallback_flags: List[float] = []

    def _process_graph(
        global_indices: torch.Tensor,
        local_mlm_mask: torch.Tensor,
        bond_sub: Optional[torch.Tensor],
        num_nodes: int,
    ) -> None:
        n_masked = int(local_mlm_mask.sum().item())
        if num_nodes <= 0:
            return
        mask_fracs.append(n_masked / num_nodes)
        masked_local = torch.where(local_mlm_mask)[0].tolist()
        connected_fracs.append(
            masked_bond_connected_fraction(masked_local, bond_sub)
        )
        bond_edges.append(float(count_undirected_bond_edges(bond_sub)))
        if want_random_walk and count_undirected_bond_edges(bond_sub) == 0:
            fallback_flags.append(1.0)
        else:
            fallback_flags.append(0.0)

    if hasattr(batch, "batch") and batch.batch is not None:
        for g in batch.batch.unique().tolist():
            dtype = get_graph_dataset_type(batch_dataset_type, int(g))
            if not is_coconut_dataset_type(dtype):
                continue
            node_mask = batch.batch == g
            global_indices = torch.where(node_mask)[0]
            n = global_indices.size(0)
            if n == 0:
                continue
            local_mlm = batch.mlm_mask[node_mask]
            node_offset = int(global_indices[0].item())
            bond_sub = None
            if bond_edge_index is not None and bond_edge_index.numel() > 0:
                bond_sub = slice_bond_edge_index(bond_edge_index, node_offset, n)
            _process_graph(global_indices, local_mlm, bond_sub, n)
    else:
        dtype = get_graph_dataset_type(batch_dataset_type, 0)
        if is_coconut_dataset_type(dtype):
            num_nodes = batch.mlm_mask.size(0)
            _process_graph(
                torch.arange(num_nodes, device=batch.mlm_mask.device),
                batch.mlm_mask,
                bond_edge_index,
                num_nodes,
            )

    if not mask_fracs:
        return None

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs)

    return {
        "mask_frac": _mean(mask_fracs),
        "mask_target_frac": target_frac,
        "mask_connected_frac": _mean(connected_fracs),
        "bond_edges_per_graph": _mean(bond_edges),
        "mask_fallback_rate": _mean(fallback_flags),
        "num_graphs": float(len(mask_fracs)),
    }
