"""Geometry hiding for masked UniMol atoms (G1, "candidate C").

Why: UniMol masks single atoms (token 0) but leaves their position, all their distances and the
distance-derived features visible, so the element is read from the geometry. A frozen checkpoint
reaches 99.9 % masked-element accuracy and drops to 34 % when the position is moved by 1 A
(core/counterfactual_eval.py, outputs/eval/g1_cf_ep330/).

What the encoder and ESA see of a masked UniMol atom after this view:

* no position: the encoder position of a masked atom is the constant 0 (the visible atoms are
  centred on their own centroid, so the masked coordinates do not even enter the centre);
* no radius edge: every edge of the stored radius graph at a masked atom is removed from the
  encoder, so no RBF, no 3 A / 5 A neighbour count and no min/mean distance of a masked atom, and
  none of its neighbours, contains a distance to it;
* binary bond topology only: the bonds of the molecule are ``0.5 < d <= bond_threshold_angstrom``
  on the clean geometry (a distance proxy, the UniMol chunks store no bond list). They reach the
  encoder as an explicit ``bond_edge_index`` (chemical coordination = bond degree) and they
  replace the radius edges at masked atoms in ESA's graph-local attention. A bond carries no
  distance value.

Visible-visible edges, distances and features are unchanged (they are legitimate context, e.g. the
angle between two neighbours of the masked atom). The threshold is one number per run: it is never
chosen from the masked atom's element, and the function has no argument that could carry one.
Protein, RNA and COCONUT graphs pass through untouched (their 1.85 A proxy / explicit bonds stay).
Clean positions remain in ``batch.pos`` for the distance losses; targets and masks are untouched.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from core.metrics.domain_mlm_metrics import canonical_domain
from core.pretraining_config import validate_unimol_bond_threshold
from data_loading.bond_contract import validate_graph_edges

TARGET_DOMAIN = "unimol"
# Same lower cut as the encoder's distance proxy (core/molecular_encoder.py): coincident atoms are no bond.
MIN_BOND_DISTANCE = 0.5


@dataclass(frozen=True)
class HiddenUniMolView:
    """Encoder/ESA inputs of one batch after hiding; all indices are those of the input tensors."""

    pos: torch.Tensor
    edge_index: torch.Tensor
    attention_edge_index: torch.Tensor
    bond_edge_index: torch.Tensor
    has_explicit_bonds: torch.Tensor
    hidden_graphs: torch.Tensor  # UniMol graphs handled by this view (their bond source is the proxy list)
    hidden_nodes: torch.Tensor  # masked atoms of those graphs
    stats: dict


def proxy_bond_edges(
    pos: torch.Tensor, edge_index: torch.Tensor, threshold: float, node_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Symmetric, duplicate-free ``[2, B]`` bond list: edges with ``0.5 < d <= threshold``.

    ``node_mask`` (optional) keeps only bonds whose source atom is in the mask. Pure geometry: no
    element enters, so the bond decision cannot depend on what a masked atom is.
    """
    threshold = validate_unimol_bond_threshold(threshold)
    if edge_index.numel() == 0:
        return edge_index.new_empty((2, 0))
    distance = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=1)
    keep = (distance > MIN_BOND_DISTANCE) & (distance <= threshold)
    if node_mask is not None:
        keep = keep & node_mask[edge_index[0]]
    bonds = edge_index[:, keep]
    num_nodes = int(pos.size(0))
    low = torch.minimum(bonds[0], bonds[1])
    high = torch.maximum(bonds[0], bonds[1])
    keys = torch.unique(low * num_nodes + high)
    low, high = keys // num_nodes, keys % num_nodes
    return torch.stack((torch.cat((low, high)), torch.cat((high, low))))


def _check(name: str, condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"{name}: {message}")


def hide_masked_unimol_geometry(
    *,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    attention_edge_index: torch.Tensor,
    batch_mapping: torch.Tensor,
    mlm_mask: torch.Tensor,
    graph_domains: Sequence[str],
    bond_edge_index: torch.Tensor,
    has_explicit_bonds: torch.Tensor,
    bond_threshold_angstrom: float,
    mol_ids: Optional[Sequence[str]] = None,
) -> HiddenUniMolView:
    """Build the hidden view of the masked atoms of every UniMol graph (see the module docstring).

    Inputs are encoder-level tensors (the compact view when a geometry-neutral graph view exists).
    ``bond_edge_index``/``has_explicit_bonds`` are the bond source the encoder would otherwise use;
    UniMol graphs must not have one (the proxy list replaces it), other graphs keep theirs.
    Inputs are never modified in place.
    """
    threshold = validate_unimol_bond_threshold(bond_threshold_angstrom)
    _check("pos", pos.ndim == 2 and pos.size(1) == 3 and torch.is_floating_point(pos),
           "pos must be a floating tensor of shape [N, 3]")
    num_nodes = int(pos.size(0))
    _check("batch_mapping", batch_mapping.ndim == 1 and batch_mapping.numel() == num_nodes
           and batch_mapping.dtype == torch.long, "batch_mapping must be a torch.long tensor with one graph id per node")
    _check("mlm_mask", mlm_mask.dtype == torch.bool and mlm_mask.shape == (num_nodes,),
           "mlm_mask must be a torch.bool tensor with one value per node")
    num_graphs = len(graph_domains)
    _check("graph_domains", isinstance(has_explicit_bonds, torch.Tensor) and has_explicit_bonds.dtype == torch.bool
           and has_explicit_bonds.shape == (num_graphs,),
           f"graph_domains has {num_graphs} entries but has_explicit_bonds is "
           f"{tuple(getattr(has_explicit_bonds, 'shape', ()))}; one value per graph is required")
    if num_nodes:
        _check("batch_mapping", int(batch_mapping.min()) >= 0 and int(batch_mapping.max()) < num_graphs,
               f"graph ids must lie in [0, {num_graphs})")
    if mol_ids is not None:
        _check("mol_ids", len(mol_ids) == num_graphs, f"mol_ids has {len(mol_ids)} entries for {num_graphs} graphs")
    validate_graph_edges(edge_index, batch_mapping, name="edge_index")
    validate_graph_edges(attention_edge_index, batch_mapping, name="attention_edge_index")
    validate_graph_edges(bond_edge_index, batch_mapping, name="bond_edge_index")
    if bond_edge_index.numel() and not bool(has_explicit_bonds[batch_mapping[bond_edge_index[0]]].all()):
        raise ValueError("has_explicit_bonds is false for a graph with explicit bonds")

    domains = [canonical_domain(domain) for domain in graph_domains]
    hidden_graphs = torch.tensor([domain == TARGET_DOMAIN for domain in domains], dtype=torch.bool, device=pos.device)
    for graph in torch.where(hidden_graphs & has_explicit_bonds)[0].tolist():
        raise ValueError(
            "unimol_masked_geometry_hiding is defined for graphs without explicit bonds (the distance proxy "
            f"replaces them): mol_id={mol_ids[graph] if mol_ids is not None else '?'} graph={graph} domain={TARGET_DOMAIN}"
        )

    target_node = hidden_graphs[batch_mapping]
    hidden_nodes = mlm_mask & target_node
    if not bool(hidden_graphs.any()):
        return HiddenUniMolView(pos, edge_index, attention_edge_index, bond_edge_index, has_explicit_bonds,
                                hidden_graphs, hidden_nodes, {"hidden_atoms": 0, "graphs": 0})

    # Binary topology from the clean geometry (the only thing the masked atom keeps).
    proxy = proxy_bond_edges(pos, edge_index, threshold, node_mask=target_node)
    at_masked = hidden_nodes[proxy[0]] | hidden_nodes[proxy[1]]

    new_edges = edge_index[:, ~(hidden_nodes[edge_index[0]] | hidden_nodes[edge_index[1]])]
    kept_attention = attention_edge_index[
        :, ~(hidden_nodes[attention_edge_index[0]] | hidden_nodes[attention_edge_index[1]])]
    new_attention = torch.cat((kept_attention, proxy[:, at_masked]), dim=1)

    # Encoder positions: visible atoms centred on their own centroid; masked atoms hold the constant 0.
    visible = target_node & ~hidden_nodes
    count = torch.zeros(num_graphs, device=pos.device, dtype=pos.dtype).index_add_(
        0, batch_mapping[visible], torch.ones(int(visible.sum()), device=pos.device, dtype=pos.dtype))
    total = torch.zeros(num_graphs, 3, device=pos.device, dtype=pos.dtype).index_add_(
        0, batch_mapping[visible], pos[visible])
    centre = total / count.clamp(min=1).unsqueeze(1)
    new_pos = pos.clone()
    new_pos[target_node] = pos[target_node] - centre[batch_mapping[target_node]]
    new_pos[hidden_nodes] = 0.0

    bonds = torch.cat((bond_edge_index, proxy), dim=1)
    stats = {
        "graphs": int(hidden_graphs.sum()),
        "hidden_atoms": int(hidden_nodes.sum()),
        "encoder_edges_removed": int(edge_index.size(1) - new_edges.size(1)),
        "attention_edges_removed": int(attention_edge_index.size(1) - kept_attention.size(1)),
        "attention_bond_edges_added": int(at_masked.sum()),
        "bond_edges": int(proxy.size(1)),
    }
    return HiddenUniMolView(
        pos=new_pos,
        edge_index=new_edges,
        attention_edge_index=new_attention,
        bond_edge_index=bonds,
        has_explicit_bonds=has_explicit_bonds | hidden_graphs,
        hidden_graphs=hidden_graphs,
        hidden_nodes=hidden_nodes,
        stats=stats,
    )
