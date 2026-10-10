"""Runtime explicit-bond availability and graph-local edge contracts."""

import torch


def validate_graph_edges(edge_index, graph_ids, *, name="edge_index"):
    """Validate before indexing; negative endpoints must never wrap silently."""
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"{name} must have shape [2, E]")
    if edge_index.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long")
    if edge_index.device != graph_ids.device:
        raise ValueError(f"{name} must be on the same device as graph ids")
    if edge_index.numel():
        if bool(((edge_index < 0) | (edge_index >= graph_ids.numel())).any()):
            raise ValueError(f"{name} endpoint out of range [0, {graph_ids.numel()})")
        if bool((graph_ids[edge_index[0]] != graph_ids[edge_index[1]]).any()):
            raise ValueError(f"{name} contains a cross-graph edge")


def resolve_bond_sources(batch, graph_ids, num_graphs):
    """Return validated bonds and one explicit-source boolean per graph.

    LazyUniversalDataset produces ``has_explicit_bonds`` before collation or
    node removal. Legacy absent/empty lists mean unavailable: the old producer
    also wrote empty tensors when it had no bond information. For direct callers
    without the runtime field, infer availability from the full bond endpoints.
    An explicit source stays explicit if compact filtering removes every bond.
    """
    if graph_ids.ndim != 1 or graph_ids.dtype != torch.long:
        raise ValueError("batch graph ids must be a one-dimensional torch.long tensor")
    if num_graphs < 1 or bool(((graph_ids < 0) | (graph_ids >= num_graphs)).any()):
        raise ValueError("batch graph ids must be in range [0, num_graphs)")
    bonds = getattr(batch, "bond_edge_index", None)
    availability = getattr(batch, "has_explicit_bonds", None)
    if availability is not None:
        if not isinstance(availability, torch.Tensor) or availability.dtype != torch.bool:
            raise TypeError("has_explicit_bonds must be a torch.bool tensor")
        if availability.ndim != 1 or availability.numel() != num_graphs:
            raise ValueError("has_explicit_bonds must contain one value per graph")
        if availability.device != graph_ids.device:
            raise ValueError("has_explicit_bonds must be on the same device as graph ids")
    if bonds is None:
        if availability is not None and bool(availability.any()):
            raise ValueError("has_explicit_bonds requires a bond_edge_index tensor (possibly empty)")
        bonds = torch.empty((2, 0), dtype=torch.long, device=graph_ids.device)
    validate_graph_edges(bonds, graph_ids, name="bond_edge_index")

    if availability is None:
        availability = torch.zeros(num_graphs, dtype=torch.bool, device=graph_ids.device)
        availability[graph_ids[bonds[0]]] = True
    elif bonds.numel() and not bool(availability[graph_ids[bonds[0]]].all()):
        raise ValueError("has_explicit_bonds is false for a graph with explicit bonds")
    return bonds, availability
