"""
Protein-Specific Dynamic Random Edges Transform
================================================

This transform is specialized for proteins with the 14-atom uniform representation.
It adds random long-range edges at runtime, ensuring different edges each epoch.

Key Features:
- Protein-specific: Uses CA (alpha carbon) atoms only
- Inverse distance³ weighting (matches preprocessing logic)
- GPU-optimized: Vectorized operations
- Synchronized edge_attr and deduplication/coalesce support
- Minimal overhead: ~1-2ms per protein
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from typing import Optional, List, Tuple


class ProteinDynamicRandomEdges:
    """
    Add dynamic random long-range edges to protein graphs at runtime.
    
    This transform is designed for proteins using the 14-atom uniform representation.
    It replicates the Tier 3 logic from preprocessing but generates different edges
    each epoch to prevent bias.
    """
    
    def __init__(
        self,
        num_random_edges: int = 8,
        min_distance: float = 10.0,
        seed: Optional[int] = None,
        ca_atom_idx: int = 1,
        use_inverse_distance_weighting: bool = True,
        replace_existing_random_edges: bool = True,
    ):
        self.num_random_edges = num_random_edges
        self.min_distance = min_distance
        self.seed = seed
        self.ca_atom_idx = ca_atom_idx
        self.use_inverse_distance_weighting = use_inverse_distance_weighting
        self.replace_existing_random_edges = replace_existing_random_edges
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def __call__(self, data: Data) -> Data:
        """
        Add dynamic random edges to the protein Data object.
        """
        # Only process protein data
        dtype = getattr(data, 'dataset_type', None)
        if isinstance(dtype, (list, tuple)):
            dtype_str = str(dtype[0]).lower() if len(dtype) > 0 else ''
        elif isinstance(dtype, torch.Tensor):
            dtype_str = str(dtype[0].item()).lower() if dtype.numel() > 0 else ''
        else:
            dtype_str = str(dtype).lower() if dtype is not None else ''

        if dtype_str and dtype_str not in ('protein', 'pdb'):
            return data

        # Extract CA atom indices supporting production pos_code, template_anchor_order, or legacy position_indices
        if hasattr(data, 'pos_code') and data.pos_code is not None:
            pos_codes = data.pos_code
            if isinstance(pos_codes, torch.Tensor):
                pos_codes = pos_codes.tolist()
            ca_indices = torch.tensor(
                [i for i, code in enumerate(pos_codes) if str(code).strip() == 'CA'],
                dtype=torch.long,
                device=data.pos.device
            )
        elif hasattr(data, 'template_anchor_order') and data.template_anchor_order is not None:
            ca_indices = torch.where(data.template_anchor_order == 1)[0]
        elif hasattr(data, 'position_indices') and data.position_indices is not None:
            ca_indices = torch.where(data.position_indices == self.ca_atom_idx)[0]
        else:
            if dtype_str in ('protein', 'pdb'):
                raise ValueError(
                    f"Protein Data object missing position metadata ('pos_code', 'template_anchor_order', or 'position_indices'): mol_id={getattr(data, 'mol_id', '?')}"
                )
            return data

        if len(ca_indices) < 2:
            return data  # Too few CA atoms for random edges

        # Remove existing random edges (Tier 3) keeping edge_index and edge_attr in exact sync
        if self.replace_existing_random_edges:
            data.edge_index, data.edge_attr = self._filter_existing_edges(
                data.edge_index,
                getattr(data, 'edge_attr', None),
                data.pos,
                ca_indices,
                self.min_distance
            )

        # Generate random edges
        random_edges = self._generate_random_edges(
            positions=data.pos,
            ca_indices=ca_indices,
            existing_edges=data.edge_index
        )

        if random_edges.size(1) > 0:
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                row, col = random_edges
                diff = data.pos[row] - data.pos[col]
                new_edge_attr = torch.norm(diff, dim=1, keepdim=True)
                data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

            data.edge_index = torch.cat([data.edge_index, random_edges], dim=1)

        # Deduplicate / coalesce edge_index and edge_attr
        if data.edge_index.size(1) > 0:
            num_nodes = data.pos.size(0)
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_index, data.edge_attr = coalesce(
                    data.edge_index, data.edge_attr, num_nodes=num_nodes, reduce="mean"
                )
            else:
                data.edge_index = coalesce(data.edge_index, num_nodes=num_nodes)

        return data

    def _generate_random_edges(
        self,
        positions: torch.Tensor,
        ca_indices: torch.Tensor,
        existing_edges: torch.Tensor
    ) -> torch.Tensor:
        """Generate random long-range edges between CA atoms."""
        device = positions.device
        ca_positions = positions[ca_indices]
        num_ca = len(ca_indices)

        existing_edge_set = set()
        if existing_edges.size(1) > 0:
            src_edges, dst_edges = existing_edges[0].cpu().numpy(), existing_edges[1].cpu().numpy()
            for i in range(len(src_edges)):
                existing_edge_set.add((src_edges[i], dst_edges[i]))

        edges = []
        for i in range(num_ca):
            ca_idx = ca_indices[i].item()
            current_pos = ca_positions[i:i+1]
            distances = torch.norm(ca_positions - current_pos, dim=1)

            valid_mask = (distances >= self.min_distance) & (torch.arange(num_ca, device=device) != i)
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue

            valid_distances = distances[valid_indices]
            num_to_sample = min(self.num_random_edges, len(valid_indices))

            if self.use_inverse_distance_weighting:
                weights = 1.0 / (valid_distances ** 3 + 1e-6)
                weights = weights / weights.sum()
                sampled_idx = torch.multinomial(weights, num_to_sample, replacement=False)
            else:
                perm = torch.randperm(len(valid_indices), device=device)[:num_to_sample]
                sampled_idx = perm

            sampled_neighbors = valid_indices[sampled_idx]
            for neighbor_local_idx in sampled_neighbors:
                neighbor_global_idx = ca_indices[neighbor_local_idx].item()
                if (ca_idx, neighbor_global_idx) in existing_edge_set:
                    continue
                edges.append([ca_idx, neighbor_global_idx])
                edges.append([neighbor_global_idx, ca_idx])

        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        return torch.tensor(edges, dtype=torch.long, device=device).t()

    def _filter_existing_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        positions: torch.Tensor,
        ca_indices: torch.Tensor,
        min_distance: float
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Remove existing long-range CA-CA edges from both edge_index and edge_attr."""
        if edge_index.size(1) == 0:
            return edge_index, edge_attr

        device = edge_index.device
        num_nodes = positions.size(0)
        is_ca = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        is_ca[ca_indices] = True

        src, dst = edge_index[0], edge_index[1]
        both_ca = is_ca[src] & is_ca[dst]

        dists = torch.norm(positions[src] - positions[dst], dim=-1)
        keep_mask = ~(both_ca & (dists >= min_distance))

        filtered_edge_index = edge_index[:, keep_mask]
        filtered_edge_attr = edge_attr[keep_mask] if edge_attr is not None else None
        return filtered_edge_index, filtered_edge_attr

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_random_edges={self.num_random_edges}, "
            f"min_distance={self.min_distance}, "
            f"seed={self.seed}, "
            f"ca_atom_idx={self.ca_atom_idx}, "
            f"use_inverse_distance_weighting={self.use_inverse_distance_weighting}, "
            f"replace_existing_random_edges={self.replace_existing_random_edges})"
        )


class OptimizedProteinDynamicRandomEdges(ProteinDynamicRandomEdges):
    """
    GPU-optimized version with vectorized operations.
    """
    
    def _generate_random_edges(
        self,
        positions: torch.Tensor,
        ca_indices: torch.Tensor,
        existing_edges: torch.Tensor
    ) -> torch.Tensor:
        """GPU-optimized random edge generation using vectorized operations."""
        device = positions.device
        ca_positions = positions[ca_indices]
        num_ca = len(ca_indices)

        delta_pos = ca_positions.unsqueeze(0) - ca_positions.unsqueeze(1)
        dist_matrix = torch.norm(delta_pos, dim=-1)

        candidate_mask = torch.ones_like(dist_matrix, dtype=torch.bool)
        candidate_mask.fill_diagonal_(False)
        candidate_mask &= (dist_matrix >= self.min_distance)

        edges = []
        for i in range(num_ca):
            ca_idx = ca_indices[i].item()
            candidates = torch.where(candidate_mask[i])[0]
            if len(candidates) == 0:
                continue

            num_to_sample = min(self.num_random_edges, len(candidates))
            if self.use_inverse_distance_weighting:
                valid_distances = dist_matrix[i, candidates]
                weights = 1.0 / (valid_distances ** 3 + 1e-6)
                weights = weights / weights.sum()
                sampled_idx = torch.multinomial(weights, num_to_sample, replacement=False)
            else:
                perm = torch.randperm(len(candidates), device=device)[:num_to_sample]
                sampled_idx = perm

            sampled_neighbors = candidates[sampled_idx]
            for neighbor_local_idx in sampled_neighbors:
                neighbor_global_idx = ca_indices[neighbor_local_idx].item()
                edges.append([ca_idx, neighbor_global_idx])
                edges.append([neighbor_global_idx, ca_idx])

        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        return torch.tensor(edges, dtype=torch.long, device=device).t()


def add_protein_dynamic_random_edges(
    data: Data,
    num_random_edges: int = 8,
    min_distance: float = 10.0,
    use_gpu_optimized: bool = True
) -> Data:
    """Convenience function to add dynamic random edges to a protein Data object."""
    if use_gpu_optimized:
        transform = OptimizedProteinDynamicRandomEdges(
            num_random_edges=num_random_edges,
            min_distance=min_distance
        )
    else:
        transform = ProteinDynamicRandomEdges(
            num_random_edges=num_random_edges,
            min_distance=min_distance
        )
    return transform(data)
