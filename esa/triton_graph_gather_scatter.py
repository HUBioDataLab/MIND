#!/usr/bin/env python3
"""
Custom OpenAI Triton Fused Graph Gather-Scatter Kernel for MIND Model.

Fuses PyTorch `aten::index` (X[src]) and `scatter_add_` (Y[dst] += X[src]) into a SINGLE GPU kernel.
Eliminates intermediate edge feature VRAM allocations and reduces VRAM roundtrips from 4x down to 1x!
"""

import torch
import triton
import triton.language as tl


# =====================================================================
# 🚀 Vectorized Triton Fused Gather-Scatter Kernel
# =====================================================================
@triton.jit
def _fused_gather_scatter_kernel(
    x_ptr,           # Features input (num_nodes or num_edges, hidden_dim)
    src_indices_ptr, # Source node indices (num_edges,)
    dst_indices_ptr, # Target node indices (num_edges,)
    out_ptr,         # Accumulated output (num_nodes, hidden_dim)
    num_edges,
    hidden_dim,
    stride_x_first, stride_x_dim,
    stride_out_node, stride_out_dim,
    is_edge_feature: tl.constexpr,
    BLOCK_EDGES: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    edge_start = pid * BLOCK_EDGES
    edge_offsets = edge_start + tl.arange(0, BLOCK_EDGES)
    edge_mask = edge_offsets < num_edges

    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < hidden_dim

    for e_idx in range(BLOCK_EDGES):
        curr_edge = edge_start + e_idx
        if curr_edge < num_edges:
            src_node = tl.load(src_indices_ptr + curr_edge)

            if is_edge_feature:
                x_ptrs = x_ptr + curr_edge * stride_x_first + dim_offsets * stride_x_dim
            else:
                x_ptrs = x_ptr + src_node * stride_x_first + dim_offsets * stride_x_dim

            x_vals = tl.load(x_ptrs, mask=dim_mask, other=0.0)

            out_ptrs = out_ptr + src_node * stride_out_node + dim_offsets * stride_out_dim
            tl.atomic_add(out_ptrs, x_vals, mask=dim_mask)


def triton_fused_gather_scatter(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
    """
    Python wrapper for Triton Fused Graph Gather-Scatter Kernel.

    Args:
        x: Node features (num_nodes, hidden_dim) OR Edge features (num_edges, hidden_dim)
        edge_index: Graph edge indices of shape (2, num_edges)
        num_nodes: Total number of nodes

    Returns:
        out: Accumulated node features of shape (num_nodes, hidden_dim)
    """
    assert x.is_cuda and edge_index.is_cuda

    num_edges = edge_index.size(1)
    hidden_dim = x.size(1)

    is_edge_feature = (x.size(0) == num_edges)

    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1 if is_edge_feature else x.size(0)

    src_indices = edge_index[0]
    dst_indices = edge_index[1]

    out = torch.zeros((num_nodes, hidden_dim), device=x.device, dtype=x.dtype)

    BLOCK_EDGES = 32
    BLOCK_DIM = triton.next_power_of_2(hidden_dim)
    grid = (triton.cdiv(num_edges, BLOCK_EDGES),)

    _fused_gather_scatter_kernel[grid](
        x,
        src_indices,
        dst_indices,
        out,
        num_edges,
        hidden_dim,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        is_edge_feature=is_edge_feature,
        BLOCK_EDGES=BLOCK_EDGES,
        BLOCK_DIM=BLOCK_DIM,
    )

    return out
