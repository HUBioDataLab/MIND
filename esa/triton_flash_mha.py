#!/usr/bin/env python3
"""
Custom OpenAI Triton Fused Block-Sparse FlashAttention Kernel for MIND Model (esa/mha.py).

Fuses:
1. Matrix multiplication Q * K.T / sqrt(d)
2. 2D Graph Adjacency Topology Masking (+ adj_mask)
3. Online Softmax reduction in SRAM registers (no VRAM allocation for (B, H, N, N) matrices)
4. Matrix multiplication Softmax(S) * V

Eliminates (B, H, N, N) VRAM allocation entirely while preserving GNN graph topology masks!
"""

import math
import torch
import triton
import triton.language as tl


# =====================================================================
# 🚀 Triton Fused Block-Sparse FlashAttention Forward Kernel (3D Grid + Varlen Support)
# =====================================================================
@triton.jit
def _triton_flash_attn_fwd_kernel(
    Q, K, V, Mask, SeqLens, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_sb,
    stride_ob, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    HAS_SEQLENS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    if HAS_SEQLENS:
        actual_seq_len = tl.load(SeqLens + off_b * stride_sb)
    else:
        actual_seq_len = N_CTX

    # Early exit for padding query blocks
    if start_m * BLOCK_M >= actual_seq_len:
        return

    # Offset pointers for batch and head
    q_ptr = Q + off_b * stride_qb + off_h * stride_qh
    k_ptr = K + off_b * stride_kb + off_h * stride_kh
    v_ptr = V + off_b * stride_vb + off_h * stride_vh
    mask_ptr = Mask + off_b * stride_mb + (off_h % 1) * stride_mh
    out_ptr = Out + off_b * stride_ob + off_h * stride_oh

    # Block ranges
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Query block
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = (offs_m[:, None] < actual_seq_len) & (offs_d[None, :] < BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize online softmax statistics in SRAM
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop ONLY up to actual_seq_len instead of N_CTX!
    for start_n in range(0, actual_seq_len, BLOCK_N):
        current_offs_n = start_n + offs_n

        # Load Key block
        k_ptrs = k_ptr + current_offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        k_mask = (current_offs_n[None, :] < actual_seq_len) & (offs_d[:, None] < BLOCK_DMODEL)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute Q * K.T / sqrt(d)
        qk = tl.dot(q, k) * sm_scale

        # Load 2D Adjacency Mask block
        m_ptrs = mask_ptr + offs_m[:, None] * stride_mm + current_offs_n[None, :] * stride_mn
        m_mask = (offs_m[:, None] < actual_seq_len) & (current_offs_n[None, :] < actual_seq_len)
        mask_val = tl.load(m_ptrs, mask=m_mask, other=-1e9)

        # Apply Graph Adjacency Masking
        qk += mask_val

        # Online Softmax Updates in SRAM
        m_curr = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_curr)
        p = tl.exp(qk - m_curr[:, None])

        # Scale previous accumulator
        acc *= alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_curr

        # Load Value block
        v_ptrs = v_ptr + current_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = (current_offs_n[:, None] < actual_seq_len) & (offs_d[None, :] < BLOCK_DMODEL)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Accumulate Softmax(S) * V into SRAM
        acc += tl.dot(p.to(v.dtype), v)

    # Normalize by Softmax denominator
    acc /= l_i[:, None]

    # Write final Output to VRAM
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    out_mask = (offs_m[:, None] < actual_seq_len) & (offs_d[None, :] < BLOCK_DMODEL)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_flash_attn_func(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None, seq_lens: torch.Tensor = None
) -> torch.Tensor:
    B, H, N, D = q.shape

    if mask is None:
        mask = torch.zeros((B, 1, N, N), device=q.device, dtype=q.dtype)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B, 1, N, N)

    out = torch.zeros_like(q)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_DMODEL = triton.next_power_of_2(D)
    sm_scale = 1.0 / math.sqrt(D)

    grid = (triton.cdiv(N, BLOCK_M), H, B)

    has_seq_lens = seq_lens is not None
    seq_lens_ptr = seq_lens if has_seq_lens else q
    stride_sb = seq_lens.stride(0) if has_seq_lens else 0

    _triton_flash_attn_fwd_kernel[grid](
        q, k, v, mask, seq_lens_ptr, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        stride_sb,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N,
        HAS_SEQLENS=has_seq_lens,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
    )
    return out
