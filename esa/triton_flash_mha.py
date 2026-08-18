#!/usr/bin/env python3
"""
==============================================================================
OpenAI Triton Block-Sparse & Variable-Length (Varlen) FlashAttention with Autograd
==============================================================================
Features:
1. Block-Sparse Tiling: Skips empty padding blocks for irregular molecules.
2. Varlen (Variable Sequence Length) Support: Loops ONLY up to actual molecule size.
3. Graph 2D Adjacency Masking Support: Fuses structural adjacency into attention.
4. Zero-Memory Mask Expansion: Uses 1-element expanded tensor with (0,0,0,0) strides when mask is None.
5. Exact Full Autograd Backward: Custom Triton kernels for dQ, dK, dV with BFloat16 AMP.
==============================================================================
"""

import math
import torch
import triton
import triton.language as tl


# =====================================================================
# 🚀 Triton Fused Block-Sparse FlashAttention Forward Kernel
# =====================================================================
@triton.jit
def _triton_flash_attn_fwd_kernel(
    Q, K, V, Mask, SeqLens, Out, L,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_sb,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_lm,
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
        actual_seq_len = tl.minimum(actual_seq_len, N_CTX)
    else:
        actual_seq_len = N_CTX

    if actual_seq_len <= 0 or start_m * BLOCK_M >= actual_seq_len:
        return

    q_ptr = Q + off_b * stride_qb + off_h * stride_qh
    k_ptr = K + off_b * stride_kb + off_h * stride_kh
    v_ptr = V + off_b * stride_vb + off_h * stride_vh
    mask_ptr = Mask + off_b * stride_mb + off_h * stride_mh
    out_ptr = Out + off_b * stride_ob + off_h * stride_oh
    l_ptr = L + off_b * stride_lb + off_h * stride_lh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = q_ptr + offs_m[:, None].to(tl.int64) * stride_qm + offs_d[None, :].to(tl.int64) * stride_qk
    q_mask = (offs_m[:, None] < actual_seq_len) & (offs_m[:, None] < N_CTX) & (offs_d[None, :] < BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, actual_seq_len, BLOCK_N):
        current_offs_n = start_n + offs_n
        k_ptrs = k_ptr + current_offs_n[:, None].to(tl.int64) * stride_kn + offs_d[None, :].to(tl.int64) * stride_kk
        k_mask = (current_offs_n[:, None] < actual_seq_len) & (current_offs_n[:, None] < N_CTX) & (offs_d[None, :] < BLOCK_DMODEL)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # qk = q @ k.T
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        m_ptrs = mask_ptr + offs_m[:, None].to(tl.int64) * stride_mm + current_offs_n[None, :].to(tl.int64) * stride_mn
        m_mask = (offs_m[:, None] < actual_seq_len) & (current_offs_n[None, :] < actual_seq_len) & (offs_m[:, None] < N_CTX) & (current_offs_n[None, :] < N_CTX)
        mask_val = tl.load(m_ptrs, mask=m_mask, other=-1e9)
        qk += mask_val

        m_curr = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_curr)
        p = tl.exp(qk - m_curr[:, None])

        acc = acc * alpha[:, None]
        v_ptrs = v_ptr + current_offs_n[:, None].to(tl.int64) * stride_vn + offs_d[None, :].to(tl.int64) * stride_vk
        v_mask = (current_offs_n[:, None] < actual_seq_len) & (current_offs_n[:, None] < N_CTX) & (offs_d[None, :] < BLOCK_DMODEL)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_curr

    acc /= l_i[:, None]

    out_ptrs = out_ptr + offs_m[:, None].to(tl.int64) * stride_om + offs_d[None, :].to(tl.int64) * stride_ok
    out_mask = (offs_m[:, None] < actual_seq_len) & (offs_m[:, None] < N_CTX) & (offs_d[None, :] < BLOCK_DMODEL)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=out_mask)

    l_ptrs = l_ptr + offs_m.to(tl.int64) * stride_lm
    l_mask = (offs_m < actual_seq_len) & (offs_m < N_CTX)
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=l_mask)


# =====================================================================
# 🚀 Triton Fused Block-Sparse FlashAttention Backward Kernels (dQ and dK, dV)
# =====================================================================
@triton.jit
def _triton_flash_attn_bwd_dq_kernel(
    Q, K, V, Mask, SeqLens,
    DO, DQ,
    L, D,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_sb,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
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
        actual_seq_len = tl.minimum(actual_seq_len, N_CTX)
    else:
        actual_seq_len = N_CTX

    if actual_seq_len <= 0 or start_m * BLOCK_M >= actual_seq_len:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptr = Q + off_b * stride_qb + off_h * stride_qh
    k_ptr = K + off_b * stride_kb + off_h * stride_kh
    v_ptr = V + off_b * stride_vb + off_h * stride_vh
    mask_ptr = Mask + off_b * stride_mb + off_h * stride_mh
    do_ptr = DO + off_b * stride_dob + off_h * stride_doh
    dq_ptr = DQ + off_b * stride_dqb + off_h * stride_dqh
    l_ptr = L + off_b * stride_lb + off_h * stride_lh
    d_ptr = D + off_b * stride_db + off_h * stride_dh

    q_ptrs = q_ptr + offs_m[:, None].to(tl.int64) * stride_qm + offs_d[None, :].to(tl.int64) * stride_qk
    q_mask = (offs_m[:, None] < actual_seq_len) & (offs_m[:, None] < N_CTX) & (offs_d[None, :] < BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    do_ptrs = do_ptr + offs_m[:, None].to(tl.int64) * stride_dom + offs_d[None, :].to(tl.int64) * stride_dok
    do = tl.load(do_ptrs, mask=q_mask, other=0.0)

    l_ptrs = l_ptr + offs_m.to(tl.int64) * stride_lm
    l_mask = (offs_m < actual_seq_len) & (offs_m < N_CTX)
    l_i = tl.load(l_ptrs, mask=l_mask, other=0.0)

    d_ptrs = d_ptr + offs_m.to(tl.int64) * stride_dm
    d_i = tl.load(d_ptrs, mask=l_mask, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, actual_seq_len, BLOCK_N):
        current_offs_n = start_n + offs_n

        k_ptrs = k_ptr + current_offs_n[:, None].to(tl.int64) * stride_kn + offs_d[None, :].to(tl.int64) * stride_kk
        k_mask = (current_offs_n[:, None] < actual_seq_len) & (current_offs_n[:, None] < N_CTX) & (offs_d[None, :] < BLOCK_DMODEL)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # qk = q @ k.T
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        m_ptrs = mask_ptr + offs_m[:, None].to(tl.int64) * stride_mm + current_offs_n[None, :].to(tl.int64) * stride_mn
        m_mask = (offs_m[:, None] < actual_seq_len) & (current_offs_n[None, :] < actual_seq_len) & (offs_m[:, None] < N_CTX) & (current_offs_n[None, :] < N_CTX)
        mask_val = tl.load(m_ptrs, mask=m_mask, other=-1e9)
        qk += mask_val

        p = tl.exp(qk - l_i[:, None])
        p = tl.where(m_mask, p, 0.0)

        v_ptrs = v_ptr + current_offs_n[:, None].to(tl.int64) * stride_vn + offs_d[None, :].to(tl.int64) * stride_vk
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # dp = do @ v.T
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d_i[:, None]) * sm_scale

        # dq += ds @ k
        dq += tl.dot(ds.to(k.dtype), k)

    dq_ptrs = dq_ptr + offs_m[:, None].to(tl.int64) * stride_dqm + offs_d[None, :].to(tl.int64) * stride_dqk
    tl.store(dq_ptrs, dq, mask=q_mask)


@triton.jit
def _triton_flash_attn_bwd_dkv_kernel(
    Q, K, V, Mask, SeqLens,
    DO, DK, DV,
    L, D,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_sb,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    Z, H, N_CTX,
    HAS_SEQLENS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    if HAS_SEQLENS:
        actual_seq_len = tl.load(SeqLens + off_b * stride_sb)
        actual_seq_len = tl.minimum(actual_seq_len, N_CTX)
    else:
        actual_seq_len = N_CTX

    if actual_seq_len <= 0 or start_n * BLOCK_N >= actual_seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptr = Q + off_b * stride_qb + off_h * stride_qh
    k_ptr = K + off_b * stride_kb + off_h * stride_kh
    v_ptr = V + off_b * stride_vb + off_h * stride_vh
    mask_ptr = Mask + off_b * stride_mb + off_h * stride_mh
    do_ptr = DO + off_b * stride_dob + off_h * stride_doh
    dk_ptr = DK + off_b * stride_dkb + off_h * stride_dkh
    dv_ptr = DV + off_b * stride_dvb + off_h * stride_dvh
    l_ptr = L + off_b * stride_lb + off_h * stride_lh
    d_ptr = D + off_b * stride_db + off_h * stride_dh

    k_ptrs = k_ptr + offs_n[:, None].to(tl.int64) * stride_kn + offs_d[None, :].to(tl.int64) * stride_kk
    k_mask = (offs_n[:, None] < actual_seq_len) & (offs_n[:, None] < N_CTX) & (offs_d[None, :] < BLOCK_DMODEL)
    k = tl.load(k_ptrs, mask=k_mask, other=0.0)

    v_ptrs = v_ptr + offs_n[:, None].to(tl.int64) * stride_vn + offs_d[None, :].to(tl.int64) * stride_vk
    v_mask = (offs_n[:, None] < actual_seq_len) & (offs_n[:, None] < N_CTX) & (offs_d[None, :] < BLOCK_DMODEL)
    v = tl.load(v_ptrs, mask=v_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    for start_m in range(0, actual_seq_len, BLOCK_M):
        current_offs_m = start_m + offs_m

        q_ptrs = q_ptr + current_offs_m[:, None].to(tl.int64) * stride_qm + offs_d[None, :].to(tl.int64) * stride_qk
        q_mask = (current_offs_m[:, None] < actual_seq_len) & (current_offs_m[:, None] < N_CTX) & (offs_d[None, :] < BLOCK_DMODEL)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        l_ptrs = l_ptr + current_offs_m.to(tl.int64) * stride_lm
        l_mask = (current_offs_m < actual_seq_len) & (current_offs_m < N_CTX)
        l_i = tl.load(l_ptrs, mask=l_mask, other=0.0)

        d_ptrs = d_ptr + current_offs_m.to(tl.int64) * stride_dm
        Di = tl.load(d_ptrs, mask=l_mask, other=0.0)

        # qk = q @ k.T
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        m_ptrs = mask_ptr + current_offs_m[:, None].to(tl.int64) * stride_mm + offs_n[None, :].to(tl.int64) * stride_mn
        m_mask = (current_offs_m[:, None] < actual_seq_len) & (offs_n[None, :] < actual_seq_len) & (current_offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        mask_val = tl.load(m_ptrs, mask=m_mask, other=-1e9)
        qk += mask_val

        p = tl.exp(qk - l_i[:, None])
        p = tl.where(m_mask, p, 0.0)

        do_ptrs = do_ptr + current_offs_m[:, None].to(tl.int64) * stride_dom + offs_d[None, :].to(tl.int64) * stride_dok
        do = tl.load(do_ptrs, mask=q_mask, other=0.0)

        # dv += p.T @ do
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        # dp = do @ v.T
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - Di[:, None]) * sm_scale

        # dk += ds.T @ q
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    out_mask = (offs_n[:, None] < actual_seq_len) & (offs_n[:, None] < N_CTX) & (offs_d[None, :] < BLOCK_DMODEL)
    dk_ptrs = dk_ptr + offs_n[:, None].to(tl.int64) * stride_dkn + offs_d[None, :].to(tl.int64) * stride_dkk
    tl.store(dk_ptrs, dk, mask=out_mask)

    dv_ptrs = dv_ptr + offs_n[:, None].to(tl.int64) * stride_dvn + offs_d[None, :].to(tl.int64) * stride_dvk
    tl.store(dv_ptrs, dv, mask=out_mask)


# =====================================================================
# 🚀 PyTorch Autograd Function Wrapper
# =====================================================================
class _TritonFlashAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask=None, seq_lens=None):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        B, H, N, D = q.shape

        if mask is None:
            mask = torch.zeros((1, 1, 1, 1), device=q.device, dtype=q.dtype)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 4 and mask.shape[1] > 1 and mask.shape[1] != H:
            mask = mask[:, :1, :, :]

        mask = mask.to(device=q.device, dtype=q.dtype)

        stride_mb = mask.stride(0) if mask.dim() == 4 and mask.shape[0] > 1 else 0
        stride_mh = mask.stride(1) if mask.dim() == 4 and mask.shape[1] > 1 else 0
        stride_mm = mask.stride(2) if mask.dim() == 4 and mask.shape[2] > 1 else 0
        stride_mn = mask.stride(3) if mask.dim() == 4 and mask.shape[3] > 1 else 0

        out = torch.zeros_like(q)
        L = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_DMODEL = triton.next_power_of_2(D)
        sm_scale = 1.0 / math.sqrt(D)

        grid = (triton.cdiv(N, BLOCK_M), H, B)

        if seq_lens is not None:
            seq_lens = seq_lens.to(device=q.device, dtype=torch.int32)
            if seq_lens.numel() < B:
                seq_lens_padded = torch.zeros(B, dtype=torch.int32, device=q.device)
                seq_lens_padded[:seq_lens.numel()] = seq_lens
                seq_lens = seq_lens_padded
            elif seq_lens.numel() > B:
                seq_lens = seq_lens[:B]
            seq_lens = seq_lens.contiguous()

        has_seq_lens = seq_lens is not None
        seq_lens_ptr = seq_lens if has_seq_lens else q
        stride_sb = seq_lens.stride(0) if has_seq_lens else 0

        _triton_flash_attn_fwd_kernel[grid](
            q, k, v, mask, seq_lens_ptr, out, L,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            stride_mb, stride_mh, stride_mm, stride_mn,
            stride_sb,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            B, H, N,
            HAS_SEQLENS=has_seq_lens,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, mask, seq_lens, out, L)
        ctx.sm_scale = sm_scale
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_DMODEL = BLOCK_DMODEL
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, mask, seq_lens, out, L = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N
        BLOCK_DMODEL = ctx.BLOCK_DMODEL

        B, H, N, D = q.shape

        dout = dout.contiguous()
        Delta = torch.sum(dout.float() * out.float(), dim=-1).contiguous()

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        if seq_lens is not None:
            seq_lens = seq_lens.to(device=q.device, dtype=torch.int32)
            if seq_lens.numel() < B:
                seq_lens_padded = torch.zeros(B, dtype=torch.int32, device=q.device)
                seq_lens_padded[:seq_lens.numel()] = seq_lens
                seq_lens = seq_lens_padded
            elif seq_lens.numel() > B:
                seq_lens = seq_lens[:B]
            seq_lens = seq_lens.contiguous()

        has_seq_lens = seq_lens is not None
        seq_lens_ptr = seq_lens if has_seq_lens else q
        stride_sb = seq_lens.stride(0) if has_seq_lens else 0

        stride_mb = mask.stride(0) if mask.dim() == 4 and mask.shape[0] > 1 else 0
        stride_mh = mask.stride(1) if mask.dim() == 4 and mask.shape[1] > 1 else 0
        stride_mm = mask.stride(2) if mask.dim() == 4 and mask.shape[2] > 1 else 0
        stride_mn = mask.stride(3) if mask.dim() == 4 and mask.shape[3] > 1 else 0

        # Launch dQ kernel
        grid_dq = (triton.cdiv(N, BLOCK_M), H, B)
        _triton_flash_attn_bwd_dq_kernel[grid_dq](
            q, k, v, mask, seq_lens_ptr,
            dout, dq,
            L, Delta,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            stride_mb, stride_mh, stride_mm, stride_mn,
            stride_sb,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            Delta.stride(0), Delta.stride(1), Delta.stride(2),
            B, H, N,
            HAS_SEQLENS=has_seq_lens,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
        )

        # Launch dK and dV kernel
        grid_dkv = (triton.cdiv(N, BLOCK_N), H, B)
        _triton_flash_attn_bwd_dkv_kernel[grid_dkv](
            q, k, v, mask, seq_lens_ptr,
            dout, dk, dv,
            L, Delta,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            stride_mb, stride_mh, stride_mm, stride_mn,
            stride_sb,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            Delta.stride(0), Delta.stride(1), Delta.stride(2),
            B, H, N,
            HAS_SEQLENS=has_seq_lens,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
        )

        return dq, dk, dv, None, None


def triton_flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None, seq_lens: torch.Tensor = None) -> torch.Tensor:
    """
    OpenAI Triton Block-Sparse FlashAttention with Variable-Length & Graph Masking Support.
    Supports autograd backward.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "All tensors must be on CUDA."
    return _TritonFlashAttnFunction.apply(q, k, v, mask, seq_lens)
