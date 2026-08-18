#!/usr/bin/env python3
"""
==============================================================================
Triton FlashAttention Unit & Autograd Backward Tests
==============================================================================
Validates:
  1. Forward pass numerical alignment with PyTorch Scaled Dot-Product Attention.
  2. Backward pass exact gradients (dQ, dK, dV) in BFloat16 AMP precision.
  3. Graph 2D adjacency mask handling.
  4. Variable-length (Varlen) sequence handling without padding.
==============================================================================
"""

import math
import torch
import torch.nn.functional as F

from esa.triton_flash_mha import triton_flash_attn_func


def pytorch_reference_attention(q, k, v, mask=None, seq_lens=None):
    B, H, N, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        scores = scores + mask

    if seq_lens is not None:
        pad_mask = torch.arange(N, device=q.device).unsqueeze(0) >= seq_lens.unsqueeze(1)
        pad_mask = pad_mask.view(B, 1, 1, N) | pad_mask.view(B, 1, N, 1)
        scores = scores.masked_fill(pad_mask, -1e9)

    attn_weights = F.softmax(scores, dim=-1)

    if seq_lens is not None:
        q_pad_mask = (torch.arange(N, device=q.device).unsqueeze(0) >= seq_lens.unsqueeze(1)).view(B, 1, N, 1)
        attn_weights = attn_weights.masked_fill(q_pad_mask, 0.0)

    out = torch.matmul(attn_weights, v)
    return out


def test_triton_flash_attention_backward_basic():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    B, H, N, D = 2, 4, 64, 32
    
    # BFloat16 AMP precision test (Training Precision)
    q_ref = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k_ref = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v_ref = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)

    out_ref = pytorch_reference_attention(q_ref, k_ref, v_ref)
    out_tri = triton_flash_attn_func(q_tri, k_tri, v_tri)

    assert torch.allclose(out_ref, out_tri, atol=2e-2, rtol=2e-2), f"Max diff forward: {(out_ref - out_tri).abs().max()}"

    # Check backward
    grad_output = torch.randn_like(out_ref)
    out_ref.backward(grad_output)
    out_tri.backward(grad_output)

    assert torch.allclose(q_ref.grad, q_tri.grad, atol=2e-2, rtol=2e-2), f"Max diff dQ: {(q_ref.grad - q_tri.grad).abs().max()}"
    assert torch.allclose(k_ref.grad, k_tri.grad, atol=2e-2, rtol=2e-2), f"Max diff dK: {(k_ref.grad - k_tri.grad).abs().max()}"
    assert torch.allclose(v_ref.grad, v_tri.grad, atol=2e-2, rtol=2e-2), f"Max diff dV: {(v_ref.grad - v_tri.grad).abs().max()}"
    print("✅ Triton FlashAttention basic backward test passed!")


def test_triton_flash_attention_backward_with_mask_and_varlen():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    B, H, N, D = 2, 4, 64, 32
    seq_lens = torch.tensor([45, 60], device=device, dtype=torch.int32)
    mask = torch.zeros(B, 1, N, N, device=device, dtype=torch.bfloat16)
    # Add block masking
    mask[:, :, 10:20, 10:20] = -1e4

    q_ref = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k_ref = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v_ref = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)

    out_ref = pytorch_reference_attention(q_ref, k_ref, v_ref, mask=mask, seq_lens=seq_lens)
    out_tri = triton_flash_attn_func(q_tri, k_tri, v_tri, mask=mask, seq_lens=seq_lens)

    # Check forward
    assert torch.allclose(out_ref, out_tri, atol=2e-2, rtol=2e-2), f"Max diff forward varlen: {(out_ref - out_tri).abs().max()}"

    # Check backward
    grad_output = torch.randn_like(out_ref)
    out_ref.backward(grad_output)
    out_tri.backward(grad_output)

    assert torch.allclose(q_ref.grad, q_tri.grad, atol=2e-2, rtol=2e-2), f"Max diff dQ varlen: {(q_ref.grad - q_tri.grad).abs().max()}"
    assert torch.allclose(k_ref.grad, k_tri.grad, atol=2e-2, rtol=2e-2), f"Max diff dK varlen: {(k_ref.grad - k_tri.grad).abs().max()}"
    assert torch.allclose(v_ref.grad, v_tri.grad, atol=2e-2, rtol=2e-2), f"Max diff dV varlen: {(v_ref.grad - v_tri.grad).abs().max()}"
    print("✅ Triton FlashAttention varlen + mask backward test passed!")


if __name__ == "__main__":
    print("🧪 Running Triton FlashAttention Backward Unit Tests...")
    test_triton_flash_attention_backward_basic()
    test_triton_flash_attention_backward_with_mask_and_varlen()
    print("🎉 All Triton Backward Tests Passed Successfully!")
