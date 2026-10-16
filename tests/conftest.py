"""Shared test fixtures."""

import pytest
import torch
import torch.nn.functional as F

import esa.mha_flash_varlen as flash_module


def sdpa_varlen(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0, softmax_scale=None, causal=False):
    """Per-sequence SDPA stand-in for ``flash_attn_varlen_func`` (CPU)."""
    outputs = []
    for i in range(len(cu_q) - 1):
        a, b = int(cu_q[i]), int(cu_q[i + 1])
        qi, ki, vi = (t[a:b].transpose(0, 1) for t in (q, k, v))
        outputs.append(F.scaled_dot_product_attention(qi, ki, vi).transpose(0, 1))
    return torch.cat(outputs, dim=0)


@pytest.fixture
def cpu_flash(monkeypatch):
    """Run the production flash_varlen layers on CPU; only the CUDA kernel is replaced."""
    monkeypatch.setattr(flash_module, "flash_attn_varlen_func", sdpa_varlen)
