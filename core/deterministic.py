"""Bit-reproducible CUDA inference for evaluation and embedding extraction.

CUDA float atomics (``index_add_``, scatter) sum in a run-dependent order, so two
identical forward passes differ in the last bits and bf16 autocast amplifies that
(about 1e-2 on pooled embeddings of the tiny test model). Training does not use this.
"""

import os

import torch


def enable_deterministic_algorithms() -> None:
    """Call before the first CUDA kernel; an op without a deterministic kernel then raises."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
