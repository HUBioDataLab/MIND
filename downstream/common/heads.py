"""Prediction heads shared by probing and fine-tuning."""

from __future__ import annotations

from torch import nn

HEADS = ("linear", "mlp")


def build_head(kind: str, in_dim: int, out_dim: int, *, hidden_dim: int = 512, dropout: float = 0.0) -> nn.Module:
    """``linear``: one affine map. ``mlp``: Linear-ReLU-Dropout-Linear."""
    if kind == "linear":
        return nn.Linear(in_dim, out_dim)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)
        )
    raise ValueError(f"unknown head {kind!r}; choose from {HEADS}")
