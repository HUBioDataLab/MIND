"""Type-independent local-coordinate templates for masked molecular blocks.

These points are deliberately synthetic.  They do not approximate a specific
amino-acid sidechain or RNA base.  During a later integration stage, every
masked side/base slot in one domain will receive the same ordered point cloud,
whether its target is a real atom or virtual padding.  Consequently the input
geometry cannot reveal the residue/nucleotide type through atom count or
target-dependent coordinates.

This module contains geometry only.  It does not mutate adapters, cached
coordinates, graph edges, masking, or loss targets.
"""

from __future__ import annotations

from typing import Sequence

import torch


MINIMUM_TEMPLATE_SEPARATION_ANGSTROM = 0.75


# Ten points on a radius-1.15 deterministic spherical shell.  The shell is
# translated into the protein sidechain half-space of the N-CA-C local frame.
# Coordinates are stored explicitly rather than generated with trigonometry at
# runtime, so repeated calls do not depend on a random seed or math backend.
_PROTEIN_SIDECHAIN_LOCAL_POINTS: tuple[tuple[float, float, float], ...] = (
    (-0.498727, -1.500000, -1.165000),
    (-1.605575, -0.945244, -1.395000),
    (-0.912930, -2.492116, -1.625000),
    (-0.332524, -0.629396, -1.855000),
    (-2.126744, -1.699305, -2.085000),
    (-0.034545, -2.114143, -2.315000),
    (-1.284794, -0.440582, -2.545000),
    (-1.459031, -2.383836, -2.775000),
    (-0.228569, -1.218275, -3.005000),
    (-1.463350, -1.308736, -3.235000),
)


# Eleven points on a radius-1.00 deterministic spherical shell.  The shell is
# translated away from the twelve retained RNA backbone atoms in the canonical
# C2'-C3'-C4' frame and rotated 180 degrees around local z so no point enters
# the rare OP2 corridor found by the full-source audit.  It remains connected
# internally at a 5 A spatial cutoff, while the audit checks that it never
# collides with retained backbone coordinates.
_RNA_BASE_LOCAL_POINTS: tuple[tuple[float, float, float], ...] = (
    (-2.416598, 3.500000, -2.090909),
    (-1.493908, 3.036378, -2.272727),
    (-2.073275, 4.334931, -2.454545),
    (-2.566786, 2.760728, -2.636364),
    (-1.031700, 3.671279, -2.818182),
    (-2.843755, 4.036728, -3.000000),
    (-1.744723, 2.550381, -3.181818),
    (-1.570646, 4.326695, -3.363636),
    (-2.787283, 3.212485, -3.545455),
    (-1.365577, 3.238119, -3.727273),
    (-2.176573, 3.877327, -3.909091),
)


def _template_tensor(
    points: Sequence[Sequence[float]],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"template dtype must be floating point, got {dtype}")
    # Construct a fresh tensor so a caller cannot mutate shared module state.
    return torch.tensor(points, dtype=dtype, device=device)


def protein_sidechain_template(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the same ordered ten local points for every protein residue."""
    return _template_tensor(
        _PROTEIN_SIDECHAIN_LOCAL_POINTS, dtype=dtype, device=device
    )


def rna_base_template(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the same ordered eleven local points for every RNA nucleotide."""
    return _template_tensor(_RNA_BASE_LOCAL_POINTS, dtype=dtype, device=device)


def _validate_static_template(points: Sequence[Sequence[float]], count: int) -> None:
    tensor = _template_tensor(points, dtype=torch.float64)
    if tuple(tensor.shape) != (count, 3):
        raise AssertionError(
            f"template must have shape ({count}, 3), got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise AssertionError("template contains NaN or Inf")
    distances = torch.pdist(tensor)
    if distances.numel() and distances.min() < MINIMUM_TEMPLATE_SEPARATION_ANGSTROM:
        raise AssertionError("template points are too close to one another")


_validate_static_template(_PROTEIN_SIDECHAIN_LOCAL_POINTS, 10)
_validate_static_template(_RNA_BASE_LOCAL_POINTS, 11)
