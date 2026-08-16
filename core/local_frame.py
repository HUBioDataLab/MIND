"""Deterministic local coordinate frames for molecular block templates.

This module is deliberately independent of adapters, PyG caches, and the
training model. Stage 1 validates the geometry contract here before any
runtime template integration is attempted.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


DEFAULT_FRAME_EPSILON = 1.0e-6


class LocalFrameError(ValueError):
    """A block cannot define a reliable local frame and must be quarantined."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class LocalFrame:
    """Right-handed frame whose basis vectors are stored as matrix columns."""

    origin: torch.Tensor
    basis: torch.Tensor

    def local_to_global(self, local_points: torch.Tensor) -> torch.Tensor:
        """Transform one or more row-vector points from local to global space."""
        points = torch.as_tensor(
            local_points, dtype=self.origin.dtype, device=self.origin.device
        )
        return points @ self.basis.transpose(-1, -2) + self.origin

    def global_to_local(self, global_points: torch.Tensor) -> torch.Tensor:
        """Transform one or more row-vector points from global to local space."""
        points = torch.as_tensor(
            global_points, dtype=self.origin.dtype, device=self.origin.device
        )
        return (points - self.origin) @ self.basis


def _anchor_tensor(
    value,
    name: str,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    if tuple(tensor.shape) != (3,):
        raise LocalFrameError(
            "invalid_anchor_shape",
            f"{name} must have shape (3,), got {tuple(tensor.shape)}",
        )
    if not torch.isfinite(tensor).all():
        raise LocalFrameError("nonfinite_anchor", f"{name} contains NaN or Inf")
    return tensor


def protein_n_ca_c_frame(
    atom_positions: Mapping[str, torch.Tensor],
    *,
    epsilon: float = DEFAULT_FRAME_EPSILON,
) -> LocalFrame:
    """Build a deterministic right-handed protein frame from N, CA, and C.

    Contract:
    - origin is CA;
    - local +x points from CA to C;
    - local +y is the component of CA->N perpendicular to +x;
    - local +z is ``x cross y``.

    Missing, coincident, non-finite, or collinear anchors raise a
    ``LocalFrameError`` carrying an explicit quarantine ``reason``. No silent
    fallback or invented coordinate is used.
    """
    missing = tuple(name for name in ("N", "CA", "C") if name not in atom_positions)
    if missing:
        raise LocalFrameError(
            "missing_anchor",
            f"required protein anchors absent: {', '.join(missing)}",
        )
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    ca = _anchor_tensor(atom_positions["CA"], "CA")
    n = _anchor_tensor(
        atom_positions["N"], "N", dtype=ca.dtype, device=ca.device
    )
    c = _anchor_tensor(
        atom_positions["C"], "C", dtype=ca.dtype, device=ca.device
    )

    ca_to_c = c - ca
    ca_to_n = n - ca
    c_norm = torch.linalg.vector_norm(ca_to_c)
    n_norm = torch.linalg.vector_norm(ca_to_n)
    if c_norm <= epsilon:
        raise LocalFrameError("coincident_anchor", "C and CA are coincident")
    if n_norm <= epsilon:
        raise LocalFrameError("coincident_anchor", "N and CA are coincident")

    x_axis = ca_to_c / c_norm
    n_perpendicular = ca_to_n - torch.dot(ca_to_n, x_axis) * x_axis
    perpendicular_norm = torch.linalg.vector_norm(n_perpendicular)
    if perpendicular_norm <= epsilon:
        raise LocalFrameError(
            "collinear_anchors", "N, CA, and C do not define a plane"
        )

    y_axis = n_perpendicular / perpendicular_norm
    z_axis = torch.linalg.cross(x_axis, y_axis)
    z_axis = z_axis / torch.linalg.vector_norm(z_axis)
    # Recompute y from the normalized x/z pair to minimize accumulated
    # floating-point non-orthogonality while preserving right-handedness.
    y_axis = torch.linalg.cross(z_axis, x_axis)
    basis = torch.stack((x_axis, y_axis, z_axis), dim=1)
    return LocalFrame(origin=ca, basis=basis)


def rna_c2_c3_c4_frame(
    pos_codes: Sequence[str],
    positions,
    *,
    epsilon: float = DEFAULT_FRAME_EPSILON,
) -> LocalFrame:
    """Build the canonical right-handed RNA frame from C2', C3', and C4'.

    Contract selected by the Stage 1 source-data audit:
    - origin is C3';
    - local +x points from C3' to C4';
    - local +y is the component of C3'->C2' perpendicular to +x;
    - local +z is ``x cross y``.

    ``pos_codes`` and ``positions`` intentionally use the same parallel-array
    representation as cached PyG RNA graphs. Anchor lookup is by position code,
    never atom arrival order. Missing, duplicate, coincident, non-finite, and
    collinear anchors raise ``LocalFrameError`` with an explicit quarantine
    reason. No silent fallback or invented coordinate is used.

    O2' is not an anchor here. It is the independent chirality quality-control
    reference used by the RNA target-eligibility policy.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    codes = tuple(str(code) for code in pos_codes)
    position_tensor = torch.as_tensor(positions)
    if not position_tensor.is_floating_point():
        position_tensor = position_tensor.to(torch.float32)
    expected_shape = (len(codes), 3)
    if tuple(position_tensor.shape) != expected_shape:
        raise LocalFrameError(
            "invalid_atom_table_shape",
            "positions must match pos_codes with shape "
            f"{expected_shape}, got {tuple(position_tensor.shape)}",
        )

    required = ("C2'", "C3'", "C4'")
    anchor_indices = {
        name: tuple(index for index, code in enumerate(codes) if code == name)
        for name in required
    }
    missing = tuple(name for name in required if not anchor_indices[name])
    if missing:
        raise LocalFrameError(
            "missing_anchor",
            f"required RNA anchors absent: {', '.join(missing)}",
        )
    duplicates = tuple(name for name in required if len(anchor_indices[name]) > 1)
    if duplicates:
        raise LocalFrameError(
            "duplicate_anchor",
            f"required RNA anchors duplicated: {', '.join(duplicates)}",
        )

    c3 = _anchor_tensor(position_tensor[anchor_indices["C3'"][0]], "C3'")
    c2 = _anchor_tensor(
        position_tensor[anchor_indices["C2'"][0]],
        "C2'",
        dtype=c3.dtype,
        device=c3.device,
    )
    c4 = _anchor_tensor(
        position_tensor[anchor_indices["C4'"][0]],
        "C4'",
        dtype=c3.dtype,
        device=c3.device,
    )

    c3_to_c4 = c4 - c3
    c3_to_c2 = c2 - c3
    c4_norm = torch.linalg.vector_norm(c3_to_c4)
    c2_norm = torch.linalg.vector_norm(c3_to_c2)
    if c4_norm <= epsilon:
        raise LocalFrameError("coincident_anchor", "C4' and C3' are coincident")
    if c2_norm <= epsilon:
        raise LocalFrameError("coincident_anchor", "C2' and C3' are coincident")

    x_axis = c3_to_c4 / c4_norm
    c2_perpendicular = c3_to_c2 - torch.dot(c3_to_c2, x_axis) * x_axis
    perpendicular_norm = torch.linalg.vector_norm(c2_perpendicular)
    if perpendicular_norm <= epsilon:
        raise LocalFrameError(
            "collinear_anchors", "C2', C3', and C4' do not define a plane"
        )

    y_axis = c2_perpendicular / perpendicular_norm
    z_axis = torch.linalg.cross(x_axis, y_axis)
    z_axis = z_axis / torch.linalg.vector_norm(z_axis)
    y_axis = torch.linalg.cross(z_axis, x_axis)
    basis = torch.stack((x_axis, y_axis, z_axis), dim=1)
    return LocalFrame(origin=c3, basis=basis)
