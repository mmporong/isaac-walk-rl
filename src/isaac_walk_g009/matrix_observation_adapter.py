"""Read-only projection of the G009 terrain-pair contact-force matrix."""

from __future__ import annotations

from dataclasses import dataclass

import torch


CONTACT_THRESHOLD_N = 1.0e-6
EXPECTED_BODY_COUNT = 19
EXPECTED_FILTER_COUNT = 1
EXPECTED_COMPONENT_COUNT = 3


class MatrixObservationAdapterError(ValueError):
    """Base error for fail-closed matrix observation adaptation."""


class MatrixObservationSourceMissingError(MatrixObservationAdapterError):
    """Raised when the required contact-force matrix is unavailable."""


class MatrixObservationSourceTypeError(MatrixObservationAdapterError, TypeError):
    """Raised when the source is not a ``torch.Tensor``."""


class MatrixObservationSourceShapeError(MatrixObservationAdapterError):
    """Raised when the source does not have shape ``[N, 19, 1, 3]``."""


class MatrixObservationSourceDTypeError(MatrixObservationAdapterError):
    """Raised when the source is not ``torch.float32``."""


class MatrixObservationSourceNonFiniteError(MatrixObservationAdapterError):
    """Raised when the source contains NaN or infinity."""


@dataclass(frozen=True, slots=True)
class MatrixObservation:
    """Authority-preserving outputs derived from one contact-force matrix.

    ``world_xyz`` retains the signed world-frame vector. ``magnitude`` and
    ``contact_mask`` are diagnostic projections and must not replace it as the
    authoritative representation.
    """

    world_xyz: torch.Tensor
    magnitude: torch.Tensor
    contact_mask: torch.Tensor


def adapt_terrain_pair_force_matrix_w(source: object | None) -> MatrixObservation:
    """Project ``[N, 19, 1, 3]`` normal contact force without mutating it.

    The filter dimension is reduced before the vector norm is evaluated.  No
    normalization, clipping, device transfer, fallback, or invalid-value fill
    is performed.
    """

    if source is None:
        raise MatrixObservationSourceMissingError(
            "terrain_pair_force_matrix_w is required and cannot be None"
        )
    if not isinstance(source, torch.Tensor):
        raise MatrixObservationSourceTypeError(
            "terrain_pair_force_matrix_w must be a torch.Tensor, "
            f"got {type(source).__name__}"
        )
    if source.ndim != 4 or tuple(source.shape[1:]) != (
        EXPECTED_BODY_COUNT,
        EXPECTED_FILTER_COUNT,
        EXPECTED_COMPONENT_COUNT,
    ):
        raise MatrixObservationSourceShapeError(
            "terrain_pair_force_matrix_w must have shape [N, 19, 1, 3], "
            f"got {tuple(source.shape)}"
        )
    if source.dtype is not torch.float32:
        raise MatrixObservationSourceDTypeError(
            "terrain_pair_force_matrix_w must have dtype torch.float32, "
            f"got {source.dtype}"
        )
    if not bool(torch.isfinite(source).all().item()):
        raise MatrixObservationSourceNonFiniteError(
            "terrain_pair_force_matrix_w must contain only finite values"
        )

    world_xyz = source.sum(dim=2)
    magnitude = torch.linalg.vector_norm(world_xyz, dim=-1)
    contact_mask = magnitude > CONTACT_THRESHOLD_N
    return MatrixObservation(
        world_xyz=world_xyz,
        magnitude=magnitude,
        contact_mask=contact_mask,
    )


__all__ = [
    "CONTACT_THRESHOLD_N",
    "MatrixObservation",
    "MatrixObservationAdapterError",
    "MatrixObservationSourceDTypeError",
    "MatrixObservationSourceMissingError",
    "MatrixObservationSourceNonFiniteError",
    "MatrixObservationSourceShapeError",
    "MatrixObservationSourceTypeError",
    "adapt_terrain_pair_force_matrix_w",
]
