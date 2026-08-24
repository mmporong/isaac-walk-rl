"""Deterministic difficulty-aware random height fields with evidence capture."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

EVIDENCE_SEED = 20260824
TERRAIN_EVIDENCE_REGISTRY: list[dict[str, Any]] = []


def clear_terrain_evidence() -> None:
    """Clear evidence before constructing a new generator."""

    TERRAIN_EVIDENCE_REGISTRY.clear()


def _shape(cfg: Any) -> tuple[int, int]:
    return (int(cfg.size[0] / cfg.horizontal_scale), int(cfg.size[1] / cfg.horizontal_scale))


def _realization(cfg: Any) -> int:
    value = getattr(cfg, "evidence_realization", None)
    if value is None:
        raise ValueError("evidence_realization must be assigned by EvidenceDifficultyTerrainGenerator")
    return int(value)


def raw_difficulty_rough_height_field(difficulty: float, cfg: Any) -> np.ndarray:
    """Return an exact-shape int16 field with amplitude ``0.01 + 0.05*d`` metres.

    A local ``SeedSequence`` makes every realization independent from NumPy's global
    random state.  The result is centered in discrete height units before clipping.
    """

    if not 0.0 <= float(difficulty) <= 1.0:
        raise ValueError("difficulty must be in [0, 1]")
    rows, cols = _shape(cfg)
    if rows < 3 or cols < 3:
        raise ValueError("terrain height field must be at least 3x3")
    vertical_scale = float(cfg.vertical_scale)
    if vertical_scale <= 0.0:
        raise ValueError("vertical_scale must be positive")

    realization = _realization(cfg)
    rng = np.random.default_rng(np.random.SeedSequence([EVIDENCE_SEED, realization]))
    amplitude_m = 0.01 + 0.05 * float(difficulty)
    amplitude_units = max(1, int(np.floor(amplitude_m / vertical_scale + 1.0e-12)))

    # Low-frequency waves retain traversable structure; seeded phases guarantee
    # distinct columns without relying on global random state.
    x = np.linspace(0.0, 2.0 * np.pi, rows, endpoint=False)[:, None]
    y = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False)[None, :]
    phases = rng.uniform(0.0, 2.0 * np.pi, size=4)
    field = (
        0.55 * np.sin(x + phases[0])
        + 0.45 * np.cos(y + phases[1])
        + 0.30 * np.sin(2.0 * x + y + phases[2])
        + 0.20 * np.cos(x - 2.0 * y + phases[3])
    )
    field /= float(np.max(np.abs(field)))
    discrete = np.rint(field * amplitude_units).astype(np.int32)
    discrete -= int(np.rint(discrete.mean()))
    discrete = np.clip(discrete, -amplitude_units, amplitude_units).astype(np.int16)
    if discrete.shape != (rows, cols) or discrete.dtype != np.int16:
        raise AssertionError("height-field shape/dtype contract violated")
    return discrete


def _centered_metrics(height_field: np.ndarray, cfg: Any) -> dict[str, float]:
    heights_m = height_field.astype(np.float64) * float(cfg.vertical_scale)
    heights_m -= float(heights_m.mean())
    absolute = np.abs(heights_m)
    dzdx, dzdy = np.gradient(heights_m, float(cfg.horizontal_scale), edge_order=1)
    # For z=f(x,y), the face-normal tilt is atan(||grad f||).
    slopes = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
    return {
        "height_rms_m": float(np.sqrt(np.mean(heights_m * heights_m))),
        "height_p90_abs_m": float(np.percentile(absolute, 90)),
        "face_normal_slope_rms_rad": float(np.sqrt(np.mean(slopes * slopes))),
        "face_normal_slope_p90_rad": float(np.percentile(slopes, 90)),
    }


def record_height_field_evidence(difficulty: float, cfg: Any, height_field: np.ndarray) -> dict[str, Any]:
    """Record raw evidence before the height-field decorator discards the array."""

    evidence = {
        "realization": _realization(cfg),
        "difficulty": float(difficulty),
        "amplitude_m": 0.01 + 0.05 * float(difficulty),
        "shape": list(height_field.shape),
        "dtype": str(height_field.dtype),
        "raw_sha256": hashlib.sha256(height_field.tobytes(order="C")).hexdigest(),
        "metrics": _centered_metrics(height_field, cfg),
    }
    TERRAIN_EVIDENCE_REGISTRY.append(evidence)
    return evidence


def decorated_difficulty_rough(difficulty: float, cfg: Any):
    """Decorated implementation; imported lazily to keep unit tests Sim-free."""

    from isaaclab.terrains.height_field.utils import height_field_to_mesh

    @height_field_to_mesh
    def _decorated(inner_difficulty: float, inner_cfg: Any) -> np.ndarray:
        raw = raw_difficulty_rough_height_field(inner_difficulty, inner_cfg)
        record_height_field_evidence(inner_difficulty, inner_cfg, raw)
        return raw

    return _decorated(difficulty, cfg)


def difficulty_rough_terrain(difficulty: float, cfg: Any):
    """Public Isaac Lab sub-terrain wrapper."""

    return decorated_difficulty_rough(difficulty, cfg)
