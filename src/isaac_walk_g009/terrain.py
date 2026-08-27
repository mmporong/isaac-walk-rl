"""Deterministic NumPy slope terrain used by the G009 S0 measurement gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


DEFAULT_STATIC_FRICTION = (0.80,)
DEFAULT_DYNAMIC_FRICTION = (0.60,)
SPATIAL_STATIC_FRICTION = (0.25, 0.40, 0.60, 0.80)
SPATIAL_DYNAMIC_FRICTION = (0.15, 0.28, 0.45, 0.60)


@dataclass(frozen=True)
class SlopeTerrainSpec:
    """Parameters that fully determine one sloped height and material field.

    Azimuth is measured counter-clockwise from world +x.  The resulting unit
    vector points uphill, so the default azimuth of zero makes +x uphill.
    """

    x_min_m: float = -4.0
    x_max_m: float = 4.0
    y_min_m: float = -4.0
    y_max_m: float = 4.0
    cell_size_m: float = 0.25
    slope_deg: float = 0.0
    azimuth_deg: float = 0.0
    seed: int = 20260828
    residual_amplitude_m: float = 0.0
    residual_smoothing_passes: int = 3
    static_friction: tuple[float, ...] = DEFAULT_STATIC_FRICTION
    dynamic_friction: tuple[float, ...] = DEFAULT_DYNAMIC_FRICTION


@dataclass(frozen=True)
class SlopeTerrainField:
    spec: SlopeTerrainSpec
    x_coords_m: np.ndarray
    y_coords_m: np.ndarray
    base_heights_m: np.ndarray
    residual_heights_m: np.ndarray
    heights_m: np.ndarray
    material_indices: np.ndarray
    uphill_xy: np.ndarray
    analytic_normal: np.ndarray


@dataclass(frozen=True)
class TerrainAxes:
    """World-XY direction contract induced by the slope azimuth."""

    uphill_xy: np.ndarray
    downhill_xy: np.ndarray
    contour_left_xy: np.ndarray
    contour_right_xy: np.ndarray


def _validate_spec(spec: SlopeTerrainSpec) -> None:
    numeric_values = (
        spec.x_min_m,
        spec.x_max_m,
        spec.y_min_m,
        spec.y_max_m,
        spec.cell_size_m,
        spec.slope_deg,
        spec.azimuth_deg,
        spec.residual_amplitude_m,
    )
    if not all(np.isfinite(value) for value in numeric_values):
        raise ValueError("terrain geometry and residual values must be finite")
    if not isinstance(spec.seed, int) or isinstance(spec.seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(spec.residual_smoothing_passes, int) or isinstance(spec.residual_smoothing_passes, bool):
        raise ValueError("residual_smoothing_passes must be an integer")
    if spec.cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    if spec.x_max_m <= spec.x_min_m or spec.y_max_m <= spec.y_min_m:
        raise ValueError("terrain bounds must have positive area")
    for extent in (spec.x_max_m - spec.x_min_m, spec.y_max_m - spec.y_min_m):
        cells = extent / spec.cell_size_m
        if not np.isclose(cells, round(cells)):
            raise ValueError("terrain bounds must be an integer multiple of cell_size_m")
    if not 0.0 <= spec.slope_deg < 90.0:
        raise ValueError("slope_deg must satisfy 0 <= slope_deg < 90")
    if spec.residual_amplitude_m < 0.0 or spec.residual_smoothing_passes < 0:
        raise ValueError("residual controls must be non-negative")
    if not spec.static_friction or len(spec.static_friction) != len(spec.dynamic_friction):
        raise ValueError("friction buckets must have equal non-zero length")
    if any(
        not np.isfinite(static)
        or not np.isfinite(dynamic)
        or dynamic < 0.0
        or static < dynamic
        for static, dynamic in zip(spec.static_friction, spec.dynamic_friction)
    ):
        raise ValueError("each friction bucket must satisfy static >= dynamic >= 0")


def _smooth(values: np.ndarray, passes: int) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    for _ in range(passes):
        padded = np.pad(result, 1, mode="reflect")
        result = (
            4.0 * padded[1:-1, 1:-1]
            + 2.0 * padded[:-2, 1:-1]
            + 2.0 * padded[2:, 1:-1]
            + 2.0 * padded[1:-1, :-2]
            + 2.0 * padded[1:-1, 2:]
            + padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        ) / 16.0
    return result


def _unit_scale(values: np.ndarray) -> np.ndarray:
    centered = values - float(np.mean(values))
    scale = float(np.max(np.abs(centered)))
    return np.zeros_like(centered) if scale == 0.0 else centered / scale


def generate_slope_terrain(spec: SlopeTerrainSpec) -> SlopeTerrainField:
    """Generate a base plane, separately stored residual relief, and friction mosaic."""
    _validate_spec(spec)
    x_cells = round((spec.x_max_m - spec.x_min_m) / spec.cell_size_m)
    y_cells = round((spec.y_max_m - spec.y_min_m) / spec.cell_size_m)
    x_coords = spec.x_min_m + np.arange(x_cells + 1, dtype=np.float64) * spec.cell_size_m
    y_coords = spec.y_min_m + np.arange(y_cells + 1, dtype=np.float64) * spec.cell_size_m
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")

    azimuth_rad = np.deg2rad(spec.azimuth_deg)
    uphill = np.asarray((np.cos(azimuth_rad), np.sin(azimuth_rad)), dtype=np.float64)
    gradient = np.tan(np.deg2rad(spec.slope_deg)) * uphill
    base = gradient[0] * xx + gradient[1] * yy
    normal = np.asarray((-gradient[0], -gradient[1], 1.0), dtype=np.float64)
    normal /= np.linalg.norm(normal)

    seed_sequence = np.random.SeedSequence(spec.seed)
    residual_seed, material_seed = seed_sequence.spawn(2)
    residual_rng = np.random.default_rng(residual_seed)
    residual = spec.residual_amplitude_m * _unit_scale(
        _smooth(residual_rng.standard_normal(base.shape), spec.residual_smoothing_passes)
    )
    material_rng = np.random.default_rng(material_seed)
    material_score = _smooth(material_rng.standard_normal((x_cells, y_cells)), 2)
    quantiles = np.quantile(material_score, np.linspace(0.0, 1.0, len(spec.static_friction) + 1)[1:-1])
    materials = np.digitize(material_score, quantiles, right=False).astype(np.int16)

    base_f32 = base.astype(np.float32)
    residual_f32 = residual.astype(np.float32)
    return SlopeTerrainField(
        spec=spec,
        x_coords_m=x_coords,
        y_coords_m=y_coords,
        base_heights_m=base_f32,
        residual_heights_m=residual_f32,
        heights_m=base_f32 + residual_f32,
        material_indices=materials,
        uphill_xy=uphill,
        analytic_normal=normal,
    )


def terrain_axes(azimuth_deg: float) -> TerrainAxes:
    """Return uphill, downhill, and signed contour axes for one slope azimuth."""
    azimuth_rad = np.deg2rad(azimuth_deg)
    uphill = np.asarray((np.cos(azimuth_rad), np.sin(azimuth_rad)), dtype=np.float64)
    contour_left = np.asarray((-uphill[1], uphill[0]), dtype=np.float64)
    return TerrainAxes(
        uphill_xy=uphill,
        downhill_xy=-uphill,
        contour_left_xy=contour_left,
        contour_right_xy=-contour_left,
    )


def build_mesh_arrays(field: SlopeTerrainField) -> dict[str, np.ndarray]:
    """Return stable triangle mesh arrays and a material ID for each face."""
    nx = field.x_coords_m.size
    ny = field.y_coords_m.size
    xx, yy = np.meshgrid(field.x_coords_m, field.y_coords_m, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), field.heights_m.ravel())).astype(np.float32)
    i = np.arange(nx - 1, dtype=np.int64)[:, None]
    j = np.arange(ny - 1, dtype=np.int64)[None, :]
    p00 = i * ny + j
    p10 = (i + 1) * ny + j
    p01 = i * ny + j + 1
    p11 = (i + 1) * ny + j + 1
    first = np.stack((p00, p10, p11), axis=-1)
    second = np.stack((p00, p11, p01), axis=-1)
    faces = np.stack((first, second), axis=-2).reshape(-1, 3).astype(np.int64)
    face_materials = np.repeat(field.material_indices[..., None], 2, axis=-1).reshape(-1).astype(np.int16)
    return {"points": points, "faces": faces, "face_materials": face_materials}


def mesh_arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash array names, dtypes, shapes, and bytes for reproducibility evidence."""
    digest = hashlib.sha256()
    for name in ("points", "faces", "face_materials"):
        values = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def triangle_normals(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Compute the simulation-only upward normal of every generated triangle."""
    points = np.asarray(arrays["points"], dtype=np.float64)
    faces = np.asarray(arrays["faces"], dtype=np.int64)
    edges_a = points[faces[:, 1]] - points[faces[:, 0]]
    edges_b = points[faces[:, 2]] - points[faces[:, 0]]
    normals = np.cross(edges_a, edges_b)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("mesh contains a degenerate triangle")
    normals = normals / lengths[:, None]
    normals[normals[:, 2] < 0.0] *= -1.0
    return normals


def friction_limit_ratio(slope_deg: float, static_friction: float) -> tuple[float, bool]:
    """Return tan(theta)/mu_s and the PRD physical-limit stress classification."""
    if not 0.0 <= slope_deg < 90.0:
        raise ValueError("slope_deg must satisfy 0 <= slope_deg < 90")
    if not np.isfinite(slope_deg) or not np.isfinite(static_friction):
        raise ValueError("slope_deg and static_friction must be finite")
    if static_friction < 0.0:
        raise ValueError("static_friction must be non-negative")
    ratio = float("inf") if static_friction == 0.0 else float(np.tan(np.deg2rad(slope_deg)) / static_friction)
    return ratio, ratio >= 1.0
