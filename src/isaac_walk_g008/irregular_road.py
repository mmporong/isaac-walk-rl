"""Deterministic 2-D friction and road-height field for the G008 irregular-road stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


IRREGULAR_ROAD_PRIM = "/World/irregular_road_field"
IRREGULAR_ROAD_MESH_PRIM = f"{IRREGULAR_ROAD_PRIM}/surface"
DEFAULT_STATIC_FRICTION = (0.25, 0.40, 0.60, 0.80)
DEFAULT_DYNAMIC_FRICTION = (0.15, 0.28, 0.45, 0.60)
DEFAULT_COLORS_RGB = (
    (0.16, 0.34, 0.62),
    (0.24, 0.48, 0.45),
    (0.56, 0.49, 0.30),
    (0.45, 0.43, 0.40),
)


@dataclass(frozen=True)
class IrregularRoadSpec:
    """Parameters that fully determine one irregular road field."""

    x_min_m: float = -48.0
    x_max_m: float = 48.0
    y_min_m: float = -48.0
    y_max_m: float = 48.0
    cell_size_m: float = 0.25
    seed: int = 20260826
    env_spacing_m: float = 4.0
    static_friction: tuple[float, ...] = DEFAULT_STATIC_FRICTION
    dynamic_friction: tuple[float, ...] = DEFAULT_DYNAMIC_FRICTION
    colors_rgb: tuple[tuple[float, float, float], ...] = DEFAULT_COLORS_RGB
    crown_height_m: float = 0.015
    undulation_amplitude_m: float = 0.030
    roughness_amplitude_m: float = 0.012
    pothole_depth_m: float = 0.025


@dataclass(frozen=True)
class IrregularRoadField:
    """Generated arrays used by both the USD spawner and evaluation metrics."""

    spec: IrregularRoadSpec
    x_coords_m: np.ndarray
    y_coords_m: np.ndarray
    heights_m: np.ndarray
    material_indices: np.ndarray
    local_slope_deg: np.ndarray


def validate_spec(spec: IrregularRoadSpec) -> None:
    if spec.cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    if spec.x_max_m <= spec.x_min_m or spec.y_max_m <= spec.y_min_m:
        raise ValueError("road bounds must have positive area")
    x_cells = (spec.x_max_m - spec.x_min_m) / spec.cell_size_m
    y_cells = (spec.y_max_m - spec.y_min_m) / spec.cell_size_m
    if not np.isclose(x_cells, round(x_cells)) or not np.isclose(y_cells, round(y_cells)):
        raise ValueError("road bounds must be an integer multiple of cell_size_m")
    if not spec.static_friction or len(spec.static_friction) != len(spec.dynamic_friction):
        raise ValueError("static and dynamic friction buckets must have equal non-zero length")
    if len(spec.colors_rgb) != len(spec.static_friction):
        raise ValueError("one display color is required for every friction bucket")
    if any(dynamic < 0.0 or static < dynamic for static, dynamic in zip(spec.static_friction, spec.dynamic_friction)):
        raise ValueError("each friction bucket must satisfy static >= dynamic >= 0")
    if tuple(spec.static_friction) != tuple(sorted(spec.static_friction)):
        raise ValueError("static friction buckets must be sorted")


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
    maximum = float(np.max(np.abs(centered)))
    if maximum == 0.0:
        return np.zeros_like(centered)
    return centered / maximum


def generate_irregular_road(spec: IrregularRoadSpec) -> IrregularRoadField:
    """Generate road-like heights and a non-periodic, correlated 2-D friction mosaic."""
    validate_spec(spec)
    x_cells = round((spec.x_max_m - spec.x_min_m) / spec.cell_size_m)
    y_cells = round((spec.y_max_m - spec.y_min_m) / spec.cell_size_m)
    x_coords = spec.x_min_m + np.arange(x_cells + 1, dtype=np.float64) * spec.cell_size_m
    y_coords = spec.y_min_m + np.arange(y_cells + 1, dtype=np.float64) * spec.cell_size_m
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
    rng = np.random.default_rng(spec.seed)

    # A road crown repeats around each simulation origin row, while long waves,
    # smoothed aggregate roughness, and shallow potholes break periodicity.
    local_y = np.remainder(yy + 0.5 * spec.env_spacing_m, spec.env_spacing_m) - 0.5 * spec.env_spacing_m
    crown = spec.crown_height_m * (1.0 - np.abs(local_y) / (0.5 * spec.env_spacing_m))
    long_wave = spec.undulation_amplitude_m * (
        0.52 * np.sin(2.0 * np.pi * xx / 6.7 + 0.35)
        + 0.31 * np.sin(2.0 * np.pi * (0.72 * xx + 0.38 * yy) / 4.9 - 0.8)
        + 0.17 * np.sin(2.0 * np.pi * yy / 7.9 + 1.4)
    )
    aggregate = _unit_scale(_smooth(rng.standard_normal((x_cells + 1, y_cells + 1)), 4))
    roughness = spec.roughness_amplitude_m * aggregate
    depression_source = _unit_scale(_smooth(rng.standard_normal((x_cells + 1, y_cells + 1)), 7))
    potholes = -spec.pothole_depth_m * np.square(np.clip((-depression_source - 0.35) / 0.65, 0.0, 1.0))
    heights = crown + long_wave + roughness + potholes
    heights -= float(np.mean(heights))

    # Coarse correlation makes patches road-like; a fine component keeps the
    # four feet from always seeing the same material inside one body footprint.
    coarse = _unit_scale(_smooth(rng.standard_normal((x_cells, y_cells)), 3))
    fine = _unit_scale(_smooth(rng.standard_normal((x_cells, y_cells)), 1))
    score = 0.62 * coarse + 0.38 * fine
    quantiles = np.quantile(score, np.linspace(0.0, 1.0, len(spec.static_friction) + 1)[1:-1])
    materials = np.digitize(score, quantiles, right=False).astype(np.int16)

    dz_dx = np.diff(heights, axis=0)[:, :-1] / spec.cell_size_m
    dz_dy = np.diff(heights, axis=1)[:-1, :] / spec.cell_size_m
    slopes = np.degrees(np.arctan(np.sqrt(np.square(dz_dx) + np.square(dz_dy))))
    return IrregularRoadField(
        spec=spec,
        x_coords_m=x_coords,
        y_coords_m=y_coords,
        heights_m=heights.astype(np.float32),
        material_indices=materials,
        local_slope_deg=slopes.astype(np.float32),
    )


def field_summary(field: IrregularRoadField) -> dict[str, Any]:
    """Return deterministic field statistics suitable for JSON reports."""
    spec = field.spec
    counts = np.bincount(field.material_indices.ravel(), minlength=len(spec.static_friction))
    adjacent_height_delta = max(
        float(np.max(np.abs(np.diff(field.heights_m, axis=0)))),
        float(np.max(np.abs(np.diff(field.heights_m, axis=1)))),
    )
    return {
        "seed": spec.seed,
        "bounds_m": {
            "x": [spec.x_min_m, spec.x_max_m],
            "y": [spec.y_min_m, spec.y_max_m],
        },
        "cell_size_m": spec.cell_size_m,
        "grid_cells": [int(field.material_indices.shape[0]), int(field.material_indices.shape[1])],
        "friction_buckets": [
            {
                "index": index,
                "static": static,
                "dynamic": dynamic,
                "cell_count": int(counts[index]),
                "cell_ratio": float(counts[index] / counts.sum()),
            }
            for index, (static, dynamic) in enumerate(zip(spec.static_friction, spec.dynamic_friction))
        ],
        "height_min_m": float(np.min(field.heights_m)),
        "height_max_m": float(np.max(field.heights_m)),
        "height_range_m": float(np.ptp(field.heights_m)),
        "adjacent_height_delta_max_m": adjacent_height_delta,
        "local_slope_mean_deg": float(np.mean(field.local_slope_deg)),
        "local_slope_max_deg": float(np.max(field.local_slope_deg)),
        "components": {
            "road_crown_height_m": spec.crown_height_m,
            "undulation_amplitude_m": spec.undulation_amplitude_m,
            "roughness_amplitude_m": spec.roughness_amplitude_m,
            "pothole_depth_m": spec.pothole_depth_m,
        },
    }


def sample_cells(field: IrregularRoadField, xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample material, bilinear height, and cell slope at ``[..., 2]`` world positions."""
    positions = np.asarray(xy_m, dtype=np.float64)
    if positions.shape[-1] != 2:
        raise ValueError("xy_m must have a final dimension of two")
    spec = field.spec
    fx = (positions[..., 0] - spec.x_min_m) / spec.cell_size_m
    fy = (positions[..., 1] - spec.y_min_m) / spec.cell_size_m
    ix = np.floor(fx).astype(np.int64)
    iy = np.floor(fy).astype(np.int64)
    if np.any(ix < 0) or np.any(iy < 0) or np.any(ix >= field.material_indices.shape[0]) or np.any(
        iy >= field.material_indices.shape[1]
    ):
        raise ValueError("sample position lies outside the irregular-road field")
    tx = fx - ix
    ty = fy - iy
    h00 = field.heights_m[ix, iy]
    h10 = field.heights_m[ix + 1, iy]
    h01 = field.heights_m[ix, iy + 1]
    h11 = field.heights_m[ix + 1, iy + 1]
    height = (
        (1.0 - tx) * (1.0 - ty) * h00
        + tx * (1.0 - ty) * h10
        + (1.0 - tx) * ty * h01
        + tx * ty * h11
    )
    return field.material_indices[ix, iy], height.astype(np.float32), field.local_slope_deg[ix, iy]


def build_mesh_arrays(field: IrregularRoadField) -> dict[str, np.ndarray]:
    """Convert a generated field to triangle-mesh arrays and per-face material IDs."""
    nx = field.x_coords_m.size
    ny = field.y_coords_m.size
    xx, yy = np.meshgrid(field.x_coords_m, field.y_coords_m, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), field.heights_m.ravel())).astype(np.float32)
    i = np.arange(nx - 1, dtype=np.int64)[:, None]
    j = np.arange(ny - 1, dtype=np.int64)[None, :]
    p00 = i * ny + j
    p10 = (i + 1) * ny + j
    p01 = i * ny + (j + 1)
    p11 = (i + 1) * ny + (j + 1)
    first = np.stack((p00, p10, p11), axis=-1)
    second = np.stack((p00, p11, p01), axis=-1)
    faces = np.stack((first, second), axis=-2).reshape(-1, 3).astype(np.int64)
    face_materials = np.repeat(field.material_indices[..., None], 2, axis=-1).reshape(-1).astype(np.int16)
    return {"points": points, "faces": faces, "face_materials": face_materials}


def spawn_irregular_road_field(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Any:
    """Spawn one collision mesh with a physics-material subset for every friction bucket."""
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.stage as stage_utils
    from pxr import UsdGeom, UsdPhysics

    del translation, orientation
    spec = IrregularRoadSpec(
        x_min_m=cfg.x_min_m,
        x_max_m=cfg.x_max_m,
        y_min_m=cfg.y_min_m,
        y_max_m=cfg.y_max_m,
        cell_size_m=cfg.cell_size_m,
        seed=cfg.seed,
        env_spacing_m=cfg.env_spacing_m,
        static_friction=tuple(cfg.static_friction),
        dynamic_friction=tuple(cfg.dynamic_friction),
        colors_rgb=tuple(tuple(color) for color in cfg.colors_rgb),
        crown_height_m=cfg.crown_height_m,
        undulation_amplitude_m=cfg.undulation_amplitude_m,
        roughness_amplitude_m=cfg.roughness_amplitude_m,
        pothole_depth_m=cfg.pothole_depth_m,
    )
    field = generate_irregular_road(spec)
    arrays = build_mesh_arrays(field)
    stage = stage_utils.get_current_stage()
    parent = UsdGeom.Xform.Define(stage, prim_path)
    material_paths: list[str] = []
    for index, (static, dynamic) in enumerate(zip(spec.static_friction, spec.dynamic_friction)):
        material_path = f"{prim_path}/friction_{index}_material"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=static,
            dynamic_friction=dynamic,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        material_cfg.func(material_path, material_cfg)
        material_paths.append(material_path)

    for material_index, material_path in enumerate(material_paths):
        selected_faces = arrays["faces"][arrays["face_materials"] == material_index]
        used_points, compact_faces = np.unique(selected_faces.ravel(), return_inverse=True)
        mesh_path = f"{prim_path}/friction_{material_index}_surface"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh.CreatePointsAttr(arrays["points"][used_points].tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(selected_faces))
        mesh.CreateFaceVertexIndicesAttr(compact_faces.tolist())
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(False)
        display_color = mesh.CreateDisplayColorAttr([spec.colors_rgb[material_index]])
        UsdGeom.Primvar(display_color).SetInterpolation(UsdGeom.Tokens.constant)
        sim_utils.define_collision_properties(
            mesh_path,
            sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            stage,
        )
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
        sim_utils.bind_physics_material(mesh_path, material_path, stage)

    # RayCaster 2.1.1 accepts exactly one static mesh. This full duplicate has
    # no CollisionAPI, while the four disjoint material meshes above own all contacts.
    scan_mesh = UsdGeom.Mesh.Define(stage, IRREGULAR_ROAD_MESH_PRIM)
    scan_mesh.CreatePointsAttr(arrays["points"].tolist())
    scan_mesh.CreateFaceVertexCountsAttr([3] * len(arrays["faces"]))
    scan_mesh.CreateFaceVertexIndicesAttr(arrays["faces"].ravel().tolist())
    scan_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    scan_mesh.CreateDoubleSidedAttr(False)
    return parent.GetPrim()
