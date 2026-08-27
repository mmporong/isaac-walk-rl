"""Isaac Lab spawner for the deterministic G009 slope field."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

from .terrain import (
    DEFAULT_DYNAMIC_FRICTION,
    DEFAULT_STATIC_FRICTION,
    SlopeTerrainSpec,
    build_mesh_arrays,
    generate_slope_terrain,
)


G009_SLOPE_FIELD_PRIM = "/World/g009_slope_field"
G009_SLOPE_SURFACE_PRIM = f"{G009_SLOPE_FIELD_PRIM}/surface"


@configclass
class SlopeTerrainFieldCfg(SpawnerCfg):
    """Parameters for one world-space, static G009 slope mesh."""

    func: Callable = None
    # S0 evaluates 32 trials per cell; this 56 m field covers that grid with
    # margin without constructing the multi-million-face production mesh.
    x_min_m: float = -28.0
    x_max_m: float = 28.0
    y_min_m: float = -28.0
    y_max_m: float = 28.0
    cell_size_m: float = 0.25
    slope_deg: float = 0.0
    azimuth_deg: float = 0.0
    seed: int = 20260828
    residual_amplitude_m: float = 0.0
    residual_smoothing_passes: int = 3
    static_friction: tuple[float, ...] = DEFAULT_STATIC_FRICTION
    dynamic_friction: tuple[float, ...] = DEFAULT_DYNAMIC_FRICTION

    def __post_init__(self):
        self.func = spawn_slope_terrain


def terrain_spec_from_cfg(cfg: Any) -> SlopeTerrainSpec:
    """Create the import-light terrain specification stored by the simulator cfg."""
    return SlopeTerrainSpec(
        x_min_m=cfg.x_min_m,
        x_max_m=cfg.x_max_m,
        y_min_m=cfg.y_min_m,
        y_max_m=cfg.y_max_m,
        cell_size_m=cfg.cell_size_m,
        slope_deg=cfg.slope_deg,
        azimuth_deg=cfg.azimuth_deg,
        seed=cfg.seed,
        residual_amplitude_m=cfg.residual_amplitude_m,
        residual_smoothing_passes=cfg.residual_smoothing_passes,
        static_friction=tuple(cfg.static_friction),
        dynamic_friction=tuple(cfg.dynamic_friction),
    )


@lru_cache(maxsize=8)
def terrain_artifacts(spec: SlopeTerrainSpec):
    """Reuse deterministic field and mesh arrays across spawn and reset events."""
    field = generate_slope_terrain(spec)
    return field, build_mesh_arrays(field)


def spawn_slope_terrain(
    prim_path: str,
    cfg: SlopeTerrainFieldCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Any:
    """Spawn one static USD mesh used for both collision and ray casting."""
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.stage as stage_utils
    from pxr import UsdGeom, UsdPhysics

    if prim_path != G009_SLOPE_FIELD_PRIM:
        raise ValueError(f"G009 slope field must use {G009_SLOPE_FIELD_PRIM!r}: {prim_path!r}")
    if translation not in (None, (0.0, 0.0, 0.0)) or orientation not in (None, (1.0, 0.0, 0.0, 0.0)):
        raise ValueError("G009 slope field is defined directly in world coordinates")
    if len(cfg.static_friction) != 1 or len(cfg.dynamic_friction) != 1:
        raise ValueError("S0 uses one nominal ground material; spatial friction is introduced in a later stage")

    _, arrays = terrain_artifacts(terrain_spec_from_cfg(cfg))
    stage = stage_utils.get_current_stage()
    parent = UsdGeom.Xform.Define(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, G009_SLOPE_SURFACE_PRIM)
    mesh.CreatePointsAttr(arrays["points"].tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(arrays["faces"]))
    mesh.CreateFaceVertexIndicesAttr(arrays["faces"].ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    display_color = mesh.CreateDisplayColorAttr([(0.43, 0.38, 0.30)])
    UsdGeom.Primvar(display_color).SetInterpolation(UsdGeom.Tokens.constant)

    sim_utils.define_collision_properties(
        G009_SLOPE_SURFACE_PRIM,
        sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        stage,
    )
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")

    material_path = f"{prim_path}/nominal_material"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=cfg.static_friction[0],
        dynamic_friction=cfg.dynamic_friction[0],
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )
    material_cfg.func(material_path, material_cfg)
    sim_utils.bind_physics_material(G009_SLOPE_SURFACE_PRIM, material_path, stage)
    return parent.GetPrim()


__all__ = [
    "G009_SLOPE_FIELD_PRIM",
    "G009_SLOPE_SURFACE_PRIM",
    "SlopeTerrainFieldCfg",
    "spawn_slope_terrain",
    "terrain_artifacts",
    "terrain_spec_from_cfg",
]
