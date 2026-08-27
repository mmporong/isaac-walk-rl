"""Import-light geometry primitives for the G009 mountain-slope stages."""

from .support_plane import (
    PlaneFitResult,
    SupportPlaneEstimate,
    SupportRegionResult,
    body_tilt_deg,
    classify_support_region,
    contact_gated_tangent_speeds,
    estimate_support_plane,
    fit_support_plane,
    normal_angle_error_deg,
    project_to_tangent,
    whole_body_com,
)
from .terrain import (
    DEFAULT_DYNAMIC_FRICTION,
    DEFAULT_STATIC_FRICTION,
    SPATIAL_DYNAMIC_FRICTION,
    SPATIAL_STATIC_FRICTION,
    SlopeTerrainField,
    SlopeTerrainSpec,
    TerrainAxes,
    build_mesh_arrays,
    friction_limit_ratio,
    generate_slope_terrain,
    mesh_arrays_sha256,
    terrain_axes,
    triangle_normals,
)


def register_tasks() -> None:
    """Register Isaac Lab tasks without making geometry-only imports depend on Isaac Sim."""
    from .registry import register_tasks as _register_tasks

    _register_tasks()

__all__ = [
    "DEFAULT_DYNAMIC_FRICTION",
    "DEFAULT_STATIC_FRICTION",
    "SPATIAL_DYNAMIC_FRICTION",
    "SPATIAL_STATIC_FRICTION",
    "PlaneFitResult",
    "SupportPlaneEstimate",
    "SlopeTerrainField",
    "SlopeTerrainSpec",
    "SupportRegionResult",
    "TerrainAxes",
    "body_tilt_deg",
    "build_mesh_arrays",
    "classify_support_region",
    "contact_gated_tangent_speeds",
    "estimate_support_plane",
    "fit_support_plane",
    "friction_limit_ratio",
    "generate_slope_terrain",
    "mesh_arrays_sha256",
    "normal_angle_error_deg",
    "project_to_tangent",
    "register_tasks",
    "terrain_axes",
    "triangle_normals",
    "whole_body_com",
]
