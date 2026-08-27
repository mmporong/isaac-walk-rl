from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_walk_g009.terrain import (  # noqa: E402
    SPATIAL_DYNAMIC_FRICTION,
    SPATIAL_STATIC_FRICTION,
    SlopeTerrainSpec,
    build_mesh_arrays,
    friction_limit_ratio,
    generate_slope_terrain,
    mesh_arrays_sha256,
    terrain_axes,
    triangle_normals,
)


@pytest.mark.parametrize("slope_deg", (0.0, 5.0, 10.0, 15.0, 20.0, 25.0))
def test_base_plane_has_requested_slope_and_upward_normal(slope_deg: float) -> None:
    field = generate_slope_terrain(SlopeTerrainSpec(slope_deg=slope_deg, azimuth_deg=0.0))
    dz_dx = np.diff(field.base_heights_m, axis=0) / field.spec.cell_size_m
    measured = np.degrees(np.arctan(float(np.mean(dz_dx))))
    assert measured == pytest.approx(slope_deg, abs=0.1)
    assert field.analytic_normal[2] > 0.0
    assert field.analytic_normal == pytest.approx((-np.sin(np.deg2rad(slope_deg)), 0.0, np.cos(np.deg2rad(slope_deg))))


@pytest.mark.parametrize(
    ("azimuth_deg", "expected_uphill"),
    ((0.0, (1.0, 0.0)), (90.0, (0.0, 1.0)), (180.0, (-1.0, 0.0)), (270.0, (0.0, -1.0))),
)
def test_azimuth_controls_uphill_axis_and_sign(azimuth_deg, expected_uphill) -> None:
    field = generate_slope_terrain(SlopeTerrainSpec(slope_deg=15.0, azimuth_deg=azimuth_deg))
    assert field.uphill_xy == pytest.approx(expected_uphill, abs=1e-12)


def test_direction_axes_have_the_preregistered_signs() -> None:
    axes = terrain_axes(0.0)
    assert axes.uphill_xy == pytest.approx((1.0, 0.0))
    assert axes.downhill_xy == pytest.approx((-1.0, 0.0))
    assert axes.contour_left_xy == pytest.approx((0.0, 1.0))
    assert axes.contour_right_xy == pytest.approx((0.0, -1.0))


def test_residual_is_separate_and_seeded_deterministically() -> None:
    spec = SlopeTerrainSpec(
        slope_deg=12.0,
        azimuth_deg=35.0,
        residual_amplitude_m=0.04,
        seed=7,
        static_friction=SPATIAL_STATIC_FRICTION,
        dynamic_friction=SPATIAL_DYNAMIC_FRICTION,
    )
    first = generate_slope_terrain(spec)
    duplicate = generate_slope_terrain(spec)
    changed = generate_slope_terrain(SlopeTerrainSpec(**{**spec.__dict__, "seed": 8}))
    assert np.array_equal(first.base_heights_m, changed.base_heights_m)
    assert np.array_equal(first.residual_heights_m, duplicate.residual_heights_m)
    assert not np.array_equal(first.residual_heights_m, changed.residual_heights_m)
    assert not np.array_equal(first.material_indices, changed.material_indices)
    assert first.heights_m == pytest.approx(first.base_heights_m + first.residual_heights_m, abs=1e-7)
    assert np.max(np.abs(first.residual_heights_m)) == pytest.approx(0.04, abs=1e-6)


def test_mesh_arrays_and_hash_are_reproducible() -> None:
    field = generate_slope_terrain(SlopeTerrainSpec(x_min_m=-1, x_max_m=1, y_min_m=-1, y_max_m=1, seed=2))
    arrays = build_mesh_arrays(field)
    assert arrays["points"].shape == (81, 3)
    assert arrays["faces"].shape == (128, 3)
    assert arrays["face_materials"].shape == (128,)
    assert mesh_arrays_sha256(arrays) == mesh_arrays_sha256(build_mesh_arrays(field))
    assert len(mesh_arrays_sha256(arrays)) == 64


def test_triangle_normals_match_the_analytic_base_plane() -> None:
    field = generate_slope_terrain(SlopeTerrainSpec(slope_deg=20.0, azimuth_deg=90.0))
    normals = triangle_normals(build_mesh_arrays(field))
    assert normals == pytest.approx(np.broadcast_to(field.analytic_normal, normals.shape), abs=1e-6)


def test_nominal_material_and_physical_limit_classification() -> None:
    field = generate_slope_terrain(SlopeTerrainSpec(slope_deg=20.0))
    assert field.spec.static_friction == (0.8,)
    assert field.spec.dynamic_friction == (0.6,)
    ratio, stress = friction_limit_ratio(20.0, 0.2)
    assert ratio == pytest.approx(np.tan(np.deg2rad(20.0)) / 0.2)
    assert stress
    _, nominal_stress = friction_limit_ratio(20.0, 0.8)
    assert not nominal_stress


def test_invalid_friction_or_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="static >= dynamic"):
        generate_slope_terrain(SlopeTerrainSpec(static_friction=(0.2,), dynamic_friction=(0.3,)))
    with pytest.raises(ValueError, match="integer multiple"):
        generate_slope_terrain(SlopeTerrainSpec(x_min_m=0.0, x_max_m=1.1, cell_size_m=0.25))
    with pytest.raises(ValueError, match="static >= dynamic"):
        generate_slope_terrain(SlopeTerrainSpec(static_friction=(float("nan"),), dynamic_friction=(0.3,)))
    with pytest.raises(ValueError, match="finite"):
        friction_limit_ratio(20.0, float("nan"))
    with pytest.raises(ValueError, match="terrain geometry.*finite"):
        generate_slope_terrain(SlopeTerrainSpec(azimuth_deg=float("nan")))
    with pytest.raises(ValueError, match="smoothing_passes must be an integer"):
        generate_slope_terrain(SlopeTerrainSpec(residual_smoothing_passes=1.5))
