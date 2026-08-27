from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_walk_g009.support_plane import (  # noqa: E402
    body_tilt_deg,
    classify_support_region,
    contact_gated_tangent_speeds,
    estimate_support_plane,
    fit_support_plane,
    normal_angle_error_deg,
    project_to_tangent,
    whole_body_com,
)


def _plane_points(slope_deg=20.0):
    gradient = np.tan(np.deg2rad(slope_deg))
    xy = np.asarray(((-0.2, -0.1), (0.2, -0.1), (0.2, 0.1), (-0.2, 0.1)))
    return np.column_stack((xy, gradient * xy[:, 0]))


def test_three_point_plane_fit_is_upward_and_exact() -> None:
    result = fit_support_plane(_plane_points()[:3])
    expected = np.asarray((-np.sin(np.deg2rad(20.0)), 0.0, np.cos(np.deg2rad(20.0))))
    assert result.normal == pytest.approx(expected, abs=1e-8)
    assert result.fit_sample_count == 3
    assert not result.used_fallback


def test_robust_fit_rejects_one_outlier_within_one_degree() -> None:
    samples = np.vstack((_plane_points(), (0.0, 0.0, 0.30)))
    result = fit_support_plane(samples, outlier_threshold_m=0.01)
    expected = np.asarray((-np.sin(np.deg2rad(20.0)), 0.0, np.cos(np.deg2rad(20.0))))
    assert normal_angle_error_deg(result.normal, expected) < 1.0
    assert result.fit_sample_count == 4
    assert not result.inlier_mask[-1]


def test_insufficient_and_degenerate_samples_preserve_previous_normal() -> None:
    previous = np.asarray((-0.2, 0.1, 0.97))
    previous /= np.linalg.norm(previous)
    insufficient = fit_support_plane(np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))), previous_normal=previous)
    degenerate = fit_support_plane(np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))), previous_normal=previous)
    assert insufficient.used_fallback and insufficient.fallback_reason == "insufficient_samples"
    assert degenerate.used_fallback and degenerate.fallback_reason == "degenerate_samples"
    assert not insufficient.quality_valid
    assert not degenerate.quality_valid
    assert insufficient.normal == pytest.approx(previous)
    assert degenerate.normal == pytest.approx(previous)


def test_tangent_projection_angle_error_and_body_tilt() -> None:
    normal = np.asarray((0.0, -np.sin(np.deg2rad(15.0)), np.cos(np.deg2rad(15.0))))
    projected = project_to_tangent(np.asarray((1.0, 2.0, 3.0)), normal)
    assert np.dot(projected, normal) == pytest.approx(0.0, abs=1e-12)
    assert normal_angle_error_deg((0.0, 0.0, 1.0), normal) == pytest.approx(15.0)
    assert body_tilt_deg((0.0, 0.0, 1.0), normal) == pytest.approx(15.0)
    assert body_tilt_deg((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)) == pytest.approx(180.0)


def test_support_polygon_classifies_inside_edge_and_outside() -> None:
    contacts = np.asarray(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)))
    normal = np.asarray((0.0, 0.0, 1.0))
    inside = classify_support_region(contacts, np.asarray((0.0, 0.0, 1.0)), normal)
    edge = classify_support_region(contacts, np.asarray((1.0, 0.0, 1.0)), normal)
    outside = classify_support_region(contacts, np.asarray((1.1, 0.0, 1.0)), normal)
    assert (inside.classification, inside.blocking) == ("inside", False)
    assert (edge.classification, edge.blocking) == ("edge", False)
    assert (outside.classification, outside.blocking) == ("outside", True)


def test_two_point_support_is_nonblocking_segment_diagnostic() -> None:
    contacts = np.asarray(((-0.2, 0.0, 0.0), (0.2, 0.0, 0.0)))
    result = classify_support_region(contacts, np.asarray((0.0, 0.1, 0.0)), np.asarray((0.0, 0.0, 1.0)))
    assert result.classification == "segment"
    assert result.contact_count == 2
    assert not result.blocking


def test_slip_speed_excludes_noncontact_feet() -> None:
    velocities = np.asarray(((1.0, 0.0, 1.0), (4.0, 0.0, 0.0)))
    speeds = contact_gated_tangent_speeds(velocities, np.asarray((True, False)), np.asarray((0.0, 0.0, 1.0)))
    assert speeds == pytest.approx((1.0, 0.0))


def test_whole_body_com_uses_runtime_link_mass_weighting() -> None:
    masses = np.asarray((1.0, 3.0))
    positions = np.asarray(((0.0, 0.0, 0.0), (4.0, 2.0, 0.0)))
    assert whole_body_com(masses, positions) == pytest.approx((3.0, 1.5, 0.0))


def test_nearly_collinear_samples_are_invalid_and_preserve_continuity() -> None:
    previous = np.asarray((0.0, 0.0, 1.0))
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0e-8, 0.0), (2.0, 0.0, 1.0e-8)))
    result = fit_support_plane(points, previous_normal=previous)
    assert result.used_fallback
    assert result.fallback_reason == "degenerate_samples"
    assert not result.quality_valid


def test_nonfinite_normals_and_queries_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        project_to_tangent(np.asarray((1.0, 0.0, 0.0)), np.asarray((np.nan, 0.0, 1.0)))
    contacts = np.asarray(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)))
    with pytest.raises(ValueError, match="query_xyz must be finite"):
        classify_support_region(contacts, np.asarray((np.nan, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0)))


def test_terrain_samples_replace_insufficient_contact_proxies_with_quality_metadata() -> None:
    contacts = _plane_points()[:2]
    terrain = _plane_points()
    estimate = estimate_support_plane(contacts, terrain)
    assert estimate.quality_valid
    assert estimate.used_sample_fallback
    assert estimate.sample_source == "terrain_raycast_fallback"
    assert estimate.contact_sample_count == 2
    assert estimate.terrain_sample_count == 4
    assert estimate.plane.sample_source == "terrain_raycast_fallback"


def test_robust_fit_bounds_hypotheses_for_dense_scanner_samples() -> None:
    xy = np.column_stack((np.linspace(-1.0, 1.0, 256), np.sin(np.linspace(0.0, 8.0, 256))))
    samples = np.column_stack((xy, 0.2 * xy[:, 0] - 0.1 * xy[:, 1]))
    result = fit_support_plane(samples, max_hypotheses=64)
    assert result.quality_valid
    assert result.hypothesis_count == 64
