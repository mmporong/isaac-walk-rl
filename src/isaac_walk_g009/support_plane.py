"""Support-plane and support-region geometry without Isaac or Torch imports."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np


_EPS = 1.0e-9
_MIN_PLANE_SPREAD_RATIO = 1.0e-6


@dataclass(frozen=True)
class PlaneFitResult:
    point: np.ndarray
    normal: np.ndarray
    residuals_m: np.ndarray
    inlier_mask: np.ndarray
    valid_sample_count: int
    fit_sample_count: int
    used_fallback: bool
    fallback_reason: str | None
    sample_source: str = "contact_proxy"
    hypothesis_count: int = 0

    @property
    def quality_valid(self) -> bool:
        return not self.used_fallback and self.fit_sample_count >= 3


@dataclass(frozen=True)
class SupportRegionResult:
    classification: str
    signed_margin_m: float
    blocking: bool
    contact_count: int


@dataclass(frozen=True)
class SupportPlaneEstimate:
    plane: PlaneFitResult
    sample_source: str
    used_sample_fallback: bool
    contact_sample_count: int
    terrain_sample_count: int

    @property
    def quality_valid(self) -> bool:
        return self.plane.quality_valid


def _unit(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("vector must be a finite 3-vector")
    length = float(np.linalg.norm(result))
    if length <= _EPS:
        raise ValueError("vector must be non-zero")
    return result / length


def _upward(normal: np.ndarray) -> np.ndarray:
    result = _unit(normal)
    return -result if result[2] < 0.0 else result


def _svd_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    _, singular_values, vh = np.linalg.svd(points - center, full_matrices=False)
    if (
        singular_values.size < 2
        or singular_values[0] <= _EPS
        or singular_values[1] <= _EPS
        or singular_values[1] / singular_values[0] <= _MIN_PLANE_SPREAD_RATIO
    ):
        raise ValueError("support samples are collinear or coincident")
    return center, _upward(vh[-1])


def _candidate_triples(sample_count: int, max_hypotheses: int) -> list[tuple[int, int, int]]:
    if max_hypotheses <= 0:
        raise ValueError("max_hypotheses must be positive")
    total = math.comb(sample_count, 3)
    if total <= max_hypotheses:
        return list(combinations(range(sample_count), 3))
    # A fixed RNG makes the bounded RANSAC surface byte-for-byte reproducible.
    rng = np.random.default_rng(0)
    selected: set[tuple[int, int, int]] = set()
    anchors = np.linspace(0, sample_count - 1, min(sample_count, 16), dtype=int)
    for first, second, third in combinations(np.unique(anchors).tolist(), 3):
        selected.add((first, second, third))
        if len(selected) >= max_hypotheses:
            return sorted(selected)
    while len(selected) < max_hypotheses:
        selected.add(tuple(sorted(int(value) for value in rng.choice(sample_count, 3, replace=False))))
    return sorted(selected)


def fit_support_plane(
    points_xyz: np.ndarray,
    *,
    previous_normal: np.ndarray | None = None,
    fallback_normal: np.ndarray = np.asarray((0.0, 0.0, 1.0)),
    outlier_threshold_m: float = 0.015,
    max_hypotheses: int = 128,
    sample_source: str = "contact_proxy",
) -> PlaneFitResult:
    """Fit a plane robustly, or retain a continuous normal for insufficient data.

    For four or more samples every non-degenerate triple is scored by inlier
    count, then median and maximum residual.  The best consensus is refit.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    valid = np.all(np.isfinite(points), axis=1)
    samples = points[valid]
    continuous_normal = _upward(previous_normal if previous_normal is not None else fallback_normal)
    fallback_point = np.mean(samples, axis=0) if len(samples) else np.zeros(3, dtype=np.float64)

    if len(samples) < 3:
        return PlaneFitResult(
            point=fallback_point,
            normal=continuous_normal,
            residuals_m=np.full(len(samples), np.nan),
            inlier_mask=np.zeros(len(samples), dtype=bool),
            valid_sample_count=int(len(samples)),
            fit_sample_count=0,
            used_fallback=True,
            fallback_reason="insufficient_samples",
            sample_source=sample_source,
            hypothesis_count=0,
        )

    candidates: list[tuple[tuple[float, ...], np.ndarray, np.ndarray, np.ndarray]] = []
    triples = _candidate_triples(len(samples), max_hypotheses)
    for indices in triples:
        subset = samples[list(indices)]
        try:
            point, normal = _svd_plane(subset)
        except ValueError:
            continue
        residuals = np.abs((samples - point) @ normal)
        inliers = residuals <= outlier_threshold_m
        score = (-float(np.count_nonzero(inliers)), float(np.median(residuals)), float(np.max(residuals)), *normal)
        candidates.append((score, point, normal, inliers))

    if not candidates:
        return PlaneFitResult(
            point=fallback_point,
            normal=continuous_normal,
            residuals_m=np.full(len(samples), np.nan),
            inlier_mask=np.zeros(len(samples), dtype=bool),
            valid_sample_count=int(len(samples)),
            fit_sample_count=0,
            used_fallback=True,
            fallback_reason="degenerate_samples",
            sample_source=sample_source,
            hypothesis_count=len(triples),
        )

    _, point, normal, inliers = min(candidates, key=lambda candidate: candidate[0])
    if np.count_nonzero(inliers) >= 3:
        try:
            refit = _svd_plane(samples[inliers])
        except ValueError:
            refit = None
        if refit is not None:
            point, normal = refit
    residuals = np.abs((samples - point) @ normal)
    inliers = residuals <= outlier_threshold_m
    return PlaneFitResult(
        point=point,
        normal=normal,
        residuals_m=residuals,
        inlier_mask=inliers,
        valid_sample_count=int(len(samples)),
        fit_sample_count=int(np.count_nonzero(inliers)),
        used_fallback=False,
        fallback_reason=None,
        sample_source=sample_source,
        hypothesis_count=len(triples),
    )


def estimate_support_plane(
    contact_projection_points_xyz: np.ndarray,
    terrain_sample_points_xyz: np.ndarray,
    *,
    previous_normal: np.ndarray | None = None,
    outlier_threshold_m: float = 0.015,
    max_hypotheses: int = 128,
) -> SupportPlaneEstimate:
    """Prefer contact proxies, then use RayCaster/stencil terrain samples."""
    contacts = np.asarray(contact_projection_points_xyz, dtype=np.float64)
    terrain = np.asarray(terrain_sample_points_xyz, dtype=np.float64)
    if contacts.ndim != 2 or contacts.shape[1] != 3:
        raise ValueError("contact_projection_points_xyz must have shape (N, 3)")
    if terrain.ndim != 2 or terrain.shape[1] != 3:
        raise ValueError("terrain_sample_points_xyz must have shape (N, 3)")
    contact_plane = fit_support_plane(
        contacts,
        previous_normal=previous_normal,
        outlier_threshold_m=outlier_threshold_m,
        max_hypotheses=max_hypotheses,
        sample_source="contact_proxy",
    )
    if contact_plane.quality_valid:
        return SupportPlaneEstimate(
            plane=contact_plane,
            sample_source="contact_proxy",
            used_sample_fallback=False,
            contact_sample_count=contact_plane.valid_sample_count,
            terrain_sample_count=int(np.count_nonzero(np.all(np.isfinite(terrain), axis=1))),
        )
    terrain_plane = fit_support_plane(
        terrain,
        previous_normal=previous_normal,
        outlier_threshold_m=outlier_threshold_m,
        max_hypotheses=max_hypotheses,
        sample_source="terrain_raycast_fallback",
    )
    return SupportPlaneEstimate(
        plane=terrain_plane,
        sample_source="terrain_raycast_fallback",
        used_sample_fallback=True,
        contact_sample_count=contact_plane.valid_sample_count,
        terrain_sample_count=terrain_plane.valid_sample_count,
    )


def project_to_tangent(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    unit_normal = _upward(normal)
    value = np.asarray(vector, dtype=np.float64)
    return value - np.sum(value * unit_normal, axis=-1, keepdims=True) * unit_normal


def normal_angle_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = _upward(first)
    b = _upward(second)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def body_tilt_deg(body_up: np.ndarray, support_normal: np.ndarray) -> float:
    body_axis = _unit(body_up)
    terrain_axis = _upward(support_normal)
    return float(np.degrees(np.arccos(np.clip(np.dot(body_axis, terrain_axis), -1.0, 1.0))))


def contact_gated_tangent_speeds(
    foot_velocities_w: np.ndarray,
    contact_mask: np.ndarray,
    support_normal: np.ndarray,
) -> np.ndarray:
    """Return tangent speeds for contacting feet and zero for swing feet."""
    velocities = np.asarray(foot_velocities_w, dtype=np.float64)
    contacts = np.asarray(contact_mask, dtype=bool)
    if velocities.ndim != 2 or velocities.shape[1] != 3 or contacts.shape != (velocities.shape[0],):
        raise ValueError("foot_velocities_w must be (N, 3) and contact_mask must be (N,)")
    tangent = project_to_tangent(velocities, support_normal)
    speeds = np.linalg.norm(tangent, axis=1)
    return np.where(contacts, speeds, 0.0)


def whole_body_com(link_masses: np.ndarray, link_com_positions_w: np.ndarray) -> np.ndarray:
    """Compute whole-body COM from runtime link masses and world COM positions."""
    masses = np.asarray(link_masses, dtype=np.float64)
    positions = np.asarray(link_com_positions_w, dtype=np.float64)
    if masses.ndim != 1 or positions.shape != (masses.size, 3):
        raise ValueError("link_masses must be (N,) and link_com_positions_w must be (N, 3)")
    if np.any(~np.isfinite(masses)) or np.any(masses < 0.0) or not np.all(np.isfinite(positions)):
        raise ValueError("masses and positions must be finite and masses non-negative")
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("total link mass must be positive")
    return np.sum(masses[:, None] * positions, axis=0) / total_mass


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _upward(normal)
    reference = np.asarray((0.0, 0.0, 1.0)) if abs(n[2]) < 0.9 else np.asarray((1.0, 0.0, 0.0))
    u = np.cross(reference, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted(set(map(tuple, points.tolist())))
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64)

    def cross(origin, a, b) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _segment_distance(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    segment = second - first
    length_squared = float(np.dot(segment, segment))
    if length_squared <= _EPS:
        return float(np.linalg.norm(point - first))
    fraction = np.clip(np.dot(point - first, segment) / length_squared, 0.0, 1.0)
    return float(np.linalg.norm(point - (first + fraction * segment)))


def classify_support_region(
    contacts_xyz: np.ndarray,
    query_xyz: np.ndarray,
    support_normal: np.ndarray,
    *,
    edge_tolerance_m: float = 0.005,
    segment_radius_m: float = 0.02,
) -> SupportRegionResult:
    """Classify projected COM against contact polygon; two contacts are diagnostic only."""
    contacts = np.asarray(contacts_xyz, dtype=np.float64)
    query = np.asarray(query_xyz, dtype=np.float64)
    if contacts.ndim != 2 or contacts.shape[1] != 3 or query.shape != (3,):
        raise ValueError("contacts_xyz must be (N, 3) and query_xyz must be (3,)")
    if not np.all(np.isfinite(query)):
        raise ValueError("query_xyz must be finite")
    contacts = contacts[np.all(np.isfinite(contacts), axis=1)]
    u, v = _tangent_basis(support_normal)
    projected = np.column_stack((contacts @ u, contacts @ v))
    query_2d = np.asarray((query @ u, query @ v))

    if len(contacts) == 2:
        distance = _segment_distance(query_2d, projected[0], projected[1])
        return SupportRegionResult("segment", segment_radius_m - distance, False, 2)
    if len(contacts) < 3:
        return SupportRegionResult("insufficient", float("-inf"), False, int(len(contacts)))

    hull = _convex_hull(projected)
    if len(hull) < 3:
        return SupportRegionResult("degenerate", float("-inf"), False, int(len(contacts)))
    signed_distances = []
    for first, second in zip(hull, np.roll(hull, -1, axis=0)):
        edge = second - first
        offset = query_2d - first
        cross_z = edge[0] * offset[1] - edge[1] * offset[0]
        signed_distances.append(cross_z / np.linalg.norm(edge))
    margin = float(np.min(signed_distances))
    if margin < -edge_tolerance_m:
        classification = "outside"
    elif margin <= edge_tolerance_m:
        classification = "edge"
    else:
        classification = "inside"
    return SupportRegionResult(classification, margin, classification == "outside", int(len(contacts)))
