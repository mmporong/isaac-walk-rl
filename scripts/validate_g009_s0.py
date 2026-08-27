from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from isaac_walk_g009.support_plane import normal_angle_error_deg  # noqa: E402
from isaac_walk_g009.terrain import (  # noqa: E402
    SlopeTerrainSpec,
    build_mesh_arrays,
    friction_limit_ratio,
    generate_slope_terrain,
    mesh_arrays_sha256,
    terrain_axes,
    triangle_normals,
)


SLOPE_DEGREES = (0, 5, 10, 15, 20, 25)
TERRAIN_AZIMUTH_DEGREES = (0, 90, 180, 270)
SEED = 20260828
STATIC_FRICTION = 0.8
DYNAMIC_FRICTION = 0.6
ANGLE_TOLERANCE_DEG = 0.1
NORMAL_TOLERANCE_DEG = 0.001
EXPECTED_CELL_COUNT = len(SLOPE_DEGREES) * len(TERRAIN_AZIMUTH_DEGREES)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _build_cell(slope_deg: int, azimuth_deg: int) -> dict[str, Any]:
    spec = SlopeTerrainSpec(
        slope_deg=float(slope_deg),
        azimuth_deg=float(azimuth_deg),
        seed=SEED,
        residual_amplitude_m=0.0,
        static_friction=(STATIC_FRICTION,),
        dynamic_friction=(DYNAMIC_FRICTION,),
    )
    field = generate_slope_terrain(spec)
    mesh = build_mesh_arrays(field)
    repeat_field = generate_slope_terrain(spec)
    repeat_mesh = build_mesh_arrays(repeat_field)
    normals = triangle_normals(mesh)
    analytic_normal = field.analytic_normal
    normal_errors = np.asarray(
        [normal_angle_error_deg(analytic_normal, normal) for normal in normals],
        dtype=np.float64,
    )
    measured_slope_deg = float(
        np.degrees(np.arccos(np.clip(float(np.mean(normals[:, 2])), -1.0, 1.0)))
    )
    axes = terrain_axes(float(azimuth_deg))
    mesh_sha256 = mesh_arrays_sha256(mesh)
    repeat_mesh_sha256 = mesh_arrays_sha256(repeat_mesh)
    material_sha256 = _array_sha256(mesh["face_materials"])
    repeat_material_sha256 = _array_sha256(repeat_mesh["face_materials"])
    unique_material_ids = [int(value) for value in np.unique(mesh["face_materials"])]
    static_friction_readback = float(field.spec.static_friction[0])
    dynamic_friction_readback = float(field.spec.dynamic_friction[0])
    friction_ratio, limit_stress = friction_limit_ratio(
        float(slope_deg), static_friction_readback
    )

    gates = {
        "angle_error_within_tolerance": abs(measured_slope_deg - slope_deg) <= ANGLE_TOLERANCE_DEG,
        "normals_upward": bool(np.all(normals[:, 2] > 0.0) and analytic_normal[2] > 0.0),
        "analytic_triangle_normal_match": float(np.max(normal_errors)) <= NORMAL_TOLERANCE_DEG,
        "material_zero_only": unique_material_ids == [0],
        "friction_order_valid": static_friction_readback >= dynamic_friction_readback >= 0.0,
        "same_seed_repeat_hash": (
            mesh_sha256 == repeat_mesh_sha256 and material_sha256 == repeat_material_sha256
        ),
    }
    return {
        "cell_id": f"slope-{slope_deg:02d}_azimuth-{azimuth_deg:03d}",
        "seed": SEED,
        "requested_slope_deg": float(slope_deg),
        "measured_slope_deg": measured_slope_deg,
        "slope_angle_error_deg": abs(measured_slope_deg - slope_deg),
        "terrain_azimuth_deg": float(azimuth_deg),
        "analytic_normal": _vector(analytic_normal),
        "triangle_normal_max_error_deg": float(np.max(normal_errors)),
        "triangle_normal_min_z": float(np.min(normals[:, 2])),
        "axes_xy": {
            "uphill": _vector(axes.uphill_xy),
            "downhill": _vector(axes.downhill_xy),
            "contour_left": _vector(axes.contour_left_xy),
            "contour_right": _vector(axes.contour_right_xy),
        },
        "mesh_sha256": mesh_sha256,
        "material_sha256": material_sha256,
        "same_seed_repeat_mesh_sha256": repeat_mesh_sha256,
        "same_seed_repeat_material_sha256": repeat_material_sha256,
        "material_unique_ids": unique_material_ids,
        "friction_readback": {
            "static": static_friction_readback,
            "dynamic": dynamic_friction_readback,
            "tan_theta_over_mu_s": friction_ratio,
            "physical_limit_stress": limit_stress,
        },
        "gates": gates,
        "status": "pass" if all(gates.values()) else "fail",
    }


def build_report() -> dict[str, Any]:
    cells = [
        _build_cell(slope_deg, azimuth_deg)
        for slope_deg in SLOPE_DEGREES
        for azimuth_deg in TERRAIN_AZIMUTH_DEGREES
    ]
    actual_keys = {
        (int(cell["requested_slope_deg"]), int(cell["terrain_azimuth_deg"]))
        for cell in cells
    }
    expected_keys = {
        (slope_deg, azimuth_deg)
        for slope_deg in SLOPE_DEGREES
        for azimuth_deg in TERRAIN_AZIMUTH_DEGREES
    }
    aggregate_gates = {
        "cell_matrix_complete": len(cells) == EXPECTED_CELL_COUNT and actual_keys == expected_keys,
        "all_angle_errors_within_0_1_deg": all(
            cell["gates"]["angle_error_within_tolerance"] for cell in cells
        ),
        "all_normals_upward": all(cell["gates"]["normals_upward"] for cell in cells),
        "all_analytic_triangle_normals_match": all(
            cell["gates"]["analytic_triangle_normal_match"] for cell in cells
        ),
        "all_materials_zero_only": all(cell["gates"]["material_zero_only"] for cell in cells),
        "all_friction_order_valid": all(cell["gates"]["friction_order_valid"] for cell in cells),
        "all_same_seed_repeat_hashes_match": all(
            cell["gates"]["same_seed_repeat_hash"] for cell in cells
        ),
    }
    errors = [name for name, passed in aggregate_gates.items() if not passed]
    return {
        "schema_version": 1,
        "goal_id": "g009",
        "stage_id": "S0",
        "validator_id": "g009_s0_import_light_analytic_gate",
        "status": "pass" if not errors else "fail",
        "scope": {
            "claim": "E01 import-light analytic geometry, material, and friction gate",
            "uses_isaac_runtime": False,
            "claims_isaac_usd_runtime_readback": False,
            "claims_policy_success": False,
            "limitation": (
                "This report does not claim Isaac USD runtime readback or policy success; "
                "those require separate Isaac Sim runtime evidence."
            ),
        },
        "protocol": {
            "slopes_deg": list(SLOPE_DEGREES),
            "terrain_azimuths_deg": list(TERRAIN_AZIMUTH_DEGREES),
            "seed": SEED,
            "residual_amplitude_m": 0.0,
            "nominal_friction": {"static": STATIC_FRICTION, "dynamic": DYNAMIC_FRICTION},
            "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
            "normal_tolerance_deg": NORMAL_TOLERANCE_DEG,
            "expected_cell_count": EXPECTED_CELL_COUNT,
        },
        "source_bindings": {
            "terrain_sha256": _file_sha256(REPO_ROOT / "src" / "isaac_walk_g009" / "terrain.py"),
            "support_plane_sha256": _file_sha256(
                REPO_ROOT / "src" / "isaac_walk_g009" / "support_plane.py"
            ),
            "validator_sha256": _file_sha256(Path(__file__)),
        },
        "aggregate": {
            "cell_count": len(cells),
            "pass_count": sum(cell["status"] == "pass" for cell in cells),
            "max_slope_angle_error_deg": max(cell["slope_angle_error_deg"] for cell in cells),
            "max_analytic_triangle_normal_error_deg": max(
                cell["triangle_normal_max_error_deg"] for cell in cells
            ),
            "gates": aggregate_gates,
        },
        "cells": cells,
        "errors": errors,
    }


def _write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the import-light G009 S0 analytic gate.")
    parser.add_argument("--output", required=True, type=Path, help="Atomic JSON report path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report()
    _write_json_atomic(args.output.resolve(), report)
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
