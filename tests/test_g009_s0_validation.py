from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_g009_s0.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_g009_s0", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_covers_complete_s0_matrix_and_all_gates_pass() -> None:
    validator = _load_validator()
    report = validator.build_report()

    assert report["status"] == "pass"
    assert report["aggregate"]["cell_count"] == 24
    assert report["aggregate"]["pass_count"] == 24
    assert all(report["aggregate"]["gates"].values())
    assert report["protocol"] == {
        "slopes_deg": [0, 5, 10, 15, 20, 25],
        "terrain_azimuths_deg": [0, 90, 180, 270],
        "seed": 20260828,
        "residual_amplitude_m": 0.0,
        "nominal_friction": {"static": 0.8, "dynamic": 0.6},
        "angle_tolerance_deg": 0.1,
        "normal_tolerance_deg": 0.001,
        "expected_cell_count": 24,
    }
    assert {
        (cell["requested_slope_deg"], cell["terrain_azimuth_deg"])
        for cell in report["cells"]
    } == {
        (float(slope), float(azimuth))
        for slope in range(0, 26, 5)
        for azimuth in (0, 90, 180, 270)
    }


def test_each_cell_records_geometry_material_friction_and_repeat_hashes() -> None:
    validator = _load_validator()
    report = validator.build_report()

    for cell in report["cells"]:
        assert abs(cell["measured_slope_deg"] - cell["requested_slope_deg"]) <= 0.1
        assert cell["analytic_normal"][2] > 0.0
        assert cell["triangle_normal_min_z"] > 0.0
        assert cell["triangle_normal_max_error_deg"] <= 0.001
        axes = cell["axes_xy"]
        assert axes["downhill"] == [-value for value in axes["uphill"]]
        assert axes["contour_right"] == [-value for value in axes["contour_left"]]
        assert len(cell["mesh_sha256"]) == 64
        assert len(cell["material_sha256"]) == 64
        assert cell["mesh_sha256"] == cell["same_seed_repeat_mesh_sha256"]
        assert cell["material_sha256"] == cell["same_seed_repeat_material_sha256"]
        assert cell["material_unique_ids"] == [0]
        friction = cell["friction_readback"]
        assert friction["static"] == 0.8
        assert friction["dynamic"] == 0.6
        assert friction["tan_theta_over_mu_s"] >= 0.0
        assert cell["status"] == "pass"


def test_scope_and_source_bindings_prevent_runtime_overclaim() -> None:
    validator = _load_validator()
    report = validator.build_report()

    assert report["scope"]["uses_isaac_runtime"] is False
    assert report["scope"]["claims_isaac_usd_runtime_readback"] is False
    assert report["scope"]["claims_policy_success"] is False
    assert "does not claim Isaac USD runtime readback or policy success" in report["scope"]["limitation"]
    assert set(report["source_bindings"]) == {
        "terrain_sha256",
        "support_plane_sha256",
        "validator_sha256",
    }
    assert all(len(value) == 64 for value in report["source_bindings"].values())


def test_cli_writes_atomic_json_and_stdout(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "g009_s0.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    stdout_report = json.loads(completed.stdout)
    assert stdout_report["status"] == "pass"
    assert json.loads(output.read_text(encoding="utf-8")) == stdout_report
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_main_returns_nonzero_for_failed_gate(tmp_path: Path, monkeypatch, capsys) -> None:
    validator = _load_validator()
    failure = {
        "schema_version": 1,
        "status": "fail",
        "errors": ["synthetic_gate_failure"],
    }
    monkeypatch.setattr(validator, "build_report", lambda: failure)
    output = tmp_path / "failed.json"

    assert validator.main(["--output", str(output)]) == 1
    assert json.loads(capsys.readouterr().out) == failure
    assert json.loads(output.read_text(encoding="utf-8")) == failure
