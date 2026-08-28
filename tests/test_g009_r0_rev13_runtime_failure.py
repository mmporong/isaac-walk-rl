from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = ROOT / "scripts/summarize_g009_r0_rev13_cpu_failure.py"
MEDIA_SCRIPT = ROOT / "scripts/build_g009_r0_rev13_runtime_media.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYNTHESIS = load_module("g009_rev13_cpu_failure", SUMMARY_SCRIPT)
MEDIA = load_module("g009_rev13_cpu_failure_media", MEDIA_SCRIPT)


def test_rev13_cpu_failure_is_three_run_rejected_evidence() -> None:
    summary = SYNTHESIS.summarize(SYNTHESIS.DEFAULT_REV13_INPUTS, SYNTHESIS.DEFAULT_BASELINE)
    assert summary["status"] == "rejected"
    assert summary["learned_policy_qualified"] is False
    assert summary["qualification_status"] == "not_run"
    assert summary["lineage"]["actual_articulation_solver_iterations"] == {"position": 8, "velocity": 1}
    assert summary["repeatability"]["validated_runs"] == 3
    assert len({item["execution_id"] for item in summary["repeatability"]["inputs"]}) == 3
    assert all(item["failed_checks"] == ["nonfoot_peak_force_bounded"] for item in summary["repeatability"]["inputs"])
    assert summary["failure"] == {
        "failed_check": "nonfoot_peak_force_bounded",
        "threshold_bodyweights": 15.0,
        "right_side_reset_pose_hold_peak_bodyweights": 15.97161865234375,
        "peak_body": "base",
        "peak_time_s": 0.645,
        "peak_physics_step": 129,
        "numeric_invalid_terminations": 0,
        "hard_joint_limit_terminations": 0,
    }
    assert all(summary["blocked_stages"].values())


def test_rev12_comparison_preserves_cautious_measured_changes() -> None:
    summary = SYNTHESIS.summarize(SYNTHESIS.DEFAULT_REV13_INPUTS, SYNTHESIS.DEFAULT_BASELINE)
    comparison = summary["rev12_comparison"]
    assert comparison["right_side_reset_pose_hold_peak_bodyweights"] == 9.332860946655273
    assert comparison["absolute_increase_bodyweights"] == 6.638757705688477
    assert math.isclose(comparison["relative_increase_percent"], 71.1331470985613, abs_tol=1e-12)
    details = comparison["right_side_reset_pose_hold"]
    assert math.isclose(details["max_root_angular_speed_rad_s"]["relative_change_percent"], 46.661287892538226, abs_tol=1e-12)
    assert math.isclose(details["max_joint_speed_rad_s"]["relative_change_percent"], -32.208317914505905, abs_tol=1e-12)
    assert math.isclose(details["excess_contact_delta_v_m_s"]["relative_change_percent"], -6.887119912359485, abs_tol=1e-12)
    assert math.isclose(details["peak_step_excess_contact_delta_v_m_s"]["relative_change_percent"], -5.958618563082352, abs_tol=1e-12)
    assert "does not prove" in comparison["careful_interpretation"]


def test_synthesis_rejects_duplicate_execution_or_second_failed_check(tmp_path: Path) -> None:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in SYNTHESIS.DEFAULT_REV13_INPUTS]
    reports[2]["execution"]["execution_id"] = reports[0]["execution"]["execution_id"]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(reports[2]), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct execution IDs"):
        SYNTHESIS.summarize((*SYNTHESIS.DEFAULT_REV13_INPUTS[:2], duplicate), SYNTHESIS.DEFAULT_BASELINE)

    reports[2]["execution"]["execution_id"] = "1234567812344234a234123456789abc"
    reports[2]["checks"]["no_numeric_invalid_termination"] = False
    second_failure = tmp_path / "second_failure.json"
    second_failure.write_text(json.dumps(reports[2]), encoding="utf-8")
    with pytest.raises(ValueError, match="only nonfoot_peak_force_bounded"):
        SYNTHESIS.summarize((*SYNTHESIS.DEFAULT_REV13_INPUTS[:2], second_failure), SYNTHESIS.DEFAULT_BASELINE)


def test_public_media_is_small_and_explicitly_telemetry_only() -> None:
    synthesis = MEDIA.read_synthesis(MEDIA.DEFAULT_INPUT)
    assert synthesis["learned_policy_qualified"] is False
    summary = json.loads(MEDIA.DEFAULT_SUMMARY.read_text(encoding="utf-8"))
    assert summary["telemetry_animation"] is True
    assert summary["camera_footage"] is False
    assert summary["ppo_training_run"] is False
    assert summary["status"] == "rejected"
    assert set(summary["labels"]) == {
        "PUBLIC DIAGNOSTIC",
        "TELEMETRY ANIMATION",
        "NOT CAMERA FOOTAGE",
        "NO PPO",
        "REJECTED",
    }
    for path, signature in ((MEDIA.DEFAULT_PNG, b"\x89PNG\r\n\x1a\n"), (MEDIA.DEFAULT_GIF, b"GIF8")):
        assert path.read_bytes().startswith(signature)
        assert path.stat().st_size < 10 * 1024 * 1024
    local = summary["local_video"]
    assert local["path"].startswith("%USERPROFILE%")
    assert "LIMMM" not in local["path"]
    assert local["tracked_in_git"] is False
    if MEDIA.DEFAULT_LOCAL_VIDEO.exists():
        assert local["sha256"] == MEDIA.file_sha256(MEDIA.DEFAULT_LOCAL_VIDEO)


def test_media_source_contains_prominent_non_policy_labels() -> None:
    source = MEDIA_SCRIPT.read_text(encoding="utf-8")
    assert "PUBLIC DIAGNOSTIC · TELEMETRY ANIMATION · NOT CAMERA FOOTAGE" in source
    assert "G009 R0 REV13 · NO PPO · REJECTED" in source
    assert "causality is not established" in source
