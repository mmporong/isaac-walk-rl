from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_g009_r0_gate10_attribution_media.py"


def _load():
    spec = importlib.util.spec_from_file_location("g009_gate10_attribution_media", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MEDIA = _load()


def test_bound_fullstate_report_has_three_nonqualified_events() -> None:
    report = MEDIA.read_report(MEDIA.DEFAULT_INPUT)
    assert len(report["events"]) == 3
    assert report["gate10_safety_passed"] is False
    assert report["learned_policy_qualified"] is False
    assert all(len(event["preceding_control_step_ring"]) == 16 for event in report["events"])


def test_event_series_contains_target_torque_and_contact_history() -> None:
    report = MEDIA.read_report(MEDIA.DEFAULT_INPUT)
    series = MEDIA.event_series(report["events"][0], report["contract"]["hard_joint_limit_margin_rad"])
    assert len(series["steps"]) == 16
    assert len(series["position"]) == len(series["target"]) == 16
    assert len(series["velocity"]) == len(series["torque"]) == 16
    assert len(series["action"]) == len(series["contact_bw"]) == 16
    assert series["termination_boundary"] < series["hard_lower"]
    assert series["margin_excess"] > 0.0


def test_report_validation_rejects_qualification_or_missing_ring_field(tmp_path: Path) -> None:
    report = json.loads(MEDIA.DEFAULT_INPUT.read_text(encoding="utf-8"))
    report["learned_policy_qualified"] = True
    qualified = tmp_path / "qualified.json"
    qualified.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot qualify"):
        MEDIA.read_report(qualified)

    report["learned_policy_qualified"] = False
    del report["events"][0]["preceding_control_step_ring"][0]["processed_ema_target_rad"]
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="processed_ema_target_rad"):
        MEDIA.read_report(missing)


def test_public_overlay_is_explicitly_nonqualified() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "SAFETY FAIL · NOT QUALIFIED" in source
    assert '"public_claim_eligible": False' in source
    assert '"learned_policy_qualified": False' in source


def test_local_video_lineage_is_private_and_hash_bound() -> None:
    summary = json.loads(MEDIA.DEFAULT_SUMMARY.read_text(encoding="utf-8"))
    local_video = summary["local_video"]
    assert local_video["path"] == MEDIA.portable_local_path(MEDIA.DEFAULT_LOCAL_VIDEO)
    assert local_video["path"].startswith("%USERPROFILE%")
    assert "LIMMM" not in local_video["path"]
    assert local_video["sha256"] == MEDIA.file_sha256(MEDIA.DEFAULT_LOCAL_VIDEO)
    assert local_video["tracked_in_git"] is False
    assert local_video["codec"] == "h264"
