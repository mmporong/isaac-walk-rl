#!/usr/bin/env python3
"""Validate the dedicated G009-5-E011 raw-contact media bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_g009_r0_rev18_raw_contact_media as media


DEFAULT_SIDECAR = media.DEFAULT_SIDECAR


def _record_path(record: Any, *, parent: Path, suffix: str, label: str) -> Path:
    media.require(isinstance(record, dict), f"{label} record must be an object")
    record = cast(dict[str, Any], record)
    portable = record.get("path")
    media.require(isinstance(portable, str) and bool(portable) and "\\" not in portable and ":" not in portable, f"{label} must use a portable repository path")
    portable = cast(str, portable)
    relative = Path(portable)
    media.require(not relative.is_absolute() and ".." not in relative.parts, f"{label} path escaped the repository")
    resolved = media._resolve_direct_child(media.REPO_ROOT / relative, parent, label=label, suffix=suffix, must_exist=True)
    media.require(media.repo_path(resolved) == portable, f"{label} path is not canonical")
    return resolved


def _local_video_path(record: Any) -> Path:
    media.require(isinstance(record, dict), "local video record must be an object")
    record = cast(dict[str, Any], record)
    portable = record.get("path")
    prefix = "%USERPROFILE%\\"
    media.require(isinstance(portable, str) and portable.startswith(prefix), "local video must use a portable %USERPROFILE% path")
    portable = cast(str, portable)
    relative = portable[len(prefix):].replace("\\", "/")
    media.require(".." not in Path(relative).parts, "local video path escaped USERPROFILE")
    resolved = media._resolve_direct_child(Path.home() / relative, media.LOCAL_VIDEO_DIR, label="local video", suffix=".mp4", must_exist=True)
    media.require(media.portable_local_path(resolved) == portable, "local video path is not canonical")
    return resolved


def _validate_artifact(record: Any, path: Path, *, label: str, local: bool) -> None:
    media.require(isinstance(record, dict), f"{label} record must be an object")
    record = cast(dict[str, Any], record)
    media.require(media._valid_sha256(record.get("sha256")) and record.get("sha256") == media.file_sha256(path), f"{label} hash mismatch")
    media.require(record.get("bytes") == path.stat().st_size, f"{label} byte count mismatch")
    media.require(
        record.get("tracked_in_git") is (not local)
        and record.get("git_policy") == ("local_only" if local else "git_public"),
        f"{label} Git policy mismatch",
    )


def validate_bundle(sidecar_path: Path = DEFAULT_SIDECAR) -> dict[str, Any]:
    resolved_sidecar = media._resolve_direct_child(sidecar_path, media.RUNS_DIR, label="E011 sidecar", suffix=".json", must_exist=True)
    sidecar_raw = resolved_sidecar.read_bytes()
    sidecar = json.loads(sidecar_raw.decode("utf-8"))
    media.require(
        isinstance(sidecar, dict)
        and sidecar.get("schema_version") == "g009.r0.rev18.raw_contact_visual_evidence.v1"
        and sidecar.get("goal_id") == "g009"
        and sidecar.get("stage_id") == "R0"
        and sidecar.get("stage_number") == media.STAGE_NUMBER
        and sidecar.get("revision") == "rev18"
        and sidecar.get("evidence_id") == media.EVIDENCE_ID
        and sidecar.get("status") == "diagnostic_complete"
        and sidecar.get("diagnostic_only") is True,
        "E011 diagnostic sidecar identity mismatch",
    )
    media.require(sidecar.get("integrity") == {"passed": True, "hash_bound": True}, "E011 sidecar integrity mismatch")
    expected_decision = {"outcome": "unavailable_on_gpu", "selected_lever": None}
    media.require(sidecar.get("decision") == expected_decision, "E011 sidecar decision mismatch")
    contract = sidecar.get("contract")
    media.require(isinstance(contract, dict) and contract.get("kind") == "g009_r0_diagnostic_extension", "diagnostic contract mismatch")
    contract = cast(dict[str, Any], contract)
    builder = contract.get("builder_source")
    builder_path = _record_path(builder, parent=media.REPO_ROOT / "scripts", suffix=".py", label="media builder")
    _validate_artifact(builder, builder_path, label="media builder", local=False)
    media.require(builder_path == media.BUILDER_SOURCE, "media builder source binding mismatch")
    dedicated = contract.get("dedicated_validator")
    media.require(isinstance(dedicated, dict), "dedicated validator record missing")
    dedicated = cast(dict[str, Any], dedicated)
    validator_path = _record_path(dedicated, parent=media.REPO_ROOT / "scripts", suffix=".py", label="dedicated validator")
    _validate_artifact(dedicated, validator_path, label="dedicated validator", local=False)
    media.require(
        validator_path == Path(__file__).resolve()
        and dedicated.get("command") == "%PYTHON% scripts/validate_g009_r0_rev18_raw_contact_media.py --check-only",
        "dedicated validator binding mismatch",
    )
    standard = contract.get("standard_stage_validator")
    media.require(
        isinstance(standard, dict)
        and standard.get("path") == "scripts/validate_g009_media_contract.py"
        and standard.get("compatible") is False
        and "diagnostic-only" in standard.get("reason", ""),
        "standard stage validator exception is not explicit",
    )
    provenance = sidecar.get("provenance")
    media.require(isinstance(provenance, dict), "sidecar provenance missing")
    provenance = cast(dict[str, Any], provenance)
    source_binding = provenance.get("source_binding")
    media.require(isinstance(source_binding, dict), "source binding missing")
    source_binding = cast(dict[str, Any], source_binding)
    synthesis_record = source_binding.get("synthesis")
    media.require(isinstance(synthesis_record, dict), "synthesis record missing")
    synthesis_record = cast(dict[str, Any], synthesis_record)
    synthesis_path = _record_path(synthesis_record, parent=media.RUNS_DIR, suffix=".json", label="E011 synthesis")
    synthesis_raw = synthesis_path.read_bytes()
    media.require(
        synthesis_record.get("sha256") == hashlib.sha256(synthesis_raw).hexdigest(),
        "E011 synthesis hash mismatch",
    )
    synthesis = media.read_summary(synthesis_path, raw=synthesis_raw)
    synthesis.pop("_validated_reports")
    expected_source_binding = {
        "synthesis": synthesis_record,
        "reports": synthesis["input_reports"],
        "raw_probe_source_bundle_sha256": synthesis["integrity"]["raw_probe_source_bundle_sha256"],
        "synthesis_source_bundle_sha256": synthesis["integrity"]["synthesis_source_bundle_sha256"],
        "predecessor": synthesis["integrity"]["predecessor"],
    }
    media.require(source_binding == expected_source_binding, "source/input hash lineage mismatch")
    summary_record = provenance.get("visual_summary")
    media.require(isinstance(summary_record, dict), "visual summary record missing")
    summary_record = cast(dict[str, Any], summary_record)
    summary_path = _record_path(summary_record, parent=media.RUNS_DIR, suffix=".json", label="visual summary")
    summary_raw = summary_path.read_bytes()
    _validate_artifact(summary_record, summary_path, label="visual summary", local=False)
    summary = json.loads(summary_raw.decode("utf-8"))
    media.require(
        isinstance(summary, dict)
        and summary.get("schema_version") == "g009.r0.rev18.raw_contact_visual_summary.v1"
        and summary.get("goal_id") == "g009"
        and summary.get("stage_id") == "R0"
        and summary.get("stage_number") == media.STAGE_NUMBER
        and summary.get("revision") == "rev18"
        and summary.get("evidence_id") == media.EVIDENCE_ID
        and summary.get("status") == "diagnostic_complete"
        and summary.get("diagnostic_only") is True
        and summary.get("camera_footage") is False
        and summary.get("robot_locomotion_footage") is False
        and summary.get("training_footage") is False
        and summary.get("telemetry_animation") is True
        and summary.get("learned_policy_qualified") is False
        and summary.get("physics_ground_truth_authority") is False,
        "visual summary identity/governance mismatch",
    )
    expected_result = {
        "cpu": {"raw_callback_availability": "2/2 pass"},
        "gpu": {"raw_callback_availability": "0/2 unavailable"},
        "instrumentation_bundle": "partial/unavailable",
        "physics_ground_truth_authority": False,
    }
    media.require(
        summary.get("source_binding") == source_binding
        and summary.get("decision") == expected_decision
        and summary.get("result") == expected_result
        and sidecar.get("result") == expected_result
        and summary.get("labels") == list(media.LABELS)
        and sidecar.get("labels") == list(media.LABELS)
        and summary.get("governance") == {"ppo": {"status": "not_run", "updates": 0}, "qualification": {"status": "not_run"}, "gate01": {"status": "forbidden"}},
        "visual summary labels/result/governance mismatch",
    )
    public = provenance.get("public_artifacts")
    media.require(isinstance(public, dict), "public artifact records missing")
    public = cast(dict[str, Any], public)
    png_record, gif_record = public.get("png"), public.get("gif")
    png_path = _record_path(png_record, parent=media.PUBLIC_MEDIA_DIR, suffix=".png", label="public PNG")
    gif_path = _record_path(gif_record, parent=media.PUBLIC_MEDIA_DIR, suffix=".gif", label="public GIF")
    _validate_artifact(png_record, png_path, label="public PNG", local=False)
    _validate_artifact(gif_record, gif_path, label="public GIF", local=False)
    media.validate_media(png_path, "png")
    media.validate_media(gif_path, "gif")
    video_record = provenance.get("local_video")
    media.require(isinstance(video_record, dict), "local video record missing")
    video_record = cast(dict[str, Any], video_record)
    video_path = _local_video_path(video_record)
    _validate_artifact(video_record, video_path, label="local video", local=True)
    media.validate_media(video_path, "mp4")
    media.require(summary.get("public_artifacts") == public and summary.get("local_video") == video_record, "visual summary artifact binding mismatch")
    media.require(media.DECISION_BANNER == "OUTCOME: UNAVAILABLE ON GPU · NO LEVER SELECTED", "required outcome banner changed")
    return {
        "schema_version": "g009.r0.rev18.raw_contact_media_validation.v1",
        "goal_id": "g009",
        "stage_id": "R0",
        "stage_number": media.STAGE_NUMBER,
        "evidence_id": media.EVIDENCE_ID,
        "mode": "check_only",
        "status": "pass",
        "diagnostic_only": True,
        "decision": expected_decision,
        "sidecar": {"path": media.repo_path(resolved_sidecar), "sha256": hashlib.sha256(sidecar_raw).hexdigest()},
        "checked": ["four_input_hashes", "source_bundle_hashes", "builder_source_binding", "dedicated_validator_binding", "diagnostic_governance", "public_png", "public_gif", "local_mp4", "visual_summary"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--check-only", action="store_true", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = validate_bundle(args.sidecar)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        result = {"schema_version": "g009.r0.rev18.raw_contact_media_validation.v1", "goal_id": "g009", "stage_id": "R0", "evidence_id": media.EVIDENCE_ID, "mode": "check_only", "status": "fail", "errors": [str(exc)]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
