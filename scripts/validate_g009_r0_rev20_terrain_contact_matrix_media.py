#!/usr/bin/env python3
"""Fail-closed validator for G009-5-E013 rev20 terrain-contact media."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, cast

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import build_g009_r0_rev20_terrain_contact_matrix_media as media


def _artifact(record: Any, path: Path, *, local: bool, metadata: Mapping[str, Any] | None = None) -> None:
    media.require(isinstance(record, dict), "artifact record must be an object")
    value = cast(dict[str, Any], record)
    expected_path = str(path.resolve()) if local else media.repo_path(path)
    media.require(value.get("path") == expected_path, "artifact canonical path mismatch")
    media.require(value.get("sha256") == media.file_sha256(path) and value.get("bytes") == path.stat().st_size, "artifact hash/bytes mismatch")
    media.require(
        value.get("git_policy") == ("local_only" if local else "git_public_after_review")
        and value.get("intended_for_git") is (not local)
        and isinstance(value.get("tracked_in_git_at_build"), bool),
        "artifact Git policy mismatch",
    )
    if local:
        media.require(value.get("tracked_in_git_at_build") is False, "local artifact cannot be Git-tracked at build")
    if metadata is not None:
        media.require(all(value.get(key) == expected for key, expected in metadata.items()), "artifact media metadata mismatch")


def validate_bundle(phase: str, sidecar_path: Path | None = None) -> dict[str, Any]:
    paths = media.phase_paths(phase); sidecar_path = sidecar_path or paths["sidecar"]
    sidecar_path = media.direct_child(sidecar_path, media.RUNS_DIR, ".json", "sidecar", exists=True)
    sidecar, _ = media.read_json(sidecar_path)
    expected_sequence = "13.01" if phase == "cpu-preflight" else "13.02"
    expected_sidecar_keys = {"schema_version", "goal_id", "stage_id", "stage_number", "sequence_number", "revision", "evidence_id", "phase", "status", "integrity", "labels", "claim_limits", "input_bindings", "source", "artifacts", "contract", "governance"}
    media.require(
        set(sidecar) == expected_sidecar_keys
        and sidecar.get("schema_version") == "g009.r0.rev20.terrain_contact_matrix_visual_evidence.v1"
        and sidecar.get("goal_id") == "g009" and sidecar.get("stage_id") == "R0" and sidecar.get("stage_number") == media.STAGE_NUMBER and sidecar.get("revision") == "rev20"
        and sidecar.get("evidence_id") == media.EVIDENCE_ID and sidecar.get("phase") == phase
        and sidecar.get("sequence_number") == expected_sequence and sidecar.get("status") == "diagnostic_complete",
        "sidecar identity mismatch",
    )
    expected_labels = list(media.labels_for_phase(phase))
    media.require(sidecar.get("labels") == expected_labels, "required labels mismatch")
    media.require(sidecar.get("claim_limits") == media.CLAIM_LIMITS, "claim limits mismatch")
    media.require(sidecar.get("integrity") == {"passed": True, "hash_bound": True, "all_inputs_revalidated": True, "no_overwrite": True}, "integrity/no-overwrite mismatch")
    bindings = sidecar.get("input_bindings")
    media.require(isinstance(bindings, list), "input bindings missing")
    expected_inputs = (*media.CPU_REPORTS, media.CPU_PREFLIGHT) if phase == "cpu-preflight" else (*media.FINAL_REPORTS, media.CPU_PREFLIGHT, media.FINAL_SYNTHESIS)
    canonical = media.validate_inputs(phase, expected_inputs)
    media.require(bindings == canonical["input_bindings"], "input bindings mismatch")
    for item in cast(list[Mapping[str, str]], bindings):
        path = media.REPO_ROOT / item["path"]
        media.require(item["sha256"] == media.file_sha256(path), "input hash mismatch")
    artifacts = cast(dict[str, Any], sidecar.get("artifacts", {})); public = cast(dict[str, Any], artifacts.get("public", {}))
    media.require(set(artifacts) == {"visual_summary", "public", "local_video"} and set(public) == {"png", "gif"}, "artifact set mismatch")
    summary_path, png_path, gif_path, video_path = paths["summary"], paths["png"], paths["gif"], paths["video"]
    _artifact(artifacts.get("visual_summary"), summary_path, local=False)
    _artifact(public.get("png"), png_path, local=False, metadata={"width": media.WIDTH, "height": media.HEIGHT, "representative_frame": media.FRAME_COUNT})
    _artifact(public.get("gif"), gif_path, local=False, metadata={"width": media.WIDTH, "height": media.HEIGHT, "frame_count": media.FRAME_COUNT, "duration_ms": media.FRAME_COUNT * media.FRAME_DURATION_MS})
    _artifact(artifacts.get("local_video"), video_path, local=True, metadata={"codec": "h264", "width": media.WIDTH, "height": media.HEIGHT, "fps": "30/1", "frames": 168, "duration_seconds": media.VIDEO_DURATION_SECONDS})
    media.require(
        all(record.get("tracked_in_git_at_build") is False for record in (cast(dict[str, Any], artifacts["visual_summary"]), cast(dict[str, Any], public["png"]), cast(dict[str, Any], public["gif"]), cast(dict[str, Any], artifacts["local_video"]))),
        "generated artifact tracking-at-build must be false",
    )
    for path, kind in ((png_path, "png"), (gif_path, "gif"), (video_path, "mp4")): media.validate_media(path, kind)
    from PIL import Image  # pyright: ignore[reportMissingImports]
    with Image.open(png_path) as image:
        media.require(image.format == "PNG" and image.size == (media.WIDTH, media.HEIGHT), "PNG metadata mismatch")
    with Image.open(gif_path) as image:
        media.require(image.format == "GIF" and image.size == (media.WIDTH, media.HEIGHT) and getattr(image, "n_frames", 1) == media.FRAME_COUNT, "GIF metadata mismatch")
        durations = []
        for index in range(media.FRAME_COUNT): image.seek(index); durations.append(image.info.get("duration"))
        media.require(durations == [media.FRAME_DURATION_MS] * media.FRAME_COUNT, "GIF duration mismatch")
    metadata = media.ffprobe_metadata(video_path)
    media.require(metadata == {"codec": "h264", "width": media.WIDTH, "height": media.HEIGHT, "fps": "30/1", "frames": 168, "duration_seconds": media.VIDEO_DURATION_SECONDS}, "MP4 codec/frame/dimension/duration mismatch")
    summary, _ = media.read_json(summary_path)
    expected_summary_keys = {"schema_version", "goal_id", "stage_id", "stage_number", "sequence_number", "revision", "evidence_id", "phase", "status", "labels", "claim_limits", "input_bindings", "git_commit", "source_bundle_sha256", "preflight_synthesis_source_bundle_sha256", "telemetry", "decision", "public_artifacts", "local_video", "governance"}
    media.require(
        set(summary) == expected_summary_keys
        and summary.get("schema_version") == "g009.r0.rev20.terrain_contact_matrix_visual_summary.v1"
        and summary.get("goal_id") == "g009" and summary.get("stage_id") == "R0" and summary.get("stage_number") == media.STAGE_NUMBER and summary.get("sequence_number") == expected_sequence
        and summary.get("revision") == "rev20" and summary.get("evidence_id") == media.EVIDENCE_ID and summary.get("phase") == phase and summary.get("status") == "diagnostic_complete",
        "summary identity/schema mismatch",
    )
    media.require(summary.get("labels") == expected_labels and summary.get("claim_limits") == media.CLAIM_LIMITS, "summary labels/claim limits mismatch")
    media.require(summary.get("input_bindings") == bindings and summary.get("telemetry") == {"reports": canonical["reports"], "repeatability": canonical["repeatability"]}, "summary telemetry/input binding mismatch")
    media.require(summary.get("decision") == {"outcome": canonical["decision"], "gpu_stage_authorized": canonical["decision"] == "gpu_stage_authorized"}, "summary decision mismatch")
    expected_source = {"git_commit": canonical["git_commit"], "probe_source_bundle_sha256": canonical["source_bundle_sha256"], "synthesis_source_bundle_sha256": canonical["preflight_synthesis_source_bundle_sha256"]}
    media.require(sidecar.get("source") == expected_source, "sidecar current source bundle mismatch")
    media.require(summary.get("git_commit") == canonical["git_commit"] and summary.get("source_bundle_sha256") == canonical["source_bundle_sha256"] and summary.get("preflight_synthesis_source_bundle_sha256") == canonical["preflight_synthesis_source_bundle_sha256"], "summary source bundle mismatch")
    media.require(summary.get("public_artifacts") == public and summary.get("local_video") == artifacts.get("local_video"), "summary artifact metadata mismatch")
    governance = {"diagnostic_only": True, "learned": False, "reward_computed": False, "ppo_updates": 0, "qualification_status": "not_run", "physics_ground_truth_authority": False}
    media.require(summary.get("governance") == governance and sidecar.get("governance") == governance, "governance mismatch")
    contract = cast(dict[str, Any], sidecar.get("contract", {}))
    media.require(set(contract) == {"builder", "validator"}, "tool contract set mismatch")
    _artifact(contract.get("builder"), media.BUILDER_SOURCE, local=False)
    validator_record = cast(dict[str, Any], contract.get("validator", {})); _artifact(validator_record, Path(__file__), local=False)
    media.require(validator_record.get("command") == f"%PYTHON% scripts/validate_g009_r0_rev20_terrain_contact_matrix_media.py --phase {phase} --check-only", "tool source binding mismatch")
    return {"schema_version": "g009.r0.rev20.terrain_contact_matrix_media_validation.v1", "status": "pass", "phase": phase, "sequence_number": expected_sequence, "checked": ["magic", "codec", "frames", "dimensions", "duration", "artifact_hashes", "labels", "input_bindings", "source_bundle", "claim_limits", "no_overwrite"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--phase", required=True, choices=("cpu-preflight", "final")); parser.add_argument("--sidecar", type=Path); parser.add_argument("--check-only", action="store_true", required=True); return parser


def main() -> int:
    args = build_parser().parse_args()
    try: result = validate_bundle(args.phase, args.sidecar)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, KeyError, IndexError) as exc: result = {"schema_version": "g009.r0.rev20.terrain_contact_matrix_media_validation.v1", "status": "fail", "phase": args.phase, "errors": [str(exc)]}
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
