#!/usr/bin/env python3
"""Build G009-5 R0 public media and a contract-valid evidence sidecar."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from isaac_walk_g009.media_contract import (  # noqa: E402
    MAX_PUBLIC_MEDIA_BYTES,
    canonical_json_sha256,
    file_sha256,
    validate_sidecar,
)
from evaluate_g009_r0 import OFFICIAL_PROTOCOL, git_source_state, validate_source_bundle  # noqa: E402


GOAL_ID = "g009"
STAGE_NUMBER = "G009-5"
STAGE_ID = "R0"
POSE_NAMES = ("prone", "supine", "left_side", "right_side")
CONFIG_PATH = "configs/g009_r0.json"
DEFAULT_QUANTITATIVE_PATH = "reports/runs/g009_r0_flat_quantitative_evaluation.json"
SUMMARY_PATH = "reports/runs/g009_r0_flat_visual_summary.json"
SIDECAR_PATH = "reports/runs/g009_r0_flat_visual_evidence.json"
PUBLIC_GIF_PATH = "docs/media/g009/R0/g009_5_r0_four_pose_recovery.gif"
PUBLIC_PNG_PATH = "docs/media/g009/R0/g009_5_r0_four_pose_recovery_contact_sheet.png"
LOCAL_MP4_PATH = "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\R0\\g009_5_r0_four_pose_recovery.mp4"
VISUAL_REPORT_ID = "g009_r0_flat_visual_summary"
GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _portable_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(f"path must be inside repository: {path}") from exc


def _resolve_portable(path_value: str) -> Path:
    if path_value.startswith("%USERPROFILE%\\"):
        return Path.home() / path_value.removeprefix("%USERPROFILE%\\")
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout


def _ffprobe(path: Path, executable: str) -> dict[str, Any]:
    return json.loads(
        _run(
            [
                executable,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def _stream_summary(probe: Mapping[str, Any]) -> dict[str, Any]:
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    duration = probe.get("format", {}).get("duration") or video.get("duration")
    return {
        "codec": video["codec_name"],
        "width": int(video["width"]),
        "height": int(video["height"]),
        "duration_s": None if duration is None else float(duration),
        "frame_rate": video.get("avg_frame_rate"),
    }


def validate_capture_reports(
    capture_paths: Sequence[Path], quantitative_path: Path, config_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path]]:
    _require(len(capture_paths) == len(POSE_NAMES), "exactly four pose capture reports are required")
    captures = [_read_json(path) for path in capture_paths]
    sources: list[Path] = []
    expected_recorder = {"path": "scripts/record_g009_r0.py", "sha256": file_sha256(REPO_ROOT / "scripts" / "record_g009_r0.py")}
    expected_evaluator = {"path": "scripts/evaluate_g009_r0.py", "sha256": file_sha256(REPO_ROOT / "scripts" / "evaluate_g009_r0.py")}
    for index, (expected_pose, capture) in enumerate(zip(POSE_NAMES, captures, strict=True), start=1):
        prefix = f"capture[{index - 1}]"
        _require(capture.get("goal_id") == GOAL_ID, f"{prefix}: goal_id mismatch")
        _require(capture.get("stage_number") == STAGE_NUMBER, f"{prefix}: stage_number mismatch")
        _require(capture.get("stage_id") == STAGE_ID, f"{prefix}: stage_id mismatch")
        _require(capture.get("status") == "complete", f"{prefix}: status must be complete")
        _require(capture.get("headless") is True and capture.get("offscreen") is True, f"{prefix}: headless off-screen capture required")
        pose = capture.get("pose")
        _require(isinstance(pose, Mapping), f"{prefix}: pose is required")
        _require(pose.get("pose_id") == expected_pose, f"{prefix}: expected {expected_pose}")
        _require(pose.get("index") == index, f"{prefix}: stage pose numbering mismatch")
        _require(capture.get("metrics", {}).get("stable_success") is True, f"{prefix}: stable_success is required")
        _require(capture.get("metrics", {}).get("termination_reason") == "stable_success", f"{prefix}: termination must be stable_success")
        video = capture.get("local_video")
        _require(isinstance(video, Mapping), f"{prefix}: local_video is required")
        video_path = video.get("path")
        expected_parent = PureWindowsPath("%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\R0")
        _require(
            isinstance(video_path, str)
            and PureWindowsPath(video_path).parent == expected_parent
            and PureWindowsPath(video_path).suffix.lower() == ".mp4",
            f"{prefix}: invalid local-only R0 MP4 path",
        )
        _require(video.get("git_policy") == "local_only", f"{prefix}: MP4 must be local_only")
        expected_name = f"g009_5_r0_{index:02d}_{expected_pose}_s42.mp4"
        _require(PureWindowsPath(video_path).name == expected_name, f"{prefix}: exact numbered filename mismatch")
        source = _resolve_portable(video_path).resolve()
        _require(source.is_file(), f"{prefix}: MP4 not found: {source}")
        _require(file_sha256(source) == video.get("sha256"), f"{prefix}: MP4 hash mismatch")
        _require(source.stat().st_size == video.get("bytes"), f"{prefix}: MP4 byte count mismatch")
        bindings = capture.get("source_bindings", {})
        _require(bindings.get("record_source") == expected_recorder, f"{prefix}: recorder source hash mismatch")
        _require(bindings.get("evaluator") == expected_evaluator, f"{prefix}: evaluator source hash mismatch")
        source_state = capture.get("source_state", {})
        _require(source_state.get("before") == source_state.get("after"), f"{prefix}: source state changed during capture")
        _require(source_state.get("after", {}).get("clean") is True, f"{prefix}: capture source state is dirty")
        physics = capture.get("physics_readback", {})
        _require(physics.get("effective_friction_valid") is True, f"{prefix}: effective friction provenance invalid")
        _require(physics.get("friction_combine_mode") == "multiply", f"{prefix}: friction combine mode mismatch")
        sources.append(source)

    for field in ("seed", "source_commit"):
        _require(len({capture[field] for capture in captures}) == 1, f"capture {field} values differ")
    checkpoint_values = {
        json.dumps(capture["checkpoint"], sort_keys=True) for capture in captures
    }
    _require(len(checkpoint_values) == 1, "capture checkpoint bindings differ")
    _require(len({str(path) for path in sources}) == 4, "capture MP4 paths must be distinct")
    _require(len({capture["local_video"]["sha256"] for capture in captures}) == 4, "capture MP4 hashes must be distinct")
    config_sha = file_sha256(config_path)
    for capture in captures:
        _require(
            capture.get("source_bindings", {}).get("config")
            == {"path": CONFIG_PATH, "sha256": config_sha},
            "capture config binding mismatch",
        )

    quantitative = _read_json(quantitative_path)
    _require(quantitative.get("goal_id") == GOAL_ID, "quantitative goal_id mismatch")
    _require(quantitative.get("stage_number") == STAGE_NUMBER, "quantitative stage_number mismatch")
    _require(quantitative.get("stage_id") == STAGE_ID, "quantitative stage_id mismatch")
    _require(quantitative.get("status") == "pass", "quantitative report must pass before media build")
    _require(quantitative.get("protocol_mode") == "official_qualification", "diagnostic evaluation cannot build public media")
    _require(quantitative.get("official_protocol") == OFFICIAL_PROTOCOL, "quantitative official protocol mismatch")
    _require(quantitative.get("source_bindings", {}).get("evaluator") == expected_evaluator, "quantitative evaluator source hash mismatch")
    _require(
        quantitative.get("source_bindings", {}).get("config")
        == {"path": CONFIG_PATH, "sha256": config_sha},
        "quantitative config binding mismatch",
    )
    _require(quantitative.get("seed") == captures[0]["seed"], "quantitative/capture seed mismatch")
    _require(quantitative.get("checkpoint") == captures[0]["checkpoint"], "quantitative/capture checkpoint mismatch")
    pose_order = tuple(item.get("pose_id") for item in quantitative.get("poses", []))
    _require(pose_order == POSE_NAMES, "quantitative pose order mismatch")
    _require(
        quantitative.get("aggregate", {}).get("all_pose_gate_pass") is True,
        "quantitative all-pose gate must pass",
    )
    training_binding = quantitative.get("training_binding")
    _require(isinstance(training_binding, Mapping), "quantitative training binding is required")
    _require(all(capture.get("training_binding") == training_binding for capture in captures), "training bindings differ")
    training_report_path = _resolve_portable(training_binding.get("path", "")).resolve()
    _require(training_report_path.is_file(), "bound training report is missing")
    _require(file_sha256(training_report_path) == training_binding.get("sha256"), "training report hash mismatch")
    training_report = _read_json(training_report_path)
    _require(training_report.get("artifacts", {}).get("checkpoint_sha256") == captures[0]["checkpoint"]["sha256"], "training report checkpoint mismatch")
    source_bundle = validate_source_bundle(training_binding.get("source_bundle", {}))
    source_commit = training_binding.get("repository", {}).get("commit")
    _require(quantitative.get("source_state", {}).get("before") == quantitative.get("source_state", {}).get("after"), "evaluation source state changed")
    _require(quantitative.get("source_state", {}).get("after", {}).get("commit") == source_commit, "evaluation/training commit mismatch")
    _require(all(capture.get("source_commit") == source_commit for capture in captures), "capture/training commit mismatch")
    _require(all(capture.get("quantitative_report", {}).get("sha256") == file_sha256(quantitative_path) for capture in captures), "capture/quantitative report hash mismatch")
    current = git_source_state()
    _require(current.get("clean") is True, "media build requires source-clean repository outside reports/runs")
    _require(current.get("commit") == source_commit, "media build commit mismatch")
    _require(source_bundle == training_binding.get("source_bundle"), "media build source bundle mismatch")
    checkpoint_path = _resolve_portable(captures[0]["checkpoint"]["path"]).resolve()
    _require(checkpoint_path.is_file(), "bound checkpoint is missing")
    _require(file_sha256(checkpoint_path) == captures[0]["checkpoint"]["sha256"], "bound checkpoint hash mismatch")
    quantitative_friction = quantitative.get("physics_readback", {})
    terrain = quantitative_friction.get("terrain", {})
    foot = quantitative_friction.get("foot_material_readback", {})
    effective = quantitative_friction.get("effective_foot_friction", {})
    _require(terrain.get("combine_mode") == "multiply", "quantitative friction combine mode mismatch")
    _require(effective.get("valid_for_all_envs") is True, "quantitative effective friction invalid")
    for index, capture in enumerate(captures):
        physics = capture["physics_readback"]
        _require(physics["terrain_static_friction"] == terrain["static_friction"], f"capture[{index}]: terrain static friction mismatch")
        _require(physics["terrain_dynamic_friction"] == terrain["dynamic_friction"], f"capture[{index}]: terrain dynamic friction mismatch")
        _require(physics["foot_material_static_friction_range"] == [foot["static_friction_min"], foot["static_friction_max"]], f"capture[{index}]: foot static friction mismatch")
        _require(physics["foot_material_dynamic_friction_range"] == [foot["dynamic_friction_min"], foot["dynamic_friction_max"]], f"capture[{index}]: foot dynamic friction mismatch")
        _require(physics["effective_foot_static_friction_range"] == [effective["static_friction_min"], effective["static_friction_max"]], f"capture[{index}]: effective static friction mismatch")
        _require(physics["effective_foot_dynamic_friction_range"] == [effective["dynamic_friction_min"], effective["dynamic_friction_max"]], f"capture[{index}]: effective dynamic friction mismatch")
    return captures, quantitative, sources


def _font_expression(font_path: Path) -> str:
    escaped_path = font_path.resolve().as_posix().replace(":", r"\:")
    return f"fontfile='{escaped_path}'"


def _build_composite(
    sources: Sequence[Path], captures: Sequence[Mapping[str, Any]], destination: Path, ffmpeg: str, font: Path
) -> None:
    font_expr = _font_expression(font)
    filters: list[str] = []
    labels: list[str] = []
    for index, capture in enumerate(captures):
        pose = capture["pose"]["pose_id"].replace("_", " ").upper()
        metric = capture["metrics"]
        outcome = "SUCCESS" if metric["stable_success"] else metric["termination_reason"].upper()
        filters.append(
            f"[{index}:v]scale=960:540:force_original_aspect_ratio=decrease,"
            "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
            "drawbox=x=0:y=0:w=iw:h=62:color=black@0.70:t=fill,"
            f"drawtext={font_expr}:text='G009-5 R0 | {index + 1}/4 {pose} | {outcome}':"
            "x=(w-text_w)/2:y=16:fontsize=26:fontcolor=white,setpts=PTS-STARTPTS"
            f"[v{index}]"
        )
        labels.append(f"[v{index}]")
    filters.append("".join(labels) + f"concat=n={len(sources)}:v=1:a=0[out]")
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    for source in sources:
        command.extend(["-i", str(source)])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    _run(command)


def _build_gif(source: Path, destination: Path, ffmpeg: str) -> None:
    graph = (
        "fps=6,scale=720:-2:flags=lanczos,split[base][palette_source];"
        "[palette_source]palettegen=max_colors=80:stats_mode=diff[palette];"
        "[base][palette]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    _run([ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(source), "-filter_complex", graph, "-loop", "0", str(destination)])


def _build_contact_sheet(
    sources: Sequence[Path], destination: Path, ffmpeg: str, ffprobe: str, font: Path
) -> list[float]:
    timestamps = []
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    for source in sources:
        duration = _stream_summary(_ffprobe(source, ffprobe))["duration_s"] or 0.02
        timestamp = max(0.0, float(duration) * 0.65)
        timestamps.append(timestamp)
        command.extend(["-ss", f"{timestamp:.3f}", "-i", str(source)])
    font_expr = _font_expression(font)
    filters = []
    labels = []
    for index, pose in enumerate(POSE_NAMES):
        filters.append(
            f"[{index}:v]scale=640:360:force_original_aspect_ratio=decrease,"
            "pad=640:360:(ow-iw)/2:(oh-ih)/2:black,"
            "drawbox=x=0:y=0:w=iw:h=42:color=black@0.70:t=fill,"
            f"drawtext={font_expr}:text='{index + 1}/4 {pose.upper()}':"
            f"x=(w-text_w)/2:y=9:fontsize=21:fontcolor=white[p{index}]"
        )
        labels.append(f"[p{index}]")
    filters.append("".join(labels) + "xstack=inputs=4:layout=0_0|640_0|0_360|640_360[out]")
    command.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-frames:v", "1", str(destination)])
    _run(command)
    return timestamps


def _validate_media(path: Path, kind: str) -> None:
    _require(path.is_file() and path.stat().st_size > 0, f"missing or empty {kind}: {path}")
    signature = path.read_bytes()[:8]
    if kind == "gif":
        _require(signature.startswith(GIF_SIGNATURES), "invalid GIF signature")
    if kind == "png":
        _require(signature == PNG_SIGNATURE, "invalid PNG signature")
    if kind in {"gif", "png"}:
        _require(path.stat().st_size < MAX_PUBLIC_MEDIA_BYTES, f"{kind} must be below 10 MiB")


def _artifact(
    path: Path, portable: str, evidence_type: str, git_policy: str, ffprobe: str
) -> dict[str, Any]:
    return {
        "evidence_type": evidence_type,
        "path": portable,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "git_policy": git_policy,
        **_stream_summary(_ffprobe(path, ffprobe)),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _publish_transaction(pairs: Sequence[tuple[Path, Path]], validator: Any) -> None:
    _require(len({final.resolve() for _, final in pairs}) == len(pairs), "transaction destinations must be distinct")
    for staged, _ in pairs:
        _require(staged.is_file(), f"staged transaction input missing: {staged}")
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for staged, final in pairs:
            final.parent.mkdir(parents=True, exist_ok=True)
            backup = final.with_name(final.name + f".{uuid.uuid4().hex}.bak")
            if final.exists():
                os.replace(final, backup)
                backups.append((final, backup))
            os.replace(staged, final)
            installed.append(final)
        validator()
    except Exception:
        for final in reversed(installed):
            final.unlink(missing_ok=True)
        for final, backup in backups:
            if backup.exists():
                os.replace(backup, final)
        raise
    else:
        for _, backup in backups:
            backup.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    quantitative_path = args.quantitative_report.resolve()
    capture_paths = [path.resolve() for path in args.capture_reports]
    for path in (*capture_paths, config_path, quantitative_path, args.font.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    captures, quantitative, sources = validate_capture_reports(capture_paths, quantitative_path, config_path)
    final_paths = {
        "local_mp4": _resolve_portable(LOCAL_MP4_PATH),
        "public_gif": REPO_ROOT / PUBLIC_GIF_PATH,
        "public_png": REPO_ROOT / PUBLIC_PNG_PATH,
        "summary": REPO_ROOT / SUMMARY_PATH,
        "sidecar": REPO_ROOT / SIDECAR_PATH,
    }
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not args.rebuild_existing:
        raise FileExistsError("refusing to overwrite existing outputs: " + ", ".join(map(str, existing)))
    final_paths["local_mp4"].parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="g009_r0_media_", dir=final_paths["local_mp4"].parent) as temp_name:
        temp_root = Path(temp_name)
        staged_mp4 = temp_root / "g009_5_r0.mp4"
        staged_gif = temp_root / "g009_5_r0.gif"
        staged_png = temp_root / "g009_5_r0.png"
        staged_summary = temp_root / "visual_summary.json"
        staged_sidecar = temp_root / "visual_evidence.json"
        _build_composite(sources, captures, staged_mp4, args.ffmpeg, args.font)
        _build_gif(staged_mp4, staged_gif, args.ffmpeg)
        frame_times = _build_contact_sheet(sources, staged_png, args.ffmpeg, args.ffprobe, args.font)
        for path, kind in ((staged_mp4, "mp4"), (staged_gif, "gif"), (staged_png, "png")):
            _validate_media(path, kind)

        physics_readback = {
            "poses": [
                {"pose_id": capture["pose"]["pose_id"], "readback": capture["physics_readback"]}
                for capture in captures
            ]
        }
        physics_sha = canonical_json_sha256(physics_readback)
        checkpoint = captures[0]["checkpoint"]
        source_commit = captures[0]["source_commit"]
        source_bundle = captures[0]["training_binding"]["source_bundle"]
        seed = captures[0]["seed"]
        config_sha = file_sha256(config_path)
        quantitative_path_portable = _portable_repo_path(quantitative_path)
        quantitative_sha = file_sha256(quantitative_path)
        media_binding = {
            "goal_id": GOAL_ID,
            "stage_id": STAGE_ID,
            "report_id": VISUAL_REPORT_ID,
            "source_commit": source_commit,
            "seed": seed,
            "checkpoint_sha256": checkpoint["sha256"],
            "config_sha256": config_sha,
            "physics_readback_sha256": physics_sha,
            "quantitative_report_sha256": quantitative_sha,
            "source_bundle_sha256": source_bundle["sha256"],
        }
        summary = {
            "schema_version": 1,
            "goal_id": GOAL_ID,
            "stage_number": STAGE_NUMBER,
            "stage_id": STAGE_ID,
            "report_id": VISUAL_REPORT_ID,
            "status": "complete",
            "scope": {
                "claim": "flat-ground four-pose R0 recovery playback tied to a passing quantitative report",
                "not_claimed": "slope, asymmetric-friction, disturbance, or real-robot recovery",
                "decision_source": quantitative_path_portable,
            },
            "media_binding": media_binding,
            "pose_order": list(POSE_NAMES),
            "capture_reports": [
                {
                    "path": _portable_repo_path(path),
                    "sha256": file_sha256(path),
                    "pose_id": capture["pose"]["pose_id"],
                    "local_video_sha256": capture["local_video"]["sha256"],
                }
                for path, capture in zip(capture_paths, captures, strict=True)
            ],
            "quantitative_gate": quantitative["aggregate"],
            "physics_readback": physics_readback,
            "composition": {
                "layout": "four sequential numbered pose clips; 2x2 numbered contact sheet",
                "contact_sheet_frame_times_s": frame_times,
                "ffmpeg_version": _run([args.ffmpeg, "-version"]).splitlines()[0],
            },
            "sources": {
                "builder": {"path": "scripts/build_g009_r0_media.py", "sha256": file_sha256(Path(__file__))},
                "recorder": {"path": "scripts/record_g009_r0.py", "sha256": file_sha256(REPO_ROOT / "scripts" / "record_g009_r0.py")},
                "evaluator": {"path": "scripts/evaluate_g009_r0.py", "sha256": file_sha256(REPO_ROOT / "scripts" / "evaluate_g009_r0.py")},
            },
        }
        _write_json(staged_summary, summary)
        summary_sha = file_sha256(staged_summary)
        artifacts = [
            _artifact(staged_mp4, LOCAL_MP4_PATH, "local_mp4", "local_only", args.ffprobe),
            _artifact(staged_gif, PUBLIC_GIF_PATH, "public_gif", "git_public", args.ffprobe),
            _artifact(staged_png, PUBLIC_PNG_PATH, "public_png", "git_public", args.ffprobe),
            {
                "evidence_type": "quantitative_report",
                "path": quantitative_path_portable,
                "sha256": quantitative_sha,
                "bytes": quantitative_path.stat().st_size,
                "git_policy": "git_public",
            },
            {
                "evidence_type": "visual_summary",
                "path": SUMMARY_PATH,
                "sha256": summary_sha,
                "bytes": staged_summary.stat().st_size,
                "git_policy": "git_public",
            },
        ]
        sidecar = {
            "schema_version": 1,
            "goal_id": GOAL_ID,
            "stage_number": STAGE_NUMBER,
            "stage_id": STAGE_ID,
            "status": "complete",
            "bindings": {
                "source_commit": source_commit,
                "seed": seed,
                "report_id": quantitative["report_id"],
                "checkpoint": checkpoint,
                "config": {"path": CONFIG_PATH, "sha256": config_sha},
                "quantitative_report": {"path": quantitative_path_portable, "sha256": quantitative_sha},
                "visual_summary": {"path": SUMMARY_PATH, "sha256": summary_sha},
                "physics_readback_sha256": physics_sha,
                "source_bundle": source_bundle,
            },
            "physics_readback": physics_readback,
            "scope": summary["scope"],
            "artifacts": artifacts,
            "capture_report_sha256": [file_sha256(path) for path in capture_paths],
            "builder_source_sha256": file_sha256(Path(__file__)),
        }
        errors = validate_sidecar(sidecar, REPO_ROOT, check_files=False)
        _require(not errors, "staged sidecar metadata failed validation: " + "; ".join(errors))
        _write_json(staged_sidecar, sidecar)
        sidecar_sha = file_sha256(staged_sidecar)

        def validate_published() -> None:
            published_errors = validate_sidecar(sidecar, REPO_ROOT, check_files=True)
            _require(not published_errors, "published sidecar validation failed: " + "; ".join(published_errors))
            _require(file_sha256(final_paths["sidecar"]) == sidecar_sha, "sidecar publish hash mismatch")

        _publish_transaction(
            (
                (staged_mp4, final_paths["local_mp4"]),
                (staged_gif, final_paths["public_gif"]),
                (staged_png, final_paths["public_png"]),
                (staged_summary, final_paths["summary"]),
                (staged_sidecar, final_paths["sidecar"]),
            ),
            validate_published,
        )
        return sidecar


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-reports", required=True, nargs=4, type=Path, metavar=("PRONE", "SUPINE", "LEFT_SIDE", "RIGHT_SIDE"))
    parser.add_argument("--quantitative-report", type=Path, default=REPO_ROOT / DEFAULT_QUANTITATIVE_PATH)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / CONFIG_PATH)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/arial.ttf"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--rebuild-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sidecar = build(args)
    except (FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "sidecar": SIDECAR_PATH, "artifacts": len(sidecar["artifacts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
