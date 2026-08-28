#!/usr/bin/env python3
"""Build numbered public GIF/PNG evidence from rev15 GPU camera footage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_STEM = "g009_5_r0_diag_rev15_06_gpu_right_side_force_fail"
DEFAULT_CAPTURE = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_capture_s42.json"
DEFAULT_GIF = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.gif"
DEFAULT_PNG = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.png"
DEFAULT_VISUAL = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_visual_evidence.json"
EXPECTED_LOCAL = PureWindowsPath(
    f"%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\R0\\diagnostic\\{OUTPUT_STEM}_s42.mp4"
)
REQUIRED_LABELS = (
    "DIAGNOSTIC",
    "REJECTED",
    "NO PPO",
    "RIGHT_SIDE",
    "RESET_POSE_HOLD",
    "GPU FORCE FAIL",
)
OVERLAY_TOP = "G009-5 | REV15 | DIAGNOSTIC | REJECTED | NO PPO"
OVERLAY_BOTTOM = "06 GPU RIGHT_SIDE | RESET_POSE_HOLD | FORCE 16.788 BW > 15 BW"
MAX_PUBLIC_BYTES = 10 * 1024 * 1024
EXPECTED_RUNTIME_REPORT_SHA256 = (
    "e24674a1ed33c38fbe5f12d19dc068167b9787e75323efbe55629bf059839b91"
)
EXPECTED_RUNTIME_COMMIT = "bc999d504e226011ff3d83e68a416b9049b406cb"
EXPECTED_EXECUTION_ID = "fc715b20c45242b19b86445f733fa02b"
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "218671a84f2748f7b94a426490057318b0896e2160454f6928c4277dee7435df"
)
EXPECTED_CONTRACT_SHA256 = (
    "5f29ba19458404b5009d3734294c57e79294efecc7fe03bf8c71c71656129832"
)
EXPECTED_BODY_NAMES = (
    "base",
    "FL_hip",
    "FR_hip",
    "Head_upper",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "Head_lower",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_new(staged: Path, final: Path) -> str:
    """Exclusively publish staged bytes without replacing an existing target."""

    created = False
    try:
        with staged.open("rb") as source, final.open("xb") as destination:
            created = True
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        digest = file_sha256(staged)
        require(file_sha256(final) == digest, "exclusive publish integrity mismatch")
        return digest
    except Exception:
        if created:
            final.unlink(missing_ok=True)
        raise


def cleanup_owned_outputs(published: list[tuple[Path, str]]) -> None:
    for path, expected_digest in published:
        if path.is_file() and file_sha256(path) == expected_digest:
            path.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "capture JSON root must be an object")
    return value


def resolve_portable(value: str) -> Path:
    prefix = "%USERPROFILE%\\"
    return (
        Path.home() / value.removeprefix(prefix)
        if value.startswith(prefix)
        else REPO_ROOT / value
    )


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")


def ffprobe_summary(
    path: Path, executable: str, *, require_timing: bool = True
) -> dict[str, Any]:
    value = json.loads(
        subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    video = next(
        stream for stream in value["streams"] if stream.get("codec_type") == "video"
    )
    result = {
        "codec": video.get("codec_name"),
        "width": int(video["width"]),
        "height": int(video["height"]),
    }
    if not require_timing:
        return result
    duration = value.get("format", {}).get("duration") or video.get("duration")
    frames = video.get("nb_read_frames") or video.get("nb_frames")
    require(
        duration is not None and frames is not None and int(frames) > 0,
        "timed media metadata is unavailable",
    )
    assert duration is not None and frames is not None
    return {
        **result,
        "frame_rate": video.get("avg_frame_rate"),
        "frames": int(frames),
        "duration_s": float(duration),
    }


def validate_live_capture_contract(capture: dict[str, Any]) -> None:
    require(
        capture.get("timing")
        == {
            "physics_dt_s": 0.005,
            "control_dt_s": 0.02,
            "decimation": 4,
            "rollout_steps": 150,
            "rollout_duration_s": 3.0,
        },
        "capture timing contract mismatch",
    )
    live = capture.get("live_physics_readback")
    require(isinstance(live, dict), "live physics readback is required")
    assert isinstance(live, dict)
    require(
        live.get("checks")
        == {
            "articulation_solver_iteration_counts_match_contract": True,
            "rigid_body_max_depenetration_velocity_matches_contract": True,
        },
        "live physics readback check mismatch",
    )
    solver = live.get("articulation_solver_iterations")
    require(isinstance(solver, dict), "live solver readback is required")
    assert isinstance(solver, dict)
    solver_rows = solver.get("articulations")
    require(
        isinstance(solver_rows, list)
        and len(solver_rows) == 8
        and all(
            isinstance(row, dict)
            and row.get("solver_position_iteration_count") == 16
            and row.get("solver_velocity_iteration_count") == 0
            for row in solver_rows
        ),
        "live solver iteration readback mismatch",
    )
    readback = live.get("readback")
    require(isinstance(readback, dict), "live readback payload is required")
    assert isinstance(readback, dict)
    require(
        readback.get("articulation_group_count") == 8
        and readback.get("rigid_body_count") == 152
        and readback.get("duplicate_link_prim_paths") == []
        and tuple(readback.get("authoritative_body_names", ())) == EXPECTED_BODY_NAMES,
        "live 8x19 topology mismatch",
    )
    articulations = readback.get("articulations")
    require(
        isinstance(articulations, list) and len(articulations) == 8,
        "eight live articulations are required",
    )
    assert isinstance(articulations, list)
    prim_paths: list[str] = []
    for articulation_index, articulation in enumerate(articulations):
        require(isinstance(articulation, dict), "invalid live articulation row")
        assert isinstance(articulation, dict)
        links = articulation.get("links")
        require(
            articulation.get("articulation_index") == articulation_index
            and tuple(articulation.get("authoritative_body_names", ()))
            == EXPECTED_BODY_NAMES
            and isinstance(links, list)
            and len(links) == 19,
            "live articulation topology mismatch",
        )
        assert isinstance(links, list)
        for body_index, link in enumerate(links):
            require(isinstance(link, dict), "invalid live link row")
            assert isinstance(link, dict)
            prim_path = link.get("prim_path")
            require(
                link.get("body_index") == body_index
                and link.get("body_name") == EXPECTED_BODY_NAMES[body_index]
                and isinstance(prim_path, str)
                and prim_path.endswith(f"/{EXPECTED_BODY_NAMES[body_index]}")
                and link.get("prim_valid") is True
                and link.get("usd_rigid_body_api") is True
                and link.get("physx_rigid_body_api") is True
                and link.get("max_depenetration_velocity_m_s") == 1.0
                and link.get("error") is None,
                "live link API/value mismatch",
            )
            assert isinstance(prim_path, str)
            prim_paths.append(prim_path)
    require(
        len(prim_paths) == len(set(prim_paths)) == 152,
        "live link prim paths must be unique",
    )


def validate_blocking_cell(blocking: dict[str, Any]) -> None:
    require(
        blocking.get("env_index") == 7
        and blocking.get("pose_id") == "right_side"
        and blocking.get("action_mode") == "reset_pose_hold"
        and blocking.get("min_contact_separation_m") is None,
        "GPU blocking cell identity/separation mismatch",
    )
    require(
        blocking.get("max_nonfoot_force_bodyweights") == 16.78827476501465
        and blocking.get("max_nonfoot_force_threshold_bodyweights") == 15.0
        and blocking.get("failed_check") == "nonfoot_peak_force_bounded"
        and blocking.get("max_nonfoot_force_body_name") == "base"
        and blocking.get("max_nonfoot_force_physics_step") == 129
        and blocking.get("max_nonfoot_force_time_s") == 0.645,
        "force evidence binding mismatch",
    )


def validate_capture(path: Path) -> tuple[dict[str, Any], Path]:
    require(
        path.resolve() == DEFAULT_CAPTURE.resolve(), "capture sidecar path is fixed"
    )
    capture = read_json(path)
    require(
        capture.get("camera_footage") is True
        and capture.get("telemetry_animation") is False,
        "actual camera footage is required",
    )
    require(
        capture.get("headless") is True and capture.get("offscreen") is True,
        "headless/offscreen binding mismatch",
    )
    require(
        capture.get("status") == "rejected"
        and capture.get("diagnostic_only") is True
        and capture.get("qualification_status") == "not_run",
        "diagnostic status mismatch",
    )
    require(
        capture.get("learned") is False
        and capture.get("ppo_training") is False
        and capture.get("ppo_checkpoint_used") is False
        and capture.get("qualification_passed") is None
        and capture.get("candidate_runtime_calibration_passed") is False,
        "NO PPO binding mismatch",
    )
    require(
        capture.get("completed_stages")
        == {
            "cpu_runtime_3x": True,
            "gpu_runtime_3x": True,
            "strict_rejection_synthesis": True,
        }
        and capture.get("blocked_stages")
        == {"gate01": True, "gate10": True, "ppo_training": True},
        "diagnostic stage accounting mismatch",
    )
    require(
        tuple(capture.get("labels", ())) == REQUIRED_LABELS,
        "required public labels mismatch",
    )
    require(
        capture.get("source_env_index") == 7
        and capture.get("pose_id") == "right_side"
        and capture.get("action_mode") == "reset_pose_hold"
        and capture.get("device") == "cuda:0",
        "failure cell mismatch",
    )
    validate_live_capture_contract(capture)
    binding = capture.get("original_rev15_report_binding", {})
    require(
        binding.get("sha256") == EXPECTED_RUNTIME_REPORT_SHA256
        and binding.get("execution_id") == EXPECTED_EXECUTION_ID,
        "runtime report binding mismatch",
    )
    source_binding = capture.get("source", {})
    require(isinstance(source_binding, dict), "capture source binding is required")
    assert isinstance(source_binding, dict)
    require(
        source_binding.get("original_runtime_binding")
        == {
            "commit": EXPECTED_RUNTIME_COMMIT,
            "bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
        },
        "runtime source binding mismatch",
    )
    current = source_binding.get("current_capture_binding")
    require(isinstance(current, dict), "current capture binding is required")
    assert isinstance(current, dict)
    current_bundle = current.get("source_bundle")
    require(isinstance(current_bundle, dict), "current source bundle is required")
    assert isinstance(current_bundle, dict)
    require(
        current.get("contract_sha256") == EXPECTED_CONTRACT_SHA256
        and current_bundle.get("source_bundle_sha256") == EXPECTED_SOURCE_BUNDLE_SHA256
        and current_bundle.get("clean") is True
        and current_bundle.get("all_files_present") is True,
        "current capture source binding mismatch",
    )
    blocking = binding.get("blocking_cell", {})
    require(isinstance(blocking, dict), "GPU blocking cell binding is required")
    assert isinstance(blocking, dict)
    validate_blocking_cell(blocking)
    video = capture.get("local_video", {})
    require(
        PureWindowsPath(video.get("path", "")) == EXPECTED_LOCAL,
        "local MP4 path mismatch",
    )
    source = resolve_portable(video["path"]).resolve()
    require(source.is_file(), "local-only camera MP4 is missing")
    require(
        file_sha256(source) == video.get("sha256")
        and source.stat().st_size == video.get("bytes"),
        "local MP4 integrity mismatch",
    )
    return capture, source


def artifact(
    path: Path,
    evidence_type: str,
    ffprobe: str,
    *,
    public_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "path": repo_path(public_path or path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "git_policy": "git_public",
        "evidence_type": evidence_type,
        **ffprobe_summary(
            path, ffprobe, require_timing=evidence_type != "camera_footage_still"
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    require(
        args.gif.resolve() == DEFAULT_GIF.resolve()
        and args.png.resolve() == DEFAULT_PNG.resolve()
        and args.visual.resolve() == DEFAULT_VISUAL.resolve(),
        "public artifact paths are fixed",
    )
    require(
        not any(path.exists() for path in (args.gif, args.png, args.visual)),
        "public evidence refuses overwrite",
    )
    capture, source = validate_capture(args.capture)
    source_probe = ffprobe_summary(source, args.ffprobe)
    require(
        source_probe["codec"] == "h264"
        and source_probe["width"] == 1280
        and source_probe["height"] == 720,
        "source must be H264 1280x720",
    )
    args.gif.parent.mkdir(parents=True, exist_ok=True)
    args.visual.parent.mkdir(parents=True, exist_ok=True)
    font = str(args.font).replace("\\", "/").replace(":", "\\:")
    overlay = (
        "drawbox=x=0:y=0:w=iw:h=58:color=yellow@0.92:t=fill,"
        f"drawtext=fontfile='{font}':text='{OVERLAY_TOP}':fontcolor=black:fontsize=26:x=(w-text_w)/2:y=14,"
        "drawbox=x=0:y=h-58:w=iw:h=58:color=black@0.72:t=fill,"
        f"drawtext=fontfile='{font}':text='{OVERLAY_BOTTOM}':fontcolor=white:fontsize=24:borderw=2:bordercolor=black:x=(w-text_w)/2:y=h-44"
    )
    with tempfile.TemporaryDirectory(prefix="g009-rev15-camera-") as directory:
        staging = Path(directory)
        staged_png = staging / "still.png"
        staged_gif = staging / "camera.gif"
        staged_visual = staging / "visual_evidence.json"
        subprocess.run(
            [
                args.ffmpeg,
                "-y",
                "-ss",
                "0.10",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                overlay,
                str(staged_png),
            ],
            check=True,
        )
        gif_filter = f"{overlay},fps=10,scale=960:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
        subprocess.run(
            [
                args.ffmpeg,
                "-y",
                "-t",
                "3.0",
                "-i",
                str(source),
                "-filter_complex",
                gif_filter,
                "-loop",
                "0",
                str(staged_gif),
            ],
            check=True,
        )
        require(
            staged_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            and staged_gif.read_bytes()[:6] in {b"GIF87a", b"GIF89a"},
            "media signature mismatch",
        )
        require(
            staged_png.stat().st_size <= MAX_PUBLIC_BYTES
            and staged_gif.stat().st_size <= MAX_PUBLIC_BYTES,
            "public media exceeds 10 MiB",
        )
        visual = {
            "schema_version": "g009.r0.rev15.camera_visual_evidence.v1",
            "status": "rejected",
            "diagnostic_only": True,
            "camera_footage": True,
            "telemetry_animation": False,
            "qualification_status": "not_run",
            "qualification_passed": None,
            "learned": False,
            "ppo_training": False,
            "candidate_runtime_calibration_passed": False,
            "labels": list(REQUIRED_LABELS),
            "overlay_labels": {"top": OVERLAY_TOP, "bottom": OVERLAY_BOTTOM},
            "pose_id": "right_side",
            "action_mode": "reset_pose_hold",
            "source_capture": {
                "path": repo_path(args.capture),
                "sha256": file_sha256(args.capture),
            },
            "source_builder": {
                "path": repo_path(Path(__file__)),
                "sha256": file_sha256(Path(__file__)),
            },
            "source": capture["source"],
            "original_rev15_report_binding": capture[
                "original_rev15_report_binding"
            ],
            "headless": True,
            "offscreen": True,
            "evidence_scope": capture["evidence_scope"],
            "completed_stages": capture["completed_stages"],
            "blocked_stages": capture["blocked_stages"],
            "local_video": {**capture["local_video"], **source_probe},
            "public_artifacts": {
                "gif": artifact(
                    staged_gif,
                    "camera_footage_gif",
                    args.ffprobe,
                    public_path=args.gif,
                ),
                "png": artifact(
                    staged_png,
                    "camera_footage_still",
                    args.ffprobe,
                    public_path=args.png,
                ),
            },
        }
        staged_visual.write_text(
            json.dumps(visual, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        require(
            json.loads(staged_visual.read_text(encoding="utf-8")) == visual,
            "staged camera visual evidence round-trip mismatch",
        )
        published: list[tuple[Path, str]] = []
        try:
            for staged, final in (
                (staged_png, args.png),
                (staged_gif, args.gif),
                (staged_visual, args.visual),
            ):
                published.append((final, publish_new(staged, final)))
            require(
                file_sha256(args.png)
                == visual["public_artifacts"]["png"]["sha256"]
                and file_sha256(args.gif)
                == visual["public_artifacts"]["gif"]["sha256"]
                and json.loads(args.visual.read_text(encoding="utf-8")) == visual,
                "published camera evidence integrity mismatch",
            )
        except Exception:
            cleanup_owned_outputs(published)
            raise
        return visual


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--visual", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument(
        "--font", type=Path, default=Path("C:/Windows/Fonts/arialbd.ttf")
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = build(parse_args(argv))
    print(
        json.dumps(
            {
                "gif": result["public_artifacts"]["gif"],
                "png": result["public_artifacts"]["png"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
