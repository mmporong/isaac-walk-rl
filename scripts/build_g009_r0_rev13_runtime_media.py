#!/usr/bin/env python3
"""Build public telemetry media for the rejected G009 R0 rev13 CPU runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "reports/runs/g009_r0_runtime_probe_rev13_cpu_failure_synthesis_s42.json"
DEFAULT_PNG = REPO_ROOT / "docs/media/g009/R0/diagnostic/g009_5_r0_diag_rev13_cpu_runtime_failure.png"
DEFAULT_GIF = REPO_ROOT / "docs/media/g009/R0/diagnostic/g009_5_r0_diag_rev13_cpu_runtime_failure.gif"
DEFAULT_SUMMARY = REPO_ROOT / "reports/runs/g009_r0_runtime_probe_rev13_cpu_failure_visual_summary.json"
DEFAULT_LOCAL_VIDEO = (
    Path.home()
    / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
    / "g009_5_r0_diag_rev13_cpu_runtime_failure_s42.mp4"
)
MAX_PUBLIC_BYTES = 10 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def portable_local_path(path: Path) -> str:
    return str(Path("%USERPROFILE%") / path.resolve().relative_to(Path.home().resolve()))


def read_synthesis(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "synthesis root must be an object")
    require(value.get("experiment") == "rev13_cpu_runtime_failure", "experiment mismatch")
    require(value.get("status") == "rejected", "rev13 must remain rejected")
    require(value.get("diagnostic_only") is True, "media source must be diagnostic-only")
    require(value.get("public_claim_eligible") is False, "diagnostic cannot be claim-eligible")
    require(value.get("learned_policy_qualified") is False, "diagnostic cannot qualify a policy")
    require(value.get("qualification_status") == "not_run", "qualification must be not_run")
    repeatability = value.get("repeatability", {})
    require(repeatability.get("validated_runs") == 3, "three validated runs are required")
    require(repeatability.get("distinct_execution_ids") is True, "independent executions are required")
    require(repeatability.get("identical_failure") is True, "identical failure is required")
    failure = value.get("failure", {})
    require(failure.get("failed_check") == "nonfoot_peak_force_bounded", "failed check mismatch")
    require(failure.get("threshold_bodyweights") == 15.0, "threshold mismatch")
    require(failure.get("right_side_reset_pose_hold_peak_bodyweights") == 15.97161865234375, "peak mismatch")
    require(failure.get("numeric_invalid_terminations") == 0, "numeric termination count changed")
    require(failure.get("hard_joint_limit_terminations") == 0, "hard-limit termination count changed")
    require(all(value.get("blocked_stages", {}).values()), "all downstream stages must remain blocked")
    return value


def render_frame(synthesis: dict[str, Any], progress: float, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = float(synthesis["rev12_comparison"]["right_side_reset_pose_hold_peak_bodyweights"])
    rev13 = float(synthesis["failure"]["right_side_reset_pose_hold_peak_bodyweights"])
    threshold = float(synthesis["failure"]["threshold_bodyweights"])
    shown_rev13 = baseline + (rev13 - baseline) * progress
    change = float(synthesis["rev12_comparison"]["relative_increase_percent"])
    details = synthesis["rev12_comparison"]["right_side_reset_pose_hold"]

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.18})
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#10131a")
    grid = figure.add_gridspec(3, 2, height_ratios=(0.55, 2.25, 1.35), hspace=0.42, wspace=0.3)
    title = figure.add_subplot(grid[0, :])
    title.set_facecolor("#8b1111")
    title.text(
        0.5,
        0.68,
        "PUBLIC DIAGNOSTIC · TELEMETRY ANIMATION · NOT CAMERA FOOTAGE",
        ha="center",
        va="center",
        color="white",
        fontsize=17,
        fontweight="bold",
    )
    title.text(
        0.5,
        0.22,
        "G009 R0 REV13 · NO PPO · REJECTED",
        ha="center",
        va="center",
        color="#ffe36e",
        fontsize=20,
        fontweight="bold",
    )
    title.set_xticks([])
    title.set_yticks([])
    for spine in title.spines.values():
        spine.set_visible(False)

    force_axis = figure.add_subplot(grid[1, 0])
    bars = force_axis.bar(["rev12\n8/0", "rev13\n8/1"], [baseline, shown_rev13], color=["#4c78a8", "#e45756"])
    force_axis.axhline(threshold, color="#f2cf5b", linewidth=2.4, linestyle="--", label="runtime ceiling 15 BW")
    force_axis.set_ylim(0, 18.5)
    force_axis.set_ylabel("right-side base peak [body weight]", color="white")
    force_axis.set_title("reset_pose_hold peak contact force", color="white")
    force_axis.tick_params(axis="both", colors="white")
    force_axis.legend(loc="upper left")
    for bar, value in zip(bars, (baseline, shown_rev13), strict=True):
        force_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.3f}", ha="center", fontweight="bold")
    if progress >= 0.95:
        force_axis.text(1, 17.55, f"+{change:.3f}%", ha="center", color="#c4161c", fontweight="bold", fontsize=13)

    change_axis = figure.add_subplot(grid[1, 1])
    names = ["force peak", "root angular", "joint speed", "total excess Δv", "peak-step Δv"]
    values = [
        change,
        float(details["max_root_angular_speed_rad_s"]["relative_change_percent"]),
        float(details["max_joint_speed_rad_s"]["relative_change_percent"]),
        float(details["excess_contact_delta_v_m_s"]["relative_change_percent"]),
        float(details["peak_step_excess_contact_delta_v_m_s"]["relative_change_percent"]),
    ]
    shown_values = [value * progress for value in values]
    colors = ["#e45756" if value > 0 else "#4c78a8" for value in values]
    change_axis.barh(names, shown_values, color=colors)
    change_axis.axvline(0, color="black", linewidth=1)
    change_axis.set_xlim(-40, 80)
    change_axis.set_xlabel("rev12 → rev13 change [%]", color="white")
    change_axis.set_title("same seed, CPU, 3/3 identical failure", color="white")
    change_axis.tick_params(axis="both", colors="white")
    for index, value in enumerate(shown_values):
        label_x = value + 1.2 if value >= 0 else -38.0
        change_axis.text(label_x, index, f"{value:+.1f}%", va="center", ha="left")

    note = figure.add_subplot(grid[2, :])
    note.axis("off")
    note.set_facecolor("#10131a")
    note.text(
        0.02,
        0.78,
        "Observed: base force peak 15.972 BW at 0.645 s / physics step 129 in all 3 independent CPU runs.",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    note.text(
        0.02,
        0.48,
        "Only failed check: nonfoot_peak_force_bounded · numeric invalid 0 · hard-limit termination 0",
        color="#d8dde8",
        fontsize=12,
    )
    note.text(
        0.02,
        0.18,
        "Cautious reading: lower Δv but higher force and root angular peaks is consistent with temporal\n"
        "concentration/rotation; causality is not established.",
        color="#f2cf5b",
        fontsize=11.5,
    )
    figure.savefig(destination, format="png", facecolor=figure.get_facecolor())
    plt.close(figure)


def run_ffmpeg(frames_pattern: Path, destination: Path, ffmpeg: str) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        "1.1111111111",
        "-i",
        str(frames_pattern),
        "-vf",
        "fps=30,format=yuv420p",
        "-c:v",
        "libx264",
        "-movflags",
        "+faststart",
        "-t",
        "5.4",
        str(destination),
    ]
    subprocess.run(command, check=True)


def validate_public_media(path: Path, signature: bytes) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"missing media: {path}")
    require(path.read_bytes().startswith(signature), f"invalid media signature: {path}")
    require(path.stat().st_size < MAX_PUBLIC_BYTES, f"public media exceeds 10 MiB: {path}")


def write_outputs(
    synthesis_path: Path,
    png_path: Path,
    gif_path: Path,
    summary_path: Path,
    local_video_path: Path,
    *,
    ffmpeg: str = "ffmpeg",
    overwrite: bool = False,
) -> dict[str, Any]:
    from PIL import Image

    for path in (png_path, gif_path, summary_path, local_video_path):
        require(overwrite or not path.exists(), f"refusing to overwrite output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    synthesis = read_synthesis(synthesis_path)
    with tempfile.TemporaryDirectory(prefix="g009_rev13_runtime_media_") as temporary:
        temporary_path = Path(temporary)
        frame_paths = []
        for index, progress in enumerate((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
            frame_path = temporary_path / f"frame_{index:03d}.png"
            render_frame(synthesis, progress, frame_path)
            frame_paths.append(frame_path)
        final_frame = Image.open(frame_paths[-1]).convert("RGB")
        final_frame.save(png_path, optimize=True)
        frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE, colors=96) for path in frame_paths]
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=900, loop=0, optimize=True)
        run_ffmpeg(temporary_path / "frame_%03d.png", local_video_path, ffmpeg)

    validate_public_media(png_path, b"\x89PNG\r\n\x1a\n")
    validate_public_media(gif_path, b"GIF8")
    require(local_video_path.is_file() and local_video_path.stat().st_size > 0, "local MP4 is missing")
    summary = {
        "schema_version": 1,
        "goal_id": "g009",
        "stage_number": "G009-5",
        "stage_id": "R0",
        "status": "rejected",
        "diagnostic_only": True,
        "telemetry_animation": True,
        "camera_footage": False,
        "public_claim_eligible": False,
        "qualification_status": "not_run",
        "learned_policy_qualified": False,
        "ppo_training_run": False,
        "source": {"path": repo_path(synthesis_path), "sha256": file_sha256(synthesis_path)},
        "png": {"path": repo_path(png_path), "sha256": file_sha256(png_path), "bytes": png_path.stat().st_size, "width": 1280, "height": 720},
        "gif": {"path": repo_path(gif_path), "sha256": file_sha256(gif_path), "bytes": gif_path.stat().st_size, "frames": 6, "duration_ms": 5400},
        "local_video": {
            "path": portable_local_path(local_video_path),
            "sha256": file_sha256(local_video_path),
            "bytes": local_video_path.stat().st_size,
            "tracked_in_git": False,
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "duration_seconds": 5.4,
        },
        "labels": ["PUBLIC DIAGNOSTIC", "TELEMETRY ANIMATION", "NOT CAMERA FOOTAGE", "NO PPO", "REJECTED"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--local-video", type=Path, default=DEFAULT_LOCAL_VIDEO)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--force", action="store_true", help="overwrite only the explicitly selected output paths")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = write_outputs(
        args.input,
        args.png,
        args.gif,
        args.summary,
        args.local_video,
        ffmpeg=args.ffmpeg,
        overwrite=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
