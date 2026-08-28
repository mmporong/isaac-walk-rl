#!/usr/bin/env python3
"""Build public, explicitly non-qualified charts for rev12 Gate10 attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "reports/runs/g009_r0_gate10_hard_limit_attribution_rev12_fullstate_gpu_rep01_s42.json"
)
DEFAULT_PNG = (
    REPO_ROOT
    / "docs/media/g009/R0/diagnostic/g009_5_r0_diag_rev12_gate10_fullstate_dynamics.png"
)
DEFAULT_GIF = (
    REPO_ROOT
    / "docs/media/g009/R0/diagnostic/g009_5_r0_diag_rev12_gate10_fullstate_dynamics.gif"
)
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "reports/runs/g009_r0_gate10_hard_limit_attribution_rev12_fullstate_visual_summary.json"
)
DEFAULT_LOCAL_VIDEO = (
    Path.home()
    / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
    / "g009_5_r0_diag_rev12_gate10_fullstate_dynamics_s42.mp4"
)
EXPECTED_EVENT_KEYS = ((1, 5, 338), (2, 19, 501), (3, 5, 629))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_local_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(Path.home().resolve())
    except ValueError:
        return str(resolved)
    return str(Path("%USERPROFILE%") / relative)


def read_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(report, dict), "attribution report root must be an object")
    require(report.get("outcome") == "attributed_historical_identity", "historical identity is required")
    require(report.get("attribution_contract_passed") is True, "attribution contract did not pass")
    require(report.get("gate10_safety_passed") is False, "Gate10 safety must remain failed")
    require(report.get("learned_policy_qualified") is False, "attribution cannot qualify a policy")
    require(all(report.get("checks", {}).values()), "all attribution checks must pass")
    require(all(report.get("historical_identity_checks", {}).values()), "all identity checks must pass")
    events = report.get("events")
    require(isinstance(events, list) and len(events) == 3, "exactly three attributed events are required")
    keys = tuple(
        (event.get("iteration"), event.get("rollout_control_step"), event.get("env_index"))
        for event in events
    )
    require(keys == EXPECTED_EVENT_KEYS, "attributed event topology changed")
    for event in events:
        ring = event.get("preceding_control_step_ring")
        require(isinstance(ring, list) and len(ring) == 16, "each event requires a 16-step ring")
        for frame in ring:
            for name in (
                "action_post_wrapper_clip",
                "processed_ema_target_rad",
                "applied_torque_nm",
                "joint_position_rad",
                "joint_velocity_rad_s",
            ):
                vector = frame.get(name)
                require(
                    isinstance(vector, list)
                    and len(vector) == 12
                    and all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector),
                    f"invalid ring vector: {name}",
                )
    return report


def event_series(event: dict[str, Any], margin_rad: float) -> dict[str, Any]:
    attribution = event["joint_attributions"][0]
    joint_index = int(attribution["joint_index"])
    lower = float(attribution["lower_limit_rad"])
    ring = event["preceding_control_step_ring"]
    return {
        "label": f"iter {event['iteration']} · env {event['env_index']} · {attribution['joint_name']}",
        "steps": [int(frame["global_action_step"]) for frame in ring],
        "position": [float(frame["joint_position_rad"][joint_index]) for frame in ring],
        "target": [float(frame["processed_ema_target_rad"][joint_index]) for frame in ring],
        "velocity": [float(frame["joint_velocity_rad_s"][joint_index]) for frame in ring],
        "torque": [float(frame["applied_torque_nm"][joint_index]) for frame in ring],
        "action": [float(frame["action_post_wrapper_clip"][joint_index]) for frame in ring],
        "contact_bw": [float(frame["body_force_summary"]["dominant_force_bw"]) for frame in ring],
        "contact_body": [str(frame["body_force_summary"]["dominant_body"]) for frame in ring],
        "hard_lower": lower,
        "termination_boundary": lower - margin_rad,
        "margin_excess": float(attribution["margin_excess_rad"]),
    }


def render_frame(series: dict[str, Any], event_number: int, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22})
    figure, axes = plt.subplots(3, 1, figsize=(12.8, 7.2), dpi=100, sharex=True)
    figure.suptitle(
        f"G009 R0 rev12 Gate10 full-state attribution · event {event_number}/3\n"
        "SAFETY FAIL · NOT QUALIFIED",
        fontsize=15,
        fontweight="bold",
    )
    steps = series["steps"]

    axes[0].plot(steps, series["position"], marker="o", label="actual joint position")
    axes[0].plot(steps, series["target"], marker="s", label="EMA position target")
    axes[0].axhline(series["hard_lower"], color="tab:orange", linestyle="--", label="URDF hard lower")
    axes[0].axhline(
        series["termination_boundary"], color="tab:red", linestyle=":", label="termination boundary"
    )
    axes[0].set_ylabel("joint angle [rad]")
    axes[0].legend(loc="best", ncols=2)
    axes[0].set_title(series["label"])

    axes[1].plot(steps, series["velocity"], marker="o", color="tab:blue", label="joint velocity")
    axes[1].set_ylabel("velocity [rad/s]")
    torque_axis = axes[1].twinx()
    torque_axis.plot(steps, series["torque"], marker="s", color="tab:green", label="applied torque")
    torque_axis.set_ylabel("torque [Nm]")
    lines = axes[1].get_lines() + torque_axis.get_lines()
    axes[1].legend(lines, [line.get_label() for line in lines], loc="best")

    axes[2].plot(steps, series["contact_bw"], marker="o", color="tab:red", label="dominant contact")
    axes[2].set_ylabel("contact [body weight]")
    action_axis = axes[2].twinx()
    action_axis.plot(steps, series["action"], marker="s", color="tab:purple", label="clipped action")
    action_axis.set_ylabel("action [-1, 1]")
    axes[2].set_xlabel("global policy action step")
    lines = axes[2].get_lines() + action_axis.get_lines()
    axes[2].legend(lines, [line.get_label() for line in lines], loc="upper left")

    terminal_step = steps[-1]
    terminal_body = series["contact_body"][-1]
    axes[2].annotate(
        f"{terminal_body}\n{series['contact_bw'][-1]:.2f} BW",
        xy=(terminal_step, series["contact_bw"][-1]),
        xytext=(-82, -8),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    axes[0].annotate(
        f"beyond tolerance\n{series['margin_excess']:.4f} rad",
        xy=(terminal_step, series["position"][-1]),
        xytext=(-120, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    figure.text(
        0.5,
        0.015,
        "Targets remain inside the limit and torque remains restorative; same-leg foot impact peaks at termination.",
        ha="center",
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.92))
    figure.savefig(destination, format="png")
    plt.close(figure)


def write_outputs(
    report_path: Path,
    png_path: Path,
    gif_path: Path,
    summary_path: Path,
    local_video_path: Path | None = None,
) -> dict[str, Any]:
    from PIL import Image

    for path in (png_path, gif_path, summary_path):
        require(not path.exists(), f"refusing to overwrite output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    report = read_report(report_path)
    margin_rad = float(report["contract"]["hard_joint_limit_margin_rad"])
    series = [event_series(event, margin_rad) for event in report["events"]]
    with tempfile.TemporaryDirectory(prefix="g009_gate10_attribution_media_") as temporary:
        frame_paths = []
        for index, item in enumerate(series, start=1):
            frame_path = Path(temporary) / f"event_{index}.png"
            render_frame(item, index, frame_path)
            frame_paths.append(frame_path)
        frames = [Image.open(path).convert("RGB") for path in frame_paths]
        sheet = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)), "white")
        for index, frame in enumerate(frames):
            sheet.paste(frame, (0, index * frame.height))
        sheet.save(png_path, optimize=True)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=1800,
            loop=0,
            optimize=True,
        )
    summary = {
        "schema_version": 1,
        "goal_id": "g009",
        "stage_number": "G009-5",
        "stage_id": "R0",
        "status": "diagnostic_complete",
        "diagnostic_only": True,
        "public_claim_eligible": False,
        "qualification_status": "not_run",
        "source_report": str(report_path.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_report_sha256": file_sha256(report_path),
        "event_count": 3,
        "gif": {
            "path": str(gif_path.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": file_sha256(gif_path),
            "frame_count": 3,
            "duration_ms_per_frame": 1800,
        },
        "png": {
            "path": str(png_path.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": file_sha256(png_path),
            "width": 1280,
            "height": 2160,
        },
        "local_video": None,
        "learned_policy_qualified": False,
    }
    if local_video_path is not None:
        require(local_video_path.is_file(), f"local-only video is missing: {local_video_path}")
        summary["local_video"] = {
            "path": portable_local_path(local_video_path),
            "sha256": file_sha256(local_video_path),
            "tracked_in_git": False,
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "duration_seconds": 5.4,
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = write_outputs(args.input, args.png, args.gif, args.summary, args.local_video)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
