#!/usr/bin/env python3
"""Build hash-bound G009-5-E012 rev19 contact-offset telemetry media."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = Path(__file__).resolve()
RUNS_DIR = REPO_ROOT / "reports/runs"
PUBLIC_MEDIA_DIR = REPO_ROOT / "docs/media/g009/R0/diagnostic"
LOCAL_VIDEO_DIR = Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
EVIDENCE_ID = "G009-5-E012"
STAGE_NUMBER = "12"
OUTPUT_STEM = "g009_5_r0_e012_rev19_contact_offset_intervention"
DEFAULT_SYNTHESIS = RUNS_DIR / "g009_r0_rev19_contact_offset_intervention_synthesis_2x2x2_s42.json"
DEFAULT_PREFLIGHT = RUNS_DIR / "g009_r0_rev19_contact_offset_cpu_preflight_2x2_s42.json"
DEFAULT_VIDEO = LOCAL_VIDEO_DIR / f"{OUTPUT_STEM}_s42.mp4"
DEFAULT_PREFLIGHT_PNG = PUBLIC_MEDIA_DIR / f"{OUTPUT_STEM}_01_cpu_preflight.png"
DEFAULT_FINAL_PNG = PUBLIC_MEDIA_DIR / f"{OUTPUT_STEM}_02_final_outcome.png"
DEFAULT_GIF = PUBLIC_MEDIA_DIR / f"{OUTPUT_STEM}.gif"
DEFAULT_SIDECAR = RUNS_DIR / f"{OUTPUT_STEM}_visual_evidence.json"
EXPECTED_REPORTS = (
    "reports/runs/g009_r0_rev19_contact_offset_armA_cpu_rep01_s42.json",
    "reports/runs/g009_r0_rev19_contact_offset_armA_cpu_rep02_s42.json",
    "reports/runs/g009_r0_rev19_contact_offset_armB_cpu_rep01_s42.json",
    "reports/runs/g009_r0_rev19_contact_offset_armB_cpu_rep02_s42.json",
    "reports/runs/g009_r0_rev19_contact_offset_armA_gpu_rep01_s42.json",
    "reports/runs/g009_r0_rev19_contact_offset_armA_gpu_rep02_s42.json",
    "reports/runs/g009_r0_rev19_contact_offset_armB_gpu_rep01_s42.json",
    "reports/runs/g009_r0_rev19_contact_offset_armB_gpu_rep02_s42.json",
)
EXPECTED_SLOTS = (
    "A.cpu.rep1", "A.cpu.rep2", "B.cpu.rep1", "B.cpu.rep2",
    "A.cuda:0.rep1", "A.cuda:0.rep2", "B.cuda:0.rep1", "B.cuda:0.rep2",
)
EXPECTED_REPEATABILITY_GROUPS = ("A.cpu", "A.cuda:0", "B.cpu", "B.cuda:0")
FRAME_COUNT = 8
FRAME_DURATION_MS = 700
VIDEO_FPS = 30
VIDEO_DURATION_SECONDS = FRAME_COUNT * FRAME_DURATION_MS / 1000
WIDTH = 1280
HEIGHT = 720
MAX_PUBLIC_BYTES = 10 * 1024 * 1024
REQUIRED_LABELS = (
    "TELEMETRY ANIMATION",
    "NOT CAMERA FOOTAGE",
    "DIAGNOSTIC ONLY",
    "NO PPO",
    "NOT QUALIFIED",
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


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")


def portable_local_path(path: Path) -> str:
    return str(Path("%USERPROFILE%") / path.resolve().relative_to(Path.home().resolve())).replace("/", "\\")


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def resolve_report(relative: Any) -> Path:
    require(isinstance(relative, str), "report path must be a string")
    normalized = Path(relative.replace("\\", "/"))
    require(
        not normalized.is_absolute()
        and normalized.parts[:2] == ("reports", "runs")
        and len(normalized.parts) == 3
        and normalized.suffix.lower() == ".json",
        "report binding must be a direct reports/runs JSON",
    )
    resolved = (REPO_ROOT / normalized).resolve(strict=True)
    require(resolved.parent == RUNS_DIR.resolve(), "report binding escaped reports/runs")
    return resolved


def direct_child(path: Path, parent: Path, suffix: str, label: str, *, exists: bool) -> Path:
    require(not path.is_symlink(), f"{label} must not be a symlink")
    resolved = path.resolve(strict=exists)
    require(resolved.parent == parent.resolve(), f"{label} must be a direct child of {parent}")
    require(resolved.suffix.lower() == suffix, f"{label} must use {suffix}")
    return resolved


def validate_inputs(synthesis_path: Path, preflight_path: Path) -> dict[str, Any]:
    synthesis = read_json(synthesis_path)
    require(
        synthesis.get("schema_version") == "g009.r0.rev19.contact_offset_intervention_synthesis.v1"
        and synthesis.get("evidence_id") == EVIDENCE_ID
        and synthesis.get("status") == "complete"
        and synthesis.get("input_report_count") == 8,
        "E012 synthesis identity mismatch",
    )
    integrity = synthesis.get("integrity", {})
    require(
        integrity.get("passed") is True
        and integrity.get("hash_bound") is True
        and tuple(integrity.get("exact_slots", ())) == EXPECTED_SLOTS,
        "E012 synthesis integrity mismatch",
    )
    require(
        synthesis.get("decision") == {
            "outcome": "gpu_raw_unavailable_both_arms",
            "next_step": "stop_without_gpu_contact_absence_claim",
            "selected_lever": None,
        },
        "E012 decision mismatch",
    )
    governance = synthesis.get("governance", {})
    require(
        governance.get("diagnostic_only") is True
        and governance.get("learned") is False
        and governance.get("selected_lever") is None
        and governance.get("ppo") == {"allowed": False, "status": "not_run", "updates": 0}
        and governance.get("qualification", {}).get("status") == "not_run",
        "E012 governance mismatch",
    )
    preflight_binding = synthesis.get("cpu_preflight", {}).get("binding", {})
    require(
        preflight_binding.get("path") == repo_path(preflight_path)
        and valid_sha256(preflight_binding.get("sha256"))
        and file_sha256(preflight_path) == preflight_binding.get("sha256")
        and synthesis.get("cpu_preflight", {}).get("passed") is True,
        "CPU preflight binding mismatch",
    )
    raw_observation = synthesis.get("raw_callback_observation")
    require(isinstance(raw_observation, dict), "raw callback observation must be an object")
    raw_observation_map = cast(dict[str, Any], raw_observation)
    repeatability_value = raw_observation_map.get("repeatability")
    require(
        isinstance(repeatability_value, dict)
        and set(repeatability_value) == set(EXPECTED_REPEATABILITY_GROUPS),
        "four-group repeatability contract mismatch",
    )
    repeatability = cast(dict[str, Any], repeatability_value)
    for group in EXPECTED_REPEATABILITY_GROUPS:
        group_value = repeatability.get(group)
        require(
            isinstance(group_value, dict) and group_value.get("repeatable") is True,
            f"repeatability contract mismatch: {group}",
        )
    preflight = read_json(preflight_path)
    require(
        preflight.get("schema_version") == "g009.r0.rev19.contact_offset_cpu_preflight.v1"
        and preflight.get("evidence_id") == EVIDENCE_ID
        and preflight.get("status") == "complete"
        and preflight.get("input_report_count") == 4
        and preflight.get("cpu_preflight") == {
            "passed": True,
            "raw_pass_probe_valid_safety_pass": True,
            "within_arm_repeatability_passed": True,
            "gpu_stage_allowed": True,
        },
        "CPU preflight contract mismatch",
    )
    bindings_value = synthesis.get("input_reports")
    require(
        isinstance(bindings_value, list)
        and len(bindings_value) == 8
        and all(isinstance(item, dict) for item in bindings_value),
        "exactly eight object run bindings required",
    )
    bindings = cast(list[dict[str, Any]], bindings_value)
    require(tuple(item.get("path") for item in bindings) == EXPECTED_REPORTS, "run binding order/path mismatch")
    runs: list[dict[str, Any]] = []
    for expected_slot, binding in zip(EXPECTED_SLOTS, bindings, strict=True):
        report_path = resolve_report(binding.get("path"))
        require(valid_sha256(binding.get("sha256")) and file_sha256(report_path) == binding.get("sha256"), "run hash mismatch")
        report = read_json(report_path)
        arm, device, replicate = expected_slot.rsplit(".", 2)
        replicate_index = int(replicate.removeprefix("rep"))
        require(
            report.get("schema_version") == "g009.r0.rev19.contact_offset_intervention.v1"
            and report.get("status") == "complete"
            and report.get("arm") == arm
            and report.get("device") == device
            and report.get("replicate_index") == replicate_index
            and report.get("headless") is True
            and report.get("physics_substeps") == 150,
            f"run identity mismatch: {expected_slot}",
        )
        safety = report.get("manual_probe_safety", {})
        feasibility = report.get("feasibility", {})
        offset = report.get("offset_integrity", {})
        expected_raw = device == "cpu"
        require(
            safety.get("available") is True
            and safety.get("passed") is True
            and feasibility.get("probe_valid") is True
            and feasibility.get("raw_observation_passed") is expected_raw
            and feasibility.get("offset_integrity_passed") is True,
            f"run telemetry contract mismatch: {expected_slot}",
        )
        observations = safety.get("observations", {})
        scale = offset.get("contact_offset_scale")
        require(scale == (1.0 if arm == "A" else 1.5), f"contact offset scale mismatch: {expected_slot}")
        force_bw = observations.get("all_env_non_foot_peak_force_body_weight")
        separation = observations.get("cpu_raw_minimum_separation_m", {}).get("all_env_minimum")
        require(isinstance(force_bw, (int, float)) and 0 <= force_bw <= 15.0, f"force BW mismatch: {expected_slot}")
        require((device == "cpu" and isinstance(separation, (int, float))) or (device != "cpu" and separation is None), f"CPU separation mismatch: {expected_slot}")
        runs.append({
            "slot": expected_slot,
            "arm": arm,
            "device": device,
            "replicate_index": replicate_index,
            "binding": binding,
            "callback_available": expected_raw,
            "force_body_weight": float(force_bw),
            "cpu_minimum_separation_m": float(separation) if separation is not None else None,
            "contact_offset_scale": float(scale),
            "safety_passed": True,
            "repeatability_passed": True,
        })
    preflight_reports = preflight.get("input_reports", [])
    require(preflight_reports == bindings[:4], "preflight run bindings must equal synthesis CPU bindings")
    return {"synthesis": synthesis, "preflight": preflight, "runs": runs, "bindings": bindings}


def phase_series(data: dict[str, Any], progress: float) -> dict[str, Any]:
    runs = data["runs"]
    phase_progress = min(progress * 2, 1.0) if progress <= 0.5 else min((progress - 0.5) * 2, 1.0)
    visible_runs = runs[:4] if progress <= 0.5 else runs
    return {
        "labels": [item["slot"].replace("cuda:0", "GPU") for item in visible_runs],
        "callback": [(100.0 if item["callback_available"] else 0.0) * phase_progress for item in visible_runs],
        "force": [item["force_body_weight"] * phase_progress for item in visible_runs],
        "colors": ["#58c4dc" if item["device"] == "cpu" else "#ff8c69" for item in visible_runs],
    }


def render_frame(data: dict[str, Any], progress: float, destination: Path) -> None:
    import matplotlib  # pyright: ignore[reportMissingImports]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    runs = data["runs"]
    phase = "01 · CPU PREFLIGHT" if progress <= 0.5 else "02 · FINAL CPU→GPU 2×2×2"
    series = phase_series(data, progress)
    labels, callback, force, colors = (series[key] for key in ("labels", "callback", "force", "colors"))
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0b1420")
    grid = figure.add_gridspec(3, 2, height_ratios=(0.42, 1.35, 0.9), hspace=0.45, wspace=0.3)
    title = figure.add_subplot(grid[0, :]); title.axis("off")
    title.text(0.5, 0.78, f"G009-5-E012 · 12 · {phase} · REV19 CONTACT-OFFSET INTERVENTION", ha="center", color="white", fontsize=16, fontweight="bold")
    title.text(0.5, 0.18, "TELEMETRY ANIMATION · NOT CAMERA FOOTAGE · DIAGNOSTIC ONLY · NO PPO · NOT QUALIFIED", ha="center", color="#ffd166", fontsize=10.8, fontweight="bold")
    ax1 = figure.add_subplot(grid[1, 0]); ax1.set_facecolor("#172334")
    ax1.bar(labels, callback, color=colors); ax1.set_ylim(0, 112); ax1.set_ylabel("CALLBACK AVAILABILITY [%]", color="white", fontweight="bold")
    callback_title = "CPU PREFLIGHT · RAW CALLBACK 4/4" if progress <= 0.5 else "FINAL · RAW CALLBACK CPU 4/4 / GPU 0/4"
    ax1.set_title(callback_title, color="white", fontsize=11, fontweight="bold"); ax1.tick_params(colors="white", axis="both", labelsize=7); ax1.tick_params(axis="x", rotation=35)
    ax2 = figure.add_subplot(grid[1, 1]); ax2.set_facecolor("#172334")
    ax2.bar(labels, force, color=colors); ax2.axhline(15.0, color="#ffd166", linestyle="--", linewidth=1.5); ax2.set_ylim(0, 16.5)
    ax2.set_ylabel("NON-FOOT PEAK FORCE [BW]", color="white", fontweight="bold"); ax2.set_title("MANUAL PROBE SAFETY · LIMIT 15 BW", color="white", fontsize=11, fontweight="bold"); ax2.tick_params(colors="white", axis="both", labelsize=7); ax2.tick_params(axis="x", rotation=35)
    note = figure.add_subplot(grid[2, :]); note.axis("off")
    cpu_sep = [item["cpu_minimum_separation_m"] * 1000 for item in runs if item["device"] == "cpu"]
    note.text(0.02, 0.78, "CONTACT-OFFSET SCALE  A 1.0×  |  B 1.5×", color="#d9f0ff", fontsize=12, fontweight="bold")
    note.text(0.56, 0.78, f"CPU MIN SEPARATION  {min(cpu_sep):.3f} to {max(cpu_sep):.3f} mm", color="#d9f0ff", fontsize=12, fontweight="bold")
    safety_text = "SAFETY 4/4 PASS · CPU WITHIN-ARM REPEATABILITY 2/2 PASS" if progress <= 0.5 else "SAFETY 8/8 PASS · WITHIN-GROUP REPEATABILITY 4/4 PASS"
    outcome_text = "PREFLIGHT OUTCOME: GPU STAGE AUTHORIZED · NO LEVER SELECTED" if progress <= 0.5 else "FINAL OUTCOME: GPU RAW CALLBACK UNAVAILABLE IN BOTH ARMS · NO CONTACT-ABSENCE CLAIM · NO LEVER SELECTED"
    note.text(0.02, 0.48, safety_text, color="#80ed99", fontsize=12, fontweight="bold")
    note.text(0.02, 0.18, outcome_text, color="#ffd166", fontsize=11.2, fontweight="bold")
    note.text(0.02, -0.05, "Dynamic telemetry derived from eight headless 150-step probes; it is not robot motion or training footage.", color="#c8d2df", fontsize=9.2)
    figure.savefig(destination, format="png", facecolor=figure.get_facecolor())
    plt.close(figure)


def run_ffmpeg(pattern: Path, destination: Path, ffmpeg: str) -> None:
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(1000 / FRAME_DURATION_MS),
        "-i", str(pattern), "-vf", f"fps={VIDEO_FPS},format=yuv420p", "-c:v", "libx264", "-movflags", "+faststart",
        "-t", str(VIDEO_DURATION_SECONDS), str(destination),
    ], check=True)


def validate_media(path: Path, kind: str) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"missing {kind}: {path}")
    header = path.read_bytes()[:12]
    signatures = {"png": header.startswith(b"\x89PNG\r\n\x1a\n"), "gif": header.startswith(b"GIF8"), "mp4": header[4:8] == b"ftyp"}
    require(signatures[kind], f"invalid {kind} signature")
    if kind in {"png", "gif"}:
        require(path.stat().st_size < MAX_PUBLIC_BYTES, f"{kind} exceeds 10 MiB")


def validate_raster_metadata(path: Path, kind: str) -> None:
    from PIL import Image  # pyright: ignore[reportMissingImports]

    with Image.open(path) as media:
        require(media.size == (WIDTH, HEIGHT), f"{kind} dimensions mismatch")
        require(media.format == kind.upper(), f"{kind} format metadata mismatch")
        if kind == "gif":
            require(getattr(media, "n_frames", 1) == FRAME_COUNT, "GIF frame count mismatch")
            durations: list[int] = []
            for index in range(FRAME_COUNT):
                media.seek(index)
                duration = media.info.get("duration")
                require(isinstance(duration, int), "GIF frame duration missing")
                durations.append(cast(int, duration))
            require(durations == [FRAME_DURATION_MS] * FRAME_COUNT, "GIF frame duration mismatch")


def validate_video_metadata(path: Path, *, ffprobe: str = "ffprobe") -> None:
    completed = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(completed.stdout)
    streams = metadata.get("streams")
    require(
        isinstance(streams, list) and len(streams) == 1 and isinstance(streams[0], dict),
        "MP4 stream metadata mismatch",
    )
    stream = cast(dict[str, Any], streams[0])
    format_value = metadata.get("format")
    require(isinstance(format_value, dict), "MP4 format metadata mismatch")
    duration = format_value.get("duration")
    require(
        stream.get("codec_name") == "h264"
        and stream.get("width") == WIDTH
        and stream.get("height") == HEIGHT
        and stream.get("r_frame_rate") == f"{VIDEO_FPS}/1"
        and stream.get("nb_frames") == str(int(VIDEO_FPS * VIDEO_DURATION_SECONDS))
        and isinstance(duration, str)
        and abs(float(duration) - VIDEO_DURATION_SECONDS) <= 1e-6,
        "MP4 encoded metadata mismatch",
    )


def install_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with destination.open("xb") as target:
            created = True
            with source.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    target.write(block)
            target.flush(); os.fsync(target.fileno())
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def publish_transaction(pairs: Iterable[tuple[Path, Path]]) -> None:
    materialized = tuple(pairs)
    for source, destination in materialized:
        require(source.is_file(), f"staged input missing: {source}")
        require(not destination.exists(), f"refusing to overwrite output: {destination}")
    published: list[Path] = []
    try:
        for source, destination in materialized:
            install_exclusive(source, destination); published.append(destination)
    except BaseException:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise


def artifact(path: Path, published: Path, *, local: bool = False) -> dict[str, Any]:
    return {
        "path": portable_local_path(published) if local else repo_path(published),
        "sha256": file_sha256(path), "bytes": path.stat().st_size,
        "tracked_in_git": not local, "git_policy": "local_only" if local else "git_public",
    }


def build(synthesis_path: Path, preflight_path: Path, video_path: Path, preflight_png_path: Path, final_png_path: Path, gif_path: Path, sidecar_path: Path, *, ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    synthesis_path = direct_child(synthesis_path, RUNS_DIR, ".json", "synthesis", exists=True)
    preflight_path = direct_child(preflight_path, RUNS_DIR, ".json", "preflight", exists=True)
    video_path = direct_child(video_path, LOCAL_VIDEO_DIR, ".mp4", "video", exists=False)
    preflight_png_path = direct_child(preflight_png_path, PUBLIC_MEDIA_DIR, ".png", "preflight PNG", exists=False)
    final_png_path = direct_child(final_png_path, PUBLIC_MEDIA_DIR, ".png", "final PNG", exists=False)
    gif_path = direct_child(gif_path, PUBLIC_MEDIA_DIR, ".gif", "GIF", exists=False)
    sidecar_path = direct_child(sidecar_path, RUNS_DIR, ".json", "sidecar", exists=False)
    require(len({video_path, preflight_png_path, final_png_path, gif_path, sidecar_path}) == 5, "output paths must be distinct")
    for path in (video_path, preflight_png_path, final_png_path, gif_path, sidecar_path):
        require(not path.exists(), f"refusing to overwrite output: {path}")
    inputs = validate_inputs(synthesis_path, preflight_path)
    with tempfile.TemporaryDirectory(prefix="g009-rev19-media-") as directory:
        temp = Path(directory); frames: list[Path] = []
        for index in range(FRAME_COUNT):
            frame = temp / f"frame_{index:03d}.png"
            render_frame(inputs, (index + 1) / FRAME_COUNT, frame); frames.append(frame)
        from PIL import Image  # pyright: ignore[reportMissingImports]
        staged_preflight_png, staged_final_png = temp / "preflight.png", temp / "final.png"
        staged_gif, staged_video = temp / "public.gif", temp / "local.mp4"
        with Image.open(frames[3]) as image:
            image.convert("RGB").save(staged_preflight_png, optimize=True)
        with Image.open(frames[-1]) as image:
            image.convert("RGB").save(staged_final_png, optimize=True)
        gif_frames = [Image.open(frame).convert("P", palette=Image.Palette.ADAPTIVE, colors=96) for frame in frames]
        try:
            gif_frames[0].save(staged_gif, save_all=True, append_images=gif_frames[1:], duration=FRAME_DURATION_MS, loop=0, optimize=True)
        finally:
            for frame in gif_frames: frame.close()
        run_ffmpeg(temp / "frame_%03d.png", staged_video, ffmpeg)
        for path, kind in ((staged_preflight_png, "png"), (staged_final_png, "png"), (staged_gif, "gif"), (staged_video, "mp4")): validate_media(path, kind)
        ten_inputs = [
            {"role": "run", **binding} for binding in inputs["bindings"]
        ] + [
            {"role": "cpu_preflight", "path": repo_path(preflight_path), "sha256": file_sha256(preflight_path)},
            {"role": "final_synthesis", "path": repo_path(synthesis_path), "sha256": file_sha256(synthesis_path)},
        ]
        sidecar = {
            "schema_version": "g009.r0.rev19.contact_offset_visual_evidence.v1",
            "goal_id": "g009", "stage_id": "R0", "stage_number": STAGE_NUMBER, "revision": "rev19", "evidence_id": EVIDENCE_ID,
            "status": "diagnostic_complete", "diagnostic_only": True, "telemetry_animation": True,
            "camera_footage": False, "robot_locomotion_footage": False, "training_footage": False,
            "labels": list(REQUIRED_LABELS),
            "integrity": {"passed": True, "hash_bound": True, "input_binding_count": 10, "all_inputs_verified": True},
            "input_bindings": ten_inputs,
            "sequence": [
                {"number": "12.01", "phase": "cpu_preflight", "frames": [1, 4], "input_binding_count": 5, "outcome": "gpu_stage_authorized"},
                {"number": "12.02", "phase": "final_cpu_gpu_2x2x2", "frames": [5, 8], "input_binding_count": 10, "outcome": "gpu_raw_unavailable_both_arms"},
            ],
            "telemetry": {"runs": inputs["runs"], "callback_availability": {"cpu": "4/4", "gpu": "0/4"}, "safety": "8/8 pass", "repeatability": "4/4 groups pass", "contact_offset_scale": {"A": 1.0, "B": 1.5}},
            "decision": {"outcome": "gpu_raw_unavailable_both_arms", "gpu_contact_absence_claimed": False, "selected_lever": None},
            "public_artifacts": {
                "preflight_png": {**artifact(staged_preflight_png, preflight_png_path), "width": WIDTH, "height": HEIGHT, "representative_frame": 4, "phase": "12.01_cpu_preflight"},
                "final_png": {**artifact(staged_final_png, final_png_path), "width": WIDTH, "height": HEIGHT, "representative_frame": 8, "phase": "12.02_final_cpu_gpu_2x2x2"},
                "gif": {**artifact(staged_gif, gif_path), "width": WIDTH, "height": HEIGHT, "frames": FRAME_COUNT, "fps": 1000 / FRAME_DURATION_MS, "duration_ms": FRAME_COUNT * FRAME_DURATION_MS},
            },
            "local_video": {**artifact(staged_video, video_path, local=True), "codec": "h264", "width": WIDTH, "height": HEIGHT, "fps": VIDEO_FPS, "frames": int(VIDEO_FPS * VIDEO_DURATION_SECONDS), "duration_seconds": VIDEO_DURATION_SECONDS},
            "governance": {"learned": False, "ppo": {"status": "not_run", "updates": 0}, "qualification": {"status": "not_run", "passed": None}},
            "builder": {"path": repo_path(BUILDER_SOURCE), "sha256": file_sha256(BUILDER_SOURCE)},
        }
        staged_sidecar = temp / "sidecar.json"
        staged_sidecar.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        publish_transaction(((staged_video, video_path), (staged_preflight_png, preflight_png_path), (staged_final_png, final_png_path), (staged_gif, gif_path), (staged_sidecar, sidecar_path)))
    published_outputs = (video_path, preflight_png_path, final_png_path, gif_path, sidecar_path)
    try:
        validate_bundle(
            sidecar_path,
            video_path=video_path,
            preflight_png_path=preflight_png_path,
            final_png_path=final_png_path,
            gif_path=gif_path,
        )
    except BaseException as validation_error:
        cleanup_errors: list[OSError] = []
        for path in published_outputs:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise RuntimeError(
                f"final validation failed and {len(cleanup_errors)} generated outputs could not be removed"
            ) from validation_error
        raise
    return sidecar


def validate_bundle(
    sidecar_path: Path,
    *,
    video_path: Path = DEFAULT_VIDEO,
    preflight_png_path: Path = DEFAULT_PREFLIGHT_PNG,
    final_png_path: Path = DEFAULT_FINAL_PNG,
    gif_path: Path = DEFAULT_GIF,
) -> dict[str, Any]:
    sidecar = read_json(sidecar_path)
    require(
        sidecar.get("schema_version") == "g009.r0.rev19.contact_offset_visual_evidence.v1"
        and sidecar.get("goal_id") == "g009"
        and sidecar.get("stage_id") == "R0"
        and sidecar.get("evidence_id") == EVIDENCE_ID
        and sidecar.get("stage_number") == STAGE_NUMBER
        and sidecar.get("revision") == "rev19"
        and sidecar.get("status") == "diagnostic_complete"
        and sidecar.get("diagnostic_only") is True
        and sidecar.get("telemetry_animation") is True
        and sidecar.get("camera_footage") is False
        and sidecar.get("robot_locomotion_footage") is False
        and sidecar.get("training_footage") is False,
        "sidecar identity mismatch",
    )
    require(sidecar.get("labels") == list(REQUIRED_LABELS), "required frame labels mismatch")
    require(
        sidecar.get("integrity")
        == {"passed": True, "hash_bound": True, "input_binding_count": 10, "all_inputs_verified": True},
        "sidecar integrity contract mismatch",
    )
    require(
        sidecar.get("sequence") == [
            {"number": "12.01", "phase": "cpu_preflight", "frames": [1, 4], "input_binding_count": 5, "outcome": "gpu_stage_authorized"},
            {"number": "12.02", "phase": "final_cpu_gpu_2x2x2", "frames": [5, 8], "input_binding_count": 10, "outcome": "gpu_raw_unavailable_both_arms"},
        ],
        "numbered phase sequence mismatch",
    )
    bindings_value = sidecar.get("input_bindings")
    require(
        isinstance(bindings_value, list)
        and len(bindings_value) == 10
        and all(isinstance(item, dict) for item in bindings_value),
        "ten object input bindings required",
    )
    bindings = cast(list[dict[str, Any]], bindings_value)
    require(
        [item.get("role") for item in bindings] == ["run"] * 8 + ["cpu_preflight", "final_synthesis"]
        and [item.get("path") for item in bindings]
        == list(EXPECTED_REPORTS) + [repo_path(DEFAULT_PREFLIGHT), repo_path(DEFAULT_SYNTHESIS)],
        "ten exact input bindings required",
    )
    for binding in bindings:
        path = resolve_report(binding.get("path"))
        require(valid_sha256(binding.get("sha256")) and file_sha256(path) == binding.get("sha256"), "sidecar input hash mismatch")
    canonical = validate_inputs(DEFAULT_SYNTHESIS, DEFAULT_PREFLIGHT)
    require(
        sidecar.get("telemetry")
        == {
            "runs": canonical["runs"],
            "callback_availability": {"cpu": "4/4", "gpu": "0/4"},
            "safety": "8/8 pass",
            "repeatability": "4/4 groups pass",
            "contact_offset_scale": {"A": 1.0, "B": 1.5},
        },
        "telemetry contract mismatch",
    )
    require(
        sidecar.get("decision")
        == {"outcome": "gpu_raw_unavailable_both_arms", "gpu_contact_absence_claimed": False, "selected_lever": None},
        "decision contract mismatch",
    )
    require(
        sidecar.get("governance")
        == {"learned": False, "ppo": {"status": "not_run", "updates": 0}, "qualification": {"status": "not_run", "passed": None}},
        "governance contract mismatch",
    )
    public_value = sidecar.get("public_artifacts")
    require(isinstance(public_value, dict) and set(public_value) == {"preflight_png", "final_png", "gif"}, "public artifact set mismatch")
    public = cast(dict[str, Any], public_value)
    public_contracts = {
        "preflight_png": (preflight_png_path, "png", {"width": WIDTH, "height": HEIGHT, "representative_frame": 4, "phase": "12.01_cpu_preflight"}),
        "final_png": (final_png_path, "png", {"width": WIDTH, "height": HEIGHT, "representative_frame": 8, "phase": "12.02_final_cpu_gpu_2x2x2"}),
        "gif": (gif_path, "gif", {"width": WIDTH, "height": HEIGHT, "frames": FRAME_COUNT, "fps": 1000 / FRAME_DURATION_MS, "duration_ms": FRAME_COUNT * FRAME_DURATION_MS}),
    }
    for name, (expected_path, kind, metadata) in public_contracts.items():
        record_value = public.get(name)
        require(isinstance(record_value, dict), f"{name} record must be an object")
        record = cast(dict[str, Any], record_value)
        require(
            record.get("path") == repo_path(expected_path)
            and record.get("tracked_in_git") is True
            and record.get("git_policy") == "git_public"
            and all(record.get(key) == value for key, value in metadata.items()),
            f"{name} public artifact contract mismatch",
        )
        validate_media(expected_path, kind)
        validate_raster_metadata(expected_path, kind)
        require(
            valid_sha256(record.get("sha256"))
            and file_sha256(expected_path) == record.get("sha256")
            and expected_path.stat().st_size == record.get("bytes"),
            f"{name} metadata mismatch",
        )
    video_value = sidecar.get("local_video")
    require(isinstance(video_value, dict), "local video record must be an object")
    video_record = cast(dict[str, Any], video_value)
    require(
        video_record.get("path") == portable_local_path(video_path)
        and video_record.get("tracked_in_git") is False
        and video_record.get("git_policy") == "local_only"
        and video_record.get("codec") == "h264"
        and video_record.get("width") == WIDTH
        and video_record.get("height") == HEIGHT
        and video_record.get("fps") == VIDEO_FPS
        and video_record.get("frames") == int(VIDEO_FPS * VIDEO_DURATION_SECONDS)
        and video_record.get("duration_seconds") == VIDEO_DURATION_SECONDS,
        "local video contract mismatch",
    )
    validate_media(video_path, "mp4")
    validate_video_metadata(video_path)
    require(
        valid_sha256(video_record.get("sha256"))
        and file_sha256(video_path) == video_record.get("sha256")
        and video_path.stat().st_size == video_record.get("bytes"),
        "MP4 metadata mismatch",
    )
    builder = sidecar.get("builder", {})
    require(builder.get("path") == repo_path(BUILDER_SOURCE) and builder.get("sha256") == file_sha256(BUILDER_SOURCE), "builder source hash mismatch")
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis", type=Path, default=DEFAULT_SYNTHESIS); parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO); parser.add_argument("--preflight-png", type=Path, default=DEFAULT_PREFLIGHT_PNG); parser.add_argument("--final-png", type=Path, default=DEFAULT_FINAL_PNG); parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR); parser.add_argument("--ffmpeg", default="ffmpeg"); parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    value = validate_bundle(args.sidecar) if args.check_only else build(args.synthesis, args.preflight, args.video, args.preflight_png, args.final_png, args.gif, args.sidecar, ffmpeg=args.ffmpeg)
    print(json.dumps({"status": "pass", "evidence_id": value["evidence_id"], "input_binding_count": len(value["input_bindings"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
