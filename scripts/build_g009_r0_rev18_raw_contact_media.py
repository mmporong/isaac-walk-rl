#!/usr/bin/env python3
"""Build hash-bound G009-5-E011 rev18 raw-contact feasibility media."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = Path(__file__).resolve()
RUNS_DIR = REPO_ROOT / "reports/runs"
PUBLIC_MEDIA_DIR = REPO_ROOT / "docs/media/g009/R0/diagnostic"
LOCAL_VIDEO_DIR = Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
DEDICATED_VALIDATOR = REPO_ROOT / "scripts/validate_g009_r0_rev18_raw_contact_media.py"
STANDARD_STAGE_VALIDATOR = REPO_ROOT / "scripts/validate_g009_media_contract.py"
EVIDENCE_ID = "G009-5-E011"
STAGE_NUMBER = "11"
OUTPUT_STEM = "g009_5_r0_e011_rev18_raw_contact_feasibility"
DEFAULT_INPUT = RUNS_DIR / "g009_r0_rev18_raw_contact_feasibility_synthesis_2x2_s42.json"
DEFAULT_VIDEO = LOCAL_VIDEO_DIR / f"{OUTPUT_STEM}_s42.mp4"
DEFAULT_PNG = PUBLIC_MEDIA_DIR / f"{OUTPUT_STEM}.png"
DEFAULT_GIF = PUBLIC_MEDIA_DIR / f"{OUTPUT_STEM}.gif"
DEFAULT_SUMMARY = RUNS_DIR / f"{OUTPUT_STEM}_visual_summary.json"
DEFAULT_SIDECAR = RUNS_DIR / f"{OUTPUT_STEM}_visual_evidence.json"
EXPECTED_REPORTS = (
    "reports/runs/g009_r0_rev18_raw_contact_cpu_rep01_s42.json",
    "reports/runs/g009_r0_rev18_raw_contact_cpu_rep02_s42.json",
    "reports/runs/g009_r0_rev18_raw_contact_gpu_rep01_s42.json",
    "reports/runs/g009_r0_rev18_raw_contact_gpu_rep02_s42.json",
)
EXPECTED_SLOTS = ("cpu.rep1", "cpu.rep2", "cuda:0.rep1", "cuda:0.rep2")
MAX_PUBLIC_BYTES = 10 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
GIF_SIGNATURE = b"GIF8"
LABELS = (
    EVIDENCE_ID,
    "11 · RAW CONTACT FEASIBILITY",
    "CPU 2/2 RAW PASS",
    "GPU 0/2 RAW CALLBACK AVAILABILITY",
    "DIAGNOSTIC-ONLY",
    "NOT PPO",
    "NOT QUALIFIED",
    "NO LEVER SELECTED",
    "PHYSICS GROUND TRUTH AUTHORITY: FALSE",
    "RESIDUAL INSTRUMENTATION: PARTIAL/UNAVAILABLE",
)
DECISION_BANNER = "OUTCOME: UNAVAILABLE ON GPU · NO LEVER SELECTED"


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
    relative = path.resolve().relative_to(Path.home().resolve())
    return str(Path("%USERPROFILE%") / relative).replace("/", "\\")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _resolve_direct_child(
    path: Path, parent: Path, *, label: str, suffix: str, must_exist: bool
) -> Path:
    require(not path.is_symlink(), f"{label} must not be a symlink")
    resolved = path.resolve(strict=must_exist)
    require(resolved.parent == parent.resolve(), f"{label} must be a direct child of {parent}")
    require(resolved.suffix.lower() == suffix, f"{label} must use {suffix}")
    return resolved


def _resolve_report_binding(relative: Any, label: str) -> Path:
    require(isinstance(relative, str), f"{label} path must be a string")
    normalized = Path(relative.replace("\\", "/"))
    require(
        not normalized.is_absolute()
        and normalized.parts[:2] == ("reports", "runs")
        and len(normalized.parts) == 3
        and normalized.suffix.lower() == ".json",
        f"{label} must be a direct reports/runs JSON binding",
    )
    resolved = (REPO_ROOT / normalized).resolve(strict=True)
    require(resolved.parent == RUNS_DIR.resolve(), f"{label} escaped reports/runs")
    return resolved


def _read_json_bytes(path: Path, raw: bytes | None = None) -> dict[str, Any]:
    value = json.loads((path.read_bytes() if raw is None else raw).decode("utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def read_summary(path: Path, *, raw: bytes | None = None) -> dict[str, Any]:
    value = _read_json_bytes(path, raw)
    require(
        value.get("schema_version") == "g009.r0.rev18.gpu_raw_contact_synthesis.v1"
        and value.get("evidence_id") == EVIDENCE_ID
        and value.get("goal_id") == "g009"
        and value.get("stage_id") == "R0"
        and value.get("revision") == "rev18"
        and value.get("status") == "pass"
        and value.get("input_report_count") == 4,
        "E011 synthesis identity mismatch",
    )
    bindings = value.get("input_reports")
    require(isinstance(bindings, list) and len(bindings) == 4, "exactly four input report bindings required")
    bindings = cast(list[dict[str, Any]], bindings)
    require(tuple(binding.get("path") for binding in bindings if isinstance(binding, dict)) == EXPECTED_REPORTS, "input report order/path mismatch")
    integrity = cast(dict[str, Any], value.get("integrity", {}))
    require(
        integrity.get("passed") is True
        and integrity.get("hash_bound") is True
        and integrity.get("unique_execution_ids") is True
        and tuple(integrity.get("exact_slots", ())) == EXPECTED_SLOTS
        and _valid_sha256(integrity.get("source_bundle_sha256"))
        and integrity.get("source_bundle_sha256") == integrity.get("raw_probe_source_bundle_sha256")
        and _valid_sha256(integrity.get("synthesis_source_bundle_sha256")),
        "E011 synthesis integrity mismatch",
    )
    reports: list[dict[str, Any]] = []
    for expected_slot, binding in zip(EXPECTED_SLOTS, bindings, strict=True):
        require(isinstance(binding, dict), "input report binding must be an object")
        report_path = _resolve_report_binding(binding.get("path"), "input report")
        expected_hash = binding.get("sha256")
        require(_valid_sha256(expected_hash) and file_sha256(report_path) == expected_hash, "input report hash mismatch")
        report = _read_json_bytes(report_path)
        device, replicate = expected_slot.split(".rep")
        require(
            report.get("schema_version") == "g009.r0.rev18.gpu_raw_contact.v1"
            and report.get("goal_id") == "g009"
            and report.get("stage_id") == "R0"
            and report.get("revision") == "rev18"
            and report.get("status") == "complete"
            and report.get("device") == device
            and report.get("replicate_index") == int(replicate)
            and report.get("seed") == 42
            and report.get("headless") is True
            and report.get("physics_substeps") == 150
            and report.get("governance", {}).get("ppo_updates") == 0
            and report.get("governance", {}).get("learned") is False
            and report.get("governance", {}).get("physics_ground_truth_authority") is False,
            f"{expected_slot} report identity/governance mismatch",
        )
        feasibility = report.get("feasibility", {})
        expected_raw = device == "cpu"
        require(
            feasibility.get("probe_valid") is True
            and feasibility.get("raw_observation_passed") is expected_raw
            and feasibility.get("physics_ground_truth_authority") is False
            and feasibility.get("supporting_bundle_complete") is False,
            f"{expected_slot} feasibility mismatch",
        )
        residual = report.get("residual_capability", {})
        require(
            residual.get("scene", {}).get("status") == "unavailable"
            and residual.get("source_articulation_root", {}).get("status") == "unavailable",
            f"{expected_slot} residual instrumentation must be unavailable",
        )
        reports.append(report)
    raw_contact = cast(dict[str, Any], value.get("raw_contact_feasibility", {}))
    runs = raw_contact.get("runs")
    require(
        raw_contact.get("outcome") == "unavailable_on_gpu"
        and raw_contact.get("gpu_pair_attribution_available") is False
        and raw_contact.get("cpu_control_2_of_2_passed_repeatable") is True
        and isinstance(runs, list)
        and len(runs) == 4
        and tuple(run.get("slot") for run in runs if isinstance(run, dict)) == EXPECTED_SLOTS,
        "E011 raw-contact outcome mismatch",
    )
    runs = cast(list[dict[str, Any]], runs)
    for index, run in enumerate(runs):
        require(
            isinstance(run, dict)
            and run.get("binding") == bindings[index]
            and run.get("probe_valid") is True
            and run.get("raw_observation_passed") is (index < 2)
            and run.get("instrumentation_bundle_complete") is False,
            "E011 run binding/status mismatch",
        )
    require(
        value.get("instrumentation_bundle") == {
            "status": "unavailable",
            "complete_report_count": 0,
            "required_report_count": 4,
            "independent_of_raw_contact_feasibility": True,
        },
        "E011 instrumentation bundle mismatch",
    )
    require(
        value.get("decision") == {
            "outcome": "unavailable_on_gpu",
            "next_step": "pre_registered_single_variable_intervention",
            "selected_lever": None,
        },
        "E011 decision mismatch",
    )
    governance = value.get("governance", {})
    require(
        governance.get("diagnostic_only") is True
        and governance.get("selected_lever") is None
        and governance.get("learned") is False
        and governance.get("ppo") == {"allowed": False, "status": "not_run", "updates": 0}
        and governance.get("qualification") == {"eligible": False, "status": "not_run", "passed": None}
        and governance.get("gate01") == {"allowed": False, "status": "forbidden"},
        "E011 governance mismatch",
    )
    value["_validated_reports"] = reports
    return value


def render_frame(value: dict[str, Any], progress: float, destination: Path) -> None:
    import matplotlib  # pyright: ignore[reportMissingImports]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    availability = [100.0 * progress, 100.0 * progress, 0.0, 0.0]
    labels = ["CPU R1", "CPU R2", "GPU R1", "GPU R2"]
    colors = ["#48bfe3", "#56cfe1", "#ff7b72", "#ff9b85"]
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0d1521")
    grid = figure.add_gridspec(3, 1, height_ratios=(0.55, 1.6, 1.0), hspace=0.38)
    title = figure.add_subplot(grid[0])
    title.axis("off")
    title.text(0.5, 0.77, f"{EVIDENCE_ID} · 11 · REV18 RAW CONTACT FEASIBILITY", ha="center", color="white", fontsize=18, fontweight="bold")
    title.text(0.5, 0.23, "DIAGNOSTIC-ONLY · NOT PPO · NOT QUALIFIED · NO LEVER SELECTED", ha="center", color="#ffd166", fontsize=11.5, fontweight="bold")
    axis = figure.add_subplot(grid[1])
    axis.set_facecolor("#172334")
    axis.bar(labels, availability, color=colors, width=0.63)
    axis.set_ylim(0, 112)
    axis.set_ylabel("RAW CALLBACK AVAILABILITY [%]", color="white", fontweight="bold")
    axis.set_title("2×2 CONTROLLED FEASIBILITY PROBE (150 PHYSICS STEPS EACH)", color="white", fontweight="bold")
    axis.tick_params(colors="white")
    for index, amount in enumerate(availability):
        final_label = "PASS" if index < 2 else "UNAVAILABLE"
        axis.text(index, max(amount + 3, 4), final_label, ha="center", color="white", fontsize=10, fontweight="bold")
    note = figure.add_subplot(grid[2])
    note.axis("off")
    note.text(0.02, 0.84, "CPU 2/2 RAW PASS", color="#71d9ee", fontsize=12, fontweight="bold")
    note.text(0.51, 0.84, "GPU 0/2 RAW CALLBACK AVAILABILITY", color="#ff9b85", fontsize=12, fontweight="bold")
    note.text(0.02, 0.52, DECISION_BANNER, color="#ffd166", fontsize=12, fontweight="bold")
    note.text(0.02, 0.27, "PHYSICS GROUND TRUTH AUTHORITY: FALSE", color="white", fontsize=11, fontweight="bold")
    note.text(0.51, 0.27, "RESIDUAL INSTRUMENTATION: PARTIAL/UNAVAILABLE", color="white", fontsize=11, fontweight="bold")
    note.text(0.02, 0.02, "Telemetry animation only. This is neither robot locomotion footage nor reinforcement-learning training evidence.", color="#c8d2df", fontsize=9.7)
    figure.savefig(destination, format="png", facecolor=figure.get_facecolor())
    plt.close(figure)


def run_ffmpeg(frames_pattern: Path, destination: Path, ffmpeg: str) -> None:
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", "1.25", "-i", str(frames_pattern), "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-movflags", "+faststart", "-t", "4.8", str(destination)],
        check=True,
    )


def validate_media(path: Path, kind: str) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"missing media: {path}")
    header = path.read_bytes()[:12]
    if kind == "png":
        require(header.startswith(PNG_SIGNATURE), "invalid PNG signature")
    elif kind == "gif":
        require(header.startswith(GIF_SIGNATURE), "invalid GIF signature")
    elif kind == "mp4":
        require(header[4:8] == b"ftyp", "invalid MP4 signature")
    else:
        raise ValueError(f"unknown media kind: {kind}")
    if kind in {"png", "gif"}:
        require(path.stat().st_size < MAX_PUBLIC_BYTES, f"{kind.upper()} exceeds 10 MiB")


def _install_exclusive(staged: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        destination = final.open("xb")
        created = True
        with destination:
            with staged.open("rb") as source:
                while block := source.read(1024 * 1024):
                    destination.write(block)
                destination.flush()
                os.fsync(destination.fileno())
    except BaseException:
        if created:
            final.unlink(missing_ok=True)
        raise


def _publish_transaction(pairs: Iterable[tuple[Path, Path]], validate: Callable[[], None]) -> None:
    materialized = tuple(pairs)
    for staged, final in materialized:
        require(staged.is_file(), f"staged transaction input missing: {staged}")
        require(not final.exists(), f"refusing to overwrite output: {final}")
    published: list[Path] = []
    try:
        for staged, final in materialized:
            _install_exclusive(staged, final)
            published.append(final)
        validate()
    except BaseException:
        for final in published:
            final.unlink(missing_ok=True)
        raise


def _artifact(path: Path, *, published_path: Path | None = None, local: bool = False) -> dict[str, Any]:
    label_path = published_path or path
    record: dict[str, Any] = {
        "path": portable_local_path(label_path) if local else repo_path(label_path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "tracked_in_git": not local,
        "git_policy": "local_only" if local else "git_public",
    }
    return record


def build(
    input_path: Path,
    video_path: Path,
    png_path: Path,
    gif_path: Path,
    summary_path: Path,
    sidecar_path: Path,
    *,
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    input_path = _resolve_direct_child(input_path, RUNS_DIR, label="input synthesis", suffix=".json", must_exist=True)
    video_path = _resolve_direct_child(video_path, LOCAL_VIDEO_DIR, label="local video", suffix=".mp4", must_exist=False)
    png_path = _resolve_direct_child(png_path, PUBLIC_MEDIA_DIR, label="public PNG", suffix=".png", must_exist=False)
    gif_path = _resolve_direct_child(gif_path, PUBLIC_MEDIA_DIR, label="public GIF", suffix=".gif", must_exist=False)
    summary_path = _resolve_direct_child(summary_path, RUNS_DIR, label="visual summary", suffix=".json", must_exist=False)
    sidecar_path = _resolve_direct_child(sidecar_path, RUNS_DIR, label="visual sidecar", suffix=".json", must_exist=False)
    require(len({video_path, png_path, gif_path, summary_path, sidecar_path}) == 5, "output paths must be distinct")
    for path in (video_path, png_path, gif_path, summary_path, sidecar_path):
        require(not path.exists(), f"refusing to overwrite output: {path}")
    input_raw = input_path.read_bytes()
    input_sha = hashlib.sha256(input_raw).hexdigest()
    value = read_summary(input_path, raw=input_raw)
    validated_reports = value.pop("_validated_reports")
    report_bindings = value["input_reports"]
    source_binding = {
        "synthesis": {"path": repo_path(input_path), "sha256": input_sha},
        "reports": report_bindings,
        "raw_probe_source_bundle_sha256": value["integrity"]["raw_probe_source_bundle_sha256"],
        "synthesis_source_bundle_sha256": value["integrity"]["synthesis_source_bundle_sha256"],
        "predecessor": value["integrity"]["predecessor"],
    }
    with tempfile.TemporaryDirectory(prefix="g009-rev18-raw-contact-media-") as directory:
        temp = Path(directory)
        frames: list[Path] = []
        for index, progress in enumerate((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
            frame = temp / f"frame_{index:03d}.png"
            render_frame(value, progress, frame)
            frames.append(frame)
        from PIL import Image  # pyright: ignore[reportMissingImports]

        staged_png = temp / "public.png"
        staged_gif = temp / "public.gif"
        staged_video = temp / "local.mp4"
        with Image.open(frames[-1]) as image:
            image.convert("RGB").save(staged_png, optimize=True)
        gif_frames = [Image.open(frame).convert("P", palette=Image.Palette.ADAPTIVE, colors=96) for frame in frames]
        try:
            gif_frames[0].save(staged_gif, save_all=True, append_images=gif_frames[1:], duration=800, loop=0, optimize=True)
        finally:
            for frame in gif_frames:
                frame.close()
        run_ffmpeg(temp / "frame_%03d.png", staged_video, ffmpeg)
        for path, kind in ((staged_png, "png"), (staged_gif, "gif"), (staged_video, "mp4")):
            validate_media(path, kind)
        visual = {
            "schema_version": "g009.r0.rev18.raw_contact_visual_summary.v1",
            "goal_id": "g009",
            "stage_id": "R0",
            "stage_number": STAGE_NUMBER,
            "revision": "rev18",
            "evidence_id": EVIDENCE_ID,
            "status": "diagnostic_complete",
            "diagnostic_only": True,
            "camera_footage": False,
            "robot_locomotion_footage": False,
            "training_footage": False,
            "telemetry_animation": True,
            "learned_policy_qualified": False,
            "physics_ground_truth_authority": False,
            "labels": list(LABELS),
            "decision": {"outcome": "unavailable_on_gpu", "selected_lever": None},
            "result": {
                "cpu": {"raw_callback_availability": "2/2 pass"},
                "gpu": {"raw_callback_availability": "0/2 unavailable"},
                "instrumentation_bundle": "partial/unavailable",
                "physics_ground_truth_authority": False,
            },
            "source_binding": source_binding,
            "public_artifacts": {
                "png": {**_artifact(staged_png, published_path=png_path), "width": 1280, "height": 720},
                "gif": {**_artifact(staged_gif, published_path=gif_path), "frames": 6, "duration_ms": 4800},
            },
            "local_video": {**_artifact(staged_video, published_path=video_path, local=True), "codec": "h264", "width": 1280, "height": 720, "fps": 30, "duration_seconds": 4.8},
            "governance": {"ppo": {"status": "not_run", "updates": 0}, "qualification": {"status": "not_run"}, "gate01": {"status": "forbidden"}},
        }
        staged_summary = temp / "visual_summary.json"
        staged_summary.write_text(json.dumps(visual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        sidecar = {
            "schema_version": "g009.r0.rev18.raw_contact_visual_evidence.v1",
            "goal_id": "g009",
            "stage_id": "R0",
            "stage_number": STAGE_NUMBER,
            "revision": "rev18",
            "evidence_id": EVIDENCE_ID,
            "status": "diagnostic_complete",
            "diagnostic_only": True,
            "integrity": {"passed": True, "hash_bound": True},
            "contract": {
                "kind": "g009_r0_diagnostic_extension",
                "builder_source": _artifact(BUILDER_SOURCE),
                "dedicated_validator": {**_artifact(DEDICATED_VALIDATOR), "command": "%PYTHON% scripts/validate_g009_r0_rev18_raw_contact_media.py --check-only"},
                "standard_stage_validator": {"path": repo_path(STANDARD_STAGE_VALIDATOR), "compatible": False, "reason": "E011 is diagnostic-only evidence under R0/diagnostic, not an R0 qualification sidecar."},
            },
            "provenance": {
                "source_binding": source_binding,
                "visual_summary": _artifact(staged_summary, published_path=summary_path),
                "public_artifacts": visual["public_artifacts"],
                "local_video": visual["local_video"],
            },
            "labels": list(LABELS),
            "decision": visual["decision"],
            "result": visual["result"],
        }
        staged_sidecar = temp / "visual_evidence.json"
        staged_sidecar.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        input_hashes = {Path(binding["path"]): binding["sha256"] for binding in report_bindings}

        def validate_published() -> None:
            require(file_sha256(input_path) == input_sha, "input synthesis changed during media build")
            for relative, expected_hash in input_hashes.items():
                require(file_sha256(REPO_ROOT / relative) == expected_hash, "input report changed during media build")
            validate_media(video_path, "mp4")
            validate_media(png_path, "png")
            validate_media(gif_path, "gif")

        _publish_transaction(
            ((staged_video, video_path), (staged_png, png_path), (staged_gif, gif_path), (staged_summary, summary_path), (staged_sidecar, sidecar_path)),
            validate_published,
        )
    require(len(validated_reports) == 4, "validated report count drift")
    return visual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = build(args.input, args.video, args.png, args.gif, args.summary, args.sidecar, ffmpeg=args.ffmpeg)
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
