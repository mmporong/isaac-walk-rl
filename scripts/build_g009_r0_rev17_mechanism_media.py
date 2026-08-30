#!/usr/bin/env python3
"""Build hash-bound G009-5-E010 rev17 mechanism diagnostic media."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = Path(__file__).resolve()
RUNS_DIR = REPO_ROOT / "reports/runs"
PUBLIC_MEDIA_DIR = REPO_ROOT / "docs/media/g009/R0/diagnostic"
LOCAL_VIDEO_DIR = Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
DEDICATED_VALIDATOR = REPO_ROOT / "scripts/validate_g009_r0_rev17_mechanism_media.py"
STANDARD_STAGE_VALIDATOR = REPO_ROOT / "scripts/validate_g009_media_contract.py"
EVIDENCE_ID = "G009-5-E010"
OUTPUT_STEM = "g009_5_r0_e010_rev17_mechanism_split"
CANONICAL_PREDECESSOR_PATH = (
    "reports/runs/g009_r0_rev16_synthesis_12_full_retry01_s42.json"
)
CANONICAL_PREDECESSOR_SHA256 = (
    "d39931ad6ddf6104095a6276e9b6db3a047d044d203e034f2d38f1f172e0288d"
)
DEFAULT_INPUT = RUNS_DIR / "g009_r0_rev17_mechanism_split_offline_s42.json"
DEFAULT_VIDEO = LOCAL_VIDEO_DIR / f"{OUTPUT_STEM}_s42.mp4"
DEFAULT_PNG = PUBLIC_MEDIA_DIR / f"{OUTPUT_STEM}.png"
DEFAULT_GIF = PUBLIC_MEDIA_DIR / f"{OUTPUT_STEM}.gif"
DEFAULT_SUMMARY = RUNS_DIR / f"{OUTPUT_STEM}_visual_summary.json"
DEFAULT_SIDECAR = RUNS_DIR / f"{OUTPUT_STEM}_visual_evidence.json"
MAX_PUBLIC_BYTES = 10 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
GIF_SIGNATURE = b"GIF8"
MP4_FTYP_OFFSET = 4
LABELS = (
    EVIDENCE_ID,
    "MECHANISM DIAGNOSTIC",
    "INCONCLUSIVE",
    "NO LEVER SELECTED",
    "NOT PPO",
    "NOT QUALIFIED",
    "CPU CONTACT AUTHORITY ONLY",
    "GPU CONTACT TOPOLOGY UNAVAILABLE",
)
DECISION_BANNER = "OUTCOME: INCONCLUSIVE · NO LEVER SELECTED"


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
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_direct_child(
    path: Path,
    parent: Path,
    *,
    label: str,
    suffix: str,
    must_exist: bool,
) -> Path:
    require(not path.is_symlink(), f"{label} must not be a symlink")
    resolved = path.resolve(strict=must_exist)
    require(
        resolved.parent == parent.resolve(),
        f"{label} must be a direct child of {parent}",
    )
    require(resolved.suffix.lower() == suffix, f"{label} must use {suffix}")
    return resolved


def _resolve_run_binding(relative: Any, label: str) -> Path:
    require(isinstance(relative, str), f"{label} path must be a string")
    normalized = Path(relative.replace("\\", "/"))
    require(
        not normalized.is_absolute()
        and normalized.parts[:2] == ("reports", "runs")
        and len(normalized.parts) == 3
        and normalized.suffix.lower() == ".json",
        f"{label} path must be a direct reports/runs JSON binding",
    )
    resolved = (REPO_ROOT / normalized).resolve(strict=True)
    require(
        resolved.parent == RUNS_DIR.resolve(),
        f"{label} path escaped reports/runs",
    )
    return resolved


def read_summary(path: Path, *, raw: bytes | None = None) -> dict[str, Any]:
    source_bytes = path.read_bytes() if raw is None else raw
    value = json.loads(source_bytes.decode("utf-8"))
    require(isinstance(value, dict), "summary root must be an object")
    require(
        value.get("schema_version") == "g009.r0.rev17.mechanism_split.v1",
        "rev17 mechanism schema mismatch",
    )
    require(value.get("evidence_id") == EVIDENCE_ID, "evidence id mismatch")
    require(value.get("status") == "pass", "mechanism integrity status is not PASS")
    integrity = value.get("integrity", {})
    require(integrity.get("passed") is True, "summary integrity is not PASS")
    require(integrity.get("hash_bound") is True, "summary is not hash-bound")
    require(
        integrity.get("input_report_count") == 12,
        "integrity must bind exactly 12 input reports",
    )
    require(
        integrity.get("predecessor_path") == CANONICAL_PREDECESSOR_PATH
        and integrity.get("predecessor_sha256") == CANONICAL_PREDECESSOR_SHA256,
        "canonical predecessor integrity binding is missing",
    )
    predecessor_path = _resolve_run_binding(
        integrity.get("predecessor_path"), "predecessor"
    )
    require(
        predecessor_path.is_file()
        and file_sha256(predecessor_path) == integrity.get("predecessor_sha256"),
        "predecessor hash mismatch",
    )
    bindings = integrity.get("input_reports")
    require(
        isinstance(bindings, list)
        and len(bindings) == 12
        and bindings == value.get("input_reports"),
        "exactly 12 input report bindings required",
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    require(
        isinstance(predecessor, dict)
        and predecessor.get("input_reports") == bindings,
        "input bindings do not match the canonical predecessor",
    )
    bound_paths: set[Path] = set()
    for binding in bindings:
        require(isinstance(binding, dict), "input report binding must be an object")
        report_path = _resolve_run_binding(binding.get("path"), "input report")
        expected_sha256 = binding.get("sha256")
        require(
            _valid_sha256(expected_sha256)
            and file_sha256(report_path) == expected_sha256,
            "input report hash mismatch",
        )
        bound_paths.add(report_path)
    require(len(bound_paths) == 12, "input report bindings must be unique")
    require(
        value.get("diagnostic_only") is True
        and value.get("ppo", {}).get("status") == "not_run"
        and value.get("qualification", {}).get("status") == "not_run"
        and value.get("qualification", {}).get("passed") is None,
        "diagnostic governance mismatch",
    )
    mechanism = value.get("mechanism_split", {})
    decision = mechanism.get("decision", {})
    require(
        decision.get("outcome") == "inconclusive"
        and decision.get("selected_lever") is None,
        "E010 decision must remain inconclusive with no selected lever",
    )
    runs = mechanism.get("direct_observations", {}).get("runs")
    require(isinstance(runs, list) and len(runs) == 12, "twelve measured runs required")
    require(
        [row.get("evidence") for row in runs if isinstance(row, dict)] == bindings,
        "run evidence bindings do not match the twelve input reports",
    )
    expected = [(arm, device, replicate) for arm, device in (("A", "cpu"), ("A", "cuda:0"), ("B", "cpu"), ("B", "cuda:0")) for replicate in (1, 2, 3)]
    require(
        [(row.get("arm"), row.get("device"), row.get("replicate_index")) for row in runs]
        == expected,
        "mechanism run order mismatch",
    )
    for row in runs:
        peak_window = row.get("peak_window", {})
        require(
            isinstance(peak_window.get("peak_base_impulse_n_s"), (int, float))
            and float(peak_window["peak_base_impulse_n_s"]) >= 0,
            "invalid peak impulse",
        )
        require(
            isinstance(peak_window.get("window_base_impulse_n_s"), (int, float))
            and float(peak_window["window_base_impulse_n_s"]) > 0,
            "invalid window impulse",
        )
        totals = peak_window.get("body_impulse_magnitude_totals_n_s")
        require(
            isinstance(totals, dict)
            and isinstance(totals.get("base"), (int, float))
            and sum(float(number) for number in totals.values()) > 0,
            "invalid body-load totals",
        )
        authority = row.get("contact_authority", {})
        if row["device"] == "cpu":
            require(
                authority.get("authority") == "cpu_only"
                and authority.get("availability") == "observed"
                and authority.get("topology_available") is True,
                "CPU contact authority mismatch",
            )
            require(
                isinstance(authority.get("body_pair_counts"), dict)
                and authority.get("per_physics_step_status")
                == "observed_cpu_authority"
                and isinstance(authority.get("per_physics_step"), dict)
                and set(authority["per_physics_step"]) == {"128", "129", "130"}
                and all(
                    isinstance(step, dict)
                    and isinstance(step.get("body_pair_counts"), dict)
                    and type(step.get("event_count")) is int
                    and step["event_count"] >= 0
                    and type(step.get("contact_point_count")) is int
                    and step["contact_point_count"] >= 0
                    for step in authority["per_physics_step"].values()
                ),
                "CPU contact topology payload mismatch",
            )
        else:
            require(
                authority.get("authority") == "cpu_only"
                and authority.get("availability") == "unavailable_on_gpu"
                and authority.get("topology_available") is False
                and authority.get("body_pair_counts") is None,
                "GPU contact topology must be unavailable",
            )
    return value


def _groups(value: dict[str, Any]) -> list[dict[str, Any]]:
    runs = value["mechanism_split"]["direct_observations"]["runs"]
    output: list[dict[str, Any]] = []
    for arm, device in (("A", "cpu"), ("A", "cuda:0"), ("B", "cpu"), ("B", "cuda:0")):
        selected = [row for row in runs if row["arm"] == arm and row["device"] == device]
        require(len(selected) == 3, "each group requires three replicates")
        peaks = [float(row["peak_window"]["peak_base_impulse_n_s"]) for row in selected]
        windows = [float(row["peak_window"]["window_base_impulse_n_s"]) for row in selected]
        shares = []
        for row in selected:
            totals = row["peak_window"]["body_impulse_magnitude_totals_n_s"]
            shares.append(float(totals["base"]) / sum(float(number) for number in totals.values()))
        output.append(
            {
                "arm": arm,
                "device": device,
                "peak_base_impulse_n_s": sum(peaks) / 3,
                "window_base_impulse_n_s": sum(windows) / 3,
                "base_impulse_share": sum(shares) / 3,
            }
        )
    return output


def render_frame(value: dict[str, Any], progress: float, destination: Path) -> None:
    import matplotlib  # pyright: ignore[reportMissingImports]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    rows = _groups(value)
    labels = ["A CPU", "A GPU", "B CPU", "B GPU"]
    colors = ["#4ca3dd", "#7fc8f8", "#f2a65a", "#e45756"]
    peaks = [float(row["peak_base_impulse_n_s"]) * progress for row in rows]
    shares = [float(row["base_impulse_share"]) * progress for row in rows]
    impulses = [float(row["window_base_impulse_n_s"]) for row in rows]

    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#101722")
    grid = figure.add_gridspec(3, 2, height_ratios=(0.42, 1.7, 0.72), hspace=0.43, wspace=0.30)
    title = figure.add_subplot(grid[0, :])
    title.axis("off")
    title.text(0.5, 0.76, f"{EVIDENCE_ID} · REV17 MECHANISM SPLIT", ha="center", color="white", fontsize=19, fontweight="bold")
    title.text(0.5, 0.28, "MECHANISM DIAGNOSTIC · INCONCLUSIVE · NOT PPO · NOT QUALIFIED", ha="center", color="#ffcf66", fontsize=12, fontweight="bold")

    force_axis = figure.add_subplot(grid[1, 0])
    force_axis.set_facecolor("#17202d")
    force_axis.bar(labels, peaks, color=colors)
    force_axis.set_title("PEAK BASE IMPULSE NUMERATOR [N·s]", color="white", fontweight="bold")
    force_axis.set_ylim(0, max(0.1, max(float(row["peak_base_impulse_n_s"]) for row in rows) * 1.15))
    force_axis.tick_params(colors="white")
    for index, peak in enumerate(peaks):
        force_axis.text(index, peak + 0.25, f"{peak:.3f}", ha="center", color="white", fontsize=9)

    share_axis = figure.add_subplot(grid[1, 1])
    share_axis.set_facecolor("#17202d")
    share_axis.bar(labels, shares, color=colors)
    share_axis.set_ylim(0, 1.0)
    share_axis.set_title("BASE SHARE OF BODY-LOAD PROXY", color="white", fontweight="bold")
    share_axis.tick_params(colors="white")
    for index, share in enumerate(shares):
        share_axis.text(index, share + 0.025, f"{share:.3f}", ha="center", color="white", fontsize=9)

    note = figure.add_subplot(grid[2, :])
    note.axis("off")
    note.text(0.02, 0.82, "17-step window impulse [N·s]: " + " · ".join(f"{label} {impulse:.3f}" for label, impulse in zip(labels, impulses, strict=True)), color="white", fontsize=11, fontweight="bold")
    note.text(0.02, 0.55, DECISION_BANNER, color="#ffcf66", fontsize=12, fontweight="bold")
    note.text(0.02, 0.28, "CPU CONTACT AUTHORITY ONLY", color="#8ed1fc", fontsize=12, fontweight="bold")
    note.text(0.52, 0.28, "GPU CONTACT TOPOLOGY UNAVAILABLE", color="#ff9f80", fontsize=12, fontweight="bold")
    note.text(0.02, 0.02, "Contact-load proxy only · GPU force telemetry is not authoritative contact-pair topology.", color="#d8dde8", fontsize=10)
    figure.savefig(destination, format="png", facecolor=figure.get_facecolor())
    plt.close(figure)


def run_ffmpeg(frames_pattern: Path, destination: Path, ffmpeg: str) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "1.25",
            "-i",
            str(frames_pattern),
            "-vf",
            "fps=30,format=yuv420p",
            "-c:v",
            "libx264",
            "-movflags",
            "+faststart",
            "-t",
            "4.8",
            str(destination),
        ],
        check=True,
    )


def validate_media(path: Path, kind: str) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"missing media: {path}")
    header = path.read_bytes()[:12]
    if kind == "png":
        require(header.startswith(PNG_SIGNATURE), "invalid PNG signature")
    elif kind == "gif":
        require(header.startswith(GIF_SIGNATURE), "invalid GIF signature")
        require(path.stat().st_size < MAX_PUBLIC_BYTES, "GIF exceeds 10 MiB")
    elif kind == "mp4":
        require(header[MP4_FTYP_OFFSET:MP4_FTYP_OFFSET + 4] == b"ftyp", "invalid MP4 signature")
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


def _publish_transaction(
    pairs: Iterable[tuple[Path, Path]],
    validate: Callable[[], None],
) -> None:
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


def _artifact(
    path: Path,
    *,
    published_path: Path | None = None,
    local: bool = False,
) -> dict[str, Any]:
    label_path = published_path or path
    output = {
        "path": portable_local_path(label_path) if local else repo_path(label_path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if local:
        output.update({"tracked_in_git": False, "git_policy": "local_only"})
    else:
        output.update({"tracked_in_git": True, "git_policy": "git_public"})
    return output


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
    input_path = _resolve_direct_child(
        input_path,
        RUNS_DIR,
        label="input summary",
        suffix=".json",
        must_exist=True,
    )
    video_path = _resolve_direct_child(
        video_path,
        LOCAL_VIDEO_DIR,
        label="local video",
        suffix=".mp4",
        must_exist=False,
    )
    png_path = _resolve_direct_child(
        png_path,
        PUBLIC_MEDIA_DIR,
        label="public PNG",
        suffix=".png",
        must_exist=False,
    )
    gif_path = _resolve_direct_child(
        gif_path,
        PUBLIC_MEDIA_DIR,
        label="public GIF",
        suffix=".gif",
        must_exist=False,
    )
    summary_path = _resolve_direct_child(
        summary_path,
        RUNS_DIR,
        label="visual summary",
        suffix=".json",
        must_exist=False,
    )
    sidecar_path = _resolve_direct_child(
        sidecar_path,
        RUNS_DIR,
        label="visual sidecar",
        suffix=".json",
        must_exist=False,
    )
    require(
        len({video_path, png_path, gif_path, summary_path, sidecar_path}) == 5,
        "output paths must be distinct",
    )
    for path in (video_path, png_path, gif_path, summary_path, sidecar_path):
        require(not path.exists(), f"refusing to overwrite output: {path}")
    input_raw = input_path.read_bytes()
    input_sha256 = hashlib.sha256(input_raw).hexdigest()
    value = read_summary(input_path, raw=input_raw)
    with tempfile.TemporaryDirectory(prefix="g009-rev17-mechanism-media-") as directory:
        temp = Path(directory)
        frame_paths: list[Path] = []
        for index, progress in enumerate((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
            frame = temp / f"frame_{index:03d}.png"
            render_frame(value, progress, frame)
            frame_paths.append(frame)

        from PIL import Image  # pyright: ignore[reportMissingImports]

        staged_png = temp / "public.png"
        staged_gif = temp / "public.gif"
        staged_video = temp / "local.mp4"
        with Image.open(frame_paths[-1]) as image:
            image.convert("RGB").save(staged_png, optimize=True)
        frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE, colors=96) for path in frame_paths]
        try:
            frames[0].save(staged_gif, save_all=True, append_images=frames[1:], duration=800, loop=0, optimize=True)
        finally:
            for frame in frames:
                frame.close()
        run_ffmpeg(temp / "frame_%03d.png", staged_video, ffmpeg)
        validate_media(staged_png, "png")
        validate_media(staged_gif, "gif")
        validate_media(staged_video, "mp4")

        visual = {
            "schema_version": "g009.r0.rev17.mechanism_visual_summary.v1",
            "goal_id": "g009",
            "stage_id": "R0",
            "stage_number": "10",
            "revision": "rev17",
            "evidence_id": EVIDENCE_ID,
            "status": "diagnostic_complete",
            "diagnostic_only": True,
            "camera_footage": False,
            "telemetry_animation": True,
            "learned_policy_qualified": False,
            "labels": list(LABELS),
            "decision": {"outcome": "inconclusive", "selected_lever": None},
            "source": {"path": repo_path(input_path), "sha256": input_sha256},
            "source_binding": value["integrity"],
            "contact_authority": {
                "cpu": "cpu_only",
                "gpu": "topology_unavailable",
            },
            "public_artifacts": {
                "png": {**_artifact(staged_png, published_path=png_path), "width": 1280, "height": 720},
                "gif": {**_artifact(staged_gif, published_path=gif_path), "frames": 6, "duration_ms": 4800},
            },
            "local_video": {**_artifact(staged_video, published_path=video_path, local=True), "codec": "h264", "width": 1280, "height": 720, "fps": 30, "duration_seconds": 4.8},
            "governance": {"ppo": {"status": "not_run"}, "qualification": {"status": "not_run"}},
        }
        staged_summary = temp / "visual_summary.json"
        staged_summary.write_text(json.dumps(visual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        sidecar = {
            "schema_version": "g009.r0.rev17.mechanism_visual_evidence.v1",
            "goal_id": "g009",
            "stage_id": "R0",
            "stage_number": "10",
            "revision": "rev17",
            "evidence_id": EVIDENCE_ID,
            "status": "diagnostic_complete",
            "diagnostic_only": True,
            "integrity": {"passed": True, "hash_bound": True},
            "contract": {
                "kind": "g009_r0_diagnostic_extension",
                "builder_source": _artifact(BUILDER_SOURCE),
                "dedicated_validator": {
                    **_artifact(DEDICATED_VALIDATOR),
                    "command": "%PYTHON% scripts/validate_g009_r0_rev17_mechanism_media.py --check-only",
                },
                "standard_stage_validator": {
                    "path": repo_path(STANDARD_STAGE_VALIDATOR),
                    "compatible": False,
                    "reason": "E010 is diagnostic-only evidence under R0/diagnostic, not an R0 qualification sidecar.",
                },
            },
            "provenance": {
                "input": visual["source"],
                "visual_summary": _artifact(
                    staged_summary, published_path=summary_path
                ),
                "public_artifacts": visual["public_artifacts"],
                "local_video": visual["local_video"],
            },
            "labels": list(LABELS),
            "decision": visual["decision"],
            "contact_authority": visual["contact_authority"],
        }
        staged_sidecar = temp / "visual_evidence.json"
        staged_sidecar.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

        pairs = (
            (staged_video, video_path),
            (staged_png, png_path),
            (staged_gif, gif_path),
            (staged_summary, summary_path),
            (staged_sidecar, sidecar_path),
        )

        def validate_published() -> None:
            require(
                file_sha256(input_path) == input_sha256,
                "input summary changed during media build",
            )
            validate_media(video_path, "mp4")
            validate_media(png_path, "png")
            validate_media(gif_path, "gif")

        _publish_transaction(pairs, validate_published)
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
    output = build(
        args.input,
        args.video,
        args.png,
        args.gif,
        args.summary,
        args.sidecar,
        ffmpeg=args.ffmpeg,
    )
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
