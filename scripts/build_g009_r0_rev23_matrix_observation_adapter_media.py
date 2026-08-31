#!/usr/bin/env python3
"""Build fail-closed rev23 matrix-observation telemetry media (not camera footage)."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_g009_r0_rev23_matrix_observation_adapter_runtime as synthesis


BUILDER_SOURCE = Path(__file__).resolve()
VALIDATOR_SOURCE = SCRIPT_DIR / "validate_g009_r0_rev23_matrix_observation_adapter_media.py"
RUNS_DIR = REPO_ROOT / "reports/runs"
PUBLIC_MEDIA_DIR = REPO_ROOT / "docs/media/g009/R0/diagnostic"
LOCAL_VIDEO_DIR = Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
CPU_REPORTS = tuple(REPO_ROOT / path for path in synthesis.CPU_PATHS)
FINAL_REPORTS = tuple(REPO_ROOT / path for path in synthesis.FINAL_PATHS)
CPU_PREFLIGHT = synthesis.CPU_OUTPUT
FINAL_SYNTHESIS = synthesis.FINAL_OUTPUT
EVIDENCE_ID = "G009-5-E016"
STAGE_NUMBER = "14"
WIDTH, HEIGHT = 1280, 720
FRAME_COUNT, FRAME_DURATION_MS = 8, 700
VIDEO_FPS = 30
VIDEO_DURATION_SECONDS = FRAME_COUNT * FRAME_DURATION_MS / 1000
VIDEO_FRAME_COUNT = round(VIDEO_DURATION_SECONDS * VIDEO_FPS)
MAX_PUBLIC_BYTES = 10 * 1024 * 1024
HEADER = "DIAGNOSTIC · TELEMETRY ANIMATION · NOT CAMERA FOOTAGE · READ-ONLY · NO PPO · NOT QUALIFIED"
FOOTER = "CLAIM LIMIT: NORMAL-CONTACT ADAPTER TELEMETRY ONLY · NO LOCOMOTION / TRAINING / REWARD / QUALIFICATION CLAIM"
GOVERNANCE = {
    "diagnostic_only": True,
    "learned": False,
    "reward_computed": False,
    "ppo_updates": 0,
    "qualification_status": "not_run",
    "physics_ground_truth_authority": False,
}
_PORTABLE_VALIDATION_LOCK = threading.RLock()
CLAIM_LIMITS = {
    "telemetry_animation_only": True,
    "camera_footage_claimed": False,
    "robot_motion_claimed": False,
    "training_success_claimed": False,
    "reward_claimed": False,
    "qualification_claimed": False,
    "normal_contact_force_vector_only": True,
    "tangential_or_friction_effect_claimed": False,
}


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def portable_local_path(path: Path) -> str:
    return str(Path("%USERPROFILE%") / path.resolve().relative_to(Path.home().resolve())).replace("/", "\\")


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {token}")))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return cast(dict[str, Any], value), raw


def phase_paths(phase: str) -> dict[str, Path]:
    require(phase in {"cpu", "final"}, "phase must be cpu or final")
    stem = (
        "g009_5_r0_diag_rev23_14_01_cpu_matrix_adapter_telemetry"
        if phase == "cpu"
        else "g009_5_r0_diag_rev23_14_02_final_matrix_adapter_telemetry"
    )
    return {
        "video": LOCAL_VIDEO_DIR / f"{stem}_s42.mp4",
        "gif": PUBLIC_MEDIA_DIR / f"{stem}.gif",
        "png": PUBLIC_MEDIA_DIR / f"{stem}.png",
        "summary": RUNS_DIR / f"{stem}_visual_summary.json",
        "sidecar": RUNS_DIR / f"{stem}_visual_evidence.json",
    }


DEFAULTS = phase_paths("cpu")
DEFAULT_VIDEO, DEFAULT_GIF, DEFAULT_PNG = DEFAULTS["video"], DEFAULTS["gif"], DEFAULTS["png"]
DEFAULT_SUMMARY, DEFAULT_SIDECAR = DEFAULTS["summary"], DEFAULTS["sidecar"]


def expected_inputs(phase: str) -> tuple[Path, ...]:
    return (*CPU_REPORTS, CPU_PREFLIGHT) if phase == "cpu" else (*FINAL_REPORTS, CPU_PREFLIGHT, FINAL_SYNTHESIS)


def _governance(value: Mapping[str, Any], label: str) -> None:
    governance = value.get("governance")
    require(isinstance(governance, Mapping), f"{label} governance missing")
    for key, expected in (("learned", False), ("reward_computed", False), ("ppo_updates", 0), ("qualification_status", "not_run")):
        require(governance.get(key) == expected, f"{label} governance {key} mismatch")


@contextlib.contextmanager
def _cpu_snapshot_reconstruction():
    """Keep recorded CUDA metadata, but reconstruct immutable JSON snapshots on CPU."""

    original = synthesis.probe._snapshot_tensor
    def reconstruct(value: Any, *, dtype: Any, device: str, shape: list[int], label: str) -> Any:
        return original(value, dtype=dtype, device="cpu", shape=shape, label=label)
    with _PORTABLE_VALIDATION_LOCK:
        synthesis.probe._snapshot_tensor = reconstruct
        try:
            yield
        finally:
            synthesis.probe._snapshot_tensor = original


def validate_report_portable(report: Mapping[str, Any]) -> dict[str, Any]:
    with _cpu_snapshot_reconstruction():
        return synthesis.probe.validate_report(report)


def validate_final_portable(final: Mapping[str, Any]) -> list[dict[str, Any]]:
    with _cpu_snapshot_reconstruction():
        return synthesis.validate_final_value(final)


def _telemetry(report: Mapping[str, Any], *, already_validated: bool = False) -> dict[str, Any]:
    _governance(report, str(report.get("device")))
    if not already_validated:
        validate_report_portable(report)
    runtime = cast(Mapping[str, Any], report["adapter_runtime"])
    require(runtime.get("passed") is True and runtime.get("sample_count") == 150 and runtime.get("error") is None, "adapter runtime did not pass")
    checks = runtime.get("checks")
    require(isinstance(checks, Mapping) and checks and all(value is True for value in checks.values()), "adapter runtime checks did not all pass")
    ledger = cast(Sequence[Mapping[str, Any]], runtime["step_ledger"])
    require(len(ledger) == 150 and [row.get("step") for row in ledger] == list(range(1, 151)), "step ledger must be exact 1..150")
    rows = []
    for row in ledger:
        require(all(key in row for key in ("max_magnitude_n", "contact_body_count", "zero_source_vector_count")), "ledger telemetry field missing")
        rows.append({
            "step": int(row["step"]),
            "max_magnitude_n": float(row["max_magnitude_n"]),
            "active_mask_count": int(row["contact_body_count"]),
            "zero_vector_count": int(row["zero_source_vector_count"]),
        })
    return {
        "slot": f"{report['device']}.rep{report['replicate_index']}",
        "device": report["device"],
        "replicate_index": report["replicate_index"],
        "sample_count": runtime["sample_count"],
        "passed": runtime["passed"],
        "max_magnitude_n": runtime["max_magnitude_n"],
        "zero_source_vector_count_total": runtime["zero_source_vector_count_total"],
        "step_ledger": rows,
    }


def validate_inputs(phase: str, input_paths: Sequence[Path]) -> dict[str, Any]:
    canonical = expected_inputs(phase)
    require(tuple(path.resolve(strict=True) for path in input_paths) == tuple(path.resolve() for path in canonical), "canonical input path/order mismatch")
    reports: list[dict[str, Any]]
    if phase == "cpu":
        preflight, _ = read_json(CPU_PREFLIGHT)
        bound = synthesis.validate_cpu_preflight_value(preflight)
        _governance(preflight, "CPU preflight")
        require(preflight["decision"]["outcome"] == "gpu_stage_authorized", "CPU preflight decision mismatch")
        reports = [_telemetry(report, already_validated=True) for report in bound]
        decision, repeatability = preflight["decision"]["outcome"], {"cpu": preflight["repeatability"]}
        source = preflight["integrity"]
    else:
        final, _ = read_json(FINAL_SYNTHESIS)
        bound = validate_final_portable(final)
        reports = [_telemetry(report, already_validated=True) for report in bound]
        decision, repeatability = final["decision"]["outcome"], final["repeatability"]
        source = final["integrity"]
    bindings = [{"role": "run" if path in canonical[: len(reports)] else "synthesis", "path": repo_path(path), "sha256": file_sha256(path)} for path in canonical]
    return {
        "phase": phase,
        "reports": reports,
        "input_bindings": bindings,
        "decision": decision,
        "repeatability": repeatability,
        "git_commit": source["git_commit"],
        "source_bundle_sha256": source["probe_source_bundle_sha256"],
    }


def render_frame(data: Mapping[str, Any], progress: float, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = cast(Sequence[Mapping[str, Any]], data["reports"])
    sequence = "14.01" if data["phase"] == "cpu" else "14.02"
    upto = max(1, min(150, round(progress * 150)))
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#07131f")
    grid = figure.add_gridspec(3, 1, height_ratios=(0.65, 3.7, 1.05), left=0.07, right=0.97, top=0.97, bottom=0.055, hspace=0.30)
    title = figure.add_subplot(grid[0]); title.axis("off")
    title.text(0.015, 0.78, f"{sequence} · REV23 MATRIX OBSERVATION ADAPTER", color="#7ee8d5", fontsize=17.5, fontweight="bold", ha="left", clip_on=False)
    title.text(0.015, 0.25, HEADER, color="#ffd166", fontsize=10.5, fontweight="bold", ha="left", clip_on=False)
    plots = grid[1].subgridspec(3, 1, hspace=0.18)
    specs = (("max_magnitude_n", "MAGNITUDE [N]", "#72b7ff"), ("active_mask_count", "ACTIVE MASK", "#70d6a7"), ("zero_vector_count", "ZERO VECTORS", "#ff9f68"))
    for index, (key, ylabel, color) in enumerate(specs):
        axis = figure.add_subplot(plots[index]); axis.set_facecolor("#102338")
        for report in reports:
            ledger = cast(Sequence[Mapping[str, Any]], report["step_ledger"])
            axis.plot([row["step"] for row in ledger[:upto]], [row[key] for row in ledger[:upto]], linewidth=1.5, label=str(report["slot"]))
        axis.set_xlim(1, 150); axis.set_ylabel(ylabel, color="white", fontsize=8.5); axis.tick_params(colors="white", labelsize=8); axis.grid(alpha=0.15)
        if index == 0: axis.legend(loc="upper right", ncol=len(reports), fontsize=8, framealpha=0.25)
        if index < 2: axis.tick_params(labelbottom=False)
        else: axis.set_xlabel("PHYSICS STEP / 150", color="white", fontsize=9)
    note = figure.add_subplot(grid[2]); note.axis("off")
    slots = " · ".join(f"{item['slot']} [{item['device']}] rep{item['replicate_index']}" for item in reports)
    repeat = " / ".join(f"{key.upper()} REPEATABILITY={'PASS' if value['repeatable'] else 'FAIL'}" for key, value in cast(Mapping[str, Mapping[str, Any]], data["repeatability"]).items())
    note.text(0, 0.82, f"SLOTS: {slots}", color="#d8e7f5", fontsize=9.2)
    note.text(0, 0.54, f"PROGRESS: {upto:03d}/150 · {repeat}", color="#80ed99", fontsize=10, fontweight="bold")
    note.text(0, 0.28, f"OUTCOME: {str(data['decision']).upper()}", color="#ffd166", fontsize=9.5, fontweight="bold")
    note.text(0, 0.02, FOOTER, color="#ffd166", fontsize=8.6, fontweight="bold")
    figure.savefig(destination, facecolor=figure.get_facecolor())
    plt.close(figure)


def validate_media(path: Path, kind: str) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"missing {kind}: {path}")
    header = path.read_bytes()[:12]
    require({"png": header.startswith(b"\x89PNG\r\n\x1a\n"), "gif": header.startswith(b"GIF8"), "mp4": header[4:8] == b"ftyp"}[kind], f"invalid {kind} magic")
    if kind in {"png", "gif"}: require(path.stat().st_size < MAX_PUBLIC_BYTES, f"{kind} exceeds 10 MiB")


def ffprobe_metadata(path: Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    completed = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path)], check=True, capture_output=True, text=True)
    value = json.loads(completed.stdout); stream = value["streams"][0]
    return {"codec": stream["codec_name"], "width": stream["width"], "height": stream["height"], "fps": stream["r_frame_rate"], "frames": int(stream["nb_frames"]), "duration_seconds": float(value["format"]["duration"])}


def _owned_identity(path: Path, expected: tuple[int, int, int, str]) -> bool:
    if not path.is_file(): return False
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, file_sha256(path)) == expected


def _remove_if_owned(path: Path, expected: tuple[int, int, int, str]) -> None:
    if _owned_identity(path, expected): os.unlink(path)


def _link_verified_sibling(temp: Path, destination: Path, expected_size: int, expected_sha: str) -> tuple[int, int, int, str]:
    """Link a verified sibling and recover safely if sibling cleanup is interrupted."""

    os.link(temp, destination)
    stat = destination.stat(); identity = (stat.st_dev, stat.st_ino, stat.st_size, expected_sha)
    try:
        require(os.path.samefile(temp, destination), "atomic link inode mismatch")
        require(
            _owned_identity(destination, identity) and _owned_identity(temp, identity),
            "atomic link ownership verification mismatch",
        )
        try:
            temp.unlink()
        except OSError as cleanup_error:
            require(
                destination.exists() and temp.exists() and os.path.samefile(temp, destination)
                and _owned_identity(destination, identity) and _owned_identity(temp, identity),
                "linked sibling cleanup failed after ownership drift",
            )
            try:
                os.unlink(temp)
            except OSError:
                _remove_if_owned(destination, identity)
                _remove_if_owned(temp, identity)
                raise cleanup_error
        require(not temp.exists() and _owned_identity(destination, identity), "linked sibling cleanup recovery mismatch")
        return identity
    except BaseException:
        _remove_if_owned(destination, identity)
        _remove_if_owned(temp, identity)
        raise


def _atomic_publish_file(source: Path, destination: Path) -> tuple[int, int, int, str]:
    """Copy to an fsynced sibling, then atomically hard-link without overwrite."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size, expected_sha = source.stat().st_size, file_sha256(source)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temp = Path(temp_name)
    try:
        digest = hashlib.sha256(); copied = 0
        with os.fdopen(fd, "wb") as target, source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                target.write(block); digest.update(block); copied += len(block)
            target.flush(); os.fsync(target.fileno())
        require(copied == expected_size and digest.hexdigest() == expected_sha, "staged copy hash/size mismatch")
        require(source.stat().st_size == expected_size and file_sha256(source) == expected_sha, "source changed during atomic publish")
        require(temp.stat().st_size == expected_size and file_sha256(temp) == expected_sha, "staged sibling verification mismatch")
        return _link_verified_sibling(temp, destination, expected_size, expected_sha)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _acquire_publish_lock(marker: Path) -> tuple[Path, tuple[int, int, int, str]]:
    lock = marker.parent / f".{marker.name}.publish.lock"
    marker.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{marker.name}.lock.", suffix=".tmp", dir=marker.parent)
    temp = Path(temp_name)
    try:
        token = os.urandom(32)
        with os.fdopen(fd, "wb") as stream: stream.write(token); stream.flush(); os.fsync(stream.fileno())
        identity = _link_verified_sibling(temp, lock, len(token), hashlib.sha256(token).hexdigest())
        return lock, identity
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def publish_transaction(pairs: Iterable[tuple[Path, Path]], validate: Callable[[], None]) -> None:
    pairs = tuple(pairs)
    require(len(pairs) >= 2, "transaction requires payloads and a final commit marker")
    for source, destination in pairs:
        require(source.is_file(), f"staged input missing: {source}"); require(not destination.exists(), f"refusing to overwrite output: {destination}")
    marker_source, marker_destination = pairs[-1]
    lock, lock_identity = _acquire_publish_lock(marker_destination)
    published: list[tuple[Path, tuple[int, int, int, str]]] = []
    try:
        for source, destination in pairs[:-1]:
            identity = _atomic_publish_file(source, destination); published.append((destination, identity))
        validate()
        require(all(_owned_identity(destination, identity) for destination, identity in published), "published payload ownership changed before commit marker")
        identity = _atomic_publish_file(marker_source, marker_destination); published.append((marker_destination, identity))
    except BaseException:
        for destination, identity in reversed(published): _remove_if_owned(destination, identity)
        raise
    finally:
        _remove_if_owned(lock, lock_identity)


def artifact(path: Path, published: Path, *, local: bool = False) -> dict[str, Any]:
    return {"path": portable_local_path(published) if local else repo_path(published), "sha256": file_sha256(path), "bytes": path.stat().st_size, "intended_for_git": not local, "git_policy": "local_only" if local else "git_public_after_review"}


def build(phase: str, input_paths: Sequence[Path], outputs: Mapping[str, Path], *, ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    defaults = phase_paths(phase)
    resolved = {key: value.resolve() for key, value in outputs.items()}
    require(set(resolved) == set(defaults), "output key set mismatch")
    for key, path in resolved.items():
        require(path.parent == defaults[key].parent.resolve() and path.suffix.lower() == defaults[key].suffix, f"invalid {key} output location/type")
        require(not path.exists(), f"refusing to overwrite output: {path}")
    data = validate_inputs(phase, input_paths)
    with tempfile.TemporaryDirectory(prefix=f"g009-rev23-{phase}-media-") as directory:
        temp = Path(directory); frames = []
        for index in range(FRAME_COUNT):
            frame = temp / f"frame_{index:03d}.png"; render_frame(data, (index + 1) / FRAME_COUNT, frame); frames.append(frame)
        from PIL import Image
        staged_png, staged_gif, staged_video = temp / "public.png", temp / "public.gif", temp / "local.mp4"
        with Image.open(frames[-1]) as image: image.convert("RGB").save(staged_png, optimize=True)
        gif_frames = [Image.open(frame).convert("P", palette=Image.Palette.ADAPTIVE, colors=96) for frame in frames]
        try: gif_frames[0].save(staged_gif, save_all=True, append_images=gif_frames[1:], duration=FRAME_DURATION_MS, loop=0, optimize=True)
        finally:
            for image in gif_frames: image.close()
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(1000 / FRAME_DURATION_MS), "-i", str(temp / "frame_%03d.png"), "-vf", f"fps={VIDEO_FPS},format=yuv420p", "-c:v", "libx264", "-movflags", "+faststart", "-t", str(VIDEO_DURATION_SECONDS), str(staged_video)], check=True)
        for path, kind in ((staged_png, "png"), (staged_gif, "gif"), (staged_video, "mp4")): validate_media(path, kind)
        video_metadata = ffprobe_metadata(staged_video)
        require(video_metadata == {"codec": "h264", "width": WIDTH, "height": HEIGHT, "fps": "30/1", "frames": VIDEO_FRAME_COUNT, "duration_seconds": VIDEO_DURATION_SECONDS}, "encoded MP4 metadata mismatch")
        public = {
            "gif": {**artifact(staged_gif, resolved["gif"]), "width": WIDTH, "height": HEIGHT, "frame_count": FRAME_COUNT, "duration_ms": FRAME_COUNT * FRAME_DURATION_MS},
            "png": {**artifact(staged_png, resolved["png"]), "width": WIDTH, "height": HEIGHT, "representative_frame": FRAME_COUNT},
        }
        local_video = {**artifact(staged_video, resolved["video"], local=True), **video_metadata}
        sequence = "14.01" if phase == "cpu" else "14.02"
        summary = {"schema_version": "g009.r0.rev23.matrix_observation_adapter_visual_summary.v1", "goal_id": "g009", "stage_id": "R0", "stage_number": STAGE_NUMBER, "sequence_number": sequence, "revision": "rev23", "evidence_id": EVIDENCE_ID, "phase": phase, "status": "diagnostic_complete", "labels": [sequence, HEADER], "claim_limits": CLAIM_LIMITS, "governance": GOVERNANCE, "input_bindings": data["input_bindings"], "source": {"git_commit": data["git_commit"], "source_bundle_sha256": data["source_bundle_sha256"]}, "telemetry": {"reports": data["reports"], "repeatability": data["repeatability"]}, "decision": {"outcome": data["decision"]}, "public_artifacts": public, "local_video": local_video}
        staged_summary = temp / "summary.json"; staged_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        sidecar = {"schema_version": "g009.r0.rev23.matrix_observation_adapter_visual_evidence.v1", "goal_id": "g009", "stage_id": "R0", "stage_number": STAGE_NUMBER, "sequence_number": sequence, "revision": "rev23", "evidence_id": EVIDENCE_ID, "phase": phase, "status": "diagnostic_complete", "integrity": {"passed": True, "hash_bound": True, "all_inputs_revalidated": True, "no_overwrite": True}, "labels": [sequence, HEADER], "claim_limits": CLAIM_LIMITS, "governance": GOVERNANCE, "input_bindings": data["input_bindings"], "source": summary["source"], "artifacts": {"visual_summary": artifact(staged_summary, resolved["summary"]), "public": public, "local_video": local_video}, "contract": {"builder": {"path": repo_path(BUILDER_SOURCE), "sha256": file_sha256(BUILDER_SOURCE)}, "validator": {"path": repo_path(VALIDATOR_SOURCE), "sha256": file_sha256(VALIDATOR_SOURCE), "command": f"%PYTHON% scripts/validate_g009_r0_rev23_matrix_observation_adapter_media.py --phase {phase} --check-only"}}}
        staged_sidecar = temp / "sidecar.json"; staged_sidecar.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        frozen = {REPO_ROOT / item["path"]: item["sha256"] for item in data["input_bindings"]}
        def validate_published() -> None:
            require(all(file_sha256(path) == digest for path, digest in frozen.items()), "input changed during media build")
            for path, kind in ((resolved["video"], "mp4"), (resolved["gif"], "gif"), (resolved["png"], "png")): validate_media(path, kind)
        publish_transaction(((staged_video, resolved["video"]), (staged_gif, resolved["gif"]), (staged_png, resolved["png"]), (staged_summary, resolved["summary"]), (staged_sidecar, resolved["sidecar"])), validate_published)
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--phase", required=True, choices=("cpu", "final")); parser.add_argument("--inputs", nargs="+", type=Path)
    for key in ("video", "gif", "png", "summary", "sidecar"): parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg"); return parser


def main() -> int:
    args = build_parser().parse_args(); defaults = phase_paths(args.phase)
    outputs = {key: getattr(args, key) or path for key, path in defaults.items()}
    value = build(args.phase, args.inputs or expected_inputs(args.phase), outputs, ffmpeg=args.ffmpeg)
    print(json.dumps({"status": "pass", "phase": args.phase, "sidecar": str(outputs["sidecar"]), "input_count": len(value["input_bindings"])}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
