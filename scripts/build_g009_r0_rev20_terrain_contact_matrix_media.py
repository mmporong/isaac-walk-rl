#!/usr/bin/env python3
"""Build fail-closed G009-5-E013 rev20 terrain-contact telemetry media."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_g009_r0_rev20_terrain_contact_matrix as synthesis


BUILDER_SOURCE = Path(__file__).resolve()
VALIDATOR_SOURCE = SCRIPT_DIR / "validate_g009_r0_rev20_terrain_contact_matrix_media.py"
RUNS_DIR = REPO_ROOT / "reports/runs"
PUBLIC_MEDIA_DIR = REPO_ROOT / "docs/media/g009/R0/diagnostic"
LOCAL_VIDEO_DIR = Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
EVIDENCE_ID = "G009-5-E013"
STAGE_NUMBER = "13"
WIDTH, HEIGHT = 1280, 720
FRAME_COUNT, FRAME_DURATION_MS = 8, 700
VIDEO_FPS = 30
VIDEO_DURATION_SECONDS = FRAME_COUNT * FRAME_DURATION_MS / 1000
MAX_PUBLIC_BYTES = 10 * 1024 * 1024
REQUIRED_LABELS = (
    "13.01",
    "TELEMETRY ANIMATION",
    "NOT CAMERA FOOTAGE",
    "DIAGNOSTIC ONLY",
    "NO PPO",
    "NOT QUALIFIED",
)
FINAL_LABELS = ("13.02", *REQUIRED_LABELS[1:])
CLAIM_LIMITS = {
    "telemetry_animation_only": True,
    "camera_footage_claimed": False,
    "robot_motion_claimed": False,
    "training_success_claimed": False,
    "qualification_claimed": False,
    "physics_ground_truth_authority": False,
}
CPU_REPORTS = tuple(REPO_ROOT / path for path in synthesis.CPU_PATHS)
CPU_PREFLIGHT = synthesis.CPU_OUTPUT
FINAL_REPORTS = tuple(REPO_ROOT / path for path in synthesis.FINAL_PATHS)
FINAL_SYNTHESIS = synthesis.FINAL_OUTPUT


def phase_paths(phase: str) -> dict[str, Path]:
    require(phase in {"cpu-preflight", "final"}, "phase must be cpu-preflight or final")
    stem = (
        "g009_5_r0_e013_rev20_cpu_preflight"
        if phase == "cpu-preflight"
        else "g009_5_r0_e013_rev20_final_cpu_gpu"
    )
    return {
        "video": LOCAL_VIDEO_DIR / f"{stem}_s42.mp4",
        "gif": PUBLIC_MEDIA_DIR / f"{stem}.gif",
        "png": PUBLIC_MEDIA_DIR / f"{stem}.png",
        "summary": RUNS_DIR / f"{stem}_visual_summary.json",
        "sidecar": RUNS_DIR / f"{stem}_visual_evidence.json",
    }


def labels_for_phase(phase: str) -> tuple[str, ...]:
    require(phase in {"cpu-preflight", "final"}, "phase must be cpu-preflight or final")
    return REQUIRED_LABELS if phase == "cpu-preflight" else FINAL_LABELS


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


DEFAULTS = phase_paths("cpu-preflight")
DEFAULT_VIDEO = DEFAULTS["video"]
DEFAULT_GIF = DEFAULTS["gif"]
DEFAULT_PNG = DEFAULTS["png"]
DEFAULT_SUMMARY = DEFAULTS["summary"]
DEFAULT_SIDECAR = DEFAULTS["sidecar"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def portable_local_path(path: Path) -> str:
    return str(Path("%USERPROFILE%") / path.resolve().relative_to(Path.home().resolve())).replace("/", "\\")


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {token}")))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return cast(dict[str, Any], value), raw


def direct_child(path: Path, parent: Path, suffix: str, label: str, *, exists: bool) -> Path:
    require(not path.is_symlink(), f"{label} must not be a symlink")
    resolved = path.resolve(strict=exists)
    require(resolved.parent == parent.resolve(), f"{label} must be a direct child of {parent}")
    require(resolved.suffix.lower() == suffix, f"{label} must use {suffix}")
    return resolved


def binding(path: Path, role: str) -> dict[str, str]:
    return {"role": role, "path": repo_path(path), "sha256": file_sha256(path)}


def _run_telemetry(report: Mapping[str, Any]) -> dict[str, Any]:
    synthesis.probe.validate_report(report)
    matrix = report["terrain_contact_matrix"]
    overlap = matrix["same_step_overlap"]
    safety = matrix["safety"]
    path_order = matrix["path_order"]
    max_bw = max(float(value) for value in safety["non_foot_peak_force_body_weight_per_env"])
    coverage = [len(steps) for steps in overlap["per_env_overlap_step_indices"]]
    return {
        "slot": f"{report['device']}.rep{report['replicate_index']}",
        "availability": matrix["availability_state"],
        "structural_passed": matrix["structural_probe_valid"],
        "safety_passed": matrix["safety_valid"],
        "overlap_passed": matrix["overlap_available"],
        "baseline_passed": report["baseline_snapshot"]["all_match"],
        "live_readback_passed": report["feasibility"]["run_interpretable"],
        "per_env_overlap_coverage_steps": coverage,
        "peak_force_n": float(overlap["all_env_matrix_peak_force_n"]),
        "max_non_foot_force_bw": max_bw,
        "raw_filter_paths_sha256": path_order["raw_filter_paths_sha256"],
        "logical_filter_paths_sha256": path_order["logical_filter_paths_sha256"],
        "sensor_paths_sha256": path_order["sensor_paths_sha256"],
        "source_bundle_sha256": report["source_bundle"]["source_bundle_sha256"],
        "git_commit": report["source_bundle"]["git_commit"],
    }


def validate_recomputed_document(
    stored: Mapping[str, Any], recomputed: Mapping[str, Any], output_path: Path, forbidden_execution_ids: set[str]
) -> None:
    require(set(stored) == set(recomputed), "synthesis top-level schema mismatch")
    require(
        {key: value for key, value in stored.items() if key != "execution"}
        == {key: value for key, value in recomputed.items() if key != "execution"},
        "stored synthesis differs from canonical recomputation",
    )
    execution = stored.get("execution")
    require(
        isinstance(execution, Mapping)
        and set(execution) == {"execution_id", "started_at_utc", "output_path_repo_relative", "no_overwrite"}
        and execution.get("output_path_repo_relative") == repo_path(output_path)
        and execution.get("no_overwrite") is True
        and isinstance(execution.get("started_at_utc"), str),
        "stored synthesis execution schema mismatch",
    )
    execution = cast(Mapping[str, Any], execution)
    execution_id = synthesis.probe.validate_uuid4_hex(execution.get("execution_id"), "stored synthesis execution_id")
    require(execution_id not in forbidden_execution_ids, "stored synthesis execution_id collision")


def validate_historical_synthesis_bundle(value: Any, expected_commit: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), "historical synthesis source bundle missing")
    bundle = cast(Mapping[str, Any], value)
    expected_keys = {"schema_version", "git_commit", "git_commit_valid", "source_binding_paths", "source_binding_files", "source_bundle_sha256", "clean"}
    paths = bundle.get("source_binding_paths"); files = bundle.get("source_binding_files")
    require(
        set(bundle) == expected_keys and bundle.get("schema_version") == 1 and bundle.get("git_commit_valid") is True
        and bundle.get("git_commit") == expected_commit and re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None
        and bundle.get("clean") is True and paths == list(synthesis.probe.SYNTHESIS_SOURCE_BINDING_PATHS)
        and isinstance(files, Mapping) and set(files) == set(synthesis.probe.SYNTHESIS_SOURCE_BINDING_PATHS)
        and all(valid_sha256(digest) for digest in files.values()),
        "historical synthesis source bundle schema mismatch",
    )
    files = cast(Mapping[str, str], files)
    for relative in synthesis.probe.SYNTHESIS_SOURCE_BINDING_PATHS:
        historical = subprocess.run(["git", "show", f"{expected_commit}:{relative}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout
        require(hashlib.sha256(historical).hexdigest() == files[relative], f"historical synthesis git blob mismatch: {relative}")
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    require(bundle.get("source_bundle_sha256") == hashlib.sha256(payload.encode()).hexdigest(), "historical synthesis aggregate mismatch")
    return dict(bundle)


def verify_current_file_equivalence(
    recorded_bundle: Mapping[str, Any], expected_paths: Sequence[str], *, current_files: Mapping[str, str] | None = None, dirty_paths: Sequence[str] | None = None
) -> dict[str, str]:
    paths = list(expected_paths); recorded_paths = recorded_bundle.get("source_binding_paths"); recorded_files = recorded_bundle.get("source_binding_files")
    require(recorded_paths == paths and isinstance(recorded_files, Mapping) and set(recorded_files) == set(paths), "recorded source file map mismatch")
    recorded_files = cast(Mapping[str, str], recorded_files)
    if dirty_paths is None:
        dirty_paths = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *paths], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    require(not dirty_paths, "current bound source paths must be clean")
    if current_files is None:
        current_files = {
            relative: hashlib.sha256(subprocess.run(["git", "show", f":{relative}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout).hexdigest()
            for relative in paths
        }
    observed = dict(current_files)
    require(observed == dict(recorded_files), "current bound source file hashes differ from historical bundle")
    payload = "\n".join(f"{path}:{observed[path]}" for path in sorted(observed))
    require(hashlib.sha256(payload.encode()).hexdigest() == recorded_bundle.get("source_bundle_sha256"), "current bound source aggregate differs from historical bundle")
    return observed


def historical_final_recomputation(
    entries: Sequence[tuple[dict[str, Any], dict[str, str]]], preflight: Mapping[str, Any], preflight_raw: bytes, stored: Mapping[str, Any], output_path: Path
) -> dict[str, Any]:
    require(len(entries) == 4, "historical final synthesis requires four reports")
    rows = [synthesis.row(report, item_binding) for report, item_binding in entries]
    synthesis.validate_entry_uniqueness(rows)
    require([item["slot"] for item in rows] == ["cpu.rep1", "cpu.rep2", "cuda:0.rep1", "cuda:0.rep2"], "historical final slot order mismatch")
    require(preflight["input_reports"] == [item["binding"] for item in rows[:2]], "historical final CPU bindings differ from preflight")
    expected_gpu_binding = {"status": "validated_for_gpu", "path": repo_path(CPU_PREFLIGHT), "sha256": hashlib.sha256(preflight_raw).hexdigest(), "git_commit": preflight["integrity"]["git_commit"], "probe_source_bundle_sha256": preflight["integrity"]["probe_source_bundle_sha256"], "input_reports": preflight["input_reports"]}
    require(rows[2]["cpu_preflight_binding"] == expected_gpu_binding and rows[3]["cpu_preflight_binding"] == expected_gpu_binding, "historical GPU preflight binding mismatch")
    ids = [item["execution_id"] for item in rows] + [preflight["execution"]["execution_id"]]
    require(len(set(ids)) == 5, "historical final execution collision")
    commits = {item["git_commit"] for item in rows}; source_hashes = {item["source_bundle_sha256"] for item in rows}
    require(len(commits) == len(source_hashes) == 1 and rows[0]["git_commit"] == preflight["integrity"]["git_commit"] and rows[0]["source_bundle_sha256"] == preflight["integrity"]["probe_source_bundle_sha256"], "historical final source binding drift")
    cpu_repeat, gpu_repeat = synthesis.repeatability(rows[:2]), synthesis.repeatability(rows[2:]); states = [item["availability_state"] for item in rows]
    if not all(item["baseline_passed"] and item["device_passed"] and item["live_readback_passed"] and item["external_passed"] for item in rows): outcome = "probe_invalid"
    elif not all(item["structural_probe_valid"] for item in rows): outcome = "terrain_matrix_probe_invalid"
    elif not all(item["safety_passed"] for item in rows): outcome = "safety_limit_exceeded"
    elif not cpu_repeat["repeatable"] or states[:2] != ["observed_valid", "observed_valid"]: outcome = "inconclusive_nondeterministic_gpu_forbidden"
    elif not gpu_repeat["repeatable"]: outcome = "inconclusive_nondeterministic"
    elif states[2:] == ["observed_valid", "observed_valid"]: outcome = "terrain_pair_matrix_authority_candidate_validated"
    elif states[2:] == ["unavailable", "unavailable"]: outcome = "gpu_terrain_matrix_unavailable"
    else: outcome = "inconclusive_nondeterministic"
    historical_commit = next(iter(commits)); historical_synthesis = validate_historical_synthesis_bundle(stored.get("synthesis_source_bundle"), historical_commit)
    expected = {
        "schema_version": synthesis.FINAL_SCHEMA, "evidence_id": EVIDENCE_ID, "status": "complete", "mode": "final_2x2", "input_report_count": 4, "input_reports": [item["binding"] for item in rows],
        "integrity": {"passed": True, "hash_bound": True, "unique_report_paths": True, "unique_report_sha256": True, "unique_execution_ids": True, "exact_slots": [item["slot"] for item in rows], "preflight": {"path": repo_path(CPU_PREFLIGHT), "sha256": hashlib.sha256(preflight_raw).hexdigest()}},
        "repeatability": {"cpu": cpu_repeat, "cuda:0": gpu_repeat}, "rows": rows,
        "decision": {"outcome": outcome, "next_step": "preregister_matrix_authority_safety_gate" if outcome == "terrain_pair_matrix_authority_candidate_validated" else "stop_and_fix_filter_path_or_view_only", "third_run_allowed": False},
        "claim_limits": {"terrain_pair_aggregated_normal_force_authority_candidate_only": True, "gpu_contact_absence_claimed": False, "physics_failure_claimed": False, "callback_count_used": False},
        "governance": synthesis.probe.governance(), "synthesis_source_bundle": historical_synthesis, "execution": stored.get("execution"),
    }
    validate_recomputed_document(stored, expected, output_path, set(ids))
    return expected


def validate_inputs(phase: str, input_paths: Sequence[Path]) -> dict[str, Any]:
    expected = (*CPU_REPORTS, CPU_PREFLIGHT) if phase == "cpu-preflight" else (*FINAL_REPORTS, CPU_PREFLIGHT, FINAL_SYNTHESIS)
    require(len(input_paths) == len(expected), f"{phase} requires exactly {len(expected)} inputs")
    resolved = tuple(path.resolve(strict=True) for path in input_paths)
    require(resolved == tuple(path.resolve() for path in expected), "canonical input path/order mismatch")
    for path in resolved:
        direct_child(path, RUNS_DIR, ".json", "input", exists=True)

    report_count = 2 if phase == "cpu-preflight" else 4
    report_paths = resolved[:report_count]
    expected_report_paths = synthesis.CPU_PATHS if phase == "cpu-preflight" else synthesis.FINAL_PATHS
    entries = synthesis.load_inputs(report_paths, expected_report_paths)
    preflight_path = resolved[report_count]
    preflight, preflight_raw = read_json(preflight_path)
    historical_cpu_reports = synthesis.probe.validate_cpu_preflight_value(preflight, REPO_ROOT, repo_path(preflight_path), source_bundle=None)
    historical_probe_bundle = cast(Mapping[str, Any], historical_cpu_reports[0]["source_bundle"])
    verify_current_file_equivalence(historical_probe_bundle, synthesis.probe.SOURCE_BINDING_PATHS)
    historical_preflight_synthesis = validate_historical_synthesis_bundle(preflight["synthesis_source_bundle"], str(preflight["integrity"]["git_commit"]))
    verify_current_file_equivalence(historical_preflight_synthesis, synthesis.probe.SYNTHESIS_SOURCE_BINDING_PATHS)
    input_bindings = [{"role": "run", **item_binding} for _, item_binding in entries]
    input_bindings.append(binding(preflight_path, "cpu_preflight"))
    report_execution_ids = {synthesis.execution_id(report) for report, _ in entries}
    reports = [_run_telemetry(report) for report, _ in entries]
    if phase == "cpu-preflight":
        decision = str(preflight["decision"]["outcome"])
        repeatability = {"cpu": preflight["decision"]["repeatability"]}
    else:
        decision = ""
        repeatability = {}
    if phase == "final":
        final_path = resolved[-1]
        final, _ = read_json(final_path)
        historical_final_recomputation(entries, preflight, preflight_raw, final, final_path)
        historical_final_synthesis = cast(Mapping[str, Any], final["synthesis_source_bundle"])
        verify_current_file_equivalence(historical_final_synthesis, synthesis.probe.SYNTHESIS_SOURCE_BINDING_PATHS)
        input_bindings.append(binding(final_path, "final_synthesis"))
        decision = str(final["decision"]["outcome"])
        repeatability = final["repeatability"]

    source_hashes = {item["source_bundle_sha256"] for item in reports}
    commits = {item["git_commit"] for item in reports}
    require(len(source_hashes) == 1 and len(commits) == 1, "report source bundle/commit drift")
    return {
        "phase": phase,
        "reports": reports,
        "input_bindings": input_bindings,
        "decision": decision,
        "repeatability": repeatability,
        "git_commit": commits.pop(),
        "source_bundle_sha256": source_hashes.pop(),
        "preflight_synthesis_source_bundle_sha256": preflight["integrity"]["synthesis_source_bundle_sha256"],
    }


def render_status_lines(data: Mapping[str, Any]) -> dict[str, str]:
    reports = cast(Sequence[Mapping[str, Any]], data["reports"])
    checks = ("availability", "structural_passed", "safety_passed", "overlap_passed", "baseline_passed", "live_readback_passed")
    check_labels = ("AVAILABILITY", "STRUCTURAL", "SAFETY", "OVERLAP", "BASELINE", "LIVE")
    status = " / ".join(
        f"{label}: {'PASS' if all((item[key] == 'observed_valid') if key == 'availability' else item[key] is True for item in reports) else 'FAIL'}"
        for label, key in zip(check_labels, checks, strict=True)
    )
    repeatability = cast(Mapping[str, Any], data["repeatability"])
    groups = ["cpu"] if data["phase"] == "cpu-preflight" else ["cpu", "cuda:0"]
    repeat = " / ".join(f"{group.upper()} REPEATABILITY: {'PASS' if repeatability.get(group, {}).get('repeatable') is True else 'FAIL'}" for group in groups)
    outcome = str(data["decision"]).replace("_", " ").upper()
    return {"checks": status, "repeatability": repeat, "outcome": f"OUTCOME: {outcome}"}


def validate_text_bounds(figure: Any, artists: Sequence[Any], label: str) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer(); frame = figure.bbox
    for artist in artists:
        bounds = artist.get_window_extent(renderer=renderer)
        require(
            bounds.x0 >= frame.x0 and bounds.y0 >= frame.y0 and bounds.x1 <= frame.x1 and bounds.y1 <= frame.y1,
            f"{label} text escaped the {WIDTH}x{HEIGHT} frame",
        )


def render_frame(data: Mapping[str, Any], progress: float, destination: Path) -> None:
    import matplotlib  # pyright: ignore[reportMissingImports]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    reports = data["reports"]
    status_lines = render_status_lines(data)
    sequence = "13.01" if data["phase"] == "cpu-preflight" else "13.02"
    first = reports[0]
    coverage = first["per_env_overlap_coverage_steps"]
    reveal = max(1, min(8, int(round(progress * 8))))
    colors = ["#5bd6c8" if value == 150 else "#f4b860" for value in coverage]
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#08131f")
    grid = figure.add_gridspec(3, 2, height_ratios=(0.52, 1.5, 0.88), hspace=0.38, wspace=0.28)
    title = figure.add_subplot(grid[0, :]); title.axis("off")
    phase_title = "CPU TERRAIN CONTACT MATRIX PREFLIGHT" if data["phase"] == "cpu-preflight" else "FINAL CPU/GPU TERRAIN CONTACT MATRIX"
    title.text(0.5, 0.8, f"G009-5-E013 · {sequence} · {phase_title}", ha="center", color="white", fontsize=16, fontweight="bold")
    title.text(0.5, 0.22, "TELEMETRY ANIMATION · NOT CAMERA FOOTAGE · DIAGNOSTIC ONLY · NO PPO · NOT QUALIFIED", ha="center", color="#ffd166", fontsize=11, fontweight="bold")
    ax1 = figure.add_subplot(grid[1, 0]); ax1.set_facecolor("#132337")
    ax1.bar([f"E{i}" for i in range(reveal)], coverage[:reveal], color=colors[:reveal]); ax1.axhline(150, color="white", linestyle="--", linewidth=1)
    ax1.set_ylim(0, 160); ax1.set_title("PER-ENV SAME-STEP OVERLAP COVERAGE", color="white", fontweight="bold"); ax1.set_ylabel("STEPS / 150", color="white"); ax1.tick_params(colors="white")
    ax2 = figure.add_subplot(grid[1, 1]); ax2.set_facecolor("#132337")
    labels = [item["slot"] for item in reports]
    peaks = [item["peak_force_n"] * progress for item in reports]
    ax2.bar(labels, peaks, color="#6ca0dc"); ax2.set_ylim(0, max(item["peak_force_n"] for item in reports) * 1.12)
    ax2.set_title("AGGREGATED TERRAIN-PAIR MATRIX PEAK", color="white", fontweight="bold"); ax2.set_ylabel("FORCE [N]", color="white"); ax2.tick_params(colors="white")
    note = figure.add_subplot(grid[2, :]); note.axis("off")
    checks_pass = "FAIL" not in status_lines["checks"]
    note.text(0.02, 0.82, status_lines["checks"], color="#80ed99" if checks_pass else "#ff8c69", fontsize=10.2, fontweight="bold")
    note.text(0.02, 0.53, f"{status_lines['repeatability']}   |   MAX NON-FOOT: {max(item['max_non_foot_force_bw'] for item in reports):.3f} BW   |   PEAK: {max(item['peak_force_n'] for item in reports):.2f} N", color="#d8e7f5", fontsize=10.6)
    note.text(0.02, 0.26, f"RAW FILTER HASH: {first['raw_filter_paths_sha256'][:16]}…   LOGICAL FILTER HASH: {first['logical_filter_paths_sha256'][:16]}…", color="#d8e7f5", fontsize=10.5)
    outcome_artist = note.text(0.02, 0.16, status_lines["outcome"], color="#ffd166", fontsize=10.2, fontweight="bold")
    claim_artist = note.text(0.02, -0.05, "CLAIM LIMIT: NOT A LOCOMOTION, TRAINING, QUALIFICATION, OR PHYSICS-GROUND-TRUTH CLAIM", color="#ffd166", fontsize=9.6, fontweight="bold")
    validate_text_bounds(figure, (outcome_artist, claim_artist), f"{sequence} footer")
    figure.savefig(destination, facecolor=figure.get_facecolor())
    plt.close(figure)


def validate_media(path: Path, kind: str) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"missing {kind}: {path}")
    header = path.read_bytes()[:12]
    valid = {"png": header.startswith(b"\x89PNG\r\n\x1a\n"), "gif": header.startswith(b"GIF8"), "mp4": header[4:8] == b"ftyp"}
    require(valid[kind], f"invalid {kind} magic")
    if kind in {"png", "gif"}:
        require(path.stat().st_size < MAX_PUBLIC_BYTES, f"{kind} exceeds 10 MiB")


def ffprobe_metadata(path: Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    completed = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path)], check=True, capture_output=True, text=True)
    value = json.loads(completed.stdout)
    stream = value["streams"][0]
    return {"codec": stream["codec_name"], "width": stream["width"], "height": stream["height"], "fps": stream["r_frame_rate"], "frames": int(stream["nb_frames"]), "duration_seconds": float(value["format"]["duration"])}


def _install_exclusive(source: Path, destination: Path) -> None:
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


def publish_transaction(pairs: Iterable[tuple[Path, Path]], validate: Callable[[], None]) -> None:
    pairs = tuple(pairs)
    for source, destination in pairs:
        require(source.is_file(), f"staged input missing: {source}")
        require(not destination.exists(), f"refusing to overwrite output: {destination}")
    published: list[Path] = []
    try:
        for source, destination in pairs:
            _install_exclusive(source, destination); published.append(destination)
        validate()
    except BaseException:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise


def artifact(path: Path, published: Path, *, local: bool = False) -> dict[str, Any]:
    tracked_at_build = False
    if not local:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", repo_path(published)],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        tracked_at_build = tracked.returncode == 0
    return {
        "path": str(published.resolve()) if local else repo_path(published),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "tracked_in_git_at_build": tracked_at_build,
        "intended_for_git": not local,
        "git_policy": "local_only" if local else "git_public_after_review",
    }


def build(phase: str, input_paths: Sequence[Path], outputs: Mapping[str, Path], *, ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    phase_paths(phase)
    resolved_outputs = {
        "video": direct_child(outputs["video"], LOCAL_VIDEO_DIR, ".mp4", "video", exists=False),
        "gif": direct_child(outputs["gif"], PUBLIC_MEDIA_DIR, ".gif", "GIF", exists=False),
        "png": direct_child(outputs["png"], PUBLIC_MEDIA_DIR, ".png", "PNG", exists=False),
        "summary": direct_child(outputs["summary"], RUNS_DIR, ".json", "summary", exists=False),
        "sidecar": direct_child(outputs["sidecar"], RUNS_DIR, ".json", "sidecar", exists=False),
    }
    require(len(set(resolved_outputs.values())) == 5, "output paths must be distinct")
    for path in resolved_outputs.values():
        require(not path.exists(), f"refusing to overwrite output: {path}")
    data = validate_inputs(phase, input_paths)
    phase_labels = labels_for_phase(phase)
    with tempfile.TemporaryDirectory(prefix=f"g009-rev20-{phase}-media-") as directory:
        temp = Path(directory)
        frames: list[Path] = []
        for index in range(FRAME_COUNT):
            frame = temp / f"frame_{index:03d}.png"; render_frame(data, (index + 1) / FRAME_COUNT, frame); frames.append(frame)
        from PIL import Image  # pyright: ignore[reportMissingImports]
        staged_png, staged_gif, staged_video = temp / "public.png", temp / "public.gif", temp / "local.mp4"
        with Image.open(frames[-1]) as image:
            image.convert("RGB").save(staged_png, optimize=True)
        gif_frames = [Image.open(frame).convert("P", palette=Image.Palette.ADAPTIVE, colors=96) for frame in frames]
        try:
            gif_frames[0].save(staged_gif, save_all=True, append_images=gif_frames[1:], duration=FRAME_DURATION_MS, loop=0, optimize=True)
        finally:
            for frame in gif_frames: frame.close()
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(1000 / FRAME_DURATION_MS), "-i", str(temp / "frame_%03d.png"), "-vf", f"fps={VIDEO_FPS},format=yuv420p", "-c:v", "libx264", "-movflags", "+faststart", "-t", str(VIDEO_DURATION_SECONDS), str(staged_video)], check=True)
        for path, kind in ((staged_png, "png"), (staged_gif, "gif"), (staged_video, "mp4")): validate_media(path, kind)
        video_metadata = ffprobe_metadata(staged_video)
        require(video_metadata == {"codec": "h264", "width": WIDTH, "height": HEIGHT, "fps": "30/1", "frames": 168, "duration_seconds": VIDEO_DURATION_SECONDS}, "encoded MP4 metadata mismatch")
        public = {
            "gif": {**artifact(staged_gif, resolved_outputs["gif"]), "width": WIDTH, "height": HEIGHT, "frame_count": FRAME_COUNT, "duration_ms": FRAME_COUNT * FRAME_DURATION_MS},
            "png": {**artifact(staged_png, resolved_outputs["png"]), "width": WIDTH, "height": HEIGHT, "representative_frame": FRAME_COUNT},
        }
        local_video = {**artifact(staged_video, resolved_outputs["video"], local=True), **video_metadata}
        summary = {
            "schema_version": "g009.r0.rev20.terrain_contact_matrix_visual_summary.v1", "goal_id": "g009", "stage_id": "R0", "stage_number": STAGE_NUMBER, "sequence_number": "13.01" if phase == "cpu-preflight" else "13.02", "revision": "rev20", "evidence_id": EVIDENCE_ID, "phase": phase, "status": "diagnostic_complete",
            "labels": list(phase_labels), "claim_limits": CLAIM_LIMITS, "input_bindings": data["input_bindings"], "git_commit": data["git_commit"], "source_bundle_sha256": data["source_bundle_sha256"], "preflight_synthesis_source_bundle_sha256": data["preflight_synthesis_source_bundle_sha256"],
            "telemetry": {"reports": data["reports"], "repeatability": data["repeatability"]}, "decision": {"outcome": data["decision"], "gpu_stage_authorized": data["decision"] == "gpu_stage_authorized"}, "public_artifacts": public, "local_video": local_video,
            "governance": {"diagnostic_only": True, "learned": False, "reward_computed": False, "ppo_updates": 0, "qualification_status": "not_run", "physics_ground_truth_authority": False},
        }
        staged_summary = temp / "summary.json"; staged_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        sidecar = {
            "schema_version": "g009.r0.rev20.terrain_contact_matrix_visual_evidence.v1", "goal_id": "g009", "stage_id": "R0", "stage_number": STAGE_NUMBER, "sequence_number": summary["sequence_number"], "revision": "rev20", "evidence_id": EVIDENCE_ID, "phase": phase, "status": "diagnostic_complete",
            "integrity": {"passed": True, "hash_bound": True, "all_inputs_revalidated": True, "no_overwrite": True}, "labels": list(phase_labels), "claim_limits": CLAIM_LIMITS, "input_bindings": data["input_bindings"],
            "source": {"git_commit": data["git_commit"], "probe_source_bundle_sha256": data["source_bundle_sha256"], "synthesis_source_bundle_sha256": data["preflight_synthesis_source_bundle_sha256"]},
            "artifacts": {"visual_summary": artifact(staged_summary, resolved_outputs["summary"]), "public": public, "local_video": local_video},
            "contract": {"builder": artifact(BUILDER_SOURCE, BUILDER_SOURCE), "validator": {**artifact(VALIDATOR_SOURCE, VALIDATOR_SOURCE), "command": f"%PYTHON% scripts/validate_g009_r0_rev20_terrain_contact_matrix_media.py --phase {phase} --check-only"}},
            "governance": summary["governance"],
        }
        staged_sidecar = temp / "sidecar.json"; staged_sidecar.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        input_hashes = {REPO_ROOT / item["path"]: item["sha256"] for item in data["input_bindings"]}
        def validate_published() -> None:
            require(all(file_sha256(path) == sha for path, sha in input_hashes.items()), "input changed during media build")
            for path, kind in ((resolved_outputs["video"], "mp4"), (resolved_outputs["gif"], "gif"), (resolved_outputs["png"], "png")): validate_media(path, kind)
        publish_transaction(((staged_video, resolved_outputs["video"]), (staged_gif, resolved_outputs["gif"]), (staged_png, resolved_outputs["png"]), (staged_summary, resolved_outputs["summary"]), (staged_sidecar, resolved_outputs["sidecar"])), validate_published)
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("cpu-preflight", "final"))
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--video", type=Path); parser.add_argument("--gif", type=Path); parser.add_argument("--png", type=Path); parser.add_argument("--summary", type=Path); parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main() -> int:
    args = build_parser().parse_args(); defaults = phase_paths(args.phase)
    inputs = args.inputs or (list(CPU_REPORTS) + [CPU_PREFLIGHT] if args.phase == "cpu-preflight" else list(FINAL_REPORTS) + [CPU_PREFLIGHT, FINAL_SYNTHESIS])
    outputs = {key: getattr(args, key) or value for key, value in defaults.items()}
    value = build(args.phase, inputs, outputs, ffmpeg=args.ffmpeg)
    print(json.dumps({"status": "pass", "phase": args.phase, "sidecar": outputs["sidecar"].as_posix(), "input_count": len(value["input_bindings"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
