#!/usr/bin/env python3
"""Record rev16 Arm-B CUDA right-side diagnostic footage to a local-only MP4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = REPO_ROOT / "scripts/record_g009_r0_rev15_gpu_right_side.py"
SOURCE_COMMIT = "9ac874f48a1403e0ed838beb5e75938db5873d1c"
SOURCE_ENV_INDEX = 7
POSE_ID = "right_side"
ACTION_MODE = "reset_pose_hold"
OUTPUT_STEM = "g009_5_r0_diag_rev16_08_b_gpu_right_side_force_repro"
DEFAULT_RUNTIME_REPORT = (
    REPO_ROOT / "reports/runs/g009_r0_rev16_arm_b_gpu_rep01_retry01_s42.json"
)
DEFAULT_CAPTURE_REPORT = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_capture_s42.json"
DEFAULT_OUTPUT_DIR = Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
DEFAULT_VIDEO = DEFAULT_OUTPUT_DIR / f"{OUTPUT_STEM}_s42.mp4"
EXPECTED_RUNTIME_SHA256 = (
    "9cb36810c81f892001557fd6ac3772ae3b8370e3ca1e3daab01e79b17080886d"
)
LABELS = (
    "DIAGNOSTIC",
    "REJECTED",
    "NO PPO",
    "NOT QUALIFIED",
    "RIGHT_SIDE",
    "RESET_POSE_HOLD",
)
CAPTURE_BINDING_PATHS = (
    "scripts/record_g009_r0_rev15_gpu_right_side.py",
    "scripts/record_g009_r0_rev16_b_gpu_right_side.py",
    "reports/runs/g009_r0_rev16_arm_b_gpu_rep01_retry01_s42.json",
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


def canonical_text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return "%USERPROFILE%\\" + str(resolved.relative_to(Path.home().resolve()))


def load_legacy() -> Any:
    spec = importlib.util.spec_from_file_location("g009_rev16_camera_base", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("rev15 recorder unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_paths = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "clean": not dirty_paths, "dirty_paths": dirty_paths}


def contains_expected_source_commit(commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, commit],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    )


def current_capture_source_bundle(
    report: dict[str, Any], source_state: dict[str, Any]
) -> dict[str, Any]:
    runtime_bundle = report.get("source_bundle", {})
    expected_runtime_files = runtime_bundle.get("source_binding_files", {})
    require(
        isinstance(expected_runtime_files, dict) and bool(expected_runtime_files),
        "runtime source binding files are missing",
    )
    relative_paths = sorted(set(expected_runtime_files).union(CAPTURE_BINDING_PATHS))
    raw_files: dict[str, str] = {}
    canonical_files: dict[str, str] = {}
    validated_files: dict[str, str] = {}
    missing_files: list[str] = []
    for relative in relative_paths:
        path = REPO_ROOT / relative
        if path.is_file():
            raw_files[relative] = file_sha256(path)
            canonical_files[relative] = canonical_text_sha256(path)
        else:
            missing_files.append(relative)
    require(not missing_files, f"capture source files missing: {missing_files}")
    mismatches: list[str] = []
    for relative in relative_paths:
        expected = expected_runtime_files.get(relative)
        if expected is None:
            validated_files[relative] = canonical_files[relative]
        elif expected in {raw_files[relative], canonical_files[relative]}:
            validated_files[relative] = expected
        else:
            mismatches.append(relative)
    require(not mismatches, f"runtime-bound source files drifted: {mismatches}")
    require(
        EXPECTED_RUNTIME_SHA256
        in {
            raw_files[CAPTURE_BINDING_PATHS[-1]],
            canonical_files[CAPTURE_BINDING_PATHS[-1]],
        },
        "capture runtime report hash drifted",
    )
    payload = {
        "schema_version": 1,
        "git_commit": source_state["commit"],
        "runtime_source_commit": SOURCE_COMMIT,
        "runtime_source_bundle_sha256": runtime_bundle.get("source_bundle_sha256"),
        "hash_validation": "runtime hash must equal raw checkout or CRLF-to-LF canonical SHA-256",
        "source_binding_files": validated_files,
        "checkout_raw_sha256": raw_files,
        "checkout_canonical_lf_sha256": canonical_files,
    }
    payload["source_bundle_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    payload["all_files_present"] = True
    payload["clean"] = source_state["clean"]
    payload["dirty_paths"] = source_state["dirty_paths"]
    return payload


def validate_current_capture_binding() -> dict[str, Any]:
    source_state = git_source_state()
    require(source_state["clean"] is True, "current capture source binding is dirty")
    require(
        contains_expected_source_commit(source_state["commit"]),
        "capture commit is not a descendant of the runtime source commit",
    )
    runtime = validate_runtime_report(DEFAULT_RUNTIME_REPORT)
    report = runtime["report"]
    bundle = current_capture_source_bundle(report, source_state)
    contract_sha256 = report.get("contract_sha256")
    require(
        isinstance(contract_sha256, str) and len(contract_sha256) == 64,
        "runtime contract SHA-256 is invalid",
    )
    return {"source_bundle": bundle, "contract_sha256": contract_sha256}


def validate_runtime_report(path: Path) -> dict[str, Any]:
    require(path.resolve() == DEFAULT_RUNTIME_REPORT.resolve(), "runtime path is fixed")
    require(path.is_file(), "runtime report is missing")
    require(file_sha256(path) == EXPECTED_RUNTIME_SHA256, "runtime hash mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    governance = report.get("governance", {})
    arm = report.get("contract", {}).get("arm", {})
    controlled = report.get("controlled_cell", {})
    require(
        report.get("schema_version") == "g009.r0.rev16.backend_divergence.v1"
        and report.get("status") == "complete"
        and report.get("diagnostic_only") is True,
        "rev16 diagnostic report contract mismatch",
    )
    require(
        report.get("source_bundle", {}).get("git_commit") == SOURCE_COMMIT
        and report.get("headless") is True
        and report.get("device") == "cuda:0",
        "source/headless/device binding mismatch",
    )
    require(
        arm.get("id") == "B"
        and arm.get("articulation_solver_position_iteration_count") == 16
        and arm.get("articulation_solver_velocity_iteration_count") == 0
        and arm.get("max_depenetration_velocity_m_s") == 1.0,
        "Arm B 16/0/1.0 binding mismatch",
    )
    require(
        controlled.get("source_env_index") == SOURCE_ENV_INDEX
        and controlled.get("pose_id") == POSE_ID
        and controlled.get("action_mode") == ACTION_MODE,
        "controlled cell mismatch",
    )
    require(
        governance.get("ppo") == {"allowed": False, "status": "not_run"}
        and governance.get("gate01") == {"allowed": False, "status": "forbidden"}
        and governance.get("gate10") == {"allowed": False, "status": "forbidden"}
        and governance.get("qualification")
        == {"eligible": False, "status": "not_run", "passed": None},
        "diagnostic governance mismatch",
    )
    metrics = report.get("historical_runtime_summary", {}).get("pose_metrics", [])
    matches = [
        row
        for row in metrics
        if row.get("env_index") == SOURCE_ENV_INDEX
        and row.get("pose_id") == POSE_ID
        and row.get("action_mode") == ACTION_MODE
    ]
    require(len(matches) == 1, "right-side metric cell missing")
    cell = matches[0]
    force = float(cell["max_nonfoot_force_bodyweights"])
    require(force > 15.0 and math.isfinite(force), "Arm B GPU force must exceed 15 BW")
    require(cell.get("max_nonfoot_force_body_name") == "base", "peak body mismatch")
    return {"report": report, "cell": cell, "sha256": file_sha256(path)}


def blocking_cell_payload(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_index": SOURCE_ENV_INDEX,
        "pose_id": POSE_ID,
        "action_mode": ACTION_MODE,
        "max_nonfoot_force_bodyweights": cell["max_nonfoot_force_bodyweights"],
        "max_nonfoot_force_threshold_bodyweights": 15.0,
        "max_nonfoot_force_body_name": cell["max_nonfoot_force_body_name"],
        "max_nonfoot_force_physics_step": cell["max_nonfoot_force_physics_step"],
        "result": "reproduced_over_limit",
    }


def validate_output_paths(output_dir: Path, video: Path, report: Path) -> None:
    require(output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve(), "output dir is fixed")
    require(video.resolve() == DEFAULT_VIDEO.resolve(), "local MP4 path is fixed")
    require(
        report.resolve() == DEFAULT_CAPTURE_REPORT.resolve(), "sidecar path is fixed"
    )
    require(
        not video.exists() and not report.exists(), "refusing to overwrite evidence"
    )
    require(
        not list(output_dir.glob(f"{OUTPUT_STEM.replace('_', '-')}*.mp4")),
        "raw capture already exists",
    )


def normalize_result(result: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    raw = result.get("local_raw_video", {}).get("path")
    if isinstance(raw, str) and raw.startswith("%USERPROFILE%\\"):
        raw_path = Path.home() / raw.removeprefix("%USERPROFILE%\\")
        if raw_path.is_file():
            raw_path.unlink()
    report = runtime["report"]
    return {
        "schema_version": "g009.r0.rev16.camera_capture.v1",
        "capture_id": result["capture_id"],
        "captured_at_utc": result["captured_at_utc"],
        "goal_id": "g009",
        "stage_id": "R0",
        "stage_number": "08",
        "revision": "rev16",
        "status": "rejected",
        "diagnostic_only": True,
        "camera_footage": True,
        "telemetry_animation": False,
        "headless": True,
        "offscreen": True,
        "labels": list(LABELS),
        "task": result["task"],
        "seed": 42,
        "device": "cuda:0",
        "source_env_index": SOURCE_ENV_INDEX,
        "pose_id": POSE_ID,
        "action_mode": ACTION_MODE,
        "physics": {
            "arm": "B",
            "solver": "16/0",
            "max_depenetration_velocity_m_s": 1.0,
        },
        "source": {
            "path": portable_path(DEFAULT_RUNTIME_REPORT),
            "sha256": runtime["sha256"],
            "commit": SOURCE_COMMIT,
            "bundle_sha256": report["source_bundle"]["source_bundle_sha256"],
        },
        "capture_source": result["source"]["current_capture_binding"],
        "governance": report["governance"],
        "blocking_cell": blocking_cell_payload(runtime["cell"]),
        "camera": result["camera"],
        "timing": result["timing"],
        "evidence_scope": "off-screen Isaac Sim camera playback; force values are bound to the input JSON, not inferred from pixels",
        "local_video": result["local_video"],
    }


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    runtime = validate_runtime_report(args.runtime_report)
    validate_output_paths(args.output_dir, args.video, args.capture_report)
    legacy = load_legacy()
    legacy.OUTPUT_STEM = OUTPUT_STEM
    legacy.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    legacy.DEFAULT_VIDEO = DEFAULT_VIDEO
    legacy.DEFAULT_CAPTURE_REPORT = DEFAULT_CAPTURE_REPORT
    legacy.EXPECTED_COMMIT = SOURCE_COMMIT
    legacy.EXPECTED_SOURCE_BUNDLE_SHA256 = runtime["report"]["source_bundle"][
        "source_bundle_sha256"
    ]
    legacy.EXPECTED_CONTRACT_SHA256 = runtime["report"]["contract_sha256"]
    legacy.EXPECTED_NONFOOT_PEAK_BW = runtime["cell"]["max_nonfoot_force_bodyweights"]
    legacy.EXPECTED_NONFOOT_STEP = runtime["cell"]["max_nonfoot_force_physics_step"]
    legacy.git_source_state = git_source_state
    legacy.contains_expected_source_commit = contains_expected_source_commit
    legacy.validate_current_capture_binding = validate_current_capture_binding
    legacy.validate_runtime_report = lambda _path: runtime
    legacy.validate_output_paths = validate_output_paths
    legacy.blocking_cell_payload = blocking_cell_payload
    original_write = legacy.write_json_new

    def write_normalized(path: Path, value: dict[str, Any]) -> None:
        original_write(path, normalize_result(value, runtime))

    legacy.write_json_new = write_normalized
    return normalize_result(legacy.capture(args), runtime)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-report", type=Path, default=DEFAULT_RUNTIME_REPORT)
    parser.add_argument("--capture-report", type=Path, default=DEFAULT_CAPTURE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.enable_cameras = True
    require(
        bool(args.headless) and args.device == "cuda:0",
        "--headless --device cuda:0 required",
    )
    if sys.platform == "win32" and not args.kit_args:
        args.kit_args = legacy_windows_args()
    return args


def legacy_windows_args() -> str:
    return "--/app/vulkan=false --/app/window/hideUi=true --/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_runtime_report(args.runtime_report)
    validate_output_paths(args.output_dir, args.video, args.capture_report)
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    launcher = AppLauncher(args)
    try:
        result = run_capture(args)
    finally:
        launcher.app.close()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
