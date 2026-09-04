from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from isaac_walk_g009.media_contract import (  # noqa: E402
    C0_EXECUTION_LOG_PATH,
    C0_REQUIRED_EVIDENCE,
    C0_VALIDATOR_JSON_PATH,
    GIF_COMPRESSION_ORDER,
    MAX_PUBLIC_GIF_FRAME_DURATION_MS,
    MIN_PUBLIC_GIF_FPS,
    SOURCE_VIDEO_FPS,
    STAGE_REGISTRY,
    TARGET_PUBLIC_GIF_FPS,
    count_g008_local_video_evidence,
    file_sha256,
    validate_c0_evidence,
    validate_contract,
    validate_repository_media_rules,
    validate_sidecar,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the import-light G009 media contract.")
    parser.add_argument("--sidecar", type=Path, help="Optional G009 media sidecar JSON to validate.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate sidecar metadata without opening referenced files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Optional validator JSON output (canonical: {C0_VALIDATOR_JSON_PATH}).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help=f"Optional execution log output (canonical: {C0_EXECUTION_LOG_PATH}).",
    )
    parser.add_argument(
        "--run-regressions",
        action="store_true",
        help="Run the existing G008 media pytest set and the repository validator.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print JSON without creating the required C0 receipt files.",
    )
    args = parser.parse_args()
    if args.check_only:
        if args.output is not None or args.log is not None:
            parser.error("--check-only cannot be combined with --output or --log")
    elif args.output is None or args.log is None:
        parser.error("C0 receipt mode requires both --output and --log; use --check-only otherwise")
    return args


def _decode_output(value: bytes) -> str:
    for encoding in ("utf-8", "cp949"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")


def _safe_tail(value: str) -> str:
    return "<non-UTF8 output; use sha256 receipt>" if "\ufffd" in value else value[-2000:]


def _run_regressions() -> dict[str, Any]:
    commands = (
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_g008_visual_evidence.py",
            "tests/test_g008_stage_capture.py",
            "tests/test_g008_stage_visual_evidence.py",
            "tests/test_g008_policy_comparison_visual_evidence.py",
            "tests/test_g008_irregular_road_reports.py",
            "tests/test_g008_road_curriculum_reports.py",
        ],
        ["pwsh", "-NoProfile", "-File", "scripts/validate_repository.ps1"],
    )
    executions = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        stdout = _decode_output(completed.stdout)
        stderr = _decode_output(completed.stderr)
        display_command = ["%PYTHON%", *command[1:]] if command[0] == sys.executable else command
        executions.append(
            {
                "command": display_command,
                "exit_code": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
                "stdout_tail": _safe_tail(stdout),
                "stderr_tail": _safe_tail(stderr),
            }
        )
    return {
        "status": "pass" if all(item["exit_code"] == 0 for item in executions) else "fail",
        "executions": executions,
    }


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    errors = validate_contract()
    errors.extend(validate_repository_media_rules(REPO_ROOT))
    checked = [
        "registry",
        "portable_paths",
        "public_media_size_limit",
        "smooth_gif_temporal_contract",
        "c0_governance",
        "repository_goal_path_rule",
        "g008_path_regression",
    ]
    errors.extend(validate_c0_evidence(C0_REQUIRED_EVIDENCE))
    regression = _run_regressions() if args.run_regressions else {"status": "not_run", "executions": []}
    if regression["status"] == "fail":
        errors.append("G008 media or repository regression failed")
    if args.sidecar is not None:
        checked.append("sidecar")
        if not args.sidecar.is_file():
            errors.append(f"sidecar not found: {args.sidecar}")
        else:
            try:
                sidecar = json.loads(args.sidecar.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"sidecar read failed: {exc}")
            else:
                if not isinstance(sidecar, dict):
                    errors.append("sidecar root must be an object")
                else:
                    errors.extend(
                        validate_sidecar(
                            sidecar,
                            REPO_ROOT,
                            check_files=not args.metadata_only,
                        )
                    )
    return {
        "schema_version": 1,
        "goal_id": "g009",
        "contract_id": "g009_c0_media_contract",
        "mode": "check_only" if args.check_only else "receipt",
        "status": "pass" if not errors else "fail",
        "stage_count": len(STAGE_REGISTRY),
        "stage_ids": list(STAGE_REGISTRY),
        "canonical_outputs": {
            "validator_json": C0_VALIDATOR_JSON_PATH,
            "execution_log": C0_EXECUTION_LOG_PATH,
        },
        "rule_diff": {
            "before": "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g008",
            "after": "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\<goal_id>",
            "g008_compatibility_preserved": True,
        },
        "gif_encoding": {
            "source_video_fps": SOURCE_VIDEO_FPS,
            "target_gif_fps": TARGET_PUBLIC_GIF_FPS,
            "minimum_gif_fps": MIN_PUBLIC_GIF_FPS,
            "maximum_frame_duration_ms": MAX_PUBLIC_GIF_FRAME_DURATION_MS,
            "compression_policy_order": list(GIF_COMPRESSION_ORDER),
        },
        "g008_regression": {
            "path_contract_status": "pass" if not validate_repository_media_rules(REPO_ROOT) else "fail",
            "local_video_references_checked": count_g008_local_video_evidence(REPO_ROOT),
            "execution_status": regression["status"],
            "executions": regression["executions"],
        },
        "source_bindings": {
            "agents_sha256": file_sha256(REPO_ROOT / "AGENTS.md"),
            "contract_sha256": file_sha256(REPO_ROOT / "src" / "isaac_walk_g009" / "media_contract.py"),
            "validator_sha256": file_sha256(Path(__file__)),
        },
        "checked": checked,
        "errors": errors,
    }


def main() -> int:
    args = _parse_args()
    result = build_result(args)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_lines = [
            f"contract_id={result['contract_id']}",
            f"status={result['status']}",
            f"stage_count={result['stage_count']}",
            f"g008_path_contract={result['g008_regression']['path_contract_status']}",
            f"g008_regression_execution={result['g008_regression']['execution_status']}",
            *(
                f"regression_exit={execution['exit_code']} command={' '.join(execution['command'])}"
                for execution in result["g008_regression"]["executions"]
            ),
            *(f"error={error}" for error in result["errors"]),
        ]
        if args.output is not None:
            log_lines.append(f"validator_json_sha256={hashlib.sha256(payload.encode('utf-8')).hexdigest()}")
        args.log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
