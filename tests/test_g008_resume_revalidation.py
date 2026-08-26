from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REVALIDATOR = REPO_ROOT / "scripts" / "revalidate_g008_resume_report.ps1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resumed_iteration_range_includes_loaded_checkpoint(tmp_path: Path) -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None, "pwsh is required for the PowerShell revalidator"

    log_root = tmp_path / "logs" / "unitree_go2_rough"
    source_run = log_root / "source_run"
    output_run = log_root / "output_run"
    source_run.mkdir(parents=True)
    output_run.mkdir(parents=True)

    source_checkpoint = source_run / "model_1499.pt"
    output_checkpoint = output_run / "model_1798.pt"
    source_checkpoint.write_bytes(b"source-checkpoint")
    output_checkpoint.write_bytes(b"output-checkpoint")
    (output_run / "events.out.tfevents.test").write_bytes(b"tensorboard")
    stdout = tmp_path / "stdout.log"
    stdout.write_text("Learning iteration 1798/1799\n", encoding="utf-8")

    report_path = REPO_ROOT / "reports" / "runs" / f".g008-resume-test-{uuid.uuid4().hex}.json"
    report = {
        "schema_version": 1,
        "passed": False,
        "max_iterations": 300,
        "last_iteration": 1798,
        "iteration_target": 1799,
        "resume": {
            "enabled": True,
            "load_run": "source_run",
            "checkpoint": "model_1499.pt",
        },
        "artifacts": {
            "raw_stdout": str(stdout),
            "checkpoint": str(output_checkpoint),
            "checkpoint_sha256": _sha256(output_checkpoint),
            "tensorboard_directory": str(output_run),
        },
        "success_checks": {
            "process_exit_zero": True,
            "no_traceback_or_error": True,
            "requested_iteration_reached": False,
            "log_directory_exists": True,
            "tensorboard_exists": True,
            "checkpoint_exists": True,
            "gpu_measurement_complete": True,
            "gpu_recovered_to_baseline": True,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    try:
        result = subprocess.run(
            [pwsh, "-NoProfile", "-File", str(REVALIDATOR), "-ReportPath", str(report_path)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        validated = json.loads(report_path.read_text(encoding="utf-8-sig"))
        assert validated["passed"] is True
        assert validated["last_iteration"] == 1798
        assert validated["iteration_target"] == 1799
        assert validated["resume_revalidation"]["expected_iteration"] == 1798
        assert validated["resume_revalidation"]["checkpoint_sha256_verified"] is True
    finally:
        report_path.unlink(missing_ok=True)
