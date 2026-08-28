from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "run_training.ps1"
PWSH = shutil.which("pwsh")


def _run_qualification(
    *extra: str,
    num_envs: str = "1024",
) -> subprocess.CompletedProcess[str]:
    if PWSH is None:
        pytest.skip("pwsh is unavailable")
    command = [
        PWSH,
        "-NoProfile",
        "-File",
        str(HARNESS),
        "-Task",
        "Isaac-G009-Recover-Flat-Go2-R0-v0",
        "-NumEnvs",
        num_envs,
        "-MaxIterations",
        "300",
        "-Seed",
        "42",
        "-RunName",
        "g009_qualification_guard_test",
        "-Qualification",
        *extra,
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_qualification_rejects_all_hydra_overrides() -> None:
    result = _run_qualification("-HydraOverrides", "agent.algorithm.gamma=0.95")

    assert result.returncode != 0
    assert "Hydra override" in result.stdout + result.stderr


def test_g009_qualification_rejects_noncanonical_budget() -> None:
    result = _run_qualification(num_envs="512")

    assert result.returncode != 0
    assert "num_envs=1024" in result.stdout + result.stderr
