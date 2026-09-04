from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "run_training.ps1"
PWSH = shutil.which("pwsh")


def test_source_binding_sort_matches_python_ordinal_order() -> None:
    if PWSH is None:
        pytest.skip("pwsh is unavailable")
    absolute_alias = str(ROOT / "SCRIPTS" / "RUN_TRAINING.PS1")
    paths = [
        "configs/g009_r0_rev24_gpu_throughput.json",
        "configs/g009_r0.json",
        "scripts/run_training.ps1",
        absolute_alias,
    ]
    literals = ",".join("'" + path.replace("'", "''") + "'" for path in paths)
    root_literal = str(ROOT).replace("'", "''")
    command = (
        f"$repoRoot='{root_literal}';$inputPaths=@({literals});"
        "$repoBoundary=[IO.Path]::GetFullPath($repoRoot).TrimEnd('\\')+'\\';"
        "$tracked=[Collections.Generic.Dictionary[string,string]]::new([StringComparer]::OrdinalIgnoreCase);"
        "git -C $repoRoot ls-files --full-name|ForEach-Object{if(-not $tracked.ContainsKey($_)){$tracked.Add($_,$_)}};"
        "$canonical=[Collections.Generic.Dictionary[string,string]]::new([StringComparer]::OrdinalIgnoreCase);"
        "foreach($path in $inputPaths){"
        "$full=if([IO.Path]::IsPathRooted($path)){[IO.Path]::GetFullPath($path)}else{[IO.Path]::GetFullPath((Join-Path $repoRoot $path))};"
        "if(-not $full.StartsWith($repoBoundary,[StringComparison]::OrdinalIgnoreCase)){throw 'outside repo'};"
        "if(-not(Test-Path -LiteralPath $full -PathType Leaf)){throw 'missing file'};"
        "$relative=[IO.Path]::GetRelativePath($repoRoot,$full).Replace('\\','/');"
        "$key=if($tracked.ContainsKey($relative)){$tracked[$relative]}else{$relative};"
        "if(-not $canonical.ContainsKey($key)){$canonical.Add($key,$full)}};"
        "$sorted=[string[]]@($canonical.Keys);"
        "[Array]::Sort($sorted,[System.StringComparer]::Ordinal);"
        "$sorted|ConvertTo-Json -Compress"
    )

    result = subprocess.run(
        [PWSH, "-NoProfile", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == [
        "configs/g009_r0.json",
        "configs/g009_r0_rev24_gpu_throughput.json",
        "scripts/run_training.ps1",
    ]
    harness = HARNESS.read_text(encoding="utf-8-sig")
    assert "[System.StringComparer]::OrdinalIgnoreCase" in harness
    assert "$trackedPathByCase[$relativeSourcePath]" in harness
    assert "[Array]::Sort($sortedSourceBindingPaths, [System.StringComparer]::Ordinal)" in harness


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
