from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "configs" / "g009_r0_rev28_entropy_smoke.json"
HARNESS = ROOT / "scripts" / "run_training.ps1"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = _load(
    "validate_g009_r0_rev28_entropy_smoke",
    "scripts/validate_g009_r0_rev28_entropy_smoke.py",
)
summary = _load(
    "summarize_g009_r0_rev28_entropy_smoke",
    "scripts/summarize_g009_r0_rev28_entropy_smoke.py",
)


def _runner() -> SimpleNamespace:
    return SimpleNamespace(
        algorithm=SimpleNamespace(
            entropy_coef=0.0,
            num_learning_epochs=5,
            num_mini_batches=4,
        ),
        policy=SimpleNamespace(init_noise_std=0.5),
        num_steps_per_env=24,
        max_iterations=300,
        save_interval=50,
    )


def test_preregistration_locks_single_variable_budget_and_claim_order() -> None:
    preregistration = validation.load_preregistration()
    readback = validation.validate_semantics(preregistration)

    assert readback["entropy_coef"] == 0.0
    assert preregistration["single_experimental_variable"] == {
        "name": "ppo_entropy_coefficient",
        "rejected_rev26_value": 0.01,
        "candidate_value": 0.0,
    }
    assert preregistration["training"]["transitions"] == 1024 * 24 * 50 == 1_228_800
    assert preregistration["training"]["optimizer_mini_batch_updates"] == 50 * 5 * 4 == 1000
    assert preregistration["training"]["pose_distribution"] == {
        "prone": 1.0,
        "supine": 0.0,
        "left_side": 0.0,
        "right_side": 0.0,
    }
    assert preregistration["execution_order"] == {
        "prelaunch_validator_required": True,
        "smoke_must_pass_before_full_300_iteration_training": True,
        "held_out_seed_1042_forbidden_until_full_300_training_safety_zero": True,
    }


def test_preregistration_rejects_entropy_or_frozen_contract_mutation() -> None:
    preregistration = validation.load_preregistration()
    preregistration["single_experimental_variable"]["candidate_value"] = 0.001
    with pytest.raises(ValueError, match="single experimental variable"):
        validation.validate_semantics(preregistration)

    preregistration = validation.load_preregistration()
    preregistration["frozen_contract"]["action_scale"] = 0.69
    with pytest.raises(ValueError, match="frozen contract"):
        validation.validate_semantics(preregistration)


def test_standalone_validator_does_not_import_agent_cfg() -> None:
    source = (ROOT / "scripts" / "validate_g009_r0_rev28_entropy_smoke.py").read_text(
        encoding="utf-8"
    )
    assert "isaac_walk_g009.agent_cfg" not in source
    assert validation.validate_semantics(validation.load_preregistration())["entropy_coef"] == 0.0


def test_canonical_manifest_matches_rev28_contract() -> None:
    binding = validation.validate_canonical_manifest()
    manifest = json.loads((ROOT / "configs" / "g009_r0.json").read_text(encoding="utf-8"))

    assert binding["sha256"] == validation.file_sha256(ROOT / "configs" / "g009_r0.json")
    assert manifest["contract"]["ppo"]["entropy_coefficient"] == 0.0
    assert manifest["contract"]["ppo"]["entropy_experiment_evidence"]["revision"] == "rev28"


def _zero_series() -> dict:
    return {
        "sample_count": 50,
        "latest": 0.0,
        "minimum": 0.0,
        "maximum": 0.0,
        "mean": 0.0,
        "nonzero_sample_count": 0,
    }


def _git_fixture_run(command, **_kwargs):
    if "show" in command:
        relative = command[-1].split(":", 1)[1]
        return SimpleNamespace(returncode=0, stdout=(ROOT / relative).read_bytes())
    return SimpleNamespace(returncode=0, stdout=b"")


def _raw_report(checkpoint: Path) -> dict:
    preregistration = validation.load_preregistration()
    files = {
        path: summary.file_sha256(ROOT / path)
        for path in preregistration["source_binding_paths"]
    }
    bundle = "\n".join(f"{path}:{files[path]}" for path in preregistration["source_binding_paths"])
    bundle_sha256 = __import__("hashlib").sha256(bundle.encode()).hexdigest()
    blob_files = dict(files)
    blob_bundle = "\n".join(
        f"{path}:{blob_files[path]}" for path in preregistration["source_binding_paths"]
    )
    blob_bundle_sha256 = __import__("hashlib").sha256(blob_bundle.encode()).hexdigest()
    run_directory = checkpoint.parent
    agent_yaml = run_directory / "params" / "agent.yaml"
    agent_yaml.parent.mkdir(parents=True, exist_ok=True)
    agent_yaml.write_text(
        "seed: 42\ndevice: cuda:0\nnum_steps_per_env: 24\nmax_iterations: 50\n"
        "policy:\n  init_noise_std: 0.5\nalgorithm:\n  entropy_coef: 0.0\n"
        "  num_learning_epochs: 5\n  num_mini_batches: 4\n",
        encoding="utf-8",
    )
    commit = "c" * 40
    snapshot = {"repository_commit": commit, "sha256": bundle_sha256, "files": files}
    return {
        "task": "Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0",
        "num_envs": 1024,
        "max_iterations": 50,
        "seed": 42,
        "headless": True,
        "resume": {"enabled": False, "load_run": None, "checkpoint": None},
        "effective_hydra_overrides": [],
        "qualification_mode": {
            "enabled": False,
            "preflight_passed": None,
            "policy_qualification_status": "not_run",
        },
        "entropy_smoke_mode": {
            "enabled": True,
            "preflight_passed": True,
            "runtime_algorithm_entropy_coef": 0.0,
            "held_out_evaluation_status": "forbidden_until_full_300_training_safety_zero",
            "full_300_iteration_training_status": "forbidden_until_smoke_accepted",
            "policy_qualification_status": "not_run",
        },
        "repository": {"commit": commit, "dirty": False},
        "source_bundle": {
            "sha256": bundle_sha256,
            "hash_domain": "executed_worktree_bytes",
            "matches_repository_commit": True,
            "files": files,
            "commit_blob_sha256": {"bundle": blob_bundle_sha256, "files": blob_files},
            "prelaunch": {**snapshot, "hash_domain": "executed_worktree_bytes", "matches_validated_snapshot": True},
            "postrun": {**snapshot, "hash_domain": "executed_worktree_bytes", "stable": True},
        },
        "entropy_smoke_contract": {
            "path": "configs/g009_r0_rev28_entropy_smoke.json",
            "sha256": summary.file_sha256(PREREGISTRATION),
            "source_binding_path_manifest_sha256": preregistration[
                "source_binding_path_manifest_sha256"
            ],
            "prelaunch_validation": {
                "schema_version": "g009.r0.rev28.entropy_smoke_prelaunch_validation.v1",
                "status": "pass",
                "evidence_id": "G009-5-E021",
                "preregistration": {
                    "path": "configs/g009_r0_rev28_entropy_smoke.json",
                    "sha256": summary.file_sha256(PREREGISTRATION),
                },
                "canonical_static_readback": {
                    "entropy_coef": 0.0,
                    "init_noise_std": 0.5,
                    "num_steps_per_env": 24,
                    "num_learning_epochs": 5,
                    "num_mini_batches": 4,
                },
                "source_state": {
                    "repository_commit": commit,
                    "repository_clean": True,
                    "source_paths_logically_equal_to_head": True,
                    "executed_worktree_sha256": {
                        "bundle": bundle_sha256,
                        "files": files,
                    },
                    "commit_blob_sha256": {
                        "bundle": blob_bundle_sha256,
                        "files": blob_files,
                    },
                }
            },
        },
        "tensorboard": {
            "series_summary": {
                "Episode_Termination/hard_joint_limit": _zero_series(),
                "Episode_Termination/numeric_invalid": _zero_series(),
                "Policy/mean_noise_std": {
                    "sample_count": 50,
                    "latest": 0.51,
                    "minimum": 0.5,
                    "maximum": 0.51,
                    "mean": 0.505,
                    "nonzero_sample_count": 50,
                },
            }
        },
        "training_safety_gate": {"required": True, "passed": True},
        "gpu": {
            "protected_run_safety": {
                "required": True,
                "passed": True,
                "temperature_threshold_c": 90.0,
                "sustained_sample_count": 3,
                "fatal_matches": [],
                "descendants_exited": True,
            }
        },
        "artifacts": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": summary.file_sha256(checkpoint),
            "tensorboard_directory": str(run_directory),
            "agent_yaml": str(agent_yaml),
            "agent_yaml_sha256": summary.file_sha256(agent_yaml),
        },
        "runtime_agent_config": {
            "source": "official train.py params/agent.yaml",
            "readback": {
                "entropy_coef": 0.0,
                "init_noise_std": 0.5,
                "num_steps_per_env": 24,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "max_iterations": 50,
                "device": "cuda:0",
            },
            "passed": True,
        },
        "success_checks": {
            "process_exit_zero": True,
            "no_traceback_or_error": True,
            "requested_iteration_reached": True,
            "log_directory_exists": True,
            "tensorboard_exists": True,
            "checkpoint_exists": True,
            "gpu_measurement_complete": True,
            "gpu_recovered_to_baseline": True,
            "qualification_training_safety_zero": None,
            "qualification_gpu_safety": None,
            "entropy_smoke_training_safety_zero": True,
            "entropy_smoke_gpu_safety": True,
            "entropy_smoke_source_snapshot_stable": True,
            "entropy_smoke_agent_yaml_readback": True,
            "requested_training_safety_gate_zero": None,
        },
        "run_health_passed": True,
        "passed": True,
        "qualification_passed": None,
    }


def test_summary_accepts_only_exact_zero_safety_and_records_std_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "run" / "model_49.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture-checkpoint")
    monkeypatch.setattr(summary, "checkpoint_std_vector", lambda _path: ([0.5] * 12, 49))
    monkeypatch.setattr(summary.subprocess, "run", _git_fixture_run)

    evidence = summary.validate_report(
        _raw_report(checkpoint), validation.load_preregistration(), checkpoint_path=checkpoint
    )

    assert evidence["training_safety"]["hard_joint_limit"]["sample_count"] == 50
    assert evidence["exploration_noise"]["checkpoint_std_vector"] == [0.5] * 12
    assert "non-worsening smoke acceptance gate" in evidence["exploration_noise"]["comparison_semantics"]


def test_summary_rejects_mutated_supplied_preregistration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "run" / "model_49.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture-checkpoint")
    monkeypatch.setattr(summary, "checkpoint_std_vector", lambda _path: ([0.5] * 12, 49))
    monkeypatch.setattr(summary.subprocess, "run", _git_fixture_run)
    mutated = copy.deepcopy(validation.load_preregistration())
    mutated["acceptance_gate"]["tensorboard_exact_sample_count"] = 1
    with pytest.raises(ValueError, match="supplied preregistration is not canonical"):
        summary.validate_report(_raw_report(checkpoint), mutated)


def test_core_autocrlf_hash_domains_do_not_compare_blob_to_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "run" / "model_49.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture-checkpoint")
    report = _raw_report(checkpoint)
    paths = validation.load_preregistration()["source_binding_paths"]
    blob_bytes = {path: (ROOT / path).read_bytes() for path in paths}
    blob_bytes[paths[0]] = blob_bytes[paths[0]].replace(b"\n", b"\r\n")
    blob_hashes = {
        path: __import__("hashlib").sha256(blob_bytes[path]).hexdigest() for path in paths
    }
    payload = "\n".join(f"{path}:{blob_hashes[path]}" for path in paths)
    blob_domain = {
        "bundle": __import__("hashlib").sha256(payload.encode()).hexdigest(),
        "files": blob_hashes,
    }
    report["source_bundle"]["commit_blob_sha256"] = blob_domain
    report["entropy_smoke_contract"]["prelaunch_validation"]["source_state"][
        "commit_blob_sha256"
    ] = blob_domain

    def git_run(command, **_kwargs):
        if "show" in command:
            return SimpleNamespace(
                returncode=0, stdout=blob_bytes[command[-1].split(":", 1)[1]]
            )
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(summary.subprocess, "run", git_run)
    monkeypatch.setattr(summary, "checkpoint_std_vector", lambda _path: ([0.5] * 12, 49))
    evidence = summary.validate_report(report, validation.load_preregistration())
    assert evidence["checkpoint"]["iteration"] == 49


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["tensorboard"]["series_summary"][
                "Episode_Termination/hard_joint_limit"
            ].update(maximum=0.01, nonzero_sample_count=1),
            "hard_joint_limit maximum",
        ),
        (
            lambda report: report["tensorboard"]["series_summary"][
                "Episode_Termination/numeric_invalid"
            ].update(sample_count=49),
            "numeric_invalid sample_count",
        ),
        (
            lambda report: report["gpu"]["protected_run_safety"].update(passed=False),
            "protected GPU safety failed",
        ),
        (
            lambda report: report["entropy_smoke_mode"].update(
                held_out_evaluation_status="not_run"
            ),
            "held-out gate opened",
        ),
        (
            lambda report: report["source_bundle"].update(
                files={key: "a" * 64 for key in report["source_bundle"]["files"]}
            ),
            "executed worktree file hashes mismatch",
        ),
        (
            lambda report: report["tensorboard"]["series_summary"][
                "Policy/mean_noise_std"
            ].update(latest=0.5513023735),
            "mean_noise_std worsened",
        ),
        (
            lambda report: report["source_bundle"]["postrun"].update(
                sha256="b" * 64
            ),
            "postrun bundle mismatch",
        ),
        (
            lambda report: report["success_checks"].update(unregistered_check=True),
            "harness success checks mismatch",
        ),
    ],
)
def test_summary_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    checkpoint = tmp_path / "run" / "model_49.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture-checkpoint")
    monkeypatch.setattr(summary, "checkpoint_std_vector", lambda _path: ([0.5] * 12, 49))
    monkeypatch.setattr(summary.subprocess, "run", _git_fixture_run)
    report = _raw_report(checkpoint)
    mutation(report)

    with pytest.raises(ValueError, match=message):
        summary.validate_report(report, validation.load_preregistration())


def test_summary_output_is_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    summary.write_json_no_overwrite(output, {"status": "pass"})
    with pytest.raises(ValueError, match="overwrite"):
        summary.write_json_no_overwrite(output, {"status": "replacement"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "pass"}


def test_summary_exclusive_create_has_one_race_winner(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def writer(label: str) -> None:
        barrier.wait()
        try:
            summary.write_json_no_overwrite(output, {"writer": label})
            outcomes.append("created")
        except FileExistsError:
            outcomes.append("exists")

    threads = [threading.Thread(target=writer, args=(label,)) for label in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["created", "exists"]


def test_harness_has_separate_fail_closed_entropy_smoke_mode() -> None:
    source = HARNESS.read_text(encoding="utf-8-sig")
    assert "[switch]$EntropySmoke" in source
    assert "$protectedGpuRun = [bool]($Qualification -or $EntropySmoke)" in source
    assert "scratch_required = [bool]($RequireZeroTrainingSafetyTerminations -or $EntropySmoke)" in source
    assert "entropy_smoke_training_safety_zero" in source
    assert "entropy_smoke_gpu_safety" in source
    assert "protected_run_safety" in source
    assert "model_49.pt" in source
    assert "sustained_gpu_temperature_at_or_above_" in source
    assert "Stop-VerifiedProcessTree" in source
    assert "Stop-PostExitDescendants" in source
    assert "$sourceSnapshotStable" in source
    assert "params\\agent.yaml" in source


def test_post_exit_snapshot_cleans_child_created_after_last_sample(tmp_path: Path) -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    fixture = tmp_path / "late-child-fixture.ps1"
    pid_file = tmp_path / "late-child.pid"
    fixture.write_text(
        r'''
param([string]$Harness,[string]$PidFile)
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($Harness,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'parser failure'}
$names=@('Get-DescendantProcessIds','Test-ProcessIdsExited','Stop-PostExitDescendants')
$definitions=$ast.FindAll({param($node)$node -is [Management.Automation.Language.FunctionDefinitionAst] -and $names -contains $node.Name},$true)
if($definitions.Count -ne 3){throw 'function extraction failure'}
Invoke-Expression (($definitions|ForEach-Object{$_.Extent.Text}) -join "`n")
$shell=(Get-Command pwsh).Source
$childCommand="`$child=Start-Process -FilePath '$shell' -ArgumentList '-NoProfile','-Command','Start-Sleep -Seconds 120' -WindowStyle Hidden -PassThru;[IO.File]::WriteAllText('$PidFile',[string]`$child.Id)"
$encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($childCommand))
$parent=Start-Process -FilePath $shell -ArgumentList '-NoProfile','-EncodedCommand',$encoded -WindowStyle Hidden -PassThru
$parent.WaitForExit()
$deadline=(Get-Date).AddSeconds(10)
while(-not(Test-Path -LiteralPath $PidFile) -and (Get-Date)-lt $deadline){Start-Sleep -Milliseconds 50}
if(-not(Test-Path -LiteralPath $PidFile)){throw 'child pid missing'}
$childId=[int](Get-Content -LiteralPath $PidFile -Raw)
$result=Stop-PostExitDescendants -RootProcessId $parent.Id
if($result.descendant_process_ids -notcontains $childId){Stop-Process -Id $childId -Force -ErrorAction SilentlyContinue;throw 'late child not observed'}
if(-not $result.all_processes_exited){throw 'late child not cleaned'}
''',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [pwsh, "-NoProfile", "-File", str(fixture), str(HARNESS), str(pid_file)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_harness_rejects_noncanonical_smoke_budget_before_launch() -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None, "pwsh is required by the Windows training harness"
    completed = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-File",
            str(HARNESS),
            "-Task",
            "Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0",
            "-NumEnvs",
            "512",
            "-MaxIterations",
            "50",
            "-Seed",
            "42",
            "-RunName",
            "rev28_guard_test",
            "-EntropySmoke",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode != 0
    assert "num_envs=1024" in completed.stdout + completed.stderr
