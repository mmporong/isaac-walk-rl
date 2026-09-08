from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "configs" / "g009_r0_rev30_action_scale_smoke.json"
HARNESS = ROOT / "scripts" / "run_training.ps1"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = _load(
    "validate_g009_r0_rev30_action_scale_smoke_test",
    "scripts/validate_g009_r0_rev30_action_scale_smoke.py",
)
summary = _load(
    "summarize_g009_r0_rev30_action_scale_smoke_test",
    "scripts/summarize_g009_r0_rev30_action_scale_smoke.py",
)


def test_preregistration_locks_rev30_single_variable_and_budget() -> None:
    preregistration = validation.load_preregistration()
    readback = validation.validate_semantics(preregistration)

    assert preregistration["single_experimental_variable"] == {
        "name": "normalized_joint_position_action_scale",
        "rejected_rev29_value": 0.65,
        "candidate_value": 0.60,
    }
    assert readback["action_scale"] == 0.60
    assert readback["agent_yaml"]["entropy_coef"] == 0.0
    assert preregistration["training"]["transitions"] == 1024 * 24 * 50
    assert preregistration["training"]["optimizer_mini_batch_updates"] == 1000
    assert preregistration["claim_limits"]["recovery_success_measured"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["single_experimental_variable"].update(candidate_value=0.61),
        lambda value: value["frozen_contract"].update(ppo_entropy_coefficient=0.01),
        lambda value: value["runtime_readback"]["env_yaml"].update(action_scale=0.65),
    ],
)
def test_preregistration_rejects_mutation(mutation) -> None:
    preregistration = validation.load_preregistration()
    mutation(preregistration)
    with pytest.raises(ValueError):
        validation.validate_semantics(preregistration)


def test_canonical_manifest_and_historical_evidence_are_bound() -> None:
    binding = validation.validate_canonical_manifest()
    manifest = json.loads((ROOT / "configs" / "g009_r0.json").read_text(encoding="utf-8"))
    evidence = validation.validate_historical_evidence(validation.load_preregistration())

    assert binding["sha256"] == validation.file_sha256(ROOT / "configs" / "g009_r0.json")
    assert manifest["contract"]["contract_id"] == "g009_r0_recover_rev30"
    assert manifest["contract"]["action"]["scale"] == 0.60
    assert set(evidence) == {
        "rev27_diagnostic_report",
        "rev29_training_report",
        "rev29_rejection_report",
    }


def test_summary_wrapper_uses_rev30_contract() -> None:
    assert summary.SCHEMA_VERSION == "g009.r0.rev30.action_scale_smoke_summary.v1"
    assert summary.REVISION == "rev30"
    assert summary.DEFAULT_PREREGISTRATION == PREREGISTRATION


def test_harness_rejects_noncanonical_budget_before_launch() -> None:
    completed = subprocess.run(
        [
            "pwsh",
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
            "rev30_guard_test",
            "-ActionScaleSmoke",
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
