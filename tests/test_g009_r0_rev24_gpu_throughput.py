from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/summarize_g009_r0_rev24_gpu_throughput.py"
SPEC = importlib.util.spec_from_file_location("rev24_gpu_throughput", PATH)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def report(env_count: int, *, steps: list[float] | None = None, numeric: dict | None = None) -> dict:
    step_values = steps if steps is not None else [1000, 1100, 1200, 1300, 1400]
    start_minute = 0 if env_count == 1024 else 2
    log_path = f"%USERPROFILE%\\IsaacLab\\logs\\rsl_rl\\g009_recover_r0\\run_{env_count}"
    source_files = {
        path: SUMMARY._sha256((SUMMARY.REPO_ROOT / path).read_bytes())
        for path in SUMMARY.REQUIRED_SOURCE_PATHS
    }
    source_payload = "\n".join(f"{path}:{source_files[path]}" for path in sorted(source_files))
    return {
        "schema_version": 1,
        "task": SUMMARY.TASK,
        "num_envs": env_count,
        "max_iterations": 5,
        "seed": 42,
        "headless": True,
        "training_entrypoint": {
            "path": "%USERPROFILE%\\isaac-walk-rl\\scripts\\bootstrap_benchmark_g009.py",
            "sha256": SUMMARY._sha256((SUMMARY.REPO_ROOT / SUMMARY.ENTRYPOINT).read_bytes()),
            "repository_internal": True,
        },
        "resume": {"enabled": False, "load_run": None, "checkpoint": None},
        "effective_hydra_overrides": [],
        "repository": {"commit": "b" * 40, "dirty": False},
        "source_bundle": {
            "sha256": SUMMARY._sha256(source_payload.encode("utf-8")),
            "matches_repository_commit": True,
            "files": source_files,
        },
        "exit_code": 0,
        "started_at": f"2026-09-01T12:{start_minute:02d}:00+09:00",
        "ended_at": f"2026-09-01T12:{start_minute + 1:02d}:00+09:00",
        "wall_time_seconds": 60.0,
        "gpu": {
            "device_count": 1,
            "total_mib": 12000,
            "peak_used_mib": 8000,
            "peak_utilization_gpu_percent": 99,
            "mean_utilization_gpu_percent": 85,
            "peak_temperature_c": 72,
            "peak_power_draw_w": 154.5,
            "measurement_complete": True,
            "recovered_to_baseline": True,
        },
        "performance": {
            "steps_per_second_samples": step_values,
            "mean_steps_per_second": round(sum(step_values) / len(step_values), 2),
            "median_steps_per_second": sorted(step_values)[len(step_values) // 2],
        },
        "training_safety_aggregate": {"numeric_invalid": numeric},
        "success_checks": {
            "process_exit_zero": True,
            "no_traceback_or_error": True,
            "requested_iteration_reached": True,
            "log_directory_exists": True,
            "tensorboard_exists": True,
            "checkpoint_exists": True,
            "gpu_measurement_complete": True,
            "gpu_recovered_to_baseline": True,
        },
        "run_health_passed": True,
        "qualification_mode": {
            "enabled": False,
            "preflight_passed": None,
            "policy_qualification_status": "not_run",
        },
        "training_safety_gate": {"requested": False, "required": False, "passed": None},
        "qualification_passed": None,
        "artifacts": {"tensorboard_directory": log_path},
        "log_directory_resolution": {
            "mode": "single_new_run_name_directory",
            "candidates": [log_path],
            "selected": log_path,
            "preexisting_match_count": 0,
        },
        "passed": True,
    }


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    runs = tmp_path / "reports/runs"
    runs.mkdir(parents=True)
    active = tmp_path / "configs/g009_r0.json"
    active.parent.mkdir(parents=True)
    real_active = json.loads((ROOT / "configs/g009_r0.json").read_text(encoding="utf-8"))
    active.write_text(json.dumps(real_active), encoding="utf-8")
    for relative in SUMMARY.REQUIRED_SOURCE_PATHS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    monkeypatch.setattr(SUMMARY, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(SUMMARY, "RUNS_DIR", runs)
    monkeypatch.setattr(SUMMARY, "ACTIVE_CONFIG_PATH", active)
    monkeypatch.setattr(SUMMARY, "repository_head", lambda: "b" * 40)
    monkeypatch.setattr(SUMMARY, "load_preregistration", lambda: {
        "experiment": {"claim_limit": "throughput only"},
        "source_bindings": {
            "isaaclab": {"version": "v2.1.1", "commit": SUMMARY.ISAACLAB_COMMIT},
            "official_benchmark": {"path": "scripts/benchmarks/benchmark_rsl_rl.py", "sha256": SUMMARY.BENCHMARK_SHA256},
            "repository_entrypoint": SUMMARY.ENTRYPOINT,
            "active_contract": {"path": "configs/g009_r0.json", "contract_sha256": SUMMARY.ACTIVE_CONTRACT_SHA256},
        },
        "protocol": {
            "ordered_environment_counts": [1024, 2048],
            "max_iterations": 5,
            "num_steps_per_env": 24,
            "ppo_num_learning_epochs": 5,
            "ppo_num_mini_batches": 4,
            "headless": True,
            "scratch": True,
            "resume_allowed": False,
        },
    })
    return runs


def materialize(runs: Path, name: str, value: dict) -> Path:
    path = runs / name
    # Deliberately permit non-standard NaN/Infinity tokens so the fail-closed
    # validator is exercised against corrupt producer output.
    path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
    return path


def test_two_passes_select_2048_and_preserve_claim_limits(isolated: Path) -> None:
    first = materialize(isolated, "run1024.json", report(1024))
    second = materialize(isolated, "run2048.json", report(2048))
    value = SUMMARY.synthesize(first, second)
    assert value["status"] == "complete"
    assert value["decision"] == {"passed": True, "stable_max_envs": 2048, "outcome": "throughput_2048_passed"}
    assert value["claim_limits"]["policy_qualification"] == "not_run"
    assert value["claim_limits"]["recovery_success"] == "not_measured"


def test_partial_1024_only_is_supported(isolated: Path) -> None:
    first = materialize(isolated, "run1024.json", report(1024))
    value = SUMMARY.synthesize(first)
    assert value["status"] == "partial"
    assert value["sequence_gate"]["stage_2048_authorized"] is True
    assert value["sequence_gate"]["stage_2048_input_status"] == "missing"
    assert value["decision"] == {"passed": False, "stable_max_envs": 1024, "outcome": "awaiting_2048_input"}


def test_failed_1024_forbids_2048(isolated: Path) -> None:
    bad = report(1024)
    bad["run_health_passed"] = False
    first = materialize(isolated, "run1024.json", bad)
    value = SUMMARY.synthesize(first)
    assert value["decision"]["outcome"] == "throughput_1024_failed"
    assert value["sequence_gate"]["stage_2048_authorized"] is False
    second = materialize(isolated, "run2048.json", report(2048))
    with pytest.raises(ValueError, match="forbidden"):
        SUMMARY.synthesize(first, second)


def test_2048_must_start_after_1024_ended(isolated: Path) -> None:
    first = materialize(isolated, "run1024.json", report(1024))
    second_value = report(2048)
    second_value["started_at"] = "2026-09-01T12:00:30+09:00"
    second = materialize(isolated, "run2048.json", second_value)
    with pytest.raises(ValueError, match="must start after"):
        SUMMARY.synthesize(first, second)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("task", "wrong"), "task mismatch"),
        (lambda value: value.__setitem__("num_envs", 2048), "environment count mismatch"),
        (lambda value: value.__setitem__("seed", 7), "seed mismatch"),
        (lambda value: value.__setitem__("max_iterations", 6), "iteration count mismatch"),
        (lambda value: value.__setitem__("headless", False), "headless mismatch"),
        (lambda value: value["training_entrypoint"].__setitem__("path", "scripts/bootstrap_train_g009.py"), "entrypoint mismatch"),
        (lambda value: value["resume"].__setitem__("enabled", True), "must be scratch"),
        (lambda value: value.__setitem__("effective_hydra_overrides", ["agent.num_steps_per_env=12"]), "PPO rollout/update"),
    ],
)
def test_identity_mutations_fail_closed(isolated: Path, mutation, message: str) -> None:
    value = report(1024)
    mutation(value)
    path = materialize(isolated, "run1024.json", value)
    with pytest.raises(ValueError, match=message):
        SUMMARY.synthesize(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["repository"].__setitem__("dirty", True),
        lambda value: value["repository"].__setitem__("commit", "d" * 40),
        lambda value: value.__setitem__("exit_code", 1),
        lambda value: value["source_bundle"].__setitem__("matches_repository_commit", False),
        lambda value: value["source_bundle"].__setitem__("sha256", "e" * 64),
        lambda value: value["success_checks"].__setitem__("checkpoint_exists", False),
        lambda value: value["gpu"].__setitem__("measurement_complete", False),
        lambda value: value["gpu"].__setitem__("recovered_to_baseline", False),
        lambda value: value["performance"].__setitem__("steps_per_second_samples", [1, 2, 3, 4]),
        lambda value: value["performance"].__setitem__("steps_per_second_samples", [1, 2, 3, 4, float("inf")]),
        lambda value: value["gpu"].__setitem__("total_mib", None),
        lambda value: value["gpu"].__setitem__("device_count", 2),
        lambda value: value["gpu"].__setitem__("peak_used_mib", 11000),
        lambda value: value["gpu"].__setitem__("peak_utilization_gpu_percent", None),
        lambda value: value["gpu"].__setitem__("peak_temperature_c", None),
        lambda value: value["gpu"].__setitem__("peak_power_draw_w", float("nan")),
        lambda value: value["log_directory_resolution"].__setitem__("mode", "ambiguous_new_run_name_directories"),
    ],
)
def test_gate_mutations_cannot_pass(isolated: Path, mutation) -> None:
    value = report(1024)
    mutation(value)
    path = materialize(isolated, "run1024.json", value)
    result = SUMMARY.synthesize(path)
    assert result["rows"][0]["passed"] is False
    assert result["sequence_gate"]["stage_2048_authorized"] is False
    assert result["decision"]["stable_max_envs"] is None


def test_numeric_invalid_absence_is_explicit_nonblocking(isolated: Path) -> None:
    value = report(1024, numeric=None)
    path = materialize(isolated, "run1024.json", value)
    result = SUMMARY.synthesize(path)
    row = result["rows"][0]
    assert row["passed"] is True
    assert row["numeric_invalid"] == {"availability": "unavailable", "blocking": False, "passed": True, "maximum": None}
    assert row["gpu"]["power"]["availability"] == "available"


def test_unavailable_power_blocks_the_throughput_gate(isolated: Path) -> None:
    value = report(1024)
    value["gpu"]["peak_power_draw_w"] = None
    path = materialize(isolated, "run1024.json", value)
    result = SUMMARY.synthesize(path)
    assert result["rows"][0]["gpu"]["power"]["blocking"] is True
    assert result["rows"][0]["passed"] is False


def test_numeric_invalid_nonzero_blocks(isolated: Path) -> None:
    value = report(1024, numeric={"maximum": 1.0})
    path = materialize(isolated, "run1024.json", value)
    result = SUMMARY.synthesize(path)
    assert result["rows"][0]["numeric_invalid"]["blocking"] is True
    assert result["rows"][0]["passed"] is False


def test_active_contract_mutation_fails_closed(isolated: Path) -> None:
    SUMMARY.ACTIVE_CONFIG_PATH.write_text(json.dumps({"contract_sha256": "0" * 64}), encoding="utf-8")
    path = materialize(isolated, "run1024.json", report(1024))
    with pytest.raises(ValueError, match="active contract SHA"):
        SUMMARY.synthesize(path)


def test_active_contract_payload_mutation_fails_closed(isolated: Path) -> None:
    value = json.loads(SUMMARY.ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    value["contract"]["physics"]["articulation_solver_position_iteration_count"] = 16
    SUMMARY.ACTIVE_CONFIG_PATH.write_text(json.dumps(value), encoding="utf-8")
    path = materialize(isolated, "run1024.json", report(1024))
    with pytest.raises(ValueError, match="payload hash"):
        SUMMARY.synthesize(path)
