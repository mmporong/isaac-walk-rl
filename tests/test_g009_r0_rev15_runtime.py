from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rev15_runtime", ROOT / "scripts/summarize_g009_r0_rev15_runtime.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_real_rev15_runtime_is_strictly_rejected_before_training() -> None:
    result = MODULE.summarize(MODULE.DEFAULT_CPU, MODULE.DEFAULT_GPU)
    assert result["status"] == "rejected_before_gate01"
    assert result["evidence_synthesis_valid"] is True
    assert result["candidate_runtime_calibration_passed"] is False
    assert result["learned"] is False
    assert result["ppo_training"] is False
    assert result["ppo_training_status"] == "not_run"
    assert result["qualification_status"] == "not_run"
    assert result["qualification_passed"] is None
    assert result["lineage"] == {
        "source_commit": MODULE.REV15_SOURCE_COMMIT,
        "source_bundle_sha256": MODULE.REV15_SOURCE_BUNDLE,
        "contract_sha256": MODULE.REV15_CONTRACT,
    }
    assert result["repeatability"]["unique_execution_ids"] == 6
    assert result["physics_readback"] == {
        "articulations_per_run": 8,
        "links_per_articulation": 19,
        "rigid_bodies_per_run": 152,
        "solver_position_iterations": 16,
        "solver_velocity_iterations": 0,
        "max_depenetration_velocity_m_s": 1.0,
        "all_paths_and_apis_valid": True,
    }
    assert result["device_results"]["cpu"]["runtime_passed"] is True
    assert result["device_results"]["cpu"]["progression_gate_passed"] is True
    assert result["device_results"]["gpu"]["runtime_passed"] is False
    assert result["device_results"]["gpu"]["progression_gate_passed"] is False
    assert result["device_results"]["gpu"]["runtime_passed_runs"] == 0
    assert result["device_results"]["gpu"]["progression_gate_passed_runs"] == 0
    assert result["device_results"]["gpu"]["failed_checks"] == [
        "nonfoot_peak_force_bounded"
    ]
    assert result["divergence"]["gpu_force_excess_bodyweights"] == pytest.approx(
        1.7882747650146484
    )
    assert result["divergence"]["cpu_separation_margin_mm"] == pytest.approx(
        0.6469136476516726
    )
    assert result["completed_stages"] == {
        "cpu_runtime_3x": True,
        "gpu_runtime_3x": True,
        "strict_rejection_synthesis": True,
    }
    assert result["blocked_stages"] == {
        "gate01": True,
        "gate10": True,
        "ppo_training": True,
    }
    assert all(
        len(run["sha256"]) == 64
        for device in ("cpu", "gpu")
        for run in result["repeatability"][device]["inputs"]
    )


def test_write_is_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    MODULE.write_summary(MODULE.DEFAULT_CPU, MODULE.DEFAULT_GPU, output)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        MODULE.write_summary(MODULE.DEFAULT_CPU, MODULE.DEFAULT_GPU, output)


def mutate_report(tmp_path: Path, source: Path, mutate) -> Path:
    value = json.loads(source.read_text(encoding="utf-8"))
    mutate(value)
    destination = MODULE.RUNS_DIR / f"pytest_rev15_runtime_{tmp_path.name}_{source.name}"
    value["execution"]["output_path_repo_relative"] = f"reports/runs/{destination.name}"
    destination.write_text(json.dumps(value), encoding="utf-8")
    return destination


@pytest.fixture(autouse=True)
def cleanup_mutations():
    before = set(MODULE.RUNS_DIR.glob("pytest_rev15_runtime_*"))
    yield
    for path in set(MODULE.RUNS_DIR.glob("pytest_rev15_runtime_*")) - before:
        path.unlink()


@pytest.mark.parametrize(
    "source_index,device,mutation,message",
    [
        (
            0,
            "cpu",
            lambda r: r["source_bundle"].__setitem__("source_bundle_sha256", "0" * 64),
            "source bundle digest mismatch",
        ),
        (
            0,
            "cpu",
            lambda r: r["progression_gate"].__setitem__("passed", False),
            "CPU progression gate",
        ),
        (
            0,
            "cpu",
            lambda r: r["required_crosschecks"]["cpu_contact_separation"].__setitem__(
                "passed", False
            ),
            "CPU separation authority",
        ),
        (
            0,
            "cpu",
            lambda r: r["physics_readback"]["articulation_solver_iterations"][
                "articulations"
            ][0].__setitem__("solver_position_iteration_count", 8),
            "position solver iteration",
        ),
        (
            0,
            "cpu",
            lambda r: r["physics_readback"]["rigid_body_max_depenetration_velocity"][
                "articulations"
            ][0]["links"][0].__setitem__("max_depenetration_velocity_m_s", 0.75),
            "max depenetration velocity",
        ),
        (
            0,
            "cpu",
            lambda r: r["pose_mode_metrics"][0]["termination_counts"].__setitem__(
                "numeric_invalid", 1
            ),
            "safety termination",
        ),
        (
            0,
            "gpu",
            lambda r: r["checks"].__setitem__("nonfoot_peak_force_bounded", True),
            "must fail only",
        ),
        (
            0,
            "gpu",
            lambda r: r["checks"].__setitem__("tail_angular_speed_settled", False),
            "must fail only",
        ),
        (
            0,
            "cpu",
            lambda r: r["checks"].pop("joint_speed_within_runtime_limit"),
            "all CPU runtime checks",
        ),
        (
            0,
            "gpu",
            lambda r: r["checks"].pop("joint_speed_within_runtime_limit"),
            "missing, reordered, or unexpected",
        ),
        (
            0,
            "gpu",
            lambda r: r["required_crosschecks"]["cpu_contact_separation"].__setitem__(
                "data_available", True
            ),
            "must not claim CPU separation",
        ),
        (
            0,
            "gpu",
            lambda r: r["pose_mode_metrics"][7].__setitem__(
                "max_nonfoot_force_bodyweights", "16.78827476501465"
            ),
            "finite JSON number",
        ),
        (
            0,
            "gpu",
            lambda r: r["pose_mode_metrics"][7].__setitem__(
                "max_nonfoot_force_bodyweights", 14.0
            ),
            "GPU peak force changed",
        ),
        (
            0,
            "gpu",
            lambda r: r["runtime_contract"].__setitem__("passed", True),
            "GPU runtime contract failure evidence",
        ),
        (
            0,
            "gpu",
            lambda r: r["progression_gate"].__setitem__("passed", True),
            "GPU progression gate",
        ),
        (
            0,
            "gpu",
            lambda r: r.__setitem__("device", "cuda_fake"),
            "device mismatch",
        ),
        (
            0,
            "gpu",
            lambda r: r["progression_gate"].__setitem__("device", "cuda_other"),
            "GPU progression gate",
        ),
        (
            0,
            "gpu",
            lambda r: r.__setitem__("passed", True),
            "GPU top-level passed",
        ),
        (
            0,
            "cpu",
            lambda r: r.__setitem__("contract_sha256", "0" * 64),
            "contract hash mismatch",
        ),
        (
            0,
            "cpu",
            lambda r: r["execution"].__setitem__("no_overwrite", False),
            "input must be no-overwrite",
        ),
        (
            0,
            "cpu",
            lambda r: r["execution"].__setitem__("started_at_utc", ""),
            "execution timestamp",
        ),
        (
            0,
            "cpu",
            lambda r: r["execution"].__setitem__("started_at_utc", "not-a-timeZ"),
            "execution timestamp must be valid",
        ),
        (
            0,
            "cpu",
            lambda r: r["execution"].__setitem__(
                "started_at_utc", "2026-08-28T10:00:00+09:00"
            ),
            "Z suffix",
        ),
        (
            0,
            "cpu",
            lambda r: r["qualification"].__setitem__("status", "passed"),
            "qualification must remain",
        ),
        (
            0,
            "cpu",
            lambda r: r["pose_mode_metrics"][0].__setitem__("env_index", 7),
            "pose/action environment mapping",
        ),
        (
            0,
            "cpu",
            lambda r: r["physics_readback"]["rigid_body_max_depenetration_velocity"][
                "articulations"
            ][0]["links"][0].__setitem__("physx_rigid_body_api", False),
            "link API/path readback",
        ),
    ],
)
def test_rev15_mutations_fail_closed(
    tmp_path: Path, source_index: int, device: str, mutation, message: str
) -> None:
    source = (
        MODULE.DEFAULT_CPU[source_index]
        if device == "cpu"
        else MODULE.DEFAULT_GPU[source_index]
    )
    bad = mutate_report(tmp_path, source, mutation)
    cpu = list(MODULE.DEFAULT_CPU)
    gpu = list(MODULE.DEFAULT_GPU)
    if device == "cpu":
        cpu[source_index] = bad
    else:
        gpu[source_index] = bad
    with pytest.raises(ValueError, match=message):
        MODULE.summarize(cpu, gpu)


def test_duplicate_execution_id_fails_closed(tmp_path: Path) -> None:
    source_id = json.loads(MODULE.DEFAULT_CPU[0].read_text(encoding="utf-8"))[
        "execution"
    ]["execution_id"]
    bad = mutate_report(
        tmp_path,
        MODULE.DEFAULT_CPU[1],
        lambda r: r["execution"].__setitem__("execution_id", source_id),
    )
    with pytest.raises(ValueError, match="six unique UUID4"):
        MODULE.summarize(
            (MODULE.DEFAULT_CPU[0], bad, MODULE.DEFAULT_CPU[2]), MODULE.DEFAULT_GPU
        )


def test_duplicate_execution_timestamp_fails_closed(tmp_path: Path) -> None:
    source_time = json.loads(MODULE.DEFAULT_CPU[0].read_text(encoding="utf-8"))[
        "execution"
    ]["started_at_utc"]
    bad = mutate_report(
        tmp_path,
        MODULE.DEFAULT_CPU[1],
        lambda r: r["execution"].__setitem__("started_at_utc", source_time),
    )
    with pytest.raises(ValueError, match="six unique RFC3339 UTC"):
        MODULE.summarize(
            (MODULE.DEFAULT_CPU[0], bad, MODULE.DEFAULT_CPU[2]), MODULE.DEFAULT_GPU
        )
