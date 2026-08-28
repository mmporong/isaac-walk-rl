from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rev14_tradeoff", ROOT / "scripts/summarize_g009_r0_rev14_tradeoff.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_real_rev14_tradeoff_synthesis_is_strictly_rejected() -> None:
    result = MODULE.summarize(
        MODULE.DEFAULT_CPU,
        MODULE.DEFAULT_GPU,
        MODULE.DEFAULT_BASELINE_CPU,
        MODULE.DEFAULT_BASELINE_GPU,
    )
    assert result["status"] == "rejected_before_gate01"
    assert result["learned"] is False
    assert result["qualification_status"] == "not_run"
    assert result["lineage"] == {
        "source_commit": MODULE.REV14_SOURCE_COMMIT,
        "source_bundle_sha256": MODULE.REV14_SOURCE_BUNDLE,
        "contract_sha256": MODULE.REV14_CONTRACT,
    }
    assert result["repeatability"]["unique_execution_ids"] == 6
    assert result["physics_readback"] == {
        "articulations_per_run": 8,
        "links_per_articulation": 19,
        "rigid_bodies_per_run": 152,
        "max_depenetration_velocity_m_s": 0.75,
        "all_paths_and_apis_valid": True,
    }
    assert result["tradeoff"]["separation_overrun_mm"] == pytest.approx(
        0.9901875257492063
    )
    assert result["blocked_stages"] == {
        "gate01": True,
        "gate10": True,
        "ppo_training": True,
    }
    assert result["completed_stages"] == {
        "cpu_runtime_3x": True,
        "gpu_runtime_3x": True,
        "strict_tradeoff_synthesis": True,
    }
    assert all(
        len(run["sha256"]) == 64
        for device in ("cpu", "gpu")
        for run in result["repeatability"][device]["inputs"]
    )


def test_write_is_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    MODULE.write_summary(
        MODULE.DEFAULT_CPU,
        MODULE.DEFAULT_GPU,
        MODULE.DEFAULT_BASELINE_CPU,
        MODULE.DEFAULT_BASELINE_GPU,
        output,
    )
    with pytest.raises(ValueError, match="refusing to overwrite"):
        MODULE.write_summary(
            MODULE.DEFAULT_CPU,
            MODULE.DEFAULT_GPU,
            MODULE.DEFAULT_BASELINE_CPU,
            MODULE.DEFAULT_BASELINE_GPU,
            output,
        )


def mutate_report(tmp_path: Path, source: Path, mutate) -> Path:
    value = json.loads(source.read_text(encoding="utf-8"))
    mutate(value)
    destination = (
        MODULE.RUNS_DIR / f"pytest_rev14_tradeoff_{tmp_path.name}_{source.name}"
    )
    destination.write_text(json.dumps(value), encoding="utf-8")
    value["execution"]["output_path_repo_relative"] = f"reports/runs/{destination.name}"
    destination.write_text(json.dumps(value), encoding="utf-8")
    return destination


@pytest.fixture(autouse=True)
def cleanup_mutations():
    before = set(MODULE.RUNS_DIR.glob("pytest_rev14_tradeoff_*"))
    yield
    for path in set(MODULE.RUNS_DIR.glob("pytest_rev14_tradeoff_*")) - before:
        path.unlink()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda r: r["source_bundle"].__setitem__("source_bundle_sha256", "0" * 64),
            "source bundle digest mismatch",
        ),
        (
            lambda r: r["checks"].__setitem__(next(iter(r["checks"])), False),
            "all runtime checks",
        ),
        (
            lambda r: r["physics_readback"]["rigid_body_max_depenetration_velocity"][
                "articulations"
            ][0]["links"][0].__setitem__("max_depenetration_velocity_m_s", 1.0),
            "must equal 0.75",
        ),
        (
            lambda r: r["physics_readback"]["rigid_body_max_depenetration_velocity"][
                "articulations"
            ][0]["links"][0].__setitem__("physx_rigid_body_api", False),
            "API/path",
        ),
        (lambda r: r["qualification"].__setitem__("status", "passed"), "qualification"),
        (
            lambda r: r["pose_mode_metrics"][0]["termination_counts"].__setitem__(
                "numeric_invalid", 1
            ),
            "safety termination",
        ),
    ],
)
def test_rev14_mutations_fail_closed(tmp_path: Path, mutation, message: str) -> None:
    bad = mutate_report(tmp_path, MODULE.DEFAULT_CPU[0], mutation)
    with pytest.raises(ValueError, match=message):
        MODULE.summarize(
            (bad, *MODULE.DEFAULT_CPU[1:]),
            MODULE.DEFAULT_GPU,
            MODULE.DEFAULT_BASELINE_CPU,
            MODULE.DEFAULT_BASELINE_GPU,
        )


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
            (MODULE.DEFAULT_CPU[0], bad, MODULE.DEFAULT_CPU[2]),
            MODULE.DEFAULT_GPU,
            MODULE.DEFAULT_BASELINE_CPU,
            MODULE.DEFAULT_BASELINE_GPU,
        )
