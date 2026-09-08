from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_g009_r0_rev31_reset_reachability.py"


def _load():
    spec = importlib.util.spec_from_file_location("rev31_reset_reachability_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load()


def _args(tmp_path: Path, **changes):
    values = {
        "task": probe.DEFAULT_TASK,
        "seed": 42,
        "num_envs": 8,
        "rollout_steps": 150,
        "calf_reset": -2.37,
        "output": ROOT / "reports" / "runs" / "rev31-test.json",
        "device": "cuda:0",
        "headless": True,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_contract_locks_budget_assignment_and_candidate_choices(tmp_path: Path) -> None:
    probe.validate_args(_args(tmp_path))
    assignment = probe.build_assignment()
    assert [row["pose"] for row in assignment] == list(probe.POSE_NAMES) * 2
    assert [row["action_mode"] for row in assignment] == ["zero_normalized"] * 4 + ["reset_pose_hold"] * 4
    assert probe.ACTION_SCALES == (0.70, 0.65, 0.60)
    assert probe.CALF_RESET_CHOICES == (-2.37, -2.28)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"seed": 1042}, "budget is fixed"),
        ({"num_envs": 4}, "budget is fixed"),
        ({"rollout_steps": 149}, "budget is fixed"),
        ({"calf_reset": -2.30}, "calf reset"),
        ({"device": "cpu"}, "headless cuda:0"),
        ({"headless": False}, "headless cuda:0"),
    ],
)
def test_validate_args_rejects_noncanonical_runtime(tmp_path: Path, change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        probe.validate_args(_args(tmp_path, **change))


def test_parser_exposes_only_two_calf_values() -> None:
    action = next(item for item in probe.core_parser()._actions if item.dest == "calf_reset")
    assert tuple(action.choices) == (-2.37, -2.28)
    with pytest.raises(SystemExit):
        probe.core_parser().parse_args(["--output", "x.json", "--calf-reset", "-2.3"])


def test_diagnostic_fail_is_reportable_not_operational_failure() -> None:
    checks = probe.diagnostic_checks(
        runtime_scale=0.60,
        source_stable=True,
        finite_all=False,
        max_nonfoot_force_bw=20.0,
        reset_selection_reported=True,
    )
    assert checks["all_recorded_runtime_values_finite"] is False
    assert checks["max_nonfoot_contact_force_within_15_bodyweights"] is False
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"diagnostic_result": "PASS" if diagnostic_passed else "FAIL"' in source
    assert '"action_scale_or_calf_reset_causal_proof": False' in source
    assert '"checkpoint": None' in source


def test_output_is_direct_child_and_no_overwrite(tmp_path: Path) -> None:
    outside = tmp_path / "report.json"
    with pytest.raises(ValueError, match="direct child"):
        probe.canonical_output(outside)
    output = ROOT / "reports" / "runs" / "rev31-unit-no-overwrite.json"
    output.unlink(missing_ok=True)
    probe.write_json_no_overwrite(output, {"status": "first"})
    try:
        with pytest.raises(FileExistsError, match="overwrite"):
            probe.write_json_no_overwrite(output, {"status": "second"})
    finally:
        output.unlink(missing_ok=True)


def test_import_is_cpu_light_and_does_not_import_isaac_app() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    prefix = source.split("def parse_args", 1)[0]
    assert "from isaaclab.app import AppLauncher" not in prefix
    assert "OnPolicyRunner" not in source
    assert "runner.load" not in source
