from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = ModuleType("isaac_walk_g006")
PACKAGE.__path__ = [str(ROOT / "src" / "isaac_walk_g006")]
sys.modules.setdefault("isaac_walk_g006", PACKAGE)

from isaac_walk_g006.evaluation.protocol import build_push_trials, compute_evaluation_source_bundle  # noqa: E402
from isaac_walk_g006.sweep.push_strength import (  # noqa: E402
    REPORT_EXPERIMENTAL_USE,
    REPORT_GOAL,
    build_grid_manifest,
    canonical_sha256,
    needs_phase_a2,
    per_magnitude_rates,
    protocol_diff_keys,
    select_comparison_grid,
    validate_grid,
)


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = load_script("g006s1_cli_tested", "g006s1_push_strength.py")
WRAPPER = load_script("g006s1_wrapper_tested", "evaluate_g006s1_push_strength.py")
CONTRACT = json.loads((ROOT / "configs" / "g006s1_push_strength_sweep.json").read_text(encoding="utf-8"))
BASE = json.loads((ROOT / "configs" / "g006_rough_push.json").read_text(encoding="utf-8"))
LADDER = CONTRACT["calibration"]["magnitude_ladder_mps"]


def test_contract_binds_the_frozen_g006_protocol_and_evaluator():
    assert canonical_sha256(BASE["evaluation_protocol"]) == CONTRACT["base"]["protocol_sha256"]
    queue = json.loads((ROOT / "reports" / "runs" / "g006_queue_state.json").read_text(encoding="utf-8"))
    assert queue["evaluation_source_bundle_sha256"] == CONTRACT["base"]["evaluation_source_bundle_sha256"]
    assert compute_evaluation_source_bundle(ROOT)["sha256"] == CONTRACT["base"]["evaluation_source_bundle_sha256"]


@pytest.mark.parametrize(
    ("rates", "expected"),
    [
        ({2.0: 0.9, 2.5: 0.8, 3.0: 0.6}, [2.0, 2.5, 3.0]),
        ({2.0: 0.99, 2.5: 0.94, 3.0: 0.7}, [2.0, 2.5, 3.0]),
        ({2.0: 0.99, 2.5: 0.98, 3.0: 0.95}, [2.5, 3.0, 3.5]),
        ({2.0: 0.99, 2.5: 0.99, 3.0: 0.99, 3.5: 0.97, 4.0: 0.96, 5.0: 0.5}, [3.5, 4.0, 5.0]),
        ({2.0: 0.99, 2.5: 0.99, 3.0: 0.99, 3.5: 0.97, 4.0: 0.9, 5.0: 0.5}, [3.5, 4.0, 5.0]),
    ],
)
def test_selection_rule_brackets_the_first_magnitude_at_or_below_threshold(rates, expected):
    decision = select_comparison_grid(rates, LADDER, 0.95)
    assert decision["verdict"] == "grid_selected"
    assert decision["grid_mps"] == expected
    assert rates[decision["knee_magnitude_mps"]] <= 0.95


def test_selection_rule_requests_more_calibration_or_stops():
    assert select_comparison_grid({2.0: 1.0, 2.5: 0.99, 3.0: 0.96}, LADDER, 0.95) == {
        "verdict": "need_more_calibration",
        "next_start_index": 3,
    }
    full = {value: 0.99 for value in LADDER}
    assert select_comparison_grid(full, LADDER, 0.95) == {"verdict": "no_knee_within_ladder"}
    with pytest.raises(ValueError):
        select_comparison_grid({2.5: 0.5}, LADDER, 0.95)
    assert needs_phase_a2({2.0: 0.99, 2.5: 0.97, 3.0: 0.951}, 0.95)
    assert not needs_phase_a2({2.0: 0.99, 2.5: 0.97, 3.0: 0.95}, 0.95)


def test_grid_validation_rejects_values_outside_ladder():
    assert validate_grid([2.0, 2.5, 3.0], LADDER) == [2.0, 2.5, 3.0]
    for bad in ([2.0, 2.5], [2.5, 2.0, 3.0], [2.0, 2.25, 3.0], [1.5, 2.0, 2.5]):
        with pytest.raises(ValueError):
            validate_grid(bad, LADDER)


def test_grid_manifest_changes_only_push_magnitudes():
    derived = build_grid_manifest(BASE, [3.0, 3.5, 4.0], sweep_sha256="0" * 64)
    assert protocol_diff_keys(BASE["evaluation_protocol"], derived["evaluation_protocol"]) == ["push_magnitudes_mps"]
    assert derived["derived_from"]["protocol_sha256"] == CONTRACT["base"]["protocol_sha256"]
    assert BASE["evaluation_protocol"]["push_magnitudes_mps"] == [0.5, 1.0, 1.5]
    protocol = derived["evaluation_protocol"]
    trials = build_push_trials(protocol["initial_states"], protocol["commands"], protocol["push_directions"], protocol["push_magnitudes_mps"])
    assert len(trials) == 1080
    assert {trial["push_magnitude_mps"] for trial in trials} == {3.0, 3.5, 4.0}


def _synthetic_report(phase: str, variant: str, seed: int, grid: list[float], recover) -> dict:
    protocol = build_grid_manifest(BASE, grid, sweep_sha256="0" * 64)["evaluation_protocol"]
    trials = build_push_trials(protocol["initial_states"], protocol["commands"], protocol["push_directions"], protocol["push_magnitudes_mps"])
    reports = []
    for trial in trials:
        recovered = bool(recover(trial))
        reports.append({
            "stratum_id": trial["stratum_id"],
            "paired_trial_key": trial["pair_id"],
            "push_magnitude_mps": trial["push_magnitude_mps"],
            "eligible": True,
            "recovered": recovered,
            "protocol_blocked": False,
            "tracking_error_sq_mean": 0.03,
            "torque_l2_mean": 200.0,
            "absolute_mechanical_power_mean": 35.0,
        })
    return {
        "status": "complete",
        "goal": REPORT_GOAL,
        "experimental_use": REPORT_EXPERIMENTAL_USE,
        "variant": variant,
        "training_seed": seed,
        "checkpoint": {"sha256": "c" * 64},
        "g006s1": {
            "phase": phase,
            "evaluation_source_bundle_matches_g006": True,
            "grid_manifest": {"push_magnitudes_mps": grid},
        },
        "trials": reports,
    }


def test_per_magnitude_rates_count_360_trials_each():
    report = _synthetic_report("A1", "baseline", 42, [2.0, 2.5, 3.0], lambda trial: trial["push_magnitude_mps"] < 3.0)
    rates = per_magnitude_rates(report)
    assert [counts["trials"] for counts in rates.values()] == [360, 360, 360]
    assert rates[3.0]["recovery_rate"] == 0.0 and rates[2.0]["recovery_rate"] == 1.0


def test_end_to_end_select_and_summarize(tmp_path, monkeypatch):
    monkeypatch.setattr(CLI, "RUNS", tmp_path)

    def write(phase, variant, seed, grid, recover):
        path = CLI.report_path(phase, variant, seed)
        path.write_text(json.dumps(_synthetic_report(phase, variant, seed, grid, recover)), encoding="utf-8")

    # 2.0 → 100%, 2.5 → 90% (knee), 3.0 → 50%
    def baseline(trial):
        magnitude = trial["push_magnitude_mps"]
        index = trial["initial_state_id"]
        return magnitude <= 2.0 or (magnitude == 2.5 and index != 0) or (magnitude == 3.0 and index < 5)

    write("A1", "baseline", 42, [2.0, 2.5, 3.0], baseline)
    decision = CLI.decide(CONTRACT)
    assert decision["verdict"] == "grid_selected"
    assert decision["grid_mps"] == [2.0, 2.5, 3.0]
    assert decision["knee_magnitude_mps"] == 2.5

    grid = decision["grid_mps"]
    for seed in (42, 43, 44):
        write("B", "baseline", seed, grid, baseline)
        write("B", "push_curriculum", seed, grid, baseline)
    summary = CLI.summarize(grid)
    assert summary["ceiling_exit"]["confirmed"] is True
    assert summary["ceiling_exit"]["pooled_baseline_rate_at_knee"] == pytest.approx(0.9)
    assert summary["paired_delta_push_curriculum_minus_baseline"]["primary_all_seeds"]["ci95"] == [0.0, 0.0]
    assert summary["paired_delta_push_curriculum_minus_baseline"]["direction_rule_result"] == "no_detectable_difference"

    with pytest.raises(CLI.G006_SUMMARY.ValidationError, match="differs from the pre-registered selection"):
        CLI.summarize([2.5, 3.0, 3.5])


def test_wrapper_preflight_rejects_changed_protocol_and_wrong_checkpoint(tmp_path):
    contract_path = ROOT / "configs" / "g006s1_push_strength_sweep.json"
    contract_sha = WRAPPER.EVALUATOR.file_sha256(contract_path)
    manifest = build_grid_manifest(BASE, [2.0, 2.5, 3.0], sweep_sha256=contract_sha)
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps(manifest), encoding="utf-8")
    fake_checkpoint = tmp_path / "model.pt"
    fake_checkpoint.write_bytes(b"not a checkpoint")
    # All other gates pass on the current tree; only the checkpoint hash must fail.
    with pytest.raises(RuntimeError, match="checkpoint sha256 differs"):
        WRAPPER.preflight(contract_path, grid_path, fake_checkpoint, "baseline", 42)

    manifest["evaluation_protocol"]["eval_seed"] = 1
    grid_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="only push_magnitudes_mps"):
        WRAPPER.preflight(contract_path, grid_path, fake_checkpoint, "baseline", 42)
