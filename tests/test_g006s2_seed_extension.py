from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = ModuleType("isaac_walk_g006")
PACKAGE.__path__ = [str(ROOT / "src" / "isaac_walk_g006")]
sys.modules.setdefault("isaac_walk_g006", PACKAGE)

SPEC = importlib.util.spec_from_file_location("g006s2_tool_tested", ROOT / "scripts" / "g006s2_seed_extension.py")
assert SPEC and SPEC.loader
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)

RUNS = ROOT / "reports" / "runs"
CONTRACT = json.loads((ROOT / "configs" / "g006s2_seed_extension.json").read_text(encoding="utf-8"))


def test_contract_binds_g006_training_and_evaluation_bytes():
    queue = json.loads((RUNS / "g006_queue_state.json").read_text(encoding="utf-8"))
    assert CONTRACT["base"]["training_source_bundle_sha256"] == queue["training_source_bundle_sha256"]
    assert CONTRACT["base"]["evaluation_source_bundle_sha256"] == queue["evaluation_source_bundle_sha256"]
    assert CONTRACT["base"]["training_entrypoint_sha256"] == queue["training_entrypoint_sha256"]
    assert TOOL.training_bundle(queue)["sha256"] == CONTRACT["base"]["training_source_bundle_sha256"]


def test_expected_command_changes_only_seed_run_name_and_explicit_device():
    g006 = json.loads((RUNS / "g006_production_baseline_e4096_i1500_s42.json").read_text(encoding="utf-8"))["command"]
    expected = TOOL.expected_command(g006, 45, "g006s2_production_baseline_e4096_i1500_s45")
    assert len(expected) == len(g006) + 2
    differences = [(a, b) for a, b in zip([str(x) for x in g006], [x for x in expected if x not in ("--device", "cuda:0")]) if a != b]
    assert differences == [("42", "45"), ("g006_production_baseline_e4096_i1500_s42", "g006s2_production_baseline_e4096_i1500_s45")]
    assert expected[expected.index("--seed") + 2 : expected.index("--seed") + 4] == ["--device", "cuda:0"]


def _training_report(variant: str, seed: int) -> dict:
    report = copy.deepcopy(json.loads((RUNS / f"g006_production_{variant}_e4096_i1500_s42.json").read_text(encoding="utf-8")))
    run_name = f"g006s2_production_{variant}_e4096_i1500_s{seed}"
    command = TOOL.expected_command(report["command"], seed, run_name)
    command[1] = "%USERPROFILE%\\worktrees\\isaac-walk-rl-g009-r0\\scripts\\bootstrap_train_g006.py"
    report.update({"run_name": run_name, "seed": seed, "command": command, "passed": True, "effective_hydra_overrides": []})
    return report


def _write_training(tmp_path: Path, variant: str, seed: int, report: dict) -> None:
    (tmp_path / f"g006s2_production_{variant}_e4096_i1500_s{seed}.json").write_text(json.dumps(report), encoding="utf-8")


def test_gate_passes_for_a_g006_equivalent_run(tmp_path):
    _write_training(tmp_path, "push_curriculum", 45, _training_report("push_curriculum", 45))
    result = TOOL.gate("push_curriculum", 45, runs=tmp_path, stdout_text="[INFO][AppLauncher]: Using device: cuda:0")
    assert result["passed"], result["checks"]


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    [
        (lambda r: r.update(passed=False), "harness_passed"),
        (lambda r: r.update(effective_hydra_overrides=["agent.seed=1"]), "no_hydra_overrides"),
        (lambda r: r["command"].extend(["--video"]), "command_matches_g006"),
        (lambda r: r.update(max_iterations=1499), "budget"),
        (lambda r: r["gpu"].update(peak_used_mib=9900), "peak_vram_within_limit"),
        (lambda r: r["training_entrypoint"].update(sha256="0" * 64), "entrypoint_sha256"),
    ],
)
def test_gate_rejects_each_deviation(tmp_path, mutate, failed_check):
    report = _training_report("baseline", 46)
    mutate(report)
    _write_training(tmp_path, "baseline", 46, report)
    result = TOOL.gate("baseline", 46, runs=tmp_path, stdout_text="Using device: cuda:0")
    assert not result["passed"]
    assert result["checks"][failed_check] is False


def test_gate_requires_cuda0_in_stdout(tmp_path):
    _write_training(tmp_path, "baseline", 45, _training_report("baseline", 45))
    assert TOOL.gate("baseline", 45, runs=tmp_path, stdout_text="Using device: cpu")["checks"]["device_cuda0_in_stdout"] is False


def test_five_seed_summary_uses_g006s1_reports_and_new_seeds(tmp_path):
    for variant in ("baseline", "push_curriculum"):
        for seed in (42, 43, 44):
            shutil.copy(RUNS / f"g006s1_B_{variant}_s{seed}_push.json", tmp_path)
        for seed, source_seed in ((45, 43), (46, 44)):
            report = json.loads((RUNS / f"g006s1_B_{variant}_s{source_seed}_push.json").read_text(encoding="utf-8"))
            report.update(goal="G006S2", training_seed=seed)
            (tmp_path / f"g006s2_S2_{variant}_s{seed}_push.json").write_text(json.dumps(report), encoding="utf-8")
    summary = TOOL.summarize(runs=tmp_path)
    assert summary["seeds"] == [42, 43, 44, 45, 46]
    assert summary["recovery"]["baseline"]["trials"] == 5 * 1080
    assert set(summary["recovery"]["baseline"]["per_direction"]) == {"backward", "forward", "left", "right"}
    assert summary["paired_delta_push_curriculum_minus_baseline"]["direction_rule_result"] in {
        "no_detectable_difference", "push_curriculum_higher", "baseline_higher"
    }

    stale = json.loads((tmp_path / "g006s2_S2_baseline_s45_push.json").read_text(encoding="utf-8"))
    stale["goal"] = "G006S1"
    (tmp_path / "g006s2_S2_baseline_s45_push.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(TOOL.G006_SUMMARY.ValidationError, match="not a G006S2 report"):
        TOOL.summarize(runs=tmp_path)
