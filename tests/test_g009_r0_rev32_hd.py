import sys
import json
import hashlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from record_g009_r0_rev32_hd import sample_steps
import record_g009_r0_rev32_hd as hd


def test_actual_state_sampling_is_full_speed_and_unique():
    steps = sample_steps(400)
    assert len(steps) == 240
    assert steps[0] == 0
    assert steps[-1] == 398
    assert len(set(steps)) == len(steps)
    assert set(b - a for a, b in zip(steps, steps[1:])) == {1, 2}
    assert len(steps) / 30 == 400 / 50
    assert max(abs(step / 50 - frame / 30) for frame, step in enumerate(steps)) <= 0.01


@pytest.mark.parametrize("horizon,fps", [(0, 30), (400, 60), (400, 0)])
def test_fake_fps_and_invalid_horizon_are_rejected(horizon, fps):
    with pytest.raises(ValueError):
        sample_steps(horizon, fps=fps)


@pytest.fixture
def bound_report(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "reports/runs/go2_flat_g009_r0_rev32_rear_damping_s42_20260927_0910.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    intervention = json.loads(source.with_name(source.stem + "_intervention.json").read_text(encoding="utf-8"))
    checkpoint = tmp_path / "model_49.pt"
    checkpoint.write_bytes(b"test checkpoint")
    value["artifacts"]["checkpoint"] = str(checkpoint)
    value["artifacts"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    folder = tmp_path / "reports/runs"
    folder.mkdir(parents=True)
    path = folder / source.name
    monkeypatch.setattr(hd, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hd, "validate_source_bundle", lambda bundle: bundle)
    return path, value, intervention


def save_binding(fixture):
    path, value, intervention = fixture
    path.write_text(json.dumps(value), encoding="utf-8")
    path.with_name(path.stem + "_intervention.json").write_text(json.dumps(intervention), encoding="utf-8")
    return path


def test_rejected_smoke_is_valid_diagnostic_not_qualification(bound_report):
    value, _ = hd.validate_binding(save_binding(bound_report))
    assert value["passed"] is False
    assert len(value["verified_intervention"]["sha256"]) == 64


@pytest.mark.parametrize("mutation", ["identity", "qualification", "manifest", "snapshot", "runtime", "gains",
                                      "headless", "resume", "hydra", "last_iteration", "iteration_target", "safety_gate"])
def test_wrong_smoke_bindings_are_rejected(bound_report, mutation):
    _, value, intervention = bound_report
    if mutation == "identity":
        value["run_name"] = value["run_name"].replace("rev32", "rev31")
    elif mutation == "qualification":
        value["qualification_mode"]["enabled"] = True
    elif mutation == "manifest":
        value["source_bundle"]["files"].pop("scripts/g009_r0_rev32.py")
    elif mutation == "snapshot":
        value["source_bundle"]["postrun"]["repository_commit"] = "0" * 40
    elif mutation == "runtime":
        intervention["runtime"]["seed"] = 43
    elif mutation == "gains":
        for field in ("actuator_before", "actuator_after"):
            intervention[field]["damping_by_joint"]["FR_calf_joint"] = 1.0
    elif mutation == "headless":
        value["headless"] = False
    elif mutation == "resume":
        value["resume"]["enabled"] = True
    elif mutation == "hydra":
        value["effective_hydra_overrides"] = ["env.fake=1"]
    elif mutation in ("last_iteration", "iteration_target"):
        value[mutation] -= 1
    elif mutation == "safety_gate":
        value["training_safety_gate"]["required"] = False
    with pytest.raises(ValueError):
        hd.validate_binding(save_binding(bound_report))


@pytest.mark.parametrize("mutation", ["protocol", "runtime", "gains", "qualification"])
def test_builder_cannot_bypass_training_intervention_checks(bound_report, mutation):
    import build_g009_r0_damping_hd_media as builder
    _, value, intervention = bound_report
    if mutation == "protocol":
        intervention["protocol"] = "other experiment"
    elif mutation == "runtime":
        intervention["runtime"]["seed"] = 43
    elif mutation == "gains":
        for key in ("actuator_before", "actuator_after"):
            intervention[key]["damping_by_joint"]["FR_calf_joint"] = 1.0
    else:
        intervention["qualification_eligible"] = True
    path = save_binding(bound_report)
    capture = {"diagnostic_only": True, "qualification_eligible": False,
               "training_report": {"path": str(path), "sha256": hd.file_sha256(path)}}
    with pytest.raises(ValueError):
        builder.validate_capture_binding(capture)
