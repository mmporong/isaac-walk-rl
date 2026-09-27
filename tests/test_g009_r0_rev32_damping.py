from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rev32", ROOT / "scripts/g009_r0_rev32.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config():
    actuator = SimpleNamespace(stiffness=25.0, effort_limit=23.5, saturation_effort=23.5, damping=0.5)
    return SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(actuators={"base_legs": actuator})))


def test_changes_only_two_rear_damping_values():
    cfg = config()
    MODULE.configure_damping(cfg)
    actuator = cfg.scene.robot.actuators["base_legs"]
    assert actuator.damping["RL_calf_joint"] == actuator.damping["RR_calf_joint"] == 1.0
    assert actuator.damping["FL_calf_joint"] == actuator.damping["FR_calf_joint"] == 0.5
    assert actuator.damping[".*_hip_joint"] == actuator.damping[".*_thigh_joint"] == 0.5
    assert (actuator.stiffness, actuator.effort_limit, actuator.saturation_effort) == (25.0, 23.5, 23.5)


@pytest.mark.parametrize("value", [float("nan"), 0.0, 2.0])
def test_rejects_unregistered_values(value):
    with pytest.raises(ValueError):
        MODULE.configure_damping(config(), value)


def test_live_readback_checks_every_environment_and_joint():
    names = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
    damping = torch.tensor([[1.0 if n in MODULE.REAR_CALVES else 0.5 for n in names]] * 3)
    actuator = SimpleNamespace(joint_names=names, damping=damping, stiffness=torch.full((3, 12),25.0), effort_limit=torch.full((3,12),23.5))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(actuators={"base_legs":actuator})})
    assert MODULE.actuator_readback(env)["environments_checked"] == 3
    actuator.damping[2,0] = 1.0
    with pytest.raises(RuntimeError, match="damping"):
        MODULE.actuator_readback(env)
