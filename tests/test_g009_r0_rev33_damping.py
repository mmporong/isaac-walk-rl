from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g009_r0_rev33 import actuator_readback, configure_damping


NAMES = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]


def test_front_extension_changes_only_remaining_calf_group():
    actuator = SimpleNamespace(stiffness=25.0, effort_limit=23.5, saturation_effort=23.5, damping=0.5)
    cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(actuators={"base_legs": actuator})))
    configure_damping(cfg)
    assert actuator.damping == {"FL_calf_joint": 1.0, "FR_calf_joint": 1.0,
                               "RL_calf_joint": 1.0, "RR_calf_joint": 1.0,
                               ".*_hip_joint": 0.5, ".*_thigh_joint": 0.5}
    assert actuator.effort_limit == actuator.saturation_effort == 23.5


def test_all_environment_readback_and_front_mismatch():
    actuator = SimpleNamespace(joint_names=NAMES,
                               damping=torch.tensor([[0.5] * 8 + [1.0] * 4] * 3),
                               stiffness=torch.full((3, 12), 25.0), effort_limit=torch.full((3, 12), 23.5))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(actuators={"base_legs": actuator})})
    assert actuator_readback(env)["environments_checked"] == 3
    actuator.damping[2, 9] = 0.5
    with pytest.raises(RuntimeError, match="damping"):
        actuator_readback(env)
