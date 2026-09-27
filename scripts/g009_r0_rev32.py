"""Isolated rear-calf damping intervention; historical R0 config stays intact."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REAR_CALVES = ("RL_calf_joint", "RR_calf_joint")
CANDIDATE_DAMPING = 1.0
BASELINE_DAMPING = 0.5


def configure_damping(cfg: Any, damping: float = CANDIDATE_DAMPING) -> None:
    if not math.isfinite(damping) or damping not in (BASELINE_DAMPING, CANDIDATE_DAMPING):
        raise ValueError("rev32 damping must be 0.5 or 1.0 N*m*s/rad")
    actuator = cfg.scene.robot.actuators["base_legs"]
    if actuator.stiffness != 25.0 or actuator.effort_limit != 23.5 or actuator.saturation_effort != 23.5:
        raise ValueError("pinned Go2 stiffness/effort contract mismatch")
    if actuator.damping != BASELINE_DAMPING:
        raise ValueError("intervention requires the scalar baseline damping 0.5")
    actuator.damping = {
        "FL_calf_joint": BASELINE_DAMPING,
        "FR_calf_joint": BASELINE_DAMPING,
        "RL_calf_joint": damping,
        "RR_calf_joint": damping,
        ".*_hip_joint": BASELINE_DAMPING,
        ".*_thigh_joint": BASELINE_DAMPING,
    }


def actuator_readback(env: Any, damping: float = CANDIDATE_DAMPING) -> dict[str, Any]:
    import torch

    actuator = env.scene["robot"].actuators["base_legs"]
    names = list(actuator.joint_names)
    if len(names) != 12 or len(set(names)) != 12 or not set(REAR_CALVES).issubset(names):
        raise RuntimeError("expected twelve unique Go2 joints including both rear calves")
    expected = torch.tensor(
        [damping if name in REAR_CALVES else BASELINE_DAMPING for name in names],
        device=actuator.damping.device, dtype=actuator.damping.dtype,
    )
    for name, values, targets in (
        ("damping", actuator.damping, expected),
        ("stiffness", actuator.stiffness, torch.full_like(expected, 25.0)),
        ("effort_limit", actuator.effort_limit, torch.full_like(expected, 23.5)),
    ):
        if not torch.isfinite(values).all() or not torch.allclose(values, targets.expand_as(values), atol=1e-7, rtol=0):
            raise RuntimeError(f"live actuator {name} differs from preregistration")
    return {
        "joint_names": names,
        "damping_by_joint": dict(zip(names, actuator.damping[0].detach().cpu().tolist())),
        "stiffness_by_joint": dict(zip(names, actuator.stiffness[0].detach().cpu().tolist())),
        "effort_limit_by_joint": dict(zip(names, actuator.effort_limit[0].detach().cpu().tolist())),
        "environments_checked": int(actuator.damping.shape[0]),
    }
