"""Gym registrations for G009 slope qualification and recovery tasks."""

from __future__ import annotations

import gymnasium as gym


AGENT_ENTRY_POINT = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents."
    "rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
)
TASK_ENTRY_POINTS = {
    "Isaac-G009-Velocity-Slope-Go2-S0-v0": "isaac_walk_g009.env_cfg:G009SlopeWalkEnvCfg",
}
RECOVER_TASK_ENTRY_POINTS = {
    "Isaac-G009-Recover-Flat-Go2-R0-v0": "isaac_walk_g009.recover_env_cfg:G009FlatRecoverEnvCfg",
    "Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0": (
        "isaac_walk_g009.recover_env_cfg:G009FlatRecoverMatrixEnvCfg"
    ),
    "Isaac-G009-Recover-Flat-Go2-R0-MatrixGate01-v0": (
        "isaac_walk_g009.recover_env_cfg:G009FlatRecoverMatrixGate01EnvCfg"
    ),
}
AGENT_ENTRY_POINTS = {
    "Isaac-G009-Velocity-Slope-Go2-S0-v0": AGENT_ENTRY_POINT,
    "Isaac-G009-Recover-Flat-Go2-R0-v0": "isaac_walk_g009.agent_cfg:G009RecoverPPORunnerCfg",
    "Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0": (
        "isaac_walk_g009.agent_cfg:G009RecoverPPORunnerCfg"
    ),
    "Isaac-G009-Recover-Flat-Go2-R0-MatrixGate01-v0": (
        "isaac_walk_g009.agent_cfg:G009RecoverPPORunnerCfg"
    ),
}


def register_tasks() -> None:
    for task_id, env_cfg_entry_point in (TASK_ENTRY_POINTS | RECOVER_TASK_ENTRY_POINTS).items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": env_cfg_entry_point,
                "rsl_rl_cfg_entry_point": AGENT_ENTRY_POINTS[task_id],
            },
        )


__all__ = [
    "AGENT_ENTRY_POINT",
    "AGENT_ENTRY_POINTS",
    "RECOVER_TASK_ENTRY_POINTS",
    "TASK_ENTRY_POINTS",
    "register_tasks",
]
