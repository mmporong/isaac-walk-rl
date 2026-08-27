"""Gym registration for the G009 S0 slope task."""

from __future__ import annotations

import gymnasium as gym


AGENT_ENTRY_POINT = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents."
    "rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
)
TASK_ENTRY_POINTS = {
    "Isaac-G009-Velocity-Slope-Go2-S0-v0": "isaac_walk_g009.env_cfg:G009SlopeWalkEnvCfg",
}


def register_tasks() -> None:
    for task_id, env_cfg_entry_point in TASK_ENTRY_POINTS.items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": env_cfg_entry_point,
                "rsl_rl_cfg_entry_point": AGENT_ENTRY_POINT,
            },
        )


__all__ = ["AGENT_ENTRY_POINT", "TASK_ENTRY_POINTS", "register_tasks"]
