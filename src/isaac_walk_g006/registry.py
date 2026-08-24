"""Gym registrations kept import-light through string entry points."""

from __future__ import annotations

import gymnasium as gym

BASELINE_TASK_ID = "Isaac-G006-Velocity-Rough-Go2-Baseline-v0"
PUSH_TASK_ID = "Isaac-G006-Velocity-Rough-Go2-PushCurriculum-v0"
AGENT_ENTRY_POINT = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents."
    "rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
)


def _register(task_id: str, env_entry_point: str) -> None:
    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_entry_point,
            "rsl_rl_cfg_entry_point": AGENT_ENTRY_POINT,
        },
    )


def register_tasks() -> None:
    _register(
        BASELINE_TASK_ID,
        "isaac_walk_g006.rough_env_cfg:G006RoughBaselineEnvCfg",
    )
    _register(
        PUSH_TASK_ID,
        "isaac_walk_g006.rough_env_cfg:G006RoughPushCurriculumEnvCfg",
    )
