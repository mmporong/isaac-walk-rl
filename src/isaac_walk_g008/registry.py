"""Gym registrations for the separated G008 experiment parts."""

from __future__ import annotations

import gymnasium as gym


AGENT_ENTRY_POINT = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents."
    "rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
)

TASK_ENTRY_POINTS = {
    "Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0": "isaac_walk_g008.env_cfg:G008CommandEnvCfg",
    "Isaac-G008-Velocity-Rough-Go2-Friction-S1-v0": "isaac_walk_g008.env_cfg:G008FrictionStage1EnvCfg",
    "Isaac-G008-Velocity-Rough-Go2-Friction-S2-v0": "isaac_walk_g008.env_cfg:G008FrictionStage2EnvCfg",
    "Isaac-G008-Velocity-Rough-Go2-Friction-S3-v0": "isaac_walk_g008.env_cfg:G008FrictionStage3EnvCfg",
    "Isaac-G008-Velocity-Rough-Go2-LegMass-S1-v0": "isaac_walk_g008.env_cfg:G008LegMassStage1EnvCfg",
    "Isaac-G008-Velocity-Rough-Go2-LegMass-S2-v0": "isaac_walk_g008.env_cfg:G008LegMassStage2EnvCfg",
    "Isaac-G008-Velocity-Rough-Go2-LegMass-S3-v0": "isaac_walk_g008.env_cfg:G008LegMassStage3EnvCfg",
}


def register_tasks() -> None:
    for task_id, env_entry_point in TASK_ENTRY_POINTS.items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": env_entry_point,
                "rsl_rl_cfg_entry_point": AGENT_ENTRY_POINT,
            },
        )
