"""RSL-RL runner configuration for the G009 R0 recovery policy."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)

from .recover_contracts import PPO_INIT_NOISE_STD


@configclass
class G009RecoverPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g009_recover_r0"
        self.max_iterations = 300
        self.save_interval = 50
        self.clip_actions = 1.0
        self.policy.init_noise_std = PPO_INIT_NOISE_STD


__all__ = ["G009RecoverPPORunnerCfg"]
