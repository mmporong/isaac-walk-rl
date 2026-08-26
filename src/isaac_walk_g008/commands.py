"""Balanced velocity-command generator for forward, reverse, and yaw turns."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .contracts import COMMAND_PRIMITIVE_PROBABILITY, COMMAND_PRIMITIVES


class BalancedVelocityCommand(UniformVelocityCommand):
    """Mix exact command primitives with continuous SE(2) velocity samples."""

    cfg: "BalancedVelocityCommandCfg"

    def __init__(self, cfg: "BalancedVelocityCommandCfg", env):
        if cfg.heading_command:
            raise ValueError("BalancedVelocityCommand requires direct yaw-rate commands")
        super().__init__(cfg, env)
        if not 0.0 <= cfg.primitive_probability <= 1.0:
            raise ValueError("primitive_probability must be in [0, 1]")
        if len(cfg.primitive_commands) != len(cfg.primitive_weights):
            raise ValueError("primitive command and weight counts must match")
        if abs(sum(cfg.primitive_weights) - 1.0) > 1.0e-8:
            raise ValueError("primitive weights must sum to one")
        if any(len(command) != 3 for command in cfg.primitive_commands):
            raise ValueError("each primitive command must have three components")
        self._primitive_commands = torch.tensor(cfg.primitive_commands, dtype=torch.float32, device=self.device)
        self._primitive_weights = torch.tensor(cfg.primitive_weights, dtype=torch.float32, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        resolved_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if resolved_ids.numel() == 0:
            return
        primitive_mask = torch.rand(resolved_ids.numel(), device=self.device) < self.cfg.primitive_probability
        primitive_env_ids = resolved_ids[primitive_mask]
        if primitive_env_ids.numel() == 0:
            return
        primitive_indices = torch.multinomial(
            self._primitive_weights,
            int(primitive_env_ids.numel()),
            replacement=True,
        )
        sampled = self._primitive_commands[primitive_indices]
        self.vel_command_b[primitive_env_ids] = sampled
        self.is_heading_env[primitive_env_ids] = False
        self.is_standing_env[primitive_env_ids] = torch.all(sampled == 0.0, dim=1)


@configclass
class BalancedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for a primitive-heavy, continuously supported velocity distribution."""

    class_type: type = BalancedVelocityCommand
    primitive_probability: float = COMMAND_PRIMITIVE_PROBABILITY
    primitive_commands: tuple[tuple[float, float, float], ...] = tuple(
        item.velocity_mps_radps for item in COMMAND_PRIMITIVES
    )
    primitive_weights: tuple[float, ...] = tuple(item.weight for item in COMMAND_PRIMITIVES)
