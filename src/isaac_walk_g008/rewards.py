"""G008 reward functions that extend the pinned Isaac Lab task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time_turn_aware(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    yaw_command_threshold: float,
) -> torch.Tensor:
    """Reward long steps for translation or an explicit yaw command.

    Isaac Lab's quadruped term disables this reward when the planar linear
    command is small.  That also disables it for pure in-place turns.  This
    variant keeps the original calculation and linear-command gate, then adds
    a separate yaw-rate gate with units stated in the configuration.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    command = env.command_manager.get_command(command_name)
    translation_active = torch.linalg.vector_norm(command[:, :2], dim=1) > 0.1
    turn_active = torch.abs(command[:, 2]) > yaw_command_threshold
    return reward * (translation_active | turn_active)
