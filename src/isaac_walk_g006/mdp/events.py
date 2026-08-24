"""Push event with an explicit body-frame magnitude curriculum."""

from __future__ import annotations

import math

import torch

from .curriculums import push_schedule_for_step


def push_robot_body_xy(
    env,
    env_ids: torch.Tensor,
    asset_cfg=None,
) -> None:
    """Add a yaw-rotated body-XY delta velocity and retain attached evidence."""
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply_yaw

    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    robot = env.scene[asset_cfg.name]
    if env_ids.numel() == 0:
        return
    stage, minimum, maximum = push_schedule_for_step(int(env.common_step_counter))
    count = len(env_ids)
    angle = torch.rand(count, device=robot.device) * (2.0 * math.pi)
    magnitude = minimum + torch.rand(count, device=robot.device) * (maximum - minimum)
    delta_body = torch.stack(
        (magnitude * torch.cos(angle), magnitude * torch.sin(angle), torch.zeros_like(magnitude)), dim=-1
    )
    delta_world = quat_apply_yaw(robot.data.root_quat_w[env_ids], delta_body)
    velocity_world = robot.data.root_vel_w[env_ids].clone()
    velocity_world[:, :3] += delta_world
    robot.write_root_velocity_to_sim(velocity_world, env_ids=env_ids)

    if not hasattr(env, "_g006_push_counts"):
        env._g006_push_counts = [0, 0, 0]
        env._g006_push_magnitude_sum = [0.0, 0.0, 0.0]
        env._g006_push_magnitude_min = [float("inf"), float("inf"), float("inf")]
        env._g006_push_magnitude_max = [0.0, 0.0, 0.0]
    values = magnitude.detach()
    env._g006_push_counts[stage] += count
    env._g006_push_magnitude_sum[stage] += float(values.sum().item())
    env._g006_push_magnitude_min[stage] = min(env._g006_push_magnitude_min[stage], float(values.min().item()))
    env._g006_push_magnitude_max[stage] = max(env._g006_push_magnitude_max[stage], float(values.max().item()))
