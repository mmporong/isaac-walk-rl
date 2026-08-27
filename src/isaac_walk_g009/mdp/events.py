"""Reset events that place the robot consistently on the G009 slope mesh."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from ..sim_terrain import terrain_artifacts, terrain_spec_from_cfg


def _sample_uniform(
    count: int,
    ranges: dict[str, tuple[float, float]],
    keys: tuple[str, ...],
    device: str,
) -> torch.Tensor:
    bounds = torch.tensor([ranges.get(key, (0.0, 0.0)) for key in keys], dtype=torch.float32, device=device)
    return math_utils.sample_uniform(bounds[:, 0], bounds[:, 1], (count, len(keys)), device=device)


def _surface_pose(
    spawn_cfg,
    xy_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact mesh-triangle heights and upward normals for world XY points."""
    field, arrays = terrain_artifacts(terrain_spec_from_cfg(spawn_cfg))
    xy = xy_w.detach().cpu().numpy().astype(np.float64, copy=False)
    spec = field.spec
    fx = (xy[:, 0] - spec.x_min_m) / spec.cell_size_m
    fy = (xy[:, 1] - spec.y_min_m) / spec.cell_size_m
    ix = np.floor(fx).astype(np.int64)
    iy = np.floor(fy).astype(np.int64)
    nx = field.x_coords_m.size
    ny = field.y_coords_m.size
    if np.any(ix < 0) or np.any(iy < 0) or np.any(ix >= nx - 1) or np.any(iy >= ny - 1):
        raise ValueError("G009 reset position lies outside the generated slope mesh")

    tx = fx - ix
    ty = fy - iy
    p00 = ix * ny + iy
    p10 = (ix + 1) * ny + iy
    p01 = ix * ny + iy + 1
    p11 = (ix + 1) * ny + iy + 1
    first_triangle = tx >= ty
    faces = np.where(
        first_triangle[:, None],
        np.column_stack((p00, p10, p11)),
        np.column_stack((p00, p11, p01)),
    )
    points = arrays["points"].astype(np.float64, copy=False)
    a = points[faces[:, 0]]
    b = points[faces[:, 1]]
    c = points[faces[:, 2]]
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    normal[normal[:, 2] < 0.0] *= -1.0
    height = a[:, 2] - (normal[:, 0] * (xy[:, 0] - a[:, 0]) + normal[:, 1] * (xy[:, 1] - a[:, 1])) / normal[:, 2]
    return (
        torch.as_tensor(height, dtype=xy_w.dtype, device=xy_w.device),
        torch.as_tensor(normal, dtype=xy_w.dtype, device=xy_w.device),
    )


def _orientation_on_normal(normal_w: torch.Tensor, yaw: torch.Tensor, azimuth_deg: float) -> torch.Tensor:
    """Build a body frame whose up axis is the support normal and yaw is about it."""
    azimuth = torch.deg2rad(torch.full_like(yaw, float(azimuth_deg)))
    uphill = torch.stack((torch.cos(azimuth), torch.sin(azimuth), torch.zeros_like(azimuth)), dim=-1)
    tangent_x = uphill - torch.sum(uphill * normal_w, dim=-1, keepdim=True) * normal_w
    tangent_x = torch.nn.functional.normalize(tangent_x, dim=-1)
    tangent_y = torch.linalg.cross(normal_w, tangent_x, dim=-1)
    cos_yaw = torch.cos(yaw).unsqueeze(-1)
    sin_yaw = torch.sin(yaw).unsqueeze(-1)
    body_x = cos_yaw * tangent_x + sin_yaw * tangent_y
    body_y = -sin_yaw * tangent_x + cos_yaw * tangent_y
    rotation = torch.stack((body_x, body_y, normal_w), dim=-1)
    return math_utils.quat_from_matrix(rotation)


def reset_root_state_on_slope(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_cfg_name: str = "slope_field",
) -> None:
    """Reset root height/orientation against the exact S0 mesh and record the sampled state."""
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    pose_samples = _sample_uniform(len(env_ids), pose_range, ("x", "y", "z", "yaw"), asset.device)
    positions = root_states[:, :3] + env.scene.env_origins[env_ids]
    positions[:, :2] += pose_samples[:, :2]

    spawn_cfg = getattr(env.cfg.scene, terrain_cfg_name).spawn
    surface_height, normal_w = _surface_pose(spawn_cfg, positions[:, :2])
    positions[:, 2] = surface_height + root_states[:, 2] + pose_samples[:, 2]
    orientations = _orientation_on_normal(normal_w, pose_samples[:, 3], spawn_cfg.azimuth_deg)

    velocity_samples = _sample_uniform(
        len(env_ids), velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"), asset.device
    )
    velocities = root_states[:, 7:13] + velocity_samples
    asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    reset_log = env.extras.setdefault("g009_reset", {})
    reset_log["env_ids"] = env_ids.detach().clone()
    reset_log["root_pose_w"] = torch.cat((positions, orientations), dim=-1).detach().clone()
    reset_log["root_velocity_w"] = velocities.detach().clone()
    reset_log["support_normal_w"] = normal_w.detach().clone()


__all__ = ["reset_root_state_on_slope"]
