"""Reset events for G009 slope walking and flat-ground self-righting."""

from __future__ import annotations

import math
import numpy as np
import torch

from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils

from ..recover_contracts import (
    FOLDED_JOINT_ANGLES_RAD,
    GROUND_DYNAMIC_FRICTION,
    GROUND_STATIC_FRICTION,
    POSE_CURRICULUM_PHASE_END_CONTROL_STEPS,
    POSE_CURRICULUM_PROBABILITIES,
    RECOVER_POSES,
    RESET_POSE_XY_RANGE_M,
    RESET_YAW_RANGE_RAD,
    canonical_sha256,
    recover_contract,
)
from ..sim_terrain import terrain_artifacts, terrain_spec_from_cfg


_RECOVER_POSE_NAMES = tuple(RECOVER_POSES)
_RECOVER_CONTRACT_SHA256 = canonical_sha256(recover_contract())


def _normalized_pose_probabilities(values) -> tuple[float, ...]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().flatten().tolist()
    probabilities = tuple(float(value) for value in values)
    if len(probabilities) != len(_RECOVER_POSE_NAMES):
        raise ValueError(
            f"Expected {len(_RECOVER_POSE_NAMES)} recovery pose probabilities, "
            f"received {len(probabilities)}"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("Recovery pose probabilities must be finite and non-negative")
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("Recovery pose probabilities must have a positive sum")
    return tuple(value / total for value in probabilities)


def recovery_pose_curriculum(
    env,
    env_ids,
    phase_end_control_steps: tuple[int, int] = POSE_CURRICULUM_PHASE_END_CONTROL_STEPS,
    phase_probabilities: tuple[tuple[float, ...], ...] = POSE_CURRICULUM_PROBABILITIES,
) -> dict[str, float]:
    """Set the global exact-pose reset distribution before reset events run."""
    del env_ids
    if len(phase_end_control_steps) != 2 or len(phase_probabilities) != 3:
        raise ValueError("R0 pose curriculum requires two boundaries and three phases")
    phase_boundaries = tuple(int(value) for value in phase_end_control_steps)
    first_end, second_end = phase_boundaries
    if first_end <= 0 or second_end <= first_end:
        raise ValueError("R0 pose curriculum boundaries must be strictly increasing")

    phase_cache_key = (
        phase_boundaries,
        tuple(tuple(float(value) for value in phase) for phase in phase_probabilities),
    )
    cached = getattr(env, "_g009_recover_pose_curriculum_cache", None)
    if cached is None or cached[0] != phase_cache_key:
        normalized_phases = tuple(
            _normalized_pose_probabilities(phase) for phase in phase_probabilities
        )
        cached = (phase_cache_key, normalized_phases)
        env._g009_recover_pose_curriculum_cache = cached
    else:
        normalized_phases = cached[1]

    control_step = int(env.common_step_counter)
    phase_index = 0 if control_step < first_end else 1 if control_step < second_end else 2
    probability_values = normalized_phases[phase_index]
    if (
        getattr(env, "_g009_recover_curriculum_phase", None) != phase_index
        or getattr(env, "_g009_recover_pose_probabilities", None) is None
    ):
        env._g009_recover_pose_probabilities = torch.tensor(
            probability_values,
            device=env.device,
            dtype=torch.float32,
        )
    env._g009_recover_pose_probabilities_valid = True
    env._g009_recover_curriculum_phase = phase_index
    env._g009_recover_curriculum_control_step = control_step

    state = {
        "phase_index": float(phase_index),
        "common_control_step": float(control_step),
    }
    state.update(
        {
            f"probability_{name}": probability_values[index]
            for index, name in enumerate(_RECOVER_POSE_NAMES)
        }
    )
    return state


class ApplyRecoverFootMaterial(randomize_rigid_body_material):
    """Apply R0 foot materials and retain verified PhysX readback provenance.

    Isaac Lab constructs the observation manager before startup events run.  The
    buffer is therefore allocated here, marked invalid, and populated only after
    ``set_material_properties`` has completed and the values have been read back.
    """

    def __init__(self, cfg: EventTermCfg, env) -> None:
        super().__init__(cfg, env)
        if self.num_shapes_per_body is None:
            raise ValueError("G009 foot material provenance requires explicit foot body ids")
        if len(self.asset_cfg.body_ids) != 4:
            raise ValueError(f"Expected four foot bodies, resolved {len(self.asset_cfg.body_ids)}")
        env._g009_effective_foot_friction = torch.zeros(
            (env.num_envs, 4, 2), device=self.asset.device, dtype=torch.float32
        )
        env._g009_foot_material_readback = torch.zeros_like(env._g009_effective_foot_friction)
        env._g009_effective_foot_friction_valid = torch.zeros(
            env.num_envs, device=self.asset.device, dtype=torch.bool
        )
        env._g009_r0_body_mass = None
        env._g009_r0_body_mass_valid = False

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        ground_static_friction: float,
        ground_dynamic_friction: float,
        make_consistent: bool = False,
    ) -> None:
        if env.cfg.scene.terrain.physics_material.friction_combine_mode != "multiply":
            raise ValueError("G009 R0 effective friction provenance requires multiply combine mode")
        configured_ground = (
            float(env.cfg.scene.terrain.physics_material.static_friction),
            float(env.cfg.scene.terrain.physics_material.dynamic_friction),
        )
        requested_ground = (float(ground_static_friction), float(ground_dynamic_friction))
        if configured_ground != requested_ground:
            raise ValueError(
                f"Ground material mismatch: configured={configured_ground}, requested={requested_ground}"
            )

        super().__call__(
            env,
            env_ids,
            static_friction_range,
            dynamic_friction_range,
            restitution_range,
            num_buckets,
            asset_cfg,
            make_consistent,
        )

        if env_ids is None:
            env_ids_cpu = torch.arange(env.num_envs, device="cpu", dtype=torch.long)
        else:
            env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        materials = self.asset.root_physx_view.get_material_properties()
        foot_pairs = []
        for body_id in self.asset_cfg.body_ids:
            start = sum(self.num_shapes_per_body[:body_id])
            end = start + self.num_shapes_per_body[body_id]
            if end <= start:
                raise RuntimeError(f"Foot body {body_id} has no collision shapes")
            body_material = materials[env_ids_cpu, start:end, :2]
            first_shape = body_material[:, :1, :]
            if not torch.allclose(body_material, first_shape.expand_as(body_material), atol=1.0e-6, rtol=0.0):
                raise RuntimeError(f"Foot body {body_id} has inconsistent shape material readback")
            foot_pairs.append(first_shape[:, 0, :])
        readback = torch.stack(foot_pairs, dim=1)
        if not torch.isfinite(readback).all() or (readback < 0.0).any():
            raise RuntimeError("Foot material readback is non-finite or negative")
        if (readback[..., 1] > readback[..., 0] + 1.0e-6).any():
            raise RuntimeError("Dynamic foot friction exceeds static friction")

        device_ids = env_ids_cpu.to(device=self.asset.device)
        readback_device = readback.to(device=self.asset.device, dtype=torch.float32)
        ground_pair = readback_device.new_tensor(requested_ground)
        env._g009_foot_material_readback[device_ids] = readback_device
        env._g009_effective_foot_friction[device_ids] = readback_device * ground_pair
        env._g009_effective_foot_friction_valid[device_ids] = True
        body_mass = self.asset.root_physx_view.get_masses().to(device=self.asset.device, dtype=torch.float32)
        if body_mass.shape[0] != env.num_envs or not torch.isfinite(body_mass).all() or (body_mass <= 0.0).any():
            raise RuntimeError("G009 R0 body-mass readback is invalid")
        env._g009_r0_body_mass = body_mass
        env._g009_r0_body_mass_valid = True


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


def _recover_class_ids(count: int, probabilities: torch.Tensor, assignment_mode: str) -> torch.Tensor:
    """Sample or deterministically stratify the four frozen R0 fall classes."""
    if assignment_mode == "random":
        return torch.multinomial(probabilities, count, replacement=True)
    if assignment_mode == "stratified":
        return torch.arange(count, device=probabilities.device, dtype=torch.long) % len(_RECOVER_POSE_NAMES)
    raise ValueError("assignment_mode must be 'random' or 'stratified'")


def _folded_recover_joint_positions(
    asset,
    env_ids: torch.Tensor,
    *,
    left_hip_angle: float,
    right_hip_angle: float,
    thigh_angle: float,
    calf_angle: float,
) -> torch.Tensor:
    """Build a compact Go2 seed by joint name instead of relying on USD ordering."""
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    matched = {"hip": 0, "thigh": 0, "calf": 0}
    for joint_index, joint_name in enumerate(asset.joint_names):
        if joint_name.endswith("_hip_joint"):
            joint_pos[:, joint_index] = left_hip_angle if "L_" in joint_name else right_hip_angle
            matched["hip"] += 1
        elif joint_name.endswith("_thigh_joint"):
            joint_pos[:, joint_index] = thigh_angle
            matched["thigh"] += 1
        elif joint_name.endswith("_calf_joint"):
            joint_pos[:, joint_index] = calf_angle
            matched["calf"] += 1
    if matched != {"hip": 4, "thigh": 4, "calf": 4}:
        raise ValueError(f"Expected four Go2 joints per leg segment, resolved {matched}")
    return joint_pos


def reset_root_and_joints_for_recovery(
    env,
    env_ids: torch.Tensor | None,
    pose_xy_range: tuple[float, float] = RESET_POSE_XY_RANGE_M,
    yaw_range: tuple[float, float] = RESET_YAW_RANGE_RAD,
    assignment_mode: str = "random",
    left_hip_angle: float = FOLDED_JOINT_ANGLES_RAD["left_hip"],
    right_hip_angle: float = FOLDED_JOINT_ANGLES_RAD["right_hip"],
    thigh_angle: float = FOLDED_JOINT_ANGLES_RAD["thigh"],
    calf_angle: float = FOLDED_JOINT_ANGLES_RAD["calf"],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset root and all twelve joints from one curated R0 event.

    The pose class is critic-only metadata.  ``stratified`` assignment exists
    for the four-environment runtime calibration; training uses probabilistic
    assignment from the frozen contract.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    count = len(env_ids)
    if count == 0:
        return

    probabilities = getattr(env, "_g009_recover_pose_probabilities", None)
    if probabilities is None:
        probability_values = _normalized_pose_probabilities(
            RECOVER_POSES[name].probability for name in _RECOVER_POSE_NAMES
        )
        probabilities = torch.tensor(
            probability_values,
            device=asset.device,
            dtype=torch.float32,
        )
        env._g009_recover_pose_probabilities = probabilities
        env._g009_recover_pose_probabilities_valid = True
    elif not getattr(env, "_g009_recover_pose_probabilities_valid", False):
        probability_values = _normalized_pose_probabilities(probabilities)
        probabilities = torch.tensor(
            probability_values,
            device=asset.device,
            dtype=torch.float32,
        )
        env._g009_recover_pose_probabilities = probabilities
        env._g009_recover_pose_probabilities_valid = True
    elif probabilities.device != torch.device(asset.device):
        probabilities = probabilities.to(device=asset.device)
        env._g009_recover_pose_probabilities = probabilities
    class_ids = _recover_class_ids(count, probabilities, assignment_mode)
    pose_quaternions = torch.tensor(
        [RECOVER_POSES[name].root_quaternion_wxyz for name in _RECOVER_POSE_NAMES],
        device=asset.device,
        dtype=asset.data.default_root_state.dtype,
    )[class_ids]
    root_heights = torch.tensor(
        [RECOVER_POSES[name].root_height_m for name in _RECOVER_POSE_NAMES],
        device=asset.device,
        dtype=asset.data.default_root_state.dtype,
    )[class_ids]

    xy_offset = math_utils.sample_uniform(
        float(pose_xy_range[0]),
        float(pose_xy_range[1]),
        (count, 2),
        asset.device,
    )
    yaw = math_utils.sample_uniform(
        float(yaw_range[0]),
        float(yaw_range[1]),
        (count,),
        asset.device,
    )
    zeros = torch.zeros_like(yaw)
    yaw_quaternions = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    orientations = math_utils.quat_mul(yaw_quaternions, pose_quaternions)

    positions = env.scene.env_origins[env_ids].clone()
    positions[:, :2] += xy_offset
    positions[:, 2] += root_heights
    root_pose_w = torch.cat((positions, orientations), dim=-1)
    root_velocity_w = torch.zeros((count, 6), device=asset.device, dtype=positions.dtype)
    joint_pos = _folded_recover_joint_positions(
        asset,
        env_ids,
        left_hip_angle=left_hip_angle,
        right_hip_angle=right_hip_angle,
        thigh_angle=thigh_angle,
        calf_angle=calf_angle,
    )
    joint_vel = torch.zeros_like(joint_pos)
    joint_limits = asset.data.joint_pos_limits[env_ids]
    outside_hard_limits = (joint_pos < joint_limits[..., 0]) | (joint_pos > joint_limits[..., 1])
    if outside_hard_limits.any():
        violating = torch.nonzero(outside_hard_limits, as_tuple=False)[0].tolist()
        raise ValueError(f"Folded recovery seed violates a hard joint limit at local index {violating}")

    # Root and joint state are deliberately owned by this single reset event so
    # a second inherited joint-reset term cannot overwrite the curated seed.
    asset.write_root_pose_to_sim(root_pose_w, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity_w, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    source_class = getattr(env, "_g009_recover_fall_class", None)
    if source_class is None or source_class.shape != (env.num_envs,):
        source_class = torch.zeros(env.num_envs, device=asset.device, dtype=torch.long)
        env._g009_recover_fall_class = source_class
    source_class[env_ids] = class_ids
    source_one_hot = getattr(env, "_g009_recover_fall_class_one_hot", None)
    if source_one_hot is None or source_one_hot.shape != (env.num_envs, len(_RECOVER_POSE_NAMES)):
        source_one_hot = torch.zeros(
            (env.num_envs, len(_RECOVER_POSE_NAMES)), device=asset.device, dtype=positions.dtype
        )
        env._g009_recover_fall_class_one_hot = source_one_hot
    source_one_hot[env_ids] = torch.nn.functional.one_hot(
        class_ids, num_classes=len(_RECOVER_POSE_NAMES)
    ).to(positions.dtype)

    actor_invalid = getattr(env, "_g009_actor_signal_invalid", None)
    if actor_invalid is not None:
        actor_invalid[env_ids] = False
    env._g009_ray_support_cache_step = -1

    reset_log = env.extras.setdefault("g009_recover_reset", {})
    reset_log["env_ids"] = env_ids.detach().clone()
    reset_log["source_class_ids"] = class_ids.detach().clone()
    reset_log["source_class_one_hot"] = source_one_hot[env_ids].detach().clone()
    reset_log["root_pose_w"] = root_pose_w.detach().clone()
    reset_log["root_velocity_w"] = root_velocity_w.detach().clone()
    reset_log["joint_pos"] = joint_pos.detach().clone()
    reset_log["joint_vel"] = joint_vel.detach().clone()
    reset_log["folded_joint_angles"] = positions.new_tensor(
        (left_hip_angle, right_hip_angle, thigh_angle, calf_angle)
    )
    reset_log["contract_sha256"] = _RECOVER_CONTRACT_SHA256


__all__ = [
    "ApplyRecoverFootMaterial",
    "recovery_pose_curriculum",
    "reset_root_and_joints_for_recovery",
    "reset_root_state_on_slope",
]
