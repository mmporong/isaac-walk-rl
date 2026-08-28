"""MDP terms for the G009 R0 self-righting task.

The actor-facing terms in this module use only deployable robot signals.  The
critic helpers are deliberately named ``critic_*`` because they read simulator
ground truth or training-only episode metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg


_WORLD_UP = (0.0, 0.0, 1.0)


def _world_up(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(_WORLD_UP).expand(reference.shape[0], -1)


def _upright_alignment(root_quat_w: torch.Tensor, normal_w: torch.Tensor) -> torch.Tensor:
    """Cosine alignment between body up and a unit support normal."""
    body_up_w = math_utils.quat_apply(root_quat_w, _world_up(root_quat_w))
    normal_w = torch.nn.functional.normalize(normal_w, dim=-1)
    return torch.sum(body_up_w * normal_w, dim=-1).clamp(-1.0, 1.0)


def _progress_delta(
    current: torch.Tensor,
    previous: torch.Tensor,
    scale: float = 1.0,
    discount_factor: float = 1.0,
) -> torch.Tensor:
    """Discount-compatible potential difference with an explicit scale."""
    if not math.isfinite(discount_factor) or not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be finite and in [0, 1]")
    return (discount_factor * current - previous) * scale


def _linear_gate(value: torch.Tensor, start: float, full: float) -> torch.Tensor:
    """Map a scalar state to a bounded zero-to-one gate."""
    if not math.isfinite(start) or not math.isfinite(full) or full <= start:
        raise ValueError("linear gate requires finite start < full")
    return ((value - start) / (full - start)).clamp(0.0, 1.0)


def _gated_height_potential_from_values(
    alignment: torch.Tensor,
    base_height: torch.Tensor,
    *,
    min_height: float,
    target_height: float,
    orientation_gate_start: float,
    orientation_gate_full: float,
) -> torch.Tensor:
    """Stage normalized height credit behind a soft upright gate."""
    height_score = _linear_gate(base_height, min_height, target_height)
    orientation_gate = _linear_gate(
        alignment, orientation_gate_start, orientation_gate_full
    )
    return orientation_gate * height_score


def _soft_stand_potential_from_values(
    alignment: torch.Tensor,
    base_height: torch.Tensor,
    foot_normal_load: torch.Tensor,
    *,
    min_height: float,
    target_height: float,
    contact_force_threshold: float,
    min_contacts: int,
    target_total_foot_load: float,
) -> torch.Tensor:
    """Score continuous progress toward upright, raised, loaded-foot support."""
    if min_contacts <= 0 or target_total_foot_load <= 0.0:
        raise ValueError("foot support targets must be positive")
    upright_score = ((alignment + 1.0) * 0.5).clamp(0.0, 1.0)
    height_score = _linear_gate(base_height, min_height, target_height)
    contact_score = (
        (foot_normal_load >= contact_force_threshold).sum(dim=-1).float()
        / float(min_contacts)
    ).clamp(0.0, 1.0)
    load_score = (
        foot_normal_load.sum(dim=-1) / float(target_total_foot_load)
    ).clamp(0.0, 1.0)
    support_score = 0.5 * contact_score + 0.5 * load_score
    return upright_score * height_score * support_score


def _stand_regularization_gate_from_values(
    alignment: torch.Tensor,
    base_height: torch.Tensor,
    *,
    orientation_gate_start: float,
    orientation_gate_full: float,
    height_gate_start: float,
    height_gate_full: float,
) -> torch.Tensor:
    """Return a smooth 0-to-1 gate from fallen motion to standing regulation."""
    return _linear_gate(
        alignment, orientation_gate_start, orientation_gate_full
    ) * _linear_gate(base_height, height_gate_start, height_gate_full)


def _positive_support_normal_load(
    forces_w: torch.Tensor,
    support_normal_w: torch.Tensor,
) -> torch.Tensor:
    """Project contact forces onto the positive support-normal direction."""
    normal_w = torch.nn.functional.normalize(support_normal_w, dim=-1)
    return torch.sum(forces_w * normal_w.unsqueeze(1), dim=-1).clamp_min(0.0)


def _stable_mask_from_values(
    alignment: torch.Tensor,
    base_height: torch.Tensor,
    contact_count: torch.Tensor,
    total_foot_normal_load: torch.Tensor,
    non_foot_contact_count: torch.Tensor,
    linear_speed: torch.Tensor,
    angular_speed: torch.Tensor,
    *,
    upright_threshold: float,
    min_base_height: float,
    max_base_height: float,
    min_contacts: int,
    min_total_foot_normal_load: float,
    max_linear_speed: float,
    max_angular_speed: float,
) -> torch.Tensor:
    """Pure tensor stable predicate shared by rewards and termination."""
    finite = torch.isfinite(alignment)
    finite &= torch.isfinite(base_height)
    finite &= torch.isfinite(total_foot_normal_load)
    finite &= torch.isfinite(linear_speed)
    finite &= torch.isfinite(angular_speed)
    return finite & (
        (alignment >= upright_threshold)
        & (base_height >= min_base_height)
        & (base_height <= max_base_height)
        & (contact_count >= min_contacts)
        & (total_foot_normal_load >= min_total_foot_normal_load)
        & (non_foot_contact_count == 0)
        & (linear_speed <= max_linear_speed)
        & (angular_speed <= max_angular_speed)
    )


def _asset(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> Articulation:
    return env.scene[cfg.name]


def _sensor(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> ContactSensor:
    return env.scene.sensors[cfg.name]


def _mark_actor_signal_invalid(env: ManagerBasedRLEnv, invalid: torch.Tensor) -> None:
    buffer = getattr(env, "_g009_actor_signal_invalid", None)
    if buffer is None or buffer.shape != (env.num_envs,):
        buffer = torch.zeros(env.num_envs, device=invalid.device, dtype=torch.bool)
        env._g009_actor_signal_invalid = buffer
    buffer |= invalid.to(device=buffer.device, dtype=torch.bool)


def _support_normal_w(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    normal = getattr(env, "_g009_terrain_normal_w", None)
    if normal is None:
        return _world_up(asset.data.root_quat_w)
    return normal.to(device=asset.device, dtype=asset.data.root_quat_w.dtype)


def _foot_force_magnitude(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = _sensor(env, sensor_cfg)
    return torch.linalg.vector_norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)


def _body_fixed_range_sample(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_distance: float,
    expected_ray_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized local camera ranges and a hit mask.

    ``distance_to_camera`` is already expressed in the body-fixed camera
    frame. Infinite samples are the configured no-hit/out-of-range sentinel
    and are therefore valid observations. Negative finite values and NaNs are
    invalid sensor data and fail closed through ``numeric_invalid``.
    """
    if max_distance <= 0.0:
        raise ValueError("max_distance must be positive")
    output = env.scene.sensors[sensor_cfg.name].data.output
    if "distance_to_camera" not in output:
        raise RuntimeError("G009 body-fixed range sensor lacks distance_to_camera output")
    distances = output["distance_to_camera"]
    if distances.ndim < 2 or distances.shape[0] != env.num_envs:
        raise ValueError("distance_to_camera must have leading shape (num_envs, ...)")
    distances = distances.reshape(env.num_envs, -1)
    if distances.shape[1] != expected_ray_count:
        raise ValueError(
            f"distance_to_camera must contain {expected_ray_count} rays, got {distances.shape[1]}"
        )

    finite = torch.isfinite(distances)
    invalid = torch.isnan(distances) | (finite & (distances < 0.0))
    _mark_actor_signal_invalid(env, invalid.any(dim=1))
    hit = finite & ~invalid & (distances <= max_distance)
    normalized = torch.where(
        hit,
        distances.clamp(min=0.0, max=max_distance) / max_distance,
        torch.ones_like(distances),
    )
    return normalized, hit.float()


def body_fixed_range(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("body_range_camera"),
    max_distance: float = 1.0,
    expected_ray_count: int = 15,
) -> torch.Tensor:
    """Return 15 normalized body-fixed range values; no-hit is 1.0."""
    return _body_fixed_range_sample(env, sensor_cfg, max_distance, expected_ray_count)[0]


def body_fixed_range_hit_mask(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("body_range_camera"),
    max_distance: float = 1.0,
    expected_ray_count: int = 15,
) -> torch.Tensor:
    """Return 1.0 for an in-range local camera hit and 0.0 otherwise."""
    return _body_fixed_range_sample(env, sensor_cfg, max_distance, expected_ray_count)[1]


def foot_contact_flags(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float,
) -> torch.Tensor:
    """Return one float contact flag for every resolved foot body id."""
    return (_foot_force_magnitude(env, sensor_cfg) >= contact_force_threshold).float()


four_foot_contact_state = foot_contact_flags


def normalized_foot_load(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    nominal_total_mass_kg: float = 15.019,
    gravity_magnitude: float = 9.81,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return ray-independent per-foot force magnitude / nominal robot weight."""
    if nominal_total_mass_kg <= 0.0:
        raise ValueError("nominal_total_mass_kg must be positive")
    load = _foot_force_magnitude(env, sensor_cfg)
    invalid = ~torch.isfinite(load).all(dim=-1)
    _mark_actor_signal_invalid(env, invalid)
    load = torch.nan_to_num(load, nan=0.0, posinf=0.0, neginf=0.0)
    denominator = max(float(nominal_total_mass_kg) * float(gravity_magnitude), eps)
    return load / denominator


normalized_four_foot_load = normalized_foot_load


def critic_terrain_normal_gt_b(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return simulator ground-truth support normal in the base frame."""
    asset = _asset(env, asset_cfg)
    return math_utils.quat_apply_inverse(asset.data.root_quat_w, _support_normal_w(env, asset))


terrain_normal_gt = critic_terrain_normal_gt_b


def critic_base_height_gt(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return R0 flat-ground base height relative to the environment origin."""
    asset = _asset(env, asset_cfg)
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return height.unsqueeze(-1)


base_height_gt = critic_base_height_gt


def critic_effective_foot_friction(
    env: ManagerBasedRLEnv,
    configured_static_friction: float,
    configured_dynamic_friction: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return readback-bound effective friction or fail closed after startup."""
    asset = _asset(env, asset_cfg)
    friction = getattr(env, "_g009_effective_foot_friction", None)
    valid = getattr(env, "_g009_effective_foot_friction_valid", None)
    manager_shape_inference = not hasattr(env, "observation_manager")
    if manager_shape_inference:
        pair = asset.data.root_quat_w.new_tensor(
            (configured_static_friction, configured_dynamic_friction)
        )
        return pair.expand(env.num_envs, 4, 2).reshape(env.num_envs, 8)
    if friction is None or valid is None or not bool(valid.all().item()):
        raise RuntimeError("G009 effective foot friction is unavailable or lacks startup readback provenance")
    friction = friction.to(device=asset.device, dtype=asset.data.root_quat_w.dtype)
    if friction.shape != (env.num_envs, 4, 2):
        raise ValueError("_g009_effective_foot_friction must have shape (num_envs, 4, 2)")
    if not torch.isfinite(friction).all() or (friction < 0.0).any():
        raise RuntimeError("G009 effective foot friction readback is invalid")
    return friction.reshape(env.num_envs, 8)


four_foot_effective_static_dynamic_friction = critic_effective_foot_friction


def _critic_body_masses(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    masses = getattr(env, "_g009_r0_body_mass", None)
    valid = bool(getattr(env, "_g009_r0_body_mass_valid", False))
    if masses is not None and valid:
        return masses.to(device=asset.device, dtype=asset.data.root_quat_w.dtype)
    if not hasattr(env, "observation_manager"):
        return asset.root_physx_view.get_masses().to(device=asset.device)
    raise RuntimeError("G009 R0 critic mass cache is unavailable after startup")


def critic_whole_body_com_b(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return the mass-weighted whole-body COM expressed in the base frame."""
    asset = _asset(env, asset_cfg)
    masses = _critic_body_masses(env, asset)
    com_w = torch.sum(asset.data.body_com_pos_w * masses.unsqueeze(-1), dim=1) / masses.sum(dim=1, keepdim=True)
    relative_w = com_w - asset.data.root_pos_w
    return math_utils.quat_apply_inverse(asset.data.root_quat_w, relative_w)


whole_body_com_base = critic_whole_body_com_b


def critic_total_mass(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return the initialization-read R0 mass cache as one critic value."""
    asset = _asset(env, asset_cfg)
    return _critic_body_masses(env, asset).sum(dim=1, keepdim=True)


total_mass = critic_total_mass


def critic_zero_external_wrench(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """R0 placeholder for the later three-axis disturbance-wrench channel."""
    asset = _asset(env, asset_cfg)
    return torch.zeros((env.num_envs, 3), device=asset.device, dtype=asset.data.root_quat_w.dtype)


commanded_wrench = critic_zero_external_wrench


def critic_zero_disturbance_pulse(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """R0 placeholder for the later disturbance pulse indicator."""
    asset = _asset(env, asset_cfg)
    return torch.zeros((env.num_envs, 1), device=asset.device, dtype=asset.data.root_quat_w.dtype)


normalized_pulse_time_remaining = critic_zero_disturbance_pulse


def critic_source_fall_one_hot(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return reset-time fall class, or lazy zeros before the first reset event."""
    asset = _asset(env, asset_cfg)
    source = getattr(env, "_g009_recover_fall_class", None)
    if source is None:
        return torch.zeros((env.num_envs, 4), device=asset.device, dtype=asset.data.root_quat_w.dtype)
    source = source.to(device=asset.device, dtype=asset.data.root_quat_w.dtype)
    if source.shape == (env.num_envs,):
        return torch.nn.functional.one_hot(source.long(), num_classes=4).to(asset.data.root_quat_w.dtype)
    if source.shape != (env.num_envs, 4):
        raise ValueError("_g009_recover_fall_class must have shape (num_envs,) or (num_envs, 4)")
    return source


source_fall_class_one_hot = critic_source_fall_one_hot


def _stable_predicate(
    env: ManagerBasedRLEnv,
    *,
    upright_threshold: float,
    min_base_height: float,
    max_base_height: float,
    min_contacts: int,
    max_linear_speed: float,
    max_angular_speed: float,
    contact_force_threshold: float,
    non_foot_contact_force_threshold: float,
    min_total_foot_support_ratio: float,
    nominal_total_mass_kg: float,
    gravity_magnitude: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    non_foot_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    support_normal_w = _support_normal_w(env, asset)
    alignment = _upright_alignment(asset.data.root_quat_w, support_normal_w)
    foot_forces_w = _sensor(env, sensor_cfg).data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_normal_load = _positive_support_normal_load(foot_forces_w, support_normal_w)
    contacts = (foot_normal_load >= contact_force_threshold).sum(dim=-1)
    non_foot_contacts = (
        _foot_force_magnitude(env, non_foot_sensor_cfg) >= non_foot_contact_force_threshold
    ).sum(dim=-1)
    min_total_foot_normal_load = (
        float(nominal_total_mass_kg) * float(gravity_magnitude) * float(min_total_foot_support_ratio)
    )
    return _stable_mask_from_values(
        alignment,
        asset.data.root_pos_w[:, 2],
        contacts,
        foot_normal_load.sum(dim=-1),
        non_foot_contacts,
        torch.linalg.vector_norm(asset.data.root_lin_vel_b, dim=-1),
        torch.linalg.vector_norm(asset.data.root_ang_vel_b, dim=-1),
        upright_threshold=upright_threshold,
        min_base_height=min_base_height,
        max_base_height=max_base_height,
        min_contacts=min_contacts,
        min_total_foot_normal_load=min_total_foot_normal_load,
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )


class _PotentialProgress(ManagerTermBase):
    """Stateful potential difference with reset-safe first samples."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._previous.zero_()
            self._initialized.fill_(True)
        else:
            self._previous[env_ids] = 0.0
            self._initialized[env_ids] = True

    def _difference(
        self,
        current: torch.Tensor,
        scale: float,
        step_dt: float,
        discount_factor: float,
    ) -> torch.Tensor:
        if step_dt <= 0.0:
            raise ValueError("step_dt must be positive")
        terminal = getattr(self._env, "reset_buf", None)
        terminal_potential = (
            torch.where(terminal.to(dtype=torch.bool), torch.zeros_like(current), current)
            if terminal is not None
            else current
        )
        reward = torch.where(
            self._initialized,
            _progress_delta(
                terminal_potential,
                self._previous,
                scale,
                discount_factor,
            )
            / step_dt,
            0.0,
        )
        self._previous.copy_(current)
        self._initialized.fill_(True)
        return reward


class UprightProgress(_PotentialProgress):
    """Reward positive change in body-up/support-normal alignment."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        discount_factor: float,
        scale: float = 1.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset = _asset(env, asset_cfg)
        return self._difference(
            _upright_alignment(asset.data.root_quat_w, _support_normal_w(env, asset)),
            scale,
            env.step_dt,
            discount_factor,
        )


class BaseHeightProgress(_PotentialProgress):
    """Reward positive change in base height, optionally capped at a target."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_height: float,
        discount_factor: float,
        scale: float = 1.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        height = _asset(env, asset_cfg).data.root_pos_w[:, 2].clamp_max(target_height)
        return self._difference(height, scale, env.step_dt, discount_factor)


class GatedBaseHeightProgress(_PotentialProgress):
    """Reward normalized height progress only as the base becomes upright."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_height: float,
        target_height: float,
        orientation_gate_start: float,
        orientation_gate_full: float,
        discount_factor: float,
        scale: float = 1.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset = _asset(env, asset_cfg)
        alignment = _upright_alignment(
            asset.data.root_quat_w, _support_normal_w(env, asset)
        )
        potential = _gated_height_potential_from_values(
            alignment,
            asset.data.root_pos_w[:, 2],
            min_height=min_height,
            target_height=target_height,
            orientation_gate_start=orientation_gate_start,
            orientation_gate_full=orientation_gate_full,
        )
        return self._difference(potential, scale, env.step_dt, discount_factor)


class SoftStandProgress(_PotentialProgress):
    """Reward continuous progress toward upright loaded-foot support."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_height: float,
        target_height: float,
        contact_force_threshold: float,
        min_contacts: int,
        min_total_foot_support_ratio: float,
        nominal_total_mass_kg: float,
        gravity_magnitude: float,
        discount_factor: float,
        scale: float = 1.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    ) -> torch.Tensor:
        asset = _asset(env, asset_cfg)
        support_normal_w = _support_normal_w(env, asset)
        alignment = _upright_alignment(asset.data.root_quat_w, support_normal_w)
        foot_forces_w = _sensor(env, sensor_cfg).data.net_forces_w[
            :, sensor_cfg.body_ids, :
        ]
        foot_normal_load = _positive_support_normal_load(
            foot_forces_w, support_normal_w
        )
        potential = _soft_stand_potential_from_values(
            alignment,
            asset.data.root_pos_w[:, 2],
            foot_normal_load,
            min_height=min_height,
            target_height=target_height,
            contact_force_threshold=contact_force_threshold,
            min_contacts=min_contacts,
            target_total_foot_load=(
                nominal_total_mass_kg
                * gravity_magnitude
                * min_total_foot_support_ratio
            ),
        )
        return self._difference(potential, scale, env.step_dt, discount_factor)


def stable_support(
    env: ManagerBasedRLEnv,
    upright_threshold: float,
    min_base_height: float,
    max_base_height: float,
    min_contacts: int,
    max_linear_speed: float,
    max_angular_speed: float,
    contact_force_threshold: float,
    non_foot_contact_force_threshold: float,
    min_total_foot_support_ratio: float,
    nominal_total_mass_kg: float,
    gravity_magnitude: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    non_foot_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return one only while the common stable predicate is satisfied."""
    return _stable_predicate(
        env,
        upright_threshold=upright_threshold,
        min_base_height=min_base_height,
        max_base_height=max_base_height,
        min_contacts=min_contacts,
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
        contact_force_threshold=contact_force_threshold,
        non_foot_contact_force_threshold=non_foot_contact_force_threshold,
        min_total_foot_support_ratio=min_total_foot_support_ratio,
        nominal_total_mass_kg=nominal_total_mass_kg,
        gravity_magnitude=gravity_magnitude,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
        non_foot_sensor_cfg=non_foot_sensor_cfg,
    ).float()


def upright_hold(
    env: ManagerBasedRLEnv,
    upright_threshold: float,
    min_base_height: float,
    max_base_height: float,
    min_contacts: int,
    max_linear_speed: float,
    max_angular_speed: float,
    contact_force_threshold: float,
    non_foot_contact_force_threshold: float,
    min_total_foot_support_ratio: float,
    nominal_total_mass_kg: float,
    gravity_magnitude: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    non_foot_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Hold reward weighted by current upright alignment after stabilization."""
    asset = _asset(env, asset_cfg)
    alignment = _upright_alignment(asset.data.root_quat_w, _support_normal_w(env, asset)).clamp_min(0.0)
    stable = _stable_predicate(
        env,
        upright_threshold=upright_threshold,
        min_base_height=min_base_height,
        max_base_height=max_base_height,
        min_contacts=min_contacts,
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
        contact_force_threshold=contact_force_threshold,
        non_foot_contact_force_threshold=non_foot_contact_force_threshold,
        min_total_foot_support_ratio=min_total_foot_support_ratio,
        nominal_total_mass_kg=nominal_total_mass_kg,
        gravity_magnitude=gravity_magnitude,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
        non_foot_sensor_cfg=non_foot_sensor_cfg,
    )
    return alignment * stable.float()


class StableSuccess(ManagerTermBase):
    """Terminate after the stable predicate holds for consecutive control steps."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._latched = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._counter.zero_()
            self._latched.zero_()
        else:
            self._counter[env_ids] = 0
            self._latched[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        required_consecutive_steps: int,
        upright_threshold: float,
        min_base_height: float,
        max_base_height: float,
        min_contacts: int,
        max_linear_speed: float,
        max_angular_speed: float,
        contact_force_threshold: float,
        non_foot_contact_force_threshold: float,
        min_total_foot_support_ratio: float,
        nominal_total_mass_kg: float,
        gravity_magnitude: float,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        non_foot_sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        if required_consecutive_steps <= 0:
            raise ValueError("required_consecutive_steps must be positive")
        stable = _stable_predicate(
            env,
            upright_threshold=upright_threshold,
            min_base_height=min_base_height,
            max_base_height=max_base_height,
            min_contacts=min_contacts,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            contact_force_threshold=contact_force_threshold,
            non_foot_contact_force_threshold=non_foot_contact_force_threshold,
            min_total_foot_support_ratio=min_total_foot_support_ratio,
            nominal_total_mass_kg=nominal_total_mass_kg,
            gravity_magnitude=gravity_magnitude,
            asset_cfg=asset_cfg,
            sensor_cfg=sensor_cfg,
            non_foot_sensor_cfg=non_foot_sensor_cfg,
        )
        self._counter = torch.where(stable, self._counter + 1, torch.zeros_like(self._counter))
        trigger = (self._counter >= required_consecutive_steps) & ~self._latched
        self._latched |= trigger
        return trigger


def stable_success_once(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a time-normalized pulse whose manager contribution is exactly its weight."""
    if env.step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    return env.termination_manager.get_term("stable_success").float() / env.step_dt


def full_angular_velocity_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize all three base angular-velocity axes."""
    return torch.sum(torch.square(_asset(env, asset_cfg).data.root_ang_vel_b), dim=-1)


def _state_gated_regularization_multiplier(
    env: ManagerBasedRLEnv,
    *,
    fallen_multiplier: float,
    orientation_gate_start: float,
    orientation_gate_full: float,
    height_gate_start: float,
    height_gate_full: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    if not 0.0 <= fallen_multiplier <= 1.0:
        raise ValueError("fallen_multiplier must be in [0, 1]")
    asset = _asset(env, asset_cfg)
    alignment = _upright_alignment(
        asset.data.root_quat_w, _support_normal_w(env, asset)
    )
    stand_gate = _stand_regularization_gate_from_values(
        alignment,
        asset.data.root_pos_w[:, 2],
        orientation_gate_start=orientation_gate_start,
        orientation_gate_full=orientation_gate_full,
        height_gate_start=height_gate_start,
        height_gate_full=height_gate_full,
    )
    return fallen_multiplier + (1.0 - fallen_multiplier) * stand_gate


def gated_angvel_l2(
    env: ManagerBasedRLEnv,
    fallen_multiplier: float,
    orientation_gate_start: float,
    orientation_gate_full: float,
    height_gate_start: float,
    height_gate_full: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Relax angular-velocity cost while fallen and restore it near standing."""
    return full_angular_velocity_l2(env, asset_cfg) * _state_gated_regularization_multiplier(
        env,
        fallen_multiplier=fallen_multiplier,
        orientation_gate_start=orientation_gate_start,
        orientation_gate_full=orientation_gate_full,
        height_gate_start=height_gate_start,
        height_gate_full=height_gate_full,
        asset_cfg=asset_cfg,
    )


def gated_action_rate_l2(
    env: ManagerBasedRLEnv,
    fallen_multiplier: float,
    orientation_gate_start: float,
    orientation_gate_full: float,
    height_gate_start: float,
    height_gate_full: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Relax raw action-rate cost while fallen and restore it near standing."""
    action_rate = torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1
    )
    return action_rate * _state_gated_regularization_multiplier(
        env,
        fallen_multiplier=fallen_multiplier,
        orientation_gate_start=orientation_gate_start,
        orientation_gate_full=orientation_gate_full,
        height_gate_start=height_gate_start,
        height_gate_full=height_gate_full,
        asset_cfg=asset_cfg,
    )


def mechanical_power_proxy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Sum absolute joint mechanical power ``|torque * velocity|``."""
    asset = _asset(env, asset_cfg)
    return torch.sum(
        torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]),
        dim=-1,
    )


def undesired_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float,
    min_base_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-foot contact only after the base reaches standing height.

    Body and leg contact are necessary support and pivot points during
    self-righting.  Once the base reaches the success-height region, renewed
    non-foot contact is treated as an undesired collision.
    """
    if min_base_height <= 0.0:
        raise ValueError("min_base_height must be positive")
    contact_count = (
        _foot_force_magnitude(env, sensor_cfg) > contact_force_threshold
    ).sum(dim=-1).float()
    base_is_raised = _asset(env, asset_cfg).data.root_pos_w[:, 2] >= min_base_height
    return contact_count * base_is_raised.float()


def numeric_invalid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate on non-finite root, joint, velocity, or torque state."""
    asset = _asset(env, asset_cfg)
    values = (
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.root_lin_vel_b,
        asset.data.root_ang_vel_b,
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.joint_vel[:, asset_cfg.joint_ids],
        asset.data.applied_torque[:, asset_cfg.joint_ids],
    )
    finite_per_value = [torch.isfinite(value).reshape(env.num_envs, -1).all(dim=-1) for value in values]
    invalid = torch.stack(finite_per_value).all(dim=0).logical_not()
    actor_invalid = getattr(env, "_g009_actor_signal_invalid", None)
    if actor_invalid is not None:
        invalid |= actor_invalid.to(device=invalid.device, dtype=torch.bool)
    return invalid


def hard_joint_limit_violation(
    env: ManagerBasedRLEnv,
    margin: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any resolved joint crosses its hard PhysX position limit."""
    asset = _asset(env, asset_cfg)
    position = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]
    return ((position < limits[..., 0] - margin) | (position > limits[..., 1] + margin)).any(dim=-1)


urdf_hard_joint_limit_violation = hard_joint_limit_violation


__all__ = [
    "BaseHeightProgress",
    "GatedBaseHeightProgress",
    "SoftStandProgress",
    "StableSuccess",
    "UprightProgress",
    "base_height_gt",
    "body_fixed_range",
    "body_fixed_range_hit_mask",
    "commanded_wrench",
    "critic_base_height_gt",
    "critic_effective_foot_friction",
    "critic_source_fall_one_hot",
    "critic_terrain_normal_gt_b",
    "critic_total_mass",
    "critic_whole_body_com_b",
    "critic_zero_disturbance_pulse",
    "critic_zero_external_wrench",
    "foot_contact_flags",
    "four_foot_contact_state",
    "four_foot_effective_static_dynamic_friction",
    "full_angular_velocity_l2",
    "gated_action_rate_l2",
    "gated_angvel_l2",
    "hard_joint_limit_violation",
    "mechanical_power_proxy",
    "normalized_foot_load",
    "normalized_four_foot_load",
    "normalized_pulse_time_remaining",
    "numeric_invalid",
    "stable_success_once",
    "stable_support",
    "source_fall_class_one_hot",
    "terrain_normal_gt",
    "total_mass",
    "undesired_collision",
    "upright_hold",
    "urdf_hard_joint_limit_violation",
    "whole_body_com_base",
]
