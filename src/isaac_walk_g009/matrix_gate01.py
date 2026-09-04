"""Whole-body terrain-contact matrix policy projection for G009 Gate01."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .matrix_observation_adapter import (
    MatrixObservationSourceDTypeError,
    MatrixObservationSourceMissingError,
    MatrixObservationSourceShapeError,
    MatrixObservationSourceTypeError,
    adapt_terrain_pair_force_matrix_w,
)
from .recover_contracts import GRAVITY_MAGNITUDE_M_S2, NOMINAL_TOTAL_MASS_KG


TERRAIN_FILTER_PATHS = ("/World/ground/terrain/GroundPlane/CollisionPlane",)
ORDERED_BODY_NAMES = (
    "base", "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
    "FR_hip", "FR_thigh", "FR_calf", "FR_foot", "Head_upper", "Head_lower",
    "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
    "RR_hip", "RR_thigh", "RR_calf", "RR_foot",
)
ORDERED_BODY_NAMES_SHA256 = hashlib.sha256(
    json.dumps(ORDERED_BODY_NAMES, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()
NOMINAL_BODY_WEIGHT_N = NOMINAL_TOTAL_MASS_KG * GRAVITY_MAGNITUDE_M_S2
MATRIX_BODY_COUNT = len(ORDERED_BODY_NAMES)
MATRIX_COMPONENT_COUNT = 3
MATRIX_OBSERVATION_DIM = MATRIX_BODY_COUNT * MATRIX_COMPONENT_COUNT
MATRIX_POLICY_OBSERVATION_DIM = 83 + MATRIX_OBSERVATION_DIM
MATRIX_CRITIC_OBSERVATION_DIM = MATRIX_POLICY_OBSERVATION_DIM + 24

_RUNTIME: dict[str, Any] = {}


def reset_runtime_telemetry() -> None:
    _RUNTIME.clear()
    _RUNTIME.update(
        {
            "call_count": 0,
            "all_source_finite": True,
            "all_output_finite": True,
            "source_unchanged": True,
            "positive_magnitude_count": 0,
            "nonzero_output_count": 0,
            "maximum_magnitude_n": 0.0,
            "output_minimum": 1.0,
            "output_maximum": -1.0,
            "output_variance_maximum": 0.0,
            "ordered_body_names": None,
            "ordered_body_names_sha256": None,
            "body_order_consistent": True,
            "live_contract": None,
            "source_shapes": set(),
            "output_shapes": set(),
            "source_dtypes": set(),
            "source_devices": set(),
        }
    )


def runtime_telemetry() -> dict[str, Any]:
    return {
        **{key: value for key, value in _RUNTIME.items() if not isinstance(value, set)},
        "source_shapes": sorted(_RUNTIME["source_shapes"]),
        "output_shapes": sorted(_RUNTIME["output_shapes"]),
        "source_dtypes": sorted(_RUNTIME["source_dtypes"]),
        "source_devices": sorted(_RUNTIME["source_devices"]),
    }


def _body_order_sha256(names: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _live_contract_readback(env, asset) -> dict[str, Any]:
    import omni.usd
    from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()
    solver_rows = []
    for prim_path in asset.root_physx_view.prim_paths:
        api = PhysxSchema.PhysxArticulationAPI(stage.GetPrimAtPath(prim_path))
        solver_rows.append(
            (
                int(api.GetSolverPositionIterationCountAttr().Get()),
                int(api.GetSolverVelocityIterationCountAttr().Get()),
            )
        )
    depenetration = []
    for paths in asset.root_physx_view.link_paths:
        for path in paths:
            value = PhysxSchema.PhysxRigidBodyAPI(
                stage.GetPrimAtPath(path)
            ).GetMaxDepenetrationVelocityAttr().Get()
            depenetration.append(float(value))
    action = env.action_manager.get_term("joint_pos").cfg
    return {
        "solver_position_velocity": [list(value) for value in solver_rows],
        "max_depenetration_velocity_m_s": depenetration,
        "action_scale": float(action.scale),
        "action_ema_alpha": float(action.alpha),
    }


def _policy_world_xyz_without_host_sync(source: object | None) -> torch.Tensor:
    """Validate static tensor metadata and reduce the filter axis without CUDA sync."""

    if source is None:
        raise MatrixObservationSourceMissingError("terrain_pair_force_matrix_w is required")
    if not isinstance(source, torch.Tensor):
        raise MatrixObservationSourceTypeError("terrain_pair_force_matrix_w must be a torch.Tensor")
    if source.ndim != 4 or tuple(source.shape[1:]) != (MATRIX_BODY_COUNT, 1, 3):
        raise MatrixObservationSourceShapeError(
            f"terrain_pair_force_matrix_w must have shape [N, 19, 1, 3], got {tuple(source.shape)}"
        )
    if source.dtype is not torch.float32:
        raise MatrixObservationSourceDTypeError(
            f"terrain_pair_force_matrix_w must have dtype torch.float32, got {source.dtype}"
        )
    return source.sum(dim=2)


def _sanitize_production_projection(
    env, source: torch.Tensor, flattened: torch.Tensor
) -> torch.Tensor:
    """Flag invalid rows on-device and return a finite, action-safe projection."""

    source_invalid = ~torch.isfinite(source).reshape(source.shape[0], -1).all(dim=-1)
    output_invalid = ~torch.isfinite(flattened).all(dim=-1)
    invalid = source_invalid | output_invalid
    buffer = getattr(env, "_g009_actor_signal_invalid", None)
    if buffer is None or buffer.shape != invalid.shape:
        buffer = torch.zeros_like(invalid, dtype=torch.bool)
        env._g009_actor_signal_invalid = buffer
    buffer |= invalid.to(device=buffer.device, dtype=torch.bool)
    sanitized = torch.nan_to_num(flattened, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(invalid.unsqueeze(-1), torch.zeros_like(sanitized), sanitized)


def whole_body_terrain_contact_matrix_base_normalized(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    collect_gate_telemetry: bool = False,
) -> torch.Tensor:
    """Return a bounded 57-D base-frame projection of the raw world forces.

    The raw adapter stays world-frame and read-only. This policy-only mapping
    rotates each vector into the base frame, divides by nominal body weight,
    then applies ``tanh``. It is deterministic, near-linear around zero, and
    bounds rare collision-force outliers to ``[-1, 1]``.
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    body_names = tuple(sensor.body_names)
    if body_names != ORDERED_BODY_NAMES:
        raise ValueError(
            "whole-body contact sensor order mismatch: "
            f"expected={ORDERED_BODY_NAMES!r} actual={body_names!r}"
        )
    source = sensor.data.force_matrix_w
    source_version = source._version if isinstance(source, torch.Tensor) else None
    observation = adapt_terrain_pair_force_matrix_w(source) if collect_gate_telemetry else None
    world_xyz = (
        observation.world_xyz
        if observation is not None
        else _policy_world_xyz_without_host_sync(source)
    )
    root_quat = asset.data.root_quat_w[:, None, :].expand(-1, MATRIX_BODY_COUNT, -1)
    base_xyz = math_utils.quat_apply_inverse(root_quat, world_xyz)
    bounded = torch.tanh(base_xyz / NOMINAL_BODY_WEIGHT_N)
    flattened = bounded.reshape(bounded.shape[0], MATRIX_OBSERVATION_DIM)

    if not collect_gate_telemetry:
        return _sanitize_production_projection(env, source, flattened)

    assert observation is not None
    body_hash = _body_order_sha256(body_names)
    previous_names = _RUNTIME["ordered_body_names"]
    if _RUNTIME["live_contract"] is None:
        _RUNTIME["live_contract"] = _live_contract_readback(env, asset)
    _RUNTIME["call_count"] += 1
    _RUNTIME["all_source_finite"] &= bool(torch.isfinite(source).all().item())
    _RUNTIME["all_output_finite"] &= bool(torch.isfinite(flattened).all().item())
    _RUNTIME["source_unchanged"] &= source._version == source_version
    _RUNTIME["positive_magnitude_count"] += int(observation.contact_mask.count_nonzero().item())
    _RUNTIME["nonzero_output_count"] += int(flattened.count_nonzero().item())
    _RUNTIME["maximum_magnitude_n"] = max(
        float(_RUNTIME["maximum_magnitude_n"]), float(observation.magnitude.max().item())
    )
    _RUNTIME["output_minimum"] = min(float(_RUNTIME["output_minimum"]), float(flattened.min().item()))
    _RUNTIME["output_maximum"] = max(float(_RUNTIME["output_maximum"]), float(flattened.max().item()))
    _RUNTIME["output_variance_maximum"] = max(
        float(_RUNTIME["output_variance_maximum"]), float(flattened.var(unbiased=False).item())
    )
    _RUNTIME["ordered_body_names"] = list(body_names)
    _RUNTIME["ordered_body_names_sha256"] = body_hash
    _RUNTIME["body_order_consistent"] &= previous_names is None or tuple(previous_names) == body_names
    _RUNTIME["source_shapes"].add("x".join(str(value) for value in source.shape))
    _RUNTIME["output_shapes"].add("x".join(str(value) for value in flattened.shape))
    _RUNTIME["source_dtypes"].add(str(source.dtype))
    _RUNTIME["source_devices"].add(str(source.device))
    return flattened


reset_runtime_telemetry()


__all__ = [
    "MATRIX_CRITIC_OBSERVATION_DIM",
    "MATRIX_OBSERVATION_DIM",
    "MATRIX_POLICY_OBSERVATION_DIM",
    "NOMINAL_BODY_WEIGHT_N",
    "ORDERED_BODY_NAMES",
    "ORDERED_BODY_NAMES_SHA256",
    "TERRAIN_FILTER_PATHS",
    "reset_runtime_telemetry",
    "runtime_telemetry",
    "whole_body_terrain_contact_matrix_base_normalized",
]
