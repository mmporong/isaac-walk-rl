#!/usr/bin/env python3
"""Capture rev16 CPU/GPU backend-divergence telemetry for one controlled cell.

This is a diagnostic-only A/B probe.  It cannot launch PPO or either G009
qualification gate, and a complete diagnostic report is not a qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPT_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import probe_g009_recover_runtime as runtime_probe

DEFAULT_TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
NUM_ENVS = 8
SOURCE_ENV_INDEX = 7
POSE_ID = "right_side"
ACTION_MODE = "reset_pose_hold"
ROLLOUT_STEPS = 150
PHYSICS_DT_S = 0.005
CONTROL_DECIMATION = 4
VELOCITY_SOLVER_ITERATIONS = 0
MAX_DEPENETRATION_VELOCITY_M_S = 1.0
CALLBACK_DT_ABS_TOLERANCE_S = 2.5e-10
MASS_ACCUMULATION_CONTRACT = {
    "component_source": "event_readback:_g009_r0_body_mass",
    "component_storage_dtype": "torch.float32",
    "serialized_component_dtype": "python.float",
    "canonical_sum_method": "math.fsum(serialized_float32_components)",
    "runtime_native_reduction": "not_serialized_and_not_used",
    "decision_uses_native_total": False,
}
MASS_EVIDENCE_FIELDS = (
    "mass_accumulation",
    "body_mass_kg",
    "all_env_body_mass_kg",
    "total_mass_kg",
    "all_env_total_mass_kg",
    "body_weight_n",
)
CLOCK_EVIDENCE_FIELDS = (
    "source",
    "evidence_kind",
    "contract_expected_dt_s",
    "callback_expected_float32_dt_s",
    "callback_dt_abs_tolerance_s",
    "observed_dt_min_s",
    "observed_dt_max_s",
    "max_abs_error_s",
    "mismatch_count",
    "nonfinite_count",
    "first_mismatch",
    "callback_count",
    "expected_callback_count",
    "passed",
)
ARM_POSITION_ITERATIONS = {"A": 8, "B": 16}
HISTORICAL_REFERENCES = {
    "A": {
        "contract_id": "g009_r0_recover_rev12",
        "canonical_sha256": (
            "d4b48d2b5fc1ea7684684a6324ba22fbfae767effeae45668c7310df382392e0"
        ),
        "role": "accepted_baseline_parameter_reference",
    },
    "B": {
        "contract_id": "g009_r0_recover_rev15",
        "canonical_sha256": (
            "5f29ba19458404b5009d3734294c57e79294efecc7fe03bf8c71c71656129832"
        ),
        "role": "rejected_comparison_parameter_reference",
    },
}
HISTORICAL_REPORTS = {
    ("A", "cpu"): {
        "path": "reports/runs/g009_r0_runtime_probe_rev12_cpu_rep01_s42.json",
        "sha256": "fb8bad2190389c3e964d1807a0f54ea700ddfd6919765105c04b93bfa8c7dd75",
    },
    ("A", "cuda:0"): {
        "path": "reports/runs/g009_r0_runtime_probe_rev12_gpu_rep01_s42.json",
        "sha256": "e485a3fcab5d8f8e6a793d30f76fb0a3ce346e27ed89a158409862e3e32414d1",
    },
    ("B", "cpu"): {
        "path": "reports/runs/g009_r0_runtime_probe_rev15_cpu_rep01_s42.json",
        "sha256": "426f4fe1085aeddad52c77d98fc74a55907dcc90d7084ebe8b4fde736b60e9d5",
    },
    ("B", "cuda:0"): {
        "path": "reports/runs/g009_r0_runtime_probe_rev15_gpu_rep01_s42.json",
        "sha256": "e24674a1ed33c38fbe5f12d19dc068167b9787e75323efbe55629bf059839b91",
    },
}
PREDECESSOR_REQUIREMENTS = {
    ("A", "cpu"): None,
    ("A", "cuda:0"): (3, "A.cuda:0"),
    ("B", "cpu"): (6, "B.cpu"),
    ("B", "cuda:0"): (9, "B.cuda:0"),
}
SOURCE_BINDING_PATHS = (
    "configs/g009_r0.json",
    "scripts/probe_g009_recover_runtime.py",
    "scripts/probe_g009_r0_rev16_backend_divergence.py",
    "scripts/summarize_g009_r0_rev16_backend_divergence.py",
    "reports/runs/g009_r0_runtime_probe_rev12_cpu_rep01_s42.json",
    "reports/runs/g009_r0_runtime_probe_rev12_gpu_rep01_s42.json",
    "reports/runs/g009_r0_runtime_probe_rev15_cpu_rep01_s42.json",
    "reports/runs/g009_r0_runtime_probe_rev15_gpu_rep01_s42.json",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/recover_contracts.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/registry.py",
)
_ENV_ROBOT_PATH = re.compile(r"/World/envs/env_(\d+)/Robot(?:/|$)")


def expected_pose_action_assignment() -> list[dict[str, Any]]:
    poses = ("prone", "supine", "left_side", "right_side")
    return [
        {
            "env_index": env_index,
            "class_id": env_index % 4,
            "pose_id": poses[env_index % 4],
            "action_mode": ("zero_normalized" if env_index < 4 else "reset_pose_hold"),
        }
        for env_index in range(NUM_ENVS)
    ]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rev16_contract(arm: str, device: str = "cuda:0") -> dict[str, Any]:
    """Return the immutable diagnostic contract for one A/B arm."""

    normalized = arm.upper()
    if normalized not in ARM_POSITION_ITERATIONS:
        raise ValueError("arm must be A or B")
    normalized_device = device.lower()
    if normalized_device not in {"cpu", "cuda:0"}:
        raise ValueError("device must be cpu or cuda:0")
    return {
        "schema_version": "g009.r0.rev16.backend_divergence_contract.v1",
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev16",
        "experiment": "cpu_gpu_backend_divergence",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "execution_conditions": {
            "task": DEFAULT_TASK,
            "seed": 42,
            "headless": True,
            "device": normalized_device,
            "runtime_device": normalized_device,
            "num_envs": NUM_ENVS,
        },
        "training_and_gate_policy": {
            "ppo_allowed": False,
            "gate01_allowed": False,
            "gate10_allowed": False,
        },
        "arm": {
            "id": normalized,
            "meaning": (
                "accepted_rev12_solver_baseline"
                if normalized == "A"
                else "position_solver_only_16"
            ),
            "articulation_solver_position_iteration_count": (
                ARM_POSITION_ITERATIONS[normalized]
            ),
            "articulation_solver_velocity_iteration_count": (
                VELOCITY_SOLVER_ITERATIONS
            ),
            "max_depenetration_velocity_m_s": MAX_DEPENETRATION_VELOCITY_M_S,
        },
        "historical_reference": {
            **HISTORICAL_REFERENCES[normalized],
            "reference_scope": "contract_identity_and_physics_tuple_only",
            "historical_checkpoint_loaded": False,
            "historical_training_resumed": False,
            "current_rev16_execution_is_fresh": True,
        },
        "controlled_cell": {
            "num_envs": NUM_ENVS,
            "source_env_index": SOURCE_ENV_INDEX,
            "pose_id": POSE_ID,
            "action_mode": ACTION_MODE,
            "assignment_mode": "stratified",
            "pose_xy_range_m": [0.0, 0.0],
            "yaw_range_rad": [0.0, 0.0],
        },
        "timing": {
            "physics_dt_s": PHYSICS_DT_S,
            "control_decimation": CONTROL_DECIMATION,
            "control_dt_s": PHYSICS_DT_S * CONTROL_DECIMATION,
            "control_steps": ROLLOUT_STEPS,
            "physics_steps": ROLLOUT_STEPS * CONTROL_DECIMATION,
        },
        "required_evidence": {
            "physics_substeps": [
                "contact_force_history_slot",
                "per_body_force_vector_n",
                "per_body_impulse_vector_n_s",
                "nonfoot_total_force_n",
                "nonfoot_impulse_n_s",
            ],
            "control_steps": [
                "root_state_w",
                "link_state_w",
                "joint_position_rad",
                "joint_velocity_rad_s",
                "applied_torque_nm",
                "input_action",
                "raw_action",
                "processed_ema_target_rad",
                "ema_previous_before_rad",
                "ema_previous_after_rad",
            ],
            "cpu_authority": [
                "robot_ground_contact_pair",
                "contact_position_w_m",
                "contact_normal",
                "contact_separation_m",
            ],
        },
    }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_bundle_provenance() -> dict[str, Any]:
    files = {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_BINDING_PATHS
        if (REPO_ROOT / path).is_file()
    }
    missing = [path for path in SOURCE_BINDING_PATHS if path not in files]
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *SOURCE_BINDING_PATHS,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "schema_version": 1,
        "git_commit": commit,
        "git_commit_valid": bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
        "source_binding_paths": list(SOURCE_BINDING_PATHS),
        "source_binding_files": files,
        "source_bundle_sha256": (
            hashlib.sha256(payload.encode("utf-8")).hexdigest() if files else None
        ),
        "all_files_present": not missing and len(files) == len(SOURCE_BINDING_PATHS),
        "missing_files": missing,
        "clean": not status,
        "dirty_source_paths": status,
    }


def _vector3(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        result = [float(value[index]) for index in range(3)]
    except (IndexError, TypeError, ValueError):
        try:
            result = [float(value.x), float(value.y), float(value.z)]
        except (AttributeError, TypeError, ValueError):
            return None
    return result if all(math.isfinite(item) for item in result) else None


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _is_exact_float32(value: float) -> bool:
    try:
        return _float32(value) == value
    except (OverflowError, struct.error):
        return False


class PhysicsStepClock:
    """Count pre-step notifications and audit their C++ float32 dt values."""

    def __init__(self, contract_expected_dt_s: float) -> None:
        require(
            math.isfinite(contract_expected_dt_s) and contract_expected_dt_s > 0.0,
            "physics dt contract must be finite and positive",
        )
        self.contract_expected_dt_s = float(contract_expected_dt_s)
        self.callback_expected_float32_dt_s = _float32(self.contract_expected_dt_s)
        self.callback_dt_abs_tolerance_s = CALLBACK_DT_ABS_TOLERANCE_S
        self.current_step = 0
        self.dt_mismatch_count = 0
        self.nonfinite_count = 0
        self.observed_dt_min_s: float | None = None
        self.observed_dt_max_s: float | None = None
        self.max_abs_error_s = 0.0
        self.first_mismatch: dict[str, Any] | None = None

    def __call__(self, dt_s: float) -> None:
        self.current_step += 1
        observed = float(dt_s)
        if not math.isfinite(observed):
            self.nonfinite_count += 1
            self.dt_mismatch_count += 1
            if self.first_mismatch is None:
                self.first_mismatch = {
                    "callback_index": self.current_step,
                    "observed_dt_s": None,
                    "abs_error_s": None,
                    "reason": "nonfinite",
                }
            return
        self.observed_dt_min_s = (
            observed
            if self.observed_dt_min_s is None
            else min(self.observed_dt_min_s, observed)
        )
        self.observed_dt_max_s = (
            observed
            if self.observed_dt_max_s is None
            else max(self.observed_dt_max_s, observed)
        )
        error = abs(observed - self.callback_expected_float32_dt_s)
        self.max_abs_error_s = max(self.max_abs_error_s, error)
        if error > self.callback_dt_abs_tolerance_s:
            self.dt_mismatch_count += 1
            if self.first_mismatch is None:
                self.first_mismatch = {
                    "callback_index": self.current_step,
                    "observed_dt_s": observed,
                    "abs_error_s": error,
                    "reason": "outside_float32_boundary_tolerance",
                }

    def snapshot(self) -> dict[str, Any]:
        return {
            "source": "subscribe_physics_on_step_events(pre_step=true,order=0)",
            "evidence_kind": "pre_step_notification_count",
            "contract_expected_dt_s": self.contract_expected_dt_s,
            "callback_expected_float32_dt_s": self.callback_expected_float32_dt_s,
            "callback_dt_abs_tolerance_s": self.callback_dt_abs_tolerance_s,
            "observed_dt_min_s": self.observed_dt_min_s,
            "observed_dt_max_s": self.observed_dt_max_s,
            "max_abs_error_s": self.max_abs_error_s,
            "mismatch_count": self.dt_mismatch_count,
            "nonfinite_count": self.nonfinite_count,
            "first_mismatch": self.first_mismatch,
            "callback_count": self.current_step,
            "expected_callback_count": ROLLOUT_STEPS * CONTROL_DECIMATION,
            "passed": self.current_step == ROLLOUT_STEPS * CONTROL_DECIMATION
            and self.dt_mismatch_count == 0
            and self.nonfinite_count == 0,
        }


class PhysicsStepClockEvidenceError(RuntimeError):
    """Carry partial clock evidence into the fail-closed report."""

    def __init__(
        self,
        evidence: dict[str, Any],
        mass_evidence: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("physics pre-step notification clock validation failed")
        self.evidence = evidence
        self.mass_evidence = mass_evidence


class MassEvidenceError(RuntimeError):
    """Carry canonical mass evidence while preserving the original failure."""

    def __init__(self, evidence: dict[str, Any], original_error: BaseException) -> None:
        super().__init__(str(original_error))
        self.evidence = evidence
        self.original_error = original_error


class DiagnosticEvidenceError(RuntimeError):
    """Preserve completed diagnostic preflight evidence on report rejection."""

    def __init__(
        self,
        original_error: BaseException,
        *,
        physics_step_clock: dict[str, Any],
        mass_evidence: dict[str, Any],
    ) -> None:
        super().__init__(str(original_error))
        self.original_error = original_error
        self.physics_step_clock = physics_step_clock
        self.mass_evidence = mass_evidence


def build_mass_evidence(body_mass_tensor: Any, source_env_index: int) -> dict[str, Any]:
    """Serialize native float32 components and compute canonical Python totals."""

    require(str(body_mass_tensor.dtype) == "torch.float32", "mass dtype changed")
    require(
        body_mass_tensor.ndim == 2
        and tuple(body_mass_tensor.shape) == (NUM_ENVS, 19)
        and 0 <= source_env_index < NUM_ENVS,
        "all-env body mass shape mismatch",
    )
    require(
        bool(body_mass_tensor.isfinite().all().item())
        and bool((body_mass_tensor > 0.0).all().item()),
        "runtime body mass readback is invalid",
    )
    all_components = [
        [float(value) for value in row]
        for row in body_mass_tensor.detach().cpu().tolist()
    ]
    canonical_totals = [math.fsum(row) for row in all_components]
    return {
        "mass_accumulation": dict(MASS_ACCUMULATION_CONTRACT),
        "body_mass_kg": all_components[source_env_index],
        "all_env_body_mass_kg": all_components,
        "total_mass_kg": canonical_totals[source_env_index],
        "all_env_total_mass_kg": canonical_totals,
        "body_weight_n": canonical_totals[source_env_index] * 9.81,
    }


def _allowlisted_evidence(
    evidence: dict[str, Any] | None, fields: tuple[str, ...]
) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return {field: evidence.get(field) for field in fields}


def _contact_event_name(value: Any, contact_event_types: Any | None) -> str:
    if contact_event_types is None or value is None:
        return "CONTACT_PERSIST"
    for name in ("CONTACT_FOUND", "CONTACT_PERSIST", "CONTACT_LOST"):
        if value == getattr(contact_event_types, name):
            return name
    raise ValueError(f"unsupported contact event type: {value}")


def extract_cpu_contact_points(
    contact_headers: Any,
    contact_data: Any,
    *,
    source_env_index: int,
    int_to_path: Any,
    contact_event_types: Any | None = None,
) -> dict[str, Any]:
    """Copy authoritative robot/ground contact pair, normal and separation."""

    points: list[dict[str, Any]] = []
    headers: list[dict[str, Any]] = []
    robot_ground_headers = 0
    all_env_minimum_separation_m: dict[int, float] = {}
    for header in contact_headers:
        paths = [
            str(int_to_path(header.actor0)),
            str(int_to_path(header.actor1)),
            str(int_to_path(header.collider0)),
            str(int_to_path(header.collider1)),
        ]
        if not any(path.startswith("/World/ground") for path in paths):
            continue
        match = next(
            (
                _ENV_ROBOT_PATH.search(path)
                for path in paths
                if _ENV_ROBOT_PATH.search(path)
            ),
            None,
        )
        if match is None:
            continue
        env_index = int(match.group(1))
        event_type = _contact_event_name(
            getattr(header, "type", None), contact_event_types
        )
        robot_ground_headers += 1
        start = int(header.contact_data_offset)
        end = start + int(header.num_contact_data)
        require(end >= start, "contact data range is invalid")
        header_points: list[dict[str, Any]] = []
        for datum in contact_data[start:end]:
            separation = float(datum.separation)
            position = _vector3(getattr(datum, "position", None))
            normal = _vector3(getattr(datum, "normal", None))
            reported_impulse = _vector3(getattr(datum, "impulse", None))
            point = {
                "actor0_path": paths[0],
                "actor1_path": paths[1],
                "collider0_path": paths[2],
                "collider1_path": paths[3],
                "contact_position_w_m": position,
                "contact_normal_w": normal,
                "reported_contact_impulse_n_s": reported_impulse,
                "separation_m": separation if math.isfinite(separation) else None,
            }
            header_points.append(point)
            if point["separation_m"] is not None:
                prior = all_env_minimum_separation_m.get(env_index, float("inf"))
                all_env_minimum_separation_m[env_index] = min(
                    prior, float(point["separation_m"])
                )
        require(
            event_type == "CONTACT_LOST" or bool(header_points),
            "FOUND/PERSIST contact header must include contact data",
        )
        require(
            event_type != "CONTACT_LOST" or not header_points,
            "LOST contact header must not fabricate contact data",
        )
        headers.append(
            {
                "env_index": env_index,
                "event_type": event_type,
                "actor0_path": paths[0],
                "actor1_path": paths[1],
                "collider0_path": paths[2],
                "collider1_path": paths[3],
                "contact_points": header_points,
            }
        )
        if env_index == source_env_index:
            points.extend(header_points)
    return {
        "robot_ground_header_count": robot_ground_headers,
        "headers": headers,
        "contact_points": points,
        "all_env_minimum_separation_m": all_env_minimum_separation_m,
        "complete": bool(headers)
        and all(
            point["contact_position_w_m"] is not None
            and point["contact_normal_w"] is not None
            and point["reported_contact_impulse_n_s"] is not None
            and point["separation_m"] is not None
            for point in points
        ),
    }


class CpuContactAuthorityAccumulator:
    """Retain callback data before the PhysX callback buffers expire."""

    def __init__(
        self,
        source_env_index: int,
        num_envs: int,
        int_to_path: Any,
        clock: PhysicsStepClock,
        contact_event_types: Any,
    ) -> None:
        self.source_env_index = source_env_index
        self.num_envs = num_envs
        self.int_to_path = int_to_path
        self.clock = clock
        self.contact_event_types = contact_event_types
        self.events: list[dict[str, Any]] = []
        self.error: str | None = None
        self.subsequent_error_count = 0
        self.callback_event_index = 0
        self.minimum_separation_m: list[float | None] = [None] * num_envs

    def mark_unavailable(self, error: BaseException) -> None:
        if self.error is None:
            self.error = f"{type(error).__name__}: {error}"
        else:
            self.subsequent_error_count += 1

    def __call__(self, contact_headers: Any, contact_data: Any) -> None:
        self.callback_event_index += 1
        try:
            event = extract_cpu_contact_points(
                contact_headers,
                contact_data,
                source_env_index=self.source_env_index,
                int_to_path=self.int_to_path,
                contact_event_types=self.contact_event_types,
            )
        except Exception as error:  # noqa: BLE001 - callback failures become evidence
            self.mark_unavailable(error)
            return
        for env_index, separation in event["all_env_minimum_separation_m"].items():
            prior = self.minimum_separation_m[env_index]
            self.minimum_separation_m[env_index] = (
                separation if prior is None else min(prior, separation)
            )
        target_headers = [
            header
            for header in event["headers"]
            if header["env_index"] == self.source_env_index
        ]
        if target_headers:
            self.events.append(
                {
                    "physics_step": self.clock.current_step,
                    "callback_event_index": self.callback_event_index,
                    "headers": target_headers,
                    "complete": all(
                        header["event_type"] == "CONTACT_LOST"
                        or all(
                            point["contact_position_w_m"] is not None
                            and point["contact_normal_w"] is not None
                            and point["reported_contact_impulse_n_s"] is not None
                            and point["separation_m"] is not None
                            for point in header["contact_points"]
                        )
                        for header in target_headers
                    ),
                }
            )

    def snapshot(self, device: str) -> dict[str, Any]:
        authoritative = device.strip().lower() == "cpu"
        complete = (
            authoritative
            and self.error is None
            and bool(self.events)
            and all(event["complete"] for event in self.events)
            and self.clock.snapshot()["passed"] is True
            and all(value is not None for value in self.minimum_separation_m)
        )
        return {
            "authority_device": "cpu",
            "this_run_is_authority": authoritative,
            "status": (
                "observed"
                if complete
                else "authority_unavailable"
                if authoritative
                else "unavailable_on_gpu"
            ),
            "data_available": complete if authoritative else False,
            "error": self.error,
            "subsequent_error_count": self.subsequent_error_count,
            "callback_event_count": self.callback_event_index,
            "physics_step_clock": self.clock.snapshot(),
            "events": self.events if authoritative else None,
            "all_env_minimum_separation_m": (
                self.minimum_separation_m if authoritative else None
            ),
            "passed": complete if authoritative else None,
        }


def physics_history_rows(
    force_history: Any,
    *,
    control_step: int,
    physics_dt_s: float,
    body_names: list[str],
    nonfoot_ids: list[int],
    foot_ids: list[int],
    base_body_id: int,
    total_mass_kg: float,
) -> list[dict[str, Any]]:
    """Serialize native newest-first sensor history with derived impulse."""

    if force_history.ndim != 3 or force_history.shape[0] != CONTROL_DECIMATION:
        raise ValueError("env force history must be [decimation, body, xyz]")
    if force_history.shape[1] != len(body_names) or force_history.shape[2] != 3:
        raise ValueError("force history body topology mismatch")
    if not bool(force_history.isfinite().all().item()):
        raise ValueError("force history contains non-finite telemetry")
    if not math.isfinite(total_mass_kg) or total_mass_kg <= 0.0:
        raise ValueError("total_mass_kg must be finite and positive")
    body_ids = set(range(len(body_names)))
    if base_body_id not in body_ids:
        raise ValueError("base body id is outside force topology")
    if set(nonfoot_ids) | set(foot_ids) != body_ids:
        raise ValueError("foot and non-foot ids must cover force topology")
    if set(nonfoot_ids) & set(foot_ids):
        raise ValueError("foot and non-foot ids must be disjoint")
    rows = []
    body_weight_n = total_mass_kg * 9.81
    for history_slot in range(CONTROL_DECIMATION):
        physics_step = (
            control_step * CONTROL_DECIMATION - history_slot
        )  # Isaac sensor history is newest first.
        forces = force_history[history_slot]
        impulses = forces * physics_dt_s
        force_magnitudes = forces.norm(dim=1)
        base_force_magnitude = force_magnitudes[base_body_id]
        foot_force_magnitude_sum = force_magnitudes[foot_ids].sum()
        nonfoot_force_vector = forces[nonfoot_ids].sum(dim=0)
        nonfoot_force_magnitude_sum = force_magnitudes[nonfoot_ids].sum()
        rows.append(
            {
                "physics_step": physics_step,
                "time_s": physics_step * physics_dt_s,
                "control_step": control_step,
                "contact_force_history_slot": history_slot,
                "history_slot_order": "newest_first",
                "body_names": body_names,
                "per_body_force_vector_n": forces.detach().cpu().tolist(),
                "per_body_force_magnitude_n": (
                    force_magnitudes.detach().cpu().tolist()
                ),
                "per_body_impulse_vector_n_s": impulses.detach().cpu().tolist(),
                "base_force_magnitude_n": float(base_force_magnitude.item()),
                "base_force_bodyweights": float(base_force_magnitude.item())
                / body_weight_n,
                "base_impulse_n_s": float((base_force_magnitude * physics_dt_s).item()),
                "foot_total_force_n": float(foot_force_magnitude_sum.item()),
                "foot_impulse_n_s": float(
                    (foot_force_magnitude_sum * physics_dt_s).item()
                ),
                "nonfoot_resultant_force_vector_n": (
                    nonfoot_force_vector.detach().cpu().tolist()
                ),
                "nonfoot_resultant_force_n": float(nonfoot_force_vector.norm().item()),
                "nonfoot_total_force_n": float(nonfoot_force_magnitude_sum.item()),
                "nonfoot_impulse_vector_n_s": (nonfoot_force_vector * physics_dt_s)
                .detach()
                .cpu()
                .tolist(),
                "nonfoot_impulse_n_s": float(
                    (nonfoot_force_magnitude_sum * physics_dt_s).item()
                ),
            }
        )
    return rows


def control_step_row(
    *,
    control_step: int,
    robot: Any,
    action_term: Any,
    action: Any,
    ema_previous_before: Any,
    source_env_index: int,
    termination_flags: dict[str, bool],
    control_dt_s: float,
) -> dict[str, Any]:
    link_field = (
        "body_link_state_w"
        if getattr(robot.data, "body_link_state_w", None) is not None
        else "body_state_w"
    )
    link_state = getattr(robot.data, link_field, None)
    if link_state is None:
        raise RuntimeError("robot link state tensor is unavailable")
    ema_after = getattr(action_term, "_prev_applied_actions", None)
    if ema_after is None:
        raise RuntimeError("EMA previous target tensor is unavailable")
    finite_tensors = (
        robot.data.root_state_w,
        link_state,
        robot.data.joint_pos,
        robot.data.joint_vel,
        robot.data.applied_torque,
        action,
        action_term.raw_actions,
        action_term.processed_actions,
        ema_previous_before,
        ema_after,
    )
    if not all(
        bool(value[source_env_index].isfinite().all().item())
        for value in finite_tensors
    ):
        raise RuntimeError(
            "control-step state/action telemetry contains non-finite values"
        )
    return {
        "control_step": control_step,
        "time_s": control_step * control_dt_s,
        "termination_flags": termination_flags,
        "root_state_w": robot.data.root_state_w[source_env_index]
        .detach()
        .cpu()
        .tolist(),
        "link_state_field": link_field,
        "link_names": list(robot.body_names),
        "link_state_w": link_state[source_env_index].detach().cpu().tolist(),
        "joint_names": list(robot.joint_names),
        "joint_position_rad": robot.data.joint_pos[source_env_index]
        .detach()
        .cpu()
        .tolist(),
        "joint_velocity_rad_s": robot.data.joint_vel[source_env_index]
        .detach()
        .cpu()
        .tolist(),
        "applied_torque_nm": robot.data.applied_torque[source_env_index]
        .detach()
        .cpu()
        .tolist(),
        "input_action": action[source_env_index].detach().cpu().tolist(),
        "raw_action": action_term.raw_actions[source_env_index].detach().cpu().tolist(),
        "processed_ema_target_rad": (
            action_term.processed_actions[source_env_index].detach().cpu().tolist()
        ),
        "ema_previous_before_rad": (
            ema_previous_before[source_env_index].detach().cpu().tolist()
        ),
        "ema_previous_after_rad": ema_after[source_env_index].detach().cpu().tolist(),
    }


def governance() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "qualification_eligible": False,
        "learned": False,
        "ppo": {"allowed": False, "status": "not_run"},
        "gate01": {"allowed": False, "status": "forbidden"},
        "gate10": {"allowed": False, "status": "forbidden"},
        "qualification": {"eligible": False, "status": "not_run", "passed": None},
    }


def _finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be a JSON number")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} shape mismatch")
    return [_finite_number(item, label) for item in value]


def _close(left: float, right: float, label: str, tolerance: float = 1.0e-6) -> None:
    require(
        math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance),
        f"{label} derivation mismatch",
    )


def validate_mass_evidence(topology: dict[str, Any]) -> None:
    retired_native_fields = {
        "native_total_mass_kg",
        "native_minus_canonical_kg",
        "all_env_native_total_mass_kg",
        "all_env_native_minus_canonical_kg",
    }
    require(
        set(MASS_EVIDENCE_FIELDS) <= set(topology)
        and not retired_native_fields & set(topology),
        "mass evidence fields mismatch",
    )
    require(
        topology.get("mass_accumulation") == MASS_ACCUMULATION_CONTRACT,
        "mass accumulation contract changed",
    )
    body_mass = _finite_vector(topology.get("body_mass_kg"), 19, "body mass")
    require(
        all(value > 0.0 and _is_exact_float32(value) for value in body_mass),
        "body mass must be positive exact float32 values",
    )
    all_components = topology.get("all_env_body_mass_kg")
    require(
        isinstance(all_components, list) and len(all_components) == NUM_ENVS,
        "all-env body mass shape mismatch",
    )
    all_components = cast(list[Any], all_components)
    all_components = [
        _finite_vector(row, 19, "all-env body mass") for row in all_components
    ]
    require(
        all(
            value > 0.0 and _is_exact_float32(value)
            for row in all_components
            for value in row
        ),
        "all-env body mass must be positive exact float32 values",
    )
    require(
        body_mass == all_components[SOURCE_ENV_INDEX],
        "controlled-cell body mass row mismatch",
    )
    canonical_totals = [math.fsum(row) for row in all_components]
    reported_canonical = _finite_vector(
        topology.get("all_env_total_mass_kg"), NUM_ENVS, "all-env canonical total"
    )
    for env_index, canonical in enumerate(canonical_totals):
        _close(
            reported_canonical[env_index],
            canonical,
            "all-env canonical total",
            1.0e-12,
        )
    total_mass = _finite_number(topology.get("total_mass_kg"), "total mass")
    _close(total_mass, canonical_totals[SOURCE_ENV_INDEX], "total mass", 1.0e-12)
    body_weight = _finite_number(topology.get("body_weight_n"), "body weight")
    _close(body_weight, total_mass * 9.81, "body weight", 1.0e-12)


def validate_physics_telemetry(report: dict[str, Any]) -> None:
    """Recompute every physics-row force, impulse and body-weight derivative."""

    topology = report.get("runtime_topology")
    require(isinstance(topology, dict), "runtime topology is required")
    topology = cast(dict[str, Any], topology)
    require(
        set(topology)
        == {
            "force_body_names",
            "link_body_names",
            "joint_names",
            "base_force_body_id",
            "foot_force_body_ids",
            "nonfoot_force_body_ids",
            "body_mass_body_names",
            "mass_accumulation",
            "body_mass_kg",
            "all_env_body_mass_kg",
            "total_mass_kg",
            "all_env_total_mass_kg",
            "body_weight_n",
        },
        "runtime topology keys mismatch",
    )
    body_names = topology.get("force_body_names")
    require(
        isinstance(body_names, list)
        and len(body_names) == 19
        and all(isinstance(name, str) for name in body_names),
        "runtime topology must contain 19 force body names",
    )
    body_names = cast(list[str], body_names)
    link_body_names = topology.get("link_body_names")
    mass_body_names = topology.get("body_mass_body_names")
    require(
        isinstance(link_body_names, list)
        and len(link_body_names) == 19
        and all(isinstance(name, str) for name in link_body_names)
        and isinstance(mass_body_names, list)
        and mass_body_names == link_body_names
        and len(set(body_names)) == 19
        and len(set(link_body_names)) == 19
        and set(body_names) == set(link_body_names),
        "force/link/mass body topology mismatch",
    )
    validate_mass_evidence(topology)
    body_weight = _finite_number(topology.get("body_weight_n"), "body weight")
    base_id = topology.get("base_force_body_id")
    foot_ids = topology.get("foot_force_body_ids")
    nonfoot_ids = topology.get("nonfoot_force_body_ids")
    require(type(base_id) is int and 0 <= base_id < 19, "base body id mismatch")
    expected_foot_ids = [
        index for index, name in enumerate(body_names) if name.endswith("_foot")
    ]
    expected_nonfoot_ids = [
        index for index, name in enumerate(body_names) if not name.endswith("_foot")
    ]
    require(
        isinstance(foot_ids, list)
        and isinstance(nonfoot_ids, list)
        and len(foot_ids) == 4
        and len(nonfoot_ids) == 15
        and all(type(index) is int and 0 <= index < 19 for index in foot_ids)
        and all(type(index) is int and 0 <= index < 19 for index in nonfoot_ids)
        and len(set(foot_ids)) == len(foot_ids)
        and len(set(nonfoot_ids)) == len(nonfoot_ids)
        and set(foot_ids) | set(nonfoot_ids) == set(range(19))
        and not set(foot_ids) & set(nonfoot_ids),
        "foot/nonfoot topology mismatch",
    )
    base_id = cast(int, base_id)
    foot_ids = cast(list[int], foot_ids)
    nonfoot_ids = cast(list[int], nonfoot_ids)
    require(body_names[base_id] == "base", "base force body label mismatch")
    require(
        foot_ids == expected_foot_ids and nonfoot_ids == expected_nonfoot_ids,
        "foot/nonfoot force body labels mismatch",
    )
    timing = report.get("telemetry_timing")
    require(isinstance(timing, dict), "telemetry timing is required")
    timing = cast(dict[str, Any], timing)
    require(
        timing
        == {
            "physics_dt_s": PHYSICS_DT_S,
            "control_dt_s": PHYSICS_DT_S * CONTROL_DECIMATION,
            "control_decimation": CONTROL_DECIMATION,
            "history_order": "newest_to_oldest",
            "peak_window_radius_physics_steps": 8,
        },
        "telemetry timing contract changed",
    )
    rows = report.get("physics_substep_telemetry")
    require(
        isinstance(rows, list) and len(rows) == 600,
        "exactly 600 physics rows are required",
    )
    rows = cast(list[Any], rows)
    required_keys = {
        "physics_step",
        "time_s",
        "control_step",
        "contact_force_history_slot",
        "history_slot_order",
        "body_names",
        "per_body_force_vector_n",
        "per_body_force_magnitude_n",
        "per_body_impulse_vector_n_s",
        "base_force_magnitude_n",
        "base_force_bodyweights",
        "base_impulse_n_s",
        "foot_total_force_n",
        "foot_impulse_n_s",
        "nonfoot_resultant_force_vector_n",
        "nonfoot_resultant_force_n",
        "nonfoot_total_force_n",
        "nonfoot_impulse_vector_n_s",
        "nonfoot_impulse_n_s",
    }
    for expected_step, row in enumerate(rows, 1):
        require(
            isinstance(row, dict) and set(row) == required_keys,
            "physics row keys mismatch",
        )
        expected_control = (
            expected_step + CONTROL_DECIMATION - 1
        ) // CONTROL_DECIMATION
        expected_slot = expected_control * CONTROL_DECIMATION - expected_step
        require(
            row["physics_step"] == expected_step, "physics steps must be exactly 1..600"
        )
        require(
            row["control_step"] == expected_control,
            "physics control-step mapping mismatch",
        )
        require(
            row["contact_force_history_slot"] == expected_slot,
            "physics history-slot formula mismatch",
        )
        require(
            row["history_slot_order"] == "newest_first",
            "physics history order mismatch",
        )
        _close(
            _finite_number(row["time_s"], "physics time"),
            expected_step * PHYSICS_DT_S,
            "physics time",
            1.0e-12,
        )
        require(row["body_names"] == body_names, "physics body-name order mismatch")
        forces = row["per_body_force_vector_n"]
        impulses = row["per_body_impulse_vector_n_s"]
        require(
            isinstance(forces, list) and len(forces) == 19, "force tensor must be 19x3"
        )
        require(
            isinstance(impulses, list) and len(impulses) == 19,
            "impulse tensor must be 19x3",
        )
        magnitudes = []
        for body_index, (force_value, impulse_value) in enumerate(
            zip(forces, impulses, strict=True)
        ):
            force = _finite_vector(force_value, 3, "body force")
            impulse = _finite_vector(impulse_value, 3, "body impulse")
            for axis in range(3):
                _close(impulse[axis], force[axis] * PHYSICS_DT_S, "body impulse")
            magnitudes.append(
                math.sqrt(sum(component * component for component in force))
            )
        reported_magnitudes = _finite_vector(
            row["per_body_force_magnitude_n"], 19, "body force magnitude"
        )
        for observed, expected in zip(reported_magnitudes, magnitudes, strict=True):
            _close(observed, expected, "body force magnitude")
        base_force = magnitudes[base_id]
        _close(
            _finite_number(row["base_force_magnitude_n"], "base force"),
            base_force,
            "base force",
        )
        _close(
            _finite_number(row["base_force_bodyweights"], "base BW"),
            base_force / body_weight,
            "base BW",
        )
        _close(
            _finite_number(row["base_impulse_n_s"], "base impulse"),
            base_force * PHYSICS_DT_S,
            "base impulse",
        )
        foot_total = sum(magnitudes[index] for index in foot_ids)
        nonfoot_total = sum(magnitudes[index] for index in nonfoot_ids)
        nonfoot_vector = [
            sum(forces[index][axis] for index in nonfoot_ids) for axis in range(3)
        ]
        nonfoot_resultant = math.sqrt(sum(value * value for value in nonfoot_vector))
        _close(
            _finite_number(row["foot_total_force_n"], "foot force"),
            foot_total,
            "foot force",
        )
        _close(
            _finite_number(row["foot_impulse_n_s"], "foot impulse"),
            foot_total * PHYSICS_DT_S,
            "foot impulse",
        )
        for observed, expected in zip(
            _finite_vector(
                row["nonfoot_resultant_force_vector_n"], 3, "nonfoot force vector"
            ),
            nonfoot_vector,
            strict=True,
        ):
            _close(observed, expected, "nonfoot force vector")
        _close(
            _finite_number(row["nonfoot_resultant_force_n"], "nonfoot resultant"),
            nonfoot_resultant,
            "nonfoot resultant",
        )
        _close(
            _finite_number(row["nonfoot_total_force_n"], "nonfoot total"),
            nonfoot_total,
            "nonfoot total",
        )
        for observed, expected in zip(
            _finite_vector(
                row["nonfoot_impulse_vector_n_s"], 3, "nonfoot impulse vector"
            ),
            [value * PHYSICS_DT_S for value in nonfoot_vector],
            strict=True,
        ):
            _close(observed, expected, "nonfoot impulse vector")
        _close(
            _finite_number(row["nonfoot_impulse_n_s"], "nonfoot impulse"),
            nonfoot_total * PHYSICS_DT_S,
            "nonfoot impulse",
        )


def validate_control_telemetry(report: dict[str, Any]) -> None:
    topology = report["runtime_topology"]
    joint_names = topology.get("joint_names")
    body_names = topology.get("link_body_names")
    require(
        isinstance(joint_names, list) and len(joint_names) == 12,
        "runtime topology must contain 12 joints",
    )
    active_terminations = report.get("active_terminations")
    require(
        isinstance(active_terminations, list) and len(active_terminations) > 0,
        "active terminations missing",
    )
    active_terminations = cast(list[str], active_terminations)
    rows = report.get("control_step_telemetry")
    require(
        isinstance(rows, list) and len(rows) == 150,
        "exactly 150 control rows are required",
    )
    rows = cast(list[Any], rows)
    required_keys = {
        "control_step",
        "time_s",
        "termination_flags",
        "root_state_w",
        "link_state_field",
        "link_names",
        "link_state_w",
        "joint_names",
        "joint_position_rad",
        "joint_velocity_rad_s",
        "applied_torque_nm",
        "input_action",
        "raw_action",
        "processed_ema_target_rad",
        "ema_previous_before_rad",
        "ema_previous_after_rad",
    }
    for expected_step, row in enumerate(rows, 1):
        require(
            isinstance(row, dict) and set(row) == required_keys,
            "control row keys mismatch",
        )
        require(
            row["control_step"] == expected_step, "control steps must be exactly 1..150"
        )
        _close(
            _finite_number(row["time_s"], "control time"),
            expected_step * 0.02,
            "control time",
            1.0e-12,
        )
        require(
            row["link_state_field"] == "body_link_state_w",
            "link-state authority changed",
        )
        require(
            row["link_names"] == body_names and row["joint_names"] == joint_names,
            "control topology order mismatch",
        )
        _finite_vector(row["root_state_w"], 13, "root state")
        link_state = row["link_state_w"]
        require(
            isinstance(link_state, list) and len(link_state) == 19,
            "link state must be 19x13",
        )
        for link in link_state:
            _finite_vector(link, 13, "link state")
        for field in (
            "joint_position_rad",
            "joint_velocity_rad_s",
            "applied_torque_nm",
            "input_action",
            "raw_action",
            "processed_ema_target_rad",
            "ema_previous_before_rad",
            "ema_previous_after_rad",
        ):
            _finite_vector(row[field], 12, field)
        flags = row["termination_flags"]
        require(
            isinstance(flags, dict)
            and list(flags) == active_terminations
            and all(type(value) is bool for value in flags.values()),
            "control termination flags mismatch",
        )


def validate_physics_step_clock(report: dict[str, Any]) -> None:
    clock = report.get("physics_step_clock")
    require(isinstance(clock, dict), "physics step clock evidence is required")
    clock = cast(dict[str, Any], clock)
    require(
        set(clock)
        == {
            "source",
            "evidence_kind",
            "contract_expected_dt_s",
            "callback_expected_float32_dt_s",
            "callback_dt_abs_tolerance_s",
            "observed_dt_min_s",
            "observed_dt_max_s",
            "max_abs_error_s",
            "mismatch_count",
            "nonfinite_count",
            "first_mismatch",
            "callback_count",
            "expected_callback_count",
            "passed",
        },
        "physics step clock keys mismatch",
    )
    require(
        clock["source"] == "subscribe_physics_on_step_events(pre_step=true,order=0)"
        and clock["evidence_kind"] == "pre_step_notification_count",
        "physics step clock authority naming mismatch",
    )
    contract_dt = _finite_number(clock["contract_expected_dt_s"], "clock contract dt")
    _close(contract_dt, PHYSICS_DT_S, "clock contract dt", 1.0e-12)
    timing = report.get("telemetry_timing")
    require(isinstance(timing, dict), "telemetry timing is required")
    timing = cast(dict[str, Any], timing)
    _close(
        contract_dt,
        _finite_number(timing.get("physics_dt_s"), "telemetry physics dt"),
        "clock/telemetry physics dt",
        1.0e-12,
    )
    callback_expected = _finite_number(
        clock["callback_expected_float32_dt_s"], "clock callback float32 dt"
    )
    _close(
        callback_expected,
        _float32(contract_dt),
        "clock callback float32 dt",
        0.0,
    )
    tolerance = _finite_number(
        clock["callback_dt_abs_tolerance_s"], "clock callback dt tolerance"
    )
    _close(tolerance, CALLBACK_DT_ABS_TOLERANCE_S, "clock callback dt tolerance", 0.0)
    require(tolerance > 0.0, "clock callback dt tolerance must be positive")
    callback_count = clock["callback_count"]
    expected_count = clock["expected_callback_count"]
    mismatch_count = clock["mismatch_count"]
    nonfinite_count = clock["nonfinite_count"]
    require(
        type(callback_count) is int
        and type(expected_count) is int
        and type(mismatch_count) is int
        and type(nonfinite_count) is int
        and callback_count >= 0
        and expected_count == ROLLOUT_STEPS * CONTROL_DECIMATION
        and 0 <= nonfinite_count <= mismatch_count <= callback_count,
        "physics step clock count types or ranges mismatch",
    )
    observed_min = clock["observed_dt_min_s"]
    observed_max = clock["observed_dt_max_s"]
    finite_callback_count = callback_count - nonfinite_count
    if finite_callback_count > 0:
        observed_min = _finite_number(observed_min, "observed callback dt min")
        observed_max = _finite_number(observed_max, "observed callback dt max")
        require(
            observed_min > 0.0 and observed_min <= observed_max,
            "observed callback dt range mismatch",
        )
        derived_max_error = max(
            abs(observed_min - callback_expected),
            abs(observed_max - callback_expected),
        )
    else:
        require(
            observed_min is None and observed_max is None,
            "empty callback dt range must be null",
        )
        derived_max_error = 0.0
    max_error = _finite_number(clock["max_abs_error_s"], "clock max dt error")
    require(max_error >= 0.0, "clock max dt error must be nonnegative")
    _close(max_error, derived_max_error, "clock max dt error", 1.0e-18)
    first_mismatch = clock["first_mismatch"]
    if mismatch_count == 0:
        require(
            first_mismatch is None and nonfinite_count == 0 and max_error <= tolerance,
            "zero-mismatch clock evidence is inconsistent",
        )
    else:
        require(
            isinstance(first_mismatch, dict)
            and set(first_mismatch)
            == {"callback_index", "observed_dt_s", "abs_error_s", "reason"}
            and type(first_mismatch.get("callback_index")) is int
            and 1 <= first_mismatch["callback_index"] <= callback_count
            and first_mismatch.get("reason")
            in {"nonfinite", "outside_float32_boundary_tolerance"},
            "first clock mismatch evidence is invalid",
        )
        if first_mismatch["reason"] == "nonfinite":
            require(
                first_mismatch["observed_dt_s"] is None
                and first_mismatch["abs_error_s"] is None
                and nonfinite_count > 0,
                "nonfinite clock mismatch evidence is invalid",
            )
        else:
            mismatch_observed = _finite_number(
                first_mismatch["observed_dt_s"], "first mismatch observed dt"
            )
            mismatch_error = _finite_number(
                first_mismatch["abs_error_s"], "first mismatch dt error"
            )
            _close(
                mismatch_error,
                abs(mismatch_observed - callback_expected),
                "first mismatch dt error",
                1.0e-18,
            )
            require(
                mismatch_error > tolerance,
                "first mismatch must exceed callback dt tolerance",
            )
    derived_passed = (
        callback_count == expected_count
        and mismatch_count == 0
        and nonfinite_count == 0
        and finite_callback_count == callback_count
        and max_error <= tolerance
    )
    require(
        type(clock["passed"]) is bool and clock["passed"] is derived_passed,
        "physics step clock passed derivation mismatch",
    )
    require(
        derived_passed,
        "physics step clock must prove exactly 600 pre-step notifications",
    )


def validate_contact_authority(report: dict[str, Any]) -> None:
    authority = report.get("cpu_contact_authority")
    require(isinstance(authority, dict), "contact authority object is required")
    authority = cast(dict[str, Any], authority)
    require(
        type(authority.get("subsequent_error_count")) is int
        and authority["subsequent_error_count"] >= 0,
        "contact authority subsequent-error count mismatch",
    )
    require(
        authority.get("authority_device") == "cpu", "contact authority device changed"
    )
    device = report["device"]
    if device != "cpu":
        require(
            authority.get("this_run_is_authority") is False
            and authority.get("status") == "unavailable_on_gpu"
            and authority.get("data_available") is False
            and authority.get("error") is None
            and authority.get("subsequent_error_count") == 0
            and authority.get("passed") is None
            and authority.get("events") is None
            and authority.get("all_env_minimum_separation_m") is None,
            "GPU fabricated CPU contact authority",
        )
        return
    require(
        authority.get("this_run_is_authority") is True
        and authority.get("status") == "observed"
        and authority.get("data_available") is True
        and authority.get("error") is None
        and authority.get("subsequent_error_count") == 0
        and authority.get("passed") is True,
        "CPU contact authority is incomplete",
    )
    require(
        authority.get("physics_step_clock") == report.get("physics_step_clock"),
        "contact authority clock binding mismatch",
    )
    separations = authority.get("all_env_minimum_separation_m")
    require(
        isinstance(separations, list) and len(separations) == 8,
        "CPU all-env separation summary incomplete",
    )
    for separation in cast(list[Any], separations):
        _finite_number(separation, "CPU all-env separation")
    events = authority.get("events")
    require(
        isinstance(events, list) and len(events) > 0,
        "CPU target contact events missing",
    )
    previous_callback = 0
    previous_step = 0
    point_keys = {
        "actor0_path",
        "actor1_path",
        "collider0_path",
        "collider1_path",
        "contact_position_w_m",
        "contact_normal_w",
        "reported_contact_impulse_n_s",
        "separation_m",
    }
    for event in cast(list[Any], events):
        require(isinstance(event, dict), "CPU contact event must be an object")
        physics_step = event.get("physics_step")
        callback_index = event.get("callback_event_index")
        require(
            type(physics_step) is int
            and 1 <= physics_step <= 600
            and physics_step >= previous_step,
            "CPU contact physics-step stamp mismatch",
        )
        require(
            type(callback_index) is int and callback_index > previous_callback,
            "CPU contact callback sequence mismatch",
        )
        previous_step = physics_step
        previous_callback = callback_index
        headers = event.get("headers")
        require(
            isinstance(headers, list) and len(headers) > 0,
            "CPU contact headers missing",
        )
        for header in cast(list[Any], headers):
            require(isinstance(header, dict), "CPU contact header must be an object")
            require(
                header.get("env_index") == SOURCE_ENV_INDEX,
                "CPU detailed event escaped env7",
            )
            event_type = header.get("event_type")
            require(
                event_type in {"CONTACT_FOUND", "CONTACT_PERSIST", "CONTACT_LOST"},
                "CPU contact event type mismatch",
            )
            for path_key in (
                "actor0_path",
                "actor1_path",
                "collider0_path",
                "collider1_path",
            ):
                require(
                    isinstance(header.get(path_key), str), "CPU contact path missing"
                )
            points = header.get("contact_points")
            require(isinstance(points, list), "CPU contact point list missing")
            require(
                (event_type == "CONTACT_LOST" and len(points) == 0)
                or (event_type != "CONTACT_LOST" and len(points) > 0),
                "CPU FOUND/PERSIST/LOST data contract mismatch",
            )
            for point in cast(list[Any], points):
                require(
                    isinstance(point, dict) and set(point) == point_keys,
                    "CPU contact point keys mismatch",
                )
                for path_key in (
                    "actor0_path",
                    "actor1_path",
                    "collider0_path",
                    "collider1_path",
                ):
                    require(
                        isinstance(point[path_key], str),
                        "CPU contact point path missing",
                    )
                _finite_vector(point["contact_position_w_m"], 3, "contact position")
                _finite_vector(point["contact_normal_w"], 3, "contact normal")
                _finite_vector(
                    point["reported_contact_impulse_n_s"], 3, "contact impulse"
                )
                _finite_number(point["separation_m"], "contact separation")


def _historical_pose_projection(
    report: dict[str, Any], device: str
) -> list[dict[str, Any]]:
    metrics = report.get("pose_mode_metrics")
    require(
        isinstance(metrics, list) and len(metrics) == 8,
        "historical reference missing 8 pose metrics",
    )
    metrics = cast(list[Any], metrics)
    result = []
    for item in metrics:
        require(isinstance(item, dict), "historical pose metric must be an object")
        result.append(
            {
                "env_index": item.get("env_index"),
                "pose_id": item.get("pose_id"),
                "action_mode": item.get("action_mode"),
                "max_nonfoot_force_bodyweights": item.get(
                    "max_nonfoot_force_bodyweights"
                ),
                "max_nonfoot_force_physics_step": item.get(
                    "max_nonfoot_force_physics_step"
                ),
                "max_nonfoot_force_body_index": item.get(
                    "max_nonfoot_force_body_index"
                ),
                "max_nonfoot_force_body_name": item.get("max_nonfoot_force_body_name"),
                "max_root_angular_speed_rad_s": item.get(
                    "max_root_angular_speed_rad_s"
                ),
                "max_joint_speed_rad_s": item.get("max_joint_speed_rad_s"),
                "termination_counts": item.get("termination_counts"),
                "min_contact_separation_m": (
                    item.get("min_contact_separation_m") if device == "cpu" else None
                ),
            }
        )
    return result


def _metric_matches(observed: dict[str, Any], reference: dict[str, Any]) -> bool:
    if set(observed) != set(reference):
        return False
    for key, expected in reference.items():
        actual = observed[key]
        if type(expected) in (int, float) and type(actual) in (int, float):
            if not math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-6
            ):
                return False
        elif actual != expected:
            return False
    return True


def build_historical_runtime_summary(
    arm: str,
    device: str,
    observed_pose_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = HISTORICAL_REPORTS[(arm, device)]
    path = REPO_ROOT / reference["path"]
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == reference["sha256"], "historical raw reference SHA mismatch")
    historical = json.loads(raw.decode("utf-8"))
    expected_contract = HISTORICAL_REFERENCES[arm]
    require(
        historical.get("contract_sha256") == expected_contract["canonical_sha256"],
        "historical raw contract SHA mismatch",
    )
    expected_metrics = _historical_pose_projection(historical, device)
    require(
        len(observed_pose_metrics) == 8, "observed all-env summary must contain 8 cells"
    )
    matches = all(
        _metric_matches(observed, expected)
        for observed, expected in zip(
            observed_pose_metrics, expected_metrics, strict=True
        )
    )
    candidate_checks = {
        "all_cells_finite": all(
            math.isfinite(float(item["max_nonfoot_force_bodyweights"]))
            and math.isfinite(float(item["max_root_angular_speed_rad_s"]))
            and math.isfinite(float(item["max_joint_speed_rad_s"]))
            for item in observed_pose_metrics
        ),
        "max_nonfoot_force_at_most_15_bodyweights": all(
            float(item["max_nonfoot_force_bodyweights"]) <= 15.0
            for item in observed_pose_metrics
        ),
        "safety_termination_zero": all(
            item["termination_counts"].get("numeric_invalid") == 0
            and item["termination_counts"].get("hard_joint_limit") == 0
            for item in observed_pose_metrics
        ),
        "cpu_separation_at_least_minus_1cm": (
            all(
                item["min_contact_separation_m"] is not None
                and float(item["min_contact_separation_m"]) >= -0.01
                for item in observed_pose_metrics
            )
            if device == "cpu"
            else True
        ),
    }
    checks = {
        "reference_report_sha256_exact": True,
        "reference_contract_sha256_exact": True,
        "pose_metric_fingerprint_within_1e_6": matches,
    }
    return {
        "reference_contract_id": expected_contract["contract_id"],
        "reference_contract_sha256": expected_contract["canonical_sha256"],
        "reference_report_path": reference["path"],
        "reference_report_sha256": reference["sha256"],
        "pose_metrics": observed_pose_metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "runtime_candidate_checks": candidate_checks,
        "runtime_candidate_passed": all(candidate_checks.values()),
        "matches_historical_reference": matches,
        "progression_allowed": False,
    }


def validate_predecessor_synthesis(
    path: Path | None,
    *,
    arm: str,
    device: str,
    source_bundle: dict[str, Any],
) -> dict[str, Any] | None:
    requirement = PREDECESSOR_REQUIREMENTS[(arm, device)]
    if requirement is None:
        require(path is None, "A.cpu must not accept a predecessor synthesis")
        return None
    require(path is not None, f"{arm}.{device} requires --predecessor-synthesis")
    path = cast(Path, path)
    resolved = path.expanduser().resolve(strict=True)
    require(
        resolved.parent == (REPO_ROOT / "reports" / "runs").resolve(),
        "predecessor synthesis must be a direct child of reports/runs",
    )
    raw = resolved.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, dict), "predecessor synthesis root must be an object")
    value = cast(dict[str, Any], value)
    expected_count, expected_next_group = requirement
    require(
        value.get("schema_version") == "g009.r0.rev16.backend_divergence_synthesis.v1"
        and value.get("goal_id") == "g009"
        and value.get("stage_id") == "R0"
        and value.get("revision") == "rev16"
        and value.get("status") == "complete"
        and value.get("evidence_synthesis_valid") is True,
        "predecessor synthesis identity/status/evidence is invalid",
    )
    require(
        value.get("input_report_count") == expected_count,
        "predecessor input report count mismatch",
    )
    input_reports = value.get("input_reports")
    require(
        isinstance(input_reports, list) and len(input_reports) == expected_count,
        "predecessor input report evidence is incomplete",
    )
    input_reports = cast(list[Any], input_reports)
    group_order = (("A", "cpu"), ("A", "cuda:0"), ("B", "cpu"), ("B", "cuda:0"))
    validated_inputs: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for index, evidence_value in enumerate(input_reports):
        require(
            isinstance(evidence_value, dict),
            "predecessor input evidence must be an object",
        )
        evidence = cast(dict[str, Any], evidence_value)
        require(
            set(evidence) == {"path", "sha256"},
            "predecessor input evidence keys mismatch",
        )
        relative_path = evidence.get("path")
        digest = evidence.get("sha256")
        require(
            isinstance(relative_path, str)
            and relative_path.startswith("reports/runs/")
            and Path(relative_path).name == relative_path.removeprefix("reports/runs/"),
            "predecessor input path must be a direct reports/runs child",
        )
        require(
            isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "predecessor input SHA must be lowercase SHA-256",
        )
        relative_path = cast(str, relative_path)
        digest = cast(str, digest)
        require(
            relative_path not in seen_paths, "predecessor input paths must be unique"
        )
        require(digest not in seen_hashes, "predecessor input hashes must be unique")
        seen_paths.add(relative_path)
        seen_hashes.add(digest)
        input_path = (REPO_ROOT / relative_path).resolve(strict=True)
        require(
            input_path.parent == (REPO_ROOT / "reports" / "runs").resolve(),
            "predecessor input resolved outside reports/runs",
        )
        input_raw = input_path.read_bytes()
        require(
            hashlib.sha256(input_raw).hexdigest() == digest,
            "predecessor input bytes SHA mismatch",
        )
        input_report = json.loads(input_raw.decode("utf-8"))
        require(
            isinstance(input_report, dict),
            "predecessor raw input root must be an object",
        )
        input_report = cast(dict[str, Any], input_report)
        validate_report_contract(input_report)
        group_index = index // 3
        replicate_index = index % 3 + 1
        expected_arm, expected_device = group_order[group_index]
        require(
            input_report["contract"]["arm"]["id"] == expected_arm
            and input_report.get("device") == expected_device
            and input_report.get("runtime_device") == expected_device
            and input_report.get("replicate_index") == replicate_index,
            "predecessor raw input group/order/replicate mismatch",
        )
        require(
            input_report.get("execution", {}).get("output_path_repo_relative")
            == relative_path,
            "predecessor raw execution output binding mismatch",
        )
        execution = input_report.get("execution", {})
        execution_id = execution.get("execution_id")
        started_at = execution.get("started_at_utc")
        require(
            isinstance(execution_id, str)
            and re.fullmatch(r"[0-9a-f]{32}", execution_id) is not None
            and uuid.UUID(hex=execution_id).version == 4
            and execution.get("no_overwrite") is True,
            "predecessor raw execution provenance mismatch",
        )
        require(
            isinstance(started_at, str) and started_at.endswith("Z"),
            "predecessor raw execution UTC timestamp missing",
        )
        try:
            parsed_started_at = datetime.fromisoformat(
                started_at.removesuffix("Z") + "+00:00"
            )
        except ValueError as error:
            raise ValueError(
                "predecessor raw execution UTC timestamp invalid"
            ) from error
        require(
            parsed_started_at.utcoffset() == timezone.utc.utcoffset(parsed_started_at),
            "predecessor raw execution timestamp is not UTC",
        )
        raw_source = input_report.get("source_bundle", {})
        require(
            raw_source.get("git_commit") == source_bundle.get("git_commit")
            and raw_source.get("source_bundle_sha256")
            == source_bundle.get("source_bundle_sha256"),
            "predecessor raw source binding mismatch",
        )
        historical = input_report.get("historical_runtime_summary", {})
        require(
            historical.get("passed") is True
            and historical.get("matches_historical_reference") is True
            and historical.get("runtime_candidate_passed") is (group_index < 3)
            and historical.get("progression_allowed") is False,
            "predecessor raw historical/runtime result mismatch",
        )
        validated_inputs.append(input_report)
    require(
        value.get("source_commit") == source_bundle.get("git_commit"),
        "predecessor source commit mismatch",
    )
    require(
        value.get("source_bundle_sha256") == source_bundle.get("source_bundle_sha256"),
        "predecessor source bundle mismatch",
    )
    require(
        value.get("run_matrix")
        == {
            "validated_run_count": expected_count,
            "validated_group_count": expected_count // 3,
        },
        "predecessor run matrix mismatch",
    )
    require(
        value.get("next_group") == expected_next_group,
        "predecessor next_group mismatch",
    )
    require(
        value.get("required_sequence") == ["A.cpu", "A.cuda:0", "B.cpu", "B.cuda:0"]
        and value.get("completed_group_count") == expected_count // 3,
        "predecessor required/completed sequence mismatch",
    )
    groups = value.get("groups")
    require(
        isinstance(groups, list) and len(groups) == expected_count // 3,
        "predecessor group evidence is incomplete",
    )
    for group_index, group_value in enumerate(cast(list[Any], groups)):
        require(isinstance(group_value, dict), "predecessor group must be an object")
        group = cast(dict[str, Any], group_value)
        expected_arm, expected_device = group_order[group_index]
        require(
            group.get("sequence_index") == group_index + 1
            and group.get("arm") == expected_arm
            and group.get("device") == expected_device
            and group.get("replicate_count") == 3
            and group.get("historical_reproduction_3_of_3") is True
            and group.get("runtime_candidate_expected_3_of_3") is True
            and group.get("sequence_gate_passed") is True
            and group.get("progression_allowed") is True,
            "predecessor group decision/sequence mismatch",
        )
        runs = group.get("runs")
        require(
            isinstance(runs, list) and len(runs) == 3, "predecessor group runs missing"
        )
        for replicate_offset, run_value in enumerate(cast(list[Any], runs)):
            require(
                isinstance(run_value, dict), "predecessor group run must be an object"
            )
            run = cast(dict[str, Any], run_value)
            evidence_index = group_index * 3 + replicate_offset
            raw_report = validated_inputs[evidence_index]
            require(
                run.get("evidence") == input_reports[evidence_index],
                "predecessor group run/input evidence mismatch",
            )
            require(
                run.get("arm") == expected_arm
                and run.get("device") == expected_device
                and run.get("replicate_index") == replicate_offset + 1
                and run.get("source_commit") == source_bundle.get("git_commit")
                and run.get("source_bundle_sha256")
                == source_bundle.get("source_bundle_sha256")
                and run.get("contract_sha256") == raw_report.get("contract_sha256")
                and run.get("historical_reproduction_passed") is True
                and run.get("runtime_candidate_passed") is True,
                "predecessor group run decision/source mismatch",
            )
    require(
        value.get("hypothesis")
        == {
            "decision": "pending_sequential_groups",
            "supported_3_of_3": None,
            "replicates": [],
        },
        "predecessor partial hypothesis must remain pending",
    )
    require(
        value.get("governance")
        == {
            "position16_accepted": False,
            "position16_status": "rejected_even_if_hypothesis_supported",
            "diagnostic_only": True,
            "learned": False,
            "ppo": {"allowed": False, "status": "not_run"},
            "gate01": {"allowed": False, "status": "forbidden"},
            "gate10": {"allowed": False, "status": "forbidden"},
            "qualification": {"eligible": False, "status": "not_run", "passed": None},
        },
        "predecessor governance mismatch",
    )
    return {
        "path": f"reports/runs/{resolved.name}",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_synthesis_valid": True,
        "validated_run_count": expected_count,
        "next_group": expected_next_group,
        "source_commit": value["source_commit"],
        "source_bundle_sha256": value["source_bundle_sha256"],
    }


def validate_historical_runtime_summary(report: dict[str, Any]) -> None:
    contract = report["contract"]
    arm = contract["arm"]["id"]
    device = report["device"]
    summary = report.get("historical_runtime_summary")
    require(isinstance(summary, dict), "historical runtime summary is required")
    summary = cast(dict[str, Any], summary)
    pose_metrics = summary.get("pose_metrics")
    require(isinstance(pose_metrics, list), "historical pose metrics are required")
    expected = build_historical_runtime_summary(
        arm, device, cast(list[dict[str, Any]], pose_metrics)
    )
    require(summary == expected, "historical runtime summary validation mismatch")


def validate_predecessor_binding(report: dict[str, Any]) -> None:
    arm = report["contract"]["arm"]["id"]
    device = report["device"]
    requirement = PREDECESSOR_REQUIREMENTS[(arm, device)]
    binding = report.get("predecessor_synthesis")
    if requirement is None:
        require(binding is None, "A.cpu predecessor binding must be null")
        return
    require(isinstance(binding, dict), "predecessor synthesis binding is required")
    binding = cast(dict[str, Any], binding)
    expected_count, expected_next = requirement
    require(
        binding.get("evidence_synthesis_valid") is True
        and binding.get("validated_run_count") == expected_count
        and binding.get("next_group") == expected_next,
        "predecessor synthesis sequence mismatch",
    )
    source = report["source_bundle"]
    require(
        binding.get("source_commit") == source.get("git_commit")
        and binding.get("source_bundle_sha256") == source.get("source_bundle_sha256"),
        "predecessor synthesis source mismatch",
    )


def validate_report_contract(report: dict[str, Any]) -> None:
    require(
        report.get("schema_version") == "g009.r0.rev16.backend_divergence.v1"
        and report.get("goal_id") == "g009"
        and report.get("stage_id") == "R0"
        and report.get("revision") == "rev16"
        and report.get("status") == "complete",
        "raw report identity/status mismatch",
    )
    guard = report.get("governance", {})
    require(report.get("diagnostic_only") is True, "diagnostic_only must be true")
    require(
        report.get("qualification_eligible") is False,
        "qualification_eligible must be false",
    )
    require(guard == governance(), "training/gate governance changed")
    require(
        report.get("replicate_index") in {1, 2, 3},
        "replicate_index must be 1, 2 or 3",
    )
    require(report.get("headless") is True, "rev16 report must be headless")
    require(
        str(report.get("device", "")).lower() in {"cpu", "cuda:0"},
        "rev16 device must be cpu or cuda:0",
    )
    require(
        report.get("runtime_device") == report.get("device"),
        "runtime device readback mismatch",
    )
    require(report.get("task") == DEFAULT_TASK, "task contract changed")
    require(report.get("seed") == 42, "seed contract changed")
    require(report.get("num_envs") == NUM_ENVS, "num_envs contract changed")
    require(
        report.get("rollout_steps") == ROLLOUT_STEPS,
        "rollout_steps contract changed",
    )
    assignment = report.get("pose_action_assignment")
    require(
        isinstance(assignment, dict)
        and assignment.get("class_ids") == [0, 1, 2, 3, 0, 1, 2, 3]
        and assignment.get("mapping") == expected_pose_action_assignment(),
        "pose/action assignment mismatch",
    )
    contract = report.get("contract", {})
    arm_id = contract.get("arm", {}).get("id") if isinstance(contract, dict) else None
    require(
        isinstance(arm_id, str)
        and contract == rev16_contract(arm_id, str(report["device"])),
        "rev16 contract or historical lineage changed",
    )
    require(
        report.get("contract_sha256") == canonical_sha256(contract),
        "contract hash mismatch",
    )
    execution = report.get("execution", {})
    require(execution.get("no_overwrite") is True, "no-overwrite provenance missing")
    require(bool(execution.get("execution_id")), "fresh execution ID missing")
    require(
        report.get("source_bundle", {}).get("all_files_present") is True,
        "source binding incomplete",
    )
    require(
        report.get("source_bundle", {}).get("clean") is True, "source binding dirty"
    )
    require(
        set(report["source_bundle"].get("source_binding_paths", []))
        == set(SOURCE_BINDING_PATHS),
        "source binding path set mismatch",
    )
    validate_predecessor_binding(report)
    cell = report.get("controlled_cell")
    topology = report.get("runtime_topology")
    base_force_body_id = (
        topology.get("base_force_body_id") if isinstance(topology, dict) else None
    )
    require(type(base_force_body_id) is int, "base body id mismatch")
    require(
        isinstance(cell, dict)
        and cell.get("source_env_index") == 7
        and cell.get("pose_id") == "right_side"
        and cell.get("action_mode") == "reset_pose_hold"
        and cell.get("target_body_index") == base_force_body_id
        and cell.get("target_body_name") == "base",
        "controlled cell mismatch",
    )
    live = report.get("live_physics_readback")
    require(
        isinstance(live, dict)
        and isinstance(live.get("checks"), dict)
        and bool(live["checks"])
        and all(value is True for value in live["checks"].values()),
        "live physics readback failed",
    )
    if report.get("status") == "complete":
        require(
            report.get("diagnostic_capture_complete") is True,
            "complete report must pass every diagnostic capture check",
        )
        validate_physics_telemetry(report)
        validate_control_telemetry(report)
        validate_historical_runtime_summary(report)
        validate_physics_step_clock(report)
        require(
            report.get("safety_termination_counts")
            == {"numeric_invalid": 0, "hard_joint_limit": 0},
            "safety termination telemetry is nonzero or missing",
        )
        validate_contact_authority(report)


def diagnose(args: argparse.Namespace, execution: dict[str, Any]) -> dict[str, Any]:
    require(args.headless is True, "rev16 requires --headless")
    require(
        str(args.device).lower() in {"cpu", "cuda:0"},
        "rev16 device must be cpu or cuda:0",
    )
    require(args.task == DEFAULT_TASK, "rev16 task is fixed")
    require(args.seed == 42, "rev16 seed is fixed at 42")
    normalized_arm = args.arm.upper()
    normalized_device = args.device.lower()
    contract = rev16_contract(normalized_arm, normalized_device)
    source_bundle = source_bundle_provenance()
    require(source_bundle["all_files_present"], "source binding files are missing")
    require(source_bundle["clean"], "source binding must be clean")
    predecessor = validate_predecessor_synthesis(
        args.predecessor_synthesis,
        arm=normalized_arm,
        device=normalized_device,
        source_bundle=source_bundle,
    )

    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    import isaaclab_tasks  # noqa: F401  # pyright: ignore[reportMissingImports]
    import omni.usd  # pyright: ignore[reportMissingImports]
    import torch
    from isaaclab import sim as sim_utils  # pyright: ignore[reportMissingImports]
    from isaaclab_tasks.utils import (  # pyright: ignore[reportMissingImports]
        parse_env_cfg,
    )
    from omni.physx import (  # pyright: ignore[reportMissingImports]
        get_physx_interface,
        get_physx_simulation_interface,
    )
    from omni.physx.bindings._physx import (  # pyright: ignore[reportMissingImports]
        ContactEventType,
    )
    from pxr import (  # pyright: ignore[reportMissingImports]
        PhysicsSchemaTools,
        PhysxSchema,
        UsdPhysics,
    )

    from isaac_walk_g009 import register_tasks

    register_tasks()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=NUM_ENVS)
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.contact_forces.history_length = CONTROL_DECIMATION
    env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = (
        contract["arm"]["articulation_solver_position_iteration_count"]
    )
    env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 0
    env_cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity = 1.0
    env_cfg.events.reset_base.params.update(
        {
            "assignment_mode": "stratified",
            "pose_xy_range": (0.0, 0.0),
            "yaw_range": (0.0, 0.0),
        }
    )
    env_cfg.validate()
    env = gym.make(args.task, cfg=env_cfg)
    raw_env: Any = env.unwrapped
    robot = raw_env.scene["robot"]
    sensor = raw_env.scene.sensors["contact_forces"]
    action_term = raw_env.action_manager.get_term("joint_pos")
    stage = omni.usd.get_context().get_stage()
    solver_readback = runtime_probe.articulation_solver_iteration_readback(
        stage, list(robot.root_physx_view.prim_paths), PhysxSchema
    )
    depenetration_readback = (
        runtime_probe.rigid_body_max_depenetration_velocity_readback(
            stage,
            sim_utils.find_matching_prim_paths(robot.cfg.prim_path, stage),
            list(robot.root_physx_view.prim_paths),
            [list(group) for group in robot.root_physx_view.link_paths],
            list(robot.body_names),
            PhysxSchema,
            UsdPhysics,
        )
    )
    solver_checks = runtime_probe.articulation_solver_iteration_checks(
        solver_readback,
        expected_position_count=contract["arm"][
            "articulation_solver_position_iteration_count"
        ],
        expected_velocity_count=0,
        expected_articulations=NUM_ENVS,
    )
    depenetration_checks = runtime_probe.rigid_body_max_depenetration_velocity_checks(
        depenetration_readback,
        expected_velocity_m_s=1.0,
        expected_articulation_count=NUM_ENVS,
        expected_body_names=list(robot.body_names),
    )
    require(
        math.isclose(
            float(raw_env.physics_dt), PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "runtime physics dt contract changed",
    )
    physics_clock = PhysicsStepClock(float(raw_env.physics_dt))
    authority = CpuContactAuthorityAccumulator(
        SOURCE_ENV_INDEX,
        NUM_ENVS,
        PhysicsSchemaTools.intToSdfPath,
        physics_clock,
        ContactEventType,
    )
    contact_subscription = None
    physics_clock_subscription = None
    try:
        env.reset()
        class_ids = raw_env._g009_recover_fall_class.detach().clone()
        observed_class_ids = [int(value) for value in class_ids.detach().cpu().tolist()]
        require(
            observed_class_ids == [0, 1, 2, 3, 0, 1, 2, 3],
            "8-env stratified class assignment mismatch",
        )
        hold = runtime_probe.reset_pose_hold_action_diagnostics(
            robot.data.joint_pos[4:].detach(),
            robot.data.soft_joint_pos_limits[4:].detach(),
            list(robot.joint_names),
            action_scale=float(action_term.cfg.scale),
        )
        require(bool((~hold["saturated_mask"]).all().item()), "hold action saturated")
        actions = torch.zeros(
            (NUM_ENVS, raw_env.action_manager.total_action_dim), device=raw_env.device
        )
        actions[4:] = hold["normalized_action"]
        stable_cfg = raw_env.termination_manager.get_term_cfg("stable_success")
        stable_cfg.params["required_consecutive_steps"] = ROLLOUT_STEPS + 1
        physics_clock_subscription = (
            get_physx_interface().subscribe_physics_on_step_events(
                physics_clock, True, 0
            )
        )
        if physics_clock_subscription is None:
            raise RuntimeError("physics pre-step callback subscription unavailable")
        if normalized_device == "cpu":
            try:
                contact_subscription = (
                    get_physx_simulation_interface().subscribe_contact_report_events(
                        authority
                    )
                )
                if contact_subscription is None:
                    raise RuntimeError("contact callback subscription unavailable")
            except Exception as error:  # noqa: BLE001 - unsupported CPU API fails closed
                authority.mark_unavailable(error)

        physics_rows: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
        safety_termination_counts = {"numeric_invalid": 0, "hard_joint_limit": 0}
        active_terminations = list(raw_env.termination_manager.active_terms)
        require(
            set(safety_termination_counts) <= set(active_terminations),
            "required safety termination terms are unavailable",
        )
        body_names = list(sensor.body_names)
        link_body_names = list(robot.body_names)
        require(
            len(body_names) == 19
            and len(link_body_names) == 19
            and len(set(body_names)) == 19
            and len(set(link_body_names)) == 19
            and set(body_names) == set(link_body_names),
            "sensor/robot body topology mismatch",
        )
        base_body_id = body_names.index("base")
        foot_ids = [
            index for index, name in enumerate(body_names) if name.endswith("_foot")
        ]
        nonfoot_ids = [
            index for index, name in enumerate(body_names) if not name.endswith("_foot")
        ]
        mass_evidence = build_mass_evidence(
            raw_env._g009_r0_body_mass, SOURCE_ENV_INDEX
        )
        try:
            validate_mass_evidence(mass_evidence)
        except Exception as error:
            raise MassEvidenceError(mass_evidence, error) from error
        total_mass_kg = float(mass_evidence["total_mass_kg"])
        all_env_total_mass_kg = torch.tensor(
            mass_evidence["all_env_total_mass_kg"],
            dtype=torch.float64,
            device=raw_env.device,
        )
        max_nonfoot_force_bw = torch.zeros(
            NUM_ENVS, dtype=torch.float64, device=raw_env.device
        )
        max_nonfoot_force_step = torch.full(
            (NUM_ENVS,), -1, dtype=torch.long, device=raw_env.device
        )
        max_nonfoot_body_index = torch.full_like(max_nonfoot_force_step, -1)
        max_root_angular_speed = torch.zeros(NUM_ENVS, device=raw_env.device)
        max_joint_speed = torch.zeros(NUM_ENVS, device=raw_env.device)
        all_env_termination_counts = {
            name: torch.zeros(NUM_ENVS, dtype=torch.long, device=raw_env.device)
            for name in active_terminations
        }
        for control_step in range(1, ROLLOUT_STEPS + 1):
            previous = getattr(action_term, "_prev_applied_actions", None)
            if previous is None:
                raise RuntimeError("EMA previous target unavailable before step")
            previous = previous.detach().clone()
            env.step(actions)
            termination_flags = {
                name: bool(
                    raw_env.termination_manager.get_term(name)[SOURCE_ENV_INDEX].item()
                )
                for name in active_terminations
            }
            for name in active_terminations:
                all_env_termination_counts[name] += (
                    raw_env.termination_manager.get_term(name).long()
                )
            for name in safety_termination_counts:
                safety_termination_counts[name] += int(
                    termination_flags.get(name, False)
                )
            history = sensor.data.net_forces_w_history
            if history is None:
                raise RuntimeError("contact force history unavailable")
            all_force_magnitudes = history.norm(dim=-1)
            nonfoot_history = all_force_magnitudes[:, :, nonfoot_ids]
            step_force_n, flat_index = nonfoot_history.reshape(NUM_ENVS, -1).max(dim=1)
            step_force_bw = step_force_n.to(torch.float64) / (
                all_env_total_mass_kg * 9.81
            )
            new_force_max = step_force_bw > max_nonfoot_force_bw
            history_slot = flat_index // len(nonfoot_ids)
            nonfoot_offset = flat_index % len(nonfoot_ids)
            body_index_lookup = torch.tensor(
                nonfoot_ids, dtype=torch.long, device=raw_env.device
            )
            step_body_index = body_index_lookup[nonfoot_offset]
            step_physics_index = control_step * CONTROL_DECIMATION - history_slot
            max_nonfoot_force_bw = torch.maximum(max_nonfoot_force_bw, step_force_bw)
            max_nonfoot_force_step = torch.where(
                new_force_max, step_physics_index, max_nonfoot_force_step
            )
            max_nonfoot_body_index = torch.where(
                new_force_max, step_body_index, max_nonfoot_body_index
            )
            max_root_angular_speed = torch.maximum(
                max_root_angular_speed,
                robot.data.root_ang_vel_b.norm(dim=1),
            )
            max_joint_speed = torch.maximum(
                max_joint_speed, robot.data.joint_vel.abs().max(dim=1).values
            )
            physics_rows.extend(
                physics_history_rows(
                    history[SOURCE_ENV_INDEX],
                    control_step=control_step,
                    physics_dt_s=raw_env.physics_dt,
                    body_names=body_names,
                    nonfoot_ids=nonfoot_ids,
                    foot_ids=foot_ids,
                    base_body_id=base_body_id,
                    total_mass_kg=total_mass_kg,
                )
            )
            control_rows.append(
                control_step_row(
                    control_step=control_step,
                    robot=robot,
                    action_term=action_term,
                    action=actions,
                    ema_previous_before=previous,
                    source_env_index=SOURCE_ENV_INDEX,
                    termination_flags=termination_flags,
                    control_dt_s=raw_env.step_dt,
                )
            )
        contact_authority = authority.snapshot(args.device)
        clock_snapshot = physics_clock.snapshot()
        if clock_snapshot["passed"] is not True:
            raise PhysicsStepClockEvidenceError(clock_snapshot, mass_evidence)
        if normalized_device == "cpu":
            require(
                contact_authority["error"] is None,
                f"CPU contact callback failed: {contact_authority['error']}",
            )
            require(
                contact_authority["passed"] is True,
                "CPU contact authority did not cover env7 details and all 8 "
                "environment separation values",
            )
        all_env_separation = contact_authority["all_env_minimum_separation_m"]
        observed_pose_metrics: list[dict[str, Any]] = []
        pose_names = ("prone", "supine", "left_side", "right_side")
        for env_index in range(NUM_ENVS):
            body_index = int(max_nonfoot_body_index[env_index].item())
            observed_pose_metrics.append(
                {
                    "env_index": env_index,
                    "pose_id": pose_names[env_index % 4],
                    "action_mode": (
                        "zero_normalized" if env_index < 4 else "reset_pose_hold"
                    ),
                    "max_nonfoot_force_bodyweights": float(
                        max_nonfoot_force_bw[env_index].item()
                    ),
                    "max_nonfoot_force_physics_step": int(
                        max_nonfoot_force_step[env_index].item()
                    ),
                    "max_nonfoot_force_body_index": body_index,
                    "max_nonfoot_force_body_name": (
                        body_names[body_index] if body_index >= 0 else None
                    ),
                    "max_root_angular_speed_rad_s": float(
                        max_root_angular_speed[env_index].item()
                    ),
                    "max_joint_speed_rad_s": float(max_joint_speed[env_index].item()),
                    "termination_counts": {
                        name: int(counts[env_index].item())
                        for name, counts in all_env_termination_counts.items()
                    },
                    "min_contact_separation_m": (
                        all_env_separation[env_index]
                        if normalized_device == "cpu"
                        else None
                    ),
                }
            )
        historical_summary = build_historical_runtime_summary(
            normalized_arm, normalized_device, observed_pose_metrics
        )
        safety_termination_counts = {
            name: sum(
                metric["termination_counts"].get(name, 0)
                for metric in observed_pose_metrics
            )
            for name in ("numeric_invalid", "hard_joint_limit")
        }
        report = {
            "schema_version": "g009.r0.rev16.backend_divergence.v1",
            "goal_id": "g009",
            "stage_id": "R0",
            "revision": "rev16",
            "status": "complete",
            "diagnostic_only": True,
            "qualification_eligible": False,
            "replicate_index": args.replicate_index,
            "headless": bool(args.headless),
            "device": args.device,
            "runtime_device": str(raw_env.device),
            "seed": args.seed,
            "task": args.task,
            "num_envs": NUM_ENVS,
            "rollout_steps": ROLLOUT_STEPS,
            "finished_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "execution": execution,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
            "source_bundle": source_bundle,
            "predecessor_synthesis": predecessor,
            "governance": governance(),
            "live_physics_readback": {
                "solver": solver_readback,
                "max_depenetration_velocity": depenetration_readback,
                "checks": {**solver_checks, **depenetration_checks},
            },
            "controlled_cell": {
                "source_env_index": SOURCE_ENV_INDEX,
                "pose_id": POSE_ID,
                "action_mode": ACTION_MODE,
                "target_body_index": base_body_id,
                "target_body_name": body_names[base_body_id],
            },
            "pose_action_assignment": {
                "class_ids": observed_class_ids,
                "mapping": expected_pose_action_assignment(),
            },
            "runtime_topology": {
                "force_body_names": body_names,
                "link_body_names": link_body_names,
                "joint_names": list(robot.joint_names),
                "base_force_body_id": base_body_id,
                "foot_force_body_ids": foot_ids,
                "nonfoot_force_body_ids": nonfoot_ids,
                "body_mass_body_names": link_body_names,
                **mass_evidence,
            },
            "telemetry_timing": {
                "physics_dt_s": float(raw_env.physics_dt),
                "control_dt_s": float(raw_env.step_dt),
                "control_decimation": CONTROL_DECIMATION,
                "history_order": "newest_to_oldest",
                "peak_window_radius_physics_steps": 8,
            },
            "physics_substep_telemetry": sorted(
                physics_rows, key=lambda row: row["physics_step"]
            ),
            "control_step_telemetry": control_rows,
            "physics_step_clock": clock_snapshot,
            "active_terminations": active_terminations,
            "safety_termination_counts": safety_termination_counts,
            "cpu_contact_authority": contact_authority,
            "historical_runtime_summary": historical_summary,
            "diagnostic_capture_complete": (
                all(solver_checks.values())
                and all(depenetration_checks.values())
                and len(physics_rows) == ROLLOUT_STEPS * CONTROL_DECIMATION
                and len(control_rows) == ROLLOUT_STEPS
                and clock_snapshot["passed"] is True
                and safety_termination_counts
                == {"numeric_invalid": 0, "hard_joint_limit": 0}
                and historical_summary["passed"] is True
                and (
                    contact_authority["passed"] is True
                    if args.device.lower() == "cpu"
                    else contact_authority["passed"] is None
                    and contact_authority["events"] is None
                )
            ),
        }
        try:
            validate_report_contract(report)
        except Exception as error:
            raise DiagnosticEvidenceError(
                error,
                physics_step_clock=clock_snapshot,
                mass_evidence=mass_evidence,
            ) from error
        return report
    finally:
        contact_subscription = None
        physics_clock_subscription = None
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("A", "B", "a", "b"))
    parser.add_argument("--replicate-index", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--task", default=DEFAULT_TASK, choices=(DEFAULT_TASK,))
    parser.add_argument("--seed", type=int, default=42, choices=(42,))
    parser.add_argument("--predecessor-synthesis", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


def failure_envelope(
    args: argparse.Namespace,
    execution: dict[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    """Build a non-throwing failure report while preserving the first error."""

    envelope_errors: list[str] = []
    try:
        contract = rev16_contract(args.arm, args.device)
    except Exception as envelope_error:  # noqa: BLE001 - never mask original failure
        envelope_errors.append(
            f"contract: {type(envelope_error).__name__}: {envelope_error}"
        )
        contract = None
    try:
        source_bundle = source_bundle_provenance()
    except Exception as envelope_error:  # noqa: BLE001 - never mask original failure
        envelope_errors.append(
            f"source_bundle: {type(envelope_error).__name__}: {envelope_error}"
        )
        source_bundle = {
            "all_files_present": False,
            "clean": False,
            "error": f"{type(envelope_error).__name__}: {envelope_error}",
        }
    original_error = (
        error.original_error
        if isinstance(error, (MassEvidenceError, DiagnosticEvidenceError))
        else error
    )
    clock_evidence = (
        error.evidence
        if isinstance(error, PhysicsStepClockEvidenceError)
        else error.physics_step_clock
        if isinstance(error, DiagnosticEvidenceError)
        else None
    )
    mass_evidence = (
        error.evidence
        if isinstance(error, MassEvidenceError)
        else error.mass_evidence
        if isinstance(error, (PhysicsStepClockEvidenceError, DiagnosticEvidenceError))
        else None
    )
    clock_evidence = _allowlisted_evidence(clock_evidence, CLOCK_EVIDENCE_FIELDS)
    mass_evidence = _allowlisted_evidence(mass_evidence, MASS_EVIDENCE_FIELDS)
    return {
        "schema_version": "g009.r0.rev16.backend_divergence_failure.v1",
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev16",
        "status": "failed_closed",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "replicate_index": args.replicate_index,
        "headless": bool(args.headless),
        "device": args.device,
        "runtime_device": None,
        "seed": args.seed,
        "task": args.task,
        "num_envs": NUM_ENVS,
        "rollout_steps": ROLLOUT_STEPS,
        "finished_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "execution": execution,
        "contract": contract,
        "contract_sha256": canonical_sha256(contract) if contract is not None else None,
        "source_bundle": source_bundle,
        "predecessor_synthesis": None,
        "governance": governance(),
        "diagnostic_capture_complete": False,
        "physics_step_clock": clock_evidence,
        "mass_evidence": mass_evidence,
        "error": {
            "type": type(original_error).__name__,
            "message": str(original_error),
        },
        "failure_envelope_errors": envelope_errors,
    }


def main(argv: list[str] | None = None) -> int:
    output, execution = runtime_probe.prepare_execution(
        runtime_probe.parse_prelaunch_output(argv)
    )
    args = parse_args(argv)
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    app = None
    try:
        try:
            preflight_source = source_bundle_provenance()
            require(
                preflight_source["all_files_present"],
                "source binding files are missing",
            )
            require(preflight_source["clean"], "source binding must be clean")
            validate_predecessor_synthesis(
                args.predecessor_synthesis,
                arm=args.arm.upper(),
                device=args.device.lower(),
                source_bundle=preflight_source,
            )
            app = AppLauncher(args).app
            report = diagnose(args, execution)
        except Exception as error:  # noqa: BLE001 - persist a fail-closed report
            report = failure_envelope(args, execution, error)
        runtime_probe._write_json_atomic(output, report)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "execution_id": execution["execution_id"],
                    "replicate_index": args.replicate_index,
                    "diagnostic_capture_complete": report[
                        "diagnostic_capture_complete"
                    ],
                    "qualification_eligible": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if report["diagnostic_capture_complete"] else 2
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
