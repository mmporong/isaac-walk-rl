#!/usr/bin/env python3
"""Probe whether PhysX raw contact reports are observable on CPU and GPU.

E011 is a diagnostic-only feasibility probe.  It performs exactly 150 manual
physics substeps for the rev17 B cell and never runs PPO, reward computation,
qualification, rendering, or a G009 gate.  ContactSensor and articulation
telemetry are supporting proxies only; they cannot upgrade missing raw GPU
robot-ground pair attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPT_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import probe_g009_recover_runtime as runtime_probe


DEFAULT_TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
SCHEMA_VERSION = "g009.r0.rev18.gpu_raw_contact.v1"
FAILURE_SCHEMA_VERSION = "g009.r0.rev18.gpu_raw_contact_failure.v1"
PREDECESSOR_PATH = (
    REPO_ROOT / "reports" / "runs" / "g009_r0_rev17_mechanism_split_offline_s42.json"
)
PREDECESSOR_SHA256 = (
    "48e596a8e61cf2b4fbcff0b1b6072d62431dc7c4c744b58db9107c398fd1cf97"
)
NUM_ENVS = 8
SOURCE_ENV_INDEX = 7
POSE_ID = "right_side"
ACTION_MODE = "reset_pose_hold"
PHYSICS_SUBSTEPS = 150
PHYSICS_DT_S = 0.005
POSITION_SOLVER_ITERATIONS = 16
VELOCITY_SOLVER_ITERATIONS = 0
AUTHORITY_SCOPE = "physx_contact_report_callback_observation"
_ENV_ROBOT_PATH = re.compile(r"/World/envs/env_(\d+)/Robot(?:/|$)")
SOURCE_BINDING_PATHS = (
    "configs/g009_r0.json",
    "scripts/probe_g009_recover_runtime.py",
    "scripts/probe_g009_r0_rev16_backend_divergence.py",
    "scripts/probe_g009_r0_rev18_gpu_raw_contact.py",
    "reports/runs/g009_r0_rev17_mechanism_split_offline_s42.json",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/recover_contracts.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/registry.py",
)


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def governance() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "headless_required": True,
        "rendering_allowed": False,
        "ppo_updates": 0,
        "reward_computed": False,
        "qualification_eligible": False,
        "gate_execution_allowed": False,
        "learned": False,
        "physics_ground_truth_authority": False,
        "authority_scope": AUTHORITY_SCOPE,
        "proxy_can_upgrade_raw_observation": False,
    }


def probe_contract(device: str, replicate_index: int = 1) -> dict[str, Any]:
    normalized = device.strip().lower()
    require(normalized in {"cpu", "cuda:0"}, "device must be cpu or cuda:0")
    require(replicate_index in {1, 2}, "replicate_index must be 1 or 2")
    return {
        "goal_id": "g009",
        "stage_id": "R0",
        "experiment_id": "G009-5-E011",
        "revision": "rev18",
        "predecessor": {
            "path": PREDECESSOR_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": PREDECESSOR_SHA256,
        },
        "controlled_cell": {
            "arm": "B",
            "solver_position_iterations": POSITION_SOLVER_ITERATIONS,
            "solver_velocity_iterations": VELOCITY_SOLVER_ITERATIONS,
            "seed": 42,
            "num_envs": NUM_ENVS,
            "source_env_index": SOURCE_ENV_INDEX,
            "pose_id": POSE_ID,
            "action_mode": ACTION_MODE,
            "device": normalized,
            "replicate_index": replicate_index,
        },
        "execution": {
            "manual_inner_loop": True,
            "physics_substeps": PHYSICS_SUBSTEPS,
            "physics_dt_s": PHYSICS_DT_S,
            "simulated_duration_s": PHYSICS_SUBSTEPS * PHYSICS_DT_S,
            "headless": True,
            "render": False,
            "ppo_updates": 0,
            "gate_runs": 0,
        },
        "authority": {
            "authority_scope": AUTHORITY_SCOPE,
            "physics_ground_truth_authority": False,
            "gpu_pair_attribution_requires_raw_callback": True,
            "cpu_observation_cannot_substitute_gpu": True,
            "supporting_proxy_cannot_upgrade_raw": True,
        },
    }


def source_bundle_provenance() -> dict[str, Any]:
    commit = current_git_commit()
    dirty = source_binding_status()
    files: dict[str, str] = {}
    missing: list[str] = []
    for relative in SOURCE_BINDING_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        try:
            files[relative] = committed_blob_sha256(relative, commit)
        except (subprocess.CalledProcessError, ValueError):
            missing.append(relative)
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
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
        "clean": not dirty,
        "dirty_source_paths": dirty,
    }


def current_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_binding_status() -> list[str]:
    return subprocess.run(
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


def committed_blob_sha256(relative_path: str, commit: str) -> str:
    """Return the SHA256 of one file exactly as stored in a Git commit."""

    require(relative_path in SOURCE_BINDING_PATHS, "unexpected source bundle path")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "invalid Git commit")
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def validate_source_bundle(bundle: Any) -> dict[str, Any]:
    """Validate exact bundle schema, digest, and every committed file blob."""

    require(isinstance(bundle, dict), "source bundle must be an object")
    bundle = cast(dict[str, Any], bundle)
    expected_keys = {
        "schema_version",
        "git_commit",
        "git_commit_valid",
        "source_binding_paths",
        "source_binding_files",
        "source_bundle_sha256",
        "all_files_present",
        "missing_files",
        "clean",
        "dirty_source_paths",
    }
    require(set(bundle) == expected_keys, "source bundle schema mismatch")
    commit = bundle.get("git_commit")
    require(
        isinstance(commit, str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", commit))
        and bundle.get("git_commit_valid") is True,
        "source bundle commit identity mismatch",
    )
    require(
        bundle.get("source_binding_paths") == list(SOURCE_BINDING_PATHS),
        "source bundle path order mismatch",
    )
    files = bundle.get("source_binding_files")
    require(
        isinstance(files, dict)
        and list(files) == list(SOURCE_BINDING_PATHS)
        and set(files) == set(SOURCE_BINDING_PATHS),
        "source bundle file key set mismatch",
    )
    files = cast(dict[str, Any], files)
    require(
        all(
            isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))
            for value in files.values()
        ),
        "source bundle file hash format mismatch",
    )
    require(
        bundle.get("all_files_present") is True
        and bundle.get("missing_files") == []
        and bundle.get("clean") is True
        and bundle.get("dirty_source_paths") == [],
        "source bundle must be complete and clean",
    )
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    expected_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    require(
        bundle.get("source_bundle_sha256") == expected_digest,
        "source bundle aggregate SHA256 mismatch",
    )
    for relative_path in SOURCE_BINDING_PATHS:
        require(
            files[relative_path] == committed_blob_sha256(relative_path, commit),
            f"source bundle committed blob mismatch: {relative_path}",
        )
    return bundle


def validate_predecessor() -> dict[str, str]:
    require(PREDECESSOR_PATH.is_file(), "rev17 predecessor report is missing")
    actual = _sha256(PREDECESSOR_PATH)
    require(actual == PREDECESSOR_SHA256, "rev17 predecessor SHA256 mismatch")
    return {
        "path": PREDECESSOR_PATH.relative_to(REPO_ROOT).as_posix(),
        "sha256": actual,
    }


def _finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def _vector(value: Any, size: int, label: str) -> list[float]:
    if hasattr(value, "x") and size == 3:
        value = [value.x, value.y, value.z]
    require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)), f"{label} must be a vector")
    sequence = cast(Sequence[Any], value)
    require(len(sequence) == size, f"{label} must contain {size} values")
    return [_finite_number(component, label) for component in sequence]


def _tensor_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _absolute_path(value: Any, int_to_path: Callable[[Any], Any]) -> str:
    path = str(int_to_path(value))
    require(path.startswith("/"), "contact pair paths must be absolute USD paths")
    return path


def _event_name(value: Any, contact_event_types: Any | None) -> str:
    if contact_event_types is None:
        return "CONTACT_PERSIST"
    require(value is not None, "contact event type is missing")
    for name in ("CONTACT_FOUND", "CONTACT_PERSIST", "CONTACT_LOST"):
        if value == getattr(contact_event_types, name, object()):
            return name
    raise ValueError(f"unsupported contact event type: {value}")


def _is_ground_path(path: str) -> bool:
    return path == "/World/ground" or path.startswith("/World/ground/")


def copy_contact_callback(
    contact_headers: Any,
    contact_data: Any,
    *,
    physics_step: int,
    int_to_path: Callable[[Any], Any],
    contact_event_types: Any | None,
) -> dict[str, Any]:
    """Copy raw callback buffers and reject malformed robot-ground data."""

    require(type(physics_step) is int and physics_step >= 0, "invalid callback physics step")
    headers: list[dict[str, Any]] = []
    robot_ground_datum_count = 0
    for header in contact_headers:
        paths = {
            "actor0_path": _absolute_path(header.actor0, int_to_path),
            "actor1_path": _absolute_path(header.actor1, int_to_path),
            "collider0_path": _absolute_path(header.collider0, int_to_path),
            "collider1_path": _absolute_path(header.collider1, int_to_path),
        }
        path_values = list(paths.values())
        ground = any(_is_ground_path(path) for path in path_values)
        match = next(
            (_ENV_ROBOT_PATH.search(path) for path in path_values if _ENV_ROBOT_PATH.search(path)),
            None,
        )
        if not ground or match is None:
            continue
        env_index = int(match.group(1))
        require(0 <= env_index < NUM_ENVS, "contact env index is outside the scene")
        event_type = _event_name(getattr(header, "type", None), contact_event_types)
        start = int(header.contact_data_offset)
        count = int(header.num_contact_data)
        require(start >= 0 and count >= 0, "contact data range is invalid")
        require(
            start + count <= len(contact_data),
            "contact data range exceeds the callback buffer",
        )
        points: list[dict[str, Any]] = []
        for datum in contact_data[start : start + count]:
            position = _vector(getattr(datum, "position", None), 3, "contact position")
            normal = _vector(getattr(datum, "normal", None), 3, "contact normal")
            impulse = _vector(getattr(datum, "impulse", None), 3, "contact impulse")
            separation = _finite_number(getattr(datum, "separation", None), "contact separation")
            normal_norm = math.sqrt(math.fsum(component * component for component in normal))
            require(math.isclose(normal_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-3), "contact normal must be unit length")
            points.append(
                {
                    "position_w_m": position,
                    "normal_w": normal,
                    "impulse_n_s": impulse,
                    "separation_m": separation,
                }
            )
        require(event_type == "CONTACT_LOST" or len(points) == count, "contact callback data slice is incomplete")
        require(event_type != "CONTACT_LOST" or not points, "CONTACT_LOST must not contain contact data")
        robot_ground_datum_count += len(points)
        headers.append({"env_index": env_index, "event_type": event_type, **paths, "contact_points": points})
    return {
        "physics_step": physics_step,
        "robot_ground_header_count": len(headers),
        "robot_ground_datum_count": robot_ground_datum_count,
        "headers": headers,
    }


class RawContactAccumulator:
    """Copy ephemeral PhysX contact buffers and preserve callback failures."""

    def __init__(
        self,
        int_to_path: Callable[[Any], Any],
        step_reader: Callable[[], int],
        contact_event_types: Any | None,
    ) -> None:
        self._int_to_path = int_to_path
        self._step_reader = step_reader
        self._contact_event_types = contact_event_types
        self.subscription_attempted = False
        self.subscription_succeeded = False
        self.subscription_error: str | None = None
        self.callback_count = 0
        self.malformed_callback_count = 0
        self.first_callback_error: str | None = None
        self.events: list[dict[str, Any]] = []

    def mark_subscription(self, holder: Any) -> None:
        self.subscription_attempted = True
        self.subscription_succeeded = holder is not None
        if holder is None:
            self.subscription_error = "RuntimeError: subscribe_contact_report_events returned no holder"

    def mark_subscription_error(self, error: BaseException) -> None:
        self.subscription_attempted = True
        self.subscription_succeeded = False
        self.subscription_error = f"{type(error).__name__}: {error}"

    def __call__(self, contact_headers: Any, contact_data: Any) -> None:
        self.callback_count += 1
        try:
            event = copy_contact_callback(
                contact_headers,
                contact_data,
                physics_step=self._step_reader(),
                int_to_path=self._int_to_path,
                contact_event_types=self._contact_event_types,
            )
        except Exception as error:  # callback errors are evidence, not crashes
            self.malformed_callback_count += 1
            if self.first_callback_error is None:
                self.first_callback_error = f"{type(error).__name__}: {error}"
            return
        self.events.append(event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "authority_scope": AUTHORITY_SCOPE,
            "physics_ground_truth_authority": False,
            "subscription_attempted": self.subscription_attempted,
            "subscription_succeeded": self.subscription_succeeded,
            "subscription_error": self.subscription_error,
            "callback_count": self.callback_count,
            "malformed_callback_count": self.malformed_callback_count,
            "first_callback_error": self.first_callback_error,
            "events": list(self.events),
        }


class PhysicsStepCounter:
    def __init__(self) -> None:
        self.current_step = 0
        self.observed_dt_s: list[float] = []

    def __call__(self, dt_s: float) -> None:
        self.current_step += 1
        self.observed_dt_s.append(float(dt_s))

    def snapshot(self) -> dict[str, Any]:
        finite = all(math.isfinite(value) for value in self.observed_dt_s)
        dt_match = finite and all(
            math.isclose(value, PHYSICS_DT_S, rel_tol=0.0, abs_tol=2.5e-10)
            for value in self.observed_dt_s
        )
        return {
            "source": "subscribe_physics_on_step_events(pre_step=true,order=0)",
            "callback_count": self.current_step,
            "expected_callback_count": PHYSICS_SUBSTEPS,
            "observed_dt_s": list(self.observed_dt_s),
            "passed": self.current_step == PHYSICS_SUBSTEPS and dt_match,
        }


def _proxy_row(
    *,
    physics_step: int,
    sensor: Any,
    robot: Any,
    residual_reader: "ResidualReader",
) -> dict[str, Any]:
    net_all = _tensor_list(sensor.data.net_forces_w)
    require(isinstance(net_all, list) and len(net_all) == NUM_ENVS, "net force env dimension mismatch")
    net_source = net_all[SOURCE_ENV_INDEX]
    force_matrix_value = getattr(sensor.data, "force_matrix_w", None)
    if force_matrix_value is None:
        force_matrix = {"status": "unavailable", "value": None, "error": "force_matrix_w is None"}
    else:
        matrix_all = _tensor_list(force_matrix_value)
        require(isinstance(matrix_all, list) and len(matrix_all) == NUM_ENVS, "force matrix env dimension mismatch")
        force_matrix = {"status": "observed", "value": matrix_all[SOURCE_ENV_INDEX], "error": None}
    wrench_all = _tensor_list(robot.data.body_incoming_joint_wrench_b)
    require(isinstance(wrench_all, list) and len(wrench_all) == NUM_ENVS, "incoming joint wrench env dimension mismatch")
    return {
        "physics_step": physics_step,
        "time_s": physics_step * PHYSICS_DT_S,
        "contact_sensor": {
            "net_forces_w_n": net_source,
            "force_matrix_w": force_matrix,
        },
        "incoming_joint_wrench_b": wrench_all[SOURCE_ENV_INDEX],
        "solver_residual": residual_reader.read(),
    }


class ResidualReader:
    """Observe pre-authored residual APIs without mutating live physics state."""

    _ATTRS = {
        "position_rms": "GetPhysxResidualReportingRmsResidualPositionIterationAttr",
        "position_max": "GetPhysxResidualReportingMaxResidualPositionIterationAttr",
        "velocity_rms": "GetPhysxResidualReportingRmsResidualVelocityIterationAttr",
        "velocity_max": "GetPhysxResidualReportingMaxResidualVelocityIterationAttr",
    }

    def __init__(
        self,
        physics_context: Any,
        scene_prim: Any,
        source_root_prim: Any,
        PhysxSchema: Any,
    ) -> None:
        self._scene_api = None
        self._root_api = None
        self.capability: dict[str, Any] = {
            "lifecycle_policy": (
                "observe_existing_only_after_articulation_tensor_view_initialization"
            ),
            "mutation_attempted": False,
            "physics_context_enable_attempted": False,
            "api_apply_attempted": False,
            "scene": {"status": "unavailable", "prim_path": None, "error": None},
            "source_articulation_root": {
                "status": "unavailable",
                "prim_path": None,
                "error": None,
            },
        }
        del physics_context
        self._scene_api = self._get_existing(
            scene_prim, PhysxSchema, self.capability["scene"]
        )
        self._root_api = self._get_existing(
            source_root_prim,
            PhysxSchema,
            self.capability["source_articulation_root"],
        )

    @staticmethod
    def _prim_path(prim: Any) -> str | None:
        try:
            path = str(prim.GetPath())
        except Exception:
            return None
        return path if path.startswith("/") else None

    def _get_existing(
        self, prim: Any, PhysxSchema: Any, state: dict[str, Any]
    ) -> Any:
        path = self._prim_path(prim)
        state["prim_path"] = path
        if path is None:
            state["error"] = "invalid USD prim"
            return None
        try:
            api = PhysxSchema.PhysxResidualReportingAPI.Get(
                prim.GetStage(), prim.GetPath()
            )
            require(
                bool(api),
                "pre-authored PhysxResidualReportingAPI is unavailable",
            )
            state["status"] = "preauthored_observed"
            return api
        except Exception as error:
            state["error"] = f"{type(error).__name__}: {error}"
            return None

    @classmethod
    def _read_api(cls, api: Any, label: str) -> dict[str, float]:
        values: dict[str, float] = {}
        for key, method_name in cls._ATTRS.items():
            method = getattr(api, method_name, None)
            require(callable(method), f"{label} residual attribute API missing: {method_name}")
            attribute = cast(Callable[[], Any], method)()
            values[key] = _finite_number(attribute.Get(), f"{label} {key}")
            require(values[key] >= 0.0, f"{label} residuals must be nonnegative")
        return values

    def read(self) -> dict[str, Any]:
        if self._scene_api is None or self._root_api is None:
            return {
                "status": "unavailable",
                "samples": None,
                "scene": None,
                "source_articulation_root": None,
                "error": "scene or source articulation residual API unavailable",
            }
        try:
            return {
                "status": "observed",
                "samples": "usd_physx_residual_reporting_api",
                "scene": self._read_api(self._scene_api, "scene"),
                "source_articulation_root": self._read_api(
                    self._root_api, "source articulation root"
                ),
                "error": None,
            }
        except Exception as error:
            return {
                "status": "read_error",
                "samples": None,
                "scene": None,
                "source_articulation_root": None,
                "error": f"{type(error).__name__}: {error}",
            }


def _all_finite_nested(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value) and all(_all_finite_nested(item) for item in value)
    return type(value) in (int, float) and math.isfinite(float(value))


def _finite_matrix(value: Any, rows: int, columns: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == rows
        and all(
            isinstance(row, list)
            and len(row) == columns
            and all(
                type(component) in (int, float)
                and math.isfinite(float(component))
                for component in row
            )
            for row in value
        )
    )


def _raw_event_counts_consistent(event: Any) -> bool:
    if not isinstance(event, Mapping) or set(event) != {
        "physics_step",
        "robot_ground_header_count",
        "robot_ground_datum_count",
        "headers",
    }:
        return False
    headers = event.get("headers")
    if (
        type(event.get("physics_step")) is not int
        or type(event.get("robot_ground_header_count")) is not int
        or type(event.get("robot_ground_datum_count")) is not int
        or not isinstance(headers, list)
    ):
        return False
    header_keys = {
        "env_index",
        "event_type",
        "actor0_path",
        "actor1_path",
        "collider0_path",
        "collider1_path",
        "contact_points",
    }
    point_keys = {"position_w_m", "normal_w", "impulse_n_s", "separation_m"}
    if not all(
        isinstance(header, Mapping)
        and set(header) == header_keys
        and type(header.get("env_index")) is int
        and header.get("event_type")
        in {"CONTACT_FOUND", "CONTACT_PERSIST", "CONTACT_LOST"}
        and isinstance(header.get("contact_points"), list)
        and all(
            isinstance(point, Mapping) and set(point) == point_keys
            for point in header["contact_points"]
        )
        for header in headers
    ):
        return False
    return (
        event["robot_ground_header_count"] == len(headers)
        and event["robot_ground_datum_count"]
        == sum(len(header["contact_points"]) for header in headers)
    )


def derive_feasibility(report: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed recomputation; serialized success fields are ignored."""

    device = str(report.get("device", "")).lower()
    raw = report.get("raw_contact_observation")
    rows = report.get("supporting_telemetry")
    clock = report.get("physics_step_clock")
    readback = report.get("device_readback")
    expected_gpu_enabled = device == "cuda:0"
    readback_schema_valid = (
        isinstance(readback, Mapping)
        and set(readback)
        == {
            "requested_device",
            "runtime_device",
            "physics_scene_prim_path",
            "gpu_dynamics_enabled",
            "gpu_dynamics_matches_device",
            "error",
        }
        and readback.get("requested_device") == device
        and readback.get("runtime_device") == device
        and isinstance(readback.get("physics_scene_prim_path"), str)
        and cast(str, readback.get("physics_scene_prim_path")).startswith("/")
        and type(readback.get("gpu_dynamics_enabled")) is bool
        and type(readback.get("gpu_dynamics_matches_device")) is bool
        and readback.get("error") is None
    )
    recomputed_gpu_match = (
        readback_schema_valid
        and isinstance(readback, Mapping)
        and readback.get("gpu_dynamics_enabled") is expected_gpu_enabled
    )
    observed_dt = clock.get("observed_dt_s") if isinstance(clock, Mapping) else None
    clock_schema_valid = (
        isinstance(clock, Mapping)
        and set(clock)
        == {
            "source",
            "callback_count",
            "expected_callback_count",
            "observed_dt_s",
            "passed",
        }
        and clock.get("source")
        == "subscribe_physics_on_step_events(pre_step=true,order=0)"
        and type(clock.get("passed")) is bool
    )
    clock_recomputed = (
        clock_schema_valid
        and isinstance(clock, Mapping)
        and clock.get("callback_count") == PHYSICS_SUBSTEPS
        and clock.get("expected_callback_count") == PHYSICS_SUBSTEPS
        and isinstance(observed_dt, list)
        and len(observed_dt) == PHYSICS_SUBSTEPS
        and all(
            type(value) in (int, float)
            and math.isfinite(float(value))
            and math.isclose(
                float(value), PHYSICS_DT_S, rel_tol=0.0, abs_tol=2.5e-10
            )
            for value in observed_dt
        )
    )
    raw_schema_valid = (
        isinstance(raw, Mapping)
        and set(raw)
        == {
            "authority_scope",
            "physics_ground_truth_authority",
            "subscription_attempted",
            "subscription_succeeded",
            "subscription_error",
            "callback_count",
            "malformed_callback_count",
            "first_callback_error",
            "events",
        }
        and raw.get("authority_scope") == AUTHORITY_SCOPE
        and raw.get("physics_ground_truth_authority") is False
        and type(raw.get("callback_count")) is int
        and type(raw.get("malformed_callback_count")) is int
        and cast(int, raw.get("callback_count")) >= 0
        and 0
        <= cast(int, raw.get("malformed_callback_count"))
        <= cast(int, raw.get("callback_count"))
        and isinstance(raw.get("events"), list)
        and cast(int, raw.get("callback_count"))
        == len(cast(list[Any], raw.get("events")))
        + cast(int, raw.get("malformed_callback_count"))
        and (
            (raw.get("malformed_callback_count") == 0 and raw.get("first_callback_error") is None)
            or (
                cast(int, raw.get("malformed_callback_count")) > 0
                and isinstance(raw.get("first_callback_error"), str)
                and bool(raw.get("first_callback_error"))
            )
        )
        and type(raw.get("subscription_attempted")) is bool
        and type(raw.get("subscription_succeeded")) is bool
        and (
            (
                raw.get("subscription_succeeded") is True
                and raw.get("subscription_error") is None
            )
            or (
                raw.get("subscription_succeeded") is False
                and isinstance(raw.get("subscription_error"), str)
                and bool(raw.get("subscription_error"))
            )
        )
    )
    checks: dict[str, bool] = {
        "device_supported": device in {"cpu", "cuda:0"},
        "raw_subscription_attempted": isinstance(raw, Mapping) and raw.get("subscription_attempted") is True,
        "raw_subscription_succeeded": isinstance(raw, Mapping) and raw.get("subscription_succeeded") is True,
        "raw_snapshot_counts_consistent": raw_schema_valid,
        "raw_callback_well_formed": raw_schema_valid
        and isinstance(raw, Mapping)
        and raw.get("malformed_callback_count") == 0,
        "exact_150_physics_steps": clock_recomputed,
        "force_proxy_complete": isinstance(rows, list)
        and len(rows) == PHYSICS_SUBSTEPS,
        "joint_wrench_complete": isinstance(rows, list)
        and len(rows) == PHYSICS_SUBSTEPS,
        "residual_bundle_complete": isinstance(rows, list)
        and len(rows) == PHYSICS_SUBSTEPS,
        "device_readback_matches": recomputed_gpu_match
        and isinstance(readback, Mapping)
        and readback.get("gpu_dynamics_matches_device") is recomputed_gpu_match,
        "manual_inner_loop_contract": False,
        "positive_force_stimulus_present": False,
    }
    source_points: list[tuple[int, Mapping[str, Any]]] = []
    event_steps: list[int] = []
    if isinstance(raw, Mapping) and isinstance(raw.get("events"), list):
        for event in raw["events"]:
            if (
                not _raw_event_counts_consistent(event)
            ):
                checks["raw_snapshot_counts_consistent"] = False
                checks["raw_callback_well_formed"] = False
                continue
            event_steps.append(event["physics_step"])
            headers = event.get("headers")
            if not isinstance(headers, list):
                checks["raw_callback_well_formed"] = False
                continue
            for header in headers:
                if not isinstance(header, Mapping) or header.get("env_index") != SOURCE_ENV_INDEX:
                    continue
                for point in header.get("contact_points", []):
                    if isinstance(point, Mapping):
                        source_points.append((event["physics_step"], point))
    checks["raw_steps_monotonic_aligned"] = bool(event_steps) and event_steps == sorted(event_steps) and all(1 <= step <= PHYSICS_SUBSTEPS for step in event_steps)
    checks["nonempty_source_robot_ground_datum"] = bool(source_points)
    checks["absolute_pair_paths"] = False
    checks["source_robot_ground_pair_paths"] = False
    if isinstance(raw, Mapping) and isinstance(raw.get("events"), list):
        relevant_headers = [
            header
            for event in raw["events"]
            if isinstance(event, Mapping) and isinstance(event.get("headers"), list)
            for header in event["headers"]
            if isinstance(header, Mapping) and header.get("env_index") == SOURCE_ENV_INDEX
        ]
        checks["absolute_pair_paths"] = bool(relevant_headers) and all(
            all(isinstance(header.get(key), str) and header[key].startswith("/") for key in ("actor0_path", "actor1_path", "collider0_path", "collider1_path"))
            for header in relevant_headers
        )
        checks["source_robot_ground_pair_paths"] = bool(relevant_headers) and all(
            any(
                isinstance(header.get(key), str)
                and _is_ground_path(header[key])
                for key in (
                    "actor0_path",
                    "actor1_path",
                    "collider0_path",
                    "collider1_path",
                )
            )
            and any(
                isinstance(header.get(key), str)
                and _ENV_ROBOT_PATH.search(header[key]) is not None
                and int(cast(re.Match[str], _ENV_ROBOT_PATH.search(header[key])).group(1))
                == SOURCE_ENV_INDEX
                for key in (
                    "actor0_path",
                    "actor1_path",
                    "collider0_path",
                    "collider1_path",
                )
            )
            for header in relevant_headers
        )
    checks["finite_raw_vectors_and_separation"] = bool(source_points) and all(
        _all_finite_nested(point.get("position_w_m"))
        and _all_finite_nested(point.get("normal_w"))
        and _all_finite_nested(point.get("impulse_n_s"))
        and type(point.get("separation_m")) in (int, float)
        and math.isfinite(float(point["separation_m"]))
        for _, point in source_points
    )
    checks["unit_normals"] = bool(source_points) and all(
        math.isclose(
            math.sqrt(math.fsum(float(v) ** 2 for v in cast(list[Any], point["normal_w"]))),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-3,
        )
        for _, point in source_points
        if isinstance(point.get("normal_w"), list) and len(point["normal_w"]) == 3
    ) and all(isinstance(point.get("normal_w"), list) and len(point["normal_w"]) == 3 for _, point in source_points)
    checks["nonzero_impulse"] = bool(source_points) and any(
        math.sqrt(math.fsum(float(v) ** 2 for v in cast(list[Any], point["impulse_n_s"]))) > 0.0
        for _, point in source_points
        if isinstance(point.get("impulse_n_s"), list) and len(point["impulse_n_s"]) == 3
    )
    manual = report.get("manual_inner_loop")
    expected_action_steps = list(range(1, PHYSICS_SUBSTEPS + 1, 4))
    checks["manual_inner_loop_contract"] = (
        isinstance(manual, Mapping)
        and manual.get("control_decimation") == 4
        and manual.get("action_process_steps") == expected_action_steps
        and manual.get("action_process_count") == len(expected_action_steps) == 38
        and manual.get("manager_post_step_executed") is False
        and manual.get("reward_computed") is False
        and manual.get("termination_computed") is False
        and manual.get("trajectory_equivalence_claimed") is False
        and manual.get("scope") == "capability_only"
    )
    positive_force_steps: set[int] = set()
    if isinstance(rows, list) and len(rows) == PHYSICS_SUBSTEPS:
        checks["force_proxy_complete"] = all(
            isinstance(row, Mapping)
            and row.get("physics_step") == index
            and isinstance(row.get("contact_sensor"), Mapping)
            and _finite_matrix(row["contact_sensor"].get("net_forces_w_n"), 19, 3)
            for index, row in enumerate(rows, 1)
        )
        checks["joint_wrench_complete"] = all(
            isinstance(row, Mapping)
            and _finite_matrix(row.get("incoming_joint_wrench_b"), 19, 6)
            for row in rows
        )
        checks["residual_bundle_complete"] = all(
            isinstance(row, Mapping)
            and isinstance(row.get("solver_residual"), Mapping)
            and row["solver_residual"].get("status") == "observed"
            and _all_finite_nested(list(row["solver_residual"]["scene"].values()))
            and _all_finite_nested(
                list(row["solver_residual"]["source_articulation_root"].values())
            )
            for row in rows
        )
        for row in rows:
            if (
                isinstance(row, Mapping)
                and type(row.get("physics_step")) is int
                and isinstance(row.get("contact_sensor"), Mapping)
                and _finite_matrix(
                    row["contact_sensor"].get("net_forces_w_n"), 19, 3
                )
            ):
                matrix = cast(
                    list[list[float]],
                    row["contact_sensor"]["net_forces_w_n"],
                )
                if any(
                    math.sqrt(math.fsum(float(value) ** 2 for value in vector))
                    > 0.0
                    for vector in matrix
                ):
                    positive_force_steps.add(cast(int, row["physics_step"]))
    raw_point_steps = {step for step, _ in source_points}
    checks["positive_force_stimulus_present"] = bool(positive_force_steps)
    checks["raw_force_step_overlap"] = bool(
        raw_point_steps.intersection(positive_force_steps)
    )
    raw_passed = all(
        checks[name]
        for name in (
            "raw_subscription_attempted",
            "raw_subscription_succeeded",
            "raw_callback_well_formed",
            "raw_steps_monotonic_aligned",
            "nonempty_source_robot_ground_datum",
            "absolute_pair_paths",
            "source_robot_ground_pair_paths",
            "finite_raw_vectors_and_separation",
            "unit_normals",
            "nonzero_impulse",
            "raw_force_step_overlap",
        )
    )
    probe_valid = all(
        checks[name]
        for name in (
            "device_supported",
            "exact_150_physics_steps",
            "force_proxy_complete",
            "device_readback_matches",
            "manual_inner_loop_contract",
            "positive_force_stimulus_present",
        )
    )
    bundle_complete = checks["joint_wrench_complete"] and checks["residual_bundle_complete"]
    run_feasible = raw_passed and probe_valid and bundle_complete
    return {
        "checks": checks,
        "raw_observation_passed": raw_passed,
        "probe_valid": probe_valid,
        "supporting_bundle_complete": bundle_complete,
        "run_feasible": run_feasible,
        "gpu_pair_attribution_available": (
            raw_passed and probe_valid if device == "cuda:0" else False
        ),
        "authority_scope": AUTHORITY_SCOPE,
        "physics_ground_truth_authority": False,
    }


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    require(
        set(report)
        == {
            "schema_version",
            "goal_id",
            "stage_id",
            "experiment_id",
            "revision",
            "status",
            "headless",
            "device",
            "seed",
            "replicate_index",
            "num_envs",
            "source_env_index",
            "physics_substeps",
            "physics_dt_s",
            "manual_inner_loop",
            "finished_at_utc",
            "execution",
            "contract",
            "contract_sha256",
            "predecessor",
            "source_bundle",
            "governance",
            "pose_action_assignment",
            "live_physics_readback",
            "device_readback",
            "residual_capability",
            "physics_step_clock",
            "raw_contact_observation",
            "supporting_telemetry",
            "feasibility",
        },
        "report top-level field set mismatch",
    )
    require(report.get("schema_version") == SCHEMA_VERSION, "report schema mismatch")
    require(report.get("goal_id") == "g009", "goal identity mismatch")
    require(report.get("stage_id") == "R0", "stage identity mismatch")
    require(report.get("experiment_id") == "G009-5-E011", "experiment identity mismatch")
    require(report.get("revision") == "rev18", "revision identity mismatch")
    require(report.get("status") == "complete", "report must be complete")
    require(report.get("headless") is True, "probe must be headless")
    require(report.get("seed") == 42, "seed must be 42")
    require(report.get("num_envs") == NUM_ENVS, "num_envs must be 8")
    require(
        report.get("source_env_index") == SOURCE_ENV_INDEX,
        "source env index must be 7",
    )
    require(report.get("physics_substeps") == PHYSICS_SUBSTEPS, "exactly 150 physics substeps are required")
    require(report.get("physics_dt_s") == PHYSICS_DT_S, "physics dt mismatch")
    require(report.get("governance") == governance(), "governance must remain closed")
    replicate_index = report.get("replicate_index")
    require(replicate_index in {1, 2}, "replicate_index must be 1 or 2")
    contract = probe_contract(str(report.get("device")), cast(int, replicate_index))
    require(report.get("contract") == contract, "probe contract mismatch")
    require(report.get("contract_sha256") == canonical_sha256(contract), "contract SHA256 mismatch")
    predecessor = report.get("predecessor")
    require(
        predecessor
        == {
            "path": PREDECESSOR_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": PREDECESSOR_SHA256,
        },
        "predecessor binding mismatch",
    )
    require(
        report.get("pose_action_assignment")
        == {"class_ids": [0, 1, 2, 3, 0, 1, 2, 3]},
        "pose/action assignment mismatch",
    )
    finished = report.get("finished_at_utc")
    require(
        isinstance(finished, str) and finished.endswith("Z"),
        "finished_at_utc must be UTC",
    )
    try:
        finished_time = datetime.fromisoformat(cast(str, finished)[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("finished_at_utc must be UTC") from error
    require(
        finished_time.utcoffset() == timezone.utc.utcoffset(finished_time),
        "finished_at_utc must be UTC",
    )
    require(
        isinstance(report.get("live_physics_readback"), Mapping),
        "live physics readback is required",
    )
    require(
        isinstance(report.get("residual_capability"), Mapping),
        "residual capability is required",
    )
    validate_execution_metadata(
        report.get("execution"), str(report.get("device")), cast(int, replicate_index)
    )
    validate_source_bundle(report.get("source_bundle"))
    derived = derive_feasibility(report)
    require(
        derived["checks"]["raw_snapshot_counts_consistent"] is True,
        "raw snapshot self-report mismatch",
    )
    require(report.get("feasibility") == derived, "serialized feasibility differs from fail-closed recomputation")
    return derived


def _gpu_dynamics_readback(
    PhysxSchema: Any, device: str, physics_scene_prim: Any
) -> dict[str, Any]:
    requested_gpu = device == "cuda:0"
    scene_path = ResidualReader._prim_path(physics_scene_prim)
    if scene_path is None:
        matches = False
        observed = None
        error = "physics scene prim unavailable"
    else:
        try:
            api = PhysxSchema.PhysxSceneAPI.Get(
                physics_scene_prim.GetStage(), physics_scene_prim.GetPath()
            )
            raw_observed = api.GetEnableGPUDynamicsAttr().Get()
            if type(raw_observed) is bool:
                observed = raw_observed
                matches = observed is requested_gpu
                error = None
            else:
                observed = None
                matches = False
                error = "enableGPUDynamics readback is not a bool"
        except Exception as exc:  # evidence remains fail-closed
            observed = None
            matches = False
            error = f"{type(exc).__name__}: {exc}"
    return {
        "requested_device": device,
        "runtime_device": None,
        "physics_scene_prim_path": scene_path,
        "gpu_dynamics_enabled": observed,
        "gpu_dynamics_matches_device": matches,
        "error": error,
    }


def expected_output_relative(device: str, replicate_index: int) -> str:
    normalized = device.strip().lower()
    require(normalized in {"cpu", "cuda:0"}, "device must be cpu or cuda:0")
    require(replicate_index in {1, 2}, "replicate_index must be 1 or 2")
    label = "cpu" if normalized == "cpu" else "gpu"
    return (
        f"reports/runs/g009_r0_rev18_raw_contact_{label}_"
        f"rep0{replicate_index}_s42.json"
    )


def validate_execution_metadata(
    execution: Any, device: str, replicate_index: int
) -> dict[str, Any]:
    require(isinstance(execution, dict), "execution metadata must be an object")
    execution = cast(dict[str, Any], execution)
    require(
        set(execution)
        == {
            "execution_id",
            "started_at_utc",
            "output_path_repo_relative",
            "no_overwrite",
        },
        "execution metadata key set mismatch",
    )
    execution_id = execution.get("execution_id")
    require(isinstance(execution_id, str), "execution_id must be UUID4 lowercase hex")
    try:
        parsed_uuid = uuid.UUID(hex=cast(str, execution_id))
    except (ValueError, AttributeError) as error:
        raise ValueError("execution_id must be UUID4 lowercase hex") from error
    require(
        parsed_uuid.version == 4 and parsed_uuid.hex == execution_id,
        "execution_id must be UUID4 lowercase hex",
    )
    started = execution.get("started_at_utc")
    require(
        isinstance(started, str) and started.endswith("Z"),
        "started_at_utc must be an ISO-8601 UTC timestamp",
    )
    try:
        parsed_time = datetime.fromisoformat(cast(str, started)[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("started_at_utc must be an ISO-8601 UTC timestamp") from error
    require(
        parsed_time.utcoffset() == timezone.utc.utcoffset(parsed_time),
        "started_at_utc must use UTC",
    )
    require(
        execution.get("output_path_repo_relative")
        == expected_output_relative(device, replicate_index),
        "execution output binding mismatch",
    )
    require(execution.get("no_overwrite") is True, "execution must be no-overwrite")
    return execution


def diagnose(args: argparse.Namespace, execution: dict[str, Any]) -> dict[str, Any]:
    require(args.device.lower() in {"cpu", "cuda:0"}, "device must be cpu or cuda:0")
    require(args.seed == 42, "seed must be 42")
    require(args.replicate_index in {1, 2}, "replicate_index must be 1 or 2")
    require(args.task == DEFAULT_TASK, "task is fixed")
    require(bool(args.headless), "E011 must run headless")
    device = args.device.lower()
    source_bundle = source_bundle_provenance()
    require(source_bundle["all_files_present"], "source bundle files are missing")
    require(source_bundle["clean"], "source bundle must be clean")
    validate_source_bundle(source_bundle)
    predecessor = validate_predecessor()

    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    import isaaclab_tasks  # noqa: F401  # pyright: ignore[reportMissingImports]
    import omni.usd  # pyright: ignore[reportMissingImports]
    import torch
    from isaaclab import sim as sim_utils  # pyright: ignore[reportMissingImports]
    from isaaclab_tasks.utils import parse_env_cfg  # pyright: ignore[reportMissingImports]
    from omni.physx import get_physx_interface, get_physx_simulation_interface  # pyright: ignore[reportMissingImports]
    from omni.physx.bindings._physx import ContactEventType  # pyright: ignore[reportMissingImports]
    from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics  # pyright: ignore[reportMissingImports]
    from isaac_walk_g009 import register_tasks

    register_tasks()
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=NUM_ENVS)
    env_cfg.seed = 42
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.contact_forces.history_length = 1
    env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = POSITION_SOLVER_ITERATIONS
    env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = VELOCITY_SOLVER_ITERATIONS
    env_cfg.events.reset_base.params.update({"assignment_mode": "stratified", "pose_xy_range": (0.0, 0.0), "yaw_range": (0.0, 0.0)})
    env_cfg.validate()
    env = gym.make(args.task, cfg=env_cfg)
    raw_env: Any = env.unwrapped
    robot = raw_env.scene["robot"]
    sensor = raw_env.scene.sensors["contact_forces"]
    action_term = raw_env.action_manager.get_term("joint_pos")
    stage = omni.usd.get_context().get_stage()
    solver_readback = runtime_probe.articulation_solver_iteration_readback(stage, list(robot.root_physx_view.prim_paths), PhysxSchema)
    depenetration_readback = runtime_probe.rigid_body_max_depenetration_velocity_readback(
        stage,
        sim_utils.find_matching_prim_paths(robot.cfg.prim_path, stage),
        list(robot.root_physx_view.prim_paths),
        [list(group) for group in robot.root_physx_view.link_paths],
        list(robot.body_names),
        PhysxSchema,
        UsdPhysics,
    )
    physics_context = raw_env.sim._physics_context
    physics_scene_prim = physics_context.get_current_physics_scene_prim()
    if isinstance(physics_scene_prim, str):
        physics_scene_prim = stage.GetPrimAtPath(physics_scene_prim)
    device_readback = _gpu_dynamics_readback(
        PhysxSchema, device, physics_scene_prim
    )
    device_readback["runtime_device"] = str(raw_env.device)
    step_clock = PhysicsStepCounter()
    raw_accumulator = RawContactAccumulator(PhysicsSchemaTools.intToSdfPath, lambda: step_clock.current_step, ContactEventType)
    source_root_path = list(robot.root_physx_view.prim_paths)[SOURCE_ENV_INDEX]
    source_root_prim = stage.GetPrimAtPath(source_root_path)
    residual_reader = ResidualReader(
        physics_context,
        physics_scene_prim,
        source_root_prim,
        PhysxSchema,
    )
    contact_subscription = None
    clock_subscription = None
    try:
        env.reset()
        class_ids = [int(value) for value in raw_env._g009_recover_fall_class.detach().cpu().tolist()]
        require(class_ids == [0, 1, 2, 3, 0, 1, 2, 3], "stratified reset assignment mismatch")
        hold = runtime_probe.reset_pose_hold_action_diagnostics(
            robot.data.joint_pos[4:].detach(),
            robot.data.soft_joint_pos_limits[4:].detach(),
            list(robot.joint_names),
            action_scale=float(action_term.cfg.scale),
        )
        require(bool((~hold["saturated_mask"]).all().item()), "reset-pose hold action saturated")
        actions = torch.zeros((NUM_ENVS, raw_env.action_manager.total_action_dim), device=raw_env.device)
        actions[4:] = hold["normalized_action"]
        clock_subscription = get_physx_interface().subscribe_physics_on_step_events(step_clock, True, 0)
        require(clock_subscription is not None, "physics-step subscription unavailable")
        try:
            contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(raw_accumulator)
            raw_accumulator.mark_subscription(contact_subscription)
        except Exception as error:  # unsupported API is primary feasibility evidence
            raw_accumulator.mark_subscription_error(error)
        rows: list[dict[str, Any]] = []
        action_process_steps: list[int] = []
        for physics_step in range(1, PHYSICS_SUBSTEPS + 1):
            if (physics_step - 1) % 4 == 0:
                raw_env.action_manager.process_action(actions)
                action_process_steps.append(physics_step)
            raw_env._sim_step_counter += 1
            raw_env.action_manager.apply_action()
            raw_env.scene.write_data_to_sim()
            raw_env.sim.step(render=False)
            raw_env.scene.update(dt=raw_env.physics_dt)
            rows.append(_proxy_row(physics_step=physics_step, sensor=sensor, robot=robot, residual_reader=residual_reader))
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "goal_id": "g009",
            "stage_id": "R0",
            "experiment_id": "G009-5-E011",
            "revision": "rev18",
            "status": "complete",
            "headless": True,
            "device": device,
            "seed": 42,
            "replicate_index": args.replicate_index,
            "num_envs": NUM_ENVS,
            "source_env_index": SOURCE_ENV_INDEX,
            "physics_substeps": PHYSICS_SUBSTEPS,
            "physics_dt_s": PHYSICS_DT_S,
            "manual_inner_loop": {
                "control_decimation": 4,
                "action_process_steps": action_process_steps,
                "action_process_count": len(action_process_steps),
                "manager_post_step_executed": False,
                "reward_computed": False,
                "termination_computed": False,
                "trajectory_equivalence_claimed": False,
                "scope": "capability_only",
            },
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "execution": execution,
            "contract": probe_contract(device, args.replicate_index),
            "contract_sha256": canonical_sha256(probe_contract(device, args.replicate_index)),
            "predecessor": predecessor,
            "source_bundle": source_bundle,
            "governance": governance(),
            "pose_action_assignment": {"class_ids": class_ids},
            "live_physics_readback": {"solver": solver_readback, "max_depenetration_velocity": depenetration_readback},
            "device_readback": device_readback,
            "residual_capability": residual_reader.capability,
            "physics_step_clock": step_clock.snapshot(),
            "raw_contact_observation": raw_accumulator.snapshot(),
            "supporting_telemetry": rows,
        }
        report["feasibility"] = derive_feasibility(report)
        validate_report(report)
        return report
    finally:
        contact_subscription = None
        clock_subscription = None
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, choices=(DEFAULT_TASK,))
    parser.add_argument("--seed", type=int, default=42, choices=(42,))
    parser.add_argument("--replicate-index", required=True, type=int, choices=(1, 2))
    parser.add_argument("--output", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if not getattr(args, "device_explicit", False):
        parser.error("--device must be supplied explicitly as cpu or cuda:0")
    if args.device not in {"cpu", "cuda:0"}:
        parser.error("--device must be cpu or cuda:0")
    return args


def failure_envelope(args: argparse.Namespace, execution: dict[str, Any], error: BaseException) -> dict[str, Any]:
    try:
        source_bundle = source_bundle_provenance()
    except Exception as bundle_error:  # do not mask the primary failure
        source_bundle = {
            "all_files_present": False,
            "clean": False,
            "error": f"{type(bundle_error).__name__}: {bundle_error}",
        }
    device = str(getattr(args, "device", "")).lower()
    replicate_index = getattr(args, "replicate_index", None)
    contract = (
        probe_contract(device, replicate_index)
        if device in {"cpu", "cuda:0"} and replicate_index in {1, 2}
        else None
    )
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "goal_id": "g009",
        "stage_id": "R0",
        "experiment_id": "G009-5-E011",
        "revision": "rev18",
        "status": "failed_closed",
        "headless": bool(getattr(args, "headless", False)),
        "device": getattr(args, "device", None),
        "seed": getattr(args, "seed", None),
        "replicate_index": getattr(args, "replicate_index", None),
        "physics_substeps": PHYSICS_SUBSTEPS,
        "execution": execution,
        "contract": contract,
        "contract_sha256": canonical_sha256(contract) if contract is not None else None,
        "predecessor": {
            "path": PREDECESSOR_PATH.relative_to(REPO_ROOT).as_posix(),
            "expected_sha256": PREDECESSOR_SHA256,
            "observed_sha256": (
                _sha256(PREDECESSOR_PATH) if PREDECESSOR_PATH.is_file() else None
            ),
        },
        "source_bundle": source_bundle,
        "governance": governance(),
        "gpu_pair_attribution_available": False,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def main(argv: list[str] | None = None) -> int:
    output, execution = runtime_probe.prepare_execution(runtime_probe.parse_prelaunch_output(argv))
    args = parse_args(argv)
    validate_execution_metadata(execution, args.device, args.replicate_index)
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    app = None
    try:
        try:
            validate_predecessor()
            preflight = source_bundle_provenance()
            require(preflight["all_files_present"], "source bundle files are missing")
            require(preflight["clean"], "source bundle must be clean")
            validate_source_bundle(preflight)
            app = AppLauncher(args).app
            report = diagnose(args, execution)
        except Exception as error:  # always persist a fail-closed report
            report = failure_envelope(args, execution, error)
        runtime_probe._write_json_atomic(output, report)
        feasible = bool(report.get("feasibility", {}).get("run_feasible", False))
        print(json.dumps({"output": str(output), "run_feasible": feasible, "gpu_pair_attribution_available": bool(report.get("feasibility", {}).get("gpu_pair_attribution_available", False))}, ensure_ascii=False), flush=True)
        return 0 if feasible else 2
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
