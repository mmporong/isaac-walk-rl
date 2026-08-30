#!/usr/bin/env python3
"""Run the preregistered G009-5-E013 terrain-filtered contact-matrix probe.

The probe changes only the contact observation path.  It injects one exact
terrain collider filter before ContactSensor initialization, resolves the lazy
sensor buffers, clones them, then clones the direct RigidContactView matrix in
the same physics step.  It is diagnostic-only and never runs PPO or a gate.
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
from typing import Any, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPT_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import probe_g009_r0_rev18_gpu_raw_contact as base_probe
import probe_g009_r0_rev19_contact_offset_intervention as rev19
import probe_g009_recover_runtime as runtime_probe
from isaac_walk_g009 import recover_contracts


DEFAULT_TASK = base_probe.DEFAULT_TASK
SCHEMA_VERSION = "g009.r0.rev20.terrain_contact_matrix.v1"
FAILURE_SCHEMA_VERSION = "g009.r0.rev20.terrain_contact_matrix_failure.v1"
PREREGISTRATION_PATH = REPO_ROOT / "configs/g009_r0_rev20_terrain_contact_matrix.json"
PREDECESSOR_PATH = REPO_ROOT / "reports/runs/g009_r0_rev19_contact_offset_intervention_synthesis_2x2x2_s42.json"
PREDECESSOR_SHA256 = "5d95449398b4168cc8a7d0f73d4248e77fd2257b05f5bce22e4431020f8bf576"
CPU_PREFLIGHT_PATH = REPO_ROOT / "reports/runs/g009_r0_rev20_terrain_contact_matrix_cpu_preflight_2x_s42.json"
NUM_ENVS = 8
BODY_COUNT = 19
FILTER_COUNT = 1
SOURCE_ENV_INDEX = 7
PHYSICS_STEPS = 150
PHYSICS_DT_S = 0.005
FORCE_THRESHOLD_N = 1e-6
HARD_LIMIT_MARGIN_RAD = 0.01
NON_FOOT_BW_MAX = 15.0
FILTER_PATHS = ("/World/ground/terrain/GroundPlane/CollisionPlane",)
EXPECTED_PATHS = {
    ("cpu", 1): "reports/runs/g009_r0_rev20_terrain_contact_matrix_cpu_rep01_s42.json",
    ("cpu", 2): "reports/runs/g009_r0_rev20_terrain_contact_matrix_cpu_rep02_s42.json",
    ("cuda:0", 1): "reports/runs/g009_r0_rev20_terrain_contact_matrix_gpu_rep01_s42.json",
    ("cuda:0", 2): "reports/runs/g009_r0_rev20_terrain_contact_matrix_gpu_rep02_s42.json",
}
SOURCE_BINDING_PATHS = (
    "configs/g009_r0.json",
    "configs/g009_r0_rev20_terrain_contact_matrix.json",
    "scripts/probe_g009_recover_runtime.py",
    "scripts/probe_g009_r0_rev18_gpu_raw_contact.py",
    "scripts/probe_g009_r0_rev19_contact_offset_intervention.py",
    "scripts/probe_g009_r0_rev20_terrain_contact_matrix.py",
    "reports/runs/g009_r0_rev19_contact_offset_intervention_synthesis_2x2x2_s42.json",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/recover_contracts.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/registry.py",
)
SYNTHESIS_SOURCE_BINDING_PATHS = (
    "configs/g009_r0_rev20_terrain_contact_matrix.json",
    "scripts/probe_g009_r0_rev18_gpu_raw_contact.py",
    "scripts/probe_g009_r0_rev19_contact_offset_intervention.py",
    "scripts/probe_g009_r0_rev20_terrain_contact_matrix.py",
    "scripts/summarize_g009_r0_rev20_terrain_contact_matrix.py",
    "scripts/probe_g009_recover_runtime.py",
    "src/isaac_walk_g009/recover_contracts.py",
)
_RUNTIME_ENV: Any | None = None
_PARSED_CFG_VALUES: dict[str, Any] | None = None


def capture_env(env: Any, _env_ids: Any = None, **_kwargs: Any) -> None:
    """Capture the startup environment using EventManager's `(env, env_ids)` ABI."""
    global _RUNTIME_ENV
    _RUNTIME_ENV = env


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_preregistration() -> dict[str, Any]:
    value = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "rev20 preregistration root must be an object")
    require(
        value.get("schema_version") == "g009.r0.rev20.terrain_contact_matrix_preregistration.v1"
        and value.get("evidence_id") == "G009-5-E013"
        and value.get("revision") == "rev20"
        and value.get("seed") == 42,
        "rev20 preregistration identity mismatch",
    )
    terrain = value.get("terrain_filter", {})
    view = value.get("rigid_contact_view", {})
    require(
        terrain.get("filter_prim_paths_expr") == list(FILTER_PATHS)
        and terrain.get("expected_filter_paths_sha256") == canonical_sha256(list(FILTER_PATHS))
        and terrain.get("view_filter_paths_property_shape") == [152, 1]
        and terrain.get("view_filter_paths_row_expected") == list(FILTER_PATHS)
        and terrain.get("view_filter_names_property_shape") == [152, 1]
        and terrain.get("view_filter_names_row_expected") == ["CollisionPlane"]
        and terrain.get("logical_filter_paths_sha256_field_name") == "logical_filter_paths_sha256"
        and terrain.get("raw_filter_paths_sha256_field_name") == "raw_filter_paths_sha256"
        and terrain.get("raw_view_filter_paths_sha256_required") is True
        and terrain.get("filter_path_fallback_or_wildcard_allowed") is False,
        "terrain filter contract mismatch",
    )
    require(
        view.get("expected_raw_matrix_shape") == [152, 1, 3]
        and view.get("expected_reshaped_matrix_shape") == [8, 19, 1, 3]
        and view.get("articulation_root_body_paths_source") == "robot.root_physx_view.prim_paths"
        and view.get("articulation_root_body_leaf_expected") == "base"
        and isinstance(view.get("body_namespace_derivation"), str)
        and isinstance(view.get("raw_filter_metadata_validation"), str)
        and view.get("direct_matrix_and_sensor_buffer_exact_equality_required_each_step") is True
        and view.get("direct_and_buffer_storage_alias_forbidden") is True,
        "RigidContactView contract mismatch",
    )
    cpu_binding = value.get("cpu_preflight", {}).get("exact_ordered_input_report_binding_schema", {})
    require(cpu_binding.get("item_key_order") == ["path", "sha256"] and cpu_binding.get("exact_cpu_paths") == [EXPECTED_PATHS[("cpu", 1)], EXPECTED_PATHS[("cpu", 2)]] and cpu_binding.get("count") == 2 and cpu_binding.get("duplicate_path_or_sha256_allowed") is False, "CPU preflight binding schema mismatch")
    require(value.get("repeatability", {}).get("exact_fields") == ["availability_state", "sensor_paths_sha256", "raw_filter_paths_sha256", "logical_filter_paths_sha256", "force_body_names_sha256", "raw_and_reshaped_tensor_shapes", "per_env_overlap_step_indices", "source_env_overlap_step_indices", "safety_checks"], "repeatability exact-field contract mismatch")
    return value


def validate_predecessor() -> dict[str, Any]:
    require(PREDECESSOR_PATH.is_file(), "rev19 predecessor is missing")
    observed = file_sha256(PREDECESSOR_PATH)
    require(observed == PREDECESSOR_SHA256, "rev19 predecessor hash mismatch")
    value = json.loads(PREDECESSOR_PATH.read_text(encoding="utf-8"))
    require(
        value.get("decision", {}).get("outcome") == "gpu_raw_unavailable_both_arms"
        and value.get("decision", {}).get("selected_lever") is None,
        "rev19 predecessor decision mismatch",
    )
    return {"path": PREDECESSOR_PATH.relative_to(REPO_ROOT).as_posix(), "sha256": observed}


def _git_blob_sha256(relative: str, commit: str) -> str:
    completed = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT, check=True, capture_output=True)
    return sha256_bytes(completed.stdout)


def source_bundle_provenance() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_BINDING_PATHS], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    missing = [relative for relative in SOURCE_BINDING_PATHS if not (REPO_ROOT / relative).is_file()]
    files = {relative: _git_blob_sha256(relative, commit) for relative in SOURCE_BINDING_PATHS if relative not in missing and not dirty}
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return {
        "schema_version": 1, "git_commit": commit, "git_commit_valid": bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
        "source_binding_paths": list(SOURCE_BINDING_PATHS), "source_binding_files": files,
        "source_bundle_sha256": sha256_bytes(payload.encode("utf-8")) if len(files) == len(SOURCE_BINDING_PATHS) else None,
        "all_files_present": not missing, "missing_files": missing, "clean": not dirty, "dirty_source_paths": dirty,
    }


def validate_source_bundle(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "source bundle missing")
    require(value.get("git_commit_valid") is True and value.get("all_files_present") is True and value.get("clean") is True, "source bundle must be committed and clean")
    require(value.get("source_binding_paths") == list(SOURCE_BINDING_PATHS), "source binding paths mismatch")
    files = value.get("source_binding_files")
    require(isinstance(files, Mapping) and set(files) == set(SOURCE_BINDING_PATHS), "source binding file map mismatch")
    commit = cast(str, value.get("git_commit"))
    for path in SOURCE_BINDING_PATHS:
        require(files[path] == _git_blob_sha256(path, commit), f"source binding git blob mismatch: {path}")
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    require(value.get("source_bundle_sha256") == sha256_bytes(payload.encode("utf-8")), "source bundle aggregate hash mismatch")
    return dict(value)


def validate_external_sources(isaaclab_root: Path, prereg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    prereg = prereg or load_preregistration()
    root = isaaclab_root.resolve(strict=True)
    expected = prereg["baseline_physics"]["isaaclab_external_source_binding"]["files"]
    observed: dict[str, str] = {}
    for relative, digest in expected.items():
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), "external source escaped IsaacLab root")
        observed[relative] = file_sha256(path)
        require(observed[relative] == digest, f"external IsaacLab source hash mismatch: {relative}")
    return {"root": str(root), "files": observed, "all_hashes_match": True}


def expected_output_relative(device: str, replicate: int) -> str:
    require((device, replicate) in EXPECTED_PATHS, "invalid rev20 slot")
    return EXPECTED_PATHS[(device, replicate)]


def cpu_preflight_not_required_binding() -> dict[str, Any]:
    return {"status": "not_required_for_cpu", "path": None, "sha256": None, "git_commit": None, "probe_source_bundle_sha256": None, "input_reports": []}


def validate_uuid4_hex(value: Any, label: str = "execution_id") -> str:
    require(isinstance(value, str), f"{label} missing")
    parsed = uuid.UUID(hex=value)
    require(parsed.version == 4 and parsed.hex == value, f"{label} must be lowercase UUID4 hex")
    return value


def validate_execution(execution: Mapping[str, Any], device: str, replicate: int) -> None:
    require(set(execution) == {"execution_id", "started_at_utc", "output_path_repo_relative", "no_overwrite"}, "execution schema mismatch")
    validate_uuid4_hex(execution.get("execution_id"))
    require(execution.get("output_path_repo_relative") == expected_output_relative(device, replicate) and execution.get("no_overwrite") is True, "canonical no-overwrite execution mismatch")


def inject_terrain_filter(env_cfg: Any) -> Any:
    sensor_cfg = env_cfg.scene.contact_forces
    require(getattr(sensor_cfg, "_g009_rev20_filter_locked", False) is False, "filter already locked")
    sensor_cfg.filter_prim_paths_expr = list(FILTER_PATHS)
    sensor_cfg.history_length = 1
    sensor_cfg._g009_rev20_filter_locked = True
    require(list(sensor_cfg.filter_prim_paths_expr) == list(FILTER_PATHS), "filter injection failed")
    return env_cfg


def _tensor_record(tensor: Any) -> dict[str, Any]:
    detached = tensor.detach().clone().cpu()
    values = detached.tolist()
    return {"shape": list(detached.shape), "dtype": str(detached.dtype), "sha256": canonical_sha256({"shape": list(detached.shape), "values": values}), "values": values}


def _relative_body(path: str, root: str) -> str:
    prefix = root.rstrip("/") + "/"
    require(path.startswith(prefix), f"sensor path is outside robot root: {path}")
    relative = path[len(prefix):]
    require(relative and "/" not in relative, f"sensor path must identify one body leaf: {path}")
    return relative


def read_contact_tensors(sensor: Any, physics_dt_s: float) -> tuple[Any, Any, Any, bool]:
    """Resolve lazy buffers first and preserve pre-clone storage identity."""
    net_raw = sensor.data.net_forces_w
    net = net_raw.detach().clone()
    buffer_raw = sensor.data.force_matrix_w
    require(buffer_raw is not None, "terrain-filtered force_matrix_w is unavailable")
    buffer_pointer = buffer_raw.untyped_storage().data_ptr() if hasattr(buffer_raw, "untyped_storage") else buffer_raw.data_ptr()
    buffer = buffer_raw.detach().clone()
    view = sensor.contact_physx_view
    require(type(view).__name__ == "RigidContactView" or getattr(view, "_g009_test_rigid_contact_view", False), "contact view type must be RigidContactView")
    require(callable(getattr(view, "get_contact_force_matrix", None)), "contact view method missing")
    for name in ("sensor_paths", "filter_paths", "sensor_names", "filter_names", "sensor_count", "filter_count"):
        require(hasattr(view, name), f"contact view property missing: {name}")
    direct_raw = view.get_contact_force_matrix(physics_dt_s)
    direct_pointer = direct_raw.untyped_storage().data_ptr() if hasattr(direct_raw, "untyped_storage") else direct_raw.data_ptr()
    direct = direct_raw.detach().clone()
    return net, buffer, direct, buffer_pointer != direct_pointer


class MatrixSafetyAccumulator:
    def __init__(self, prereg: Mapping[str, Any] | None = None, requested_device: str = "cpu") -> None:
        self.prereg = dict(prereg or load_preregistration())
        self.requested_device = requested_device
        self.samples = 0
        self.error: str | None = None
        self.sensor_paths: list[str] | None = None
        self.filter_paths: list[list[str]] | None = None
        self.body_names: list[str] | None = None
        self.mass_body_names: list[str] | None = None
        self.direct_buffer_parity_steps: list[int] = []
        self.storage_independent_steps: list[int] = []
        self.overlap_steps: list[list[int]] = [[] for _ in range(NUM_ENVS)]
        self.all_peak = 0.0; self.source_peak = 0.0; self.all_integral = 0.0; self.source_integral = 0.0
        self.nonfoot_peak: list[float] = [0.0] * NUM_ENVS
        self.hard_limit_violation_steps: list[int] = []
        self.mass_record: dict[str, Any] | None = None
        self.mass_hash: str | None = None
        self.mass_changed_steps: list[int] = []
        self.joint_position_finite_steps: list[int] = []
        self.raw_shape: list[int] | None = None
        self.reshaped_shape: list[int] | None = None
        self.buffer_shape: list[int] | None = None
        self.net_shape: list[int] | None = None
        self.step_ledger: list[dict[str, Any]] = []
        self.robot_root_paths: list[str] | None = None
        self.body_namespace_paths: list[str] | None = None
        self.view_metadata: dict[str, Any] | None = None

    def observe(self, step: int, sensor: Any, robot: Any, physics_dt_s: float, torch_module: Any) -> None:
        try:
            require(step == self.samples + 1 and 1 <= step <= PHYSICS_STEPS, "matrix sample step mismatch")
            torch = torch_module
            net, buffer, direct, original_storage_independent = read_contact_tensors(sensor, physics_dt_s)
            tensor_devices = {"net_force": str(net.device), "sensor_buffer": str(buffer.device), "direct_matrix": str(direct.device)}
            require(set(tensor_devices.values()) == {self.requested_device}, "contact tensor device does not match requested device")
            view = sensor.contact_physx_view
            raw_shape, reshaped_shape = list(direct.shape), [NUM_ENVS, BODY_COUNT, FILTER_COUNT, 3]
            require(raw_shape == [NUM_ENVS * BODY_COUNT, FILTER_COUNT, 3], "direct matrix raw shape mismatch")
            reshaped = direct.reshape(*reshaped_shape)
            require(list(buffer.shape) == reshaped_shape and list(net.shape) == [NUM_ENVS, BODY_COUNT, 3], "sensor matrix/net shape mismatch")
            require(original_storage_independent, "direct return and lazy sensor buffer share storage before clone")
            require(torch.equal(reshaped, buffer), "direct matrix and lazy sensor buffer differ")
            sensor_paths = [str(path) for path in view.sensor_paths]
            filter_paths = [[str(path) for path in row] for row in view.filter_paths]
            body_names = [str(name) for name in sensor.body_names]
            mass_names = [str(name) for name in robot.body_names]
            roots = [str(path) for path in robot.root_physx_view.prim_paths]
            body_namespaces = [root.rsplit("/", 1)[0] for root in roots]
            view_metadata = {"sensor_count": int(view.sensor_count), "filter_count": int(view.filter_count), "sensor_names": [str(value) for value in view.sensor_names], "filter_names": [[str(value) for value in row] for row in view.filter_names]}
            sensor_names = view_metadata["sensor_names"]
            names_env_major = len(sensor_names) == NUM_ENVS * BODY_COUNT and all(sensor_names[env_index * BODY_COUNT:(env_index + 1) * BODY_COUNT] == body_names for env_index in range(NUM_ENVS))
            terrain_contract = self.prereg["terrain_filter"]; view_contract = self.prereg["rigid_contact_view"]
            filter_shape = terrain_contract["view_filter_paths_property_shape"]; name_shape = terrain_contract["view_filter_names_property_shape"]
            expected_filter_rows = [list(terrain_contract["view_filter_paths_row_expected"]) for _ in range(int(filter_shape[0]))]; expected_filter_name_rows = [list(terrain_contract["view_filter_names_row_expected"]) for _ in range(int(name_shape[0]))]
            root_leaf_valid = all(root.rsplit("/", 1)[-1] == view_contract["articulation_root_body_leaf_expected"] for root in roots)
            require(len(sensor_paths) == 152 and len(set(sensor_paths)) == 152 and filter_shape == name_shape == [152, 1] and filter_paths == expected_filter_rows and view_metadata["sensor_count"] == 152 and view_metadata["filter_count"] == 1 and names_env_major and view_metadata["filter_names"] == expected_filter_name_rows and root_leaf_valid, "view path/count contract mismatch")
            require(len(body_names) == BODY_COUNT and len(set(body_names)) == BODY_COUNT and len(roots) == NUM_ENVS, "body/root count mismatch")
            for env_index, namespace in enumerate(body_namespaces):
                chunk = sensor_paths[env_index * BODY_COUNT:(env_index + 1) * BODY_COUNT]
                require([_relative_body(path, namespace) for path in chunk] == body_names, f"env {env_index} sensor path/body order mismatch")
            require(len(set(mass_names)) == BODY_COUNT and set(mass_names) == set(body_names), "force/mass body inventory mismatch")
            if self.sensor_paths is None:
                self.sensor_paths, self.filter_paths, self.body_names, self.mass_body_names = sensor_paths, filter_paths, body_names, mass_names
                self.robot_root_paths, self.body_namespace_paths, self.view_metadata = roots, body_namespaces, view_metadata
                self.raw_shape, self.reshaped_shape, self.buffer_shape, self.net_shape = raw_shape, reshaped_shape, list(buffer.shape), list(net.shape)
            else:
                require((sensor_paths, filter_paths, body_names, mass_names, roots, body_namespaces, view_metadata) == (self.sensor_paths, self.filter_paths, self.body_names, self.mass_body_names, self.robot_root_paths, self.body_namespace_paths, self.view_metadata), "matrix path/body order drift")
            finite = bool(torch.isfinite(net).all().item() and torch.isfinite(buffer).all().item() and torch.isfinite(direct).all().item())
            require(finite, "non-finite contact matrix observation")
            matrix_body = torch.linalg.vector_norm(buffer.sum(dim=2), dim=2)
            net_body = torch.linalg.vector_norm(net, dim=2)
            overlap = (matrix_body > FORCE_THRESHOLD_N) & (net_body > FORCE_THRESHOLD_N)
            for env_index in range(NUM_ENVS):
                if bool(overlap[env_index].any().item()): self.overlap_steps[env_index].append(step)
            all_step_peak = float(matrix_body.max().item()); source_step_peak = float(matrix_body[SOURCE_ENV_INDEX].max().item())
            self.all_peak = max(self.all_peak, all_step_peak); self.source_peak = max(self.source_peak, source_step_peak)
            self.all_integral += all_step_peak * physics_dt_s; self.source_integral += source_step_peak * physics_dt_s
            joint_pos = robot.data.joint_pos.detach(); limits = robot.data.joint_pos_limits.detach()
            joint_position_finite = bool(torch.isfinite(joint_pos).all().item())
            if joint_position_finite: self.joint_position_finite_steps.append(step)
            hard_limit_pass = bool(((joint_pos >= limits[..., 0] - HARD_LIMIT_MARGIN_RAD) & (joint_pos <= limits[..., 1] + HARD_LIMIT_MARGIN_RAD)).all().item())
            if not hard_limit_pass: self.hard_limit_violation_steps.append(step)
            nonfoot = torch.tensor(["foot" not in name.lower() for name in body_names], device=net.device, dtype=torch.bool)
            per_env_nonfoot = torch.linalg.vector_norm(net[:, nonfoot, :], dim=2).amax(dim=1)
            self.nonfoot_peak = [max(old, float(new)) for old, new in zip(self.nonfoot_peak, per_env_nonfoot.detach().cpu().tolist(), strict=True)]
            mass = robot.data.default_mass.detach().clone()
            require(list(mass.shape) == [NUM_ENVS, BODY_COUNT] and bool(torch.isfinite(mass).all().item()) and bool((mass > 0).all().item()), "default mass invalid")
            record = _tensor_record(mass)
            if self.mass_record is None: self.mass_record, self.mass_hash = record, record["sha256"]
            elif record["sha256"] != self.mass_hash: self.mass_changed_steps.append(step)
            self.step_ledger.append({
                "step": step,
                "direct_matrix_sha256": rev19.tensor_snapshot(reshaped)["sha256"],
                "sensor_buffer_sha256": rev19.tensor_snapshot(buffer)["sha256"],
                "storage_independent_before_clone": original_storage_independent,
                "finite": finite,
                "matrix_positive_body_indices_by_env": [[int(index) for index in torch.nonzero(matrix_body[env] > FORCE_THRESHOLD_N, as_tuple=False).flatten().cpu().tolist()] for env in range(NUM_ENVS)],
                "net_positive_body_indices_by_env": [[int(index) for index in torch.nonzero(net_body[env] > FORCE_THRESHOLD_N, as_tuple=False).flatten().cpu().tolist()] for env in range(NUM_ENVS)],
                "matrix_body_magnitude_n_by_env": matrix_body.detach().clone().cpu().tolist(),
                "net_body_magnitude_n_by_env": net_body.detach().clone().cpu().tolist(),
                "matrix_body_magnitude_sha256": canonical_sha256({"shape": [8, 19], "values": matrix_body.detach().clone().cpu().tolist()}),
                "net_body_magnitude_sha256": canonical_sha256({"shape": [8, 19], "values": net_body.detach().clone().cpu().tolist()}),
                "joint_lower_margin_rad_by_env": (joint_pos - limits[..., 0]).detach().clone().cpu().tolist(),
                "joint_upper_margin_rad_by_env": (limits[..., 1] - joint_pos).detach().clone().cpu().tolist(),
                "all_env_matrix_peak_force_n": all_step_peak,
                "source_env_matrix_peak_force_n": source_step_peak,
                "non_foot_peak_force_n_per_env": [float(value) for value in per_env_nonfoot.detach().cpu().tolist()],
                "hard_joint_limit_with_margin": hard_limit_pass,
                "joint_position_finite": joint_position_finite,
                "mass_tensor_sha256": record["sha256"],
                "tensor_devices": tensor_devices,
            })
            self.direct_buffer_parity_steps.append(step); self.storage_independent_steps.append(step); self.samples += 1
        except Exception as error:
            if str(error) == "matrix sample step mismatch":
                raise
            if self.error is None: self.error = f"{type(error).__name__}: {error}"
            # A runtime-observed matrix contract failure is a diagnostic result, not
            # an infrastructure failure. Preserve one immutable ledger row per step
            # so the canonical report can classify it as structural-invalid.
            if self.mass_record is None:
                mass = robot.data.default_mass.detach().clone(); self.mass_record = _tensor_record(mass); self.mass_hash = self.mass_record["sha256"]
                self.mass_body_names = [str(name) for name in robot.body_names]; self.body_names = list(self.mass_body_names)
            self.step_ledger.append({"step": step, "structural_error": self.error})
            self.samples += 1

    def snapshot(self) -> dict[str, Any]:
        require(self.mass_record is not None and self.body_names is not None and self.mass_body_names is not None, f"matrix observations unavailable: {self.error}")
        mass_record = self.mass_record; body_names = self.body_names; mass_body_names = self.mass_body_names
        assert mass_record is not None and body_names is not None and mass_body_names is not None
        masses = [sum(row) for row in mass_record["values"]]
        body_weights = [mass * 9.81 for mass in masses]
        ratios = [force / weight for force, weight in zip(self.nonfoot_peak, body_weights, strict=True)]
        checks = {
            "exact_150_samples": self.samples == PHYSICS_STEPS,
            "view_filter_shape_order_finite": self.error is None and self.raw_shape == [152, 1, 3] and self.reshaped_shape == [8, 19, 1, 3],
            "direct_matrix_sensor_buffer_parity_150_of_150": self.direct_buffer_parity_steps == list(range(1, PHYSICS_STEPS + 1)),
            "direct_and_buffer_storage_independent_150_of_150": self.storage_independent_steps == list(range(1, PHYSICS_STEPS + 1)),
            "same_body_positive_force_overlap_8_of_8": all(len(steps) >= 1 for steps in self.overlap_steps),
            "source_env_7_overlap": len(self.overlap_steps[SOURCE_ENV_INDEX]) >= 1,
            "finite_joint_position_and_contact_force": self.error is None and self.joint_position_finite_steps == list(range(1, PHYSICS_STEPS + 1)),
            "hard_joint_limit_with_margin": self.error is None and not self.hard_limit_violation_steps,
            "all_env_non_foot_peak_force_within_15_bw": self.error is None and max(ratios) <= NON_FOOT_BW_MAX,
            "source_env_non_foot_peak_force_within_15_bw": self.error is None and ratios[SOURCE_ENV_INDEX] <= NON_FOOT_BW_MAX,
            "force_and_mass_body_name_inventories_match": set(body_names) == set(mass_body_names),
            "default_mass_8x19_finite_positive_unchanged": self.error is None and not self.mass_changed_steps,
            "collection_error_absent": self.error is None,
        }
        structural_keys = ("exact_150_samples", "view_filter_shape_order_finite", "direct_matrix_sensor_buffer_parity_150_of_150", "direct_and_buffer_storage_independent_150_of_150")
        safety_keys = tuple(key for key in checks if key not in structural_keys + ("same_body_positive_force_overlap_8_of_8", "source_env_7_overlap"))
        structural_valid = all(checks[key] for key in structural_keys)
        safety_valid = all(checks[key] for key in safety_keys)
        contract_valid = structural_valid
        overlap_valid = checks["same_body_positive_force_overlap_8_of_8"] and checks["source_env_7_overlap"]
        availability = "observed_valid" if structural_valid and overlap_valid else ("unavailable" if structural_valid else "invalid")
        return {
            "availability_state": availability,
            "sample_count": self.samples,
            "requested_device": self.requested_device,
            "parity_step_indices": self.direct_buffer_parity_steps,
            "storage_independent_step_indices": self.storage_independent_steps,
            "step_ledger": self.step_ledger,
            "path_order": {"articulation_root_body_paths": self.robot_root_paths or [], "body_namespace_paths": self.body_namespace_paths or [], "view_metadata": self.view_metadata or {}, "sensor_paths": self.sensor_paths or [], "sensor_paths_sha256": canonical_sha256(self.sensor_paths or []), "filter_paths": self.filter_paths or [], "raw_filter_paths_sha256": canonical_sha256(self.filter_paths or []), "logical_filter_paths_sha256": canonical_sha256(list(FILTER_PATHS)), "force_body_names": body_names, "force_body_names_sha256": canonical_sha256(body_names)},
            "shapes": {"raw": self.raw_shape, "reshaped": self.reshaped_shape, "sensor_buffer": self.buffer_shape, "net_force": self.net_shape},
            "same_step_overlap": {"per_env_overlap_step_indices": self.overlap_steps, "source_env_overlap_step_indices": self.overlap_steps[SOURCE_ENV_INDEX], "all_env_matrix_peak_force_n": self.all_peak, "source_env_matrix_peak_force_n": self.source_peak, "all_env_matrix_force_integral_n_s": self.all_integral, "source_env_matrix_force_integral_n_s": self.source_integral},
            "safety": {"mass_tensor": mass_record, "mass_body_names": mass_body_names, "mass_body_names_sha256": canonical_sha256(mass_body_names), "per_env_body_weight_n": body_weights, "non_foot_peak_force_n_per_env": self.nonfoot_peak, "non_foot_peak_force_body_weight_per_env": ratios, "hard_joint_limit_violation_steps": self.hard_limit_violation_steps, "mass_changed_steps": self.mass_changed_steps},
            "checks": checks, "structural_probe_valid": structural_valid, "safety_valid": safety_valid,
            "overlap_available": overlap_valid, "contract_valid": contract_valid, "passed": structural_valid and safety_valid, "error": self.error,
        }


def baseline_snapshot_from_values(prereg: Mapping[str, Any], values: Mapping[str, Any], live_tensor_hashes: Mapping[str, str], raw_checks: Mapping[str, bool] | None = None) -> dict[str, Any]:
    expected = prereg["baseline_physics"]["expected_snapshot_contracts"]
    snapshots: dict[str, Any] = {}
    for name in ("material", "action", "motor", "reset", "timing"):
        actual = values.get(name)
        digest = canonical_sha256(actual)
        snapshots[name] = {"value": actual, "sha256": digest, "expected_sha256": expected[name]["sha256"], "matches": actual == expected[name]["value"] and digest == expected[name]["sha256"]}
    expected_tensors = prereg["baseline_physics"]["runtime_readback_contract"]["motor"]["resolved_tensor_expected_sha256"]
    snapshots["motor_live_tensors"] = {"sha256": dict(live_tensor_hashes), "expected_sha256": dict(expected_tensors), "matches": dict(live_tensor_hashes) == dict(expected_tensors)}
    snapshots["raw_runtime_checks"] = dict(raw_checks or {})
    snapshots["all_match"] = all(item["matches"] for key, item in snapshots.items() if key not in {"all_match", "raw_runtime_checks"}) and all(snapshots["raw_runtime_checks"].values())
    return snapshots


def _scalar(value: Any, label: str) -> float:
    if hasattr(value, "detach"):
        tensor = value.detach().clone().cpu()
        require(tensor.numel() > 0 and bool(tensor.isfinite().all().item()), f"{label} tensor invalid")
        first = float(tensor.reshape(-1)[0].item())
        require(bool((tensor == tensor.reshape(-1)[0]).all().item()), f"{label} tensor is not uniform")
        return first
    result = float(value)
    require(math.isfinite(result), f"{label} is non-finite")
    return result


def _config_value(value: Any) -> Any:
    if isinstance(value, tuple): return list(value)
    return value


def capture_parse_cfg_values(env_cfg: Any) -> dict[str, Any]:
    action = env_cfg.actions.joint_pos
    motor = env_cfg.scene.robot.actuators["base_legs"]
    reset_params = env_cfg.events.reset_base.params
    terrain_material = env_cfg.scene.terrain.physics_material
    foot_params = env_cfg.events.physics_material.params
    return {
        "material_cfg": {
            "effective_static_dynamic": [float(terrain_material.static_friction) * float(foot_params["static_friction_range"][0]), float(terrain_material.dynamic_friction) * float(foot_params["dynamic_friction_range"][0])],
            "foot_static_dynamic": [float(foot_params["static_friction_range"][0]), float(foot_params["dynamic_friction_range"][0])],
            "friction_combine_mode": str(terrain_material.friction_combine_mode),
            "ground_static_dynamic": [float(terrain_material.static_friction), float(terrain_material.dynamic_friction)],
        },
        "action": {"alpha": float(action.alpha), "asset_soft_joint_limit_factor": float(env_cfg.scene.robot.soft_joint_pos_limit_factor), "rescale_to_limits": bool(action.rescale_to_limits), "scale": float(action.scale), "type": action.class_type.__name__.removesuffix("Cfg")},
        "motor_raw": {"actuator_class": type(motor).__name__, "actuator_group": "base_legs", "armature": _config_value(motor.armature), "damping": float(motor.damping), "effort_limit": float(motor.effort_limit), "effort_limit_sim": _config_value(motor.effort_limit_sim), "friction": float(motor.friction), "joint_names_expr": list(motor.joint_names_expr), "saturation_effort": float(motor.saturation_effort), "stiffness": float(motor.stiffness), "velocity_limit": float(motor.velocity_limit), "velocity_limit_sim": _config_value(motor.velocity_limit_sim)},
        "reset_cfg": {"assignment_mode": str(reset_params["assignment_mode"]), "pose_xy_range_m": list(reset_params["pose_xy_range"]), "yaw_range_rad": list(reset_params["yaw_range"])},
        "timing": {"control_decimation": int(env_cfg.decimation), "control_dt_s": float(env_cfg.decimation * env_cfg.sim.dt), "physics_dt_s": float(env_cfg.sim.dt)},
        "solver": {"position": int(env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count), "velocity": int(env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count), "max_depenetration_velocity": float(env_cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity)},
    }


def ground_material_live_readback() -> dict[str, Any]:
    import omni.usd  # pyright: ignore[reportMissingImports]
    from pxr import UsdShade  # pyright: ignore[reportMissingImports]
    stage = omni.usd.get_context().get_stage()
    collision = stage.GetPrimAtPath(FILTER_PATHS[0])
    require(collision.IsValid(), "ground collision prim missing")
    material, _relationship = UsdShade.MaterialBindingAPI(collision).ComputeBoundMaterial(materialPurpose="physics")
    require(material and material.GetPrim().IsValid(), "bound ground material missing")
    prim = material.GetPrim()
    return {
        "static": float(prim.GetAttribute("physics:staticFriction").Get()),
        "dynamic": float(prim.GetAttribute("physics:dynamicFriction").Get()),
        "combine": str(prim.GetAttribute("physxMaterial:frictionCombineMode").Get()),
        "material_path": str(prim.GetPath()),
    }


def collect_baseline_runtime(raw_env: Any, robot: Any, prereg: Mapping[str, Any], parsed_cfg: Mapping[str, Any], hold_diagnostics: Mapping[str, Any], torch_module: Any) -> dict[str, Any]:
    torch = torch_module
    expected = prereg["baseline_physics"]["expected_snapshot_contracts"]
    action_term = raw_env.action_manager.get_term("joint_pos")
    action_cfg = action_term.cfg
    actuator = robot.actuators["base_legs"]
    ground = ground_material_live_readback()
    foot = raw_env._g009_foot_material_readback.detach().clone()
    effective = raw_env._g009_effective_foot_friction.detach().clone()
    require(list(foot.shape) == [8, 4, 2] and list(effective.shape) == [8, 4, 2], "foot/effective material shape mismatch")
    values = {
        "material": dict(parsed_cfg["material_cfg"]),
        "action": dict(parsed_cfg["action"]),
        "motor": {
            "raw_config": dict(parsed_cfg["motor_raw"]),
            "resolved": {
                "actuator_class": type(actuator).__name__, "armature": _scalar(actuator.armature, "armature"), "damping": _scalar(actuator.damping, "damping"),
                "effort_limit": _scalar(actuator.effort_limit, "effort_limit"), "effort_limit_sim": _scalar(actuator.effort_limit_sim, "effort_limit_sim"), "friction": _scalar(actuator.friction, "friction"),
                "stiffness": _scalar(actuator.stiffness, "stiffness"), "velocity_limit": _scalar(actuator.velocity_limit, "velocity_limit"),
            },
        },
        "reset": {},
        "timing": dict(parsed_cfg["timing"]),
    }
    live_tensors = {
        "stiffness": actuator.stiffness, "damping": actuator.damping, "armature": actuator.armature, "friction": actuator.friction,
        "effort_limit": actuator.effort_limit, "velocity_limit": actuator.velocity_limit, "effort_limit_sim": actuator.effort_limit_sim,
        "default_joint_armature": robot.data.default_joint_armature,
    }
    live_hashes: dict[str, str] = {}
    live_records: dict[str, dict[str, Any]] = {}
    for name, tensor in live_tensors.items():
        require(list(tensor.shape) == [8, 12] and bool(torch.isfinite(tensor).all().item()), f"{name} live tensor invalid")
        live_records[name] = _tensor_record(tensor); live_hashes[name] = live_records[name]["sha256"]
    class_ids = [int(value) for value in raw_env._g009_recover_fall_class.detach().clone().cpu().tolist()]
    root_pose = robot.data.root_state_w[:, :7].detach().clone()
    root_velocity = robot.data.root_state_w[:, 7:13].detach().clone()
    joint_velocity = robot.data.joint_vel.detach().clone()
    previous = action_term._prev_applied_actions.detach().clone()
    hold_action = hold_diagnostics["normalized_action"].detach().clone()
    hold_target = hold_diagnostics["reachable_target"].detach().clone()
    full_actions = torch.zeros((8, int(raw_env.action_manager.total_action_dim)), device=hold_action.device, dtype=hold_action.dtype)
    full_actions[4:] = hold_action
    reset_expected = expected["reset"]["value"]
    reset_log = raw_env.extras.get("g009_recover_reset", {})
    require(reset_log and all(name in reset_log for name in ("source_class_ids", "root_pose_w", "root_velocity_w", "joint_pos", "joint_vel", "folded_joint_angles")), "reset event log incomplete")
    logged_pose = reset_log["root_pose_w"].detach().clone(); logged_joint = reset_log["joint_pos"].detach().clone()
    local_xyz = logged_pose[:, :3] - raw_env.scene.env_origins
    expected_pose = torch.zeros_like(root_pose)
    for env_index, class_id in enumerate(class_ids):
        definition = reset_expected["pose_definitions_in_class_id_order"][class_id]
        expected_pose[env_index, 2] = float(definition["root_height_m"])
        expected_pose[env_index, 3:7] = torch.tensor(definition["root_quaternion_wxyz"], device=root_pose.device, dtype=root_pose.dtype)
    expected_pose[:, :2] = 0.0
    actual_local_pose = torch.cat((local_xyz, logged_pose[:, 3:7]), dim=1)
    expected_joint = torch.zeros_like(robot.data.joint_pos)
    for joint_index, name in enumerate(robot.joint_names):
        if name.endswith("_hip_joint"):
            expected_joint[:, joint_index] = reset_expected["folded_joint_angles_rad"]["left_hip" if name.startswith(("FL_", "RL_")) else "right_hip"]
        elif name.endswith("_thigh_joint"):
            expected_joint[:, joint_index] = reset_expected["folded_joint_angles_rad"]["thigh"]
        elif name.endswith("_calf_joint"):
            expected_joint[:, joint_index] = reset_expected["folded_joint_angles_rad"]["calf"]
        else:
            raise ValueError(f"unmapped reset joint: {name}")
    pose_names = list(recover_contracts.RECOVER_POSES)
    values["reset"] = {
        "action_modes_by_env": ["zero_normalized"] * 4 + ["reset_pose_hold"] * 4,
        "assignment_mode": parsed_cfg["reset_cfg"]["assignment_mode"], "class_ids": class_ids,
        "folded_joint_angles_rad": dict(recover_contracts.FOLDED_JOINT_ANGLES_RAD),
        "joint_velocity_rad_s": 0.0,
        "pose_definitions_in_class_id_order": [{"name": pose.name, "root_height_m": pose.root_height_m, "root_quaternion_wxyz": list(pose.root_quaternion_wxyz)} for pose in recover_contracts.RECOVER_POSES.values()],
        "pose_xy_range_m": list(parsed_cfg["reset_cfg"]["pose_xy_range_m"]), "root_velocity_m_s_and_rad_s": [0.0] * 6,
        "source_env_index": SOURCE_ENV_INDEX, "source_env_pose": pose_names[class_ids[SOURCE_ENV_INDEX]], "yaw_range_rad": list(parsed_cfg["reset_cfg"]["yaw_range_rad"]),
    }
    raw_checks = {
        "foot_material_all_8x4_exact": bool(torch.allclose(foot, torch.tensor([1.0, 1.0], device=foot.device), atol=1e-6, rtol=0.0)),
        "effective_material_all_8x4_exact": bool(torch.allclose(effective, torch.tensor([0.8, 0.6], device=effective.device), atol=1e-6, rtol=0.0)),
        "reset_class_ids_exact": class_ids == [0, 1, 2, 3, 0, 1, 2, 3],
        "reset_root_velocity_zero": bool(torch.allclose(root_velocity, torch.zeros_like(root_velocity), atol=1e-6, rtol=0.0)),
        "reset_joint_velocity_zero": bool(torch.allclose(joint_velocity, torch.zeros_like(joint_velocity), atol=1e-6, rtol=0.0)),
        "current_root_state_matches_reset_log": bool(torch.allclose(root_pose, logged_pose, atol=1e-6, rtol=0.0) and torch.allclose(root_velocity, reset_log["root_velocity_w"], atol=1e-6, rtol=0.0)),
        "current_joint_state_matches_reset_log": bool(torch.allclose(robot.data.joint_pos, logged_joint, atol=1e-6, rtol=0.0) and torch.allclose(robot.data.joint_vel, reset_log["joint_vel"], atol=1e-6, rtol=0.0)),
        "reset_root_pose_exact_8_env": bool(torch.allclose(actual_local_pose, expected_pose, atol=1e-6, rtol=0.0)),
        "reset_folded_joint_state_exact_8_env": bool(torch.allclose(logged_joint, expected_joint, atol=1e-6, rtol=0.0)),
        "ema_history_equals_reset_joint_position": bool(torch.allclose(previous, robot.data.joint_pos, atol=1e-6, rtol=0.0)),
        "zero_action_envs_0_to_3_exact": bool((full_actions[:4] == 0).all().item()),
        "hold_action_envs_4_to_7_finite_bounded_unsaturated": bool(torch.isfinite(hold_action).all().item() and (hold_action.abs() <= 1.0).all().item() and (~hold_diagnostics["saturated_mask"]).all().item()),
        "hold_target_envs_4_to_7_equal_folded_state": bool(torch.allclose(hold_target, robot.data.joint_pos[4:], atol=1e-6, rtol=0.0)),
        "root_pose_finite_8x7": list(root_pose.shape) == [8, 7] and bool(torch.isfinite(root_pose).all().item()),
        "timing_sources_exact": float(raw_env.physics_dt) == float(raw_env.cfg.sim.dt) == 0.005 and int(raw_env.cfg.decimation) == 4 and float(raw_env.step_dt) == 0.02,
        "ground_material_attributes_finite": all(math.isfinite(float(ground[key])) for key in ("static", "dynamic")),
        "live_action_cfg_matches_parse_cfg": {"alpha": float(action_cfg.alpha), "asset_soft_joint_limit_factor": float(robot.cfg.soft_joint_pos_limit_factor), "rescale_to_limits": bool(action_cfg.rescale_to_limits), "scale": float(action_cfg.scale), "type": type(action_term).__name__} == parsed_cfg["action"],
        "parse_cfg_ground_material_matches_live": all(abs(actual - expected_value) <= 1e-6 for actual, expected_value in zip((ground["static"], ground["dynamic"]), parsed_cfg["material_cfg"]["ground_static_dynamic"], strict=True)) and ground["combine"] == parsed_cfg["material_cfg"]["friction_combine_mode"],
        "parse_cfg_solver_8_0_depenetration_1": parsed_cfg["solver"] == {"position": 8, "velocity": 0, "max_depenetration_velocity": 1.0},
        "solver_effort_limits_match": list(robot.data.joint_effort_limits.shape) == [8, 12] and bool(torch.allclose(robot.data.joint_effort_limits, actuator.effort_limit_sim, atol=0.0, rtol=0.0)),
        "solver_velocity_limits_match": list(robot.data.joint_vel_limits.shape) == [8, 12] and bool(torch.allclose(robot.data.joint_vel_limits, actuator.velocity_limit_sim, atol=0.0, rtol=0.0)),
        "default_joint_properties_match": all(list(tensor.shape) == [8, 12] and bool(torch.isfinite(tensor).all().item()) for tensor in (robot.data.default_joint_armature, robot.data.default_joint_stiffness, robot.data.default_joint_damping, robot.data.default_joint_friction_coeff)),
    }
    snapshot = baseline_snapshot_from_values(prereg, values, live_hashes, raw_checks)
    action_values = full_actions.detach().clone().cpu().tolist()
    snapshot["action_assignment"] = {"shape": list(full_actions.shape), "values": action_values, "sha256": canonical_sha256({"shape": list(full_actions.shape), "values": action_values}), "zero_envs": [0, 1, 2, 3], "hold_envs": [4, 5, 6, 7]}
    snapshot["reset_runtime_evidence"] = {"logged_root_pose": rev19.tensor_snapshot(logged_pose), "current_root_pose": rev19.tensor_snapshot(root_pose), "logged_joint_pos": rev19.tensor_snapshot(logged_joint), "current_joint_pos": rev19.tensor_snapshot(robot.data.joint_pos), "ema_previous_targets": rev19.tensor_snapshot(previous), "hold_normalized_action": rev19.tensor_snapshot(hold_action), "hold_reachable_target": rev19.tensor_snapshot(hold_target)}
    snapshot["runtime_observations"] = {
        "material": {"foot": _tensor_record(foot), "effective": _tensor_record(effective), "ground": ground},
        "action": {"live_cfg": {"alpha": float(action_cfg.alpha), "asset_soft_joint_limit_factor": float(robot.cfg.soft_joint_pos_limit_factor), "rescale_to_limits": bool(action_cfg.rescale_to_limits), "scale": float(action_cfg.scale), "type": type(action_term).__name__}, "parsed_cfg": dict(parsed_cfg["action"])},
        "motor": {"live_tensors": live_records, "velocity_limit_sim": _tensor_record(actuator.velocity_limit_sim), "joint_effort_limits": _tensor_record(robot.data.joint_effort_limits), "joint_velocity_limits": _tensor_record(robot.data.joint_vel_limits), "default_joint_stiffness": _tensor_record(robot.data.default_joint_stiffness), "default_joint_damping": _tensor_record(robot.data.default_joint_damping), "default_joint_friction": _tensor_record(robot.data.default_joint_friction_coeff)},
        "reset": {"class_ids": class_ids, "env_origins": _tensor_record(raw_env.scene.env_origins), "root_velocity": _tensor_record(root_velocity), "joint_velocity": _tensor_record(joint_velocity), "logged_root_velocity": _tensor_record(reset_log["root_velocity_w"]), "logged_joint_velocity": _tensor_record(reset_log["joint_vel"]), "joint_names": list(robot.joint_names), "saturated_mask": _tensor_record(hold_diagnostics["saturated_mask"])},
        "timing": {"physics_dt": float(raw_env.physics_dt), "cfg_sim_dt": float(raw_env.cfg.sim.dt), "decimation": int(raw_env.cfg.decimation), "step_dt": float(raw_env.step_dt), "parsed": dict(parsed_cfg["timing"]), "solver_parsed": dict(parsed_cfg["solver"])},
    }
    contact_offsets = rev19.tensor_snapshot(robot.root_physx_view.get_contact_offsets().detach().clone())
    rest_offsets = rev19.tensor_snapshot(robot.root_physx_view.get_rest_offsets().detach().clone())
    mass = rev19.tensor_snapshot(robot.data.default_mass.detach().clone())
    force_names = [str(name) for name in raw_env.scene.sensors["contact_forces"].body_names]
    mass_names = [str(name) for name in robot.body_names]
    invariant_checks = {
        "contact_offset_tensor_hash": contact_offsets["sha256"] == prereg["baseline_physics"]["expected_contact_offset_tensor_sha256"],
        "rest_offset_tensor_hash": rest_offsets["sha256"] == prereg["baseline_physics"]["expected_rest_offset_tensor_sha256"],
        "mass_tensor_hash": mass["sha256"] == prereg["baseline_physics"]["expected_mass_tensor_sha256"],
        "mass_body_order_hash": canonical_sha256(mass_names) == prereg["baseline_physics"]["expected_mass_body_names_sha256"],
        "force_body_order_hash": canonical_sha256(force_names) == prereg["baseline_physics"]["expected_force_body_names_sha256"],
    }
    snapshot["invariants"] = {"contact_offsets": contact_offsets, "rest_offsets": rest_offsets, "mass": mass, "mass_body_names": mass_names, "force_body_names": force_names, "checks": invariant_checks}
    snapshot["all_match"] = snapshot["all_match"] and all(invariant_checks.values())
    return snapshot


def governance() -> dict[str, Any]:
    return {"diagnostic_only": True, "learned": False, "reward_computed": False, "ppo_updates": 0, "gate_execution_allowed": False, "qualification_eligible": False, "qualification_status": "not_run", "physics_ground_truth_authority": False}


def probe_contract(device: str, replicate: int) -> dict[str, Any]:
    prereg = load_preregistration()
    return {
        "schema_version": 1, "experiment_id": "G009-5-E013", "revision": "rev20", "slot": f"{device}.rep{replicate}",
        "baseline_source_cell": prereg["baseline_physics"]["source_cell"], "single_changed_axis": "contact observation path only",
        "terrain_filter_paths": list(FILTER_PATHS), "solver_iterations": [8, 0], "contact_offset_scale": 1.0,
        "runtime": {"num_envs": 8, "physics_steps": 150, "physics_dt_s": 0.005, "headless": True, "render": False},
        "governance": governance(), "canonical_output": expected_output_relative(device, replicate),
    }


def derive_feasibility(report: Mapping[str, Any]) -> dict[str, Any]:
    matrix = report.get("terrain_contact_matrix", {})
    baseline = report.get("baseline_snapshot", {})
    device = report.get("device_readback", {})
    external = report.get("external_source_binding", {})
    valid = bool(matrix.get("structural_probe_valid") is True and matrix.get("safety_valid") is True and baseline.get("all_match") is True and device.get("gpu_dynamics_matches_device") is True and external.get("all_hashes_match") is True and live_readback_valid(report))
    return {"probe_valid": valid, "availability_state": matrix.get("availability_state", "invalid"), "run_interpretable": valid, "matrix_authority_candidate": valid and matrix.get("availability_state") == "observed_valid"}


def recompute_matrix_payload(matrix: Mapping[str, Any], requested_device: str | None = None) -> dict[str, Any]:
    path_order = cast(Mapping[str, Any], matrix.get("path_order", {})); shapes = cast(Mapping[str, Any], matrix.get("shapes", {})); ledger = matrix.get("step_ledger")
    sensor_paths = path_order.get("sensor_paths"); roots = path_order.get("articulation_root_body_paths"); namespaces = path_order.get("body_namespace_paths"); body_names = path_order.get("force_body_names"); filter_paths = path_order.get("filter_paths")
    view_metadata = path_order.get("view_metadata", {})
    require(isinstance(sensor_paths, list) and isinstance(roots, list) and isinstance(namespaces, list) and isinstance(body_names, list) and isinstance(filter_paths, list), "matrix path payload missing")
    assert isinstance(sensor_paths, list) and isinstance(roots, list) and isinstance(namespaces, list) and isinstance(body_names, list) and isinstance(filter_paths, list)
    mapping_valid = len(sensor_paths) == 152 and len(set(sensor_paths)) == 152 and len(roots) == len(namespaces) == 8 and namespaces == [str(root).rsplit("/", 1)[0] for root in roots] and len(body_names) == 19 and len(set(body_names)) == 19
    if mapping_valid:
        for env_index, namespace in enumerate(namespaces):
            chunk = sensor_paths[env_index * BODY_COUNT:(env_index + 1) * BODY_COUNT]
            mapping_valid = mapping_valid and [_relative_body(path, namespace) for path in chunk] == body_names
    path_hashes_valid = path_order.get("sensor_paths_sha256") == canonical_sha256(sensor_paths) and path_order.get("raw_filter_paths_sha256") == canonical_sha256(filter_paths) and path_order.get("logical_filter_paths_sha256") == canonical_sha256(list(FILTER_PATHS)) and path_order.get("force_body_names_sha256") == canonical_sha256(body_names) and "filter_paths_sha256" not in path_order
    metadata_sensor_names = view_metadata.get("sensor_names") if isinstance(view_metadata, Mapping) else None
    prereg = load_preregistration(); terrain_contract = prereg["terrain_filter"]; view_contract = prereg["rigid_contact_view"]
    expected_filter_rows = [list(terrain_contract["view_filter_paths_row_expected"]) for _ in range(int(terrain_contract["view_filter_paths_property_shape"][0]))]; expected_filter_name_rows = [list(terrain_contract["view_filter_names_row_expected"]) for _ in range(int(terrain_contract["view_filter_names_property_shape"][0]))]
    mapping_valid = mapping_valid and all(str(root).rsplit("/", 1)[-1] == view_contract["articulation_root_body_leaf_expected"] for root in roots)
    metadata_valid = isinstance(metadata_sensor_names, list) and len(metadata_sensor_names) == NUM_ENVS * BODY_COUNT and all(metadata_sensor_names[env_index * BODY_COUNT:(env_index + 1) * BODY_COUNT] == body_names for env_index in range(NUM_ENVS)) and view_metadata.get("sensor_count") == 152 and view_metadata.get("filter_count") == 1 and view_metadata.get("filter_names") == expected_filter_name_rows
    shape_valid = shapes == {"raw": [152, 1, 3], "reshaped": [8, 19, 1, 3], "sensor_buffer": [8, 19, 1, 3], "net_force": [8, 19, 3]}
    require(isinstance(ledger, list), "matrix step ledger missing")
    assert isinstance(ledger, list)
    ledger_rows = cast(list[Mapping[str, Any]], ledger)
    exact_steps = [item.get("step") for item in ledger_rows] == list(range(1, PHYSICS_STEPS + 1))
    parity = exact_steps and all(item.get("direct_matrix_sha256") == item.get("sensor_buffer_sha256") for item in ledger_rows)
    storage = exact_steps and all(item.get("storage_independent_before_clone") is True for item in ledger_rows)
    requested_device = requested_device or str(matrix.get("requested_device"))
    device_valid = matrix.get("requested_device") == requested_device and exact_steps and all(item.get("tensor_devices") == {"net_force": requested_device, "sensor_buffer": requested_device, "direct_matrix": requested_device} for item in ledger_rows)
    contact_finite = exact_steps and all(item.get("finite") is True and math.isfinite(float(cast(float, item.get("all_env_matrix_peak_force_n")))) and math.isfinite(float(cast(float, item.get("source_env_matrix_peak_force_n")))) for item in ledger_rows)
    joint_finite = exact_steps and all(item.get("joint_position_finite") is True for item in ledger_rows)
    overlap_steps: list[list[int]] = [[] for _ in range(NUM_ENVS)]
    all_peaks: list[float] = []; source_peaks: list[float] = []; nonfoot_steps: list[list[float]] = []
    for item in ledger_rows:
        if item.get("structural_error"):
            matrix_magnitudes = [[0.0] * BODY_COUNT for _ in range(NUM_ENVS)]; net_magnitudes = [[0.0] * BODY_COUNT for _ in range(NUM_ENVS)]
            matrix_indices = [[] for _ in range(NUM_ENVS)]; net_indices = [[] for _ in range(NUM_ENVS)]
            all_peaks.append(0.0); source_peaks.append(0.0); nonfoot_steps.append([0.0] * NUM_ENVS)
            continue
        matrix_magnitudes = item.get("matrix_body_magnitude_n_by_env"); net_magnitudes = item.get("net_body_magnitude_n_by_env")
        require(isinstance(matrix_magnitudes, list) and isinstance(net_magnitudes, list) and len(matrix_magnitudes) == len(net_magnitudes) == NUM_ENVS and all(isinstance(row, list) and len(row) == BODY_COUNT for row in matrix_magnitudes + net_magnitudes), "ledger body magnitude payload mismatch")
        assert isinstance(matrix_magnitudes, list) and isinstance(net_magnitudes, list)
        require(all(math.isfinite(float(value)) and float(value) >= 0.0 for row in matrix_magnitudes + net_magnitudes for value in row), "ledger body magnitude non-finite")
        require(item.get("matrix_body_magnitude_sha256") == canonical_sha256({"shape": [8, 19], "values": matrix_magnitudes}) and item.get("net_body_magnitude_sha256") == canonical_sha256({"shape": [8, 19], "values": net_magnitudes}), "ledger body magnitude hash mismatch")
        matrix_indices = [[index for index, value in enumerate(row) if float(value) > FORCE_THRESHOLD_N] for row in matrix_magnitudes]
        net_indices = [[index for index, value in enumerate(row) if float(value) > FORCE_THRESHOLD_N] for row in net_magnitudes]
        require(item.get("matrix_positive_body_indices_by_env") == matrix_indices and item.get("net_positive_body_indices_by_env") == net_indices, "ledger positive indices differ from raw magnitudes")
        require(isinstance(matrix_indices, list) and isinstance(net_indices, list) and len(matrix_indices) == len(net_indices) == NUM_ENVS, "ledger positive-index payload mismatch")
        for env_index in range(NUM_ENVS):
            require(all(type(index) is int and 0 <= index < BODY_COUNT for index in matrix_indices[env_index] + net_indices[env_index]), "ledger body index invalid")
            if set(matrix_indices[env_index]) & set(net_indices[env_index]): overlap_steps[env_index].append(int(item["step"]))
        raw_all_peak = max(float(value) for row in matrix_magnitudes for value in row); raw_source_peak = max(float(value) for value in matrix_magnitudes[SOURCE_ENV_INDEX])
        require(float(item["all_env_matrix_peak_force_n"]) == raw_all_peak and float(item["source_env_matrix_peak_force_n"]) == raw_source_peak, "ledger peak differs from raw magnitudes")
        lower = item.get("joint_lower_margin_rad_by_env"); upper = item.get("joint_upper_margin_rad_by_env")
        require(isinstance(lower, list) and isinstance(upper, list) and len(lower) == len(upper) == NUM_ENVS and all(isinstance(row, list) and len(row) == 12 for row in lower + upper), "ledger joint margin payload mismatch")
        assert isinstance(lower, list) and isinstance(upper, list)
        raw_hard_safe = all(math.isfinite(float(value)) and float(value) >= -HARD_LIMIT_MARGIN_RAD for row in lower + upper for value in row)
        require(item.get("hard_joint_limit_with_margin") is raw_hard_safe, "ledger hard-limit status differs from raw margins")
        all_peaks.append(raw_all_peak); source_peaks.append(raw_source_peak)
        nonfoot_indices = [index for index, name in enumerate(body_names) if "foot" not in str(name).lower()]
        recomputed_nonfoot = [max(float(net_magnitudes[env_index][body_index]) for body_index in nonfoot_indices) for env_index in range(NUM_ENVS)]
        nonfoot = item.get("non_foot_peak_force_n_per_env"); require(isinstance(nonfoot, list) and len(nonfoot) == NUM_ENVS and all(math.isfinite(float(value)) and float(value) >= 0 for value in nonfoot), "ledger non-foot payload invalid")
        assert isinstance(nonfoot, list)
        require([float(value) for value in nonfoot] == recomputed_nonfoot, "ledger non-foot peak differs from net body magnitudes")
        nonfoot_steps.append([float(value) for value in nonfoot])
    safety = cast(Mapping[str, Any], matrix.get("safety", {})); mass = cast(Mapping[str, Any], safety.get("mass_tensor", {})); mass_values = mass.get("values")
    require(mass.get("shape") == [8, 19] and isinstance(mass_values, list) and len(mass_values) == 8 and all(len(row) == 19 for row in mass_values), "mass payload shape mismatch")
    assert isinstance(mass_values, list)
    mass_hash = canonical_sha256({"shape": [8, 19], "values": mass_values}); require(mass.get("sha256") == mass_hash, "mass payload hash mismatch")
    body_weights = [sum(float(value) for value in row) * 9.81 for row in mass_values]
    nonfoot_peak = [max(step[env] for step in nonfoot_steps) for env in range(NUM_ENVS)]
    ratios = [force / weight for force, weight in zip(nonfoot_peak, body_weights, strict=True)]
    error_absent = matrix.get("error") is None
    mass_unchanged = exact_steps and error_absent and all(item.get("mass_tensor_sha256") == mass_hash for item in ledger_rows)
    hard_safe = exact_steps and error_absent and all(item.get("hard_joint_limit_with_margin") is True for item in ledger_rows)
    recomputed_checks = {
        "exact_150_samples": exact_steps and matrix.get("sample_count") == PHYSICS_STEPS,
        "view_filter_shape_order_finite": mapping_valid and path_hashes_valid and metadata_valid and filter_paths == expected_filter_rows and shape_valid and device_valid and contact_finite,
        "direct_matrix_sensor_buffer_parity_150_of_150": parity and matrix.get("parity_step_indices") == list(range(1, PHYSICS_STEPS + 1)),
        "direct_and_buffer_storage_independent_150_of_150": storage and matrix.get("storage_independent_step_indices") == list(range(1, PHYSICS_STEPS + 1)),
        "same_body_positive_force_overlap_8_of_8": all(overlap_steps), "source_env_7_overlap": bool(overlap_steps[SOURCE_ENV_INDEX]),
        "finite_joint_position_and_contact_force": contact_finite and joint_finite, "hard_joint_limit_with_margin": hard_safe,
        "all_env_non_foot_peak_force_within_15_bw": error_absent and max(ratios) <= NON_FOOT_BW_MAX, "source_env_non_foot_peak_force_within_15_bw": error_absent and ratios[SOURCE_ENV_INDEX] <= NON_FOOT_BW_MAX,
        "force_and_mass_body_name_inventories_match": set(body_names) == set(safety.get("mass_body_names", [])),
        "default_mass_8x19_finite_positive_unchanged": mass_unchanged and all(value > 0 and math.isfinite(value) for row in mass_values for value in row),
        "collection_error_absent": error_absent,
    }
    structural_keys = ("exact_150_samples", "view_filter_shape_order_finite", "direct_matrix_sensor_buffer_parity_150_of_150", "direct_and_buffer_storage_independent_150_of_150")
    safety_keys = tuple(key for key in recomputed_checks if key not in structural_keys + ("same_body_positive_force_overlap_8_of_8", "source_env_7_overlap"))
    structural_valid = all(recomputed_checks[key] for key in structural_keys); safety_valid = all(recomputed_checks[key] for key in safety_keys)
    contract_valid = structural_valid; overlap_valid = recomputed_checks["same_body_positive_force_overlap_8_of_8"] and recomputed_checks["source_env_7_overlap"]
    availability = "observed_valid" if structural_valid and overlap_valid else ("unavailable" if structural_valid else "invalid")
    summary = {"per_env_overlap_step_indices": overlap_steps, "source_env_overlap_step_indices": overlap_steps[SOURCE_ENV_INDEX], "all_env_matrix_peak_force_n": max(all_peaks), "source_env_matrix_peak_force_n": max(source_peaks), "all_env_matrix_force_integral_n_s": sum(value * PHYSICS_DT_S for value in all_peaks), "source_env_matrix_force_integral_n_s": sum(value * PHYSICS_DT_S for value in source_peaks)}
    require(matrix.get("checks") == recomputed_checks and matrix.get("structural_probe_valid") is structural_valid and matrix.get("safety_valid") is safety_valid and matrix.get("overlap_available") is overlap_valid and matrix.get("contract_valid") is contract_valid and matrix.get("passed") is (structural_valid and safety_valid) and matrix.get("availability_state") == availability, "serialized matrix checks/status differ from ledger recomputation")
    serialized_summary = matrix.get("same_step_overlap", {})
    require(all(serialized_summary.get(key) == summary[key] for key in ("per_env_overlap_step_indices", "source_env_overlap_step_indices", "all_env_matrix_peak_force_n", "source_env_matrix_peak_force_n")) and all(abs(float(serialized_summary.get(key)) - summary[key]) <= 1e-12 for key in ("all_env_matrix_force_integral_n_s", "source_env_matrix_force_integral_n_s")), "serialized matrix summary differs from ledger recomputation")
    require(safety.get("per_env_body_weight_n") == body_weights and safety.get("non_foot_peak_force_n_per_env") == nonfoot_peak and safety.get("non_foot_peak_force_body_weight_per_env") == ratios, "serialized safety summary differs from ledger recomputation")
    return {"checks": recomputed_checks, "structural_probe_valid": structural_valid, "safety_valid": safety_valid, "overlap_available": overlap_valid, "contract_valid": contract_valid, "availability_state": availability}


def validate_baseline_payload(value: Any, prereg: Mapping[str, Any]) -> bool:
    require(isinstance(value, Mapping), "baseline payload missing")
    expected = prereg["baseline_physics"]["expected_snapshot_contracts"]
    matches: list[bool] = []
    for name in ("material", "action", "motor", "reset", "timing"):
        item = value.get(name); require(isinstance(item, Mapping), f"baseline {name} missing")
        digest = canonical_sha256(item.get("value")); expected_digest = expected[name]["sha256"]
        recomputed = item.get("value") == expected[name]["value"] and digest == expected_digest
        require(item.get("sha256") == digest and item.get("expected_sha256") == expected_digest and item.get("matches") is recomputed, f"baseline {name} hash/status mismatch")
        matches.append(recomputed)
    motor_live = value.get("motor_live_tensors", {}); expected_live = prereg["baseline_physics"]["runtime_readback_contract"]["motor"]["resolved_tensor_expected_sha256"]
    require(motor_live.get("expected_sha256") == expected_live and motor_live.get("sha256") == expected_live and motor_live.get("matches") is True, "baseline motor tensor hash mismatch")
    required_raw_checks = {
        "foot_material_all_8x4_exact", "effective_material_all_8x4_exact", "reset_class_ids_exact", "reset_root_velocity_zero", "reset_joint_velocity_zero",
        "current_root_state_matches_reset_log", "current_joint_state_matches_reset_log", "reset_root_pose_exact_8_env", "reset_folded_joint_state_exact_8_env",
        "ema_history_equals_reset_joint_position", "zero_action_envs_0_to_3_exact", "hold_action_envs_4_to_7_finite_bounded_unsaturated",
        "hold_target_envs_4_to_7_equal_folded_state", "root_pose_finite_8x7", "timing_sources_exact", "ground_material_attributes_finite",
        "live_action_cfg_matches_parse_cfg", "parse_cfg_ground_material_matches_live", "parse_cfg_solver_8_0_depenetration_1",
        "solver_effort_limits_match", "solver_velocity_limits_match", "default_joint_properties_match",
    }
    raw_checks = value.get("raw_runtime_checks"); require(isinstance(raw_checks, Mapping) and set(raw_checks) == required_raw_checks and all(type(check) is bool for check in raw_checks.values()), "baseline raw runtime checks malformed")
    invariants = value.get("invariants", {}); required_invariant_checks = {"contact_offset_tensor_hash", "rest_offset_tensor_hash", "mass_tensor_hash", "mass_body_order_hash", "force_body_order_hash"}
    require(isinstance(invariants, Mapping) and isinstance(invariants.get("checks"), Mapping) and set(invariants["checks"]) == required_invariant_checks and all(type(check) is bool for check in invariants["checks"].values()), "baseline invariant checks malformed")
    def validate_tensor_evidence(record: Any, expected_shape: list[int] | None = None) -> str:
        require(isinstance(record, Mapping) and isinstance(record.get("values"), list) and isinstance(record.get("shape"), list), "runtime tensor evidence malformed")
        if expected_shape is not None: require(record["shape"] == expected_shape, "runtime tensor evidence shape mismatch")
        digest = canonical_sha256({"shape": record["shape"], "values": record["values"]})
        require(record.get("sha256") == digest, "runtime tensor evidence hash mismatch")
        return digest
    contact_hash = validate_tensor_evidence(invariants.get("contact_offsets")); rest_hash = validate_tensor_evidence(invariants.get("rest_offsets")); mass_hash = validate_tensor_evidence(invariants.get("mass"), [8, 19])
    recomputed_invariants = {
        "contact_offset_tensor_hash": contact_hash == prereg["baseline_physics"]["expected_contact_offset_tensor_sha256"],
        "rest_offset_tensor_hash": rest_hash == prereg["baseline_physics"]["expected_rest_offset_tensor_sha256"],
        "mass_tensor_hash": mass_hash == prereg["baseline_physics"]["expected_mass_tensor_sha256"],
        "mass_body_order_hash": canonical_sha256(invariants.get("mass_body_names")) == prereg["baseline_physics"]["expected_mass_body_names_sha256"],
        "force_body_order_hash": canonical_sha256(invariants.get("force_body_names")) == prereg["baseline_physics"]["expected_force_body_names_sha256"],
    }
    require(invariants["checks"] == recomputed_invariants, "baseline invariant statuses differ from raw evidence")
    evidence = value.get("reset_runtime_evidence"); required_evidence = {"logged_root_pose", "current_root_pose", "logged_joint_pos", "current_joint_pos", "ema_previous_targets", "hold_normalized_action", "hold_reachable_target"}
    require(isinstance(evidence, Mapping) and set(evidence) == required_evidence, "reset runtime evidence schema mismatch")
    for name, shape in {"logged_root_pose": [8, 7], "current_root_pose": [8, 7], "logged_joint_pos": [8, 12], "current_joint_pos": [8, 12], "ema_previous_targets": [8, 12], "hold_normalized_action": [4, 12], "hold_reachable_target": [4, 12]}.items(): validate_tensor_evidence(evidence[name], shape)
    action_assignment = value.get("action_assignment", {}); action_values = action_assignment.get("values")
    require(action_assignment.get("shape") == [8, 12] and isinstance(action_values, list) and len(action_values) == 8 and all(isinstance(row, list) and len(row) == 12 and all(math.isfinite(float(item)) for item in row) for row in action_values) and action_assignment.get("zero_envs") == [0, 1, 2, 3] and action_assignment.get("hold_envs") == [4, 5, 6, 7] and action_assignment.get("sha256") == canonical_sha256({"shape": [8, 12], "values": action_values}), "action assignment evidence malformed")
    require(all(float(item) == 0.0 for row in action_values[:4] for item in row) and all(abs(float(item)) <= 1.0 for row in action_values[4:] for item in row), "action assignment modes invalid")
    observations = value.get("runtime_observations"); require(isinstance(observations, Mapping) and set(observations) == {"material", "action", "motor", "reset", "timing"}, "runtime observations schema mismatch")
    assert isinstance(observations, Mapping)
    def record_values(record: Any, shape: list[int]) -> list[Any]:
        validate_tensor_evidence(record, shape); assert isinstance(record, Mapping) and isinstance(record.get("values"), list)
        return cast(list[Any], record["values"])
    def close_nested(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
        if isinstance(left, list) and isinstance(right, list): return len(left) == len(right) and all(close_nested(a, b, tolerance) for a, b in zip(left, right, strict=True))
        if isinstance(left, (int, float)) and isinstance(right, (int, float)): return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance
        return left == right
    material_obs = cast(Mapping[str, Any], observations["material"]); foot_values = record_values(material_obs.get("foot"), [8, 4, 2]); effective_values = record_values(material_obs.get("effective"), [8, 4, 2]); ground = cast(Mapping[str, Any], material_obs.get("ground", {}))
    action_obs = cast(Mapping[str, Any], observations["action"]); motor_obs = cast(Mapping[str, Any], observations["motor"]); reset_obs = cast(Mapping[str, Any], observations["reset"]); timing_obs = cast(Mapping[str, Any], observations["timing"])
    require(set(material_obs) == {"foot", "effective", "ground"} and set(ground) == {"static", "dynamic", "combine", "material_path"}, "material runtime observation schema mismatch")
    require(set(action_obs) == {"live_cfg", "parsed_cfg"}, "action runtime observation schema mismatch")
    require(set(motor_obs) == {"live_tensors", "velocity_limit_sim", "joint_effort_limits", "joint_velocity_limits", "default_joint_stiffness", "default_joint_damping", "default_joint_friction"}, "motor runtime observation schema mismatch")
    require(set(reset_obs) == {"class_ids", "env_origins", "root_velocity", "joint_velocity", "logged_root_velocity", "logged_joint_velocity", "joint_names", "saturated_mask"}, "reset runtime observation schema mismatch")
    require(set(timing_obs) == {"physics_dt", "cfg_sim_dt", "decimation", "step_dt", "parsed", "solver_parsed"}, "timing runtime observation schema mismatch")
    motor_records = cast(Mapping[str, Any], motor_obs.get("live_tensors", {})); require(set(motor_records) == set(expected_live), "motor live tensor evidence keys mismatch")
    recomputed_live_hashes = {name: validate_tensor_evidence(motor_records[name], [8, 12]) for name in expected_live}
    require(recomputed_live_hashes == expected_live, "motor live tensor raw evidence mismatch")
    effort_values = record_values(motor_records["effort_limit_sim"], [8, 12]); joint_effort_values = record_values(motor_obs.get("joint_effort_limits"), [8, 12])
    velocity_sim_values = record_values(motor_obs.get("velocity_limit_sim"), [8, 12]); joint_velocity_limit_values = record_values(motor_obs.get("joint_velocity_limits"), [8, 12])
    default_property_values = [record_values(motor_obs.get(name), [8, 12]) for name in ("default_joint_stiffness", "default_joint_damping", "default_joint_friction")]
    reset_runtime = cast(Mapping[str, Any], evidence); logged_pose = cast(list[Any], reset_runtime["logged_root_pose"]["values"]); current_pose = cast(list[Any], reset_runtime["current_root_pose"]["values"]); logged_joint = cast(list[Any], reset_runtime["logged_joint_pos"]["values"]); current_joint = cast(list[Any], reset_runtime["current_joint_pos"]["values"]); previous_targets = cast(list[Any], reset_runtime["ema_previous_targets"]["values"]); hold_values = cast(list[Any], reset_runtime["hold_normalized_action"]["values"]); hold_target_values = cast(list[Any], reset_runtime["hold_reachable_target"]["values"])
    class_ids = reset_obs.get("class_ids"); env_origins = record_values(reset_obs.get("env_origins"), [8, 3]); root_velocity_values = record_values(reset_obs.get("root_velocity"), [8, 6]); joint_velocity_values = record_values(reset_obs.get("joint_velocity"), [8, 12]); logged_root_velocity = record_values(reset_obs.get("logged_root_velocity"), [8, 6]); logged_joint_velocity = record_values(reset_obs.get("logged_joint_velocity"), [8, 12]); saturated_values = record_values(reset_obs.get("saturated_mask"), [4, 12]); joint_names = reset_obs.get("joint_names")
    require(isinstance(class_ids, list) and len(class_ids) == NUM_ENVS and all(type(item) is int and 0 <= item < 4 for item in class_ids) and isinstance(joint_names, list) and len(joint_names) == 12, "reset observation identifiers malformed")
    assert isinstance(class_ids, list) and isinstance(joint_names, list)
    reset_contract = expected["reset"]["value"]; definitions = reset_contract["pose_definitions_in_class_id_order"]; folded = reset_contract["folded_joint_angles_rad"]
    pose_exact = True
    for env_index, class_id in enumerate(class_ids):
        definition = definitions[int(class_id)]; local = [float(logged_pose[env_index][axis]) - float(env_origins[env_index][axis]) for axis in range(3)]
        pose_exact = pose_exact and abs(local[0]) <= 1e-6 and abs(local[1]) <= 1e-6 and abs(local[2] - float(definition["root_height_m"])) <= 1e-6 and close_nested(logged_pose[env_index][3:7], definition["root_quaternion_wxyz"])
    expected_joint_values: list[list[float]] = []
    for _env_index in range(NUM_ENVS):
        row: list[float] = []
        for name in joint_names:
            if str(name).endswith("_hip_joint"): row.append(float(folded["left_hip" if str(name).startswith(("FL_", "RL_")) else "right_hip"]))
            elif str(name).endswith("_thigh_joint"): row.append(float(folded["thigh"]))
            elif str(name).endswith("_calf_joint"): row.append(float(folded["calf"]))
            else: row.append(float("nan"))
        expected_joint_values.append(row)
    live_action = action_obs.get("live_cfg"); parsed_action = action_obs.get("parsed_cfg")
    parsed_timing = timing_obs.get("parsed"); solver_parsed = timing_obs.get("solver_parsed")
    recomputed_raw_checks = {
        "foot_material_all_8x4_exact": close_nested(foot_values, [[[1.0, 1.0]] * 4 for _ in range(8)]),
        "effective_material_all_8x4_exact": close_nested(effective_values, [[[0.8, 0.6]] * 4 for _ in range(8)]),
        "reset_class_ids_exact": class_ids == reset_contract["class_ids"],
        "reset_root_velocity_zero": close_nested(root_velocity_values, [[0.0] * 6 for _ in range(8)]), "reset_joint_velocity_zero": close_nested(joint_velocity_values, [[0.0] * 12 for _ in range(8)]),
        "current_root_state_matches_reset_log": close_nested(current_pose, logged_pose) and close_nested(root_velocity_values, logged_root_velocity), "current_joint_state_matches_reset_log": close_nested(current_joint, logged_joint) and close_nested(joint_velocity_values, logged_joint_velocity),
        "reset_root_pose_exact_8_env": pose_exact, "reset_folded_joint_state_exact_8_env": close_nested(logged_joint, expected_joint_values), "ema_history_equals_reset_joint_position": close_nested(previous_targets, current_joint),
        "zero_action_envs_0_to_3_exact": close_nested(action_values[:4], [[0.0] * 12 for _ in range(4)]), "hold_action_envs_4_to_7_finite_bounded_unsaturated": close_nested(action_values[4:], hold_values) and all(math.isfinite(float(item)) and abs(float(item)) <= 1.0 for row in hold_values for item in row) and not any(bool(item) for row in saturated_values for item in row),
        "hold_target_envs_4_to_7_equal_folded_state": close_nested(hold_target_values, current_joint[4:]), "root_pose_finite_8x7": all(math.isfinite(float(item)) for row in current_pose for item in row),
        "timing_sources_exact": timing_obs.get("physics_dt") == timing_obs.get("cfg_sim_dt") == 0.005 and timing_obs.get("decimation") == 4 and timing_obs.get("step_dt") == 0.02 and parsed_timing == expected["timing"]["value"],
        "ground_material_attributes_finite": all(isinstance(ground.get(key), (int, float)) and math.isfinite(float(ground[key])) for key in ("static", "dynamic")), "live_action_cfg_matches_parse_cfg": live_action == parsed_action == expected["action"]["value"],
        "parse_cfg_ground_material_matches_live": close_nested([ground.get("static"), ground.get("dynamic")], expected["material"]["value"]["ground_static_dynamic"]) and ground.get("combine") == expected["material"]["value"]["friction_combine_mode"],
        "parse_cfg_solver_8_0_depenetration_1": solver_parsed == {"position": 8, "velocity": 0, "max_depenetration_velocity": 1.0}, "solver_effort_limits_match": close_nested(joint_effort_values, effort_values), "solver_velocity_limits_match": close_nested(joint_velocity_limit_values, velocity_sim_values),
        "default_joint_properties_match": all(all(math.isfinite(float(item)) for row in values for item in row) for values in default_property_values),
    }
    require(raw_checks == recomputed_raw_checks, "baseline raw runtime checks differ from raw observations")
    require(value.get("all_match") is (all(matches) and all(raw_checks.values()) and all(invariants["checks"].values())), "baseline aggregate status mismatch")
    return bool(value.get("all_match"))


def live_readback_valid(report: Mapping[str, Any]) -> bool:
    live = report.get("live_physics_readback", {}); solver = live.get("solver", {})
    solver_checks = runtime_probe.articulation_solver_iteration_checks(solver, expected_position_count=8, expected_velocity_count=0, expected_articulations=NUM_ENVS)
    try:
        mass_names = report["baseline_snapshot"]["invariants"]["mass_body_names"]
    except (KeyError, TypeError):
        return False
    depen_checks = runtime_probe.rigid_body_max_depenetration_velocity_checks(live.get("max_depenetration_velocity", {}), expected_velocity_m_s=1.0, expected_articulation_count=NUM_ENVS, expected_body_names=mass_names)
    clock = report.get("physics_step_clock", {}); observed_dt = clock.get("observed_dt_s")
    clock_pass = clock.get("callback_count") == PHYSICS_STEPS and clock.get("expected_callback_count") == PHYSICS_STEPS and isinstance(observed_dt, list) and len(observed_dt) == PHYSICS_STEPS and all(isinstance(value, (int, float)) and math.isfinite(float(value)) and abs(float(value) - PHYSICS_DT_S) <= 1e-9 for value in observed_dt)
    return bool(solver_checks.get("articulation_solver_iteration_counts_match_contract") and depen_checks.get("rigid_body_max_depenetration_velocity_matches_contract") and clock.get("passed") is clock_pass and clock_pass)


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    require(report.get("schema_version") == SCHEMA_VERSION and report.get("experiment_id") == "G009-5-E013" and report.get("status") == "complete", "rev20 report identity mismatch")
    device, replicate = str(report.get("device")), report.get("replicate_index")
    require(device in {"cpu", "cuda:0"} and replicate in {1, 2}, "rev20 report slot mismatch")
    validate_execution(cast(Mapping[str, Any], report.get("execution")), device, cast(int, replicate))
    contract = probe_contract(device, cast(int, replicate)); require(report.get("contract") == contract and report.get("contract_sha256") == canonical_sha256(contract), "rev20 contract/hash mismatch")
    validate_source_bundle(report.get("source_bundle"))
    require(report.get("predecessor") == validate_predecessor(), "predecessor binding mismatch")
    require(report.get("governance") == governance(), "governance mismatch")
    matrix = report.get("terrain_contact_matrix", {}); require(isinstance(matrix, Mapping), "terrain matrix payload missing")
    recomputed_matrix = recompute_matrix_payload(cast(Mapping[str, Any], matrix), device)
    prereg = load_preregistration(); validate_baseline_payload(report.get("baseline_snapshot"), prereg)
    external = report.get("external_source_binding", {}); expected_external = prereg["baseline_physics"]["isaaclab_external_source_binding"]["files"]
    require(isinstance(external, Mapping) and external.get("files") == expected_external and external.get("all_hashes_match") is True and isinstance(external.get("root"), str), "external source binding mismatch")
    readback = report.get("device_readback", {})
    recomputed_device_match = readback.get("requested_device") == device and readback.get("runtime_device") == device and readback.get("gpu_dynamics_enabled") is (device == "cuda:0") and readback.get("error") is None
    require(readback.get("gpu_dynamics_matches_device") is recomputed_device_match, "device/GPU dynamics serialized status mismatch")
    live_readback_valid(report)
    derived = derive_feasibility(report)
    require(report.get("feasibility") == derived, "feasibility recomputation mismatch")
    return derived


def recompute_cpu_preflight_decision(reports: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    require(len(reports) == 2, "CPU preflight recomputation requires two reports")
    matrices = [cast(Mapping[str, Any], report["terrain_contact_matrix"]) for report in reports]
    exact_fields = [
        (matrix["availability_state"], matrix["path_order"]["sensor_paths_sha256"], matrix["path_order"]["raw_filter_paths_sha256"], matrix["path_order"]["logical_filter_paths_sha256"], matrix["path_order"]["force_body_names_sha256"], matrix["shapes"]["raw"], matrix["shapes"]["reshaped"], matrix["same_step_overlap"]["per_env_overlap_step_indices"], matrix["same_step_overlap"]["source_env_overlap_step_indices"], matrix["checks"])
        for matrix in matrices
    ]
    numeric_fields = [[float(matrix["same_step_overlap"][key]) for key in ("all_env_matrix_peak_force_n", "source_env_matrix_peak_force_n", "all_env_matrix_force_integral_n_s", "source_env_matrix_force_integral_n_s")] for matrix in matrices]
    abs_tol, rel_tol = 1e-5, 1e-6
    numeric_match = all(abs(left - right) <= max(abs_tol, rel_tol * max(abs(left), abs(right))) for left, right in zip(numeric_fields[0], numeric_fields[1], strict=True))
    repeatability = {"exact_fields_match": exact_fields[0] == exact_fields[1], "numeric_fields_within_tolerance": numeric_match, "repeatable": exact_fields[0] == exact_fields[1] and numeric_match, "absolute_tolerance": abs_tol, "relative_tolerance": rel_tol}
    baseline_device_source_valid = all(report["baseline_snapshot"]["all_match"] is True and report["device_readback"]["gpu_dynamics_matches_device"] is True and report["external_source_binding"]["all_hashes_match"] is True and live_readback_valid(report) for report in reports)
    if not baseline_device_source_valid: outcome = "probe_invalid"
    elif not all(matrix["structural_probe_valid"] is True for matrix in matrices): outcome = "terrain_matrix_probe_invalid"
    elif not all(matrix["safety_valid"] is True for matrix in matrices): outcome = "safety_limit_exceeded"
    elif [matrix["availability_state"] for matrix in matrices] == ["unavailable", "unavailable"]: outcome = "cpu_terrain_matrix_unavailable_gpu_forbidden"
    elif [matrix["availability_state"] for matrix in matrices] != ["observed_valid", "observed_valid"] or not repeatability["repeatable"]: outcome = "inconclusive_nondeterministic_gpu_forbidden"
    else: outcome = "gpu_stage_authorized"
    return outcome, repeatability


def validate_cpu_preflight_value(value: Mapping[str, Any], repo_root: Path, expected_output_relative_path: str, source_bundle: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    prereg = load_preregistration(); binding_contract = prereg["cpu_preflight"]["exact_ordered_input_report_binding_schema"]
    expected_top_level_keys = {
        "schema_version", "evidence_id", "status", "mode", "input_report_count",
        "input_reports", "integrity", "cpu_preflight", "decision", "governance",
        "synthesis_source_bundle", "execution",
    }
    require(set(value) == expected_top_level_keys, "CPU preflight top-level schema mismatch")
    require(value.get("schema_version") == "g009.r0.rev20.terrain_contact_matrix_cpu_preflight.v1" and value.get("evidence_id") == "G009-5-E013" and value.get("status") == "complete" and value.get("mode") == "cpu_preflight_2x" and value.get("input_report_count") == 2, "CPU preflight identity mismatch")
    require(value.get("governance") == governance(), "CPU preflight governance mismatch")
    inputs = value.get("input_reports"); expected_key_order = list(binding_contract["item_key_order"]); expected_paths = list(binding_contract["exact_cpu_paths"])
    require(isinstance(inputs, list) and len(inputs) == binding_contract["count"] and all(isinstance(item, Mapping) and list(item.keys()) == expected_key_order for item in inputs), "CPU preflight input binding key order/schema mismatch")
    assert isinstance(inputs, list)
    require([item["path"] for item in inputs] == expected_paths and all(re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])) is not None for item in inputs) and len({item["path"] for item in inputs}) == 2 and len({item["sha256"] for item in inputs}) == 2, "CPU preflight exact inputs mismatch")
    reports: list[dict[str, Any]] = []
    for item in inputs:
        report_path = (repo_root / item["path"]).resolve(strict=True); require(report_path.is_relative_to(repo_root.resolve()), "CPU input escaped repository")
        report_raw = report_path.read_bytes(); require(sha256_bytes(report_raw) == item["sha256"], "CPU input report hash mismatch")
        report = json.loads(report_raw.decode("utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {token}")))
        validate_report(report); reports.append(report)
    require([f"{report['device']}.rep{report['replicate_index']}" for report in reports] == ["cpu.rep1", "cpu.rep2"], "CPU input report slot mismatch")
    report_source_digests = {report["source_bundle"]["source_bundle_sha256"] for report in reports}; report_commits = {report["source_bundle"]["git_commit"] for report in reports}
    require(len(report_source_digests) == len(report_commits) == 1, "CPU report source binding drift")
    if source_bundle is not None: require(report_source_digests == {source_bundle.get("source_bundle_sha256")} and report_commits == {source_bundle.get("git_commit")}, "CPU reports/current source bundle mismatch")
    outcome, repeatability = recompute_cpu_preflight_decision(reports); require(outcome == "gpu_stage_authorized", f"CPU reports do not authorize GPU: {outcome}")
    decision = value.get("decision"); require(isinstance(decision, Mapping) and set(decision) == {"outcome", "third_run_allowed", "repeatability"} and decision.get("outcome") == outcome and decision.get("third_run_allowed") is False and decision.get("repeatability") == repeatability, "CPU preflight decision differs from report recomputation")
    required_checks = all(report["terrain_contact_matrix"]["structural_probe_valid"] is True and report["terrain_contact_matrix"]["safety_valid"] is True and report["baseline_snapshot"]["all_match"] is True and report["device_readback"]["gpu_dynamics_matches_device"] is True and report["external_source_binding"]["all_hashes_match"] is True and live_readback_valid(report) for report in reports)
    cpu_status = value.get("cpu_preflight"); expected_cpu_status = {"passed": True, "required_checks_passed": required_checks, "within_cpu_repeatability_passed": repeatability["repeatable"], "gpu_stage_allowed": True}
    require(cpu_status == expected_cpu_status, "CPU preflight status fields differ from report recomputation")
    synthesis = value.get("synthesis_source_bundle"); require(isinstance(synthesis, Mapping) and set(synthesis) == {"schema_version", "git_commit", "git_commit_valid", "source_binding_paths", "source_binding_files", "source_bundle_sha256", "clean"}, "CPU synthesis source bundle schema mismatch")
    assert isinstance(synthesis, Mapping)
    synthesis_paths = synthesis.get("source_binding_paths"); synthesis_files = synthesis.get("source_binding_files")
    require(synthesis.get("schema_version") == 1 and synthesis.get("git_commit_valid") is True and isinstance(synthesis.get("git_commit"), str) and re.fullmatch(r"[0-9a-f]{40}", synthesis["git_commit"]) is not None and synthesis.get("clean") is True and synthesis_paths == list(SYNTHESIS_SOURCE_BINDING_PATHS) and isinstance(synthesis_files, Mapping) and set(synthesis_files) == set(SYNTHESIS_SOURCE_BINDING_PATHS) and all(re.fullmatch(r"[0-9a-f]{64}", str(digest)) is not None for digest in synthesis_files.values()), "CPU synthesis source bundle fields mismatch")
    assert isinstance(synthesis_files, Mapping)
    synthesis_payload = "\n".join(f"{path}:{synthesis_files[path]}" for path in sorted(synthesis_files)); require(synthesis.get("source_bundle_sha256") == sha256_bytes(synthesis_payload.encode()), "CPU synthesis source aggregate mismatch")
    if (repo_root / ".git").exists():
        for relative in SYNTHESIS_SOURCE_BINDING_PATHS: require(synthesis_files[relative] == sha256_bytes(subprocess.run(["git", "show", f"{synthesis['git_commit']}:{relative}"], cwd=repo_root, check=True, capture_output=True).stdout), f"CPU synthesis git blob mismatch: {relative}")
    require(report_commits == {synthesis["git_commit"]}, "CPU report/synthesis commit mismatch")
    integrity = value.get("integrity"); expected_integrity_keys = {"passed", "hash_bound", "unique_report_paths", "unique_report_sha256", "unique_execution_ids", "exact_slots", "git_commit", "probe_source_bundle_sha256", "synthesis_source_bundle_sha256"}
    require(isinstance(integrity, Mapping) and set(integrity) == expected_integrity_keys and integrity.get("passed") is True and integrity.get("hash_bound") is True and integrity.get("unique_report_paths") is True and integrity.get("unique_report_sha256") is True and integrity.get("unique_execution_ids") is True and integrity.get("exact_slots") == ["cpu.rep1", "cpu.rep2"] and integrity.get("git_commit") == synthesis["git_commit"] and integrity.get("probe_source_bundle_sha256") == next(iter(report_source_digests)) and integrity.get("synthesis_source_bundle_sha256") == synthesis["source_bundle_sha256"], "CPU preflight integrity mismatch")
    execution = value.get("execution"); require(isinstance(execution, Mapping) and set(execution) == {"execution_id", "started_at_utc", "output_path_repo_relative", "no_overwrite"} and execution.get("output_path_repo_relative") == expected_output_relative_path and execution.get("no_overwrite") is True and isinstance(execution.get("started_at_utc"), str), "CPU preflight execution identity mismatch")
    assert isinstance(execution, Mapping)
    preflight_execution_id = validate_uuid4_hex(execution.get("execution_id"), "CPU preflight execution_id"); input_ids = [validate_uuid4_hex(report["execution"]["execution_id"]) for report in reports]
    require(len(set(input_ids + [preflight_execution_id])) == 3, "CPU preflight/input execution_id collision")
    return reports


def validate_cpu_preflight_artifact(path: Path, source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    require(resolved == CPU_PREFLIGHT_PATH.resolve(), "GPU must bind canonical CPU preflight")
    raw = resolved.read_bytes(); value = json.loads(raw.decode("utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {token}")))
    validate_cpu_preflight_value(value, REPO_ROOT, resolved.relative_to(REPO_ROOT).as_posix(), source_bundle)
    inputs = value["input_reports"]
    return {"status": "validated_for_gpu", "path": resolved.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_bytes(raw), "git_commit": source_bundle["git_commit"], "probe_source_bundle_sha256": source_bundle["source_bundle_sha256"], "input_reports": inputs}


def prelaunch_validate(args: argparse.Namespace) -> dict[str, Any]:
    """Complete every immutable/source/preflight check before AppLauncher."""
    prereg = load_preregistration(); validate_predecessor(); validate_external_sources(args.isaaclab_root, prereg)
    source = validate_source_bundle(source_bundle_provenance())
    binding = validate_cpu_preflight_artifact(args.cpu_preflight, source) if args.device == "cuda:0" else cpu_preflight_not_required_binding()
    return {"preregistration": prereg, "source_bundle": source, "cpu_preflight_binding": binding}


def diagnose(args: argparse.Namespace, execution: dict[str, Any]) -> dict[str, Any]:
    global _RUNTIME_ENV, _PARSED_CFG_VALUES
    prereg = load_preregistration(); validate_predecessor()
    source = validate_source_bundle(source_bundle_provenance())
    external = validate_external_sources(args.isaaclab_root, prereg)
    import isaaclab_tasks.utils as task_utils  # pyright: ignore[reportMissingImports]
    import torch
    from isaaclab.managers import EventTermCfg  # pyright: ignore[reportMissingImports]

    accumulator = MatrixSafetyAccumulator(prereg, args.device)
    baseline: dict[str, Any] = {}
    original_parse, original_proxy = task_utils.parse_env_cfg, base_probe._proxy_row
    original_hold = runtime_probe.reset_pose_hold_action_diagnostics
    saved = {"POSITION_SOLVER_ITERATIONS": base_probe.POSITION_SOLVER_ITERATIONS, "VELOCITY_SOLVER_ITERATIONS": base_probe.VELOCITY_SOLVER_ITERATIONS, "PREDECESSOR_PATH": base_probe.PREDECESSOR_PATH, "PREDECESSOR_SHA256": base_probe.PREDECESSOR_SHA256, "SOURCE_BINDING_PATHS": base_probe.SOURCE_BINDING_PATHS, "expected_output_relative": base_probe.expected_output_relative}

    def wrapped_parse(*parse_args: Any, **parse_kwargs: Any) -> Any:
        global _PARSED_CFG_VALUES
        cfg = inject_terrain_filter(original_parse(*parse_args, **parse_kwargs))
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 0
        cfg.events.reset_base.params.update({"assignment_mode": "stratified", "pose_xy_range": (0.0, 0.0), "yaw_range": (0.0, 0.0)})
        _PARSED_CFG_VALUES = capture_parse_cfg_values(cfg)
        cfg.events.rev20_capture_env = EventTermCfg(func=capture_env, mode="startup")
        return cfg

    def wrapped_hold(*hold_args: Any, **hold_kwargs: Any) -> Any:
        result = original_hold(*hold_args, **hold_kwargs)
        require(_RUNTIME_ENV is not None and _PARSED_CFG_VALUES is not None, "pre-step baseline sources unavailable")
        runtime_env = _RUNTIME_ENV; parsed_cfg_values = _PARSED_CFG_VALUES
        assert runtime_env is not None and parsed_cfg_values is not None
        baseline.update(collect_baseline_runtime(runtime_env, runtime_env.scene["robot"], prereg, parsed_cfg_values, result, torch))
        return result

    def wrapped_proxy(*proxy_args: Any, **proxy_kwargs: Any) -> dict[str, Any]:
        row = original_proxy(*proxy_args, **proxy_kwargs)
        accumulator.observe(int(proxy_kwargs["physics_step"]), proxy_kwargs["sensor"], proxy_kwargs["robot"], PHYSICS_DT_S, torch)
        return row

    try:
        task_utils.parse_env_cfg = wrapped_parse; base_probe._proxy_row = wrapped_proxy; runtime_probe.reset_pose_hold_action_diagnostics = wrapped_hold
        base_probe.POSITION_SOLVER_ITERATIONS = 8; base_probe.VELOCITY_SOLVER_ITERATIONS = 0
        base_probe.PREDECESSOR_PATH = PREDECESSOR_PATH; base_probe.PREDECESSOR_SHA256 = PREDECESSOR_SHA256; base_probe.SOURCE_BINDING_PATHS = SOURCE_BINDING_PATHS
        base_probe.expected_output_relative = expected_output_relative
        base_report = base_probe.diagnose(argparse.Namespace(**vars(args)), execution)
    finally:
        task_utils.parse_env_cfg = original_parse; base_probe._proxy_row = original_proxy; runtime_probe.reset_pose_hold_action_diagnostics = original_hold; _RUNTIME_ENV = None; _PARSED_CFG_VALUES = None
        for name, value in saved.items(): setattr(base_probe, name, value)
    report = dict(base_report)
    report.update({"schema_version": SCHEMA_VERSION, "experiment_id": "G009-5-E013", "revision": "rev20"})
    report["contract"] = probe_contract(args.device, args.replicate_index); report["contract_sha256"] = canonical_sha256(report["contract"])
    report["predecessor"] = validate_predecessor(); report["source_bundle"] = source; report["external_source_binding"] = external
    report["cpu_preflight_binding"] = getattr(args, "_cpu_preflight_binding", cpu_preflight_not_required_binding())
    report["terrain_filter"] = {"filter_prim_paths_expr": list(FILTER_PATHS), "filter_paths_sha256": canonical_sha256(list(FILTER_PATHS)), "injected_before_view_initialization": True, "fallback_used": False}
    report["terrain_contact_matrix"] = accumulator.snapshot(); report["baseline_snapshot"] = baseline; report["governance"] = governance()
    report["feasibility"] = derive_feasibility(report); validate_report(report)
    return report


def build_core_help_parser() -> argparse.ArgumentParser:
    """Build import-free help for probe-owned options only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, choices=(DEFAULT_TASK,)); parser.add_argument("--seed", type=int, default=42, choices=(42,))
    parser.add_argument("--replicate-index", required=True, type=int, choices=(1, 2)); parser.add_argument("--cpu-preflight", type=Path); parser.add_argument("--isaaclab-root", type=Path, default=REPO_ROOT.parent / "IsaacLab"); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0")); parser.add_argument("--headless", action="store_true", required=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, choices=(DEFAULT_TASK,)); parser.add_argument("--seed", type=int, default=42, choices=(42,))
    parser.add_argument("--replicate-index", required=True, type=int, choices=(1, 2)); parser.add_argument("--cpu-preflight", type=Path); parser.add_argument("--isaaclab-root", type=Path, default=REPO_ROOT.parent / "IsaacLab"); parser.add_argument("--output", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser); args = parser.parse_args(argv)
    if not getattr(args, "device_explicit", False) or args.device not in {"cpu", "cuda:0"}: parser.error("--device must be supplied explicitly as cpu or cuda:0")
    if args.device == "cuda:0" and args.cpu_preflight is None: parser.error("GPU runs require --cpu-preflight")
    if args.device == "cpu" and args.cpu_preflight is not None: parser.error("CPU runs must not supply --cpu-preflight")
    if getattr(args, "headless", None) is not True: parser.error("--headless is required")
    return args


def failure_envelope(args: argparse.Namespace, execution: Mapping[str, Any], error: BaseException) -> dict[str, Any]:
    return {"schema_version": FAILURE_SCHEMA_VERSION, "experiment_id": "G009-5-E013", "revision": "rev20", "status": "failed_closed", "device": getattr(args, "device", None), "replicate_index": getattr(args, "replicate_index", None), "execution": dict(execution), "governance": governance(), "error": {"type": type(error).__name__, "message": str(error)}}


def failed_attempt_path(args: argparse.Namespace, execution: Mapping[str, Any]) -> Path:
    device = str(args.device).replace(":", "_"); replicate = int(args.replicate_index); execution_id = validate_uuid4_hex(execution.get("execution_id"))
    return Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic/failed_attempts/rev20" / f"g009_r0_rev20_terrain_contact_matrix_{device}_rep{replicate:02d}_{execution_id}.json"


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if any(token in {"-h", "--help"} for token in effective_argv):
        build_core_help_parser().print_help()
        return 0
    output, execution = runtime_probe.prepare_execution(runtime_probe.parse_prelaunch_output(argv)); args = parse_args(argv)
    validate_execution(execution, args.device, args.replicate_index)
    try:
        prelaunch = prelaunch_validate(args)
        args._cpu_preflight_binding = prelaunch["cpu_preflight_binding"]
    except Exception as error:
        print(json.dumps({"status": "prelaunch_rejected_without_consuming_canonical_output", "error": {"type": type(error).__name__, "message": str(error)}}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 2
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]
    app = None
    try:
        try:
            app = AppLauncher(args).app; report = diagnose(args, execution); validate_report(report)
        except Exception as error:
            failure_path = failed_attempt_path(args, execution); runtime_probe._write_json_atomic(failure_path, failure_envelope(args, execution, error))
            print(json.dumps({"status": "runtime_failed_without_consuming_canonical_output", "failure_report": str(failure_path)}, ensure_ascii=False), file=sys.stderr, flush=True)
            return 2
        runtime_probe._write_json_atomic(output, report)
        valid = bool(report["feasibility"]["run_interpretable"])
        print(json.dumps({"output": str(output), "run_interpretable": valid, "availability_state": report["feasibility"]["availability_state"]}, ensure_ascii=False), flush=True)
        return 0 if valid else 2
    finally:
        if app is not None: app.close()


if __name__ == "__main__":
    raise SystemExit(main())
