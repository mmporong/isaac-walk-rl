#!/usr/bin/env python3
"""Run the G009-5-E016 read-only matrix observation adapter runtime probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPT_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))
sys.modules.setdefault(Path(__file__).stem, sys.modules[__name__])

import probe_g009_recover_runtime as runtime_probe
import probe_g009_r0_rev20_terrain_contact_matrix as rev20
import summarize_g009_r0_rev22_read_only_matrix_observation_adapter as rev22_verifier


DEFAULT_TASK = rev20.DEFAULT_TASK
SCHEMA_VERSION = "g009.r0.rev23.read_only_matrix_observation_adapter_runtime.v1"
FAILURE_SCHEMA_VERSION = "g009.r0.rev23.read_only_matrix_observation_adapter_runtime_failure.v1"
PREREGISTRATION_PATH = REPO_ROOT / "configs/g009_r0_rev23_read_only_matrix_observation_adapter_runtime.json"
PREDECESSOR_PATH = REPO_ROOT / "reports/runs/g009_r0_rev22_read_only_matrix_observation_adapter_preregistration_s42.json"
PREDECESSOR_SHA256 = "a8e536c7f5b739b983c8d1ce05c701b725b46b8926acc33b457c56ff9fad2343"
RUNTIME_PARENT_SYNTHESIS_PATH = REPO_ROOT / "reports/runs/g009_r0_rev20_terrain_contact_matrix_synthesis_2x2_s42.json"
RUNTIME_PARENT_SYNTHESIS_SHA256 = "dcb8f446a212390f94f9ae5ccad97d9e770f9b8f5961f5ffb0c920f8d62580b3"
CPU_PREFLIGHT_PATH = REPO_ROOT / "reports/runs/g009_r0_rev23_matrix_observation_adapter_cpu_preflight_2x_s42.json"
NUM_ENVS = 8
BODY_COUNT = 19
FILTER_COUNT = 1
COMPONENT_COUNT = 3
PHYSICS_STEPS = 150
PHYSICS_DT_S = 0.005
CONTACT_THRESHOLD_N = 1.0e-6
REPRESENTATIVE_STEPS = (1, 50, 100, 150)
EXPECTED_PATHS = {
    ("cpu", 1): "reports/runs/g009_r0_rev23_matrix_observation_adapter_cpu_rep01_s42.json",
    ("cpu", 2): "reports/runs/g009_r0_rev23_matrix_observation_adapter_cpu_rep02_s42.json",
    ("cuda:0", 1): "reports/runs/g009_r0_rev23_matrix_observation_adapter_gpu_rep01_s42.json",
    ("cuda:0", 2): "reports/runs/g009_r0_rev23_matrix_observation_adapter_gpu_rep02_s42.json",
}
SOURCE_BINDING_PATHS = (
    "configs/g009_r0_rev23_read_only_matrix_observation_adapter_runtime.json",
    "src/isaac_walk_g009/matrix_observation_adapter.py",
    "scripts/probe_g009_r0_rev23_matrix_observation_adapter_runtime.py",
    "scripts/summarize_g009_r0_rev23_matrix_observation_adapter_runtime.py",
    "scripts/probe_g009_r0_rev20_terrain_contact_matrix.py",
    "configs/g009_r0_rev20_terrain_contact_matrix.json",
    "scripts/summarize_g009_r0_rev22_read_only_matrix_observation_adapter.py",
    "reports/runs/g009_r0_rev22_read_only_matrix_observation_adapter_preregistration_s42.json",
)
SYNTHESIS_SOURCE_BINDING_PATHS = SOURCE_BINDING_PATHS


class OperationalVerificationError(RuntimeError):
    """Signal a local verification or I/O failure, not an experiment rejection."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON: {error}") from error
    require(isinstance(value, dict), f"{label} root must be an object")
    return cast(dict[str, Any], value)


def _read_strict_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    return strict_json_bytes(raw, label), raw


def expected_output_relative(device: str, replicate: int) -> str:
    require((device, replicate) in EXPECTED_PATHS, "invalid rev23 runtime slot")
    return EXPECTED_PATHS[(device, replicate)]


def governance() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "learned": False,
        "reward_computed": False,
        "ppo_updates": 0,
        "gate_execution_allowed": False,
        "qualification_eligible": False,
        "qualification_status": "not_run",
        "physics_ground_truth_authority": False,
    }


def claim_limits() -> dict[str, Any]:
    return {
        "adapter_runtime_observed_after_pass": True,
        "policy_observation_connected": False,
        "reward_computed": False,
        "ppo_training_executed": False,
        "normal_contact_force_vector_claim_allowed": True,
        "total_contact_force_claim_allowed": False,
        "tangential_friction_force_claim_allowed": False,
        "friction_effect_directly_observed_claim_allowed": False,
        "walking_turning_slope_or_self_recovery_qualified": False,
        "physics_ground_truth_authority": False,
    }


def load_preregistration() -> dict[str, Any]:
    value, _ = _read_strict_json(PREREGISTRATION_PATH, "rev23 preregistration")
    require(
        value.get("schema_version")
        == "g009.r0.rev23.read_only_matrix_observation_adapter_runtime_preregistration.v1"
        and value.get("evidence_id") == "G009-5-E016"
        and value.get("revision") == "rev23"
        and value.get("seed") == 42,
        "rev23 preregistration identity mismatch",
    )
    predecessor = cast(Mapping[str, Any], value.get("predecessor", {}))
    require(
        predecessor.get("path") == PREDECESSOR_PATH.relative_to(REPO_ROOT).as_posix()
        and predecessor.get("sha256") == PREDECESSOR_SHA256
        and predecessor.get("required_outcome") == "read_only_matrix_observation_adapter_preregistration_passed"
        and predecessor.get("required_next_step") == "implement_and_run_read_only_matrix_observation_adapter_runtime_probe"
        and predecessor.get("required_adapter_contract_sha256")
        == "05105dbb7cf8646d0c7a5bf667cc9ab78de76131819a9654e43d9465a31d5b43"
        and predecessor.get("full_read_only_verification_required") is True,
        "rev23 predecessor contract mismatch",
    )
    runtime = cast(Mapping[str, Any], value.get("runtime", {}))
    require(
        runtime.get("task") == DEFAULT_TASK
        and runtime.get("num_envs") == NUM_ENVS
        and runtime.get("physics_steps") == PHYSICS_STEPS
        and runtime.get("physics_dt_s") == PHYSICS_DT_S
        and runtime.get("control_dt_s") == 0.02
        and runtime.get("headless") is True
        and runtime.get("render") is False
        and runtime.get("devices") == ["cpu", "cuda:0"]
        and runtime.get("replicates_per_device") == 2
        and runtime.get("representative_snapshot_steps") == list(REPRESENTATIVE_STEPS),
        "rev23 runtime contract mismatch",
    )
    adapter = cast(Mapping[str, Any], value.get("adapter_implementation", {}))
    require(
        adapter.get("module") == "isaac_walk_g009.matrix_observation_adapter"
        and adapter.get("callable") == "adapt_terrain_pair_force_matrix_w"
        and adapter.get("source_shape") == [NUM_ENVS, BODY_COUNT, FILTER_COUNT, COMPONENT_COUNT]
        and adapter.get("source_dtype") == "torch.float32"
        and adapter.get("world_xyz_formula") == "source.sum(dim=2)"
        and adapter.get("world_xyz_shape") == [NUM_ENVS, BODY_COUNT, COMPONENT_COUNT]
        and adapter.get("magnitude_shape") == [NUM_ENVS, BODY_COUNT]
        and adapter.get("contact_mask_shape") == [NUM_ENVS, BODY_COUNT]
        and adapter.get("contact_threshold_n") == CONTACT_THRESHOLD_N
        and adapter.get("normalization") == "none"
        and adapter.get("clipping") == "none"
        and adapter.get("invalid_source") == "fail_closed_without_zero_fill"
        and adapter.get("source_output_alias_allowed") is False
        and adapter.get("output_output_alias_allowed") is False,
        "rev23 adapter implementation contract mismatch",
    )
    outputs = cast(Mapping[str, Any], value.get("outputs", {}))
    require(
        outputs.get("cpu_rep1") == EXPECTED_PATHS[("cpu", 1)]
        and outputs.get("cpu_rep2") == EXPECTED_PATHS[("cpu", 2)]
        and outputs.get("gpu_rep1") == EXPECTED_PATHS[("cuda:0", 1)]
        and outputs.get("gpu_rep2") == EXPECTED_PATHS[("cuda:0", 2)]
        and outputs.get("cpu_preflight") == CPU_PREFLIGHT_PATH.relative_to(REPO_ROOT).as_posix()
        and outputs.get("immutable_no_overwrite") is True
        and outputs.get("pass_only_canonical_write") is True,
        "rev23 output contract mismatch",
    )
    require(value.get("governance") == governance() and value.get("claim_limits") == claim_limits(), "rev23 governance mismatch")
    return value


def validate_predecessor() -> dict[str, Any]:
    require(PREDECESSOR_PATH.is_file(), "rev22 predecessor is missing")
    raw = PREDECESSOR_PATH.read_bytes()
    require(sha256_bytes(raw) == PREDECESSOR_SHA256, "rev22 predecessor SHA-256 mismatch")
    verified = rev22_verifier.verify_artifact(PREDECESSOR_PATH)
    value = strict_json_bytes(raw, "rev22 predecessor")
    require(
        verified.get("decision") == value.get("decision")
        and verified.get("adapter_contract_sha256") == value.get("adapter_contract_sha256")
        and verified.get("rev22_source_binding") == value.get("rev22_source_binding")
        and value.get("decision", {}).get("passed") is True
        and value.get("decision", {}).get("outcome") == "read_only_matrix_observation_adapter_preregistration_passed"
        and value.get("decision", {}).get("next_step") == "implement_and_run_read_only_matrix_observation_adapter_runtime_probe"
        and value.get("adapter_contract_sha256")
        == "05105dbb7cf8646d0c7a5bf667cc9ab78de76131819a9654e43d9465a31d5b43",
        "rev22 predecessor decision/full verification mismatch",
    )
    return {
        "path": PREDECESSOR_PATH.relative_to(REPO_ROOT).as_posix(),
        "sha256": PREDECESSOR_SHA256,
        "outcome": value["decision"]["outcome"],
        "next_step": value["decision"]["next_step"],
        "adapter_contract_sha256": value["adapter_contract_sha256"],
        "full_verification_passed": True,
    }


def validate_runtime_parent_synthesis() -> dict[str, Any]:
    require(RUNTIME_PARENT_SYNTHESIS_PATH.is_file(), "rev20 runtime-parent synthesis is missing")
    value, raw = _read_strict_json(RUNTIME_PARENT_SYNTHESIS_PATH, "rev20 runtime-parent synthesis")
    require(sha256_bytes(raw) == RUNTIME_PARENT_SYNTHESIS_SHA256, "rev20 runtime-parent synthesis SHA-256 mismatch")
    require(
        value.get("schema_version") == "g009.r0.rev20.terrain_contact_matrix_synthesis.v1"
        and value.get("decision", {}).get("outcome") == "terrain_pair_matrix_authority_candidate_validated",
        "rev20 runtime-parent synthesis decision mismatch",
    )
    return {
        "path": RUNTIME_PARENT_SYNTHESIS_PATH.relative_to(REPO_ROOT).as_posix(),
        "sha256": RUNTIME_PARENT_SYNTHESIS_SHA256,
        "outcome": value["decision"]["outcome"],
    }


def _git_bytes(args: list[str]) -> bytes:
    try:
        completed = subprocess.run(["git", *args], cwd=REPO_ROOT, check=False, capture_output=True)
    except OSError as error:
        raise OperationalVerificationError(f"git execution failed: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise OperationalVerificationError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def source_bundle_provenance(paths: tuple[str, ...] = SOURCE_BINDING_PATHS) -> dict[str, Any]:
    commit = _git_bytes(["rev-parse", "HEAD"]).decode("ascii").strip()
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "HEAD commit is malformed")
    dirty = _git_bytes(["status", "--porcelain=v1", "--untracked-files=all", "--", *paths]).decode("utf-8", errors="replace").splitlines()
    require(not dirty, "rev23 source paths must be committed and clean: " + "; ".join(dirty))
    files: dict[str, str] = {}
    for relative in paths:
        path = REPO_ROOT / relative
        require(path.is_file(), f"rev23 source path is missing: {relative}")
        worktree = path.read_bytes()
        blob = _git_bytes(["show", f"{commit}:{relative}"])
        require(worktree == blob, f"rev23 worktree differs from commit blob: {relative}")
        files[relative] = sha256_bytes(blob)
    payload = "\n".join(f"{path}:{files[path]}" for path in paths)
    value = {
        "schema_version": 1,
        "git_commit": commit,
        "source_binding_paths": list(paths),
        "source_binding_files": files,
        "source_bundle_sha256": sha256_bytes(payload.encode("utf-8")),
        "path_scoped_clean": True,
    }
    return validate_recorded_source_bundle(value, paths)


def validate_recorded_source_bundle(
    value: Mapping[str, Any], paths: tuple[str, ...] = SOURCE_BINDING_PATHS
) -> dict[str, Any]:
    """Validate recorded source bytes against their commit, independent of current HEAD."""

    require(
        set(value)
        == {
            "schema_version",
            "git_commit",
            "source_binding_paths",
            "source_binding_files",
            "source_bundle_sha256",
            "path_scoped_clean",
        },
        "rev23 source bundle schema mismatch",
    )
    commit = value.get("git_commit")
    files = value.get("source_binding_files")
    require(
        value.get("schema_version") == 1
        and isinstance(commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", commit) is not None
        and value.get("source_binding_paths") == list(paths)
        and isinstance(files, Mapping)
        and list(files) == list(paths)
        and value.get("path_scoped_clean") is True,
        "rev23 source bundle fields mismatch",
    )
    assert isinstance(commit, str) and isinstance(files, Mapping)
    dirty = _git_bytes(
        ["status", "--porcelain=v1", "--untracked-files=all", "--", *paths]
    ).decode("utf-8", errors="replace").splitlines()
    require(not dirty, "rev23 bound source paths are not clean: " + "; ".join(dirty))
    observed: dict[str, str] = {}
    for relative in paths:
        recorded_digest = files.get(relative)
        require(
            isinstance(recorded_digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", recorded_digest) is not None,
            f"rev23 source digest is malformed: {relative}",
        )
        blob = _git_bytes(["show", f"{commit}:{relative}"])
        try:
            worktree = (REPO_ROOT / relative).read_bytes()
        except OSError as error:
            raise OperationalVerificationError(
                f"cannot read rev23 bound source path {relative}: {error}"
            ) from error
        require(worktree == blob, f"rev23 worktree differs from recorded commit blob: {relative}")
        observed[relative] = sha256_bytes(blob)
        require(observed[relative] == recorded_digest, f"rev23 source digest mismatch: {relative}")
    payload = "\n".join(f"{path}:{observed[path]}" for path in paths)
    require(
        value.get("source_bundle_sha256") == sha256_bytes(payload.encode("utf-8")),
        "rev23 source bundle aggregate mismatch",
    )
    return dict(value)


def validate_uuid4_hex(value: Any, label: str = "execution_id") -> str:
    require(isinstance(value, str), f"{label} missing")
    parsed = uuid.UUID(hex=value)
    require(parsed.version == 4 and parsed.hex == value, f"{label} must be lowercase UUID4 hex")
    return value


def validate_execution(execution: Mapping[str, Any], device: str, replicate: int) -> None:
    require(set(execution) == {"execution_id", "started_at_utc", "output_path_repo_relative", "no_overwrite"}, "execution schema mismatch")
    validate_uuid4_hex(execution.get("execution_id"))
    require(
        execution.get("output_path_repo_relative") == expected_output_relative(device, replicate)
        and execution.get("no_overwrite") is True,
        "canonical no-overwrite execution mismatch",
    )


def cpu_preflight_not_required_binding() -> dict[str, Any]:
    return {
        "status": "not_required_for_cpu",
        "path": None,
        "sha256": None,
        "git_commit": None,
        "probe_source_bundle_sha256": None,
        "input_reports": [],
    }


def validate_cpu_preflight_artifact(path: Path, source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    require(resolved == CPU_PREFLIGHT_PATH.resolve(), "GPU requires the canonical rev23 CPU preflight")
    value, raw = _read_strict_json(resolved, "rev23 CPU preflight")
    import summarize_g009_r0_rev23_matrix_observation_adapter_runtime as rev23_summary

    rev23_summary.validate_cpu_preflight_value(
        value,
        REPO_ROOT,
        resolved.relative_to(REPO_ROOT).as_posix(),
        source_bundle,
    )
    expected_inputs = cast(list[dict[str, str]], value["input_reports"])
    return {
        "status": "validated_for_gpu",
        "path": resolved.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_bytes(raw),
        "git_commit": source_bundle["git_commit"],
        "probe_source_bundle_sha256": source_bundle["source_bundle_sha256"],
        "input_reports": expected_inputs,
    }


def _tensor_pointer(tensor: Any) -> int:
    return int(tensor.untyped_storage().data_ptr() if hasattr(tensor, "untyped_storage") else tensor.data_ptr())


def _tensor_value_sha256(tensor: Any) -> str:
    detached = tensor.detach().clone().contiguous().cpu()
    header = canonical_json({"shape": list(detached.shape), "dtype": str(detached.dtype)}).encode("utf-8")
    return sha256_bytes(header + b"\0" + detached.numpy().tobytes(order="C"))


def _tensor_metadata(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "storage_data_ptr": _tensor_pointer(tensor),
        "storage_offset": int(tensor.storage_offset()),
        "version": int(tensor._version),
        "exact_values_sha256": _tensor_value_sha256(tensor),
    }


class AdapterRuntimeAccumulator:
    def __init__(self, requested_device: str, adapter_contract_sha256: str) -> None:
        self.requested_device = requested_device
        self.adapter_contract_sha256 = adapter_contract_sha256
        self.samples = 0
        self.error: str | None = None
        self.step_ledger: list[dict[str, Any]] = []
        self.representative_snapshots: list[dict[str, Any]] = []
        self.source_mutation_steps: list[int] = []
        self.oracle_mismatch_steps: list[int] = []
        self.alias_violation_steps: list[int] = []
        self.source_contract_violation_steps: list[int] = []
        self.zero_semantics_violation_steps: list[int] = []
        self.device_violation_steps: list[int] = []
        self.nonfinite_steps: list[int] = []
        self.max_magnitude_n = 0.0
        self.magnitude_integral_n_s = 0.0
        self.zero_source_vector_count_total = 0

    def observe(self, step: int, sensor: Any, torch_module: Any) -> None:
        torch = torch_module
        try:
            require(step == self.samples + 1 and 1 <= step <= PHYSICS_STEPS, "adapter sample step mismatch")
            source = sensor.data.force_matrix_w
            require(source is not None, "terrain_pair_force_matrix_w unavailable")
            before_metadata = _tensor_metadata(source)
            before_values = source.detach().clone()
            before_version = int(source._version)

            from isaac_walk_g009.matrix_observation_adapter import adapt_terrain_pair_force_matrix_w

            result = adapt_terrain_pair_force_matrix_w(source)
            after_metadata = _tensor_metadata(source)
            source_exact = bool(torch.equal(source, before_values))
            source_unchanged = (
                before_metadata == after_metadata
                and int(source._version) == before_version
                and source_exact
            )

            oracle_xyz = before_values.sum(dim=2)
            single_filter_oracle = before_values[:, :, 0, :].clone()
            oracle_magnitude = torch.linalg.vector_norm(oracle_xyz, dim=-1)
            oracle_mask = oracle_magnitude > CONTACT_THRESHOLD_N
            xyz_equal = bool(torch.equal(result.world_xyz, oracle_xyz) and torch.equal(oracle_xyz, single_filter_oracle))
            magnitude_equal = bool(torch.equal(result.magnitude, oracle_magnitude))
            mask_equal = bool(torch.equal(result.contact_mask, oracle_mask))
            oracle_equal = xyz_equal and magnitude_equal and mask_equal

            source_pointer = _tensor_pointer(source)
            output_pointers = {
                "world_xyz": _tensor_pointer(result.world_xyz),
                "magnitude": _tensor_pointer(result.magnitude),
                "contact_mask": _tensor_pointer(result.contact_mask),
            }
            source_output_non_alias = all(pointer != source_pointer for pointer in output_pointers.values())
            output_output_non_alias = len(set(output_pointers.values())) == len(output_pointers)
            alias_valid = source_output_non_alias and output_output_non_alias

            shapes_valid = (
                list(source.shape) == [NUM_ENVS, BODY_COUNT, FILTER_COUNT, COMPONENT_COUNT]
                and list(result.world_xyz.shape) == [NUM_ENVS, BODY_COUNT, COMPONENT_COUNT]
                and list(result.magnitude.shape) == [NUM_ENVS, BODY_COUNT]
                and list(result.contact_mask.shape) == [NUM_ENVS, BODY_COUNT]
            )
            dtypes_valid = (
                source.dtype is torch.float32
                and result.world_xyz.dtype is source.dtype
                and result.magnitude.dtype is source.dtype
                and result.contact_mask.dtype is torch.bool
            )
            devices = {
                "source": str(source.device),
                "world_xyz": str(result.world_xyz.device),
                "magnitude": str(result.magnitude.device),
                "contact_mask": str(result.contact_mask.device),
            }
            device_valid = set(devices.values()) == {self.requested_device}
            finite = bool(
                torch.isfinite(source).all().item()
                and torch.isfinite(result.world_xyz).all().item()
                and torch.isfinite(result.magnitude).all().item()
            )
            zero_source = torch.all(source == 0.0, dim=-1).all(dim=-1)
            zero_count = int(zero_source.sum().item())
            self.zero_source_vector_count_total += zero_count
            zero_semantics = bool(
                torch.all(result.world_xyz[zero_source] == 0.0).item()
                and torch.all(result.magnitude[zero_source] == 0.0).item()
                and torch.all(~result.contact_mask[zero_source]).item()
            ) if zero_count else True
            max_magnitude = float(result.magnitude.max().item())
            self.max_magnitude_n = max(self.max_magnitude_n, max_magnitude)
            self.magnitude_integral_n_s += max_magnitude * PHYSICS_DT_S

            if not source_unchanged:
                self.source_mutation_steps.append(step)
            if not oracle_equal:
                self.oracle_mismatch_steps.append(step)
            if not alias_valid:
                self.alias_violation_steps.append(step)
            if not (shapes_valid and dtypes_valid):
                self.source_contract_violation_steps.append(step)
            if not zero_semantics:
                self.zero_semantics_violation_steps.append(step)
            if not device_valid:
                self.device_violation_steps.append(step)
            if not finite:
                self.nonfinite_steps.append(step)

            output_metadata = {
                "world_xyz": _tensor_metadata(result.world_xyz),
                "magnitude": _tensor_metadata(result.magnitude),
                "contact_mask": _tensor_metadata(result.contact_mask),
            }
            row = {
                "step": step,
                "source_before": before_metadata,
                "source_after": after_metadata,
                "output_metadata": output_metadata,
                "devices": devices,
                "source_unchanged": source_unchanged,
                "world_xyz_oracle_equal": xyz_equal,
                "magnitude_oracle_equal": magnitude_equal,
                "contact_mask_oracle_equal": mask_equal,
                "source_output_non_alias": source_output_non_alias,
                "output_output_non_alias": output_output_non_alias,
                "shape_contract_valid": shapes_valid,
                "dtype_contract_valid": dtypes_valid,
                "device_contract_valid": device_valid,
                "finite": finite,
                "zero_source_vector_count": zero_count,
                "zero_contact_semantics_valid": zero_semantics,
                "contact_body_count": int(result.contact_mask.sum().item()),
                "max_magnitude_n": max_magnitude,
                "per_env_max_magnitude_n": [float(value) for value in result.magnitude.amax(dim=1).detach().cpu().tolist()],
                "per_env_contact_body_count": [int(value) for value in result.contact_mask.sum(dim=1).detach().cpu().tolist()],
            }
            self.step_ledger.append(row)
            if step in REPRESENTATIVE_STEPS:
                self.representative_snapshots.append(
                    {
                        "step": step,
                        "source": source.detach().clone().cpu().tolist(),
                        "world_xyz": result.world_xyz.detach().clone().cpu().tolist(),
                        "magnitude": result.magnitude.detach().clone().cpu().tolist(),
                        "contact_mask": result.contact_mask.detach().clone().cpu().tolist(),
                    }
                )
            self.samples += 1
        except Exception as error:
            if str(error) == "adapter sample step mismatch":
                raise
            if self.error is None:
                self.error = f"{type(error).__name__}: {error}"
            self.step_ledger.append({"step": step, "adapter_error": f"{type(error).__name__}: {error}"})
            self.samples += 1

    def snapshot(self) -> dict[str, Any]:
        complete_steps = [row for row in self.step_ledger if "adapter_error" not in row]
        first = complete_steps[0] if complete_steps else {}
        source_before = cast(Mapping[str, Any], first.get("source_before", {}))
        outputs = cast(Mapping[str, Any], first.get("output_metadata", {}))
        xyz = cast(Mapping[str, Any], outputs.get("world_xyz", {}))
        magnitude = cast(Mapping[str, Any], outputs.get("magnitude", {}))
        mask = cast(Mapping[str, Any], outputs.get("contact_mask", {}))
        checks = {
            "exact_150_samples": self.samples == PHYSICS_STEPS,
            "source_available_150_of_150": len(complete_steps) == PHYSICS_STEPS,
            "source_contract_150_of_150": not self.source_contract_violation_steps,
            "source_unchanged_150_of_150": not self.source_mutation_steps,
            "world_xyz_oracle_150_of_150": not self.oracle_mismatch_steps and all(row.get("world_xyz_oracle_equal") is True for row in complete_steps),
            "magnitude_oracle_150_of_150": not self.oracle_mismatch_steps and all(row.get("magnitude_oracle_equal") is True for row in complete_steps),
            "contact_mask_oracle_150_of_150": not self.oracle_mismatch_steps and all(row.get("contact_mask_oracle_equal") is True for row in complete_steps),
            "source_output_non_alias_150_of_150": not self.alias_violation_steps and all(row.get("source_output_non_alias") is True for row in complete_steps),
            "output_output_non_alias_150_of_150": not self.alias_violation_steps and all(row.get("output_output_non_alias") is True for row in complete_steps),
            "zero_contact_semantics_150_of_150": not self.zero_semantics_violation_steps,
            "finite_150_of_150": not self.nonfinite_steps and all(row.get("finite") is True for row in complete_steps),
            "requested_device_preserved_150_of_150": not self.device_violation_steps,
            "representative_snapshots_exact": [item.get("step") for item in self.representative_snapshots] == list(REPRESENTATIVE_STEPS),
            "collection_error_absent": self.error is None,
        }
        return {
            "schema_version": "g009.r0.rev23.adapter_runtime_observation.v1",
            "adapter_contract_sha256": self.adapter_contract_sha256,
            "requested_device": self.requested_device,
            "sample_count": self.samples,
            "source_shape": source_before.get("shape"),
            "world_xyz_shape": xyz.get("shape"),
            "magnitude_shape": magnitude.get("shape"),
            "contact_mask_shape": mask.get("shape"),
            "source_dtype": source_before.get("dtype"),
            "world_xyz_dtype": xyz.get("dtype"),
            "magnitude_dtype": magnitude.get("dtype"),
            "contact_mask_dtype": mask.get("dtype"),
            "source_device": source_before.get("device"),
            "world_xyz_device": xyz.get("device"),
            "magnitude_device": magnitude.get("device"),
            "contact_mask_device": mask.get("device"),
            "step_ledger": self.step_ledger,
            "representative_snapshots": self.representative_snapshots,
            "source_mutation_steps": self.source_mutation_steps,
            "oracle_mismatch_steps": self.oracle_mismatch_steps,
            "alias_violation_steps": self.alias_violation_steps,
            "source_contract_violation_steps": self.source_contract_violation_steps,
            "zero_semantics_violation_steps": self.zero_semantics_violation_steps,
            "device_violation_steps": self.device_violation_steps,
            "nonfinite_steps": self.nonfinite_steps,
            "zero_source_vector_count_total": self.zero_source_vector_count_total,
            "max_magnitude_n": self.max_magnitude_n,
            "magnitude_integral_n_s": self.magnitude_integral_n_s,
            "checks": checks,
            "passed": all(checks.values()),
            "error": self.error,
        }


def runtime_contract(device: str, replicate: int, adapter_contract_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": "G009-5-E016",
        "revision": "rev23",
        "slot": f"{device}.rep{replicate}",
        "single_changed_axis": "read-only matrix observation adapter runtime implementation only",
        "adapter_contract_sha256": adapter_contract_sha256,
        "runtime_parent": "G009-5-E013 rev20 unchanged terrain-pair matrix runtime",
        "runtime": {
            "num_envs": NUM_ENVS,
            "physics_steps": PHYSICS_STEPS,
            "physics_dt_s": PHYSICS_DT_S,
            "headless": True,
            "render": False,
        },
        "governance": governance(),
        "canonical_output": expected_output_relative(device, replicate),
    }


@contextmanager
def _patched_rev20_output_contract() -> Iterator[None]:
    original = rev20.expected_output_relative
    rev20.expected_output_relative = expected_output_relative
    try:
        yield
    finally:
        rev20.expected_output_relative = original


def _runtime_parent_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(report)
    for key in (
        "evidence_id",
        "rev22_predecessor",
        "runtime_parent",
        "rev23_source_bundle",
        "adapter_runtime",
        "adapter_decision",
        "claim_limits",
    ):
        projected.pop(key, None)
    projected.update(
        {
            "schema_version": rev20.SCHEMA_VERSION,
            "experiment_id": "G009-5-E013",
            "revision": "rev20",
        }
    )
    device = str(projected["device"])
    replicate = int(projected["replicate_index"])
    with _patched_rev20_output_contract():
        contract = rev20.probe_contract(device, replicate)
        projected["contract"] = contract
        projected["contract_sha256"] = rev20.canonical_sha256(contract)
        projected["feasibility"] = rev20.derive_feasibility(projected)
        rev20.validate_report(projected)
    return projected


_TENSOR_METADATA_KEYS = {
    "shape",
    "dtype",
    "device",
    "stride",
    "storage_data_ptr",
    "storage_offset",
    "version",
    "exact_values_sha256",
}


def _plain_int(value: Any) -> bool:
    return type(value) is int


def _finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _validate_tensor_metadata(
    value: Any,
    *,
    label: str,
    shape: list[int],
    dtype: str,
    device: str,
) -> dict[str, Any]:
    require(isinstance(value, Mapping) and set(value) == _TENSOR_METADATA_KEYS, f"{label} metadata schema mismatch")
    metadata = dict(value)
    stride = metadata.get("stride")
    require(
        metadata.get("shape") == shape
        and metadata.get("dtype") == dtype
        and metadata.get("device") == device
        and isinstance(stride, list)
        and len(stride) == len(shape)
        and all(_plain_int(item) and item >= 0 for item in stride)
        and _plain_int(metadata.get("storage_data_ptr"))
        and metadata["storage_data_ptr"] > 0
        and _plain_int(metadata.get("storage_offset"))
        and metadata["storage_offset"] >= 0
        and _plain_int(metadata.get("version"))
        and metadata["version"] >= 0
        and isinstance(metadata.get("exact_values_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", metadata["exact_values_sha256"]) is not None,
        f"{label} metadata fields mismatch",
    )
    return metadata


def _validate_snapshot_leaf_types(value: Any, *, boolean: bool, label: str) -> None:
    if isinstance(value, list):
        for item in value:
            _validate_snapshot_leaf_types(item, boolean=boolean, label=label)
        return
    if boolean:
        require(type(value) is bool, f"{label} contains a non-boolean leaf")
    else:
        require(_finite_number(value), f"{label} contains a non-finite or non-numeric leaf")


def _snapshot_tensor(value: Any, *, dtype: Any, device: str, shape: list[int], label: str) -> Any:
    import torch

    _validate_snapshot_leaf_types(value, boolean=dtype is torch.bool, label=label)
    try:
        tensor = torch.tensor(value, dtype=dtype, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{label} tensor reconstruction failed: {error}") from error
    require(list(tensor.shape) == shape, f"{label} shape mismatch")
    return tensor


def validate_adapter_runtime(value: Mapping[str, Any], requested_device: str) -> None:
    expected_top_keys = {
        "schema_version",
        "adapter_contract_sha256",
        "requested_device",
        "sample_count",
        "source_shape",
        "world_xyz_shape",
        "magnitude_shape",
        "contact_mask_shape",
        "source_dtype",
        "world_xyz_dtype",
        "magnitude_dtype",
        "contact_mask_dtype",
        "source_device",
        "world_xyz_device",
        "magnitude_device",
        "contact_mask_device",
        "step_ledger",
        "representative_snapshots",
        "source_mutation_steps",
        "oracle_mismatch_steps",
        "alias_violation_steps",
        "source_contract_violation_steps",
        "zero_semantics_violation_steps",
        "device_violation_steps",
        "nonfinite_steps",
        "zero_source_vector_count_total",
        "max_magnitude_n",
        "magnitude_integral_n_s",
        "checks",
        "passed",
        "error",
    }
    require(set(value) == expected_top_keys, "adapter runtime top-level schema mismatch")
    require(value.get("schema_version") == "g009.r0.rev23.adapter_runtime_observation.v1", "adapter runtime schema mismatch")
    require(
        value.get("adapter_contract_sha256") == "05105dbb7cf8646d0c7a5bf667cc9ab78de76131819a9654e43d9465a31d5b43"
        and value.get("requested_device") == requested_device
        and value.get("sample_count") == PHYSICS_STEPS,
        "adapter runtime identity mismatch",
    )
    require(
        value.get("source_shape") == [NUM_ENVS, BODY_COUNT, FILTER_COUNT, COMPONENT_COUNT]
        and value.get("world_xyz_shape") == [NUM_ENVS, BODY_COUNT, COMPONENT_COUNT]
        and value.get("magnitude_shape") == [NUM_ENVS, BODY_COUNT]
        and value.get("contact_mask_shape") == [NUM_ENVS, BODY_COUNT]
        and value.get("source_dtype") == value.get("world_xyz_dtype") == value.get("magnitude_dtype") == "torch.float32"
        and value.get("contact_mask_dtype") == "torch.bool"
        and value.get("source_device") == value.get("world_xyz_device") == value.get("magnitude_device") == value.get("contact_mask_device") == requested_device,
        "adapter runtime shape/dtype/device mismatch",
    )
    ledger = value.get("step_ledger")
    require(
        isinstance(ledger, list)
        and len(ledger) == PHYSICS_STEPS
        and all(isinstance(row, Mapping) for row in ledger)
        and [row.get("step") for row in ledger] == list(range(1, PHYSICS_STEPS + 1)),
        "adapter step ledger mismatch",
    )
    require(
        value.get("source_mutation_steps") == []
        and value.get("oracle_mismatch_steps") == []
        and value.get("alias_violation_steps") == []
        and value.get("source_contract_violation_steps") == []
        and value.get("zero_semantics_violation_steps") == []
        and value.get("device_violation_steps") == []
        and value.get("nonfinite_steps") == []
        and value.get("error") is None,
        "adapter runtime violation ledger is not empty",
    )
    row_keys = {
        "step",
        "source_before",
        "source_after",
        "output_metadata",
        "devices",
        "source_unchanged",
        "world_xyz_oracle_equal",
        "magnitude_oracle_equal",
        "contact_mask_oracle_equal",
        "source_output_non_alias",
        "output_output_non_alias",
        "shape_contract_valid",
        "dtype_contract_valid",
        "device_contract_valid",
        "finite",
        "zero_source_vector_count",
        "zero_contact_semantics_valid",
        "contact_body_count",
        "max_magnitude_n",
        "per_env_max_magnitude_n",
        "per_env_contact_body_count",
    }
    boolean_fields = (
        "source_unchanged",
        "world_xyz_oracle_equal",
        "magnitude_oracle_equal",
        "contact_mask_oracle_equal",
        "source_output_non_alias",
        "output_output_non_alias",
        "shape_contract_valid",
        "dtype_contract_valid",
        "device_contract_valid",
        "finite",
        "zero_contact_semantics_valid",
    )
    total_zero_count = 0
    ledger_max = 0.0
    ledger_integral = 0.0
    rows_by_step: dict[int, Mapping[str, Any]] = {}
    for row_value in cast(list[Mapping[str, Any]], ledger):
        require(set(row_value) == row_keys, "adapter ledger row schema mismatch")
        step = cast(int, row_value["step"])
        rows_by_step[step] = row_value
        source_before = _validate_tensor_metadata(
            row_value.get("source_before"),
            label=f"step {step} source_before",
            shape=[NUM_ENVS, BODY_COUNT, FILTER_COUNT, COMPONENT_COUNT],
            dtype="torch.float32",
            device=requested_device,
        )
        source_after = _validate_tensor_metadata(
            row_value.get("source_after"),
            label=f"step {step} source_after",
            shape=[NUM_ENVS, BODY_COUNT, FILTER_COUNT, COMPONENT_COUNT],
            dtype="torch.float32",
            device=requested_device,
        )
        require(source_after == source_before, f"step {step} source metadata/value mutation")
        output_metadata = row_value.get("output_metadata")
        require(
            isinstance(output_metadata, Mapping)
            and set(output_metadata) == {"world_xyz", "magnitude", "contact_mask"},
            f"step {step} output metadata schema mismatch",
        )
        assert isinstance(output_metadata, Mapping)
        xyz_metadata = _validate_tensor_metadata(
            output_metadata.get("world_xyz"),
            label=f"step {step} world_xyz",
            shape=[NUM_ENVS, BODY_COUNT, COMPONENT_COUNT],
            dtype="torch.float32",
            device=requested_device,
        )
        magnitude_metadata = _validate_tensor_metadata(
            output_metadata.get("magnitude"),
            label=f"step {step} magnitude",
            shape=[NUM_ENVS, BODY_COUNT],
            dtype="torch.float32",
            device=requested_device,
        )
        mask_metadata = _validate_tensor_metadata(
            output_metadata.get("contact_mask"),
            label=f"step {step} contact_mask",
            shape=[NUM_ENVS, BODY_COUNT],
            dtype="torch.bool",
            device=requested_device,
        )
        source_pointer = source_before["storage_data_ptr"]
        output_pointers = [
            xyz_metadata["storage_data_ptr"],
            magnitude_metadata["storage_data_ptr"],
            mask_metadata["storage_data_ptr"],
        ]
        require(
            all(pointer != source_pointer for pointer in output_pointers)
            and len(set(output_pointers)) == len(output_pointers),
            f"step {step} tensor alias metadata mismatch",
        )
        devices = row_value.get("devices")
        require(
            devices
            == {
                "source": requested_device,
                "world_xyz": requested_device,
                "magnitude": requested_device,
                "contact_mask": requested_device,
            },
            f"step {step} device ledger mismatch",
        )
        require(all(row_value.get(field) is True for field in boolean_fields), f"step {step} boolean evidence failed")
        zero_count = row_value.get("zero_source_vector_count")
        contact_count = row_value.get("contact_body_count")
        require(
            _plain_int(zero_count)
            and 0 <= zero_count <= NUM_ENVS * BODY_COUNT
            and _plain_int(contact_count)
            and 0 <= contact_count <= NUM_ENVS * BODY_COUNT,
            f"step {step} contact count mismatch",
        )
        per_env_max = row_value.get("per_env_max_magnitude_n")
        per_env_contact = row_value.get("per_env_contact_body_count")
        require(
            isinstance(per_env_max, list)
            and len(per_env_max) == NUM_ENVS
            and all(_finite_number(item) and float(item) >= 0.0 for item in per_env_max)
            and isinstance(per_env_contact, list)
            and len(per_env_contact) == NUM_ENVS
            and all(_plain_int(item) and 0 <= item <= BODY_COUNT for item in per_env_contact)
            and sum(per_env_contact) == contact_count,
            f"step {step} per-environment metrics mismatch",
        )
        row_max = row_value.get("max_magnitude_n")
        require(
            _finite_number(row_max)
            and float(row_max) >= 0.0
            and float(row_max) == max(float(item) for item in per_env_max),
            f"step {step} maximum magnitude mismatch",
        )
        total_zero_count += cast(int, zero_count)
        ledger_max = max(ledger_max, float(row_max))
        ledger_integral += float(row_max) * PHYSICS_DT_S
    require(
        value.get("zero_source_vector_count_total") == total_zero_count
        and _finite_number(value.get("max_magnitude_n"))
        and float(cast(float, value.get("max_magnitude_n"))) == ledger_max
        and _finite_number(value.get("magnitude_integral_n_s"))
        and math.isclose(
            float(cast(float, value.get("magnitude_integral_n_s"))),
            ledger_integral,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ),
        "adapter runtime aggregate mismatch",
    )
    checks = value.get("checks")
    expected_checks = {
        "exact_150_samples",
        "source_available_150_of_150",
        "source_contract_150_of_150",
        "source_unchanged_150_of_150",
        "world_xyz_oracle_150_of_150",
        "magnitude_oracle_150_of_150",
        "contact_mask_oracle_150_of_150",
        "source_output_non_alias_150_of_150",
        "output_output_non_alias_150_of_150",
        "zero_contact_semantics_150_of_150",
        "finite_150_of_150",
        "requested_device_preserved_150_of_150",
        "representative_snapshots_exact",
        "collection_error_absent",
    }
    require(
        isinstance(checks, Mapping)
        and set(checks) == expected_checks
        and all(item is True for item in checks.values())
        and value.get("passed") is True,
        "adapter runtime checks failed",
    )
    snapshots = value.get("representative_snapshots")
    require(
        isinstance(snapshots, list)
        and len(snapshots) == len(REPRESENTATIVE_STEPS)
        and all(isinstance(item, Mapping) for item in snapshots)
        and [item.get("step") for item in snapshots] == list(REPRESENTATIVE_STEPS),
        "adapter representative snapshots mismatch",
    )
    import torch

    for snapshot in cast(list[Mapping[str, Any]], snapshots):
        require(
            set(snapshot) == {"step", "source", "world_xyz", "magnitude", "contact_mask"},
            "adapter representative snapshot schema mismatch",
        )
        step = cast(int, snapshot["step"])
        row = rows_by_step[step]
        source = _snapshot_tensor(
            snapshot.get("source"),
            dtype=torch.float32,
            device=requested_device,
            shape=[NUM_ENVS, BODY_COUNT, FILTER_COUNT, COMPONENT_COUNT],
            label=f"step {step} snapshot source",
        )
        world_xyz = _snapshot_tensor(
            snapshot.get("world_xyz"),
            dtype=torch.float32,
            device=requested_device,
            shape=[NUM_ENVS, BODY_COUNT, COMPONENT_COUNT],
            label=f"step {step} snapshot world_xyz",
        )
        magnitude = _snapshot_tensor(
            snapshot.get("magnitude"),
            dtype=torch.float32,
            device=requested_device,
            shape=[NUM_ENVS, BODY_COUNT],
            label=f"step {step} snapshot magnitude",
        )
        contact_mask = _snapshot_tensor(
            snapshot.get("contact_mask"),
            dtype=torch.bool,
            device=requested_device,
            shape=[NUM_ENVS, BODY_COUNT],
            label=f"step {step} snapshot contact_mask",
        )
        oracle_xyz = source.sum(dim=2)
        oracle_magnitude = torch.linalg.vector_norm(oracle_xyz, dim=-1)
        oracle_mask = oracle_magnitude > CONTACT_THRESHOLD_N
        require(
            torch.equal(world_xyz, oracle_xyz)
            and torch.equal(magnitude, oracle_magnitude)
            and torch.equal(contact_mask, oracle_mask),
            f"step {step} representative snapshot oracle mismatch",
        )
        source_zero = torch.all(source == 0.0, dim=-1).all(dim=-1)
        expected_per_env_max = [float(item) for item in magnitude.amax(dim=1).detach().cpu().tolist()]
        expected_per_env_contact = [int(item) for item in contact_mask.sum(dim=1).detach().cpu().tolist()]
        output_metadata = cast(Mapping[str, Mapping[str, Any]], row["output_metadata"])
        source_before = cast(Mapping[str, Any], row["source_before"])
        require(
            source_before["exact_values_sha256"] == _tensor_value_sha256(source)
            and output_metadata["world_xyz"]["exact_values_sha256"] == _tensor_value_sha256(world_xyz)
            and output_metadata["magnitude"]["exact_values_sha256"] == _tensor_value_sha256(magnitude)
            and output_metadata["contact_mask"]["exact_values_sha256"] == _tensor_value_sha256(contact_mask),
            f"step {step} representative snapshot hash mismatch",
        )
        require(
            row["zero_source_vector_count"] == int(source_zero.sum().item())
            and row["contact_body_count"] == int(contact_mask.sum().item())
            and row["max_magnitude_n"] == float(magnitude.max().item())
            and row["per_env_max_magnitude_n"] == expected_per_env_max
            and row["per_env_contact_body_count"] == expected_per_env_contact
            and bool(torch.all(world_xyz[source_zero] == 0.0).item())
            and bool(torch.all(magnitude[source_zero] == 0.0).item())
            and bool(torch.all(~contact_mask[source_zero]).item()),
            f"step {step} representative snapshot ledger mismatch",
        )


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    require(
        report.get("schema_version") == SCHEMA_VERSION
        and report.get("experiment_id") == "G009-5-E016"
        and report.get("evidence_id") == "G009-5-E016"
        and report.get("revision") == "rev23"
        and report.get("status") == "complete",
        "rev23 report identity mismatch",
    )
    device = str(report.get("device"))
    replicate = int(cast(int, report.get("replicate_index")))
    require(device in {"cpu", "cuda:0"} and replicate in {1, 2}, "rev23 report slot mismatch")
    validate_execution(cast(Mapping[str, Any], report.get("execution", {})), device, replicate)
    predecessor = validate_predecessor()
    parent = validate_runtime_parent_synthesis()
    require(report.get("rev22_predecessor") == predecessor and report.get("runtime_parent") == parent, "rev23 predecessor/runtime-parent mismatch")
    source = validate_recorded_source_bundle(
        cast(Mapping[str, Any], report.get("rev23_source_bundle", {}))
    )
    contract = runtime_contract(device, replicate, predecessor["adapter_contract_sha256"])
    require(report.get("contract") == contract and report.get("contract_sha256") == canonical_sha256(contract), "rev23 runtime contract/hash mismatch")
    require(report.get("governance") == governance() and report.get("claim_limits") == claim_limits(), "rev23 governance/claim limits mismatch")
    _runtime_parent_projection(report)
    adapter = cast(Mapping[str, Any], report.get("adapter_runtime", {}))
    validate_adapter_runtime(adapter, device)
    parent_valid = bool(report.get("feasibility", {}).get("run_interpretable") is True)
    expected_outcome = "read_only_matrix_observation_adapter_runtime_run_passed" if parent_valid and adapter.get("passed") is True else "read_only_matrix_observation_adapter_runtime_run_failed"
    expected_decision = {
        "passed": expected_outcome.endswith("_passed"),
        "outcome": expected_outcome,
        "next_step": "await_same_device_repeatability_or_synthesis",
    }
    require(report.get("adapter_decision") == expected_decision, "rev23 adapter decision mismatch")
    return expected_decision


def diagnose(args: argparse.Namespace, execution: dict[str, Any]) -> dict[str, Any]:
    prereg = load_preregistration()
    predecessor = validate_predecessor()
    parent = validate_runtime_parent_synthesis()
    source_bundle = source_bundle_provenance()
    rev20.validate_external_sources(args.isaaclab_root, rev20.load_preregistration())

    import torch

    holder: dict[str, AdapterRuntimeAccumulator] = {}
    original_accumulator = rev20.MatrixSafetyAccumulator

    class CombinedAccumulator(original_accumulator):
        def __init__(self, *combined_args: Any, **combined_kwargs: Any) -> None:
            super().__init__(*combined_args, **combined_kwargs)
            adapter_accumulator = AdapterRuntimeAccumulator(args.device, predecessor["adapter_contract_sha256"])
            holder["adapter"] = adapter_accumulator
            self._rev23_adapter_accumulator = adapter_accumulator

        def observe(self, step: int, sensor: Any, robot: Any, physics_dt_s: float, torch_module: Any) -> None:
            super().observe(step, sensor, robot, physics_dt_s, torch_module)
            self._rev23_adapter_accumulator.observe(step, sensor, torch_module)

    rev20.MatrixSafetyAccumulator = CombinedAccumulator
    try:
        with _patched_rev20_output_contract():
            base_report = rev20.diagnose(args, execution)
    finally:
        rev20.MatrixSafetyAccumulator = original_accumulator
    require("adapter" in holder, "rev23 adapter accumulator was not constructed")

    report = dict(base_report)
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": "G009-5-E016",
            "evidence_id": "G009-5-E016",
            "revision": "rev23",
        }
    )
    contract = runtime_contract(args.device, args.replicate_index, predecessor["adapter_contract_sha256"])
    adapter_runtime = holder["adapter"].snapshot()
    report["contract"] = contract
    report["contract_sha256"] = canonical_sha256(contract)
    report["rev22_predecessor"] = predecessor
    report["runtime_parent"] = parent
    report["rev23_source_bundle"] = source_bundle
    report["adapter_runtime"] = adapter_runtime
    report["claim_limits"] = claim_limits()
    parent_valid = bool(report.get("feasibility", {}).get("run_interpretable") is True)
    passed = parent_valid and adapter_runtime["passed"] is True
    report["adapter_decision"] = {
        "passed": passed,
        "outcome": "read_only_matrix_observation_adapter_runtime_run_passed" if passed else "read_only_matrix_observation_adapter_runtime_run_failed",
        "next_step": "await_same_device_repeatability_or_synthesis",
    }
    report["governance"] = governance()
    return report


def build_core_help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, choices=(DEFAULT_TASK,))
    parser.add_argument("--seed", type=int, default=42, choices=(42,))
    parser.add_argument("--replicate-index", required=True, type=int, choices=(1, 2))
    parser.add_argument("--cpu-preflight", type=Path)
    parser.add_argument("--isaaclab-root", type=Path, default=REPO_ROOT.parent / "IsaacLab")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    parser.add_argument("--headless", action="store_true", required=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return rev20.parse_args(argv)


def parse_prelaunch_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse only rev23 gate fields without importing Isaac Lab."""

    parser = build_core_help_parser()
    args, _unknown = parser.parse_known_args(argv)
    return args


def failure_envelope(args: argparse.Namespace, execution: Mapping[str, Any], error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "experiment_id": "G009-5-E016",
        "revision": "rev23",
        "status": "failed_closed",
        "device": getattr(args, "device", None),
        "replicate_index": getattr(args, "replicate_index", None),
        "execution": dict(execution),
        "governance": governance(),
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def failed_attempt_path(args: argparse.Namespace, execution: Mapping[str, Any]) -> Path:
    device = str(args.device).replace(":", "_")
    replicate = int(args.replicate_index)
    execution_id = validate_uuid4_hex(execution.get("execution_id"))
    return Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic/failed_attempts/rev23" / f"g009_r0_rev23_matrix_observation_adapter_{device}_rep{replicate:02d}_{execution_id}.json"


def prelaunch_validate(args: argparse.Namespace) -> dict[str, Any]:
    """Complete every local/source/preflight gate before AppLauncher is imported."""

    load_preregistration()
    validate_predecessor()
    validate_runtime_parent_synthesis()
    source_bundle = source_bundle_provenance()
    rev20_preregistration = rev20.load_preregistration()
    rev20.validate_source_bundle(rev20.source_bundle_provenance())
    rev20.validate_external_sources(args.isaaclab_root, rev20_preregistration)
    return (
        validate_cpu_preflight_artifact(args.cpu_preflight, source_bundle)
        if args.device == "cuda:0"
        else cpu_preflight_not_required_binding()
    )


def _record_failed_attempt(
    args: argparse.Namespace,
    execution: Mapping[str, Any],
    error: BaseException,
    *,
    status: str,
    diagnostic_report: Mapping[str, Any] | None = None,
) -> Path:
    failure_path = failed_attempt_path(args, execution)
    envelope = failure_envelope(args, execution, error)
    if diagnostic_report is not None:
        envelope["diagnostic_report"] = dict(diagnostic_report)
    runtime_probe._write_json_atomic(failure_path, envelope)
    print(
        json.dumps(
            {"status": status, "failure_report": str(failure_path)},
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )
    return failure_path


def _publish_canonical_report(output: Path, report: Mapping[str, Any]) -> str | None:
    """Publish once and recover only a verified post-link temp cleanup failure."""

    payload = (json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        runtime_probe._write_json_atomic(output, dict(report))
        return None
    except FileExistsError:
        raise
    except OSError as error:
        linked_by_this_process = False
        try:
            linked_by_this_process = (
                output.is_file()
                and temporary.is_file()
                and output.read_bytes() == payload
                and temporary.read_bytes() == payload
                and output.samefile(temporary)
            )
        except OSError:
            linked_by_this_process = False
        if not linked_by_this_process:
            raise
        cleanup_retry = "temporary cleanup retry succeeded"
        try:
            temporary.unlink()
        except OSError as cleanup_error:
            cleanup_retry = (
                "temporary cleanup retry failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        return (
            f"post-link temporary cleanup warning: {type(error).__name__}: {error}; "
            f"{cleanup_retry}"
        )


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if any(token in {"-h", "--help"} for token in effective_argv):
        build_core_help_parser().print_help()
        return 0
    try:
        output, execution = runtime_probe.prepare_execution(runtime_probe.parse_prelaunch_output(argv))
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "prelaunch_rejected_without_consuming_canonical_output",
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    prelaunch_args = parse_prelaunch_args(argv)
    validate_execution(execution, prelaunch_args.device, prelaunch_args.replicate_index)
    try:
        cpu_preflight_binding = prelaunch_validate(prelaunch_args)
    except (
        OperationalVerificationError,
        rev22_verifier.OperationalVerificationError,
        subprocess.SubprocessError,
        OSError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "prelaunch_operational_verification_failed_without_consuming_canonical_output",
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 3
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "prelaunch_rejected_without_consuming_canonical_output",
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2

    args = parse_args(argv)
    require(
        all(
            getattr(args, field) == getattr(prelaunch_args, field)
            for field in (
                "task",
                "seed",
                "replicate_index",
                "cpu_preflight",
                "isaaclab_root",
                "output",
                "device",
                "headless",
            )
        ),
        "full launcher arguments differ from prelaunch gate arguments",
    )
    validate_execution(execution, args.device, args.replicate_index)
    args._cpu_preflight_binding = cpu_preflight_binding

    from isaaclab.app import AppLauncher

    app = None
    try:
        app = AppLauncher(args).app
        report = diagnose(args, execution)
    except Exception as error:
        if app is not None:
            try:
                app.close()
            except Exception as close_error:
                error = RuntimeError(
                    f"{type(error).__name__}: {error}; app close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            app = None
        _record_failed_attempt(
            args,
            execution,
            error,
            status="runtime_failed_without_consuming_canonical_output",
        )
        return 2

    passed = bool(report["adapter_decision"]["passed"])
    if passed:
        try:
            validate_report(report)
        except Exception as error:
            try:
                app.close()
            except Exception as close_error:
                error = RuntimeError(
                    f"{type(error).__name__}: {error}; app close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            app = None
            _record_failed_attempt(
                args,
                execution,
                error,
                status="runtime_report_validation_failed_without_consuming_canonical_output",
                diagnostic_report=report,
            )
            return 2

    try:
        app.close()
        app = None
    except Exception as error:
        app = None
        _record_failed_attempt(
            args,
            execution,
            error,
            status="runtime_shutdown_failed_without_consuming_canonical_output",
            diagnostic_report=report,
        )
        return 3

    if not passed:
        _record_failed_attempt(
            args,
            execution,
            RuntimeError("adapter runtime decision did not pass"),
            status="runtime_rejected_without_consuming_canonical_output",
            diagnostic_report=report,
        )
        return 2
    try:
        cleanup_warning = _publish_canonical_report(output, report)
    except Exception as error:
        output_created = output.exists()
        _record_failed_attempt(
            args,
            execution,
            error,
            status=(
                "canonical_publish_failed_after_requested_output_creation"
                if output_created
                else "canonical_publish_failed_without_consuming_requested_output"
            ),
            diagnostic_report=report,
        )
        return 3
    print(
        json.dumps(
            {
                "output": str(output),
                "adapter_runtime_passed": True,
                "sample_count": report["adapter_runtime"]["sample_count"],
                "cleanup_warning": cleanup_warning,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
