#!/usr/bin/env python3
"""Run the E015 static preregistration gate with fail-closed no-overwrite I/O."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import summarize_g009_r0_rev22_read_only_matrix_observation_adapter as evaluator


PREREGISTRATION_PATH = REPO_ROOT / "configs/g009_r0_rev22_read_only_matrix_observation_adapter.json"
PREDECESSOR_PATH = REPO_ROOT / "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json"
OUTPUT_PATH = REPO_ROOT / "reports/runs/g009_r0_rev22_read_only_matrix_observation_adapter_preregistration_s42.json"
FAILURE_ROOT = REPO_ROOT.parent / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic/failed_attempts/rev22"
FAILURE_SCHEMA = "g009.r0.rev22.read_only_matrix_observation_adapter_preregistration_failure.v1"
PASS_REASON = evaluator.PASS_REASON


class GateReject(RuntimeError):
    def __init__(self, reason: str, message: str, *, checks: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.reason = reason
        self.checks = list(checks or [])


class OperationalFailure(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class ExclusiveWriteFailure(OSError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def sha256_bytes(value: bytes) -> str:
    return evaluator.sha256_bytes(value)


def _new_execution() -> dict[str, Any]:
    return {
        "execution_id": uuid.uuid4().hex,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "output_path_repo_relative": OUTPUT_PATH.relative_to(REPO_ROOT).as_posix(),
        "no_overwrite": True,
    }


def _canonical_repo_path(relative: str, *, must_exist: bool, reason: str) -> Path:
    if not isinstance(relative, str) or not relative or "*" in relative or "?" in relative:
        raise GateReject(reason, "path must be one fixed repository-relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise GateReject(reason, f"unsafe repository path: {relative}")
    try:
        root = REPO_ROOT.resolve(strict=True)
        cursor = root
        for part in candidate.parts:
            cursor = cursor / part
            if not cursor.exists():
                break
            attributes = getattr(cursor.lstat(), "st_file_attributes", 0)
            if cursor.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise GateReject(reason, f"symlink/reparse traversal is forbidden: {relative}")
        resolved = (root / candidate).resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise GateReject(reason, f"missing canonical path: {relative}") from exc
    except OSError as exc:
        raise OperationalFailure("runner_input_io_error", str(exc)) from exc
    if not resolved.is_relative_to(root):
        raise GateReject(reason, f"path escaped repository: {relative}")
    return resolved


def _git_bytes(args: list[str], *, missing_reason: str) -> bytes:
    try:
        git_env = dict(os.environ)
        git_env.update({"LC_ALL": "C", "LANG": "C"})
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=REPO_ROOT, check=True, capture_output=True, env=git_env).stdout
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace")
        missing_markers = ("does not exist", "not in the working tree", "invalid object name", "bad object", "unknown revision", "ambiguous argument")
        if any(marker in detail.lower() for marker in missing_markers):
            raise GateReject(missing_reason, detail) from exc
        raise OperationalFailure("runner_input_io_error", detail or str(exc)) from exc
    except OSError as exc:
        raise OperationalFailure("runner_input_io_error", str(exc)) from exc


def _git_commit() -> str:
    commit = _git_bytes(["rev-parse", "HEAD"], missing_reason="rev22_source_provenance_invalid").decode().strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise GateReject("rev22_source_provenance_invalid", "HEAD commit is malformed")
    return commit


def _git_blob(commit: str, relative: str) -> bytes:
    return _git_bytes(["show", f"{commit}:{relative}"], missing_reason="rev22_source_provenance_invalid")


def source_bundle_provenance(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    try:
        paths = list(preregistration["rev22_source_binding"]["ordered_paths"])
    except (KeyError, TypeError) as exc:
        raise GateReject("rev22_preregistration_invalid", "rev22 source path list missing") from exc
    if paths != list(evaluator.REQUIRED_SOURCE_PATHS):
        raise GateReject("rev22_source_provenance_invalid", "rev22 source path order mismatch")
    commit = _git_commit()
    dirty = _git_bytes(["status", "--porcelain=v1", "--untracked-files=all", "--", *paths], missing_reason="rev22_source_provenance_invalid").decode("utf-8", errors="replace").splitlines()
    if dirty:
        raise GateReject("rev22_source_provenance_invalid", "rev22 source paths must be committed and clean: " + "; ".join(dirty))
    files: dict[str, str] = {}
    for relative in paths:
        path = _canonical_repo_path(relative, must_exist=True, reason="rev22_source_provenance_invalid")
        try:
            worktree = path.read_bytes()
        except OSError as exc:
            raise OperationalFailure("runner_input_io_error", str(exc)) from exc
        blob = _git_blob(commit, relative)
        if worktree != blob:
            raise GateReject("rev22_source_provenance_invalid", f"worktree differs from HEAD blob: {relative}")
        files[relative] = sha256_bytes(blob)
    payload = "\n".join(f"{path}:{files[path]}" for path in paths)
    return {"schema_version": 1, "git_commit": commit, "source_binding_paths": paths, "source_binding_files": files, "source_bundle_sha256": sha256_bytes(payload.encode("utf-8")), "path_scoped_clean": True}


def _read(path: Path, *, reason: str) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise GateReject(reason, f"missing canonical input: {path}") from exc
    except OSError as exc:
        raise OperationalFailure("runner_input_io_error", str(exc)) from exc


def _load_preregistration() -> tuple[dict[str, Any], bytes]:
    canonical = _canonical_repo_path("configs/g009_r0_rev22_read_only_matrix_observation_adapter.json", must_exist=True, reason="rev22_preregistration_invalid")
    try:
        configured = PREREGISTRATION_PATH.resolve(strict=True)
    except OSError as exc:
        raise GateReject("rev22_preregistration_invalid", str(exc)) from exc
    if configured != canonical:
        raise GateReject("rev22_preregistration_invalid", "configured preregistration is not canonical")
    raw = _read(canonical, reason="rev22_preregistration_invalid")
    try:
        return evaluator.validate_preregistration_bytes(raw), raw
    except evaluator.GateValidationError as exc:
        raise GateReject(exc.reason, exc.detail) from exc


def _validate_output(preregistration: Mapping[str, Any]) -> Path:
    relative = preregistration["output_contract"]["canonical_path"]
    output = _canonical_repo_path(relative, must_exist=False, reason="canonical_output_path_invalid")
    if output != OUTPUT_PATH.resolve() or output.parent != (REPO_ROOT / "reports/runs").resolve():
        raise GateReject("canonical_output_path_invalid", "output path is not canonical direct reports/runs child")
    if output.exists():
        raise GateReject("canonical_output_already_exists", "canonical output already exists")
    return output


def _load_predecessor(preregistration: Mapping[str, Any]) -> bytes:
    relative = preregistration["predecessor"]["path"]
    path = _canonical_repo_path(relative, must_exist=True, reason="rev21_predecessor_missing_or_path_invalid")
    if path != PREDECESSOR_PATH.resolve(strict=True):
        raise GateReject("rev21_predecessor_missing_or_path_invalid", "predecessor path differs from canonical")
    raw = _read(path, reason="rev21_predecessor_missing_or_path_invalid")
    if sha256_bytes(raw) != preregistration["predecessor"]["sha256"]:
        raise GateReject("rev21_predecessor_sha256_mismatch", "predecessor SHA-256 mismatch")
    try:
        evaluator.verify_predecessor_fresh(path)
    except evaluator.GateValidationError as exc:
        raise GateReject("rev21_predecessor_full_verification_failed", exc.detail) from exc
    return raw


def _evaluation_failure(projection: Mapping[str, Any]) -> GateReject | None:
    decision = projection.get("decision")
    if not isinstance(decision, Mapping):
        raise OperationalFailure("runner_internal_error", "evaluator decision missing")
    primary = decision.get("primary_reason")
    passed = decision.get("passed")
    if passed is True:
        if decision != {"passed": True, "outcome": PASS_REASON, "primary_reason": PASS_REASON, "next_step": evaluator.NEXT_STEP}:
            raise OperationalFailure("runner_internal_error", "contradictory evaluator PASS")
        evaluator.validate_complete_reason_ledger(projection.get("checks"), PASS_REASON)
        return None
    if passed is not False or not isinstance(primary, str) or primary not in evaluator.REASON_PRIORITY[:-1] or decision.get("outcome") != primary:
        raise OperationalFailure("runner_internal_error", "contradictory evaluator rejection")
    checks = evaluator.validate_complete_reason_ledger(projection.get("checks"), primary)
    return GateReject(primary, f"evidence rejected: {primary}", checks=checks)


def _artifact(projection: Mapping[str, Any], source: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    rejection = _evaluation_failure(projection)
    if rejection is not None:
        raise rejection
    value = dict(projection)
    value["rev22_source_binding"] = dict(source)
    value["execution"] = dict(execution)
    evaluator.validate_artifact_value(value, projection, source)
    return value


def _owned_final_matches(temporary: Path, final: Path, payload: bytes) -> bool:
    try:
        return os.path.samefile(temporary, final) and final.read_bytes() == payload
    except OSError:
        return False


def write_json_exclusive_owned(path: Path, value: Mapping[str, Any] | bytes, *, link: Callable[[os.PathLike[str] | str, os.PathLike[str] | str], Any] = os.link) -> None:
    payload = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExclusiveWriteFailure("temp", str(exc)) from exc
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    installed = False
    try:
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise ExclusiveWriteFailure("temp", str(exc)) from exc
        try:
            link(temporary, path)
        except FileExistsError:
            raise
        except OSError as exc:
            raise ExclusiveWriteFailure("final", str(exc)) from exc
        installed = True
        if not _owned_final_matches(temporary, path, payload):
            raise ExclusiveWriteFailure("final", "installed final identity/payload mismatch")
    except Exception:
        if installed and _owned_final_matches(temporary, path, payload):
            path.unlink(missing_ok=True)
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def run(*, check_only: bool = False, _execution: Mapping[str, Any] | None = None) -> dict[str, Any]:
    execution = dict(_execution or _new_execution())
    preregistration, preregistration_raw = _load_preregistration()
    output = _validate_output(preregistration)
    source = source_bundle_provenance(preregistration)
    predecessor_raw = _load_predecessor(preregistration)
    projection = evaluator.evaluate_evidence(preregistration_raw, predecessor_raw, source)
    rejection = _evaluation_failure(projection)
    if rejection is not None:
        raise rejection
    artifact = _artifact(projection, source, execution)
    if check_only:
        return artifact
    try:
        write_json_exclusive_owned(output, artifact)
    except FileExistsError as exc:
        raise GateReject("canonical_output_already_exists", str(exc)) from exc
    except ExclusiveWriteFailure as exc:
        reason = "runner_temp_write_failed" if exc.stage == "temp" else "runner_final_install_failed"
        raise OperationalFailure(reason, str(exc)) from exc
    except (PermissionError, OSError) as exc:
        raise OperationalFailure("runner_final_install_failed", str(exc)) from exc
    return artifact


def _failure_path(execution: Mapping[str, Any]) -> Path:
    execution_id = execution.get("execution_id")
    if not isinstance(execution_id, str) or re.fullmatch(r"[0-9a-f]{32}", execution_id) is None:
        raise OperationalFailure("runner_failure_envelope_write_failed", "invalid execution ID")
    return FAILURE_ROOT / f"g009_5_e015_rev22_read_only_matrix_observation_adapter_{execution_id}.json"


def _failure_envelope(error: GateReject | OperationalFailure, execution: Mapping[str, Any], exit_code: int) -> dict[str, Any]:
    checks = evaluator.complete_reason_ledger(error.checks, error.reason) if isinstance(error, GateReject) and error.reason in evaluator.REASON_PRIORITY else []
    return {"schema_version": FAILURE_SCHEMA, "evidence_id": "G009-5-E015", "status": "rejected" if exit_code == 2 else "operational_error", "primary_reason": error.reason, "exit_code": exit_code, "execution": dict(execution), "checks": checks, "error": {"type": type(error).__name__, "message": str(error)}, "governance": evaluator.GOVERNANCE}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    execution = _new_execution()
    try:
        artifact = run(check_only=args.check_only, _execution=execution)
    except GateReject as exc:
        if not args.check_only:
            try:
                write_json_exclusive_owned(_failure_path(execution), _failure_envelope(exc, execution, 2))
            except Exception as envelope_error:
                print(json.dumps({"status": "operational_error", "primary_reason": "runner_failure_envelope_write_failed", "original_reason": exc.reason, "message": str(envelope_error)}, ensure_ascii=False), file=sys.stderr)
                return 3
        print(json.dumps({"status": "rejected", "primary_reason": exc.reason, "check_only": bool(args.check_only), "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except OperationalFailure as exc:
        if not args.check_only:
            try:
                write_json_exclusive_owned(_failure_path(execution), _failure_envelope(exc, execution, 3))
            except Exception as envelope_error:
                print(json.dumps({"status": "operational_error", "primary_reason": "runner_failure_envelope_write_failed", "original_reason": exc.reason, "message": str(envelope_error)}, ensure_ascii=False), file=sys.stderr)
                return 3
        print(json.dumps({"status": "operational_error", "primary_reason": exc.reason, "check_only": bool(args.check_only), "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3
    except Exception as exc:
        wrapped = OperationalFailure("runner_internal_error", str(exc))
        if not args.check_only:
            try:
                write_json_exclusive_owned(_failure_path(execution), _failure_envelope(wrapped, execution, 3))
            except Exception as envelope_error:
                print(json.dumps({"status": "operational_error", "primary_reason": "runner_failure_envelope_write_failed", "original_reason": wrapped.reason, "message": str(envelope_error)}, ensure_ascii=False), file=sys.stderr)
                return 3
        print(json.dumps({"status": "operational_error", "primary_reason": wrapped.reason, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3
    print(json.dumps({"status": "passed", "outcome": artifact["decision"]["outcome"], "check_only": bool(args.check_only), "output": str(OUTPUT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
