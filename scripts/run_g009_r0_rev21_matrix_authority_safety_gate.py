#!/usr/bin/env python3
"""Run the static, fail-closed G009-5-E014 matrix-authority safety gate.

This command never imports Isaac, launches the simulator, computes rewards, or
runs PPO.  It validates the immutable rev20 evidence chain and writes one
canonical PASS artifact with exclusive-create semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
sys.dont_write_bytecode = True
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import summarize_g009_r0_rev21_matrix_authority_safety_gate as evaluator


PREREGISTRATION_PATH = (
    REPO_ROOT / "configs/g009_r0_rev21_matrix_authority_safety_gate.json"
)
RUNS_DIR = REPO_ROOT / "reports/runs"
SYNTHESIS_PATH = (
    RUNS_DIR / "g009_r0_rev20_terrain_contact_matrix_synthesis_2x2_s42.json"
)
OUTPUT_PATH = RUNS_DIR / "g009_r0_rev21_matrix_authority_safety_gate_s42.json"
FAILURE_ROOT = (
    Path.home()
    / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic/failed_attempts/rev21"
)
PASS_REASON = "matrix_authority_safety_gate_passed_for_diagnostic_preregistration"
FAILURE_SCHEMA = "g009.r0.rev21.matrix_authority_safety_gate_failure.v1"


class GateReject(Exception):
    """A deterministic contract rejection (process exit 2)."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        checks: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.checks = [dict(item) for item in (checks or [])]


class OperationalFailure(Exception):
    """An unexpected local I/O or subprocess failure (process exit 3)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class ExclusiveWriteFailure(OSError):
    """A staged exclusive-writer failure that preserves ownership context."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _new_execution() -> dict[str, Any]:
    try:
        relative = OUTPUT_PATH.absolute().relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError:
        relative = str(OUTPUT_PATH)
    return {
        "execution_id": uuid.uuid4().hex,
        "started_at_utc": _utc_now(),
        "output_path_repo_relative": relative,
        "no_overwrite": True,
    }


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _path_has_reparse_below_root(root: Path, candidate: Path) -> bool:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return True
    current = root
    if _is_reparse_point(current):
        return True
    for part in relative.parts:
        current = current / part
        if _is_reparse_point(current):
            return True
        if not current.exists():
            break
    return False


def _canonical_repo_path(relative: str, *, must_exist: bool, reason: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise GateReject(reason, "canonical path must be a non-empty POSIX path")
    lexical_root = REPO_ROOT.absolute()
    lexical = (REPO_ROOT / Path(relative)).absolute()
    try:
        lexical.relative_to(lexical_root)
    except ValueError as exc:
        raise GateReject(reason, f"canonical path escaped repository: {relative}") from exc
    if _path_has_reparse_below_root(lexical_root, lexical):
        raise GateReject(reason, f"canonical path crosses a reparse point: {relative}")
    try:
        resolved_root = REPO_ROOT.resolve(strict=True)
        resolved = lexical.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise GateReject(reason, f"canonical path is missing: {relative}") from exc
    except PermissionError as exc:
        raise OperationalFailure(
            "runner_input_io_error", f"cannot resolve canonical path: {relative}: {exc}"
        ) from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise GateReject(reason, f"resolved path escaped repository: {relative}") from exc
    return resolved


def _validate_output_path(preregistration: Mapping[str, Any]) -> Path:
    try:
        relative = preregistration["output_contract"]["canonical_path"]
    except (KeyError, TypeError) as exc:
        raise GateReject(
            "rev21_preregistration_invalid", "output contract is missing"
        ) from exc
    output = _canonical_repo_path(
        relative, must_exist=False, reason="canonical_output_path_invalid"
    )
    try:
        runs_root = RUNS_DIR.resolve(strict=True)
    except OSError as exc:
        raise OperationalFailure(
            "runner_input_io_error", f"cannot resolve reports/runs: {exc}"
        ) from exc
    if output.parent != runs_root or output.suffix != ".json" or output.name == ".json":
        raise GateReject(
            "canonical_output_path_invalid",
            "output must be the direct canonical reports/runs JSON",
        )
    if output != OUTPUT_PATH.resolve(strict=False):
        raise GateReject(
            "canonical_output_path_invalid", "configured output differs from runner constant"
        )
    if output.exists():
        raise GateReject(
            "canonical_output_already_exists", f"canonical output already exists: {output}"
        )
    return output


def _run_git_bytes(args: list[str], *, missing_reason: str) -> bytes:
    git_environment = os.environ.copy()
    git_environment["LC_ALL"] = "C"
    git_environment["LANG"] = "C"
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            env=git_environment,
        )
    except OSError as exc:
        raise OperationalFailure(
            "runner_input_io_error", f"cannot execute git {' '.join(args)}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        lowered = detail.lower()
        command = args[0] if args else ""
        expected_missing = (
            command == "show"
            and any(
                marker in lowered
                for marker in (
                    "does not exist in",
                    "invalid object name",
                    "bad object",
                    "unknown revision or path not in the working tree",
                )
            )
        ) or (
            command == "rev-parse"
            and any(
                marker in lowered
                for marker in (
                    "ambiguous argument 'head'",
                    "needed a single revision",
                )
            )
        )
        message = detail or f"git {' '.join(args)} failed with {completed.returncode}"
        if expected_missing:
            raise GateReject(missing_reason, message)
        raise OperationalFailure("runner_input_io_error", message)
    return completed.stdout


def _git_commit() -> str:
    raw = _run_git_bytes(["rev-parse", "HEAD"], missing_reason="rev21_source_provenance_invalid")
    commit = raw.decode("ascii", errors="strict").strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise GateReject(
            "rev21_source_provenance_invalid", "HEAD is not a lowercase 40-hex commit"
        )
    return commit


def _git_blob(commit: str, relative: str, *, missing_reason: str) -> bytes:
    return _run_git_bytes(["show", f"{commit}:{relative}"], missing_reason=missing_reason)


def source_bundle_provenance(
    preregistration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the exact committed rev21 config/evaluator/runner source paths."""

    if preregistration is None:
        try:
            raw = PREREGISTRATION_PATH.read_bytes()
            preregistration = evaluator.validate_preregistration_bytes(raw)
        except OSError as exc:
            raise GateReject("rev21_preregistration_invalid", str(exc)) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise GateReject("rev21_preregistration_invalid", str(exc)) from exc
    try:
        paths = list(preregistration["rev21_source_binding"]["ordered_paths"])
    except (KeyError, TypeError) as exc:
        raise GateReject(
            "rev21_preregistration_invalid", "rev21 source path list is missing"
        ) from exc
    if paths != list(evaluator.REQUIRED_SOURCE_PATHS):
        raise GateReject(
            "rev21_source_provenance_invalid", "rev21 ordered source paths mismatch"
        )
    commit = _git_commit()
    status_raw = _run_git_bytes(
        ["status", "--porcelain=v1", "--untracked-files=all", "--", *paths],
        missing_reason="rev21_source_provenance_invalid",
    )
    dirty = status_raw.decode("utf-8", errors="replace").splitlines()
    if dirty:
        raise GateReject(
            "rev21_source_provenance_invalid",
            "rev21 source paths must be committed and path-scoped clean: "
            + "; ".join(dirty),
        )
    files: dict[str, str] = {}
    for relative in paths:
        path = _canonical_repo_path(
            relative, must_exist=True, reason="rev21_source_provenance_invalid"
        )
        try:
            worktree = path.read_bytes()
        except PermissionError as exc:
            raise OperationalFailure(
                "runner_input_io_error", f"cannot read rev21 source {relative}: {exc}"
            ) from exc
        blob = _git_blob(
            commit, relative, missing_reason="rev21_source_provenance_invalid"
        )
        if worktree != blob:
            raise GateReject(
                "rev21_source_provenance_invalid",
                f"worktree bytes differ from HEAD blob: {relative}",
            )
        files[relative] = sha256_bytes(blob)
    payload = "\n".join(f"{path}:{files[path]}" for path in paths).encode("utf-8")
    return {
        "schema_version": 1,
        "git_commit": commit,
        "source_binding_paths": paths,
        "source_binding_files": files,
        "source_bundle_sha256": sha256_bytes(payload),
        "path_scoped_clean": True,
    }


def _read_bytes(path: Path, *, missing_reason: str) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise GateReject(missing_reason, f"missing canonical evidence: {path}") from exc
    except PermissionError as exc:
        raise OperationalFailure(
            "runner_input_io_error", f"cannot read canonical evidence {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise OperationalFailure(
            "runner_input_io_error", f"cannot read canonical evidence {path}: {exc}"
        ) from exc


def _load_preregistration() -> tuple[dict[str, Any], bytes]:
    canonical = _canonical_repo_path(
        "configs/g009_r0_rev21_matrix_authority_safety_gate.json",
        must_exist=True,
        reason="rev21_preregistration_invalid",
    )
    try:
        configured = PREREGISTRATION_PATH.resolve(strict=True)
    except OSError as exc:
        raise GateReject("rev21_preregistration_invalid", str(exc)) from exc
    if configured != canonical:
        raise GateReject(
            "rev21_preregistration_invalid",
            "runner preregistration path differs from the canonical config",
        )
    raw = _read_bytes(canonical, missing_reason="rev21_preregistration_invalid")
    try:
        value = evaluator.validate_preregistration_bytes(raw)
    except evaluator.GateValidationError as exc:
        raise GateReject("rev21_preregistration_invalid", str(exc)) from exc
    return value, raw


def _load_historical_blobs(
    preregistration: Mapping[str, Any],
) -> dict[str, bytes]:
    try:
        binding = preregistration["historical_source_binding"]
        commit = binding["commit"]
        paths = list(binding["ordered_unique_paths"])
    except (KeyError, TypeError) as exc:
        raise GateReject(
            "rev21_preregistration_invalid", "historical source binding is missing"
        ) from exc
    if commit != preregistration["predecessor"]["historical_source_commit"]:
        raise GateReject(
            "rev21_preregistration_invalid", "historical commit bindings disagree"
        )
    if paths != list(evaluator.HISTORICAL_SOURCE_PATHS):
        raise GateReject(
            "rev21_preregistration_invalid", "historical source path order mismatch"
        )
    return {
        relative: _git_blob(
            commit, relative, missing_reason="rev20_source_provenance_invalid"
        )
        for relative in paths
    }


def _load_evidence(
    preregistration: Mapping[str, Any],
) -> tuple[bytes, dict[str, bytes], bytes, dict[str, bytes]]:
    predecessor = preregistration["predecessor"]
    synthesis_path = _canonical_repo_path(
        predecessor["path"],
        must_exist=True,
        reason="rev20_synthesis_missing_or_path_invalid",
    )
    if synthesis_path != SYNTHESIS_PATH.resolve(strict=True):
        raise GateReject(
            "rev20_synthesis_missing_or_path_invalid",
            "configured predecessor differs from canonical synthesis",
        )
    synthesis_bytes = _read_bytes(
        synthesis_path, missing_reason="rev20_synthesis_missing_or_path_invalid"
    )
    if sha256_bytes(synthesis_bytes) != predecessor["sha256"]:
        raise GateReject(
            "rev20_synthesis_sha256_mismatch", "rev20 synthesis SHA-256 mismatch"
        )
    raw_by_path: dict[str, bytes] = {}
    for relative in preregistration["evidence_chain"]["ordered_raw_report_paths"]:
        path = _canonical_repo_path(
            relative, must_exist=True, reason="rev20_evidence_chain_binding_invalid"
        )
        raw_by_path[relative] = _read_bytes(
            path, missing_reason="rev20_evidence_chain_binding_invalid"
        )
    preflight_relative = predecessor["cpu_preflight_path"]
    preflight_path = _canonical_repo_path(
        preflight_relative,
        must_exist=True,
        reason="rev20_evidence_chain_binding_invalid",
    )
    preflight_bytes = _read_bytes(
        preflight_path, missing_reason="rev20_evidence_chain_binding_invalid"
    )
    if sha256_bytes(preflight_bytes) != predecessor["cpu_preflight_sha256"]:
        raise GateReject(
            "rev20_evidence_chain_binding_invalid", "CPU preflight SHA-256 mismatch"
        )
    historical_blobs = _load_historical_blobs(preregistration)
    return synthesis_bytes, raw_by_path, preflight_bytes, historical_blobs


def _evaluation_failure(result: Mapping[str, Any]) -> GateReject | None:
    decision = result.get("decision")
    if not isinstance(decision, Mapping):
        raise OperationalFailure(
            "runner_internal_error", "evaluator decision is missing or malformed"
        )
    passed = decision.get("passed")
    outcome = decision.get("outcome")
    primary = decision.get("primary_reason")
    next_step = decision.get("next_step")
    if passed is True:
        if not (
            outcome == primary == PASS_REASON
            and next_step == "preregister_read_only_matrix_observation_adapter"
        ):
            raise OperationalFailure(
                "runner_internal_error", "evaluator PASS decision is contradictory"
            )
        try:
            evaluator.validate_complete_reason_ledger(
                result.get("checks"), PASS_REASON
            )
        except evaluator.GateValidationError as exc:
            raise OperationalFailure("runner_internal_error", str(exc)) from exc
        return None
    if passed is not False or not isinstance(primary, str) or primary == PASS_REASON:
        raise OperationalFailure(
            "runner_internal_error", "evaluator rejection decision is contradictory"
        )
    if outcome != primary or primary not in evaluator.REASON_PRIORITY[:-1]:
        raise OperationalFailure(
            "runner_internal_error", "evaluator rejection reason is invalid"
        )
    reason = primary
    checks = result.get("checks")
    try:
        completed_checks = evaluator.validate_complete_reason_ledger(checks, reason)
    except evaluator.GateValidationError as exc:
        raise OperationalFailure("runner_internal_error", str(exc)) from exc
    return GateReject(
        reason,
        str(decision.get("message") or f"evidence evaluation rejected: {reason}"),
        checks=completed_checks,
    )


def _artifact(
    projection: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    decision = projection.get("decision")
    if not (
        isinstance(decision, Mapping)
        and decision.get("passed") is True
        and decision.get("outcome") == PASS_REASON
        and decision.get("primary_reason") == PASS_REASON
        and decision.get("next_step")
        == "preregister_read_only_matrix_observation_adapter"
    ):
        raise OperationalFailure(
            "runner_internal_error", "cannot build a PASS artifact from a contradictory projection"
        )
    checks = projection.get("checks")
    try:
        completed_checks = evaluator.validate_complete_reason_ledger(checks, PASS_REASON)
    except evaluator.GateValidationError as exc:
        raise OperationalFailure("runner_internal_error", str(exc)) from exc
    value = dict(projection)
    value["checks"] = completed_checks
    value["rev21_source_binding"] = dict(source_binding)
    value["execution"] = dict(execution)
    return value


def _owned_final_matches(temporary: Path, final: Path, payload: bytes) -> bool:
    try:
        return os.path.samefile(temporary, final) and final.read_bytes() == payload
    except OSError:
        return False


def write_json_exclusive_owned(
    path: Path,
    value: Mapping[str, Any] | bytes,
    *,
    link: Callable[[os.PathLike[str] | str, os.PathLike[str] | str], Any] = os.link,
) -> None:
    """Install complete bytes without overwriting or deleting another writer's final."""

    payload = (
        value
        if isinstance(value, bytes)
        else (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExclusiveWriteFailure("temp", f"cannot prepare output parent: {exc}") from exc
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    installed_by_this_execution = False
    try:
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise ExclusiveWriteFailure("temp", f"temporary write failed: {exc}") from exc
        try:
            link(temporary, path)
        except FileExistsError:
            raise
        except OSError as exc:
            raise ExclusiveWriteFailure("final", f"exclusive final install failed: {exc}") from exc
        installed_by_this_execution = True
        if not _owned_final_matches(temporary, path, payload):
            raise ExclusiveWriteFailure(
                "final", "exclusive final install identity or payload mismatch"
            )
    except Exception:
        if installed_by_this_execution and _owned_final_matches(temporary, path, payload):
            path.unlink(missing_ok=True)
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # A fully installed final is valid; never roll it back for temp cleanup.
            pass


def run(
    *,
    check_only: bool = False,
    _execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    execution = dict(_execution or _new_execution())
    preregistration, preregistration_bytes = _load_preregistration()
    output = _validate_output_path(preregistration)
    source_binding = source_bundle_provenance(preregistration)
    synthesis_bytes, raw_by_path, preflight_bytes, historical_blobs = _load_evidence(
        preregistration
    )
    projection = evaluator.evaluate_evidence(
        preregistration_bytes,
        synthesis_bytes,
        raw_by_path,
        preflight_bytes,
        historical_blobs,
    )
    rejection = _evaluation_failure(projection)
    if rejection is not None:
        raise rejection
    artifact = _artifact(projection, source_binding, execution)
    if check_only:
        return artifact
    try:
        write_json_exclusive_owned(output, artifact)
    except FileExistsError as exc:
        raise GateReject("canonical_output_already_exists", str(exc)) from exc
    except ExclusiveWriteFailure as exc:
        reason = (
            "runner_temp_write_failed"
            if exc.stage == "temp"
            else "runner_final_install_failed"
        )
        raise OperationalFailure(reason, str(exc)) from exc
    except (PermissionError, OSError) as exc:
        raise OperationalFailure("runner_final_install_failed", str(exc)) from exc
    return artifact


def _failure_path(execution: Mapping[str, Any]) -> Path:
    execution_id = execution.get("execution_id")
    if not isinstance(execution_id, str) or re.fullmatch(r"[0-9a-f]{32}", execution_id) is None:
        raise OperationalFailure(
            "runner_failure_envelope_write_failed", "invalid failure execution_id"
        )
    return FAILURE_ROOT / (
        "g009_5_e014_rev21_matrix_authority_safety_gate_"
        f"{execution_id}.json"
    )


def _failure_envelope(
    error: GateReject | OperationalFailure,
    execution: Mapping[str, Any],
    exit_code: int,
) -> dict[str, Any]:
    return {
        "schema_version": FAILURE_SCHEMA,
        "evidence_id": "G009-5-E014",
        "status": "rejected" if exit_code == 2 else "operational_error",
        "primary_reason": error.reason,
        "exit_code": exit_code,
        "execution": dict(execution),
        "checks": error.checks if isinstance(error, GateReject) else {},
        "error": {"type": type(error).__name__, "message": str(error)},
        "governance": {
            "diagnostic_only": True,
            "learned": False,
            "reward_computed": False,
            "ppo_updates": 0,
            "qualification_eligible": False,
            "qualification_status": "not_run",
            "physics_ground_truth_authority": False,
        },
    }


def _complete_gate_rejection_checks(error: GateReject) -> None:
    if error.reason not in evaluator.REASON_PRIORITY:
        return
    try:
        error.checks = evaluator.complete_reason_ledger(error.checks, error.reason)
    except evaluator.GateValidationError as exc:
        raise OperationalFailure("runner_internal_error", str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run the full gate without writing canonical, temp, or failure artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execution = _new_execution()
    try:
        artifact = run(check_only=args.check_only, _execution=execution)
    except GateReject as exc:
        _complete_gate_rejection_checks(exc)
        if not args.check_only:
            try:
                write_json_exclusive_owned(
                    _failure_path(execution), _failure_envelope(exc, execution, 2)
                )
            except Exception as envelope_error:
                print(
                    json.dumps(
                        {
                            "status": "operational_error",
                            "primary_reason": "runner_failure_envelope_write_failed",
                            "original_reason": exc.reason,
                            "message": str(envelope_error),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                return 3
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "primary_reason": exc.reason,
                    "message": str(exc),
                    "check_only": bool(args.check_only),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    except OperationalFailure as exc:
        if not args.check_only:
            try:
                write_json_exclusive_owned(
                    _failure_path(execution), _failure_envelope(exc, execution, 3)
                )
            except Exception as envelope_error:
                print(
                    json.dumps(
                        {
                            "status": "operational_error",
                            "primary_reason": "runner_failure_envelope_write_failed",
                            "original_reason": exc.reason,
                            "message": str(envelope_error),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                return 3
        print(
            json.dumps(
                {
                    "status": "operational_error",
                    "primary_reason": exc.reason,
                    "message": str(exc),
                    "check_only": bool(args.check_only),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 3
    except Exception as exc:
        wrapped = OperationalFailure("runner_internal_error", str(exc))
        if not args.check_only:
            try:
                write_json_exclusive_owned(
                    _failure_path(execution), _failure_envelope(wrapped, execution, 3)
                )
            except Exception as envelope_error:
                print(
                    json.dumps(
                        {
                            "status": "operational_error",
                            "primary_reason": "runner_failure_envelope_write_failed",
                            "original_reason": wrapped.reason,
                            "message": str(envelope_error),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                return 3
        print(
            json.dumps(
                {
                    "status": "operational_error",
                    "primary_reason": wrapped.reason,
                    "message": str(exc),
                    "check_only": bool(args.check_only),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 3
    print(
        json.dumps(
            {
                "status": "passed",
                "outcome": artifact.get("decision", {}).get("outcome"),
                "check_only": bool(args.check_only),
                "output": str(OUTPUT_PATH),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
