#!/usr/bin/env python3
"""Fail-closed CPU preflight and final synthesis for G009-5-E016 rev23."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import probe_g009_r0_rev23_matrix_observation_adapter_runtime as probe


RUNS_DIR = REPO_ROOT / "reports/runs"
CPU_SCHEMA = "g009.r0.rev23.matrix_observation_adapter_cpu_preflight.v1"
FINAL_SCHEMA = "g009.r0.rev23.matrix_observation_adapter_runtime_synthesis.v1"
CPU_OUTPUT = RUNS_DIR / "g009_r0_rev23_matrix_observation_adapter_cpu_preflight_2x_s42.json"
FINAL_OUTPUT = RUNS_DIR / "g009_r0_rev23_matrix_observation_adapter_synthesis_2x2_s42.json"
CPU_PATHS = tuple(probe.EXPECTED_PATHS[("cpu", replicate)] for replicate in (1, 2))
FINAL_PATHS = CPU_PATHS + tuple(probe.EXPECTED_PATHS[("cuda:0", replicate)] for replicate in (1, 2))


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return probe.sha256_bytes(value)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    return probe.strict_json_bytes(raw, label), raw


def _binding(path: Path, raw: bytes) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    require(resolved.parent == RUNS_DIR.resolve(), "input must be a direct reports/runs JSON")
    return {"path": resolved.relative_to(REPO_ROOT.resolve()).as_posix(), "sha256": sha256_bytes(raw)}


def load_inputs(paths: Sequence[Path], expected_paths: Sequence[str]) -> list[tuple[dict[str, Any], dict[str, str]]]:
    require(len(paths) == len(expected_paths), f"exactly {len(expected_paths)} reports required")
    entries: list[tuple[dict[str, Any], dict[str, str]]] = []
    for index, path in enumerate(paths):
        value, raw = _read_json(path, f"rev23 input report {index + 1}")
        probe.validate_report(value)
        entries.append((value, _binding(path, raw)))
    bindings = [item for _, item in entries]
    require([item["path"] for item in bindings] == list(expected_paths), "canonical input path/order mismatch")
    require(len({item["path"] for item in bindings}) == len(bindings), "duplicate report path")
    require(len({item["sha256"] for item in bindings}) == len(bindings), "duplicate report SHA")
    return entries


def _repeatability_contract() -> tuple[tuple[str, ...], tuple[str, ...], float, float]:
    prereg = probe.load_preregistration()
    contract = cast(Mapping[str, Any], cast(Mapping[str, Any], prereg["gates"])["repeatability"])
    exact = tuple(cast(Sequence[str], contract["exact_fields"]))
    numeric = tuple(cast(Sequence[str], contract["numeric_fields"]))
    absolute = float(contract["absolute_tolerance"])
    relative = float(contract["relative_tolerance"])
    require(exact and numeric and absolute >= 0.0 and relative >= 0.0, "invalid repeatability contract")
    return exact, numeric, absolute, relative


def _numeric_close(left: Any, right: Any, absolute: float, relative: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool) or not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    left_value, right_value = float(left), float(right)
    return math.isfinite(left_value) and math.isfinite(right_value) and abs(left_value - right_value) <= max(
        absolute, relative * max(abs(left_value), abs(right_value))
    )


def repeatability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) == 2, "repeatability requires exactly two rows")
    exact_fields, numeric_fields, absolute, relative = _repeatability_contract()
    exact_matches = {field: rows[0][field] == rows[1][field] for field in exact_fields}
    numeric_matches = {
        field: _numeric_close(rows[0][field], rows[1][field], absolute, relative) for field in numeric_fields
    }
    return {
        "exact_fields": list(exact_fields),
        "numeric_fields": list(numeric_fields),
        "exact_field_matches": exact_matches,
        "numeric_field_matches": numeric_matches,
        "exact_fields_match": all(exact_matches.values()),
        "numeric_fields_within_tolerance": all(numeric_matches.values()),
        "repeatable": all(exact_matches.values()) and all(numeric_matches.values()),
        "absolute_tolerance": absolute,
        "relative_tolerance": relative,
    }


def _execution_id(report: Mapping[str, Any]) -> str:
    execution = cast(Mapping[str, Any], report.get("execution", {}))
    return probe.validate_uuid4_hex(execution.get("execution_id"))


def row(report: Mapping[str, Any], item_binding: Mapping[str, str]) -> dict[str, Any]:
    probe.validate_report(report)
    adapter = cast(Mapping[str, Any], report["adapter_runtime"])
    feasibility = cast(Mapping[str, Any], report["feasibility"])
    decision = cast(Mapping[str, Any], report["adapter_decision"])
    source = cast(Mapping[str, Any], report["rev23_source_bundle"])
    return {
        "slot": f"{report['device']}.rep{report['replicate_index']}",
        "device": report["device"],
        "replicate_index": report["replicate_index"],
        "binding": dict(item_binding),
        "execution_id": _execution_id(report),
        "git_commit": source["git_commit"],
        "source_bundle_sha256": source["source_bundle_sha256"],
        "source_bundle": dict(source),
        "cpu_preflight_binding": report["cpu_preflight_binding"],
        "runtime_parent_passed": feasibility.get("run_interpretable") is True,
        "adapter_decision_passed": decision.get("passed") is True
        and decision.get("outcome") == "read_only_matrix_observation_adapter_runtime_run_passed",
        "adapter_runtime_passed": adapter.get("passed") is True,
        **{field: adapter[field] for field in _repeatability_contract()[0] + _repeatability_contract()[1]},
    }


def _validate_unique(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len({item["binding"]["path"] for item in rows}) == len(rows), "duplicate report path")
    require(len({item["binding"]["sha256"] for item in rows}) == len(rows), "duplicate report SHA")
    require(len({item["execution_id"] for item in rows}) == len(rows), "duplicate execution_id")


def _validate_source_bundle(bundle: Mapping[str, Any]) -> None:
    paths = list(probe.SYNTHESIS_SOURCE_BINDING_PATHS)
    files = bundle.get("source_binding_files")
    require(
        bundle.get("schema_version") == 1
        and isinstance(bundle.get("git_commit"), str)
        and re.fullmatch(r"[0-9a-f]{40}", str(bundle["git_commit"])) is not None
        and bundle.get("source_binding_paths") == paths
        and isinstance(files, Mapping)
        and list(files) == paths
        and all(re.fullmatch(r"[0-9a-f]{64}", str(files[path])) is not None for path in paths)
        and bundle.get("path_scoped_clean") is True,
        "synthesis source bundle schema mismatch",
    )
    assert isinstance(files, Mapping)
    payload = "\n".join(f"{path}:{files[path]}" for path in paths)
    require(bundle.get("source_bundle_sha256") == sha256_bytes(payload.encode("utf-8")), "synthesis source bundle aggregate mismatch")


def synthesis_source_bundle_provenance() -> dict[str, Any]:
    bundle = probe.source_bundle_provenance(probe.SYNTHESIS_SOURCE_BINDING_PATHS)
    _validate_source_bundle(bundle)
    return bundle


def _new_execution(output: Path, forbidden_ids: set[str]) -> dict[str, Any]:
    execution_id = uuid.uuid4().hex
    require(execution_id not in forbidden_ids, "synthesis execution_id collides with input")
    return {
        "execution_id": execution_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "output_path_repo_relative": output.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "no_overwrite": True,
    }


def _validate_mode_output(mode: str, output: Path) -> Path:
    require(mode in {"cpu-preflight", "final"}, "invalid synthesis mode")
    expected = CPU_OUTPUT if mode == "cpu-preflight" else FINAL_OUTPUT
    resolved = output.resolve()
    require(
        resolved == expected.resolve(),
        f"{mode} output must be the exact canonical path: {expected.relative_to(REPO_ROOT).as_posix()}",
    )
    return resolved


def _validate_common_rows(rows: Sequence[Mapping[str, Any]], expected_slots: Sequence[str], source_bundle: Mapping[str, Any]) -> None:
    _validate_unique(rows)
    require([item["slot"] for item in rows] == list(expected_slots), "canonical slot order mismatch")
    require(all(item["runtime_parent_passed"] for item in rows), "runtime parent validation failed")
    require(all(item["adapter_runtime_passed"] and item["adapter_decision_passed"] for item in rows), "adapter runtime validation failed")
    require({item["git_commit"] for item in rows} == {source_bundle["git_commit"]}, "report/synthesis git commit drift")
    require(
        {item["source_bundle_sha256"] for item in rows} == {source_bundle["source_bundle_sha256"]},
        "report/synthesis source bundle drift",
    )
    require(all(item["source_bundle"] == source_bundle for item in rows), "report/synthesis source bundle contents drift")


def cpu_preflight(entries: Sequence[tuple[dict[str, Any], dict[str, str]]], output: Path = CPU_OUTPUT) -> dict[str, Any]:
    _validate_mode_output("cpu-preflight", output)
    require(len(entries) == 2, "CPU preflight requires exactly two reports")
    rows = [row(report, item_binding) for report, item_binding in entries]
    source_bundle = synthesis_source_bundle_provenance()
    _validate_common_rows(rows, ("cpu.rep1", "cpu.rep2"), source_bundle)
    repeated = repeatability(rows)
    require(repeated["repeatable"] is True, "CPU adapter runtime is not repeatable; GPU stage forbidden")
    ids = {item["execution_id"] for item in rows}
    return {
        "schema_version": CPU_SCHEMA,
        "evidence_id": "G009-5-E016",
        "status": "complete",
        "mode": "cpu_preflight_2x",
        "input_report_count": 2,
        "input_reports": [item["binding"] for item in rows],
        "integrity": {
            "passed": True,
            "hash_bound": True,
            "unique_report_paths": True,
            "unique_report_sha256": True,
            "unique_execution_ids": True,
            "exact_slots": [item["slot"] for item in rows],
            "git_commit": source_bundle["git_commit"],
            "probe_source_bundle_sha256": source_bundle["source_bundle_sha256"],
            "synthesis_source_bundle_sha256": source_bundle["source_bundle_sha256"],
        },
        "cpu_preflight": {
            "passed": True,
            "runtime_parent_passed": True,
            "adapter_150_of_150_passed": True,
            "within_cpu_repeatability_passed": True,
            "gpu_stage_allowed": True,
        },
        "repeatability": repeated,
        "decision": {"outcome": "gpu_stage_authorized", "next_step": "run_cuda_runtime_replicates_1_and_2", "third_run_allowed": False},
        "governance": probe.governance(),
        "claim_limits": probe.claim_limits(),
        "synthesis_source_bundle": source_bundle,
        "execution": _new_execution(output, ids),
    }


def _load_bound_reports(
    input_bindings: Any, expected_paths: Sequence[str], repo_root: Path
) -> list[tuple[dict[str, Any], dict[str, str]]]:
    require(isinstance(input_bindings, list) and len(input_bindings) == len(expected_paths), "input report binding count mismatch")
    require(
        all(isinstance(item, Mapping) and list(item) == ["path", "sha256"] for item in input_bindings),
        "input report binding schema mismatch",
    )
    require([item["path"] for item in input_bindings] == list(expected_paths), "input report binding path/order mismatch")
    result: list[tuple[dict[str, Any], dict[str, str]]] = []
    for index, item in enumerate(input_bindings):
        relative = str(item["path"])
        path = (repo_root / relative).resolve(strict=True)
        require(path.parent == (repo_root / "reports/runs").resolve(), "bound input must be a direct reports/runs JSON")
        value, raw = _read_json(path, f"bound rev23 report {index + 1}")
        require(sha256_bytes(raw) == item["sha256"], "bound input report SHA-256 mismatch")
        probe.validate_report(value)
        result.append((value, {"path": relative, "sha256": str(item["sha256"])}))
    return result


def validate_cpu_preflight_value(
    value: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
    expected_output_relative_path: str = CPU_OUTPUT.relative_to(REPO_ROOT).as_posix(),
    source_bundle: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    require(
        set(value)
        == {
            "schema_version",
            "evidence_id",
            "status",
            "mode",
            "input_report_count",
            "input_reports",
            "integrity",
            "cpu_preflight",
            "repeatability",
            "decision",
            "governance",
            "claim_limits",
            "synthesis_source_bundle",
            "execution",
        },
        "CPU preflight top-level schema mismatch",
    )
    require(
        value.get("schema_version") == CPU_SCHEMA
        and value.get("evidence_id") == "G009-5-E016"
        and value.get("status") == "complete"
        and value.get("mode") == "cpu_preflight_2x"
        and value.get("input_report_count") == 2,
        "CPU preflight identity mismatch",
    )
    entries = _load_bound_reports(value.get("input_reports"), CPU_PATHS, repo_root)
    rows = [row(report, binding) for report, binding in entries]
    stored_bundle = cast(Mapping[str, Any], value.get("synthesis_source_bundle", {}))
    _validate_source_bundle(stored_bundle)
    if source_bundle is not None:
        require(stored_bundle == source_bundle, "CPU preflight/current source bundle mismatch")
    _validate_common_rows(rows, ("cpu.rep1", "cpu.rep2"), stored_bundle)
    repeated = repeatability(rows)
    require(repeated["repeatable"] is True and value.get("repeatability") == repeated, "CPU repeatability mismatch")
    integrity = cast(Mapping[str, Any], value.get("integrity", {}))
    require(
        set(integrity)
        == {
            "passed",
            "hash_bound",
            "unique_report_paths",
            "unique_report_sha256",
            "unique_execution_ids",
            "exact_slots",
            "git_commit",
            "probe_source_bundle_sha256",
            "synthesis_source_bundle_sha256",
        }
        and integrity.get("passed") is True
        and integrity.get("hash_bound") is True
        and integrity.get("unique_report_paths") is True
        and integrity.get("unique_report_sha256") is True
        and integrity.get("unique_execution_ids") is True
        and integrity.get("exact_slots") == ["cpu.rep1", "cpu.rep2"]
        and integrity.get("git_commit") == stored_bundle["git_commit"]
        and integrity.get("probe_source_bundle_sha256") == stored_bundle["source_bundle_sha256"]
        and integrity.get("synthesis_source_bundle_sha256") == stored_bundle["source_bundle_sha256"],
        "CPU preflight integrity mismatch",
    )
    require(
        value.get("cpu_preflight")
        == {
            "passed": True,
            "runtime_parent_passed": True,
            "adapter_150_of_150_passed": True,
            "within_cpu_repeatability_passed": True,
            "gpu_stage_allowed": True,
        }
        and value.get("decision")
        == {"outcome": "gpu_stage_authorized", "next_step": "run_cuda_runtime_replicates_1_and_2", "third_run_allowed": False}
        and value.get("governance") == probe.governance()
        and value.get("claim_limits") == probe.claim_limits(),
        "CPU preflight decision/governance mismatch",
    )
    execution = cast(Mapping[str, Any], value.get("execution", {}))
    require(
        set(execution) == {"execution_id", "started_at_utc", "output_path_repo_relative", "no_overwrite"}
        and execution.get("output_path_repo_relative") == expected_output_relative_path
        and execution.get("no_overwrite") is True
        and isinstance(execution.get("started_at_utc"), str),
        "CPU preflight execution mismatch",
    )
    preflight_id = probe.validate_uuid4_hex(execution.get("execution_id"), "CPU preflight execution_id")
    require(preflight_id not in {item["execution_id"] for item in rows}, "CPU preflight/input execution_id collision")
    return [report for report, _ in entries]


def _read_preflight(path: Path, source_bundle: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    resolved = path.resolve(strict=True)
    require(resolved == CPU_OUTPUT.resolve(), "final synthesis requires canonical CPU preflight")
    value, raw = _read_json(resolved, "rev23 CPU preflight")
    validate_cpu_preflight_value(value, REPO_ROOT, CPU_OUTPUT.relative_to(REPO_ROOT).as_posix(), source_bundle)
    return value, raw


def final_synthesis(
    entries: Sequence[tuple[dict[str, Any], dict[str, str]]], preflight_path: Path, output: Path = FINAL_OUTPUT
) -> dict[str, Any]:
    _validate_mode_output("final", output)
    require(len(entries) == 4, "final synthesis requires exactly four reports")
    rows = [row(report, item_binding) for report, item_binding in entries]
    source_bundle = synthesis_source_bundle_provenance()
    _validate_common_rows(rows, ("cpu.rep1", "cpu.rep2", "cuda:0.rep1", "cuda:0.rep2"), source_bundle)
    preflight, preflight_raw = _read_preflight(preflight_path, source_bundle)
    require(preflight["input_reports"] == [item["binding"] for item in rows[:2]], "final CPU bindings differ from preflight")
    expected_binding = {
        "status": "validated_for_gpu",
        "path": CPU_OUTPUT.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_bytes(preflight_raw),
        "git_commit": source_bundle["git_commit"],
        "probe_source_bundle_sha256": source_bundle["source_bundle_sha256"],
        "input_reports": preflight["input_reports"],
    }
    require(
        rows[2]["cpu_preflight_binding"] == expected_binding and rows[3]["cpu_preflight_binding"] == expected_binding,
        "GPU reports do not exact-bind the same immutable CPU preflight",
    )
    all_ids = [item["execution_id"] for item in rows] + [preflight["execution"]["execution_id"]]
    require(len(set(all_ids)) == 5, "report/preflight execution_id collision")
    cpu_repeat, gpu_repeat = repeatability(rows[:2]), repeatability(rows[2:])
    require(cpu_repeat["repeatable"] is True and gpu_repeat["repeatable"] is True, "CPU/GPU adapter runtime is not repeatable")
    return {
        "schema_version": FINAL_SCHEMA,
        "evidence_id": "G009-5-E016",
        "status": "complete",
        "mode": "final_2x2",
        "input_report_count": 4,
        "input_reports": [item["binding"] for item in rows],
        "integrity": {
            "passed": True,
            "hash_bound": True,
            "unique_report_paths": True,
            "unique_report_sha256": True,
            "unique_execution_ids": True,
            "exact_slots": [item["slot"] for item in rows],
            "git_commit": source_bundle["git_commit"],
            "probe_source_bundle_sha256": source_bundle["source_bundle_sha256"],
            "synthesis_source_bundle_sha256": source_bundle["source_bundle_sha256"],
            "cpu_preflight": {"path": expected_binding["path"], "sha256": expected_binding["sha256"]},
        },
        "repeatability": {"cpu": cpu_repeat, "cuda:0": gpu_repeat},
        "rows": rows,
        "decision": {
            "outcome": "read_only_matrix_observation_adapter_runtime_2x2_validated",
            "next_step": "preregister_and_run_gpu_throughput_ladder_before_matrix_gate01",
            "third_run_allowed": False,
        },
        "governance": probe.governance(),
        "claim_limits": probe.claim_limits(),
        "synthesis_source_bundle": source_bundle,
        "execution": _new_execution(output, set(all_ids)),
    }


def validate_final_value(
    value: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
    expected_output_relative_path: str = FINAL_OUTPUT.relative_to(REPO_ROOT).as_posix(),
    source_bundle: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    require(
        set(value)
        == {
            "schema_version",
            "evidence_id",
            "status",
            "mode",
            "input_report_count",
            "input_reports",
            "integrity",
            "repeatability",
            "rows",
            "decision",
            "governance",
            "claim_limits",
            "synthesis_source_bundle",
            "execution",
        },
        "final synthesis top-level schema mismatch",
    )
    require(
        value.get("schema_version") == FINAL_SCHEMA
        and value.get("evidence_id") == "G009-5-E016"
        and value.get("status") == "complete"
        and value.get("mode") == "final_2x2"
        and value.get("input_report_count") == 4,
        "final synthesis identity mismatch",
    )
    entries = _load_bound_reports(value.get("input_reports"), FINAL_PATHS, repo_root)
    rows = [row(report, binding) for report, binding in entries]
    stored_bundle = cast(Mapping[str, Any], value.get("synthesis_source_bundle", {}))
    _validate_source_bundle(stored_bundle)
    if source_bundle is not None:
        require(stored_bundle == source_bundle, "final/current source bundle mismatch")
    _validate_common_rows(rows, ("cpu.rep1", "cpu.rep2", "cuda:0.rep1", "cuda:0.rep2"), stored_bundle)
    preflight_path = repo_root / CPU_OUTPUT.relative_to(REPO_ROOT)
    preflight, preflight_raw = _read_json(preflight_path, "bound rev23 CPU preflight")
    validate_cpu_preflight_value(preflight, repo_root, CPU_OUTPUT.relative_to(REPO_ROOT).as_posix(), stored_bundle)
    require(preflight["input_reports"] == [item["binding"] for item in rows[:2]], "final CPU bindings differ from preflight")
    expected_binding = {
        "status": "validated_for_gpu",
        "path": CPU_OUTPUT.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_bytes(preflight_raw),
        "git_commit": stored_bundle["git_commit"],
        "probe_source_bundle_sha256": stored_bundle["source_bundle_sha256"],
        "input_reports": preflight["input_reports"],
    }
    require(rows[2]["cpu_preflight_binding"] == expected_binding == rows[3]["cpu_preflight_binding"], "final GPU preflight binding mismatch")
    cpu_repeat, gpu_repeat = repeatability(rows[:2]), repeatability(rows[2:])
    require(value.get("repeatability") == {"cpu": cpu_repeat, "cuda:0": gpu_repeat}, "final repeatability mismatch")
    require(value.get("rows") == rows, "final rows differ from report recomputation")
    require(
        value.get("decision")
        == {
            "outcome": "read_only_matrix_observation_adapter_runtime_2x2_validated",
            "next_step": "preregister_and_run_gpu_throughput_ladder_before_matrix_gate01",
            "third_run_allowed": False,
        }
        and value.get("governance") == probe.governance()
        and value.get("claim_limits") == probe.claim_limits(),
        "final decision/governance mismatch",
    )
    integrity = cast(Mapping[str, Any], value.get("integrity", {}))
    require(
        set(integrity)
        == {
            "passed",
            "hash_bound",
            "unique_report_paths",
            "unique_report_sha256",
            "unique_execution_ids",
            "exact_slots",
            "git_commit",
            "probe_source_bundle_sha256",
            "synthesis_source_bundle_sha256",
            "cpu_preflight",
        }
        and integrity.get("passed") is True
        and integrity.get("hash_bound") is True
        and integrity.get("unique_report_paths") is True
        and integrity.get("unique_report_sha256") is True
        and integrity.get("unique_execution_ids") is True
        and integrity.get("exact_slots") == ["cpu.rep1", "cpu.rep2", "cuda:0.rep1", "cuda:0.rep2"]
        and integrity.get("git_commit") == stored_bundle["git_commit"]
        and integrity.get("probe_source_bundle_sha256") == stored_bundle["source_bundle_sha256"]
        and integrity.get("synthesis_source_bundle_sha256") == stored_bundle["source_bundle_sha256"]
        and integrity.get("cpu_preflight") == {"path": expected_binding["path"], "sha256": expected_binding["sha256"]},
        "final integrity mismatch",
    )
    execution = cast(Mapping[str, Any], value.get("execution", {}))
    require(
        set(execution) == {"execution_id", "started_at_utc", "output_path_repo_relative", "no_overwrite"}
        and execution.get("output_path_repo_relative") == expected_output_relative_path
        and execution.get("no_overwrite") is True
        and isinstance(execution.get("started_at_utc"), str),
        "final execution mismatch",
    )
    synthesis_id = probe.validate_uuid4_hex(execution.get("execution_id"), "final execution_id")
    ids = [item["execution_id"] for item in rows] + [preflight["execution"]["execution_id"], synthesis_id]
    require(len(set(ids)) == 6, "final/report/preflight execution_id collision")
    return [report for report, _ in entries]


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    require(
        resolved in {CPU_OUTPUT.resolve(), FINAL_OUTPUT.resolve()},
        "output must be an exact rev23 canonical synthesis path",
    )
    require(not resolved.exists(), "refusing to overwrite output")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("cpu-preflight", "final"))
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--cpu-preflight", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_mode_output(args.mode, args.output)
    if args.mode == "cpu-preflight":
        require(args.cpu_preflight is None, "CPU mode must not bind a preflight")
        value = cpu_preflight(load_inputs(args.inputs, CPU_PATHS), args.output)
    else:
        require(args.cpu_preflight is not None, "final mode requires --cpu-preflight")
        value = final_synthesis(load_inputs(args.inputs, FINAL_PATHS), args.cpu_preflight, args.output)
    write_json_exclusive(args.output, value)
    print(json.dumps({"output": str(args.output), "outcome": value["decision"]["outcome"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
