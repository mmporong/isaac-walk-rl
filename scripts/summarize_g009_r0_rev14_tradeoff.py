#!/usr/bin/env python3
"""Create a fail-closed rev14 CPU/GPU trade-off evidence synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "reports/runs"
DEFAULT_CPU = tuple(
    RUNS_DIR / f"g009_r0_runtime_probe_rev14_actualtopology_cpu_rep{i:02d}_s42.json"
    for i in range(1, 4)
)
DEFAULT_GPU = tuple(
    RUNS_DIR / f"g009_r0_runtime_probe_rev14_actualtopology_gpu_rep{i:02d}_s42.json"
    for i in range(1, 4)
)
DEFAULT_BASELINE_CPU = RUNS_DIR / "g009_r0_runtime_probe_rev12_cpu_rep01_s42.json"
DEFAULT_BASELINE_GPU = RUNS_DIR / "g009_r0_runtime_probe_rev12_gpu_rep01_s42.json"
DEFAULT_OUTPUT = (
    RUNS_DIR / "g009_r0_runtime_probe_rev14_tradeoff_synthesis_3x3_s42.json"
)

REV14_CONTRACT = "744c53d3c8d1e608f849af405c7d0fad314b01234fc4cb9a4ab1000c69140506"
REV12_CONTRACT = "d4b48d2b5fc1ea7684684a6324ba22fbfae767effeae45668c7310df382392e0"
REV14_SOURCE_COMMIT = "e9c1eff15bb2679c67e325546a749dbe7f98b07c"
REV14_SOURCE_BUNDLE = "5c3cfa41a9c6b61a5579ed48ed17eb4f0f363eeebb9f970b61eada09fca8bacc"
EXPECTED_BODY_NAMES = (
    "base",
    "FL_hip",
    "FR_hip",
    "Head_upper",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "Head_lower",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)
EXPECTED_PRIMARY_CPU = 8.50235366821289
EXPECTED_GLOBAL_CPU = 13.943856239318848
EXPECTED_GLOBAL_GPU = 12.610370635986328
EXPECTED_BASELINE_CPU_PRIMARY = 9.332860946655273
EXPECTED_WORST_SEPARATION = -0.010990187525749207
SEPARATION_THRESHOLD = -0.01


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.resolve(strict=True)
    require(
        resolved.parent == RUNS_DIR.resolve(), f"input must be in reports/runs: {path}"
    )
    raw = resolved.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value, {
        "path": f"reports/runs/{resolved.name}",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def finite_number(value: Any, label: str) -> float:
    require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def metric(report: dict[str, Any], pose: str, mode: str) -> dict[str, Any]:
    matches = [
        m
        for m in report.get("pose_mode_metrics", [])
        if m.get("pose_id") == pose and m.get("action_mode") == mode
    ]
    require(len(matches) == 1, f"exactly one {pose}/{mode} metric is required")
    return matches[0]


def termination_totals(report: dict[str, Any]) -> dict[str, int]:
    metrics = report.get("pose_mode_metrics")
    require(
        isinstance(metrics, list) and len(metrics) == 8,
        "eight pose metrics are required",
    )
    assert isinstance(metrics, list)
    totals = {"numeric_invalid": 0, "hard_joint_limit": 0}
    for item in metrics:
        counts = item.get("termination_counts") if isinstance(item, dict) else None
        require(isinstance(counts, dict), "termination_counts must be an object")
        assert isinstance(counts, dict)
        for name in totals:
            value = counts.get(name)
            require(
                type(value) is int and value >= 0, f"invalid {name} termination count"
            )
            assert isinstance(value, int)
            totals[name] += value
    return totals


def validate_execution(report: dict[str, Any], evidence: dict[str, str]) -> str:
    execution = report.get("execution")
    require(isinstance(execution, dict), "execution is required")
    assert isinstance(execution, dict)
    execution_id = execution.get("execution_id")
    require(isinstance(execution_id, str), "execution_id must be UUID4 hex")
    assert isinstance(execution_id, str)
    try:
        parsed = uuid.UUID(hex=execution_id)
    except ValueError as exc:
        raise ValueError("execution_id must be UUID4 hex") from exc
    require(
        parsed.version == 4 and parsed.hex == execution_id,
        "execution_id must be lowercase UUID4 hex",
    )
    require(
        execution.get("no_overwrite") is True, "input execution must be no-overwrite"
    )
    require(
        execution.get("output_path_repo_relative") == evidence["path"],
        "execution path binding mismatch",
    )
    return execution_id


def validate_lineage(report: dict[str, Any], contract: str) -> tuple[str, str, str]:
    require(report.get("contract_sha256") == contract, "contract hash mismatch")
    bundle = report.get("source_bundle")
    require(isinstance(bundle, dict), "source_bundle is required")
    assert isinstance(bundle, dict)
    require(bundle.get("git_commit_valid") is True, "source commit must be validated")
    require(
        bundle.get("clean") is True and bundle.get("all_files_present") is True,
        "source bundle must be clean and complete",
    )
    require(
        bundle.get("missing_files") == [] and bundle.get("dirty_source_paths") == [],
        "source bundle lists unresolved files",
    )
    commit = bundle.get("git_commit")
    digest = bundle.get("source_bundle_sha256")
    require(
        isinstance(commit, str)
        and len(commit) == 40
        and all(c in "0123456789abcdef" for c in commit),
        "invalid source commit",
    )
    assert isinstance(commit, str)
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest),
        "invalid source bundle hash",
    )
    assert isinstance(digest, str)
    paths = bundle.get("source_binding_paths")
    files = bundle.get("source_binding_files")
    require(
        isinstance(paths, list) and len(paths) > 0 and len(paths) == len(set(paths)),
        "source binding paths must be non-empty and unique",
    )
    assert isinstance(paths, list)
    require(
        isinstance(files, dict) and set(files) == set(paths),
        "source binding files must match paths",
    )
    assert isinstance(files, dict)
    require(
        all(
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)
            for value in files.values()
        ),
        "invalid source binding file hash",
    )
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    require(
        hashlib.sha256(payload.encode("utf-8")).hexdigest() == digest,
        "source bundle digest mismatch",
    )
    return commit, digest, contract


def validate_common(report: dict[str, Any], expected_device: str) -> None:
    require(report.get("schema_version") == 3, "probe schema must be 3")
    require(
        report.get("goal_id") == "g009" and report.get("stage_id") == "R0",
        "goal/stage mismatch",
    )
    require(report.get("probe") == "flat_recover_runtime_calibration", "probe mismatch")
    require(report.get("task") == "Isaac-G009-Recover-Flat-Go2-R0-v0", "task mismatch")
    require(
        report.get("seed") == 42 and report.get("headless") is True,
        "seed/headless mismatch",
    )
    device = report.get("device")
    require(
        device == "cpu"
        if expected_device == "cpu"
        else isinstance(device, str) and device.startswith("cuda"),
        "device mismatch",
    )
    require(report.get("run_health", {}).get("passed") is True, "run_health must pass")
    require(
        report.get("runtime_contract", {}).get("passed") is True,
        "runtime_contract must pass",
    )
    checks = report.get("checks")
    require(
        isinstance(checks, dict)
        and len(checks) > 0
        and all(type(v) is bool and v for v in checks.values()),
        "all runtime checks must be true",
    )
    require(report.get("passed") is True, "runtime probe must pass")
    qualification = report.get("qualification")
    require(
        isinstance(qualification, dict)
        and qualification.get("status") == "not_run"
        and qualification.get("passed") is None,
        "qualification must be not_run/null",
    )
    require(
        termination_totals(report) == {"numeric_invalid": 0, "hard_joint_limit": 0},
        "safety termination count must be zero",
    )


def validate_readback(report: dict[str, Any]) -> None:
    readback = report.get("physics_readback", {}).get(
        "rigid_body_max_depenetration_velocity"
    )
    require(isinstance(readback, dict), "rigid-body readback is required")
    require(
        readback.get("articulation_group_count") == 8
        and readback.get("rigid_body_count") == 152,
        "readback must cover 8x19 rigid bodies",
    )
    require(
        readback.get("duplicate_link_prim_paths") == [], "readback paths must be unique"
    )
    require(
        readback.get("authoritative_body_names") == list(EXPECTED_BODY_NAMES),
        "authoritative body order mismatch",
    )
    expected_containers = [f"/World/envs/env_{index}/Robot" for index in range(8)]
    expected_roots = [f"{container}/base" for container in expected_containers]
    expected_groups = [
        [f"{container}/{body_name}" for body_name in EXPECTED_BODY_NAMES]
        for container in expected_containers
    ]
    require(
        readback.get("robot_container_prim_paths") == expected_containers,
        "top-level robot container paths mismatch",
    )
    require(
        readback.get("articulation_prim_paths") == expected_roots,
        "top-level articulation paths mismatch",
    )
    require(
        readback.get("authoritative_link_path_groups") == expected_groups,
        "top-level link path groups mismatch",
    )
    articulations = readback.get("articulations")
    require(
        isinstance(articulations, list) and len(articulations) == 8,
        "eight articulation readbacks are required",
    )
    observed_paths: set[str] = set()
    for index, articulation in enumerate(articulations):
        require(
            articulation.get("articulation_index") == index,
            "articulation index mismatch",
        )
        container = f"/World/envs/env_{index}/Robot"
        require(
            articulation.get("robot_container_prim_path") == container,
            "robot container path mismatch",
        )
        require(
            articulation.get("articulation_prim_path") == f"{container}/base",
            "articulation root path mismatch",
        )
        require(
            articulation.get("root_link_prim_path") == f"{container}/base",
            "root link path mismatch",
        )
        require(
            articulation.get("authoritative_body_names") == list(EXPECTED_BODY_NAMES),
            "per-articulation body order mismatch",
        )
        links = articulation.get("links")
        require(
            isinstance(links, list) and len(links) == 19,
            "each articulation must expose 19 links",
        )
        for body_index, (body_name, link) in enumerate(
            zip(EXPECTED_BODY_NAMES, links, strict=True)
        ):
            expected_path = f"{container}/{body_name}"
            require(
                link.get("body_index") == body_index
                and link.get("body_name") == body_name,
                "link identity mismatch",
            )
            require(
                link.get("prim_path") == expected_path
                and expected_path not in observed_paths,
                "link path mismatch or duplicate",
            )
            observed_paths.add(expected_path)
            require(
                link.get("prim_valid") is True
                and link.get("usd_rigid_body_api") is True
                and link.get("physx_rigid_body_api") is True,
                "link API/path readback invalid",
            )
            require(link.get("error") is None, "link readback error must be null")
            value = finite_number(
                link.get("max_depenetration_velocity_m_s"), "max depenetration velocity"
            )
            require(
                math.isclose(value, 0.75, abs_tol=1e-12),
                "max depenetration velocity must equal 0.75 m/s",
            )
    require(len(observed_paths) == 152, "readback must contain 152 unique paths")


def extrema(report: dict[str, Any]) -> tuple[float, float, float | None]:
    metrics = report["pose_mode_metrics"]
    primary = finite_number(
        metric(report, "right_side", "reset_pose_hold").get(
            "max_nonfoot_force_bodyweights"
        ),
        "primary force",
    )
    global_peak = max(
        finite_number(item.get("max_nonfoot_force_bodyweights"), "global force")
        for item in metrics
    )
    separations = [item.get("min_contact_separation_m") for item in metrics]
    available = [
        finite_number(value, "contact separation")
        for value in separations
        if value is not None
    ]
    return primary, global_peak, min(available) if available else None


def validate_rev14(
    report: dict[str, Any], evidence: dict[str, str], device: str
) -> dict[str, Any]:
    validate_common(report, device)
    validate_readback(report)
    execution_id = validate_execution(report, evidence)
    primary, global_peak, separation = extrema(report)
    crosscheck = report.get("required_crosschecks", {}).get("cpu_contact_separation")
    require(
        isinstance(crosscheck, dict) and crosscheck.get("authority_device") == "cpu",
        "CPU separation crosscheck missing",
    )
    if device == "cpu":
        require(
            crosscheck.get("this_run_is_authority") is True
            and crosscheck.get("data_available") is True,
            "CPU must be separation authority",
        )
        require(
            crosscheck.get("status") == "observed"
            and crosscheck.get("passed") is False
            and crosscheck.get("threshold_passed") is False,
            "CPU separation must reproduce the rejection",
        )
        require(
            math.isclose(primary, EXPECTED_PRIMARY_CPU, abs_tol=1e-12),
            "CPU primary force changed",
        )
        require(
            math.isclose(global_peak, EXPECTED_GLOBAL_CPU, abs_tol=1e-12),
            "CPU global force changed",
        )
        require(
            separation is not None
            and math.isclose(separation, EXPECTED_WORST_SEPARATION, abs_tol=1e-12),
            "CPU worst separation changed",
        )
    else:
        require(
            crosscheck.get("this_run_is_authority") is False
            and crosscheck.get("data_available") is False,
            "GPU cannot claim separation authority",
        )
        require(
            crosscheck.get("status") == "requires_cpu_crosscheck"
            and crosscheck.get("passed") is None,
            "GPU separation status mismatch",
        )
        require(
            math.isclose(global_peak, EXPECTED_GLOBAL_GPU, abs_tol=1e-12),
            "GPU global force changed",
        )
        require(separation is None, "GPU must not fabricate contact separation")
    return {
        **evidence,
        "execution_id": execution_id,
        "primary_force_bodyweights": primary,
        "global_force_bodyweights": global_peak,
        "worst_separation_m": separation,
    }


def validate_baseline(
    report: dict[str, Any], evidence: dict[str, str], device: str
) -> dict[str, Any]:
    validate_common(report, device)
    lineage = validate_lineage(report, REV12_CONTRACT)
    validate_execution(report, evidence)
    primary, global_peak, separation = extrema(report)
    if device == "cpu":
        require(
            math.isclose(primary, EXPECTED_BASELINE_CPU_PRIMARY, abs_tol=1e-12),
            "rev12 CPU primary force changed",
        )
        require(
            separation is not None and separation >= SEPARATION_THRESHOLD,
            "rev12 CPU separation baseline must pass",
        )
    return {
        **evidence,
        "source_commit": lineage[0],
        "source_bundle_sha256": lineage[1],
        "primary_force_bodyweights": primary,
        "global_force_bodyweights": global_peak,
        "worst_separation_m": separation,
    }


def summarize(
    cpu_paths: Iterable[Path],
    gpu_paths: Iterable[Path],
    baseline_cpu_path: Path,
    baseline_gpu_path: Path,
) -> dict[str, Any]:
    cpu_paths, gpu_paths = tuple(cpu_paths), tuple(gpu_paths)
    require(
        len(cpu_paths) == len(gpu_paths) == 3,
        "exactly three CPU and three GPU reports are required",
    )
    all_paths = cpu_paths + gpu_paths
    require(
        len({p.resolve() for p in all_paths}) == 6,
        "six distinct rev14 paths are required",
    )
    loaded = [read_json(path) for path in all_paths]
    reports = [item[0] for item in loaded]
    evidence = [item[1] for item in loaded]
    lineage = [validate_lineage(report, REV14_CONTRACT) for report in reports]
    require(
        len(set(lineage)) == 1,
        "all six rev14 reports must share source commit, bundle, and contract",
    )
    require(
        lineage[0] == (REV14_SOURCE_COMMIT, REV14_SOURCE_BUNDLE, REV14_CONTRACT),
        "rev14 source lineage does not match the preregistered clean run",
    )
    cpu_runs = [validate_rev14(reports[i], evidence[i], "cpu") for i in range(3)]
    gpu_runs = [validate_rev14(reports[i], evidence[i], "gpu") for i in range(3, 6)]
    require(
        len({run["execution_id"] for run in cpu_runs + gpu_runs}) == 6,
        "six unique UUID4 execution IDs are required",
    )
    cpu_signature = {
        (
            run["primary_force_bodyweights"],
            run["global_force_bodyweights"],
            run["worst_separation_m"],
        )
        for run in cpu_runs
    }
    gpu_signature = {
        (
            run["primary_force_bodyweights"],
            run["global_force_bodyweights"],
            run["worst_separation_m"],
        )
        for run in gpu_runs
    }
    require(
        len(cpu_signature) == len(gpu_signature) == 1,
        "CPU/GPU three-run semantic results must be reproducible",
    )
    baseline_cpu, baseline_cpu_evidence = read_json(baseline_cpu_path)
    baseline_gpu, baseline_gpu_evidence = read_json(baseline_gpu_path)
    baseline_cpu_summary = validate_baseline(baseline_cpu, baseline_cpu_evidence, "cpu")
    baseline_gpu_summary = validate_baseline(baseline_gpu, baseline_gpu_evidence, "gpu")
    require(
        validate_lineage(baseline_cpu, REV12_CONTRACT)
        == validate_lineage(baseline_gpu, REV12_CONTRACT),
        "rev12 CPU/GPU baseline lineage mismatch",
    )
    exceedance_m = abs(EXPECTED_WORST_SEPARATION - SEPARATION_THRESHOLD)
    require(
        EXPECTED_PRIMARY_CPU <= EXPECTED_BASELINE_CPU_PRIMARY,
        "rev14 CPU primary force must not exceed rev12",
    )
    return {
        "schema_version": 1,
        "goal_id": "g009",
        "stage_number": "G009-5",
        "stage_id": "R0",
        "experiment": "rev14_max_depenetration_velocity_tradeoff",
        "status": "rejected_before_gate01",
        "learned": False,
        "qualification_status": "not_run",
        "conclusion": "rev14 runtime checks and force gates pass, but strict synthesis rejects the CPU separation overrun",
        "lineage": {
            "source_commit": lineage[0][0],
            "source_bundle_sha256": lineage[0][1],
            "contract_sha256": lineage[0][2],
        },
        "repeatability": {
            "cpu": {
                "required_runs": 3,
                "validated_runs": 3,
                "semantically_identical": True,
                "inputs": cpu_runs,
            },
            "gpu": {
                "required_runs": 3,
                "validated_runs": 3,
                "semantically_identical": True,
                "inputs": gpu_runs,
            },
            "unique_execution_ids": 6,
        },
        "physics_readback": {
            "articulations_per_run": 8,
            "links_per_articulation": 19,
            "rigid_bodies_per_run": 152,
            "max_depenetration_velocity_m_s": 0.75,
            "all_paths_and_apis_valid": True,
        },
        "tradeoff": {
            "cpu_primary_right_side_reset_pose_hold_bodyweights": EXPECTED_PRIMARY_CPU,
            "rev12_cpu_primary_bodyweights": EXPECTED_BASELINE_CPU_PRIMARY,
            "cpu_primary_improvement_bodyweights": EXPECTED_BASELINE_CPU_PRIMARY
            - EXPECTED_PRIMARY_CPU,
            "cpu_global_peak_bodyweights": EXPECTED_GLOBAL_CPU,
            "gpu_global_peak_bodyweights": EXPECTED_GLOBAL_GPU,
            "cpu_worst_separation_m": EXPECTED_WORST_SEPARATION,
            "separation_threshold_m": SEPARATION_THRESHOLD,
            "separation_overrun_m": exceedance_m,
            "separation_overrun_mm": exceedance_m * 1000.0,
            "strict_decision": "reject",
        },
        "safety": {
            "numeric_invalid_terminations": 0,
            "hard_joint_limit_terminations": 0,
        },
        "rev12_baselines": {"cpu": baseline_cpu_summary, "gpu": baseline_gpu_summary},
        "completed_stages": {
            "cpu_runtime_3x": True,
            "gpu_runtime_3x": True,
            "strict_tradeoff_synthesis": True,
        },
        "blocked_stages": {"gate01": True, "gate10": True, "ppo_training": True},
    }


def write_summary(
    cpu_paths: Iterable[Path],
    gpu_paths: Iterable[Path],
    baseline_cpu_path: Path,
    baseline_gpu_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    require(not output_path.exists(), f"refusing to overwrite output: {output_path}")
    summary = summarize(cpu_paths, gpu_paths, baseline_cpu_path, baseline_gpu_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", nargs=3, type=Path, default=DEFAULT_CPU)
    parser.add_argument("--gpu", nargs=3, type=Path, default=DEFAULT_GPU)
    parser.add_argument("--baseline-cpu", type=Path, default=DEFAULT_BASELINE_CPU)
    parser.add_argument("--baseline-gpu", type=Path, default=DEFAULT_BASELINE_GPU)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            write_summary(
                args.cpu, args.gpu, args.baseline_cpu, args.baseline_gpu, args.output
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
