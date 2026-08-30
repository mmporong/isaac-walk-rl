#!/usr/bin/env python3
"""Build the offline G009 R0 rev17 mechanism split from canonical rev16 evidence.

This program is deliberately diagnostic-only.  It binds to the exact twelve raw
reports named by the canonical rev16 synthesis, validates them again, and keeps
direct observations, temporal signatures, and causal inferences in separate
top-level sections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "reports/runs"
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import summarize_g009_r0_rev16_backend_divergence as rev16

CANONICAL_SYNTHESIS = RUNS_DIR / "g009_r0_rev16_synthesis_12_full_retry01_s42.json"
CANONICAL_SYNTHESIS_SHA256 = (
    "d39931ad6ddf6104095a6276e9b6db3a047d044d203e034f2d38f1f172e0288d"
)
FOCUS_STEPS = (128, 129, 130)
WINDOW_RADIUS = 8
EXPECTED_BODY_COUNT = 19
EXPECTED_JOINT_COUNT = 12
CONTROL_FIELDS = (
    "input_action",
    "raw_action",
    "processed_ema_target_rad",
    "ema_previous_before_rad",
    "ema_previous_after_rad",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be a JSON number")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def finite_vector(value: Any, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} shape changed")
    return [finite_number(item, label) for item in value]


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def concentration_index(peak_impulse: Any, window_impulse: Any) -> float:
    numerator = finite_number(peak_impulse, "peak impulse numerator")
    denominator = finite_number(window_impulse, "window impulse denominator")
    require(numerator >= 0.0, "peak impulse numerator must be nonnegative")
    require(denominator > 0.0, "17-step base impulse denominator must be positive")
    return numerator / denominator


def _read_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value, raw


def _canonical_inputs(
    synthesis_path: Path = CANONICAL_SYNTHESIS,
) -> tuple[dict[str, Any], dict[str, str], list[tuple[dict[str, Any], dict[str, str]]]]:
    resolved = synthesis_path.resolve(strict=True)
    require(resolved.parent == RUNS_DIR.resolve(), "synthesis must be in reports/runs")
    synthesis, synthesis_raw = _read_json_object(resolved)
    synthesis_hash = sha256_bytes(synthesis_raw)
    require(
        synthesis_hash == CANONICAL_SYNTHESIS_SHA256,
        "canonical rev16 synthesis hash mismatch",
    )
    require(
        synthesis.get("schema_version")
        == "g009.r0.rev16.backend_divergence_synthesis.v1"
        and synthesis.get("revision") == "rev16"
        and synthesis.get("input_report_count") == 12,
        "canonical rev16 synthesis contract mismatch",
    )
    bindings = synthesis.get("input_reports")
    require(isinstance(bindings, list) and len(bindings) == 12, "exactly 12 bindings required")
    assert isinstance(bindings, list)
    entries: list[tuple[dict[str, Any], dict[str, str]]] = []
    for index, binding in enumerate(bindings):
        require(isinstance(binding, dict), f"input binding {index} must be an object")
        assert isinstance(binding, dict)
        relative = binding.get("path")
        expected_hash = binding.get("sha256")
        require(
            isinstance(relative, str)
            and relative.startswith("reports/runs/")
            and Path(relative).parent.as_posix() == "reports/runs",
            f"input binding {index} path escaped reports/runs",
        )
        require(
            isinstance(expected_hash, str) and len(expected_hash) == 64,
            f"input binding {index} hash changed",
        )
        assert isinstance(relative, str)
        assert isinstance(expected_hash, str)
        path = (REPO_ROOT / relative).resolve(strict=True)
        require(path.parent == RUNS_DIR.resolve(), f"input binding {index} escaped reports/runs")
        report, raw = _read_json_object(path)
        actual_hash = sha256_bytes(raw)
        require(actual_hash == expected_hash, f"raw report hash mismatch: {relative}")
        entries.append((report, {"path": relative, "sha256": actual_hash}))
    # The rev16 validator is retained as the authoritative lineage, historical,
    # control-trace, and sequential-predecessor check.
    reproduced = rev16.synthesize_loaded(entries)
    require(
        reproduced.get("input_reports") == bindings,
        "rev16 canonical input binding did not reproduce",
    )
    return synthesis, {"path": f"reports/runs/{resolved.name}", "sha256": synthesis_hash}, entries


def _body_name_from_actor(path: Any) -> str | None:
    if not isinstance(path, str) or "/Robot/" not in path:
        return None
    return path.split("/Robot/", 1)[1].split("/", 1)[0]


def vector_magnitude(vector: list[float]) -> float:
    return math.sqrt(math.fsum(component * component for component in vector))


def _validate_and_measure_row(
    report: dict[str, Any], row: dict[str, Any], expected_step: int, physics_dt: float
) -> dict[str, Any]:
    require(row.get("physics_step") == expected_step, "physics steps must be contiguous 1..600")
    expected_control = (expected_step + 3) // 4
    expected_slot = expected_control * 4 - expected_step
    require(
        row.get("control_step") == expected_control
        and row.get("contact_force_history_slot") == expected_slot
        and row.get("history_slot_order") == "newest_first",
        f"physics history mapping changed at step {expected_step}",
    )
    topology = report.get("runtime_topology")
    require(isinstance(topology, dict), "runtime topology missing")
    assert isinstance(topology, dict)
    expected_names = topology.get("force_body_names")
    names = row.get("body_names")
    require(
        isinstance(names, list)
        and len(names) == EXPECTED_BODY_COUNT
        and names == expected_names
        and len(set(names)) == len(names)
        and all(isinstance(name, str) and name for name in names),
        f"body index/name alignment changed at step {expected_step}",
    )
    assert isinstance(names, list)
    force_vectors = row.get("per_body_force_vector_n")
    magnitudes = row.get("per_body_force_magnitude_n")
    impulse_vectors = row.get("per_body_impulse_vector_n_s")
    require(
        isinstance(force_vectors, list)
        and isinstance(magnitudes, list)
        and isinstance(impulse_vectors, list)
        and len(force_vectors) == len(magnitudes) == len(impulse_vectors) == len(names),
        f"per-body telemetry shape changed at step {expected_step}",
    )
    assert isinstance(force_vectors, list)
    assert isinstance(magnitudes, list)
    assert isinstance(impulse_vectors, list)
    vectors: list[list[float]] = []
    impulses: list[list[float]] = []
    mags: list[float] = []
    for index, name in enumerate(names):
        vector = finite_vector(force_vectors[index], 3, f"{name} force vector")
        magnitude = finite_number(magnitudes[index], f"{name} force magnitude")
        impulse = finite_vector(impulse_vectors[index], 3, f"{name} impulse vector")
        require(magnitude >= 0.0, f"{name} force magnitude must be nonnegative")
        require(
            math.isclose(math.sqrt(math.fsum(component * component for component in vector)), magnitude, rel_tol=2e-6, abs_tol=1e-7),
            f"{name} force magnitude/vector mismatch at step {expected_step}",
        )
        require(
            all(math.isclose(impulse[axis], vector[axis] * physics_dt, rel_tol=2e-6, abs_tol=1e-8) for axis in range(3)),
            f"{name} force/impulse mismatch at step {expected_step}",
        )
        vectors.append(vector)
        impulses.append(impulse)
        mags.append(magnitude)
    magnitude_sum = math.fsum(mags)
    vector_sum = [math.fsum(vector[axis] for vector in vectors) for axis in range(3)]
    impulse_sum = [math.fsum(vector[axis] for vector in impulses) for axis in range(3)]
    resultant = math.sqrt(math.fsum(component * component for component in vector_sum))
    recorded_nonfoot = finite_vector(row.get("nonfoot_resultant_force_vector_n"), 3, "nonfoot resultant")
    nonfoot_ids = topology.get("nonfoot_force_body_ids")
    foot_ids = topology.get("foot_force_body_ids")
    require(isinstance(nonfoot_ids, list) and isinstance(foot_ids, list), "body partitions missing")
    assert isinstance(nonfoot_ids, list)
    assert isinstance(foot_ids, list)
    require(sorted(nonfoot_ids + foot_ids) == list(range(len(names))), "body partitions changed")
    derived_nonfoot = [math.fsum(vectors[index][axis] for index in nonfoot_ids) for axis in range(3)]
    require(
        all(math.isclose(derived_nonfoot[axis], recorded_nonfoot[axis], rel_tol=2e-6, abs_tol=1e-6) for axis in range(3)),
        f"nonfoot vector sum mismatch at step {expected_step}",
    )
    base_index = topology.get("base_force_body_id")
    require(type(base_index) is int and names[base_index] == "base", "base body mapping changed")
    assert isinstance(base_index, int)
    base_impulse = finite_number(row.get("base_impulse_n_s"), "base impulse")
    require(
        math.isclose(base_impulse, math.sqrt(math.fsum(component * component for component in impulses[base_index])), rel_tol=2e-6, abs_tol=1e-8),
        f"base impulse mismatch at step {expected_step}",
    )
    return {
        "physics_step": expected_step,
        "time_s": finite_number(row.get("time_s"), "physics time"),
        "body_force_magnitude_sum_n": magnitude_sum,
        "body_force_vector_sum_n": vector_sum,
        "body_force_vector_sum_magnitude_n": resultant,
        "vector_to_magnitude_sum_ratio": resultant / magnitude_sum if magnitude_sum > 0.0 else None,
        "body_impulse_vector_sum_n_s": impulse_sum,
        "base_force_bodyweights": finite_number(row.get("base_force_bodyweights"), "base force BW"),
        "base_impulse_n_s": base_impulse,
        "body_loads": [
            {
                "body_index": index,
                "body_name": name,
                "force_magnitude_n": mags[index],
                "magnitude_share": mags[index] / magnitude_sum if magnitude_sum > 0.0 else 0.0,
                "force_vector_n": vectors[index],
                "impulse_vector_n_s": impulses[index],
            }
            for index, name in enumerate(names)
        ],
    }


def _cpu_contact_metrics(events: list[Any], selected_steps: set[int]) -> dict[str, Any]:
    selected = [
        event
        for event in events
        if isinstance(event, dict) and event.get("physics_step") in selected_steps
    ]
    pair_counts: Counter[str] = Counter()
    impulse_sum = [0.0, 0.0, 0.0]
    minimum_separation: float | None = None
    point_count = 0
    header_count = 0
    for event in selected:
        require(event.get("complete") is True, "CPU contact event is incomplete")
        headers = event.get("headers")
        require(isinstance(headers, list), "CPU contact headers missing")
        assert isinstance(headers, list)
        for header in headers:
            require(isinstance(header, dict), "CPU contact header must be an object")
            assert isinstance(header, dict)
            header_count += 1
            left = _body_name_from_actor(header.get("actor0_path")) or str(
                header.get("actor0_path")
            )
            right = _body_name_from_actor(header.get("actor1_path")) or str(
                header.get("actor1_path")
            )
            pair_counts[f"{left}<->{right}"] += 1
            points = header.get("contact_points")
            require(isinstance(points, list), "CPU contact points missing")
            assert isinstance(points, list)
            for point in points:
                require(isinstance(point, dict), "CPU contact point must be an object")
                assert isinstance(point, dict)
                point_count += 1
                vector = finite_vector(
                    point.get("reported_contact_impulse_n_s"),
                    3,
                    "reported contact impulse",
                )
                for axis in range(3):
                    impulse_sum[axis] += vector[axis]
                separation = finite_number(
                    point.get("separation_m"), "contact separation"
                )
                minimum_separation = (
                    separation
                    if minimum_separation is None
                    else min(minimum_separation, separation)
                )
    return {
        "event_count": len(selected),
        "header_count": header_count,
        "contact_point_count": point_count,
        "reported_impulse_vector_sum_n_s": impulse_sum,
        "body_pair_counts": dict(sorted(pair_counts.items())),
        "minimum_separation_m": minimum_separation,
    }


def _contact_authority(
    report: dict[str, Any], selected_steps: set[int]
) -> dict[str, Any]:
    device = report.get("device")
    authority = report.get("cpu_contact_authority")
    require(isinstance(authority, dict), "cpu contact authority block missing")
    assert isinstance(authority, dict)
    if device == "cuda:0":
        require(
            authority.get("authority_device") == "cpu"
            and authority.get("this_run_is_authority") is False
            and authority.get("status") == "unavailable_on_gpu"
            and authority.get("data_available") is False
            and authority.get("events") is None
            and authority.get("passed") is None,
            "GPU contact topology must remain explicitly unavailable",
        )
        return {
            "authority": "cpu_only",
            "availability": "unavailable_on_gpu",
            "topology_available": False,
            "event_count": None,
            "contact_point_count": None,
            "reported_impulse_vector_sum_n_s": None,
            "body_pair_counts": None,
            "minimum_separation_m": None,
            "per_physics_step": None,
            "per_physics_step_status": "unavailable_on_gpu",
        }
    require(device == "cpu", "device must be cpu or cuda:0")
    events = authority.get("events")
    require(
        authority.get("authority_device") == "cpu"
        and authority.get("this_run_is_authority") is True
        and authority.get("status") == "observed"
        and authority.get("data_available") is True
        and authority.get("passed") is True
        and isinstance(events, list),
        "CPU contact authority contract changed",
    )
    assert isinstance(events, list)
    aggregate = _cpu_contact_metrics(events, selected_steps)
    return {
        "authority": "cpu_only",
        "availability": "observed",
        "topology_available": True,
        **aggregate,
        "per_physics_step": {
            str(step): _cpu_contact_metrics(events, {step}) for step in FOCUS_STEPS
        },
        "per_physics_step_status": "observed_cpu_authority",
    }


def _validate_control_rows(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("control_step_telemetry")
    require(isinstance(rows, list) and len(rows) == 150, "exactly 150 control rows are required")
    assert isinstance(rows, list)
    topology = report.get("runtime_topology")
    require(isinstance(topology, dict), "runtime topology missing")
    assert isinstance(topology, dict)
    expected_links = topology.get("link_body_names")
    expected_joints = topology.get("joint_names")
    observations: dict[str, dict[str, Any]] = {}
    for expected_step, row in enumerate(rows, 1):
        require(
            isinstance(row, dict) and row.get("control_step") == expected_step,
            "control steps must be contiguous 1..150",
        )
        assert isinstance(row, dict)
        time_s = finite_number(row.get("time_s"), "control time")
        require(
            math.isclose(time_s, expected_step * 0.02, abs_tol=1e-12),
            f"control time changed at step {expected_step}",
        )
        flags = row.get("termination_flags")
        require(
            isinstance(flags, dict)
            and set(flags)
            == {"time_out", "stable_success", "numeric_invalid", "hard_joint_limit"}
            and all(type(value) is bool for value in flags.values()),
            f"termination flags changed at control step {expected_step}",
        )
        assert isinstance(flags, dict)
        root = finite_vector(row.get("root_state_w"), 13, "root state")
        require(
            row.get("link_state_field") == "body_link_state_w"
            and row.get("link_names") == expected_links,
            f"link state contract changed at control step {expected_step}",
        )
        link_states = row.get("link_state_w")
        require(
            isinstance(link_states, list)
            and len(link_states) == EXPECTED_BODY_COUNT,
            f"link state count changed at control step {expected_step}",
        )
        assert isinstance(link_states, list)
        links = [finite_vector(state, 13, "link state") for state in link_states]
        require(
            row.get("joint_names") == expected_joints,
            f"joint names changed at control step {expected_step}",
        )
        joint_position = finite_vector(
            row.get("joint_position_rad"), EXPECTED_JOINT_COUNT, "joint position"
        )
        joint_velocity = finite_vector(
            row.get("joint_velocity_rad_s"), EXPECTED_JOINT_COUNT, "joint velocity"
        )
        torque = finite_vector(
            row.get("applied_torque_nm"), EXPECTED_JOINT_COUNT, "applied torque"
        )
        traces = {
            field: finite_vector(row.get(field), EXPECTED_JOINT_COUNT, field)
            for field in CONTROL_FIELDS
        }
        if expected_step in (32, 33):
            observations[str(expected_step)] = {
                "control_step": expected_step,
                "time_s": time_s,
                "interpretation_label": "control-bucket state/context; not an instantaneous state measurement for every mapped physics substep",
                "root_linear_speed_m_s": vector_magnitude(root[7:10]),
                "root_angular_speed_rad_s": vector_magnitude(root[10:13]),
                "max_link_linear_speed_m_s": max(
                    vector_magnitude(state[7:10]) for state in links
                ),
                "max_link_angular_speed_rad_s": max(
                    vector_magnitude(state[10:13]) for state in links
                ),
                "max_joint_speed_rad_s": max(abs(value) for value in joint_velocity),
                "max_abs_applied_torque_nm": max(abs(value) for value in torque),
                "joint_position_rad": joint_position,
                "joint_velocity_rad_s": joint_velocity,
                "applied_torque_nm": torque,
                "action_and_ema_trace": traces,
                "termination_flags": dict(flags),
            }
    return {
        "control_row_count": len(rows),
        "physics_to_control_bucket_mapping": [
            {
                "physics_step": step,
                "control_step": (step + 3) // 4,
                "contact_force_history_slot": ((step + 3) // 4) * 4 - step,
            }
            for step in FOCUS_STEPS
        ],
        "mapping_interpretation_label": "bucket/history mapping only; control telemetry must not be interpreted as an instantaneous state at each physics substep",
        "selected_control_buckets": observations,
    }


def _validate_group_semantic_identity(
    group: str, runs: list[dict[str, Any]]
) -> dict[str, Any]:
    require(len(runs) == 3, f"{group} requires exactly three replicates")
    fields = (
        "physics_substep_telemetry_sha256",
        "control_step_telemetry_sha256",
        "cpu_contact_authority_sha256",
    )
    hashes: dict[str, str] = {}
    for field in fields:
        values = [run["semantic_payload_hashes"][field] for run in runs]
        require(
            len(set(values)) == 1,
            f"{group} replicate semantic payload mismatch: {field}",
        )
        hashes[field] = values[0]
    return {
        "group": group,
        "replicate_count": 3,
        "identical_3_of_3": True,
        "canonical_json_sha256": hashes,
    }


def _measure_run(
    report: dict[str, Any],
    evidence: dict[str, str],
    canonical_run: dict[str, Any],
    arm: str,
    device: str,
) -> dict[str, Any]:
    require(report.get("device") == device, "canonical device binding changed")
    timing = report.get("telemetry_timing")
    require(isinstance(timing, dict), "telemetry timing missing")
    assert isinstance(timing, dict)
    physics_dt = finite_number(timing.get("physics_dt_s"), "physics dt")
    require(math.isclose(physics_dt, 0.005, abs_tol=1e-12), "physics dt changed")
    require(timing.get("peak_window_radius_physics_steps") == WINDOW_RADIUS, "17-step window contract changed")
    rows = report.get("physics_substep_telemetry")
    require(isinstance(rows, list) and len(rows) == 600, "exactly 600 physics rows are required")
    assert isinstance(rows, list)
    measured = [
        _validate_and_measure_row(report, row, expected_step, physics_dt)
        for expected_step, row in enumerate(rows, 1)
        if isinstance(row, dict)
    ]
    require(len(measured) == 600, "every physics row must be an object")
    peak_step = max(measured, key=lambda item: item["base_force_bodyweights"])["physics_step"]
    require(peak_step == canonical_run.get("peak_base_force_physics_step"), "rev16 peak step did not reproduce")
    window_first, window_last = peak_step - WINDOW_RADIUS, peak_step + WINDOW_RADIUS
    require(window_first >= 1 and window_last <= 600, "peak lacks a complete 17-step window")
    peak_impulse = measured[peak_step - 1]["base_impulse_n_s"]
    window_impulse = math.fsum(measured[step - 1]["base_impulse_n_s"] for step in range(window_first, window_last + 1))
    concentration = concentration_index(peak_impulse, window_impulse)
    require(
        math.isclose(concentration, finite_number(canonical_run.get("concentration_index"), "canonical concentration"), rel_tol=1e-12, abs_tol=1e-15),
        "rev16 concentration value did not reproduce",
    )
    focus = [measured[step - 1] for step in FOCUS_STEPS]
    selected_steps = set(range(window_first, window_last + 1)) | set(FOCUS_STEPS)
    return {
        "evidence": evidence,
        "arm": arm,
        "device": device,
        "replicate_index": report.get("replicate_index"),
        "physics_row_count": len(measured),
        "focus_steps": focus,
        "control_context": _validate_control_rows(report),
        "contact_authority": _contact_authority(report, selected_steps),
        "semantic_payload_hashes": {
            "physics_substep_telemetry_sha256": rev16.canonical_sha256(rows),
            "control_step_telemetry_sha256": rev16.canonical_sha256(
                report["control_step_telemetry"]
            ),
            "cpu_contact_authority_sha256": rev16.canonical_sha256(
                report["cpu_contact_authority"]
            ),
        },
        "peak_window": {
            "radius_physics_steps": WINDOW_RADIUS,
            "step_count": 17,
            "first_physics_step": window_first,
            "peak_physics_step": peak_step,
            "last_physics_step": window_last,
            "peak_base_force_bodyweights": measured[peak_step - 1][
                "base_force_bodyweights"
            ],
            "peak_base_impulse_n_s": peak_impulse,
            "window_base_impulse_n_s": window_impulse,
            "concentration_index": concentration,
            "rev16_concentration_reproduced": True,
            "body_impulse_magnitude_totals_n_s": {
                name: math.fsum(
                    math.sqrt(math.fsum(component * component for component in measured[step - 1]["body_loads"][index]["impulse_vector_n_s"]))
                    for step in range(window_first, window_last + 1)
                )
                for index, name in enumerate(rows[0]["body_names"])
            },
        },
    }


def synthesize(synthesis_path: Path = CANONICAL_SYNTHESIS) -> dict[str, Any]:
    canonical, binding, entries = _canonical_inputs(synthesis_path)
    canonical_runs = [
        (run, group["arm"], group["device"])
        for group in canonical["groups"]
        for run in group["runs"]
    ]
    require(len(canonical_runs) == len(entries) == 12, "canonical run matrix changed")
    runs = [
        _measure_run(report, evidence, canonical_run, arm, device)
        for (report, evidence), (canonical_run, arm, device) in zip(
            entries, canonical_runs, strict=True
        )
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[f"{run['arm']}.{run['device']}"].append(run)
    concentration = {
        group: [run["peak_window"]["concentration_index"] for run in group_runs]
        for group, group_runs in grouped.items()
    }
    semantic_identity = [
        _validate_group_semantic_identity(group, group_runs)
        for group, group_runs in grouped.items()
    ]
    ratios = [
        grouped["B.cuda:0"][index]["peak_window"]["concentration_index"]
        / grouped["B.cpu"][index]["peak_window"]["concentration_index"]
        for index in range(3)
    ]
    b_comparisons: list[dict[str, Any]] = []
    for index in range(3):
        cpu = grouped["B.cpu"][index]["peak_window"]
        gpu = grouped["B.cuda:0"][index]["peak_window"]
        cpu_body = cpu["body_impulse_magnitude_totals_n_s"]
        gpu_body = gpu["body_impulse_magnitude_totals_n_s"]
        cpu_total = math.fsum(cpu_body.values())
        gpu_total = math.fsum(gpu_body.values())
        cpu_rear_hips = cpu_body["FR_hip"] + cpu_body["RR_hip"]
        gpu_rear_hips = gpu_body["FR_hip"] + gpu_body["RR_hip"]
        b_comparisons.append(
            {
                "replicate_index": index + 1,
                "gpu_over_cpu_peak_base_force_percent_change": (
                    gpu["peak_base_force_bodyweights"]
                    / cpu["peak_base_force_bodyweights"]
                    - 1.0
                )
                * 100.0,
                "gpu_over_cpu_all_body_impulse_magnitude_window_percent_change": (
                    gpu_total / cpu_total - 1.0
                )
                * 100.0,
                "gpu_over_cpu_base_window_impulse_percent_change": (
                    gpu["window_base_impulse_n_s"]
                    / cpu["window_base_impulse_n_s"]
                    - 1.0
                )
                * 100.0,
                "cpu_base_share_of_all_body_impulse_magnitude": cpu_body["base"]
                / cpu_total,
                "gpu_base_share_of_all_body_impulse_magnitude": gpu_body["base"]
                / gpu_total,
                "gpu_over_cpu_fr_rr_hip_impulse_magnitude_percent_change": (
                    gpu_rear_hips / cpu_rear_hips - 1.0
                )
                * 100.0,
            }
        )
    hypothesis_replicates = canonical["hypothesis"]["replicates"]
    require(
        isinstance(hypothesis_replicates, list) and len(hypothesis_replicates) == 3,
        "canonical rev16 hypothesis matrix changed",
    )
    assert isinstance(hypothesis_replicates, list)
    first_physics_force_divergences = [
        replicate["derived"]["first_divergence_pairs"]["b_cpu_vs_b_gpu"][
            "first_physics_divergence"
        ]
        for replicate in hypothesis_replicates
    ]
    first_control_divergences = [
        replicate["derived"]["first_divergence_pairs"]["b_cpu_vs_b_gpu"][
            "first_control_divergence"
        ]
        for replicate in hypothesis_replicates
    ]
    require(
        all(
            replicate["derived"]["max_action_ema_trace_abs_error"] == 0.0
            for replicate in hypothesis_replicates
        )
        and all(
            item == first_physics_force_divergences[0]
            for item in first_physics_force_divergences
        )
        and first_physics_force_divergences[0].get("step") == 128
        and first_physics_force_divergences[0].get("variable")
        == "base_force_bodyweights"
        and all(item == first_control_divergences[0] for item in first_control_divergences)
        and first_control_divergences[0].get("step") == 1,
        "canonical action/EMA or force/control-divergence signature changed",
    )
    return {
        "schema_version": "g009.r0.rev17.mechanism_split.v1",
        "evidence_id": "G009-5-E010",
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev17",
        "status": "pass",
        "mode": "offline_reanalysis_of_immutable_rev16_reports",
        "integrity": {
            "passed": True,
            "hash_bound": True,
            "predecessor_path": binding["path"],
            "predecessor_sha256": binding["sha256"],
            "canonical_rev16_synthesis": binding,
            "input_report_count": 12,
            "input_reports": canonical["input_reports"],
        },
        "diagnostic_only": True,
        "ppo": {"allowed": False, "status": "not_run"},
        "qualification": {"eligible": False, "status": "not_run", "passed": None},
        "canonical_rev16_synthesis": binding,
        "input_report_count": 12,
        "input_reports": canonical["input_reports"],
        "mechanism_split": {
            "decision": {
                "outcome": "inconclusive",
                "selected_lever": None,
            },
            "direct_observations": {
                "scope": {
                    "focus_physics_steps": list(FOCUS_STEPS),
                    "peak_window_radius_physics_steps": WINDOW_RADIUS,
                    "peak_window_step_count": 17,
                },
                "runs": runs,
            },
            "temporal_signatures": {
                "concentration_index_by_group": concentration,
                "b_gpu_over_b_cpu_concentration_ratio_by_replicate": ratios,
                "all_rev16_concentration_values_reproduced": all(
                    run["peak_window"]["rev16_concentration_reproduced"]
                    for run in runs
                ),
                "focus_interpretation": "steps 128-130 are reported without replacing each run's independently located peak window",
                "b_cpu_gpu_mechanism_comparison": b_comparisons,
                "action_and_ema_trace_max_abs_error": 0.0,
                "b_cpu_gpu_first_control_state_divergence_step": first_control_divergences[
                    0
                ]["step"],
                "b_cpu_gpu_first_physics_force_divergence": {
                    "status": "observed_in_force_aggregation",
                    "replicate_count": 3,
                    "identical_3_of_3": True,
                    **first_physics_force_divergences[0],
                },
                "b_cpu_gpu_contact_topology_divergence": {
                    "status": "unavailable_on_gpu",
                    "step": None,
                    "reason": "GPU contact-pair authority is unavailable, so a CPU/GPU contact-topology divergence step cannot be observed",
                },
                "replicate_semantic_identity": semantic_identity,
            },
            "causal_inferences": {
                "decision": {
                    "outcome": "inconclusive",
                    "selected_lever": None,
                    "reason": "force redistribution is observed, but the immutable reports do not isolate one causal solver lever",
                },
                "supported_claims": [
                    "the canonical rev16 base-impulse concentration values reproduce from raw 600-step telemetry",
                    "CPU contact-pair topology is observable only in CPU authority reports",
                    "GPU reports provide force aggregation but no contact-pair authority",
                ],
                "not_supported_claims": [
                    "a specific solver parameter caused the CPU/GPU divergence",
                    "GPU contact pairs are equivalent to CPU contact pairs",
                    "PPO or qualification may begin from this diagnostic alone",
                ],
                "next_action": "keep selected_lever=null, then add authoritative constraint/contact instrumentation or run a preregistered single-variable intervention probe",
            },
        },
        "governance": {
            "diagnostic_only": True,
            "learned": False,
            "ppo": {"allowed": False, "status": "not_run"},
            "gate01": {"allowed": False, "status": "forbidden"},
            "qualification": {"eligible": False, "status": "not_run", "passed": None},
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis", type=Path, default=CANONICAL_SYNTHESIS)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    require(output.parent == RUNS_DIR.resolve(), "output must be a direct child of reports/runs")
    report = synthesize(args.synthesis)
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    report["synthesis_execution_id"] = uuid.uuid4().hex
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite output: {output}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
