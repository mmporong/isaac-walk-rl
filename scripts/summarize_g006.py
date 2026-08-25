#!/usr/bin/env python3
"""Strict validation and paired summary for the complete G006 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaac_walk_g006.evaluation.protocol import (  # noqa: E402
    compute_evaluation_source_bundle,
    deterministic_hierarchical_paired_bootstrap,
    validate_success_criteria,
    wilson_interval,
)


class ValidationError(ValueError):
    """Raised when evidence does not satisfy the fixed G006 contract."""


PORTABLE_TOKEN_RE = re.compile(
    r"^(?P<token>%USERPROFILE%|%REPO_ROOT%|%ISAACLAB_ROOT%)(?P<suffix>(?:[\\/].*)?)$",
    re.IGNORECASE,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> bool:
    resolved, resolved_root = path.resolve(), root.resolve()
    return resolved == resolved_root or resolved.is_relative_to(resolved_root)


def resolve_portable_path(
    value: str,
    relative_to: Path | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    isaaclab_root: Path | None = None,
) -> Path:
    if "%" in value:
        match = PORTABLE_TOKEN_RE.fullmatch(value)
        if match is None:
            raise ValidationError("portable_token_invalid")
        token = match.group("token").upper()
        roots = {
            "%USERPROFILE%": Path.home(),
            "%REPO_ROOT%": repo_root,
            "%ISAACLAB_ROOT%": isaaclab_root,
        }
        selected_root = roots[token]
        if selected_root is None:
            raise ValidationError("isaaclab_root_required")
        suffix = match.group("suffix").lstrip("\\/").replace("\\", os.sep)
        resolved = (selected_root / suffix).resolve() if suffix else selected_root.resolve()
        if not _within(resolved, selected_root):
            raise ValidationError("portable_token_escape")
        return resolved

    path = Path(value)
    if not path.is_absolute():
        base = relative_to or repo_root
        resolved = (base / path).resolve()
        if _within(resolved, base):
            return resolved
        raise ValidationError("path_outside_allowed_roots")

    resolved = path.resolve()
    allowed_roots = [Path.home(), repo_root, isaaclab_root]
    if any(root is not None and _within(resolved, root) for root in allowed_roots):
        return resolved
    raise ValidationError("path_outside_allowed_roots")


def portable_path(
    path: Path,
    relative_to: Path | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    isaaclab_root: Path | None = None,
) -> str:
    """Serialize a resolved path without binding evidence to a local username."""

    resolved = path.resolve()
    repository = repo_root.resolve()
    if _within(resolved, repository):
        return resolved.relative_to(repository).as_posix() or "."
    if relative_to is not None and _within(resolved, relative_to):
        base = relative_to.resolve()
        return resolved.relative_to(base).as_posix() or "."
    home = Path.home().resolve()
    if _within(resolved, home):
        suffix = resolved.relative_to(home)
        return "%USERPROFILE%" + ("\\" + str(suffix) if suffix.parts else "")
    if isaaclab_root is not None and _within(resolved, isaaclab_root):
        suffix = resolved.relative_to(isaaclab_root.resolve())
        return "%ISAACLAB_ROOT%" + ("\\" + str(suffix) if suffix.parts else "")
    raise ValidationError("path_outside_portable_roots")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("json_read_failed") from exc
    require(isinstance(value, dict), "json_root_not_object")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_failure_message(exc: Exception) -> str:
    if not isinstance(exc, ValidationError):
        return "unexpected_error"
    message = str(exc)
    home = str(Path.home())
    if (
        (home and home.casefold() in message.casefold())
        or re.search(r"(?i)(?:[a-z]:[\\/]|/(?:home|users)/)", message)
    ):
        return "validation_failed"
    return message


def compute_declared_source_bundle(files: list[dict[str, Any]]) -> dict[str, Any]:
    require(isinstance(files, list) and files, "source bundle files are required")
    digest = hashlib.sha256()
    normalized = []
    paths = [str(item.get("path", "")).replace("\\", "/") for item in files]
    require(paths == sorted(paths) and len(set(paths)) == len(paths), "source bundle paths must be unique and ordinal-sorted")
    for item, relative in zip(files, paths):
        require(relative and not relative.startswith("/") and ".." not in Path(relative).parts, "invalid source bundle path")
        path = (REPO_ROOT / Path(relative)).resolve()
        require(path.is_file() and path.is_relative_to(REPO_ROOT.resolve()), f"source bundle file missing/outside repo: {relative}")
        raw = path.read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        require(item.get("sha256") == actual_hash, f"source bundle file hash mismatch: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
        normalized.append({"path": relative, "sha256": actual_hash})
    return {"sha256": digest.hexdigest(), "files": normalized}


def validate_manifest(manifest: dict[str, Any]) -> tuple[list[int], dict[str, Any], str]:
    require(manifest.get("goal") == "G006", "manifest goal must be G006")
    seeds = manifest.get("training", {}).get("seeds")
    require(seeds == [42, 43, 44], "training seeds must be [42,43,44]")
    variants = manifest.get("variants")
    require(isinstance(variants, list) and [item.get("name") for item in variants] == ["baseline", "push_curriculum"], "two ordered variants required")
    require(variants[0].get("normalized_cfg_difference_from_baseline") == [], "baseline normalized diff must be empty")
    require(variants[1].get("normalized_cfg_difference_from_baseline") == ["events.push_robot"], "push curriculum must differ only at events.push_robot")
    protocol = manifest.get("evaluation_protocol")
    require(isinstance(protocol, dict), "evaluation_protocol is required")
    require(protocol.get("push_trial_count") == 1080 and protocol.get("guardrail_trial_count") == 90, "trial counts must be 1080 and 90")
    require(protocol.get("bootstrap_resamples") == 10_000, "bootstrap_resamples must be 10000")
    validate_success_criteria(protocol.get("success_criteria"))
    agent_cfg = manifest.get("training", {}).get("agent_learning_config")
    require(isinstance(agent_cfg, dict), "agent_learning_config is required")
    require(canonical_sha256(agent_cfg) == manifest["training"].get("agent_learning_config_sha256"), "agent learning config hash mismatch")
    return [int(seed) for seed in seeds], protocol, canonical_sha256(protocol)


def validate_terrain_evidence(value: Any) -> None:
    require(isinstance(value, dict), "terrain_evidence object missing")
    selected = value.get("selected_tiles")
    require(isinstance(selected, list) and len(selected) == 30, "terrain evidence requires 30 tiles")
    require(len({item.get("raw_sha256") for item in selected}) == 30, "raw terrain hashes must be unique")
    require(len({item.get("mesh_sha256") for item in selected}) == 30, "mesh terrain hashes must be unique")
    metrics = ("height_rms_m", "height_p90_abs_m", "face_normal_slope_rms_rad", "face_normal_slope_p90_rad")
    for col in range(10):
        rows = {item.get("row"): item for item in selected if item.get("col") == col}
        require(set(rows) == {1, 4, 8}, f"terrain col {col} is missing difficulty rows")
        for metric in metrics:
            values = [float(rows[row]["metrics"][metric]) for row in (1, 4, 8)]
            require(values[0] < values[1] < values[2], f"terrain monotonicity failed col={col} metric={metric}")


def validate_eval_report(
    report: dict[str, Any], *, mode: str, variant: str, seed: int, protocol_hash: str, checkpoint_hash: str
) -> None:
    require(report.get("schema_version") == 1 and report.get("goal") == "G006" and report.get("status") == "complete", "evaluation identity/status invalid")
    require(report.get("protocol_compliant") is True and report.get("experimental_use") == "g006_production_evaluation", "production summary rejects smoke-only evidence")
    require(report.get("mode") == mode and report.get("variant") == variant and report.get("training_seed") == seed, "evaluation binding mismatch")
    require(report.get("protocol", {}).get("sha256") == protocol_hash, "evaluation protocol hash mismatch")
    require(report.get("checkpoint", {}).get("sha256") == checkpoint_hash, "evaluation checkpoint hash mismatch")
    validate_success_criteria(report.get("success_criteria"))
    current_bundle = compute_evaluation_source_bundle(REPO_ROOT)
    require(report.get("evaluation_source_bundle_sha256") == current_bundle["sha256"], "evaluation source bundle hash mismatch")
    require(report.get("evaluation_source_bundle_files") == current_bundle["files"], "evaluation source bundle files mismatch")
    trials = report.get("trials")
    expected = 1080 if mode == "push" else 90
    require(isinstance(trials, list) and len(trials) == expected, f"{mode} report requires {expected} trials")
    require(len({trial.get("trial_id") for trial in trials}) == expected, f"{mode} trial IDs must be unique")
    expected_pair = f"{variant}-s{seed}"
    require(all(trial.get("pair_id") == expected_pair for trial in trials), f"{mode} pair IDs invalid")
    common_trial_keys = {
        "trial_id", "pair_id", "paired_trial_key", "stratum_id", "excluded_reason",
        "tracking_error_sq_mean", "yaw_error_sq_mean", "torque_l2_mean",
        "absolute_mechanical_power_mean", "action_rate_l2_mean",
    }
    mode_trial_keys = (
        {"eligible", "criterion_met", "recovered", "failed", "recovery_failed", "recovery_step", "prepush_failure", "survived_to_horizon", "physical_failure", "protocol_blocked"}
        if mode == "push"
        else {"guardrail_eligible", "guardrail_survived", "survived_to_horizon", "protocol_blocked"}
    )
    require(all(common_trial_keys | mode_trial_keys <= set(trial) for trial in trials), f"{mode} trial schema incomplete")
    require(all(not bool(trial["protocol_blocked"]) for trial in trials), f"{mode} contains protocol-blocked trial")
    if mode == "push":
        for trial in trials:
            expected_recovered = bool(trial["criterion_met"]) and bool(trial["survived_to_horizon"])
            require(bool(trial["recovered"]) == expected_recovered, "recovered must require criterion and horizon survival")
            require(bool(trial["failed"]) == (bool(trial["eligible"]) and not expected_recovered), "failed/recovered mismatch")
            require(bool(trial["recovery_failed"]) == (bool(trial["eligible"]) and not bool(trial["criterion_met"])), "recovery_failed/criterion mismatch")
            require(bool(trial["physical_failure"]) == (bool(trial["eligible"]) and not bool(trial["survived_to_horizon"])), "physical failure/survival mismatch")
            if trial["criterion_met"]:
                require(225 <= int(trial["recovery_step"]) <= 450, "criterion recovery_step outside completed-step window")
            else:
                require(trial.get("recovery_step") is None, "non-criterion trial cannot have recovery_step")
    cells = report.get("cells")
    expected_cells = 108 if mode == "push" else 9
    require(isinstance(cells, list) and len(cells) == expected_cells, f"{mode} cell schema/count invalid")
    require(all(cell.get("trial_count") == 10 and isinstance(cell.get("raw_metrics"), dict) for cell in cells), f"{mode} cell raw metrics/count invalid")
    aggregate = report.get("aggregate")
    require(isinstance(aggregate, dict) and aggregate.get("trial_count") == expected, "evaluation aggregate missing/count invalid")
    require(aggregate.get("survived_to_horizon_count") == sum(bool(trial["survived_to_horizon"]) for trial in trials), "aggregate horizon survival mismatch")
    if mode == "push":
        eligible = [trial for trial in trials if trial["eligible"]]
        require(aggregate.get("eligible_count") == len(eligible), "aggregate eligible mismatch")
        require(aggregate.get("criterion_met_count") == sum(bool(trial["criterion_met"]) for trial in eligible), "aggregate criterion mismatch")
        require(aggregate.get("recovered_count") == sum(bool(trial["recovered"]) for trial in eligible), "aggregate recovered mismatch")
    validate_terrain_evidence(report.get("terrain_evidence"))
    runtime = report.get("runtime")
    require(isinstance(runtime, dict), "runtime evidence missing")
    require(runtime.get("task") and "Play" not in runtime["task"], "full rough task required")
    require(runtime.get("terrain_levels_runtime") is None, "runtime terrain_levels must be None")
    require(runtime.get("observation_corruption") is False and runtime.get("events_enabled") == [], "evaluation corruption/events must be disabled")
    require(float(runtime.get("base_contact_threshold_n", -1)) == 1.0, "base contact threshold must be 1 N")
    require(runtime.get("push_injection_completed_steps") == 200 and runtime.get("horizon_completed_step") == 600, "runtime completed-step timing mismatch")
    require(runtime.get("preliminary") is False and runtime.get("exit_code") == 0, "runtime preliminary/exit gate failed")
    require(runtime.get("app_close_completed") is True and runtime.get("finalized_after_process_exit") is True, "runtime was not finalized after process exit")
    require(runtime.get("gpu_measurement_complete") is True, "GPU measurement incomplete")
    require(runtime.get("process_recovered") is True and runtime.get("gpu_recovered_to_baseline") is True, "evaluation process/GPU did not recover")
    require(runtime.get("fatal_patterns") == [], "fatal runtime patterns found")
    boundary_count = int(report["aggregate"].get("boundary_violation_count", 0))
    require(boundary_count == 0, "tile boundary violation protocol-blocked")
    require(int(report["aggregate"].get("auto_reset_excluded_count", 0)) == 0, "auto-reset poison protocol-blocked")


def _training_report(
    job: dict[str, Any],
    queue_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    isaaclab_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    path = resolve_portable_path(
        str(job["report_path"]), queue_path.parent,
        repo_root=repo_root, isaaclab_root=isaaclab_root,
    )
    report = read_json(path)
    require(report.get("passed") is True, f"training report failed: {job.get('id')}")
    require(report.get("task") == job.get("task") and int(report.get("seed")) == int(job.get("seed")), "training report identity mismatch")
    checkpoint = resolve_portable_path(
        str(report.get("artifacts", {}).get("checkpoint")), queue_path.parent,
        repo_root=repo_root, isaaclab_root=isaaclab_root,
    )
    checkpoint_hash = report.get("artifacts", {}).get("checkpoint_sha256")
    require(checkpoint.is_file() and file_sha256(checkpoint) == checkpoint_hash == job.get("checkpoint_sha256"), "checkpoint hash mismatch")
    return report, checkpoint


def build_paired_recovery_deltas(
    baseline_by_seed: dict[int, list[dict[str, Any]]],
    curriculum_by_seed: dict[int, list[dict[str, Any]]],
) -> dict[int, dict[str, list[float]]]:
    """Validate pair keys/strata and produce 108x10 deltas per seed."""

    require(set(baseline_by_seed) == set(curriculum_by_seed), "paired seed mismatch")
    paired: dict[int, dict[str, list[float]]] = {}
    for seed in sorted(baseline_by_seed):
        baseline = {trial["paired_trial_key"]: trial for trial in baseline_by_seed[seed]}
        curriculum = {trial["paired_trial_key"]: trial for trial in curriculum_by_seed[seed]}
        require(set(baseline) == set(curriculum) and len(baseline) == 1080, f"paired trial mismatch seed={seed}")
        paired[seed] = {}
        for key in sorted(baseline):
            require(baseline[key]["stratum_id"] == curriculum[key]["stratum_id"], "paired stratum mismatch")
            stratum = baseline[key]["stratum_id"]
            paired[seed].setdefault(stratum, []).append(
                float(bool(curriculum[key]["recovered"])) - float(bool(baseline[key]["recovered"]))
            )
        require(len(paired[seed]) == 108 and all(len(values) == 10 for values in paired[seed].values()), "paired 108x10 structure invalid")
    return paired


def summarize(
    manifest_path: Path,
    queue_path: Path,
    *,
    isaaclab_root: Path | None = None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    queue = read_json(queue_path)
    seeds, protocol, protocol_hash = validate_manifest(manifest)
    require(
        queue.get("goal") == "G006"
        and queue.get("mode") == "production"
        and queue.get("status") in {"summarizing", "complete"},
        "production queue must be summarizing or complete",
    )
    require(queue.get("config_sha256") == canonical_sha256(manifest), "queue manifest hash mismatch")
    require(queue.get("protocol_sha256") == protocol_hash, "queue protocol hash mismatch")
    source_bundles = queue.get("source_bundles")
    require(isinstance(source_bundles, dict) and set(source_bundles) == {"training", "evaluation"}, "queue source_bundles missing")
    current_training_bundle = compute_declared_source_bundle(source_bundles["training"].get("files"))
    current_evaluation_bundle = compute_evaluation_source_bundle(REPO_ROOT)
    require(source_bundles["training"] == current_training_bundle, "queue training source bundle mismatch")
    require(source_bundles["evaluation"] == current_evaluation_bundle, "queue evaluation source bundle mismatch")
    require(queue.get("training_source_bundle_sha256") == current_training_bundle["sha256"], "queue training source bundle hash mismatch")
    require(queue.get("evaluation_source_bundle_sha256") == current_evaluation_bundle["sha256"], "queue evaluation source bundle hash mismatch")
    bootstrap_path = REPO_ROOT / "scripts" / "bootstrap_train_g006.py"
    evaluator_path = REPO_ROOT / "scripts" / "evaluate_push_recovery.py"
    require(queue.get("training_entrypoint_sha256") == file_sha256(bootstrap_path), "training entrypoint hash mismatch")
    jobs = queue.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 6, "exactly six production jobs required")
    expected = {(variant, seed) for variant in ("baseline", "push_curriculum") for seed in seeds}
    actual = {(job.get("variant"), int(job.get("seed"))) for job in jobs}
    require(actual == expected and all(job.get("status") == "complete" for job in jobs), "2x3 complete job matrix required")

    evidence_jobs: list[dict[str, Any]] = []
    reports: dict[tuple[str, int, str], dict[str, Any]] = {}
    for job in jobs:
        variant, seed = str(job["variant"]), int(job["seed"])
        training, checkpoint = _training_report(
            job, queue_path, repo_root=REPO_ROOT, isaaclab_root=isaaclab_root,
        )
        require(job.get("training_source_bundle_sha256") == current_training_bundle["sha256"], "job training source bundle mismatch")
        require(job.get("evaluation_source_bundle_sha256") == current_evaluation_bundle["sha256"], "job evaluation source bundle mismatch")
        require(training.get("training_source_bundle_sha256") == current_training_bundle["sha256"], "training report source bundle mismatch")
        require(training.get("training_entrypoint", {}).get("sha256") == file_sha256(bootstrap_path), "training report entrypoint hash mismatch")
        require(job.get("push_script_sha256") == file_sha256(evaluator_path), "push evaluator script hash mismatch")
        require(job.get("guardrail_script_sha256") == file_sha256(evaluator_path), "guardrail evaluator script hash mismatch")
        checkpoint_hash = file_sha256(checkpoint)
        task_by_variant = {item["name"]: item["task"] for item in manifest["variants"]}
        require(job.get("task") == task_by_variant[variant], "job task/config variant mismatch")
        mode_evidence: dict[str, Any] = {}
        for mode, path_key in (("push", "push_report_path"), ("guardrail", "guardrail_report_path")):
            path = resolve_portable_path(
                str(job[path_key]), queue_path.parent,
                repo_root=REPO_ROOT, isaaclab_root=isaaclab_root,
            )
            report = read_json(path)
            validate_eval_report(report, mode=mode, variant=variant, seed=seed, protocol_hash=protocol_hash, checkpoint_hash=checkpoint_hash)
            reports[(variant, seed, mode)] = report
            mode_evidence[mode] = {
                "path": portable_path(
                    path, queue_path.parent,
                    repo_root=REPO_ROOT, isaaclab_root=isaaclab_root,
                ),
                "sha256": file_sha256(path),
            }
        evidence_jobs.append({
            "variant": variant,
            "training_seed": seed,
            "task": job["task"],
            "normalized_cfg_difference_from_baseline": next(item["normalized_cfg_difference_from_baseline"] for item in manifest["variants"] if item["name"] == variant),
            "training_report_sha256": file_sha256(resolve_portable_path(
                str(job["report_path"]), queue_path.parent,
                repo_root=REPO_ROOT, isaaclab_root=isaaclab_root,
            )),
            "checkpoint_sha256": checkpoint_hash,
            "evaluation": mode_evidence,
        })

    recovery_by_variant: dict[str, dict[str, Any]] = {}
    guardrail_by_variant: dict[str, dict[str, Any]] = {}
    for variant in ("baseline", "push_curriculum"):
        recovery_success = recovery_total = guard_success = guard_total = 0
        push_survived = criterion_met = prepush_failures = boundary_violations = auto_reset_exclusions = 0
        raw_values = {
            key: []
            for key in (
                "tracking_error_sq_mean",
                "yaw_error_sq_mean",
                "torque_l2_mean",
                "absolute_mechanical_power_mean",
                "action_rate_l2_mean",
            )
        }
        for seed in seeds:
            push_trials = reports[(variant, seed, "push")]["trials"]
            recovery_success += sum(bool(trial["recovered"]) for trial in push_trials if trial["eligible"])
            recovery_total += sum(bool(trial["eligible"]) for trial in push_trials)
            push_survived += sum(bool(trial["survived_to_horizon"]) for trial in push_trials if trial["eligible"])
            criterion_met += sum(bool(trial["criterion_met"]) for trial in push_trials if trial["eligible"])
            prepush_failures += sum(bool(trial.get("prepush_failure")) for trial in push_trials)
            boundary_violations += sum(trial.get("excluded_reason") == "tile_boundary" for trial in push_trials)
            auto_reset_exclusions += sum(trial.get("excluded_reason") == "auto_reset_poison" for trial in push_trials)
            for trial in push_trials:
                for key in raw_values:
                    if trial.get(key) is not None:
                        raw_values[key].append(float(trial[key]))
            guard_trials = reports[(variant, seed, "guardrail")]["trials"]
            guard_success += sum(bool(trial["guardrail_survived"]) for trial in guard_trials if trial["guardrail_eligible"])
            guard_total += sum(bool(trial["guardrail_eligible"]) for trial in guard_trials)
        require(recovery_total > 0 and guard_total > 0, f"empty denominator for {variant}")
        recovery_by_variant[variant] = {
            "successes": recovery_success,
            "total": recovery_total,
            "rate": recovery_success / recovery_total,
            "wilson95": list(wilson_interval(recovery_success, recovery_total)),
            "criterion_met_count": criterion_met,
            "survived_to_horizon_count": push_survived,
            "survival_rate": push_survived / recovery_total,
            "survival_wilson95": list(wilson_interval(push_survived, recovery_total)),
            "prepush_failure_count": prepush_failures,
            "boundary_violation_count": boundary_violations,
            "auto_reset_excluded_count": auto_reset_exclusions,
            "raw_metrics": {
                key: (math.fsum(values) / len(values) if values else None)
                for key, values in raw_values.items()
            },
        }
        guardrail_by_variant[variant] = {
            "successes": guard_success,
            "total": guard_total,
            "survival_rate": guard_success / guard_total,
            "wilson95": list(wilson_interval(guard_success, guard_total)),
        }

    paired = build_paired_recovery_deltas(
        {seed: reports[("baseline", seed, "push")]["trials"] for seed in seeds},
        {seed: reports[("push_curriculum", seed, "push")]["trials"] for seed in seeds},
    )

    bootstrap = deterministic_hierarchical_paired_bootstrap(
        paired, bootstrap_seed=int(protocol.get("eval_seed", 20260824)), draws=int(protocol["bootstrap_resamples"])
    )
    recovery_delta = recovery_by_variant["push_curriculum"]["rate"] - recovery_by_variant["baseline"]["rate"]
    guardrail_delta = guardrail_by_variant["push_curriculum"]["survival_rate"] - guardrail_by_variant["baseline"]["survival_rate"]
    return {
        "schema_version": 1,
        "goal": "G006",
        "status": "complete",
        "manifest": {
            "path": portable_path(
                manifest_path, queue_path.parent,
                repo_root=REPO_ROOT, isaaclab_root=isaaclab_root,
            ),
            "sha256": file_sha256(manifest_path),
            "protocol_sha256": protocol_hash,
        },
        "queue_state": {
            "path": portable_path(
                queue_path, queue_path.parent,
                repo_root=REPO_ROOT, isaaclab_root=isaaclab_root,
            ),
        },
        "jobs": evidence_jobs,
        "comparisons": {
            "recovery": recovery_by_variant,
            "guardrail": guardrail_by_variant,
            "paired_recovery_rate_delta": recovery_delta,
            "guardrail_survival_rate_delta": guardrail_delta,
            "tradeoff": {
                "recovery_gain": recovery_delta,
                "guardrail_survival_cost": -guardrail_delta,
                "interpretation": "descriptive paired comparison only",
            },
        },
        "wilson_intervals": {"recovery": recovery_by_variant, "guardrail": guardrail_by_variant},
        "bootstrap": bootstrap,
        "warnings": [
            "Training has n=3 seeds per variant; no statistical-significance claim is made.",
            "The paired hierarchical bootstrap uses fixed 108 strata, equal stratum weight, and 10,000 resamples.",
            "Mechanical power is a simulation proxy and is not electrical energy consumption.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--queue-state", required=True, type=Path)
    parser.add_argument("--isaaclab-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = summarize(
            args.manifest.resolve(), args.queue_state.resolve(),
            isaaclab_root=args.isaaclab_root.resolve(),
        )
        write_json_atomic(args.output.resolve(), result)
        print(json.dumps({"status": "complete", "output": str(args.output.resolve())}), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "goal": "G006",
            "status": "failed",
            "error": {
                "type": type(exc).__name__,
                "message": safe_failure_message(exc),
            },
            "jobs": [],
            "comparisons": {},
            "wilson_intervals": {},
            "bootstrap": None,
            "warnings": [],
        }
        write_json_atomic(args.output.resolve(), failure)
        print(json.dumps(failure["error"]), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
