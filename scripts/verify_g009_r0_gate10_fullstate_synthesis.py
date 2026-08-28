#!/usr/bin/env python3
"""Verify the published rev12 Gate10 full-state three-run synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = tuple(
    REPO_ROOT
    / f"reports/runs/g009_r0_gate10_hard_limit_attribution_rev12_fullstate_gpu_rep0{index}_s42.json"
    for index in range(1, 4)
)
DEFAULT_SYNTHESIS = (
    REPO_ROOT
    / "reports/runs/g009_r0_gate10_hard_limit_attribution_rev12_fullstate_synthesis_3x3_s42.json"
)
CANONICALIZATION = (
    "json.dumps(events, ensure_ascii=False, sort_keys=True, "
    "separators=(',', ':'), allow_nan=False)"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify(report_paths: tuple[Path, ...], synthesis_path: Path) -> dict[str, Any]:
    require(len(report_paths) == 3, "exactly three full-state reports are required")
    reports = [load_json_object(path) for path in report_paths]
    synthesis = load_json_object(synthesis_path)

    event_hashes = [canonical_sha256(report.get("events")) for report in reports]
    require(len(set(event_hashes)) == 1, "full event payloads differ across repetitions")
    expected_event_hash = synthesis.get("full_event_payload_sha256")
    require(expected_event_hash == event_hashes[0], "synthesis full event payload hash mismatch")
    require(
        synthesis.get("full_event_payload_canonicalization") == CANONICALIZATION,
        "synthesis canonicalization contract mismatch",
    )

    synthesis_reports = synthesis.get("reports")
    if not isinstance(synthesis_reports, list) or len(synthesis_reports) != 3:
        raise ValueError("invalid report index")
    execution_ids: list[str] = []
    action_hashes: list[str] = []
    for path, report, indexed in zip(report_paths, reports, synthesis_reports, strict=True):
        if not isinstance(indexed, dict):
            raise ValueError("invalid indexed report")
        relative = str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
        require(indexed.get("path") == relative, f"indexed path mismatch: {relative}")
        require(indexed.get("sha256") == file_sha256(path), f"file hash mismatch: {relative}")
        execution = report.get("execution")
        if not isinstance(execution, dict):
            raise ValueError(f"missing execution metadata: {relative}")
        execution_id = execution.get("execution_id")
        if not isinstance(execution_id, str) or len(execution_id) == 0:
            raise ValueError(f"invalid execution id: {relative}")
        require(indexed.get("execution_id") == execution_id, f"execution id mismatch: {relative}")
        execution_ids.append(execution_id)

        runtime = report.get("runtime_reproduction")
        if not isinstance(runtime, dict):
            raise ValueError(f"missing runtime reproduction: {relative}")
        action_hash = runtime.get("ppo_action_stream_sha256")
        if not isinstance(action_hash, str) or len(action_hash) != 64:
            raise ValueError(f"invalid action hash: {relative}")
        action_hashes.append(action_hash)
        require(runtime.get("act_count") == 240, f"unexpected action count: {relative}")
        require(runtime.get("update_count") == 10, f"unexpected update count: {relative}")
        require(
            runtime.get("hard_joint_limit_event_counts") == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            f"unexpected hard-limit topology: {relative}",
        )
        require(report.get("outcome") == "attributed_historical_identity", f"bad outcome: {relative}")
        require(report.get("attribution_contract_passed") is True, f"attribution failed: {relative}")
        require(
            report.get("historical_trajectory_identity_confirmed") is True,
            f"historical identity failed: {relative}",
        )
        require(all(report.get("checks", {}).values()), f"attribution check failed: {relative}")
        require(
            all(report.get("historical_identity_checks", {}).values()),
            f"historical identity check failed: {relative}",
        )
        require(report.get("gate10_safety_passed") is False, f"safety status changed: {relative}")
        require(report.get("learned_policy_qualified") is False, f"qualification status changed: {relative}")

    require(len(set(execution_ids)) == 3, "execution ids are not independent")
    require(len(set(action_hashes)) == 1, "PPO action streams differ")
    require(synthesis.get("ppo_action_stream_sha256") == action_hashes[0], "action hash mismatch")
    qualification = synthesis.get("qualification")
    if not isinstance(qualification, dict):
        raise ValueError("missing synthesis qualification")
    require(qualification.get("gate10_safety_passed") is False, "synthesis safety status changed")
    require(qualification.get("learned_policy_qualified") is False, "synthesis qualification changed")
    require(qualification.get("gate50_allowed") is False, "synthesis opened Gate50")

    return {
        "status": "pass",
        "report_count": 3,
        "independent_execution_ids": True,
        "full_event_payload_sha256": event_hashes[0],
        "ppo_action_stream_sha256": action_hashes[0],
        "gate10_safety_passed": False,
        "learned_policy_qualified": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs=3, default=DEFAULT_REPORTS)
    parser.add_argument("--synthesis", type=Path, default=DEFAULT_SYNTHESIS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = verify(tuple(args.reports), args.synthesis)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
