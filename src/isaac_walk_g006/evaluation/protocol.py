"""Pure protocol primitives for deterministic G006 evaluation and statistics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

PUSH_INJECTION_COMPLETED_STEPS = 200
PUSH_STEP = PUSH_INJECTION_COMPLETED_STEPS  # compatibility alias
RECOVERY_START_STEP = 201
RECOVERY_END_STEP = 450
TOTAL_STEPS = 600
RECOVERY_DWELL_STEPS = 25
TILE_HALF_EXTENT_M = 11.5
DIFFICULTY_ROWS = (1, 4, 8)
TERRAIN_COLS = tuple(range(10))
EXPECTED_SUCCESS_CRITERIA = {
    "lin_vel_error_mps_max": 0.30,
    "yaw_rate_error_radps_max": 0.30,
    "roll_abs_rad_max": 0.35,
    "pitch_abs_rad_max": 0.35,
    "consecutive_post_push_samples": 25,
    "recovery_completed_step_start": 201,
    "recovery_completed_step_end": 450,
    "horizon_completed_step": 600,
    "push_injection_completed_steps": 200,
    "base_contact_allowed": False,
    "survival_to_horizon_required": True,
}


def validate_success_criteria(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value != EXPECTED_SUCCESS_CRITERIA:
        raise ValueError(f"success_criteria must exactly equal {EXPECTED_SUCCESS_CRITERIA}")
    return dict(value)


def compute_evaluation_source_bundle(repo_root: Path) -> dict[str, Any]:
    """Hash sorted path+NUL+raw-bytes+NUL for the fixed evaluation source closure."""

    root = repo_root.resolve()
    paths = [root / "scripts" / "evaluate_push_recovery.py"]
    paths.extend(sorted((root / "src" / "isaac_walk_g006" / "evaluation").rglob("*.py")))
    files = []
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
        files.append({"path": relative, "sha256": file_hash})
    if not files:
        raise ValueError("evaluation source bundle is empty")
    return {"sha256": digest.hexdigest(), "files": files}


def pair_id(terrain_row: int, push_direction_deg: int, push_magnitude_mps: float, initial_state: int) -> str:
    return f"r{terrain_row:02d}_d{push_direction_deg:03d}_m{push_magnitude_mps:.2f}_i{initial_state:02d}"


def build_push_trials(
    initial_states: Sequence[Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]] | None = None,
    directions: Sequence[Mapping[str, Any]] | None = None,
    magnitudes: Sequence[float] = (0.5, 1.0, 1.5),
) -> list[dict[str, Any]]:
    """Build exactly 108 strata x 10 literal initial-state repeats = 1080 trials."""

    if len(initial_states) != 10:
        raise ValueError("exactly 10 literal initial states are required")
    if commands is None:
        commands = (
            {"id": "forward", "base_velocity": [0.75, 0.0, 0.0]},
            {"id": "lateral", "base_velocity": [0.0, 0.5, 0.0]},
            {"id": "turning", "base_velocity": [0.5, 0.0, 0.5]},
        )
    if directions is None:
        directions = (
            {"id": "forward", "body_xy": [1.0, 0.0]},
            {"id": "backward", "body_xy": [-1.0, 0.0]},
            {"id": "left", "body_xy": [0.0, 1.0]},
            {"id": "right", "body_xy": [0.0, -1.0]},
        )
    if len(commands) != 3 or len(directions) != 4 or len(magnitudes) != 3:
        raise ValueError("push matrix must be 3 commands x 4 directions x 3 magnitudes")
    trials: list[dict[str, Any]] = []
    for row in DIFFICULTY_ROWS:
        for command in commands:
            for direction in directions:
                for magnitude in magnitudes:
                    stratum = f"r{row:02d}_cmd{command['id']}_dir{direction['id']}_m{float(magnitude):.2f}"
                    for repeat, initial_state in enumerate(initial_states):
                        trials.append({
                            "trial_id": f"push_{stratum}_c{repeat:02d}",
                            "pair_id": f"{stratum}_c{repeat:02d}",
                            "stratum_id": stratum,
                            "terrain_row": row,
                            "terrain_col": repeat,
                            "command_id": command["id"],
                            "command": list(command["base_velocity"]),
                            "push_direction_id": direction["id"],
                            "push_direction_body_xy": list(direction["body_xy"]),
                            "push_magnitude_mps": float(magnitude),
                            "initial_state_id": repeat,
                            "initial_state": dict(initial_state),
                        })
    if len(trials) != 1080 or len({trial["trial_id"] for trial in trials}) != 1080:
        raise AssertionError("push trial cardinality contract violated")
    return trials


def build_guardrail_trials(
    initial_states: Sequence[Mapping[str, Any]], commands: Sequence[Mapping[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Build 90 no-push trials: three terrain strata x three command cases x ten repeats."""

    if len(initial_states) != 10:
        raise ValueError("exactly 10 literal initial states are required")
    if commands is None:
        commands = (
            {"id": "forward", "base_velocity": [0.75, 0.0, 0.0]},
            {"id": "lateral", "base_velocity": [0.0, 0.5, 0.0]},
            {"id": "turning", "base_velocity": [0.5, 0.0, 0.5]},
        )
    if len(commands) != 3:
        raise ValueError("guardrail matrix requires three commands")
    trials: list[dict[str, Any]] = []
    for row in DIFFICULTY_ROWS:
        for command in commands:
            for repeat, initial_state in enumerate(initial_states):
                trials.append({
                    "trial_id": f"guard_r{row:02d}_cmd{command['id']}_c{repeat:02d}",
                    "stratum_id": f"r{row:02d}_cmd{command['id']}",
                    "terrain_row": row,
                    "terrain_col": repeat,
                    "command_id": command["id"],
                    "command": list(command["base_velocity"]),
                    "initial_state_id": repeat,
                    "initial_state": dict(initial_state),
                })
    if len(trials) != 90:
        raise AssertionError("guardrail trial cardinality contract violated")
    return trials


def yaw_from_wxyz(quaternion: Sequence[float]) -> float:
    if len(quaternion) != 4:
        raise ValueError("quaternion must be wxyz")
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def body_xy_to_world(delta_v_body: Sequence[float], root_quaternion_wxyz: Sequence[float]) -> tuple[float, float, float]:
    """Rotate a body-frame XY velocity impulse by yaw only."""

    if len(delta_v_body) != 2:
        raise ValueError("delta_v_body must contain x and y")
    yaw = yaw_from_wxyz(root_quaternion_wxyz)
    x, y = (float(value) for value in delta_v_body)
    return (math.cos(yaw) * x - math.sin(yaw) * y, math.sin(yaw) * x + math.cos(yaw) * y, 0.0)


def tile_boundary_violation(position_xy: Sequence[float], terrain_origin_xy: Sequence[float]) -> bool:
    if len(position_xy) != 2 or len(terrain_origin_xy) != 2:
        raise ValueError("position and origin must be XY")
    return any(abs(float(position_xy[index]) - float(terrain_origin_xy[index])) > TILE_HALF_EXTENT_M for index in range(2))


@dataclass
class PushRecoveryStateMachine:
    """Per-trial state machine immune to termination auto-reset poisoning."""

    tracking_error_limit: float
    angular_error_limit: float
    roll_limit: float
    pitch_limit: float
    dwell_required: int = RECOVERY_DWELL_STEPS
    push_enabled: bool = True
    active: bool = True
    pushed: bool = False
    eligible: bool = False
    criterion_met: bool = False
    recovered: bool = False
    failed: bool = False
    survived_to_horizon: bool = False
    protocol_blocked: bool = False
    excluded_reason: str | None = None
    dwell: int = 0
    recovery_step: int | None = None
    prepush_failure: bool = False
    last_completed_step: int = 0
    sample_steps: list[int] = field(default_factory=list)

    def mark_push(self, completed_steps: int) -> None:
        """Mark the pre-step injection performed after exactly 200 completed steps."""

        if not self.push_enabled or not self.active:
            return
        if completed_steps != PUSH_INJECTION_COMPLETED_STEPS or self.last_completed_step != completed_steps:
            raise ValueError("push must be injected at completed_steps=200")
        self.pushed = True
        self.eligible = True

    def observe(
        self,
        completed_step: int,
        *,
        tracking_error: float = 0.0,
        angular_error: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        terminated: bool = False,
        auto_reset_detected: bool = False,
        boundary_violation: bool = False,
    ) -> None:
        if completed_step != self.last_completed_step + 1:
            raise ValueError("completed steps must be contiguous and one-based")
        if not 1 <= completed_step <= TOTAL_STEPS:
            raise ValueError("completed step outside protocol horizon")
        self.last_completed_step = completed_step
        if not self.active:
            return
        if auto_reset_detected:
            self.active = False
            self.protocol_blocked = True
            self.excluded_reason = "auto_reset_poison"
            return
        if boundary_violation:
            self.active = False
            self.protocol_blocked = True
            self.failed = self.pushed
            self.excluded_reason = "tile_boundary"
            return
        if terminated:
            self.active = False
            if completed_step <= PUSH_INJECTION_COMPLETED_STEPS:
                self.prepush_failure = True
                self.eligible = True
                self.failed = True
            else:
                self.failed = True
            return
        if RECOVERY_START_STEP <= completed_step <= RECOVERY_END_STEP and self.pushed and not self.criterion_met:
            self.sample_steps.append(completed_step)
            in_band = (
                tracking_error <= self.tracking_error_limit
                and angular_error <= self.angular_error_limit
                and abs(roll) <= self.roll_limit
                and abs(pitch) <= self.pitch_limit
            )
            self.dwell = self.dwell + 1 if in_band else 0
            if self.dwell >= self.dwell_required:
                self.criterion_met = True
                self.recovery_step = completed_step
        if completed_step == TOTAL_STEPS:
            self.survived_to_horizon = True
            self.recovered = self.criterion_met
            self.failed = not self.recovered if self.push_enabled else False
            self.active = False

    def finalize(self) -> dict[str, Any]:
        self.recovered = self.criterion_met and self.survived_to_horizon and not self.protocol_blocked
        if self.push_enabled and self.eligible:
            self.failed = not self.recovered
        return {
            "eligible": self.eligible,
            "criterion_met": self.criterion_met,
            "recovered": self.recovered,
            "failed": self.failed,
            "recovery_failed": self.eligible and not self.criterion_met,
            "survived_to_horizon": self.survived_to_horizon,
            "physical_failure": self.eligible and not self.survived_to_horizon,
            "protocol_blocked": self.protocol_blocked,
            "prepush_failure": self.prepush_failure,
            "excluded_reason": self.excluded_reason,
            "recovery_step": self.recovery_step,
            "recovery_sample_count": len(self.sample_steps),
        }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    if total == 0:
        return (None, None)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def deterministic_hierarchical_paired_bootstrap(
    paired_deltas: Mapping[int, Mapping[str, Sequence[float]]],
    *,
    bootstrap_seed: int,
    draws: int = 10_000,
) -> dict[str, Any]:
    """Seed→fixed-stratum→repeat paired bootstrap with equal stratum weights."""

    if draws <= 0 or not paired_deltas:
        raise ValueError("bootstrap requires positive draws and paired data")
    seeds = sorted(paired_deltas)
    strata = sorted(next(iter(paired_deltas.values())))
    if len(strata) != 108:
        raise ValueError("exactly 108 fixed strata are required")
    for seed in seeds:
        if sorted(paired_deltas[seed]) != strata:
            raise ValueError("strata mismatch across seeds")
        if any(not paired_deltas[seed][stratum] for stratum in strata):
            raise ValueError("each stratum requires paired repeats")
    repeat_counts = {len(paired_deltas[seed][stratum]) for seed in seeds for stratum in strata}
    if len(repeat_counts) != 1:
        raise ValueError("all seed/stratum cells must have equal repeat counts")
    repeat_count = repeat_counts.pop()
    data = np.asarray(
        [[[float(value) for value in paired_deltas[seed][stratum]] for stratum in strata] for seed in seeds],
        dtype=np.float64,
    )
    rng = np.random.default_rng(np.random.SeedSequence([int(bootstrap_seed), 108, draws]))
    estimates = np.empty(draws, dtype=np.float64)
    # Chunked vectorization keeps memory bounded while retaining the exact
    # seed→fixed-stratum→repeat hierarchy and equal stratum weights.
    chunk_size = 250
    for start in range(0, draws, chunk_size):
        stop = min(draws, start + chunk_size)
        count = stop - start
        sampled_seed_indices = rng.integers(0, len(seeds), size=(count, len(seeds)))
        sampled_data = data[sampled_seed_indices]
        repeat_indices = rng.integers(
            0,
            repeat_count,
            size=(count, len(seeds), len(strata), repeat_count),
        )
        sampled_repeats = np.take_along_axis(sampled_data, repeat_indices, axis=3)
        estimates[start:stop] = sampled_repeats.mean(axis=3).mean(axis=2).mean(axis=1)
    return {
        "seed": int(bootstrap_seed),
        "draws": draws,
        "estimate": float(estimates.mean()),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "samples_sha256": hashlib.sha256(estimates.tobytes()).hexdigest(),
    }
