"""Run the pinned G009 trainer with diagnostic-only pre-reset attribution."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
POSE_NAMES = ("prone", "supine", "left_side", "right_side")
FIXED_TASK = "Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0"
FIXED_NUM_ENVS = 1024
FIXED_MAX_ITERATIONS = 50
FIXED_SEED = 42
FIXED_DEVICE = "cuda:0"
ROLLOUT_STEPS_PER_ITERATION = 24
DEFAULT_EVENT_CAP = 512
CALF_HYDRA_PREFIX = "env.events.reset_base.params.calf_angle="
SOURCE_PATHS = (
    "scripts/bootstrap_train_g009.py",
    "scripts/bootstrap_train_g009_rev31_attribution.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/mdp/recover.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: _sha256(REPO_ROOT / name) for name in SOURCE_PATHS}


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite attribution value: {label}={result!r}")
    return result


def _as_list(tensor: Any, label: str) -> list[float]:
    values = tensor.detach().cpu().tolist()
    if not isinstance(values, list):
        values = [values]
    return [_finite(value, f"{label}[{index}]") for index, value in enumerate(values)]


class TrainingAttributionObserver:
    """Capture hard-limit terminal state at RecorderManager's pre-reset boundary."""

    def __init__(self, env: Any, *, event_cap: int) -> None:
        if event_cap < 0:
            raise ValueError("event_cap must be non-negative")
        self.env = env
        self.event_cap = event_cap
        self.control_step = 0
        self.hard_termination_total = 0
        self.hard_termination_by_pose: Counter[str] = Counter()
        self.hard_termination_by_joint: Counter[str] = Counter()
        self.hard_termination_joint_event_total = 0
        self.samples: list[dict[str, Any]] = []
        self.sample_overflow_count = 0
        self.pose_observations: Counter[str] = Counter()
        self.error: str | None = None
        self.rng_neutral = True

        action_term = env.action_manager.get_term("joint_pos")
        self.joint_names = list(action_term._joint_names)
        raw_ids = action_term._joint_ids
        if isinstance(raw_ids, slice):
            self.joint_ids = list(range(len(env.scene["robot"].joint_names)))[raw_ids]
        else:
            self.joint_ids = list(raw_ids)
        if len(self.joint_ids) != len(self.joint_names) or not self.joint_names:
            raise RuntimeError("action joint ids/names are inconsistent")
        self._observe_poses()

    def _observe_poses(self) -> None:
        pose_ids = getattr(self.env, "_g009_recover_fall_class", None)
        if pose_ids is None:
            return
        for raw in pose_ids.detach().cpu().tolist():
            index = int(raw)
            name = POSE_NAMES[index] if 0 <= index < len(POSE_NAMES) else f"unknown:{index}"
            self.pose_observations[name] += 1

    def capture(self, env_ids: Any) -> None:
        import torch

        hard = self.env.termination_manager.get_term("hard_joint_limit")
        robot = self.env.scene["robot"]
        action_term = self.env.action_manager.get_term("joint_pos")
        sensor = self.env.scene["contact_forces"]
        pose_ids = getattr(self.env, "_g009_recover_fall_class", None)
        positions = robot.data.joint_pos[:, self.joint_ids]
        velocities = robot.data.joint_vel[:, self.joint_ids]
        limits = robot.data.joint_pos_limits[:, self.joint_ids]
        torques = robot.data.applied_torque[:, self.joint_ids]
        forces = torch.linalg.vector_norm(sensor.data.net_forces_w, dim=-1)
        body_names = list(sensor.body_names)
        foot_mask = torch.tensor(
            [name.lower().endswith("foot") for name in body_names],
            dtype=torch.bool,
            device=forces.device,
        )
        if not bool(foot_mask.any().item()) or bool(foot_mask.all().item()):
            raise RuntimeError("contact sensor must expose both foot and non-foot bodies")
        total_mass = self.env._g009_r0_body_mass.sum(dim=1)

        for raw_env_index in env_ids:
            env_index = int(raw_env_index)
            if not bool(hard[env_index].item()):
                continue
            pose_index = int(pose_ids[env_index].item()) if pose_ids is not None else -1
            pose = POSE_NAMES[pose_index] if 0 <= pose_index < len(POSE_NAMES) else f"unknown:{pose_index}"
            violations: list[dict[str, Any]] = []
            for local_index, joint_name in enumerate(self.joint_names):
                q = _finite(positions[env_index, local_index].item(), f"q.{joint_name}")
                lower = _finite(limits[env_index, local_index, 0].item(), f"lower.{joint_name}")
                upper = _finite(limits[env_index, local_index, 1].item(), f"upper.{joint_name}")
                if q < lower or q > upper:
                    limit_side = "lower" if q < lower else "upper"
                    hard_limit_excess = lower - q if limit_side == "lower" else q - upper
                    self.hard_termination_by_joint[joint_name] += 1
                    self.hard_termination_joint_event_total += 1
                    violations.append(
                        {
                            "joint_name": joint_name,
                            "joint_index": int(self.joint_ids[local_index]),
                            "limit_side": limit_side,
                            "hard_limit_excess_rad": hard_limit_excess,
                            "q_rad": q,
                            "qdot_rad_s": _finite(
                                velocities[env_index, local_index].item(), f"qdot.{joint_name}"
                            ),
                            "lower_rad": lower,
                            "upper_rad": upper,
                            "applied_torque_nm": _finite(
                                torques[env_index, local_index].item(), f"torque.{joint_name}"
                            ),
                            "processed_target_rad": _finite(
                                action_term.processed_actions[env_index, local_index].item(),
                                f"target.{joint_name}",
                            ),
                        }
                    )
            if not violations:
                raise RuntimeError(
                    f"hard termination without a joint outside hard limits: env={env_index}"
                )
            mass = _finite(total_mass[env_index].item(), "total_mass")
            if mass <= 0.0:
                raise RuntimeError(f"invalid robot mass: {mass}")
            bodyweight = mass * 9.81
            foot_max = _finite(forces[env_index, foot_mask].max().item(), "foot_force") / bodyweight
            nonfoot_max = (
                _finite(forces[env_index, ~foot_mask].max().item(), "nonfoot_force") / bodyweight
            )
            self.hard_termination_total += 1
            self.hard_termination_by_pose[pose] += 1
            sample = {
                "control_step": self.control_step,
                "iteration_index_zero_based": (self.control_step - 1) // ROLLOUT_STEPS_PER_ITERATION,
                "iteration_number_one_based": ((self.control_step - 1) // ROLLOUT_STEPS_PER_ITERATION) + 1,
                "episode_step": int(self.env.episode_length_buf[env_index].item()),
                "env_index": env_index,
                "pose": pose,
                "violations": violations,
                "root_position_w_m": _as_list(robot.data.root_pos_w[env_index], "root_position"),
                "root_quaternion_wxyz": _as_list(robot.data.root_quat_w[env_index], "root_quaternion"),
                "max_foot_net_contact_bodyweights": foot_max,
                "max_nonfoot_net_contact_bodyweights": nonfoot_max,
            }
            if len(self.samples) < self.event_cap:
                self.samples.append(sample)
            else:
                self.sample_overflow_count += 1

    def report(self) -> dict[str, Any]:
        return {
            "hard_termination_env_count": self.hard_termination_total,
            "hard_termination_joint_event_count": self.hard_termination_joint_event_total,
            "hard_termination_by_pose": dict(sorted(self.hard_termination_by_pose.items())),
            "hard_termination_by_joint": dict(sorted(self.hard_termination_by_joint.items())),
            "event_sample_cap": self.event_cap,
            "event_sample_count": len(self.samples),
            "event_sample_overflow_count": self.sample_overflow_count,
            "event_samples": self.samples,
            "pose_count_at_install": dict(sorted(self.pose_observations.items())),
            "control_steps_recorded": self.control_step,
            "observer_rng_neutral": self.rng_neutral,
            "instrumentation_error": self.error,
        }


def install_pre_reset_observer(env: Any, observer: TrainingAttributionObserver) -> tuple[Any, Any]:
    """Install fail-closed instance wrappers and return the original callables."""
    import torch

    manager = env.recorder_manager
    if len(manager.active_terms) != 0:
        raise RuntimeError("attribution requires zero active recorder terms")
    original_pre_reset = manager.record_pre_reset
    original_step = env.step

    def observed_pre_reset(env_ids: Any, force_export_or_skip: Any = None) -> Any:
        cpu_before = torch.get_rng_state().clone()
        cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
        try:
            observer.capture(env_ids)
        except Exception as error:
            observer.error = f"{type(error).__name__}: {error}"
            raise
        cpu_after = torch.get_rng_state()
        cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        neutral = bool(torch.equal(cpu_before, cpu_after)) and len(cuda_before) == len(cuda_after)
        neutral &= all(bool(torch.equal(before, after)) for before, after in zip(cuda_before, cuda_after))
        observer.rng_neutral &= neutral
        if not neutral:
            observer.error = "observer changed torch RNG state"
            raise RuntimeError(observer.error)
        return original_pre_reset(env_ids, force_export_or_skip)

    def observed_step(actions: Any) -> Any:
        observer.control_step += 1
        return original_step(actions)

    manager.record_pre_reset = observed_pre_reset
    env.step = observed_step
    return original_pre_reset, original_step


class AttributionReportWriter:
    """Write the attribution report exactly once, from whichever path runs first.

    The upstream Isaac Lab trainer calls ``simulation_app.close()`` right after
    ``main()`` returns, and that call ends the process without unwinding this
    module's ``finally`` block. The report is therefore written at ``env.close()``
    on the normal path and at ``finally`` only when an exception keeps the
    process alive.
    """

    def __init__(self, output: Path) -> None:
        self.output = output
        self.written = False
        self.trigger: str | None = None

    def write(self, build_payload: Any, *, trigger: str) -> bool:
        if self.written:
            return False
        payload = build_payload()
        payload["report_written_at"] = trigger
        _write_json_no_overwrite(self.output, payload)
        self.written = True
        self.trigger = trigger
        return True


def install_close_finalizer(env: Any, writer: AttributionReportWriter, build_payload: Any) -> Any:
    """Record the attribution report before the simulator closes the process."""
    original_close = env.close

    def observed_close(*args: Any, **kwargs: Any) -> Any:
        writer.write(build_payload, trigger="env_close")
        return original_close(*args, **kwargs)

    env.close = observed_close
    return original_close


def _write_json_no_overwrite(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite attribution report: {path}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_and_strip_custom_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attribution-output", type=Path)
    parser.add_argument("--attribution-event-cap", type=int, default=DEFAULT_EVENT_CAP)
    parser.add_argument("--calf-reset", type=float, choices=(-2.37, -2.28))
    custom, forwarded = parser.parse_known_args(argv)
    calf_tokens = [value for value in forwarded if value.startswith(CALF_HYDRA_PREFIX)]
    if calf_tokens:
        if len(calf_tokens) != 1:
            raise ValueError("duplicate calf reset Hydra override is forbidden")
        hydra_calf = float(calf_tokens[0][len(CALF_HYDRA_PREFIX) :])
        if hydra_calf not in (-2.37, -2.28):
            raise ValueError("calf reset Hydra override must be exactly -2.37 or -2.28")
        if custom.calf_reset is not None and custom.calf_reset != hydra_calf:
            raise ValueError("conflicting --calf-reset and Hydra calf reset override")
        custom.calf_reset = hydra_calf
        forwarded.remove(calf_tokens[0])
    standard = argparse.ArgumentParser(add_help=False)
    standard.add_argument("--task", required=True)
    standard.add_argument("--num_envs", required=True, type=int)
    standard.add_argument("--max_iterations", required=True, type=int)
    standard.add_argument("--seed", required=True, type=int)
    standard.add_argument("--device", required=True)
    standard.add_argument("--run_name", required=True)
    standard.add_argument("--headless", action="store_true")
    known, unknown = standard.parse_known_args(forwarded)
    if unknown:
        raise ValueError(f"rev31 attribution forbids resume/Hydra/unknown arguments: {unknown}")
    expected = (FIXED_TASK, FIXED_NUM_ENVS, FIXED_MAX_ITERATIONS, FIXED_SEED, FIXED_DEVICE, True)
    actual = (known.task, known.num_envs, known.max_iterations, known.seed, known.device, known.headless)
    if actual != expected:
        raise ValueError(f"rev31 runtime is fixed: expected={expected!r} actual={actual!r}")
    if custom.attribution_event_cap < 0:
        raise ValueError("--attribution-event-cap must be non-negative")
    if custom.attribution_output is None:
        custom.attribution_output = REPO_ROOT / "reports" / "runs" / f"{known.run_name}_attribution.json"
    custom.runtime = vars(known)
    return custom, forwarded


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    custom, forwarded = parse_and_strip_custom_args(values)
    output = custom.attribution_output.resolve()
    if output.parent != (REPO_ROOT / "reports" / "runs").resolve() or output.suffix.lower() != ".json":
        raise ValueError("attribution output must be a direct JSON child of reports/runs")
    if output.exists() or output.with_suffix(".json.tmp").exists():
        raise FileExistsError(f"refusing to overwrite attribution report: {output}")

    hashes_before = source_hashes()
    writer = AttributionReportWriter(output)
    observer: TrainingAttributionObserver | None = None
    raw_env: Any = None
    originals: tuple[Any, Any] | None = None
    actual_calf_reset: float | None = None
    actual_env_cfg: dict[str, Any] | None = None
    status = "training_failed"
    caught: BaseException | None = None
    import gymnasium as gym
    import bootstrap_train_g009

    original_make = gym.make

    def build_report(status_value: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol": "g009_r0_rev31_training_time_pre_reset_attribution_v1",
            "status": status_value,
            "diagnostic_only": True,
            "qualification_eligible": False,
            "runtime": {
                **custom.runtime,
                "rollout_steps_per_iteration": ROLLOUT_STEPS_PER_ITERATION,
                "physics_budget_env_steps": FIXED_NUM_ENVS * FIXED_MAX_ITERATIONS * ROLLOUT_STEPS_PER_ITERATION,
                "requested_calf_reset_override_rad": custom.calf_reset,
                "actual_cfg_calf_reset_rad": actual_calf_reset,
                "calf_reset_override_applied": custom.calf_reset is not None,
            },
            "actual_environment_cfg": actual_env_cfg,
            "attribution": observer.report() if observer is not None else None,
            "source_hashes_before": hashes_before,
            "source_hashes_after": source_hashes(),
            "source_hashes_stable": hashes_before == source_hashes(),
            "error": None if caught is None else f"{type(caught).__name__}: {caught}",
        }

    def instrumented_make(task: str, **kwargs: Any) -> Any:
        nonlocal observer, raw_env, originals, actual_calf_reset, actual_env_cfg
        if task != FIXED_TASK:
            raise RuntimeError(f"unexpected task passed to gym.make: {task}")
        cfg = kwargs.get("cfg")
        if custom.calf_reset is not None:
            cfg.events.reset_base.params["calf_angle"] = custom.calf_reset
        default_calf = inspect.signature(cfg.events.reset_base.func).parameters["calf_angle"].default
        actual_calf_reset = _finite(
            cfg.events.reset_base.params.get("calf_angle", default_calf), "cfg.calf_angle"
        )
        actual_env_cfg = {
            "task": task,
            "num_envs": int(cfg.scene.num_envs),
            "seed": int(cfg.seed),
            "device": str(cfg.sim.device),
            "decimation": int(cfg.decimation),
            "episode_length_s": _finite(cfg.episode_length_s, "cfg.episode_length_s"),
            "calf_reset_rad": actual_calf_reset,
        }
        raw_env = original_make(task, **kwargs)
        observer = TrainingAttributionObserver(raw_env.unwrapped, event_cap=custom.attribution_event_cap)
        originals = install_pre_reset_observer(raw_env.unwrapped, observer)
        install_close_finalizer(raw_env, writer, lambda: build_report("training_complete"))
        return raw_env

    gym.make = instrumented_make
    previous_argv = sys.argv
    sys.argv = [str(Path(__file__).resolve()), *forwarded]
    try:
        bootstrap_train_g009.main()
        status = "training_complete"
    except BaseException as error:
        caught = error
    finally:
        sys.argv = previous_argv
        gym.make = original_make
        if originals is not None and raw_env is not None:
            raw_env.unwrapped.recorder_manager.record_pre_reset = originals[0]
            raw_env.unwrapped.step = originals[1]
        writer.write(lambda: build_report(status), trigger="process_finally")
    if caught is not None:
        raise caught
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
