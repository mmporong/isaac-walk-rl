#!/usr/bin/env python3
"""Attribute rev12 Gate10 hard-limit events across all ten PPO updates.

This is an instrumented, scratch reproduction of the canonical Gate10 run.  It
uses the official ``OnPolicyRunner.learn`` loop and calls the original PPO
``update`` method ten times.  No checkpoint is loaded.  Historical trajectory
identity is reported only when the hard-limit series, generated checkpoint
hashes, action/update counts, and training-core source bundle all match the
bound Gate10 evidence exactly.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
EXPECTED_SEED = 42
EXPECTED_NUM_ENVS = 1024
EXPECTED_ROLLOUT_STEPS = 24
EXPECTED_ITERATIONS = 10
EXPECTED_ACT_COUNT = 240
EXPECTED_UPDATE_COUNT = 10
EXPECTED_DEVICE = "cuda:0"
EXPECTED_CONTACT_HISTORY_LENGTH = 3
RING_BUFFER_STEPS = 16
ACTION_MODE = "stochastic_ppo_train"
PROTOCOL_ID = "g009_r0_rev12_gate10_full_update_pre_reset_attribution_v1"
EXPECTED_TRAINING_CORE_SHA256 = "2471c64c7fa107005c199ce8c27f42d4e9782b59452c4376e7ca981125aafffa"
EXPECTED_MODEL_0_SHA256 = "52f45ef5ae9d3c98ced51132e7fb6b5e8d78d0721a7efd9657f3fdc46ea17017"
EXPECTED_MODEL_9_SHA256 = "b4bf026c446a72072ddf464aef8e5b3275b4d3f1cb1ad8980718139de2702cd2"
EXPECTED_HARD_EVENT_COUNTS = (0, 1, 1, 1, 0, 0, 0, 0, 0, 0)
EXPECTED_HARD_SERIES = tuple(count / 24.0 for count in EXPECTED_HARD_EVENT_COUNTS)
EXPECTED_BOUND_TENSORBOARD_HARD_SERIES = (
    0.0,
    0.0416666679084301,
    0.0416666679084301,
    0.0416666679084301,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
HARD_LIMIT_TAG = "Episode_Termination/hard_joint_limit"
POSE_NAMES = ("prone", "supine", "left_side", "right_side")
TRAINING_CORE_SHA256 = {
    "configs/g009_r0.json": "87355a5e927b2025d9cead696bf3be1a0fd07cad53ca1cd1e9b7867b3d01eaac",
    "scripts/bootstrap_train_g009.py": "4d5cc0776bc97a075854e7bfb801992ce0ea3a3f0e330f37c4f52edace669a58",
    "scripts/run_training.ps1": "b2910517adb83ac402eeda9389a67a844663d578247fae006483efa12b67a533",
    "src/isaac_walk_g009/agent_cfg.py": "0a1ab242e4c13023b72b319f3d0ff461131b12c8745596aadbc40a2e4d39b79b",
    "src/isaac_walk_g009/mdp/__init__.py": "c5e13ec110a341e974cd303d8e44bfadb696a97ea2db2185d7a01fece74ccfd5",
    "src/isaac_walk_g009/mdp/events.py": "642c34b0931c45ec0533d03c19f6e9fe574b4cf47cb79414b484ba6b1b3e4ea8",
    "src/isaac_walk_g009/mdp/recover.py": "a825707a076d02fb597f2e6a779170ad34dac54037c9ef535e9f54fbfc4f3599",
    "src/isaac_walk_g009/recover_contracts.py": "26eabe7661e980caae7684e5383e6da88d3b183b657d8d9a2d8482f052ced360",
    "src/isaac_walk_g009/recover_env_cfg.py": "dadf6ff10f20aebd216eb0a004e51aab119b38ef61e2c782d7bc23ad6a14d73a",
    "src/isaac_walk_g009/registry.py": "82b9f0d85bf3027789d7308b210e0505a3b5ab14ee38cfe155a206f303ff8d93",
}
DIAGNOSTIC_SOURCE_PATHS = (
    "scripts/attribute_g009_r0_gate10.py",
    "tests/test_g009_r0_gate10_attribution.py",
)
EXPECTED_OFFICIAL_RUNTIME_SOURCE_SHA256 = {
    "isaaclab_manager_based_rl_env": "d699f2d0467e851cdfb16b5c8e7aab3eb0d00ba7f1f13ba54d52f5e459743485",
    "isaaclab_recorder_manager": "06d1d419e4415a742d9ad714fe7406d115c7ddab2da35b724d63b05f8aed167a",
    "isaaclab_rsl_rl_vecenv_wrapper": "df54debea2947b1905a0b25db80eb5344d0cf27b136d98378c09af8a87e34869",
    "rsl_rl_on_policy_runner": "5ae9e87aabd62ecc9cf1ffdced0866875e0613f396af69877daaf517f7d2e33d",
    "rsl_rl_ppo": "efc1acf07745562d610c17ed28297e188c68101f3c38c68a0b029f68f5467cdc",
    "rsl_rl_actor_critic": "51ce8c6504a7434f86e5188ccb11e9fc981e46f764c00e32be3e8febe7112ae2",
    "rsl_rl_rollout_storage": "456450bbba03f9f5818f9ec43d2f3bd5d14d2d27f4bf2308838dd580f32ffcf0",
    "isaaclab_upstream_train": "8b995f75ac57ce7403973ff1f3f2715fbff9563ef2cdcdc321a7edc5dd15f5df",
    "isaaclab_ema_action": "fa560606fcb53d796dfc4bd4c45b0379381af5828bd58433a17cadc5d7d979be",
    "isaaclab_seed_source": "236f808e7792dcdba31ec37dd1c9ce18be14a9531a84a4648d34341d8ce8cdec",
    "isaacsim_torch_set_seed": "686be6e545b36cf4c51a9b6e7f9b05f177b6c93af7fdc3f7d710c7a24fde75e6",
}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite report or temporary file: {path}")
    created = False
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if created and temporary.exists():
            temporary.unlink()


def canonical_report_output(output: Path) -> tuple[Path, str]:
    reports_root = (REPO_ROOT / "reports" / "runs").resolve()
    resolved = output.expanduser().resolve()
    if resolved.parent != reports_root or resolved.suffix != ".json" or resolved.name == ".json":
        raise ValueError("output must be a direct-child JSON in canonical reports/runs")
    if resolved.exists() or resolved.with_suffix(".json.tmp").exists():
        raise FileExistsError(f"refusing to overwrite existing report: {resolved}")
    return resolved, resolved.relative_to(REPO_ROOT.resolve()).as_posix()


def prepare_execution(output: Path) -> tuple[Path, dict[str, Any]]:
    resolved, relative = canonical_report_output(output)
    return resolved, {
        "execution_id": uuid.uuid4().hex,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "output_path_repo_relative": relative,
        "no_overwrite": True,
    }


def parse_prelaunch_output(argv: list[str] | None = None) -> Path:
    values = list(sys.argv[1:] if argv is None else argv)
    if "-h" in values or "--help" in values:
        return REPO_ROOT / "reports" / "runs" / "_gate10_attribution_help_only.json"
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True, type=Path)
    args, _ = parser.parse_known_args(values)
    return args.output


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    home = str(Path.home().resolve())
    resolved = str(path.resolve())
    return "%USERPROFILE%" + resolved[len(home) :] if resolved.lower().startswith(home.lower()) else resolved


def _expand_user_path(value: str) -> Path:
    return Path(value.replace("%USERPROFILE%", str(Path.home()))).expanduser().resolve()


def training_core_provenance() -> dict[str, Any]:
    actual = {
        relative: _file_sha256(REPO_ROOT / relative)
        for relative in TRAINING_CORE_SHA256
        if (REPO_ROOT / relative).is_file()
    }
    payload = "\n".join(f"{name}:{actual[name]}" for name in sorted(actual))
    aggregate = hashlib.sha256(payload.encode("utf-8")).hexdigest() if actual else None
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *TRAINING_CORE_SHA256],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "expected_files": TRAINING_CORE_SHA256,
        "actual_files": actual,
        "expected_sha256": EXPECTED_TRAINING_CORE_SHA256,
        "sha256": aggregate,
        "exact_match": actual == TRAINING_CORE_SHA256 and aggregate == EXPECTED_TRAINING_CORE_SHA256,
        "clean": not status,
        "dirty_paths": status,
    }


def diagnostic_source_provenance() -> dict[str, Any]:
    files = {relative: _file_sha256(REPO_ROOT / relative) for relative in DIAGNOSTIC_SOURCE_PATHS}
    payload = "\n".join(f"{name}:{files[name]}" for name in sorted(files))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *DIAGNOSTIC_SOURCE_PATHS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "git_commit": commit,
        "git_commit_valid": bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
        "files": files,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "clean": not status,
        "dirty_paths": status,
    }


def official_runtime_source_provenance() -> dict[str, str]:
    home = Path.home()
    paths = {
        "isaaclab_manager_based_rl_env": home / "IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
        "isaaclab_recorder_manager": home / "IsaacLab/source/isaaclab/isaaclab/managers/recorder_manager.py",
        "isaaclab_rsl_rl_vecenv_wrapper": home / "IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py",
        "rsl_rl_on_policy_runner": home / "IsaacLab/_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
        "rsl_rl_ppo": home / "IsaacLab/_isaac_sim/kit/python/Lib/site-packages/rsl_rl/algorithms/ppo.py",
        "rsl_rl_actor_critic": home / "IsaacLab/_isaac_sim/kit/python/Lib/site-packages/rsl_rl/modules/actor_critic.py",
        "rsl_rl_rollout_storage": home / "IsaacLab/_isaac_sim/kit/python/Lib/site-packages/rsl_rl/storage/rollout_storage.py",
        "isaaclab_upstream_train": home / "IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py",
        "isaaclab_ema_action": home / "IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions_to_limits.py",
        "isaaclab_seed_source": home / "IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py",
        "isaacsim_torch_set_seed": home / "IsaacLab/_isaac_sim/exts/isaacsim.core.utils/isaacsim/core/utils/torch/maths.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"official runtime source missing: {missing}")
    return {name: _file_sha256(path) for name, path in paths.items()}


def validate_training_report(path: Path) -> dict[str, Any]:
    reports_root = (REPO_ROOT / "reports" / "runs").resolve()
    resolved = path.expanduser().resolve()
    if resolved.parent != reports_root or not resolved.is_file() or resolved.suffix != ".json":
        raise ValueError("training report must be an existing direct-child reports/runs JSON")
    report = json.loads(resolved.read_text(encoding="utf-8"))
    run_name = report.get("run_name")
    if not isinstance(run_name, str) or not re.fullmatch(
        r"go2_flat_recover_rev12_prone_gate10_s42_\d{8}-\d{6}", run_name
    ):
        raise ValueError("training report is not the canonical rev12 prone Gate10 run")
    expected = {
        "task": DEFAULT_TASK,
        "num_envs": EXPECTED_NUM_ENVS,
        "max_iterations": EXPECTED_ITERATIONS,
        "seed": EXPECTED_SEED,
        "headless": True,
        "exit_code": 0,
        "last_iteration": 9,
        "iteration_target": EXPECTED_ITERATIONS,
        "run_health_passed": True,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("training report protocol/completion contract mismatch")
    if report.get("resume") != {"enabled": False, "load_run": None, "checkpoint": None}:
        raise ValueError("Gate10 must be a scratch run without resume")
    if report.get("effective_hydra_overrides") != []:
        raise ValueError("Gate10 report must not contain Hydra overrides")
    qualification = report.get("qualification_mode", {})
    if qualification.get("enabled") is not False or qualification.get("policy_qualification_status") != "not_run":
        raise ValueError("Gate10 attribution is not policy qualification")
    repository = report.get("repository", {})
    if repository.get("dirty") is not False or not re.fullmatch(r"[0-9a-f]{40}", str(repository.get("commit", ""))):
        raise ValueError("Gate10 report must bind a clean full source commit")
    source = report.get("source_bundle", {})
    if (
        source.get("files") != TRAINING_CORE_SHA256
        or source.get("sha256") != EXPECTED_TRAINING_CORE_SHA256
        or source.get("matches_repository_commit") is not True
    ):
        raise ValueError("bound Gate10 training source bundle mismatch")
    core = training_core_provenance()
    if not core["exact_match"]:
        raise ValueError("current training core no longer matches original Gate10")
    artifact_dir = _expand_user_path(report["artifacts"]["tensorboard_directory"])
    bound_hard_series = load_bound_hard_series(artifact_dir, report)
    model_paths = {0: artifact_dir / "model_0.pt", 9: artifact_dir / "model_9.pt"}
    model_hashes = {index: _file_sha256(model_paths[index]) for index in model_paths if model_paths[index].is_file()}
    if model_hashes != {0: EXPECTED_MODEL_0_SHA256, 9: EXPECTED_MODEL_9_SHA256}:
        raise ValueError("bound Gate10 model_0/model_9 checkpoint hash mismatch")
    series = report.get("tensorboard", {}).get("series_summary", {}).get(
        "Episode_Termination/hard_joint_limit", {}
    )
    if series.get("sample_count") != 10 or series.get("nonzero_sample_count") != 3:
        raise ValueError("bound Gate10 hard-limit summary mismatch")
    return {
        "path": resolved.relative_to(REPO_ROOT).as_posix(),
        "sha256": _file_sha256(resolved),
        "run_name": run_name,
        "training_commit": report["repository"]["commit"],
        "training_core_sha256": source["sha256"],
        "hard_joint_limit_series": bound_hard_series,
        "model_0": {"path": _portable_path(model_paths[0]), "sha256": model_hashes[0], "loaded": False},
        "model_9": {"path": _portable_path(model_paths[9]), "sha256": model_hashes[9], "loaded": False},
    }


def _scalar_summary(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("TensorBoard hard-limit series is empty or non-finite")
    return {
        "sample_count": len(values),
        "latest": values[-1],
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
        "nonzero_sample_count": sum(abs(value) > 1.0e-12 for value in values),
    }


def validate_bound_hard_series(samples: list[dict[str, Any]], stored_summary: dict[str, Any]) -> list[float]:
    if [sample.get("step") for sample in samples] != list(range(EXPECTED_ITERATIONS)):
        raise ValueError("bound Gate10 hard-limit TensorBoard steps are not exactly 0..9")
    values = [float(sample["value"]) for sample in samples]
    if len(values) != len(EXPECTED_BOUND_TENSORBOARD_HARD_SERIES) or not all(
        actual == expected for actual, expected in zip(values, EXPECTED_BOUND_TENSORBOARD_HARD_SERIES)
    ):
        raise ValueError("bound Gate10 hard-limit TensorBoard value order mismatch")
    recomputed = _scalar_summary(values)
    for field in ("sample_count", "nonzero_sample_count"):
        if recomputed[field] != stored_summary.get(field):
            raise ValueError(f"bound Gate10 stored hard-limit {field} mismatch")
    for field in ("latest", "minimum", "maximum", "mean"):
        stored_value = stored_summary.get(field)
        if not isinstance(stored_value, (int, float)) or isinstance(stored_value, bool):
            raise ValueError(f"bound Gate10 stored hard-limit {field} is missing or non-numeric")
        if not math.isclose(float(recomputed[field]), float(stored_value), rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ValueError(f"bound Gate10 stored hard-limit {field} mismatch")
    return values


def load_bound_hard_series(tensorboard_dir: Path, report: dict[str, Any]) -> list[float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - Isaac bundled Python supplies TensorBoard
        raise RuntimeError("TensorBoard is required to bind the original Gate10 series") from error
    if not tensorboard_dir.is_dir():
        raise ValueError("bound Gate10 TensorBoard directory is missing")
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if HARD_LIMIT_TAG not in set(accumulator.Tags().get("scalars", [])):
        raise ValueError("bound Gate10 TensorBoard hard-limit tag is missing")
    samples = [
        {"step": int(item.step), "wall_time": float(item.wall_time), "value": float(item.value)}
        for item in accumulator.Scalars(HARD_LIMIT_TAG)
    ]
    stored = report.get("tensorboard", {}).get("series_summary", {}).get(HARD_LIMIT_TAG, {})
    return validate_bound_hard_series(samples, stored)


def validate_protocol_args(args: argparse.Namespace) -> None:
    actual = (
        args.task,
        args.seed,
        args.num_envs,
        args.rollout_steps,
        args.iterations,
        args.headless,
        args.device,
    )
    expected = (
        DEFAULT_TASK,
        EXPECTED_SEED,
        EXPECTED_NUM_ENVS,
        EXPECTED_ROLLOUT_STEPS,
        EXPECTED_ITERATIONS,
        True,
        EXPECTED_DEVICE,
    )
    if actual != expected:
        raise ValueError("protocol requires seed42, cuda:0, headless scratch, 1024 env, 24 steps x 10 iterations")


def _tensor_bytes(tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def torch_rng_state_sha256(torch_module) -> str:
    digest = hashlib.sha256(_tensor_bytes(torch_module.get_rng_state()))
    if torch_module.cuda.is_available():
        for state in torch_module.cuda.get_rng_state_all():
            digest.update(_tensor_bytes(state))
    return digest.hexdigest()


def torch_rng_component_sha256(torch_module) -> dict[str, Any]:
    return {
        "cpu": hashlib.sha256(_tensor_bytes(torch_module.get_rng_state())).hexdigest(),
        "cuda": [
            hashlib.sha256(_tensor_bytes(state)).hexdigest()
            for state in (torch_module.cuda.get_rng_state_all() if torch_module.cuda.is_available() else [])
        ],
    }


def _hash_state(value: Any, digest) -> None:
    if hasattr(value, "detach") and hasattr(value, "dtype"):
        digest.update(b"tensor")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: repr(item)):
            digest.update(repr(key).encode("utf-8"))
            _hash_state(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _hash_state(item, digest)
    else:
        digest.update(repr(value).encode("utf-8"))


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_state(value, digest)
    return digest.hexdigest()


def policy_state_sha256(policy) -> str:
    return state_sha256(policy.state_dict())


def optimizer_state_sha256(optimizer) -> str:
    return state_sha256(optimizer.state_dict())


def joint_limit_attributions(
    *, position: list[float], lower: list[float], upper: list[float], joint_names: list[str], margin_rad: float
) -> list[dict[str, Any]]:
    count = len(joint_names)
    if not (count == len(position) == len(lower) == len(upper) and count > 0):
        raise ValueError("joint vectors must have one non-empty identical dimension")
    if not math.isfinite(margin_rad) or margin_rad < 0.0:
        raise ValueError("margin must be finite and non-negative")
    records = []
    for index, (actual, lo, hi, name) in enumerate(zip(position, lower, upper, joint_names)):
        if not name or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (actual, lo, hi)):
            raise ValueError("joint state must be finite")
        if lo >= hi:
            raise ValueError("joint lower limit must be below upper limit")
        if actual < lo - margin_rad or actual > hi + margin_rad:
            side = "lower" if actual < lo - margin_rad else "upper"
            raw = lo - actual if side == "lower" else actual - hi
            records.append(
                {
                    "joint_index": index,
                    "joint_name": name,
                    "actual_position_rad": float(actual),
                    "lower_limit_rad": float(lo),
                    "upper_limit_rad": float(hi),
                    "violated_side": side,
                    "raw_excess_rad": float(raw),
                    "margin_excess_rad": float(raw - margin_rad),
                    "predicate_recomputed": True,
                }
            )
    return records


def leg_chain_relation(joint_name: str, body_name: str) -> str:
    match = re.match(r"^(FL|FR|RL|RR)_", joint_name)
    if body_name == "base":
        return "base"
    if match and body_name.startswith(match.group(1) + "_"):
        return "same_leg_chain"
    if re.match(r"^(FL|FR|RL|RR)_", body_name):
        return "other_leg_chain"
    return "unclassified"


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return False


def _close_vectors(left: list[float], right: list[float], tolerance: float = 1.0e-6) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance) for a, b in zip(left, right)
    )


def validate_attribution_result(
    *, events: list[dict[str, Any]], observed_termination_keys, margin_rad: float
) -> dict[str, bool]:
    observed = Counter(observed_termination_keys)
    attributed: list[tuple[int, int, int]] = []
    valid = True
    clamp_valid = True
    ema_valid = True
    ring_valid = True
    for event in events:
        try:
            key = (int(event["iteration"]), int(event["rollout_control_step"]), int(event["env_index"]))
            attributed.append(key)
            expected_records = joint_limit_attributions(
                position=event["joint_position_rad"],
                lower=event["joint_lower_limit_rad"],
                upper=event["joint_upper_limit_rad"],
                joint_names=event["joint_names"],
                margin_rad=margin_rad,
            )
            names = event["joint_names"]
            joint_count = len(names)
            clamp_expected = [max(-1.0, min(1.0, float(v))) for v in event["ppo_sample_pre_wrapper_clip"]]
            clamp_ok = _close_vectors(event["action_term_raw_post_wrapper_clip"], clamp_expected, 1.0e-7)
            scale = float(event["action_scale"])
            alpha = float(event["ema_alpha"])
            soft_lower = event["joint_soft_lower_limit_rad"]
            soft_upper = event["joint_soft_upper_limit_rad"]
            pre_ema_expected = [
                0.5 * (max(-1.0, min(1.0, raw * scale)) + 1.0) * (hi - lo) + lo
                for raw, lo, hi in zip(clamp_expected, soft_lower, soft_upper)
            ]
            ema_expected = [
                max(lo, min(hi, alpha * current + (1.0 - alpha) * previous))
                for current, previous, lo, hi in zip(
                    pre_ema_expected, event["ema_previous_target_rad"], soft_lower, soft_upper
                )
            ]
            ema_ok = (
                scale == 0.70
                and alpha == 0.2
                and _close_vectors(event["pre_ema_scaled_target_rad"], pre_ema_expected)
                and _close_vectors(event["ema_expected_target_rad"], ema_expected)
                and _close_vectors(event["processed_ema_target_rad"], ema_expected)
            )
            history = event["contact_sensor_history"]
            ring = event["preceding_control_step_ring"]
            global_steps = [int(frame["global_action_step"]) for frame in ring]
            primary_joint = expected_records[0]["joint_name"] if expected_records else ""
            terminal_frame = ring[-1] if ring else {}
            terminal_matches = bool(ring) and (
                terminal_frame.get("episode_control_step") == event["episode_control_step"]
                and terminal_frame.get("sim_step_counter") == event["sim_step_counter"]
                and _close_vectors(terminal_frame.get("action_post_wrapper_clip", []), event["action_term_raw_post_wrapper_clip"])
                and _close_vectors(terminal_frame.get("processed_ema_target_rad", []), event["processed_ema_target_rad"])
                and _close_vectors(terminal_frame.get("applied_torque_nm", []), event["applied_torque_nm"])
                and _close_vectors(terminal_frame.get("joint_position_rad", []), event["joint_position_rad"])
                and _close_vectors(terminal_frame.get("joint_velocity_rad_s", []), event["joint_velocity_rad_s"])
                and _close_vectors(terminal_frame.get("root_pose", {}).get("position_w_m", []), event["root_pose"]["position_w_m"])
                and _close_vectors(
                    terminal_frame.get("root_pose", {}).get("quaternion_wxyz", []),
                    event["root_pose"]["quaternion_wxyz"],
                )
                and _close_vectors(
                    terminal_frame.get("root_twist", {}).get("linear_velocity_w_m_s", []),
                    event["root_twist"]["linear_velocity_w_m_s"],
                )
                and _close_vectors(
                    terminal_frame.get("root_twist", {}).get("angular_velocity_w_rad_s", []),
                    event["root_twist"]["angular_velocity_w_rad_s"],
                )
            )
            ring_ok = (
                len(ring) == min(RING_BUFFER_STEPS, int(event["global_action_step"]))
                and global_steps == list(range(event["global_action_step"] - len(ring) + 1, event["global_action_step"] + 1))
                and all(
                    frame["iteration"] == (frame["global_action_step"] - 1) // EXPECTED_ROLLOUT_STEPS
                    and frame["rollout_control_step"] == (frame["global_action_step"] - 1) % EXPECTED_ROLLOUT_STEPS + 1
                    and frame["env_index"] == event["env_index"]
                    and len(frame["action_post_wrapper_clip"]) == joint_count
                    and len(frame["processed_ema_target_rad"]) == joint_count
                    and len(frame["applied_torque_nm"]) == joint_count
                    and len(frame["joint_position_rad"]) == joint_count
                    and len(frame["joint_velocity_rad_s"]) == joint_count
                    and len(frame["root_pose"]["position_w_m"]) == 3
                    and len(frame["root_pose"]["quaternion_wxyz"]) == 4
                    and len(frame["root_twist"]["linear_velocity_w_m_s"]) == 3
                    and len(frame["root_twist"]["angular_velocity_w_rad_s"]) == 3
                    and frame["phase"]
                    == ("terminal_pre_reset" if index == len(ring) - 1 else "post_step_after_manager_reset_for_terminated_envs")
                    and frame["body_force_summary"]["body_names"] == history["body_names"]
                    and len(frame["body_force_summary"]["body_names"]) == len(frame["body_force_summary"]["net_forces_w_n"])
                    and all(len(vector) == 3 for vector in frame["body_force_summary"]["net_forces_w_n"])
                    and _body_force_summary_valid(frame["body_force_summary"], primary_joint)
                    for index, frame in enumerate(ring)
                )
                and terminal_matches
            )
            contact_shape_ok = (
                history["sensor_name"] == "contact_forces"
                and history["history_length"] == EXPECTED_CONTACT_HISTORY_LENGTH
                and len(history["net_forces_w_history_n"]) == EXPECTED_CONTACT_HISTORY_LENGTH
                and bool(history["body_names"])
                and len(set(history["body_names"])) == len(history["body_names"])
                and all(
                    len(sample) == len(history["body_names"]) and all(len(vector) == 3 for vector in sample)
                    for sample in history["net_forces_w_history_n"]
                )
            )
            vector_names = (
                "joint_position_rad",
                "joint_lower_limit_rad",
                "joint_upper_limit_rad",
                "joint_soft_lower_limit_rad",
                "joint_soft_upper_limit_rad",
                "ppo_sample_pre_wrapper_clip",
                "action_term_raw_post_wrapper_clip",
                "pre_ema_scaled_target_rad",
                "ema_previous_target_rad",
                "ema_expected_target_rad",
                "processed_ema_target_rad",
                "joint_velocity_rad_s",
                "applied_torque_nm",
            )
            shapes_ok = (
                joint_count == 12
                and all(len(event[name]) == joint_count for name in vector_names)
                and len(event["root_pose"]["position_w_m"]) == 3
                and len(event["root_pose"]["quaternion_wxyz"]) == 4
                and len(event["root_twist"]["linear_velocity_w_m_s"]) == 3
                and len(event["root_twist"]["angular_velocity_w_rad_s"]) == 3
            )
            clamp_valid &= clamp_ok
            ema_valid &= ema_ok
            ring_valid &= ring_ok
            valid &= (
                key in observed
                and 0 <= key[0] < EXPECTED_ITERATIONS
                and 1 <= key[1] <= EXPECTED_ROLLOUT_STEPS
                and 0 <= key[2] < EXPECTED_NUM_ENVS
                and event["global_action_step"] == key[0] * EXPECTED_ROLLOUT_STEPS + key[1]
                and event.get("action_mode") == ACTION_MODE
                and int(event["pose_id"]) == 0
                and event.get("pose_name") == "prone"
                and event.get("joint_attributions") == expected_records
                and len(expected_records) > 0
                and shapes_ok
                and contact_shape_ok
                and clamp_ok
                and ema_ok
                and ring_ok
                and _all_finite(event)
            )
        except (IndexError, KeyError, TypeError, ValueError):
            valid = clamp_valid = ema_valid = ring_valid = False
    exact = Counter(attributed) == observed
    return {
        "hard_joint_limit_reproduced": bool(observed),
        "termination_key_multiset_matches_attribution": exact,
        "termination_and_attribution_counts_match": len(events) == sum(observed.values()) == len(attributed),
        "predicate_recomputed_and_records_valid": valid and exact,
        "action_wrapper_clamp_exact": clamp_valid and bool(events),
        "ema_target_recomputed_exact": ema_valid and bool(events),
        "preceding_16_step_ring_valid": ring_valid and bool(events),
        "all_event_values_finite": bool(events) and all(_all_finite(event) for event in events),
    }


def _body_force_summary_valid(summary: dict[str, Any], joint_name: str) -> bool:
    try:
        names = summary["body_names"]
        forces = summary["net_forces_w_n"]
        magnitudes = [math.sqrt(sum(float(component) ** 2 for component in vector)) for vector in forces]
        dominant_index = max(range(len(magnitudes)), key=magnitudes.__getitem__)
        dominant_body = names[dominant_index]
        total_mass = float(summary["total_robot_mass_kg"])
        return (
            total_mass > 0.0
            and summary["dominant_body"] == dominant_body
            and math.isclose(summary["dominant_force_n"], magnitudes[dominant_index], rel_tol=0.0, abs_tol=1.0e-5)
            and math.isclose(
                summary["dominant_force_bw"], magnitudes[dominant_index] / (total_mass * 9.81), rel_tol=0.0, abs_tol=1.0e-6
            )
            and summary["violated_joint_leg_chain_relation"] == leg_chain_relation(joint_name, dominant_body)
        )
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def hard_series_from_keys(keys) -> list[float]:
    counts = Counter(int(key[0]) for key in keys)
    return [counts[index] / EXPECTED_ROLLOUT_STEPS for index in range(EXPECTED_ITERATIONS)]


def hard_event_counts_from_keys(keys) -> list[int]:
    counts = Counter(int(key[0]) for key in keys)
    return [counts[index] for index in range(EXPECTED_ITERATIONS)]


def historical_identity_checks(
    *,
    hard_event_counts: list[int],
    hard_series: list[float],
    model_hashes: dict[int, str],
    act_count: int,
    update_count: int,
    core_sha256: str,
) -> dict[str, bool]:
    return {
        "training_core_sha256_exact": core_sha256 == EXPECTED_TRAINING_CORE_SHA256,
        "hard_limit_event_counts_exact": hard_event_counts == list(EXPECTED_HARD_EVENT_COUNTS),
        "hard_limit_logger_series_consistent": len(hard_series) == len(EXPECTED_BOUND_TENSORBOARD_HARD_SERIES)
        and all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=2.0e-9)
            for a, b in zip(hard_series, EXPECTED_BOUND_TENSORBOARD_HARD_SERIES)
        ),
        "model_0_sha256_exact": model_hashes.get(0) == EXPECTED_MODEL_0_SHA256,
        "model_9_sha256_exact": model_hashes.get(9) == EXPECTED_MODEL_9_SHA256,
        "act_count_exact_240": act_count == EXPECTED_ACT_COUNT,
        "update_count_exact_10": update_count == EXPECTED_UPDATE_COUNT,
    }


class Gate10PreResetObserver:
    """Capture terminal state and RNG-neutral preceding control-step snapshots."""

    def __init__(self, env, collector_state: dict[str, Any]):
        self.env = env
        self.collector_state = collector_state
        self.ring: deque[dict[str, Any]] = deque(maxlen=RING_BUFFER_STEPS - 1)

    def _frame_tensors(self, action_override=None, *, phase: str) -> dict[str, Any]:
        import torch

        robot = self.env.scene["robot"]
        sensor = self.env.scene["contact_forces"]
        action_term = self.env.action_manager.get_term("joint_pos")
        forces = sensor.data.net_forces_w
        magnitudes = torch.linalg.vector_norm(forces, dim=-1)
        dominant_force, dominant_body = magnitudes.max(dim=1)
        total_mass = self.env._g009_r0_body_mass.sum(dim=1)
        return {
            "tag": dict(self.collector_state["current_tag"]),
            "phase": phase,
            "action": (action_term.raw_actions if action_override is None else action_override).detach().clone(),
            "processed_ema_target": action_term.processed_actions.detach().clone(),
            "applied_torque": robot.data.applied_torque[:, action_term._joint_ids].detach().clone(),
            "episode_control_step": self.env.episode_length_buf.detach().clone(),
            "sim_step_counter": int(self.env._sim_step_counter),
            "joint_position": robot.data.joint_pos[:, action_term._joint_ids].detach().clone(),
            "joint_velocity": robot.data.joint_vel[:, action_term._joint_ids].detach().clone(),
            "root_position": robot.data.root_pos_w.detach().clone(),
            "root_quaternion": robot.data.root_quat_w.detach().clone(),
            "root_linear_velocity": robot.data.root_lin_vel_w.detach().clone(),
            "root_angular_velocity": robot.data.root_ang_vel_w.detach().clone(),
            "dominant_body": dominant_body.detach().clone(),
            "dominant_force_n": dominant_force.detach().clone(),
            "dominant_force_bw": (dominant_force / (total_mass * 9.81)).detach().clone(),
            "body_forces": forces.detach().clone(),
            "total_mass": total_mass.detach().clone(),
        }

    def append_completed_step(self, clipped_actions) -> None:
        self.ring.append(
            self._frame_tensors(clipped_actions, phase="post_step_after_manager_reset_for_terminated_envs")
        )

    def _frame_for_env(self, frame: dict[str, Any], env_index: int, joint_name: str, body_names: list[str]) -> dict[str, Any]:
        body_name = body_names[int(frame["dominant_body"][env_index].item())]
        return {
            **frame["tag"],
            "phase": frame["phase"],
            "env_index": env_index,
            "episode_control_step": int(frame["episode_control_step"][env_index].item()),
            "sim_step_counter": int(frame["sim_step_counter"]),
            "action_post_wrapper_clip": frame["action"][env_index].cpu().tolist(),
            "processed_ema_target_rad": frame["processed_ema_target"][env_index].cpu().tolist(),
            "applied_torque_nm": frame["applied_torque"][env_index].cpu().tolist(),
            "joint_position_rad": frame["joint_position"][env_index].cpu().tolist(),
            "joint_velocity_rad_s": frame["joint_velocity"][env_index].cpu().tolist(),
            "root_pose": {
                "position_w_m": frame["root_position"][env_index].cpu().tolist(),
                "quaternion_wxyz": frame["root_quaternion"][env_index].cpu().tolist(),
            },
            "root_twist": {
                "linear_velocity_w_m_s": frame["root_linear_velocity"][env_index].cpu().tolist(),
                "angular_velocity_w_rad_s": frame["root_angular_velocity"][env_index].cpu().tolist(),
            },
            "body_force_summary": {
                "body_names": body_names,
                "net_forces_w_n": frame["body_forces"][env_index].cpu().tolist(),
                "total_robot_mass_kg": float(frame["total_mass"][env_index].item()),
                "dominant_body": body_name,
                "dominant_force_n": float(frame["dominant_force_n"][env_index].item()),
                "dominant_force_bw": float(frame["dominant_force_bw"][env_index].item()),
                "violated_joint_leg_chain_relation": leg_chain_relation(joint_name, body_name),
            },
        }

    def capture(self, env_ids) -> None:
        from isaac_walk_g009.recover_contracts import ACTION_EMA_ALPHA, SOLVER_JOINT_LIMIT_TOLERANCE_RAD

        if not self.collector_state.get("current_tag"):
            return
        hard = self.env.termination_manager.get_term("hard_joint_limit")
        robot = self.env.scene["robot"]
        sensor = self.env.scene["contact_forces"]
        action_term = self.env.action_manager.get_term("joint_pos")
        joint_ids = action_term._joint_ids
        joint_names = list(action_term._joint_names)
        body_names = list(sensor.body_names)
        terminal = self._frame_tensors(phase="terminal_pre_reset")
        pose_ids = getattr(self.env, "_g009_recover_fall_class", None)
        for env_id in env_ids:
            env_index = int(env_id)
            if not bool(hard[env_index].item()):
                continue
            try:
                position = robot.data.joint_pos[env_index, joint_ids].detach().cpu().tolist()
                limits = robot.data.joint_pos_limits[env_index, joint_ids]
                lower = limits[:, 0].detach().cpu().tolist()
                upper = limits[:, 1].detach().cpu().tolist()
                records = joint_limit_attributions(
                    position=position,
                    lower=lower,
                    upper=upper,
                    joint_names=joint_names,
                    margin_rad=SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
                )
                primary_joint = records[0]["joint_name"] if records else "unknown"
                soft = robot.data.soft_joint_pos_limits[env_index, joint_ids]
                raw = action_term.raw_actions[env_index]
                scale = action_term._scale
                scaled = (raw * scale).clamp(-1.0, 1.0)
                pre_ema = 0.5 * (scaled + 1.0) * (soft[:, 1] - soft[:, 0]) + soft[:, 0]
                previous = self.collector_state["current_ema_previous"][env_index]
                expected_ema = (ACTION_EMA_ALPHA * pre_ema + (1.0 - ACTION_EMA_ALPHA) * previous).clamp(
                    soft[:, 0], soft[:, 1]
                )
                pose_id = int(pose_ids[env_index].item()) if pose_ids is not None else -1
                history = sensor.data.net_forces_w_history[env_index].detach().cpu()
                ring = [*self.ring, terminal][-RING_BUFFER_STEPS:]
                tag = dict(self.collector_state["current_tag"])
                event = {
                    **tag,
                    "episode_control_step": int(self.env.episode_length_buf[env_index].item()),
                    "sim_step_counter": int(self.env._sim_step_counter),
                    "env_index": env_index,
                    "pose_id": pose_id,
                    "pose_name": POSE_NAMES[pose_id] if 0 <= pose_id < len(POSE_NAMES) else None,
                    "action_mode": ACTION_MODE,
                    "joint_names": joint_names,
                    "joint_position_rad": position,
                    "joint_lower_limit_rad": lower,
                    "joint_upper_limit_rad": upper,
                    "joint_soft_lower_limit_rad": soft[:, 0].detach().cpu().tolist(),
                    "joint_soft_upper_limit_rad": soft[:, 1].detach().cpu().tolist(),
                    "joint_attributions": records,
                    "action_scale": float(action_term._scale),
                    "ema_alpha": float(ACTION_EMA_ALPHA),
                    "ppo_sample_pre_wrapper_clip": self.collector_state["current_ppo_actions"][env_index].cpu().tolist(),
                    "action_term_raw_post_wrapper_clip": raw.detach().cpu().tolist(),
                    "pre_ema_scaled_target_rad": pre_ema.detach().cpu().tolist(),
                    "ema_previous_target_rad": previous.detach().cpu().tolist(),
                    "ema_expected_target_rad": expected_ema.detach().cpu().tolist(),
                    "processed_ema_target_rad": action_term.processed_actions[env_index].detach().cpu().tolist(),
                    "joint_velocity_rad_s": robot.data.joint_vel[env_index, joint_ids].detach().cpu().tolist(),
                    "applied_torque_nm": robot.data.applied_torque[env_index, joint_ids].detach().cpu().tolist(),
                    "root_pose": {
                        "position_w_m": robot.data.root_pos_w[env_index].detach().cpu().tolist(),
                        "quaternion_wxyz": robot.data.root_quat_w[env_index].detach().cpu().tolist(),
                    },
                    "root_twist": {
                        "linear_velocity_w_m_s": robot.data.root_lin_vel_w[env_index].detach().cpu().tolist(),
                        "angular_velocity_w_rad_s": robot.data.root_ang_vel_w[env_index].detach().cpu().tolist(),
                    },
                    "contact_sensor_history": {
                        "sensor_name": "contact_forces",
                        "history_length": int(history.shape[0]),
                        "body_names": body_names,
                        "net_forces_w_history_n": history.tolist(),
                    },
                    "preceding_control_step_ring": [
                        self._frame_for_env(frame, env_index, primary_joint, body_names) for frame in ring
                    ],
                }
                if not records:
                    self.collector_state["errors"].append(f"hard termination without recomputed violation: {tag} env={env_index}")
                self.collector_state["events"].append(event)
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
                self.collector_state["errors"].append(f"pre-reset attribution failed: env={env_index}: {error}")


def install_pre_reset_observer(recorder_manager, observer: Gate10PreResetObserver):
    if len(recorder_manager.active_terms) != 0:
        raise RuntimeError("exact-RNG attribution requires zero active recorder terms")
    original = recorder_manager.record_pre_reset

    def observed_pre_reset(env_ids, force_export_or_skip=None):
        import torch

        hashes_before = torch_rng_component_sha256(torch)
        cpu_before = torch.get_rng_state().clone()
        cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
        observer.capture(env_ids)
        cpu_after = torch.get_rng_state()
        cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        hashes_after = torch_rng_component_sha256(torch)
        observer.collector_state.setdefault("observer_rng_hash_checks", []).append(
            {"before": hashes_before, "after": hashes_after, "identical": hashes_before == hashes_after}
        )
        observer.collector_state["observer_rng_neutral"] &= bool(torch.equal(cpu_before, cpu_after))
        observer.collector_state["observer_rng_neutral"] &= len(cuda_before) == len(cuda_after) and all(
            bool(torch.equal(before, after)) for before, after in zip(cuda_before, cuda_after)
        )
        return original(env_ids, force_export_or_skip)

    recorder_manager.record_pre_reset = observed_pre_reset
    if len(recorder_manager.active_terms) != 0:
        raise RuntimeError("pre-reset observer changed active recorder terms")
    return original


def run(args: argparse.Namespace, execution: dict[str, Any]) -> dict[str, Any]:
    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    import torch
    from rsl_rl.runners import OnPolicyRunner  # pyright: ignore[reportMissingImports]

    import isaaclab_tasks  # noqa: F401  # pyright: ignore[reportMissingImports]
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # pyright: ignore[reportMissingImports]
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # pyright: ignore[reportMissingImports]
    from isaac_walk_g009 import register_tasks
    from isaac_walk_g009.recover_contracts import (
        ACTION_EMA_ALPHA,
        ACTION_SCALE,
        POSE_CURRICULUM_PROBABILITIES,
        PPO_INIT_NOISE_STD,
        SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
        canonical_sha256,
        recover_contract,
    )

    validate_protocol_args(args)
    training_binding = validate_training_report(args.training_report)
    core = training_core_provenance()
    diagnostic_source = diagnostic_source_provenance()
    runtime_sources = official_runtime_source_provenance()
    collector: dict[str, Any] = {
        "events": [],
        "errors": [],
        "current_tag": None,
        "current_ppo_actions": None,
        "current_ema_previous": None,
        "observer_rng_neutral": True,
        "observer_rng_hash_checks": [],
    }
    register_tasks()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = agent_cfg.seed = args.seed
    env_cfg.sim.device = agent_cfg.device = args.device
    agent_cfg.max_iterations = args.iterations
    agent_cfg.run_name = training_binding["run_name"]
    agent_cfg.resume = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    import tempfile

    diagnostic_log_dir = tempfile.TemporaryDirectory(prefix="g009_gate10_attribution_")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=diagnostic_log_dir.name, device=args.device)
    if runner.num_steps_per_env != EXPECTED_ROLLOUT_STEPS:
        raise RuntimeError("official runner rollout length is not 24")
    sensor = env.unwrapped.scene["contact_forces"]
    if sensor.cfg.history_length != EXPECTED_CONTACT_HISTORY_LENGTH:
        raise RuntimeError("contact sensor history_length changed from the bound training config")
    observer = Gate10PreResetObserver(env.unwrapped, collector)
    install_pre_reset_observer(env.unwrapped.recorder_manager, observer)
    observed_keys: list[tuple[int, int, int]] = []
    update_records: list[dict[str, Any]] = []
    action_samples = []
    original_act = runner.alg.act
    original_update = runner.alg.update
    original_step = env.step
    action_count = 0

    def observed_act(obs, privileged_obs):
        nonlocal action_count
        actions = original_act(obs, privileged_obs)
        action_count += 1
        collector["current_tag"] = {
            "iteration": (action_count - 1) // EXPECTED_ROLLOUT_STEPS,
            "rollout_control_step": (action_count - 1) % EXPECTED_ROLLOUT_STEPS + 1,
            "global_action_step": action_count,
        }
        collector["current_ppo_actions"] = actions.detach().clone()
        action_samples.append(actions.detach().clone())
        return actions

    def observed_step(actions):
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        collector["current_ema_previous"] = action_term._prev_applied_actions.detach().clone()
        result = original_step(actions)
        hard = env.unwrapped.termination_manager.get_term("hard_joint_limit")
        tag = collector["current_tag"]
        for env_index in torch.nonzero(hard, as_tuple=False).flatten().cpu().tolist():
            observed_keys.append((tag["iteration"], tag["rollout_control_step"], int(env_index)))
        observer.append_completed_step(actions.clamp(-1.0, 1.0))
        return result

    def observed_update(*update_args, **update_kwargs):
        index = len(update_records)
        record = {
            "update_index": index,
            "policy_before_sha256": policy_state_sha256(runner.alg.policy),
            "optimizer_before_sha256": optimizer_state_sha256(runner.alg.optimizer),
        }
        result = original_update(*update_args, **update_kwargs)
        record.update(
            {
                "policy_after_sha256": policy_state_sha256(runner.alg.policy),
                "optimizer_after_sha256": optimizer_state_sha256(runner.alg.optimizer),
                "original_update_returned": True,
            }
        )
        update_records.append(record)
        return result

    runner.alg.act = observed_act
    runner.alg.update = observed_update
    env.step = observed_step
    rng_before = torch_rng_state_sha256(torch)
    recorder_zero_before = len(env.unwrapped.recorder_manager.active_terms) == 0
    model_hashes: dict[int, str] = {}
    checkpoint_files: list[str] = []
    try:
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        for index in (0, 9):
            path = Path(diagnostic_log_dir.name) / f"model_{index}.pt"
            if path.is_file():
                model_hashes[index] = _file_sha256(path)
        checkpoint_files = sorted(
            str(path.relative_to(diagnostic_log_dir.name)).replace("\\", "/")
            for path in Path(diagnostic_log_dir.name).rglob("model_*.pt")
        )
    finally:
        recorder_zero_after = len(env.unwrapped.recorder_manager.active_terms) == 0
        rng_after = torch_rng_state_sha256(torch)
        if runner.writer is not None:
            runner.writer.close()
        env.close()
        diagnostic_log_dir.cleanup()

    hard_event_counts = hard_event_counts_from_keys(observed_keys)
    hard_series = hard_series_from_keys(observed_keys)
    events = collector["events"]
    attribution_checks = validate_attribution_result(
        events=events,
        observed_termination_keys=observed_keys,
        margin_rad=SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
    )
    execution_checks = {
        "collector_errors_absent": not collector["errors"],
        "training_core_exact_and_clean": bool(core["exact_match"] and core["clean"]),
        "diagnostic_source_commit_bound_and_clean": bool(
            diagnostic_source["git_commit_valid"] and diagnostic_source["clean"]
        ),
        "official_runtime_source_hashes_exact": runtime_sources == EXPECTED_OFFICIAL_RUNTIME_SOURCE_SHA256,
        "official_runner_learn_completed": runner.current_learning_iteration == 9,
        "official_act_count_exact": action_count == EXPECTED_ACT_COUNT,
        "original_update_count_exact": len(update_records) == EXPECTED_UPDATE_COUNT,
        "all_original_updates_returned": len(update_records) == EXPECTED_UPDATE_COUNT
        and all(item["original_update_returned"] for item in update_records),
        "active_recorder_terms_remained_zero": recorder_zero_before and recorder_zero_after,
        "pre_reset_observer_rng_neutral": bool(collector["observer_rng_neutral"]),
        "contact_sensor_history_length_unchanged": sensor.cfg.history_length == EXPECTED_CONTACT_HISTORY_LENGTH,
        "scratch_checkpoint_load_absent": True,
        "pose_curriculum_prone_only": tuple(POSE_CURRICULUM_PROBABILITIES[0]) == (1.0, 0.0, 0.0, 0.0),
        "action_and_noise_contract_unchanged": ACTION_SCALE == 0.70
        and ACTION_EMA_ALPHA == 0.2
        and PPO_INIT_NOISE_STD == 0.5,
    }
    identity_checks = historical_identity_checks(
        hard_event_counts=hard_event_counts,
        hard_series=hard_series,
        model_hashes=model_hashes,
        act_count=action_count,
        update_count=len(update_records),
        core_sha256=core["sha256"],
    )
    attribution_passed = bool(observed_keys) and all(attribution_checks.values()) and all(execution_checks.values())
    historical_identity = attribution_passed and all(identity_checks.values())
    if not observed_keys:
        outcome = "not_reproduced"
    elif attribution_passed:
        outcome = "attributed_historical_identity" if historical_identity else "attributed_fresh_reproduction"
    else:
        outcome = "invalid"
    action_digest = hashlib.sha256()
    for sample in action_samples:
        action_digest.update(_tensor_bytes(sample))
    return {
        "schema_version": 1,
        "probe_id": "g009_r0_gate10_full_update_hard_joint_limit_attribution",
        "execution": execution,
        "protocol": {
            "id": PROTOCOL_ID,
            "task": args.task,
            "seed": args.seed,
            "device": args.device,
            "headless": bool(args.headless),
            "scratch": True,
            "checkpoint_loaded": False,
            "num_envs": args.num_envs,
            "rollout_steps_per_iteration": args.rollout_steps,
            "iterations": args.iterations,
            "official_runner_method": "OnPolicyRunner.learn",
            "original_ppo_update_required": True,
            "pre_reset_hook": "instance wrapper around RecorderManager.record_pre_reset",
            "active_recorder_term_count": 0,
            "ring_buffer_control_steps": RING_BUFFER_STEPS,
        },
        "contract": {
            "id": "g009_r0_recover_rev12",
            "sha256": canonical_sha256(recover_contract()),
            "hard_joint_limit_margin_rad": SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
        },
        "training_binding": training_binding,
        "training_core": core,
        "diagnostic_source": diagnostic_source,
        "runtime_reproduction": {
            "torch_rng_state_before_learn_sha256": rng_before,
            "torch_rng_state_after_learn_sha256": rng_after,
            "observer_rng_neutral": bool(collector["observer_rng_neutral"]),
            "observer_rng_hash_checks": collector["observer_rng_hash_checks"],
            "official_runtime_source_sha256": runtime_sources,
            "expected_official_runtime_source_sha256": EXPECTED_OFFICIAL_RUNTIME_SOURCE_SHA256,
            "ppo_action_stream_sha256": action_digest.hexdigest(),
            "act_count": action_count,
            "update_count": len(update_records),
            "updates": update_records,
            "generated_checkpoint_files": checkpoint_files,
            "generated_checkpoint_sha256": {f"model_{index}.pt": value for index, value in sorted(model_hashes.items())},
            "hard_joint_limit_series": hard_series,
            "hard_joint_limit_event_counts": hard_event_counts,
            "observed_termination_key_multiset": [
                {"iteration": key[0], "rollout_control_step": key[1], "env_index": key[2], "count": count}
                for key, count in sorted(Counter(observed_keys).items())
            ],
        },
        "counts": {
            "hard_joint_limit_termination_env_events": len(observed_keys),
            "attributed_env_events": len(events),
            "attributed_joint_records": sum(len(event.get("joint_attributions", ())) for event in events),
        },
        "events": events,
        "collector_errors": collector["errors"],
        "checks": {**attribution_checks, **execution_checks},
        "historical_identity_checks": identity_checks,
        "outcome": outcome,
        "attribution_contract_passed": attribution_passed,
        "historical_trajectory_identity_confirmed": historical_identity,
        "historical_identity_semantics": (
            "true only when source bundle, exact hard series, model_0/model_9 hashes, act240, and update10 all match"
        ),
        "gate10_safety_passed": False,
        "safety_gate_passed": False,
        "learned_policy_qualified": False,
        "interpretation": {
            "qualification_scope": "Attribution only; this report never promotes Gate10 safety or learned-policy qualification.",
            "identity_mismatch": "Any identity mismatch is retained as a fresh reproduction only.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--num-envs", type=int, default=EXPECTED_NUM_ENVS)
    parser.add_argument("--rollout-steps", type=int, default=EXPECTED_ROLLOUT_STEPS)
    parser.add_argument("--iterations", type=int, default=EXPECTED_ITERATIONS)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(
            "usage: attribute_g009_r0_gate10.py --training-report REPORT --output REPORTS_RUNS_JSON "
            "[--task TASK] [--seed 42] [--num-envs 1024] [--rollout-steps 24] "
            "[--iterations 10] --headless --device cuda:0"
        )
        return 0
    output, execution = prepare_execution(parse_prelaunch_output())
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    args = build_parser().parse_args()
    app_launcher = AppLauncher(args)
    try:
        report = run(args, execution)
        _write_json_atomic(output, report)
        print(json.dumps({"output": str(output), "outcome": report["outcome"]}, ensure_ascii=False))
        return 0 if report["attribution_contract_passed"] else 2
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
