from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = _load("evaluate_g009_r0", "scripts/evaluate_g009_r0.py")
recorder = _load("record_g009_r0", "scripts/record_g009_r0.py")
media = _load("build_g009_r0_media", "scripts/build_g009_r0_media.py")


def test_scripts_are_import_light() -> None:
    assert evaluation.STAGE_NUMBER == recorder.STAGE_NUMBER == media.STAGE_NUMBER == "G009-5"
    assert evaluation.STAGE_ID == recorder.STAGE_ID == media.STAGE_ID == "R0"
    assert evaluation.POSE_NAMES == recorder.POSE_NAMES == media.POSE_NAMES


def test_sample_summary_is_deterministic_and_uses_nearest_rank_p95() -> None:
    result = evaluation.summarize_samples([5.0, 1.0, 3.0, 2.0, 4.0])
    assert result == {
        "count": 5,
        "min": 1.0,
        "median": 3.0,
        "mean": 3.0,
        "p95": 5.0,
        "max": 5.0,
    }
    assert evaluation.summarize_samples([])["median"] is None


def test_pose_gate_blocks_safety_termination_even_with_high_success_rate() -> None:
    accumulator = evaluation._new_accumulator()
    accumulator.update(
        {
            "episode_count": 10,
            "success_count": 9,
            "numeric_invalid_count": 1,
            "recovery_times_s": [2.0] * 9,
        }
    )
    result = evaluation.finalize_pose_metrics(
        accumulator,
        minimum_success_rate=0.8,
        maximum_median_recovery_time_s=4.0,
    )
    assert result["success_rate"] == 0.9
    assert result["gate_checks"]["success_rate"] is True
    assert result["gate_checks"]["no_safety_termination"] is False
    assert result["gate_pass"] is False


@pytest.mark.parametrize(
    ("raw_violation", "expected"),
    [(0.009, True), (0.011, False)],
)
def test_pose_gate_blocks_raw_joint_limit_violation_above_solver_tolerance(
    raw_violation: float, expected: bool
) -> None:
    accumulator = evaluation._new_accumulator()
    accumulator.update(
        {
            "episode_count": 1,
            "success_count": 1,
            "recovery_times_s": [1.0],
            "max_raw_hard_joint_limit_violation_rad": raw_violation,
        }
    )

    result = evaluation.finalize_pose_metrics(
        accumulator,
        minimum_success_rate=0.8,
        maximum_median_recovery_time_s=4.0,
    )

    assert result["gate_checks"]["joint_limit_violation_within_solver_tolerance"] is expected
    assert result["gate_pass"] is expected


def test_report_preserves_four_pose_blocking_cells() -> None:
    args = argparse.Namespace(
        task=evaluation.DEFAULT_TASK,
        seed=42,
        device="cuda:0",
        headless=True,
        num_envs=256,
        horizon_steps=400,
        minimum_success_rate=0.8,
        maximum_median_recovery_time_s=4.0,
    )
    accumulators = {}
    for pose in evaluation.POSE_NAMES:
        accumulator = evaluation._new_accumulator()
        accumulator.update(
            {
                "episode_count": 2,
                "success_count": 2,
                "recovery_times_s": [1.0, 2.0],
            }
        )
        accumulators[pose] = accumulator
    report = evaluation.build_report(
        args=args,
        checkpoint={"path": "%USERPROFILE%\\checkpoint.pt", "sha256": "a" * 64},
        step_dt_s=0.02,
        pose_accumulators=accumulators,
        physics_readback={"terrain": {"type": "plane"}},
        training_binding={"source_bundle": {"sha256": "b" * 64}},
        source_state_before={"commit": "c" * 40, "clean": True},
        source_state_after={"commit": "c" * 40, "clean": True},
    )
    assert report["status"] == "pass"
    assert report["aggregate"] == {
        "episode_count": 8,
        "success_count": 8,
        "success_rate": 1.0,
        "safety_termination_count": 0,
        "all_pose_gate_pass": True,
    }
    assert [item["pose_id"] for item in report["poses"]] == list(evaluation.POSE_NAMES)


def test_local_video_names_keep_stage_and_pose_numbering() -> None:
    assert recorder.output_name("prone", 42) == "g009_5_r0_01_prone_s42.mp4"
    assert recorder.output_name("right_side", 42) == "g009_5_r0_04_right_side_s42.mp4"
    assert "--/app/vulkan=false" in recorder.WINDOWS_KIT_ARGS
    assert recorder.validate_output_dir(recorder.DEFAULT_OUTPUT_DIR) == recorder.DEFAULT_OUTPUT_DIR.resolve()
    with pytest.raises(ValueError, match="local-only"):
        recorder.validate_output_dir(ROOT / "docs" / "media" / "g009" / "R0")


def test_official_recorder_keeps_initial_frame_and_removes_terminal_reset() -> None:
    assert recorder.recorded_frame_count(25, terminated=True) == 25
    assert recorder.recorded_frame_count(25, terminated=False) == 26
    assert recorder.recorded_frame_count(1, terminated=True) == 1


def test_media_paths_follow_r0_contract_and_mp4_stays_local_only() -> None:
    assert media.LOCAL_MP4_PATH.startswith(
        "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\R0\\"
    )
    assert media.PUBLIC_GIF_PATH.startswith("docs/media/g009/R0/")
    assert media.PUBLIC_PNG_PATH.startswith("docs/media/g009/R0/")
    assert not media.LOCAL_MP4_PATH.startswith("docs/")
    assert media.SUMMARY_PATH.startswith("reports/runs/g009_r0_flat")
    assert media.SIDECAR_PATH.startswith("reports/runs/g009_r0_flat")


def test_media_signatures_and_public_size_are_checked(tmp_path: Path) -> None:
    gif = tmp_path / "capture.gif"
    png = tmp_path / "capture.png"
    gif.write_bytes(b"GIF89a" + b"x" * 16)
    png.write_bytes(media.PNG_SIGNATURE + b"x" * 16)
    media._validate_media(gif, "gif")
    media._validate_media(png, "png")
    gif.write_bytes(b"notgif")
    with pytest.raises(ValueError, match="GIF signature"):
        media._validate_media(gif, "gif")


def test_media_builder_requires_all_four_capture_reports() -> None:
    with pytest.raises(SystemExit):
        media.parse_args(["--capture-reports", "a.json", "b.json", "c.json"])


def test_publish_transaction_rolls_back_all_outputs(tmp_path: Path) -> None:
    staged = tmp_path / "staged.gif"
    final = tmp_path / "final.gif"
    staged.write_bytes(b"new")
    final.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="post-publish"):
        media._publish_transaction(
            ((staged, final),), lambda: (_ for _ in ()).throw(RuntimeError("post-publish"))
        )
    assert final.read_bytes() == b"old"
    assert not staged.exists()


def test_official_protocol_cannot_be_relaxed_without_diagnostic() -> None:
    args = argparse.Namespace(**evaluation.OFFICIAL_PROTOCOL, diagnostic=False)
    args.minimum_success_rate = 0.1
    with pytest.raises(ValueError, match="official protocol values are fixed"):
        evaluation.protocol_mode(args)
    args.diagnostic = True
    assert evaluation.protocol_mode(args) == "diagnostic_only"


def test_diagnostic_result_never_becomes_public_pass() -> None:
    args = argparse.Namespace(
        **{**evaluation.OFFICIAL_PROTOCOL, "minimum_success_rate": 0.1},
        diagnostic=True,
        device="cuda:0",
        headless=True,
    )
    accumulators = {}
    for pose in evaluation.POSE_NAMES:
        accumulator = evaluation._new_accumulator()
        accumulator.update({"episode_count": 1, "success_count": 1, "recovery_times_s": [1.0]})
        accumulators[pose] = accumulator
    state = {"commit": "c" * 40, "clean": True}
    report = evaluation.build_report(
        args=args,
        checkpoint={"path": "checkpoint.pt", "sha256": "a" * 64},
        step_dt_s=0.02,
        pose_accumulators=accumulators,
        physics_readback={},
        training_binding={},
        source_state_before=state,
        source_state_after=state,
    )
    assert report["status"] == "diagnostic"
    assert report["protocol_mode"] == "diagnostic_only"


def test_training_report_requires_scratch_1024x300_and_exact_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    checkpoint = tmp_path / "model_299.pt"
    source.write_text("x = 1\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(evaluation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluation,
        "git_source_state",
        lambda: {"commit": "c" * 40, "clean": True, "dirty_paths": [], "source_dirty_paths": []},
    )
    files = {"source.py": evaluation.file_sha256(source)}
    report = {
        "task": evaluation.DEFAULT_TASK,
        "seed": 42,
        "num_envs": 1024,
        "max_iterations": 300,
        "headless": True,
        "resume": {"enabled": False},
        "qualification_mode": {
            "enabled": True,
            "preflight_passed": True,
            "policy_qualification_status": "not_run",
        },
        "run_health_passed": True,
        "repository": {"commit": "c" * 40, "dirty": False},
        "source_bundle": {"sha256": evaluation.source_bundle_sha256(files), "files": files},
        "artifacts": {"checkpoint_sha256": evaluation.file_sha256(checkpoint)},
    }
    report_path = tmp_path / "training.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert evaluation.validate_training_report(report_path, checkpoint)["source_bundle"]["files"] == files
    report["qualification_mode"]["preflight_passed"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="qualification_preflight"):
        evaluation.validate_training_report(report_path, checkpoint)
    report["qualification_mode"]["preflight_passed"] = True
    report["max_iterations"] = 299
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="max_iterations"):
        evaluation.validate_training_report(report_path, checkpoint)


def test_publish_transaction_rolls_back_multiple_outputs_when_second_install_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged_a, staged_b = tmp_path / "a.new", tmp_path / "b.new"
    final_a, final_b = tmp_path / "a.out", tmp_path / "b.out"
    staged_a.write_bytes(b"new-a")
    staged_b.write_bytes(b"new-b")
    final_a.write_bytes(b"old-a")
    final_b.write_bytes(b"old-b")
    real_replace = media.os.replace
    calls = 0

    def failing_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if Path(source) == staged_b:
            raise OSError("injected second install failure")
        real_replace(source, destination)

    monkeypatch.setattr(media.os, "replace", failing_replace)
    with pytest.raises(OSError, match="injected"):
        media._publish_transaction(((staged_a, final_a), (staged_b, final_b)), lambda: None)
    assert final_a.read_bytes() == b"old-a"
    assert final_b.read_bytes() == b"old-b"


def test_publish_transaction_rejects_missing_stage_before_touching_outputs(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    final = tmp_path / "final"
    final.write_bytes(b"old")
    with pytest.raises(ValueError, match="staged transaction input missing"):
        media._publish_transaction(((missing, final),), lambda: None)
    assert final.read_bytes() == b"old"


def test_media_builder_rejects_diagnostic_evaluation_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    record_source = scripts / "record_g009_r0.py"
    eval_source = scripts / "evaluate_g009_r0.py"
    record_source.write_text("record\n", encoding="utf-8")
    eval_source.write_text("evaluate\n", encoding="utf-8")
    config = tmp_path / "g009_r0.json"
    config.write_text("{}\n", encoding="utf-8")
    video_by_name = {}
    captures = []
    state = {"commit": "c" * 40, "clean": True}
    checkpoint = {"path": "%USERPROFILE%\\checkpoint.pt", "sha256": "a" * 64}
    for index, pose in enumerate(media.POSE_NAMES, start=1):
        name = f"g009_5_r0_{index:02d}_{pose}_s42.mp4"
        video = tmp_path / name
        video.write_bytes(f"video-{pose}".encode())
        video_by_name[name] = video
        captures.append(
            {
                "goal_id": media.GOAL_ID,
                "stage_number": media.STAGE_NUMBER,
                "stage_id": media.STAGE_ID,
                "status": "complete",
                "headless": True,
                "offscreen": True,
                "pose": {"pose_id": pose, "index": index},
                "metrics": {"stable_success": True, "termination_reason": "stable_success"},
                "local_video": {
                    "path": f"%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\R0\\{name}",
                    "sha256": media.file_sha256(video),
                    "bytes": video.stat().st_size,
                    "git_policy": "local_only",
                },
                "source_bindings": {
                    "record_source": {"path": "scripts/record_g009_r0.py", "sha256": media.file_sha256(record_source)},
                    "evaluator": {"path": "scripts/evaluate_g009_r0.py", "sha256": media.file_sha256(eval_source)},
                    "config": {"path": media.CONFIG_PATH, "sha256": media.file_sha256(config)},
                },
                "source_state": {"before": state, "after": state},
                "physics_readback": {"effective_friction_valid": True, "friction_combine_mode": "multiply"},
                "seed": 42,
                "source_commit": "c" * 40,
                "checkpoint": checkpoint,
            }
        )
    quantitative = {
        "goal_id": media.GOAL_ID,
        "stage_number": media.STAGE_NUMBER,
        "stage_id": media.STAGE_ID,
        "status": "pass",
        "protocol_mode": "diagnostic_only",
        "seed": 42,
        "checkpoint": checkpoint,
        "poses": [{"pose_id": pose} for pose in media.POSE_NAMES],
        "aggregate": {"all_pose_gate_pass": True},
    }
    capture_paths = [tmp_path / f"capture-{index}.json" for index in range(4)]
    quantitative_path = tmp_path / "quantitative.json"
    documents = {**dict(zip(capture_paths, captures, strict=True)), quantitative_path: quantitative}
    monkeypatch.setattr(media, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(media, "_read_json", lambda path: documents[path])
    monkeypatch.setattr(media, "_resolve_portable", lambda value: video_by_name[Path(value.replace("\\", "/")).name])
    with pytest.raises(ValueError, match="diagnostic evaluation"):
        media.validate_capture_reports(capture_paths, quantitative_path, config)
