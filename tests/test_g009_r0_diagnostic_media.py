from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = _load("g009_r0_pilot_analysis", "scripts/analyze_g009_r0_pilot.py")
diagnostic = _load("g009_r0_diagnostic_media", "scripts/build_g009_r0_diagnostic_media.py")
official = _load("g009_r0_official_media_for_diagnostic_test", "scripts/build_g009_r0_media.py")


def _series(values: list[float]) -> list[dict[str, float | int]]:
    return [{"step": index, "wall_time": float(index), "value": value} for index, value in enumerate(values)]


def _pilot_series() -> dict[str, list[dict[str, float | int]]]:
    result = {tag: _series([0.0] * 50) for tag in analysis.EXPECTED_TAGS}
    result["Episode_Reward/stable_support"] = _series([0.0] * 42 + [0.001] * 8)
    result["Episode_Reward/upright_hold"] = _series([0.0] * 42 + [0.002] * 8)
    result["Episode_Termination/hard_joint_limit"] = _series([0.1] * 23 + [0.0] * 27)
    result["Curriculum/recover_pose_distribution/probability_prone"] = _series([1.0] * 49 + [0.9791667])
    result["Curriculum/recover_pose_distribution/phase_index"] = _series([0.0] * 49 + [0.0416667])
    result["Curriculum/recover_pose_distribution/probability_left_side"] = _series([0.0] * 49 + [0.0104167])
    result["Curriculum/recover_pose_distribution/probability_right_side"] = _series([0.0] * 49 + [0.0104167])
    result["Policy/mean_noise_std"] = _series([0.5 + index * 0.0008 for index in range(50)])
    result["Train/mean_reward"] = _series([float(index) / 10.0 for index in range(50)])
    return result


def test_analysis_reproduces_tail_and_never_qualifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_path = tmp_path / "training.json"
    report_path.write_text("{}", encoding="utf-8")
    (tmp_path / "events.out.tfevents.fixture").write_bytes(b"event-bundle")
    monkeypatch.setattr(analysis, "REPO_ROOT", tmp_path)
    report = {
        "run_name": analysis.EXPECTED_RUN_NAME,
        "repository": {"commit": "c" * 40},
        "source_bundle": {"sha256": "b" * 64},
        "artifacts": {
            "tensorboard_directory": "%USERPROFILE%\\logs\\tb",
            "checkpoint": "%USERPROFILE%\\logs\\model_49.pt",
            "checkpoint_sha256": "a" * 64,
        },
    }
    result = analysis.build_analysis(report, report_path, tmp_path, _pilot_series())
    assert result["status"] == "diagnostic_complete"
    assert result["qualification_allowed"] is False
    assert set(result["qualification_block_reasons"]) == {
        "diagnostic_pilot_never_qualifies",
        "hard_joint_limit_nonzero",
        "strict_success_zero",
        "prone_curriculum_boundary_leak",
    }
    assert len(result["tail_10_iterations"]) == 10
    assert result["tail_10_iterations"][0]["iteration"] == 40
    assert result["observations"]["hard_joint_limit"]["nonzero_sample_count"] == 23
    assert len(result["tensorboard"]["event_bundle_sha256"]) == 64


def test_analysis_rejects_misaligned_tensorboard_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_path = tmp_path / "training.json"
    report_path.write_text("{}", encoding="utf-8")
    (tmp_path / "events.out.tfevents.fixture").write_bytes(b"event-bundle")
    monkeypatch.setattr(analysis, "REPO_ROOT", tmp_path)
    series = _pilot_series()
    series["Episode_Reward/stable_support"][10]["step"] = 999
    with pytest.raises(ValueError, match="step alignment"):
        analysis.build_analysis(
            {
                "run_name": analysis.EXPECTED_RUN_NAME,
                "repository": {"commit": "c" * 40},
                "source_bundle": {"sha256": "b" * 64},
                "artifacts": {"tensorboard_directory": "tb", "checkpoint": "model", "checkpoint_sha256": "a" * 64},
            },
            report_path,
            tmp_path,
            series,
        )


def test_dynamic_analysis_rejects_boolean_iteration_count(tmp_path: Path) -> None:
    report_path = tmp_path / "training.json"
    report_path.write_text(
        json.dumps(
            {
                "run_name": "go2_flat_recover_rev10_prone_gate01_s42_fixture",
                "task": analysis.EXPECTED_TASK,
                "num_envs": 1024,
                "max_iterations": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="iteration count is invalid"):
        analysis.validate_pilot_report(
            report_path,
            revision="rev10",
            gate_label="gate01",
            output_stem="g009_5_r0_diag_rev10_gate01_01_prone",
            expected_run_name="go2_flat_recover_rev10_prone_gate01_s42_fixture",
        )


def test_gate01_analysis_uses_report_iteration_count_and_stays_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "training.json"
    report_path.write_text("{}", encoding="utf-8")
    (tmp_path / "events.out.tfevents.fixture").write_bytes(b"event-bundle")
    monkeypatch.setattr(analysis, "REPO_ROOT", tmp_path)
    series = {tag: values[:1] for tag, values in _pilot_series().items()}
    result = analysis.build_analysis(
        {
            "run_name": "go2_flat_recover_rev10_prone_gate01_s42_fixture",
            "max_iterations": 1,
            "repository": {"commit": "c" * 40},
            "source_bundle": {"sha256": "b" * 64},
            "artifacts": {
                "tensorboard_directory": "tb",
                "checkpoint": "model_0.pt",
                "checkpoint_sha256": "a" * 64,
            },
        },
        report_path,
        tmp_path,
        series,
        revision="rev10",
        gate_label="gate01",
        output_stem="g009_5_r0_diag_rev10_gate01_01_prone",
    )
    assert result["pilot"]["iterations"] == 1
    assert len(result["tail_10_iterations"]) == 1
    assert result["qualification_allowed"] is False
    assert result["public_claim_eligible"] is False
    assert "관절 한계 종료가 관측됨" in result["interpretation"]
    assert "prone 커리큘럼 경계 누수가 관측되지 않음" in result["interpretation"]
    assert "정책 자격 평가는 수행하지 않았다" in result["interpretation"]


def _capture_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_stem: str | None = None
) -> tuple[Path, Path, dict, dict]:
    video = tmp_path / (diagnostic.EXPECTED_LOCAL_NAME if output_stem is None else f"{output_stem}_s42.mp4")
    checkpoint = tmp_path / "model_49.pt"
    training = tmp_path / "training.json"
    tensorboard = tmp_path / "tensorboard"
    tensorboard.mkdir()
    event_file = tensorboard / "events.out.tfevents.fixture"
    event_file.write_bytes(b"event-bundle")
    video.write_bytes(b"mp4-diagnostic")
    checkpoint.write_bytes(b"checkpoint")
    training.write_text('{"training": true}\n', encoding="utf-8")
    portable_video = f"%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\R0\\diagnostic\\{video.name}"
    state = {"commit": "c" * 40, "clean": True}
    capture = {
        "goal_id": "g009",
        "stage_number": "G009-5",
        "stage_id": "R0",
        "status": "diagnostic_complete",
        "diagnostic_only": True,
        "public_claim_eligible": False,
        "qualification_status": "not_run",
        "policy_result": "failure",
        "strict_success": 0,
        "revision": "rev9" if output_stem is None else "rev10",
        "gate_label": "pilot" if output_stem is None else "gate10",
        "output_stem": output_stem,
        "task": "Isaac-G009-Recover-Flat-Go2-R0-v0",
        "seed": 42,
        "headless": True,
        "offscreen": True,
        "pose": {"index": 1, "pose_id": "prone", "source_class_id": 0},
        "source_state": {"before": state, "after": state},
        "capture_commit": state["commit"],
        "metrics": {"stable_success": False, "termination_reason": "capture_horizon"},
        "local_video": {
            "path": portable_video,
            "sha256": diagnostic.file_sha256(video),
            "bytes": video.stat().st_size,
            "git_policy": "local_only",
        },
        "checkpoint": {"path": "checkpoint.pt", "sha256": diagnostic.file_sha256(checkpoint)},
        "training_binding": {
            "path": "training.json",
            "sha256": diagnostic.file_sha256(training),
            "checkpoint_sha256": diagnostic.file_sha256(checkpoint),
            "run_name": (
                analysis.EXPECTED_RUN_NAME
                if output_stem is None
                else "go2_flat_recover_rev10_prone_gate10_s42_fixture"
            ),
        },
        "source_bindings": {
            "record_source": {
                "path": diagnostic.EXPECTED_RECORD_SOURCE,
                "sha256": diagnostic.file_sha256(ROOT / diagnostic.EXPECTED_RECORD_SOURCE),
            },
            "config": {
                "path": diagnostic.EXPECTED_CONFIG_SOURCE,
                "sha256": diagnostic.file_sha256(ROOT / diagnostic.EXPECTED_CONFIG_SOURCE),
            },
        },
    }
    event_files = {event_file.name: diagnostic.file_sha256(event_file)}
    event_payload = "\n".join(f"{name}:{digest}" for name, digest in event_files.items())
    analysis_doc = {
        "schema_version": "g009.r0.pilot_analysis.v2",
        "status": "diagnostic_complete",
        "diagnostic_only": True,
        "qualification_allowed": False,
        "public_claim_eligible": False,
        "revision": "rev9" if output_stem is None else "rev10",
        "gate_label": "pilot" if output_stem is None else "gate10",
        "output_stem": output_stem,
        "analysis_source": {
            "path": diagnostic.EXPECTED_ANALYSIS_SOURCE,
            "sha256": diagnostic.file_sha256(ROOT / diagnostic.EXPECTED_ANALYSIS_SOURCE),
        },
        "tensorboard": {
            "path": "%USERPROFILE%\\logs\\g009-r0-rev9-tensorboard",
            "event_bundle_sha256": hashlib.sha256(event_payload.encode("utf-8")).hexdigest(),
            "event_files": event_files,
        },
        "checkpoint": {"sha256": diagnostic.file_sha256(checkpoint)},
        "training_report": {"sha256": diagnostic.file_sha256(training)},
        "qualification_block_reasons": [
            "diagnostic_pilot_never_qualifies",
            "hard_joint_limit_nonzero",
            "strict_success_zero",
            "prone_curriculum_boundary_leak",
        ],
    }
    analysis_doc["training_report"]["run_name"] = (
        analysis.EXPECTED_RUN_NAME
        if output_stem is None
        else "go2_flat_recover_rev10_prone_gate10_s42_fixture"
    )
    capture_path, analysis_path = tmp_path / "capture.json", tmp_path / "analysis.json"
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis_doc), encoding="utf-8")

    def resolve(value: str) -> Path:
        if value == portable_video:
            return video
        if value == "checkpoint.pt":
            return checkpoint
        if value == "training.json":
            return training
        if value == "%USERPROFILE%\\logs\\g009-r0-rev9-tensorboard":
            return tensorboard
        raise AssertionError(value)

    monkeypatch.setattr(diagnostic, "resolve_portable_path", resolve)
    monkeypatch.setattr(
        diagnostic,
        "git_blob_sha256_candidates",
        lambda _commit, relative: frozenset({diagnostic.file_sha256(ROOT / relative)}),
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_diagnostic_training_report",
        lambda *_args, **_kwargs: capture["training_binding"],
    )
    return capture_path, analysis_path, capture, analysis_doc


def test_diagnostic_capture_and_hash_binding_are_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture_path, analysis_path, _, _ = _capture_fixture(tmp_path, monkeypatch)
    capture, result, video = diagnostic.validate_capture(capture_path, analysis_path)
    assert capture["diagnostic_only"] is True
    assert result["qualification_allowed"] is False
    assert video.name == diagnostic.EXPECTED_LOCAL_NAME


def test_diagnostic_builder_preserves_single_playback_success_as_not_qualified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_path, analysis_path, capture, _ = _capture_fixture(tmp_path, monkeypatch)
    capture["metrics"]["stable_success"] = True
    capture["strict_success"] = 1
    capture["policy_result"] = "single_playback_success"
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    accepted, _, _ = diagnostic.validate_capture(capture_path, analysis_path)
    assert accepted["public_claim_eligible"] is False
    assert accepted["qualification_status"] == "not_run"
    assert "SINGLE PLAYBACK SUCCESS" in diagnostic._overlay_filter(diagnostic.DEFAULT_FONT, 1)


def test_diagnostic_builder_requires_integer_zero_strict_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_path, analysis_path, capture, _ = _capture_fixture(tmp_path, monkeypatch)
    capture["strict_success"] = False
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    with pytest.raises(ValueError, match="integer 0 or 1"):
        diagnostic.validate_capture(capture_path, analysis_path)


def test_diagnostic_builder_rejects_inconsistent_playback_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_path, analysis_path, capture, _ = _capture_fixture(tmp_path, monkeypatch)
    capture["metrics"]["stable_success"] = True
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    with pytest.raises(ValueError, match="strict_success/playback mismatch"):
        diagnostic.validate_capture(capture_path, analysis_path)


def test_diagnostic_builder_rejects_tampered_tensorboard_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_path, analysis_path, _, analysis_doc = _capture_fixture(tmp_path, monkeypatch)
    tensorboard_path = diagnostic.resolve_portable_path(analysis_doc["tensorboard"]["path"])
    next(tensorboard_path.glob("events.out.tfevents.*")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="event file hash mismatch"):
        diagnostic.validate_capture(capture_path, analysis_path)


def test_diagnostic_builder_rejects_official_capture_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture_path, analysis_path, capture, _ = _capture_fixture(tmp_path, monkeypatch)
    capture.update(status="complete", diagnostic_only=False, public_claim_eligible=True)
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic_complete"):
        diagnostic.validate_capture(capture_path, analysis_path)


def test_official_builder_rejects_diagnostic_capture_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostic_capture = {
        "goal_id": "g009",
        "stage_number": "G009-5",
        "stage_id": "R0",
        "status": "diagnostic_complete",
        "diagnostic_only": True,
    }
    capture_paths = [tmp_path / f"capture-{index}.json" for index in range(4)]
    monkeypatch.setattr(official, "_read_json", lambda path: diagnostic_capture)
    with pytest.raises(ValueError, match="status must be complete"):
        official.validate_capture_reports(capture_paths, tmp_path / "quantitative.json", tmp_path / "config.json")


def test_public_paths_are_isolated_under_diagnostic_namespace() -> None:
    assert diagnostic.DEFAULT_GIF.as_posix().endswith("docs/media/g009/R0/diagnostic/g009_5_r0_diag_rev9_01_prone.gif")
    assert diagnostic.DEFAULT_PNG.as_posix().endswith("docs/media/g009/R0/diagnostic/g009_5_r0_diag_rev9_01_prone_still.png")
    assert diagnostic.DEFAULT_SIDECAR.name == "g009_r0_diag_rev9_01_prone_visual_evidence.json"
    assert "four_pose_recovery" not in diagnostic.DEFAULT_GIF.name
    assert "fontsize=26" in diagnostic._overlay_filter(diagnostic.DEFAULT_FONT, 0)
    assert "y=ih-48" in diagnostic._overlay_filter(diagnostic.DEFAULT_FONT, 0)


def test_media_signatures_size_and_transaction_rollback(tmp_path: Path) -> None:
    gif, png = tmp_path / "a.gif", tmp_path / "b.png"
    gif.write_bytes(b"GIF89a" + b"x" * 8)
    png.write_bytes(diagnostic.PNG_SIGNATURE + b"x" * 8)
    diagnostic._validate_media(gif, "gif")
    diagnostic._validate_media(png, "png")
    staged, final = tmp_path / "new", tmp_path / "final"
    staged.write_bytes(b"new")
    final.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="injected"):
        diagnostic._publish_transaction(((staged, final),), lambda: (_ for _ in ()).throw(RuntimeError("injected")))
    assert final.read_bytes() == b"old"


def test_analysis_and_public_media_outputs_cannot_be_overwritten(tmp_path: Path) -> None:
    analysis_output = tmp_path / "analysis.json"
    analysis_output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        analysis.validate_new_output_path(analysis_output)
    existing = tmp_path / "existing.gif"
    existing.write_bytes(b"GIF89a")
    with pytest.raises(FileExistsError):
        diagnostic.validate_new_public_paths((tmp_path / "new.png", existing))


def test_diagnostic_cli_defaults_do_not_overlap_official_outputs() -> None:
    args = diagnostic.parse_args([])
    assert args.gif != official.REPO_ROOT / official.PUBLIC_GIF_PATH
    assert args.png != official.REPO_ROOT / official.PUBLIC_PNG_PATH
    assert args.sidecar != official.REPO_ROOT / official.SIDECAR_PATH
    assert isinstance(args, argparse.Namespace)


def test_diagnostic_media_paths_are_fixed(tmp_path: Path) -> None:
    args = diagnostic.parse_args(["--gif", str(tmp_path / "official-overwrite.gif")])
    with pytest.raises(ValueError, match="paths are fixed"):
        diagnostic.validate_fixed_paths(args)


def test_rev10_gate_media_identity_and_numbered_paths_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stem = "g009_5_r0_diag_rev10_gate10_01_prone"
    capture_path, analysis_path, _, analysis_doc = _capture_fixture(tmp_path, monkeypatch, stem)
    analysis_doc["qualification_block_reasons"] = ["diagnostic_pilot_never_qualifies"]
    analysis_path.write_text(json.dumps(analysis_doc), encoding="utf-8")
    capture, accepted_analysis, video = diagnostic.validate_capture(
        capture_path,
        analysis_path,
        revision="rev10",
        gate_label="gate10",
        output_stem=stem,
        expected_run_name="go2_flat_recover_rev10_prone_gate10_s42_fixture",
    )
    assert capture["public_claim_eligible"] is False
    assert accepted_analysis["qualification_allowed"] is False
    assert video.name == f"{stem}_s42.mp4"
    paths = diagnostic.expected_paths(stem)
    assert paths["gif"].name == f"{stem}.gif"
    assert paths["png"].name == f"{stem}_still.png"
    assert paths["capture_report"].name == f"{stem}_capture_s42.json"


def test_media_rejects_capture_training_binding_that_differs_from_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stem = "g009_5_r0_diag_rev10_gate10_01_prone"
    capture_path, analysis_path, capture, analysis_doc = _capture_fixture(tmp_path, monkeypatch, stem)
    analysis_doc["qualification_block_reasons"] = ["diagnostic_pilot_never_qualifies"]
    analysis_path.write_text(json.dumps(analysis_doc), encoding="utf-8")
    revalidated = json.loads(json.dumps(capture["training_binding"]))
    capture["training_binding"]["repository"] = {"commit": "b" * 40, "clean": True}
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    monkeypatch.setattr(diagnostic, "validate_diagnostic_training_report", lambda *_args, **_kwargs: revalidated)
    with pytest.raises(ValueError, match="does not match revalidation"):
        diagnostic.validate_capture(
            capture_path,
            analysis_path,
            revision="rev10",
            gate_label="gate10",
            output_stem=stem,
            expected_run_name="go2_flat_recover_rev10_prone_gate10_s42_fixture",
        )


def test_rev10_gate_cli_derives_all_outputs_and_rejects_partial_identity() -> None:
    stem = "g009_5_r0_diag_rev10_gate01_01_prone"
    args = diagnostic.parse_args(
        [
            "--revision", "rev10", "--gate-label", "gate01", "--output-stem", stem,
            "--expected-run-name", "go2_flat_recover_rev10_prone_gate01_s42_fixture",
        ]
    )
    expected = diagnostic.expected_paths(stem)
    for name, path in expected.items():
        assert getattr(args, name) == path
    diagnostic.validate_fixed_paths(args)
    with pytest.raises(ValueError, match="must be supplied together"):
        diagnostic.parse_args(["--revision", "rev10"])


@pytest.mark.parametrize("gate_label", ["gate01", "gate10", "gate50"])
def test_rev11_gate_cli_derives_exact_numbered_outputs(gate_label: str) -> None:
    stem = f"g009_5_r0_diag_rev11_{gate_label}_01_prone"
    run_name = f"go2_flat_recover_rev11_prone_{gate_label}_s42_fixture"
    args = diagnostic.parse_args(
        [
            "--revision", "rev11", "--gate-label", gate_label, "--output-stem", stem,
            "--expected-run-name", run_name,
        ]
    )
    diagnostic.validate_fixed_paths(args)
    expected = diagnostic.expected_paths(stem)
    assert args.capture_report == expected["capture_report"]
    assert args.analysis_report == expected["analysis_report"]
    assert args.gif == expected["gif"]
    assert args.png == expected["png"]
    assert args.summary == expected["summary"]
    assert args.sidecar == expected["sidecar"]


def test_rev11_media_rejects_run_name_with_only_substring_identity() -> None:
    stem = "g009_5_r0_diag_rev11_gate01_01_prone"
    args = diagnostic.parse_args(
        [
            "--revision", "rev11", "--gate-label", "gate01", "--output-stem", stem,
            "--expected-run-name", "prefix_go2_flat_recover_rev11_prone_gate01_s42_fixture",
        ]
    )
    with pytest.raises(ValueError, match="identity is not canonical"):
        diagnostic.validate_fixed_paths(args)


@pytest.mark.parametrize(
    "bad_stem",
    [
        "g009_5_r0_diag_rev10_gate01_01_prone_suffix",
        "g009_5_r0_diag_rev10_gate01_01_prone/../../escape",
    ],
)
def test_rev10_media_rejects_noncanonical_output_stem(bad_stem: str) -> None:
    args = diagnostic.parse_args(
        [
            "--revision", "rev10", "--gate-label", "gate01", "--output-stem", bad_stem,
            "--expected-run-name", "go2_flat_recover_rev10_prone_gate01_s42_fixture",
        ]
    )
    with pytest.raises(ValueError, match="not canonical"):
        diagnostic.validate_fixed_paths(args)


def test_committed_rev9_capture_and_analysis_validate_when_local_video_available() -> None:
    capture = json.loads(diagnostic.DEFAULT_CAPTURE.read_text(encoding="utf-8"))
    video = diagnostic.resolve_portable_path(capture["local_video"]["path"])
    if not video.is_file():
        pytest.skip("local-only rev9 MP4 is unavailable on this host")
    accepted_capture, accepted_analysis, accepted_video = diagnostic.validate_capture(
        diagnostic.DEFAULT_CAPTURE, diagnostic.DEFAULT_ANALYSIS
    )
    assert accepted_capture["capture_commit"] == "1ba2859d6817faa49f8d49465274ca00a4377efe"
    assert accepted_analysis["schema_version"] == "g009.r0.pilot_analysis.v2"
    assert accepted_video == video.resolve()
