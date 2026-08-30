from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_g009_r0_rev19_contact_offset_media.py"
SPEC = importlib.util.spec_from_file_location("g009_rev19_contact_offset_media", BUILDER)
assert SPEC and SPEC.loader
MEDIA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEDIA)


def test_defaults_continue_e012_numbering_and_keep_video_local() -> None:
    assert MEDIA.EVIDENCE_ID == "G009-5-E012"
    assert MEDIA.STAGE_NUMBER == "12"
    assert MEDIA.DEFAULT_VIDEO.as_posix().endswith("IsaacLab/logs/visual_evidence/g009/R0/diagnostic/g009_5_r0_e012_rev19_contact_offset_intervention_s42.mp4")
    assert MEDIA.DEFAULT_PREFLIGHT_PNG.name.endswith("_01_cpu_preflight.png")
    assert MEDIA.DEFAULT_FINAL_PNG.name.endswith("_02_final_outcome.png")
    assert MEDIA.REQUIRED_LABELS == (
        "TELEMETRY ANIMATION", "NOT CAMERA FOOTAGE", "DIAGNOSTIC ONLY", "NO PPO", "NOT QUALIFIED"
    )


def test_canonical_ten_input_contract_and_telemetry_validate() -> None:
    value = MEDIA.validate_inputs(MEDIA.DEFAULT_SYNTHESIS, MEDIA.DEFAULT_PREFLIGHT)
    assert len(value["runs"]) == 8
    assert [item["slot"] for item in value["runs"]] == list(MEDIA.EXPECTED_SLOTS)
    assert [item["callback_available"] for item in value["runs"]] == [True] * 4 + [False] * 4
    assert [item["contact_offset_scale"] for item in value["runs"]] == [1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5]
    assert all(item["safety_passed"] and item["repeatability_passed"] for item in value["runs"])
    assert all(item["force_body_weight"] <= 15.0 for item in value["runs"])
    assert all(item["cpu_minimum_separation_m"] is not None for item in value["runs"][:4])
    assert all(item["cpu_minimum_separation_m"] is None for item in value["runs"][4:])


def test_phase_series_keep_bar_color_and_label_counts_aligned() -> None:
    data = MEDIA.validate_inputs(MEDIA.DEFAULT_SYNTHESIS, MEDIA.DEFAULT_PREFLIGHT)
    for progress, expected_count in ((0.25, 4), (0.5, 4), (0.75, 8), (1.0, 8)):
        series = MEDIA.phase_series(data, progress)
        assert {len(series[key]) for key in ("labels", "callback", "force", "colors")} == {expected_count}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["decision"].update(selected_lever="contact_offset"), "decision"),
        (lambda value: value["governance"]["ppo"].update(updates=1), "governance"),
        (lambda value: value["input_reports"][0].update(sha256="0" * 64), "hash"),
        (lambda value: value.update(input_reports=None), "eight object run bindings"),
        (lambda value: value["raw_callback_observation"]["repeatability"]["A.cuda:0"].update(repeatable=False), "repeatability contract"),
    ],
)
def test_synthesis_tampering_fails_closed(tmp_path: Path, mutation, message: str) -> None:
    value = json.loads(MEDIA.DEFAULT_SYNTHESIS.read_text(encoding="utf-8"))
    mutation(value)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        MEDIA.validate_inputs(path, MEDIA.DEFAULT_PREFLIGHT)


def test_every_rendered_frame_contains_scope_labels_and_numbered_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes

    data = MEDIA.validate_inputs(MEDIA.DEFAULT_SYNTHESIS, MEDIA.DEFAULT_PREFLIGHT)
    for progress, expected_phase in ((0.5, "01 · CPU PREFLIGHT"), (1.0, "02 · FINAL CPU→GPU 2×2×2")):
        rendered: list[str] = []
        original = Axes.text

        def capture(self, x, y, text, *args, **kwargs):
            rendered.append(text)
            return original(self, x, y, text, *args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(Axes, "text", capture)
            destination = tmp_path / f"frame_{progress}.png"
            MEDIA.render_frame(data, progress, destination)
        assert destination.read_bytes().startswith(b"\x89PNG")
        combined = " ".join(rendered)
        assert expected_phase in combined
        for label in MEDIA.REQUIRED_LABELS:
            assert label in combined


def test_build_creates_two_phase_hash_bound_bundle_and_local_mp4() -> None:
    token = uuid.uuid4().hex
    video = MEDIA.LOCAL_VIDEO_DIR / f"test_e012_{token}.mp4"
    preflight_png = MEDIA.PUBLIC_MEDIA_DIR / f"test_e012_{token}_01.png"
    final_png = MEDIA.PUBLIC_MEDIA_DIR / f"test_e012_{token}_02.png"
    gif = MEDIA.PUBLIC_MEDIA_DIR / f"test_e012_{token}.gif"
    sidecar = MEDIA.RUNS_DIR / f"test_e012_{token}.json"
    outputs = (video, preflight_png, final_png, gif, sidecar)
    for path in outputs:
        path.unlink(missing_ok=True)
    try:
        value = MEDIA.build(MEDIA.DEFAULT_SYNTHESIS, MEDIA.DEFAULT_PREFLIGHT, video, preflight_png, final_png, gif, sidecar)
        assert value["integrity"] == {"passed": True, "hash_bound": True, "input_binding_count": 10, "all_inputs_verified": True}
        assert [phase["number"] for phase in value["sequence"]] == ["12.01", "12.02"]
        assert len(value["input_bindings"]) == 10
        assert value["decision"]["gpu_contact_absence_claimed"] is False
        assert value["local_video"]["tracked_in_git"] is False
        assert value["local_video"]["codec"] == "h264"
        assert value["public_artifacts"]["gif"]["frames"] == MEDIA.FRAME_COUNT
        assert value["public_artifacts"]["preflight_png"]["phase"] == "12.01_cpu_preflight"
        assert value["public_artifacts"]["final_png"]["phase"] == "12.02_final_cpu_gpu_2x2x2"
        assert MEDIA.validate_bundle(
            sidecar,
            video_path=video,
            preflight_png_path=preflight_png,
            final_png_path=final_png,
            gif_path=gif,
        )["status"] == "diagnostic_complete"
        assert video.read_bytes()[4:8] == b"ftyp"
        assert preflight_png.stat().st_size < MEDIA.MAX_PUBLIC_BYTES
        assert final_png.stat().st_size < MEDIA.MAX_PUBLIC_BYTES
        assert gif.stat().st_size < MEDIA.MAX_PUBLIC_BYTES
        assert list((ROOT / "docs").rglob("*.mp4")) == []
    finally:
        for path in outputs:
            path.unlink(missing_ok=True)


def test_build_rolls_back_all_outputs_when_final_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = uuid.uuid4().hex
    video = MEDIA.LOCAL_VIDEO_DIR / f"test_e012_rollback_{token}.mp4"
    preflight_png = MEDIA.PUBLIC_MEDIA_DIR / f"test_e012_rollback_{token}_01.png"
    final_png = MEDIA.PUBLIC_MEDIA_DIR / f"test_e012_rollback_{token}_02.png"
    gif = MEDIA.PUBLIC_MEDIA_DIR / f"test_e012_rollback_{token}.gif"
    sidecar = MEDIA.RUNS_DIR / f"test_e012_rollback_{token}.json"
    outputs = (video, preflight_png, final_png, gif, sidecar)

    def fail_validation(*args, **kwargs):
        raise ValueError("forced final validation failure")

    monkeypatch.setattr(MEDIA, "validate_bundle", fail_validation)
    with pytest.raises(ValueError, match="forced final validation failure"):
        MEDIA.build(
            MEDIA.DEFAULT_SYNTHESIS,
            MEDIA.DEFAULT_PREFLIGHT,
            video,
            preflight_png,
            final_png,
            gif,
            sidecar,
        )
    assert all(not path.exists() for path in outputs)


def test_validator_rejects_input_hash_tamper(tmp_path: Path) -> None:
    value = json.loads(MEDIA.DEFAULT_SIDECAR.read_text(encoding="utf-8"))
    value["input_bindings"][0]["sha256"] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="input hash"):
        MEDIA.validate_bundle(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["integrity"].update(all_inputs_verified=False), "integrity contract"),
        (lambda value: value["telemetry"].update(safety="0/8 pass"), "telemetry contract"),
        (lambda value: value["decision"].update(selected_lever="contact_offset"), "decision contract"),
        (lambda value: value["governance"]["ppo"].update(updates=1), "governance contract"),
        (lambda value: value["public_artifacts"]["gif"].update(path="docs/media/g009/R0/diagnostic/other.gif"), "public artifact contract"),
        (lambda value: value["local_video"].update(path="%USERPROFILE%\\other.mp4"), "local video contract"),
    ],
)
def test_validator_rejects_sidecar_contract_tamper(tmp_path: Path, mutation, message: str) -> None:
    value = json.loads(MEDIA.DEFAULT_SIDECAR.read_text(encoding="utf-8"))
    mutation(value)
    path = tmp_path / "tampered-sidecar.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        MEDIA.validate_bundle(path)


def test_cli_exposes_phase_outputs_and_check_mode() -> None:
    help_text = MEDIA.build_parser().format_help()
    for option in ("--synthesis", "--preflight", "--video", "--preflight-png", "--final-png", "--gif", "--sidecar", "--check-only"):
        assert option in help_text
