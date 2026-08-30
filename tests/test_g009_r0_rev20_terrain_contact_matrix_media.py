from __future__ import annotations

import importlib.util
import copy
import json
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_g009_r0_rev20_terrain_contact_matrix_media.py"
SPEC = importlib.util.spec_from_file_location("rev20_media", BUILDER); assert SPEC and SPEC.loader
MEDIA = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MEDIA)


def load_validator():
    validator_spec = importlib.util.spec_from_file_location("rev20_validator", ROOT / "scripts/validate_g009_r0_rev20_terrain_contact_matrix_media.py"); assert validator_spec and validator_spec.loader
    validator = importlib.util.module_from_spec(validator_spec); validator_spec.loader.exec_module(validator)
    return validator


def test_cpu_defaults_and_phase_aware_cli() -> None:
    assert MEDIA.DEFAULT_VIDEO == Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic/g009_5_r0_e013_rev20_cpu_preflight_s42.mp4"
    assert MEDIA.DEFAULT_GIF.name == "g009_5_r0_e013_rev20_cpu_preflight.gif"
    assert MEDIA.DEFAULT_PNG.name == "g009_5_r0_e013_rev20_cpu_preflight.png"
    assert MEDIA.DEFAULT_SUMMARY.name.endswith("_cpu_preflight_visual_summary.json")
    assert MEDIA.DEFAULT_SIDECAR.name.endswith("_cpu_preflight_visual_evidence.json")
    assert MEDIA.phase_paths("final")["video"].name == "g009_5_r0_e013_rev20_final_cpu_gpu_s42.mp4"
    assert MEDIA.labels_for_phase("cpu-preflight")[0] == "13.01"
    assert MEDIA.labels_for_phase("final")[0] == "13.02"
    help_text = MEDIA.build_parser().format_help()
    for option in ("--phase", "--inputs", "--video", "--gif", "--png", "--summary", "--sidecar"): assert option in help_text


def test_canonical_cpu_inputs_are_revalidated_and_expose_required_telemetry() -> None:
    value = MEDIA.validate_inputs("cpu-preflight", (*MEDIA.CPU_REPORTS, MEDIA.CPU_PREFLIGHT))
    assert value["decision"] == "gpu_stage_authorized"
    assert value["repeatability"]["cpu"]["repeatable"] is True
    assert len(value["input_bindings"]) == 3 and len(value["reports"]) == 2
    for report in value["reports"]:
        assert report["availability"] == "observed_valid"
        assert all(report[key] for key in ("structural_passed", "safety_passed", "overlap_passed", "baseline_passed", "live_readback_passed"))
        assert len(report["per_env_overlap_coverage_steps"]) == 8
        assert report["peak_force_n"] > 0 and report["max_non_foot_force_bw"] <= 15
        assert report["raw_filter_paths_sha256"] != report["logical_filter_paths_sha256"]


@pytest.mark.parametrize("mutation,message", [
    (lambda value: value["cpu_preflight"].update(passed=False), "CPU preflight"),
    (lambda value: value["decision"]["repeatability"].update(repeatable=False), "CPU preflight"),
    (lambda value: value["input_reports"][0].update(sha256="0" * 64), "CPU input report hash"),
])
def test_cpu_preflight_tampering_fails_closed(tmp_path: Path, mutation, message: str) -> None:
    value = json.loads(MEDIA.CPU_PREFLIGHT.read_text(encoding="utf-8")); mutation(value)
    with pytest.raises(ValueError, match=message):
        MEDIA.synthesis.probe.validate_cpu_preflight_value(value, ROOT, MEDIA.repo_path(MEDIA.CPU_PREFLIGHT), None)


@pytest.mark.parametrize("mutation,message", [
    (lambda value: value["integrity"].update(git_commit="0" * 40), "integrity"),
    (lambda value: value.update(governance={}), "governance"),
])
def test_cpu_preflight_integrity_commit_and_governance_tamper_fail_closed(mutation, message: str) -> None:
    value = json.loads(MEDIA.CPU_PREFLIGHT.read_text(encoding="utf-8")); mutation(value)
    with pytest.raises(ValueError, match=message):
        MEDIA.synthesis.probe.validate_cpu_preflight_value(value, ROOT, MEDIA.repo_path(MEDIA.CPU_PREFLIGHT), None)


def test_current_equivalence_allows_different_head_when_bound_files_match() -> None:
    preflight = json.loads(MEDIA.CPU_PREFLIGHT.read_text(encoding="utf-8")); historical = copy.deepcopy(preflight["synthesis_source_bundle"])
    historical["git_commit"] = "0" * 40
    observed = MEDIA.verify_current_file_equivalence(
        historical,
        MEDIA.synthesis.probe.SYNTHESIS_SOURCE_BINDING_PATHS,
        current_files=historical["source_binding_files"],
        dirty_paths=[],
    )
    assert observed == historical["source_binding_files"]


def test_current_equivalence_rejects_changed_bound_file() -> None:
    preflight = json.loads(MEDIA.CPU_PREFLIGHT.read_text(encoding="utf-8")); historical = preflight["synthesis_source_bundle"]
    changed = dict(historical["source_binding_files"]); changed[next(iter(changed))] = "0" * 64
    with pytest.raises(ValueError, match="file hashes"):
        MEDIA.verify_current_file_equivalence(historical, MEDIA.synthesis.probe.SYNTHESIS_SOURCE_BINDING_PATHS, current_files=changed, dirty_paths=[])


def test_current_equivalence_rejects_dirty_bound_path() -> None:
    preflight = json.loads(MEDIA.CPU_PREFLIGHT.read_text(encoding="utf-8")); historical = preflight["synthesis_source_bundle"]
    with pytest.raises(ValueError, match="must be clean"):
        MEDIA.verify_current_file_equivalence(historical, MEDIA.synthesis.probe.SYNTHESIS_SOURCE_BINDING_PATHS, current_files=historical["source_binding_files"], dirty_paths=[" M scripts/probe.py"])


def test_every_frame_contains_number_and_claim_limit_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes
    data = MEDIA.validate_inputs("cpu-preflight", (*MEDIA.CPU_REPORTS, MEDIA.CPU_PREFLIGHT))
    rendered: list[str] = []; original = Axes.text
    def capture(self, x, y, text, *args, **kwargs): rendered.append(text); return original(self, x, y, text, *args, **kwargs)
    monkeypatch.setattr(Axes, "text", capture)
    output = tmp_path / "frame.png"; MEDIA.render_frame(data, 0.5, output)
    combined = " ".join(rendered)
    for label in MEDIA.REQUIRED_LABELS: assert label in combined
    assert "CLAIM LIMIT: NOT A LOCOMOTION, TRAINING, QUALIFICATION, OR PHYSICS-GROUND-TRUTH CLAIM" in combined
    assert output.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("phase", ["cpu-preflight", "final"])
def test_footer_text_bbox_stays_inside_1280x720_frame(phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = (*MEDIA.CPU_REPORTS, MEDIA.CPU_PREFLIGHT) if phase == "cpu-preflight" else (*MEDIA.FINAL_REPORTS, MEDIA.CPU_PREFLIGHT, MEDIA.FINAL_SYNTHESIS)
    data = MEDIA.validate_inputs(phase, inputs)
    observed: list[tuple[str, float, float, float, float]] = []
    original = MEDIA.validate_text_bounds
    def capture(figure, artists, label):
        original(figure, artists, label); renderer = figure.canvas.get_renderer()
        for artist in artists:
            bounds = artist.get_window_extent(renderer=renderer)
            observed.append((artist.get_text(), bounds.x0, bounds.y0, bounds.x1, bounds.y1))
    monkeypatch.setattr(MEDIA, "validate_text_bounds", capture)
    output = tmp_path / f"{phase}.png"; MEDIA.render_frame(data, 1.0, output)
    assert len(observed) == 2
    assert observed[0][0].startswith("OUTCOME:")
    assert observed[1][0] == "CLAIM LIMIT: NOT A LOCOMOTION, TRAINING, QUALIFICATION, OR PHYSICS-GROUND-TRUTH CLAIM"
    assert all(0 <= x0 < x1 <= MEDIA.WIDTH and 0 <= y0 < y1 <= MEDIA.HEIGHT for _, x0, y0, x1, y1 in observed)


def test_final_failure_render_never_claims_all_pass_or_gpu_authorized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = MEDIA.validate_inputs("cpu-preflight", (*MEDIA.CPU_REPORTS, MEDIA.CPU_PREFLIGHT))
    failed = copy.deepcopy(data); failed["phase"] = "final"; failed["decision"] = "safety_limit_exceeded"
    failed["reports"] = failed["reports"] + copy.deepcopy(failed["reports"])
    failed["reports"][2]["safety_passed"] = False
    failed["repeatability"] = {"cpu": {"repeatable": True}, "cuda:0": {"repeatable": False}}
    lines = MEDIA.render_status_lines(failed)
    assert "SAFETY: FAIL" in lines["checks"] and "CUDA:0 REPEATABILITY: FAIL" in lines["repeatability"]
    assert "GPU STAGE AUTHORIZED" not in " ".join(lines.values())
    from matplotlib.axes import Axes
    rendered: list[str] = []; original = Axes.text
    def capture(self, x, y, text, *args, **kwargs): rendered.append(text); return original(self, x, y, text, *args, **kwargs)
    monkeypatch.setattr(Axes, "text", capture)
    output = tmp_path / "failed-final.png"; MEDIA.render_frame(failed, 1.0, output)
    combined = " ".join(rendered)
    assert "13.02" in combined and "SAFETY: FAIL" in combined and "OUTCOME: SAFETY LIMIT EXCEEDED" in combined
    assert "GPU STAGE AUTHORIZED" not in combined


def test_build_is_hash_bound_no_overwrite_and_validator_passes() -> None:
    token = uuid.uuid4().hex
    outputs = {"video": MEDIA.LOCAL_VIDEO_DIR / f"test_rev20_{token}.mp4", "gif": MEDIA.PUBLIC_MEDIA_DIR / f"test_rev20_{token}.gif", "png": MEDIA.PUBLIC_MEDIA_DIR / f"test_rev20_{token}.png", "summary": MEDIA.RUNS_DIR / f"test_rev20_{token}_summary.json", "sidecar": MEDIA.RUNS_DIR / f"test_rev20_{token}_sidecar.json"}
    try:
        MEDIA.build("cpu-preflight", (*MEDIA.CPU_REPORTS, MEDIA.CPU_PREFLIGHT), outputs)
        assert all(path.exists() for path in outputs.values())
        with pytest.raises(ValueError, match="overwrite"): MEDIA.build("cpu-preflight", (*MEDIA.CPU_REPORTS, MEDIA.CPU_PREFLIGHT), outputs)
    finally:
        for path in outputs.values(): path.unlink(missing_ok=True)


def test_canonical_bundle_validates_when_present() -> None:
    if MEDIA.DEFAULT_VIDEO.exists():
        validator = load_validator()
        assert validator.validate_bundle("cpu-preflight")["status"] == "pass"
    else:
        assert MEDIA.DEFAULT_GIF.is_file() and MEDIA.DEFAULT_PNG.is_file()
        assert MEDIA.DEFAULT_SUMMARY.is_file() and MEDIA.DEFAULT_SIDECAR.is_file()


def test_validator_rejects_forged_public_path(tmp_path: Path) -> None:
    if not MEDIA.DEFAULT_VIDEO.exists(): return
    validator = load_validator(); value = json.loads(MEDIA.DEFAULT_SIDECAR.read_text(encoding="utf-8"))
    value["artifacts"]["public"]["png"]["path"] = "docs/media/g009/R0/diagnostic/forged.png"
    sidecar = MEDIA.RUNS_DIR / f"test_rev20_forged_{uuid.uuid4().hex}.json"
    try:
        sidecar.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError, match="canonical path"): validator.validate_bundle("cpu-preflight", sidecar)
    finally: sidecar.unlink(missing_ok=True)


@pytest.mark.parametrize("mutation,message", [
    (lambda value: value.update(phase="final"), "identity"),
    (lambda value: value["decision"].update(outcome="safety_limit_exceeded"), "decision"),
    (lambda value: value.update(source_bundle_sha256="0" * 64), "source bundle"),
    (lambda value: value["public_artifacts"]["gif"].update(frame_count=7), "artifact metadata"),
    (lambda value: value["local_video"].update(duration_seconds=1.0), "artifact metadata"),
])
def test_validator_rejects_summary_core_tamper(monkeypatch: pytest.MonkeyPatch, mutation, message: str) -> None:
    if not MEDIA.DEFAULT_VIDEO.exists(): return
    validator = load_validator(); tampered = json.loads(MEDIA.DEFAULT_SUMMARY.read_text(encoding="utf-8")); mutation(tampered)
    original = validator.media.read_json
    def fake_read(path: Path):
        if path.resolve() == MEDIA.DEFAULT_SUMMARY.resolve(): return tampered, b"tampered"
        return original(path)
    monkeypatch.setattr(validator.media, "read_json", fake_read)
    with pytest.raises(ValueError, match=message): validator.validate_bundle("cpu-preflight")
