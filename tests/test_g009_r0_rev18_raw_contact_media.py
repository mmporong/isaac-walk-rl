from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_g009_r0_rev18_raw_contact_media.py"
VALIDATOR_PATH = ROOT / "scripts/validate_g009_r0_rev18_raw_contact_media.py"
SPEC = importlib.util.spec_from_file_location("g009_rev18_raw_contact_media", BUILDER_PATH)
assert SPEC and SPEC.loader
MEDIA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEDIA)


def synthesis_fixture() -> dict:
    return json.loads(MEDIA.DEFAULT_INPUT.read_text(encoding="utf-8"))


def write_synthesis(path: Path, value: dict | None = None) -> Path:
    path.write_text(json.dumps(value or synthesis_fixture()), encoding="utf-8", newline="\n")
    return path


def test_numbered_defaults_labels_and_local_only_video() -> None:
    assert MEDIA.EVIDENCE_ID == "G009-5-E011"
    assert MEDIA.STAGE_NUMBER == "11"
    assert MEDIA.DEFAULT_VIDEO.as_posix().endswith("IsaacLab/logs/visual_evidence/g009/R0/diagnostic/g009_5_r0_e011_rev18_raw_contact_feasibility_s42.mp4")
    assert MEDIA.DEFAULT_PNG.as_posix().endswith("docs/media/g009/R0/diagnostic/g009_5_r0_e011_rev18_raw_contact_feasibility.png")
    assert MEDIA.DEFAULT_GIF.as_posix().endswith("docs/media/g009/R0/diagnostic/g009_5_r0_e011_rev18_raw_contact_feasibility.gif")
    for label in (
        "G009-5-E011",
        "11 · RAW CONTACT FEASIBILITY",
        "CPU 2/2 RAW PASS",
        "GPU 0/2 RAW CALLBACK AVAILABILITY",
        "DIAGNOSTIC-ONLY",
        "NOT PPO",
        "NOT QUALIFIED",
        "NO LEVER SELECTED",
        "PHYSICS GROUND TRUTH AUTHORITY: FALSE",
        "RESIDUAL INSTRUMENTATION: PARTIAL/UNAVAILABLE",
    ):
        assert label in MEDIA.LABELS
    assert MEDIA.DECISION_BANNER == "OUTCOME: UNAVAILABLE ON GPU · NO LEVER SELECTED"


def test_canonical_synthesis_and_four_reports_validate() -> None:
    value = MEDIA.read_summary(MEDIA.DEFAULT_INPUT)
    assert value["raw_contact_feasibility"]["outcome"] == "unavailable_on_gpu"
    assert len(value["_validated_reports"]) == 4
    assert [binding["path"] for binding in value["input_reports"]] == list(MEDIA.EXPECTED_REPORTS)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(status="failed"), "identity"),
        (lambda value: value["integrity"].update(hash_bound=False), "integrity"),
        (lambda value: value["raw_contact_feasibility"].update(outcome="gpu_pair_attribution_available"), "outcome"),
        (lambda value: value["decision"].update(selected_lever="contact_offset"), "decision"),
        (lambda value: value["governance"]["ppo"].update(status="complete"), "governance"),
        (lambda value: value["instrumentation_bundle"].update(status="complete"), "instrumentation"),
    ],
)
def test_synthesis_contract_fails_closed(tmp_path: Path, mutate, message: str) -> None:
    value = synthesis_fixture()
    mutate(value)
    path = write_synthesis(tmp_path / "bad.json", value)
    with pytest.raises(ValueError, match=message):
        MEDIA.read_summary(path)


def test_report_hash_and_order_are_exact(tmp_path: Path) -> None:
    bad_hash = synthesis_fixture()
    bad_hash["input_reports"][0]["sha256"] = "0" * 64
    path = write_synthesis(tmp_path / "hash.json", bad_hash)
    with pytest.raises(ValueError, match="hash mismatch"):
        MEDIA.read_summary(path)
    bad_order = synthesis_fixture()
    bad_order["input_reports"][0], bad_order["input_reports"][1] = bad_order["input_reports"][1], bad_order["input_reports"][0]
    path = write_synthesis(tmp_path / "order.json", bad_order)
    with pytest.raises(ValueError, match="order/path"):
        MEDIA.read_summary(path)


def test_transaction_refuses_overwrite_and_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staged_a, staged_b = tmp_path / "a.stage", tmp_path / "b.stage"
    final_a, final_b = tmp_path / "a.final", tmp_path / "b.final"
    staged_a.write_bytes(b"a")
    staged_b.write_bytes(b"b")
    final_a.write_bytes(b"owned")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        MEDIA._publish_transaction(((staged_a, final_a),), lambda: None)
    assert final_a.read_bytes() == b"owned"
    final_a.unlink()
    original = MEDIA._install_exclusive
    calls = 0

    def fail_second(staged: Path, final: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected")
        original(staged, final)

    monkeypatch.setattr(MEDIA, "_install_exclusive", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        MEDIA._publish_transaction(((staged_a, final_a), (staged_b, final_b)), lambda: None)
    assert not final_a.exists() and not final_b.exists()


def test_transaction_removes_partial_file_on_copy_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staged, final = tmp_path / "large.stage", tmp_path / "partial.final"
    staged.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    original_open = Path.open

    class FailingReader:
        def __init__(self) -> None:
            self.stream = original_open(staged, "rb")
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.stream.close()

        def read(self, size: int) -> bytes:
            self.calls += 1
            if self.calls == 2:
                raise OSError("mid-copy")
            return self.stream.read(size)

    def patched_open(path: Path, *args, **kwargs):
        if path == staged and args and args[0] == "rb":
            return FailingReader()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", patched_open)
    with pytest.raises(OSError, match="mid-copy"):
        MEDIA._publish_transaction(((staged, final),), lambda: None)
    assert not final.exists()


def test_rendered_frame_contains_exact_governance_and_scope_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes

    rendered: list[str] = []
    original = Axes.text

    def capture(self, x, y, text, *args, **kwargs):
        rendered.append(text)
        return original(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", capture)
    destination = tmp_path / "frame.png"
    MEDIA.render_frame(synthesis_fixture(), 1.0, destination)
    assert destination.read_bytes().startswith(MEDIA.PNG_SIGNATURE)
    for label in (
        MEDIA.DECISION_BANNER,
        "CPU 2/2 RAW PASS",
        "GPU 0/2 RAW CALLBACK AVAILABILITY",
        "PHYSICS GROUND TRUTH AUTHORITY: FALSE",
        "RESIDUAL INSTRUMENTATION: PARTIAL/UNAVAILABLE",
    ):
        assert label in rendered
    assert any("neither robot locomotion footage nor reinforcement-learning training evidence" in text for text in rendered)


def test_build_and_dedicated_validator_create_hash_bound_bundle() -> None:
    stem = f"test_rev18_media_{uuid.uuid4().hex}"
    input_path = MEDIA.RUNS_DIR / f"{stem}_input.json"
    video = MEDIA.LOCAL_VIDEO_DIR / f"{stem}.mp4"
    png = MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.png"
    gif = MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.gif"
    summary = MEDIA.RUNS_DIR / f"{stem}_summary.json"
    sidecar = MEDIA.RUNS_DIR / f"{stem}_sidecar.json"
    outputs = (input_path, video, png, gif, summary, sidecar)
    for path in outputs:
        path.unlink(missing_ok=True)
    input_path.write_bytes(MEDIA.DEFAULT_INPUT.read_bytes())
    try:
        value = MEDIA.build(input_path, video, png, gif, summary, sidecar)
        assert value["evidence_id"] == "G009-5-E011"
        assert value["stage_number"] == "11"
        assert value["diagnostic_only"] is True
        assert value["robot_locomotion_footage"] is False
        assert value["training_footage"] is False
        assert value["physics_ground_truth_authority"] is False
        assert value["decision"] == {"outcome": "unavailable_on_gpu", "selected_lever": None}
        assert video.read_bytes()[4:8] == b"ftyp"
        assert png.read_bytes().startswith(MEDIA.PNG_SIGNATURE)
        assert gif.read_bytes().startswith(MEDIA.GIF_SIGNATURE)
        assert png.stat().st_size < 10 * 1024 * 1024
        assert gif.stat().st_size < 10 * 1024 * 1024
        sidecar_value = json.loads(sidecar.read_text(encoding="utf-8"))
        assert sidecar_value["integrity"] == {"passed": True, "hash_bound": True}
        assert len(sidecar_value["provenance"]["source_binding"]["reports"]) == 4
        assert sidecar_value["provenance"]["source_binding"]["synthesis"]["sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
        assert sidecar_value["provenance"]["local_video"]["tracked_in_git"] is False
        completed = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH), "--check-only", "--sidecar", str(sidecar)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        receipt = json.loads(completed.stdout)
        assert receipt["status"] == "pass"
        assert receipt["stage_number"] == "11"
        assert "four_input_hashes" in receipt["checked"]
        sidecar_value["provenance"]["public_artifacts"]["png"]["sha256"] = "0" * 64
        sidecar.write_text(json.dumps(sidecar_value), encoding="utf-8")
        rejected = subprocess.run(
            [sys.executable, str(VALIDATOR_PATH), "--check-only", "--sidecar", str(sidecar)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert rejected.returncode == 1
        assert json.loads(rejected.stdout)["status"] == "fail"
        assert list((ROOT / "docs").rglob("*.mp4")) == []
    finally:
        for path in outputs:
            path.unlink(missing_ok=True)


def test_builder_refuses_existing_output_before_render(monkeypatch: pytest.MonkeyPatch) -> None:
    stem = f"test_rev18_existing_{uuid.uuid4().hex}"
    input_path = MEDIA.RUNS_DIR / f"{stem}_input.json"
    video = MEDIA.LOCAL_VIDEO_DIR / f"{stem}.mp4"
    png = MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.png"
    gif = MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.gif"
    summary = MEDIA.RUNS_DIR / f"{stem}_summary.json"
    sidecar = MEDIA.RUNS_DIR / f"{stem}_sidecar.json"
    outputs = (input_path, video, png, gif, summary, sidecar)
    for path in outputs:
        path.unlink(missing_ok=True)
    input_path.write_bytes(MEDIA.DEFAULT_INPUT.read_bytes())
    gif.parent.mkdir(parents=True, exist_ok=True)
    gif.write_bytes(b"owned")
    try:
        monkeypatch.setattr(MEDIA, "render_frame", lambda *_args: pytest.fail("render must not run"))
        with pytest.raises(ValueError, match="refusing to overwrite"):
            MEDIA.build(input_path, video, png, gif, summary, sidecar)
        assert gif.read_bytes() == b"owned"
    finally:
        for path in outputs:
            path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("video", ROOT / "docs/bad.mp4", "local video"),
        ("png", ROOT / "reports/bad.png", "public PNG"),
        ("gif", ROOT / "reports/bad.gif", "public GIF"),
        ("summary", ROOT / "docs/bad.json", "visual summary"),
        ("sidecar", ROOT / "docs/bad.json", "visual sidecar"),
    ],
)
def test_build_rejects_output_outside_policy(field: str, bad: Path, message: str) -> None:
    values = {
        "input": MEDIA.DEFAULT_INPUT,
        "video": MEDIA.LOCAL_VIDEO_DIR / f"policy_{uuid.uuid4().hex}.mp4",
        "png": MEDIA.PUBLIC_MEDIA_DIR / f"policy_{uuid.uuid4().hex}.png",
        "gif": MEDIA.PUBLIC_MEDIA_DIR / f"policy_{uuid.uuid4().hex}.gif",
        "summary": MEDIA.RUNS_DIR / f"policy_{uuid.uuid4().hex}.json",
        "sidecar": MEDIA.RUNS_DIR / f"policy_{uuid.uuid4().hex}.json",
    }
    values[field] = bad
    with pytest.raises(ValueError, match=message):
        MEDIA.build(values["input"], values["video"], values["png"], values["gif"], values["summary"], values["sidecar"])


def test_cli_exposes_all_explicit_paths() -> None:
    help_text = MEDIA.build_parser().format_help()
    for option in ("--input", "--video", "--png", "--gif", "--summary", "--sidecar", "--ffmpeg"):
        assert option in help_text
