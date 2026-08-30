from __future__ import annotations

import hashlib
import importlib.util
import json
import uuid
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/build_g009_r0_rev17_mechanism_media.py"
    spec = importlib.util.spec_from_file_location("rev17_mechanism_media", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MEDIA = load_module()
VALIDATOR_PATH = ROOT / "scripts/validate_g009_r0_rev17_mechanism_media.py"


def fixture() -> dict:
    predecessor = json.loads(
        (ROOT / "reports/runs/g009_r0_rev16_synthesis_12_full_retry01_s42.json").read_text(encoding="utf-8")
    )
    predecessor_path = "reports/runs/g009_r0_rev16_synthesis_12_full_retry01_s42.json"
    predecessor_sha = hashlib.sha256((ROOT / predecessor_path).read_bytes()).hexdigest()
    runs = []
    binding_index = 0
    for arm, device, peak, window, share in (
        ("A", "cpu", 0.9, 2.1, 0.48),
        ("A", "cuda:0", 0.8, 2.0, 0.45),
        ("B", "cpu", 1.3, 2.8, 0.67),
        ("B", "cuda:0", 1.7, 3.1, 0.79),
    ):
        for replicate in (1, 2, 3):
            evidence = predecessor["input_reports"][binding_index]
            binding_index += 1
            runs.append(
                {
                    "evidence": evidence,
                    "arm": arm,
                    "device": device,
                    "replicate_index": replicate,
                    "peak_window": {
                        "peak_base_impulse_n_s": peak,
                        "window_base_impulse_n_s": window,
                        "body_impulse_magnitude_totals_n_s": {
                            "base": share,
                            "FL_hip": 1.0 - share,
                        },
                    },
                    "contact_authority": {
                        "authority": "cpu_only",
                        "availability": "observed" if device == "cpu" else "unavailable_on_gpu",
                        "topology_available": device == "cpu",
                        "body_pair_counts": {} if device == "cpu" else None,
                        "per_physics_step": (
                            {
                                str(step): {
                                    "event_count": 0,
                                    "header_count": 0,
                                    "contact_point_count": 0,
                                    "reported_impulse_vector_sum_n_s": [0.0, 0.0, 0.0],
                                    "body_pair_counts": {},
                                    "minimum_separation_m": None,
                                }
                                for step in (128, 129, 130)
                            }
                            if device == "cpu"
                            else None
                        ),
                        "per_physics_step_status": (
                            "observed_cpu_authority"
                            if device == "cpu"
                            else "unavailable_on_gpu"
                        ),
                    },
                }
            )
    return {
        "schema_version": "g009.r0.rev17.mechanism_split.v1",
        "evidence_id": "G009-5-E010",
        "status": "pass",
        "diagnostic_only": True,
        "integrity": {
            "passed": True,
            "hash_bound": True,
            "predecessor_path": predecessor_path,
            "predecessor_sha256": predecessor_sha,
            "input_report_count": 12,
            "input_reports": predecessor["input_reports"],
        },
        "input_reports": predecessor["input_reports"],
        "ppo": {"allowed": False, "status": "not_run"},
        "qualification": {"eligible": False, "status": "not_run", "passed": None},
        "mechanism_split": {
            "direct_observations": {"runs": runs},
            "temporal_signatures": {},
            "causal_inferences": {},
            "decision": {"outcome": "inconclusive", "selected_lever": None},
        },
    }


def write_fixture(path: Path, value: dict | None = None) -> Path:
    path.write_text(json.dumps(value or fixture()), encoding="utf-8")
    return path


def test_defaults_and_required_labels_are_numbered_and_local_only() -> None:
    assert MEDIA.EVIDENCE_ID == "G009-5-E010"
    assert MEDIA.DEFAULT_VIDEO.suffix == ".mp4"
    assert MEDIA.DEFAULT_VIDEO.is_relative_to(Path.home())
    assert MEDIA.DEFAULT_PNG.as_posix().endswith("docs/media/g009/R0/diagnostic/g009_5_r0_e010_rev17_mechanism_split.png")
    for label in (
        "G009-5-E010",
        "MECHANISM DIAGNOSTIC",
        "INCONCLUSIVE",
        "NO LEVER SELECTED",
        "NOT PPO",
        "NOT QUALIFIED",
        "CPU CONTACT AUTHORITY ONLY",
        "GPU CONTACT TOPOLOGY UNAVAILABLE",
    ):
        assert label in MEDIA.LABELS
    assert MEDIA.DECISION_BANNER == "OUTCOME: INCONCLUSIVE · NO LEVER SELECTED"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(status="rejected"), "not PASS"),
        (lambda value: value["integrity"].update(passed=False), "integrity"),
        (lambda value: value["integrity"].update(hash_bound=False), "hash-bound"),
        (lambda value: value["ppo"].update(status="complete"), "governance"),
        (lambda value: value["qualification"].update(status="pass"), "governance"),
        (lambda value: value["mechanism_split"]["direct_observations"]["runs"][3]["contact_authority"].update(topology_available=True), "GPU contact topology"),
    ],
)
def test_input_contract_fails_closed(tmp_path: Path, mutate, message: str) -> None:
    value = fixture()
    mutate(value)
    path = write_fixture(tmp_path / "bad.json", value)
    with pytest.raises(ValueError, match=message):
        MEDIA.read_summary(path)


def test_transaction_rejects_overwrite_and_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_stage = tmp_path / "first.stage"
    second_stage = tmp_path / "second.stage"
    first_stage.write_bytes(b"first")
    second_stage.write_bytes(b"second")
    first_final = tmp_path / "first.final"
    second_final = tmp_path / "second.final"
    first_final.write_bytes(b"owned")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        MEDIA._publish_transaction(((first_stage, first_final),), lambda: None)
    assert first_final.read_bytes() == b"owned"

    first_final.unlink()
    calls = 0
    original = MEDIA._install_exclusive

    def fail_second(staged: Path, final: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected")
        original(staged, final)

    monkeypatch.setattr(MEDIA, "_install_exclusive", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        MEDIA._publish_transaction(
            ((first_stage, first_final), (second_stage, second_final)),
            lambda: None,
        )
    assert not first_final.exists() and not second_final.exists()


def test_transaction_removes_partial_file_when_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "large.stage"
    staged.write_bytes(b"x" * (2 * 1024 * 1024 + 7))
    final = tmp_path / "partial.final"
    original_open = Path.open

    class FailingReader:
        def __init__(self) -> None:
            self.stream = original_open(staged, "rb")
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size: int) -> bytes:
            self.calls += 1
            if self.calls == 2:
                raise OSError("injected mid-copy failure")
            return self.stream.read(size)

    def patched_open(path: Path, *args, **kwargs):
        if path == staged and args and args[0] == "rb":
            return FailingReader()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", patched_open)
    with pytest.raises(OSError, match="mid-copy"):
        MEDIA._publish_transaction(((staged, final),), lambda: None)
    assert not final.exists()


def test_transaction_removes_created_file_when_staged_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "open.stage"
    staged.write_bytes(b"payload")
    final = tmp_path / "open.final"
    original_open = Path.open

    def patched_open(path: Path, *args, **kwargs):
        if path == staged and args and args[0] == "rb":
            raise OSError("injected staged-open failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", patched_open)
    with pytest.raises(OSError, match="staged-open"):
        MEDIA._publish_transaction(((staged, final),), lambda: None)
    assert not final.exists()


def test_transaction_removes_partial_file_when_flush_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "flush.stage"
    staged.write_bytes(b"complete payload")
    final = tmp_path / "flush.final"
    original_open = Path.open

    class FailingWriter:
        def __init__(self) -> None:
            self.stream = original_open(final, "xb")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, value: bytes) -> int:
            return self.stream.write(value)

        def flush(self) -> None:
            raise OSError("injected flush failure")

        def fileno(self) -> int:
            return self.stream.fileno()

    def patched_open(path: Path, *args, **kwargs):
        if path == final and args and args[0] == "xb":
            return FailingWriter()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", patched_open)
    with pytest.raises(OSError, match="flush"):
        MEDIA._publish_transaction(((staged, final),), lambda: None)
    assert not final.exists()


def test_transaction_removes_partial_file_when_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "fsync.stage"
    staged.write_bytes(b"complete payload")
    final = tmp_path / "fsync.final"
    monkeypatch.setattr(
        MEDIA.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("injected fsync failure")),
    )
    with pytest.raises(OSError, match="fsync"):
        MEDIA._publish_transaction(((staged, final),), lambda: None)
    assert not final.exists()


def test_rendered_frame_contains_exact_no_lever_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.axes import Axes

    destination = tmp_path / "frame.png"
    rendered_text: list[str] = []
    original_text = Axes.text

    def capture_text(self, x, y, text, *args, **kwargs):
        rendered_text.append(text)
        return original_text(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", capture_text)
    MEDIA.render_frame(fixture(), 1.0, destination)
    assert destination.read_bytes().startswith(MEDIA.PNG_SIGNATURE)
    assert MEDIA.DECISION_BANNER in rendered_text


def test_build_creates_hash_bound_outputs_without_public_mp4(tmp_path: Path) -> None:
    stem = f"test_rev17_media_{uuid.uuid4().hex}"
    input_path = MEDIA.RUNS_DIR / f"{stem}_input.json"
    video = MEDIA.LOCAL_VIDEO_DIR / f"{stem}.mp4"
    png = MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.png"
    gif = MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.gif"
    summary = MEDIA.RUNS_DIR / f"{stem}_summary.json"
    sidecar = MEDIA.RUNS_DIR / f"{stem}_sidecar.json"
    outputs = (input_path, video, png, gif, summary, sidecar)
    for path in outputs:
        path.unlink(missing_ok=True)
    write_fixture(input_path)
    try:
        value = MEDIA.build(input_path, video, png, gif, summary, sidecar)
        assert value["evidence_id"] == "G009-5-E010"
        assert value["diagnostic_only"] is True
        assert value["learned_policy_qualified"] is False
        assert value["decision"] == {
            "outcome": "inconclusive",
            "selected_lever": None,
        }
        assert value["governance"] == {
            "ppo": {"status": "not_run"},
            "qualification": {"status": "not_run"},
        }
        assert video.is_file() and video.read_bytes()[4:8] == b"ftyp"
        assert png.read_bytes().startswith(MEDIA.PNG_SIGNATURE)
        assert gif.read_bytes().startswith(MEDIA.GIF_SIGNATURE)
        assert png.stat().st_size < 10 * 1024 * 1024
        assert gif.stat().st_size < 10 * 1024 * 1024
        sidecar_value = json.loads(sidecar.read_text(encoding="utf-8"))
        assert sidecar_value["integrity"] == {"passed": True, "hash_bound": True}
        assert sidecar_value["decision"] == value["decision"]
        assert sidecar_value["provenance"]["input"]["sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
        assert sidecar_value["provenance"]["public_artifacts"]["png"]["sha256"] == hashlib.sha256(png.read_bytes()).hexdigest()
        assert sidecar_value["provenance"]["public_artifacts"]["gif"]["sha256"] == hashlib.sha256(gif.read_bytes()).hexdigest()
        assert sidecar_value["provenance"]["local_video"]["tracked_in_git"] is False
        assert sidecar_value["goal_id"] == "g009"
        assert sidecar_value["stage_id"] == "R0"
        assert sidecar_value["stage_number"] == "10"
        assert sidecar_value["contract"]["standard_stage_validator"]["compatible"] is False
        assert sidecar_value["contract"]["builder_source"]["sha256"] == hashlib.sha256(
            MEDIA.BUILDER_SOURCE.read_bytes()
        ).hexdigest()
        assert sidecar_value["contract"]["dedicated_validator"]["sha256"] == hashlib.sha256(
            VALIDATOR_PATH.read_bytes()
        ).hexdigest()
        completed = __import__("subprocess").run(
            [
                __import__("sys").executable,
                str(VALIDATOR_PATH),
                "--check-only",
                "--sidecar",
                str(sidecar),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        receipt = json.loads(completed.stdout)
        assert receipt["status"] == "pass"
        assert receipt["decision"] == value["decision"]
        sidecar_value["provenance"]["public_artifacts"]["png"]["sha256"] = "0" * 64
        sidecar.write_text(json.dumps(sidecar_value), encoding="utf-8")
        rejected = __import__("subprocess").run(
            [
                __import__("sys").executable,
                str(VALIDATOR_PATH),
                "--check-only",
                "--sidecar",
                str(sidecar),
            ],
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


def test_builder_refuses_any_existing_output_before_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stem = f"test_rev17_existing_{uuid.uuid4().hex}"
    input_path = MEDIA.RUNS_DIR / f"{stem}_input.json"
    video = MEDIA.LOCAL_VIDEO_DIR / f"{stem}.mp4"
    png = MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.png"
    gif = MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.gif"
    summary = MEDIA.RUNS_DIR / f"{stem}_summary.json"
    sidecar = MEDIA.RUNS_DIR / f"{stem}_sidecar.json"
    outputs = (input_path, video, png, gif, summary, sidecar)
    for path in outputs:
        path.unlink(missing_ok=True)
    write_fixture(input_path)
    gif.parent.mkdir(parents=True, exist_ok=True)
    gif.write_bytes(b"user-owned")
    try:
        monkeypatch.setattr(MEDIA, "render_frame", lambda *_args: pytest.fail("render must not run"))
        with pytest.raises(ValueError, match="refusing to overwrite"):
            MEDIA.build(input_path, video, png, gif, summary, sidecar)
        assert gif.read_bytes() == b"user-owned"
    finally:
        for path in outputs:
            path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("field", "bad_path", "message"),
    [
        ("video", ROOT / "docs" / "bad.mp4", "local video"),
        ("png", ROOT / "reports" / "bad.png", "public PNG"),
        ("gif", ROOT / "reports" / "bad.gif", "public GIF"),
        ("summary", ROOT / "docs" / "bad.json", "visual summary"),
        ("sidecar", ROOT / "docs" / "bad.json", "visual sidecar"),
    ],
)
def test_build_rejects_outputs_outside_policy_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, bad_path: Path, message: str
) -> None:
    stem = f"test_rev17_policy_{uuid.uuid4().hex}"
    values = {
        "input": MEDIA.RUNS_DIR / f"{stem}_input.json",
        "video": MEDIA.LOCAL_VIDEO_DIR / f"{stem}.mp4",
        "png": MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.png",
        "gif": MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.gif",
        "summary": MEDIA.RUNS_DIR / f"{stem}_summary.json",
        "sidecar": MEDIA.RUNS_DIR / f"{stem}_sidecar.json",
    }
    write_fixture(values["input"])
    values[field] = bad_path
    monkeypatch.setattr(MEDIA, "render_frame", lambda *_args: pytest.fail("render must not run"))
    try:
        with pytest.raises(ValueError, match=message):
            MEDIA.build(
                values["input"],
                values["video"],
                values["png"],
                values["gif"],
                values["summary"],
                values["sidecar"],
            )
    finally:
        values["input"].unlink(missing_ok=True)


def test_read_summary_rejects_binding_escape_and_cpu_topology_drift(tmp_path: Path) -> None:
    escaped = fixture()
    escaped["integrity"]["predecessor_path"] = "reports/runs/../escape.json"
    escaped_path = write_fixture(tmp_path / "escaped.json", escaped)
    with pytest.raises(ValueError, match="canonical predecessor"):
        MEDIA.read_summary(escaped_path)

    cpu_drift = fixture()
    cpu_drift["mechanism_split"]["direct_observations"]["runs"][0][
        "contact_authority"
    ]["availability"] = "unknown"
    drift_path = write_fixture(tmp_path / "cpu-drift.json", cpu_drift)
    with pytest.raises(ValueError, match="CPU contact authority"):
        MEDIA.read_summary(drift_path)

    run_drift = fixture()
    run_drift["mechanism_split"]["direct_observations"]["runs"][1]["evidence"] = (
        run_drift["input_reports"][0]
    )
    run_drift_path = write_fixture(tmp_path / "run-drift.json", run_drift)
    with pytest.raises(ValueError, match="run evidence bindings"):
        MEDIA.read_summary(run_drift_path)


def test_build_rejects_input_outside_reports_runs_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = write_fixture(tmp_path / "outside.json")
    monkeypatch.setattr(MEDIA, "render_frame", lambda *_args: pytest.fail("render must not run"))
    with pytest.raises(ValueError, match="input summary"):
        MEDIA.build(
            input_path,
            MEDIA.DEFAULT_VIDEO,
            MEDIA.DEFAULT_PNG,
            MEDIA.DEFAULT_GIF,
            MEDIA.DEFAULT_SUMMARY,
            MEDIA.DEFAULT_SIDECAR,
        )


def test_build_rolls_back_all_outputs_if_input_changes_during_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stem = f"test_rev17_input_drift_{uuid.uuid4().hex}"
    input_path = MEDIA.RUNS_DIR / f"{stem}_input.json"
    video = MEDIA.LOCAL_VIDEO_DIR / f"{stem}.mp4"
    png = MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.png"
    gif = MEDIA.PUBLIC_MEDIA_DIR / f"{stem}.gif"
    summary = MEDIA.RUNS_DIR / f"{stem}_summary.json"
    sidecar = MEDIA.RUNS_DIR / f"{stem}_sidecar.json"
    outputs = (video, png, gif, summary, sidecar)
    write_fixture(input_path)
    original_ffmpeg = MEDIA.run_ffmpeg

    def mutate_after_ffmpeg(frames_pattern: Path, destination: Path, ffmpeg: str) -> None:
        original_ffmpeg(frames_pattern, destination, ffmpeg)
        input_path.write_bytes(input_path.read_bytes() + b"\n")

    monkeypatch.setattr(MEDIA, "run_ffmpeg", mutate_after_ffmpeg)
    try:
        with pytest.raises(ValueError, match="changed during media build"):
            MEDIA.build(input_path, video, png, gif, summary, sidecar)
        assert all(not path.exists() for path in outputs)
    finally:
        input_path.unlink(missing_ok=True)
        for path in outputs:
            path.unlink(missing_ok=True)


def test_cli_exposes_explicit_input_and_all_outputs() -> None:
    help_text = MEDIA.build_parser().format_help()
    for option in ("--input", "--video", "--png", "--gif", "--summary", "--sidecar"):
        assert option in help_text
