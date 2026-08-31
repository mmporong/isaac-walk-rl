from __future__ import annotations

import copy
import importlib.util
import json
import os
import threading
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_g009_r0_rev23_matrix_observation_adapter_media.py"
SPEC = importlib.util.spec_from_file_location("rev23_media", BUILDER); assert SPEC and SPEC.loader
MEDIA = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MEDIA)


def load_validator():
    path = ROOT / "scripts/validate_g009_r0_rev23_matrix_observation_adapter_media.py"
    spec = importlib.util.spec_from_file_location("rev23_media_validator", path); assert spec and spec.loader
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def test_paths_numbering_and_cli_contract() -> None:
    assert MEDIA.DEFAULT_VIDEO == Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic/g009_5_r0_diag_rev23_14_01_cpu_matrix_adapter_telemetry_s42.mp4"
    assert MEDIA.phase_paths("final")["video"].name == "g009_5_r0_diag_rev23_14_02_final_matrix_adapter_telemetry_s42.mp4"
    assert MEDIA.DEFAULT_GIF.name == "g009_5_r0_diag_rev23_14_01_cpu_matrix_adapter_telemetry.gif"
    assert MEDIA.phase_paths("final")["png"].name == "g009_5_r0_diag_rev23_14_02_final_matrix_adapter_telemetry.png"
    help_text = MEDIA.build_parser().format_help()
    for option in ("--phase", "--inputs", "--video", "--gif", "--png", "--summary", "--sidecar", "--ffmpeg"): assert option in help_text


@pytest.mark.parametrize("phase,count,slots", [
    ("cpu", 2, ["cpu.rep1", "cpu.rep2"]),
    ("final", 4, ["cpu.rep1", "cpu.rep2", "cuda:0.rep1", "cuda:0.rep2"]),
])
def test_canonical_inputs_revalidate_and_expose_exact_ledger(phase: str, count: int, slots: list[str]) -> None:
    value = MEDIA.validate_inputs(phase, MEDIA.expected_inputs(phase))
    assert len(value["reports"]) == count
    assert [report["slot"] for report in value["reports"]] == slots
    for report in value["reports"]:
        assert report["passed"] is True and report["sample_count"] == 150
        assert len(report["step_ledger"]) == 150
        assert [row["step"] for row in report["step_ledger"]] == list(range(1, 151))
        assert all(set(row) == {"step", "max_magnitude_n", "active_mask_count", "zero_vector_count"} for row in report["step_ledger"])
    assert value["repeatability"]["cpu"]["repeatable"] is True
    if phase == "final": assert value["repeatability"]["cuda:0"]["repeatable"] is True


@pytest.mark.parametrize("key,bad", [("learned", True), ("reward_computed", True), ("ppo_updates", 1), ("qualification_status", "passed")])
def test_governance_fails_closed(key: str, bad: object) -> None:
    report = json.loads(MEDIA.CPU_REPORTS[0].read_text(encoding="utf-8")); report["governance"][key] = bad
    with pytest.raises(ValueError, match=f"governance {key} mismatch"): MEDIA._telemetry(report)


def test_portable_gpu_validation_passes_without_cuda_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    original = torch.tensor; devices: list[object] = []
    def cpu_only(*args, **kwargs):
        devices.append(kwargs.get("device")); assert kwargs.get("device") != "cuda:0"; return original(*args, **kwargs)
    monkeypatch.setattr(torch, "tensor", cpu_only)
    for path in MEDIA.FINAL_REPORTS[2:]: MEDIA.validate_report_portable(json.loads(path.read_text(encoding="utf-8")))
    assert devices and set(devices) == {"cpu"}


@pytest.mark.parametrize("mutation", [
    lambda report: report["adapter_runtime"]["step_ledger"][0].update(source_unchanged=False),
    lambda report: report["adapter_runtime"]["step_ledger"][0]["source_before"].update(exact_values_sha256="0" * 64),
    lambda report: report["adapter_runtime"]["step_ledger"][0]["output_metadata"]["world_xyz"].update(storage_data_ptr=report["adapter_runtime"]["step_ledger"][0]["source_before"]["storage_data_ptr"]),
    lambda report: report["adapter_runtime"].update(zero_source_vector_count_total=0),
    lambda report: report["adapter_runtime"]["checks"].pop("finite_150_of_150"),
    lambda report: report["adapter_runtime"]["checks"].update(finite_150_of_150=False),
    lambda report: report["adapter_runtime"]["representative_snapshots"].pop(),
    lambda report: report["adapter_runtime"]["representative_snapshots"][0]["source"][0][0][0].__setitem__(0, 999.0),
    lambda report: report["adapter_runtime"]["representative_snapshots"][0]["world_xyz"][0][0].__setitem__(0, 999.0),
    lambda report: report.update(adapter_decision={}),
])
def test_portable_gpu_report_mutations_fail_closed(mutation) -> None:
    report = json.loads(MEDIA.FINAL_REPORTS[2].read_text(encoding="utf-8")); mutation(report)
    with pytest.raises(ValueError): MEDIA.validate_report_portable(report)


@pytest.mark.parametrize("field", ["source_mutation_steps", "oracle_mismatch_steps", "alias_violation_steps", "source_contract_violation_steps", "zero_semantics_violation_steps", "device_violation_steps", "nonfinite_steps"])
def test_portable_gpu_rejects_each_nonempty_violation_ledger(field: str) -> None:
    report = json.loads(MEDIA.FINAL_REPORTS[2].read_text(encoding="utf-8")); report["adapter_runtime"][field] = [1]
    with pytest.raises(ValueError, match="violation ledger"): MEDIA.validate_report_portable(report)


def test_portable_final_rejects_false_row_and_self_consistent_binding_tamper() -> None:
    final = json.loads(MEDIA.FINAL_SYNTHESIS.read_text(encoding="utf-8"))
    false_row = copy.deepcopy(final); false_row["rows"][2]["adapter_runtime_passed"] = False
    with pytest.raises(ValueError): MEDIA.validate_final_portable(false_row)
    rebound = copy.deepcopy(final); rebound["input_reports"][2]["sha256"] = "0" * 64; rebound["rows"][2]["binding"]["sha256"] = "0" * 64
    with pytest.raises(ValueError): MEDIA.validate_final_portable(rebound)


def test_every_frame_contains_required_disclaimer_and_actual_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes
    data = MEDIA.validate_inputs("cpu", MEDIA.expected_inputs("cpu")); rendered: list[str] = []; original = Axes.text
    def capture(self, x, y, text, *args, **kwargs): rendered.append(text); return original(self, x, y, text, *args, **kwargs)
    monkeypatch.setattr(Axes, "text", capture)
    output = tmp_path / "frame.png"; MEDIA.render_frame(data, 0.5, output)
    combined = " ".join(rendered)
    assert MEDIA.HEADER in combined and MEDIA.FOOTER in combined and "14.01" in combined and "PROGRESS: 075/150" in combined
    assert "cpu.rep1 [cpu] rep1" in combined and output.read_bytes().startswith(b"\x89PNG")


def test_final_is_side_by_side_observation_not_device_equivalence_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes
    data = MEDIA.validate_inputs("final", MEDIA.expected_inputs("final")); rendered: list[str] = []; original = Axes.text
    def capture(self, x, y, text, *args, **kwargs): rendered.append(text); return original(self, x, y, text, *args, **kwargs)
    monkeypatch.setattr(Axes, "text", capture)
    MEDIA.render_frame(data, 1.0, tmp_path / "final.png"); combined = " ".join(rendered).upper()
    assert "CPU REPEATABILITY=PASS" in combined and "CUDA:0 REPEATABILITY=PASS" in combined
    assert "EQUIVALENT" not in combined and "DEVICE MATCH" not in combined


def test_build_is_hash_bound_and_no_overwrite() -> None:
    token = uuid.uuid4().hex
    outputs = {"video": MEDIA.LOCAL_VIDEO_DIR / f"test_rev23_{token}.mp4", "gif": MEDIA.PUBLIC_MEDIA_DIR / f"test_rev23_{token}.gif", "png": MEDIA.PUBLIC_MEDIA_DIR / f"test_rev23_{token}.png", "summary": MEDIA.RUNS_DIR / f"test_rev23_{token}_summary.json", "sidecar": MEDIA.RUNS_DIR / f"test_rev23_{token}_sidecar.json"}
    try:
        value = MEDIA.build("cpu", MEDIA.expected_inputs("cpu"), outputs)
        assert all(path.exists() for path in outputs.values())
        assert value["governance"] == MEDIA.GOVERNANCE and value["claim_limits"] == MEDIA.CLAIM_LIMITS
        with pytest.raises(ValueError, match="overwrite"): MEDIA.build("cpu", MEDIA.expected_inputs("cpu"), outputs)
    finally:
        for path in outputs.values(): path.unlink(missing_ok=True)


def _transaction_files(tmp_path: Path) -> tuple[tuple[tuple[Path, Path], ...], list[bytes]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    payloads = [b"video-bytes", b"summary-bytes", b"sidecar-marker"]
    sources = []
    for index, payload in enumerate(payloads):
        source = tmp_path / f"source-{index}.bin"; source.write_bytes(payload); sources.append(source)
    destinations = [tmp_path / "out" / f"destination-{index}.bin" for index in range(3)]
    return tuple(zip(sources, destinations, strict=True)), payloads


def _assert_no_publish_temps(directory: Path) -> None:
    if directory.exists(): assert not [path for path in directory.iterdir() if path.name.startswith(".")]


def test_atomic_transaction_publishes_sidecar_last_and_preserves_bytes(tmp_path: Path) -> None:
    pairs, payloads = _transaction_files(tmp_path); events: list[str] = []
    def validate():
        events.append("validate"); assert all(destination.exists() for _, destination in pairs[:-1]); assert not pairs[-1][1].exists()
    MEDIA.publish_transaction(pairs, validate)
    assert events == ["validate"] and [destination.read_bytes() for _, destination in pairs] == payloads
    _assert_no_publish_temps(pairs[-1][1].parent)


def test_atomic_transaction_concurrent_one_wins_one_fails(tmp_path: Path) -> None:
    pairs, payloads = _transaction_files(tmp_path); barrier = threading.Barrier(2); outcomes: list[str] = []
    def run():
        barrier.wait()
        try: MEDIA.publish_transaction(pairs, lambda: None); outcomes.append("pass")
        except (FileExistsError, ValueError): outcomes.append("fail")
    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert sorted(outcomes) == ["fail", "pass"] and [destination.read_bytes() for _, destination in pairs] == payloads
    _assert_no_publish_temps(pairs[-1][1].parent)


@pytest.mark.parametrize("fault_call", [1, 2, 3, 4])
def test_atomic_transaction_link_fault_rolls_back_without_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_call: int) -> None:
    pairs, _ = _transaction_files(tmp_path); original = os.link; calls = 0
    def faulty(source, destination, *args, **kwargs):
        nonlocal calls; calls += 1
        if calls == fault_call: raise OSError("injected link fault")
        return original(source, destination, *args, **kwargs)
    monkeypatch.setattr(MEDIA.os, "link", faulty)
    with pytest.raises(OSError, match="injected"): MEDIA.publish_transaction(pairs, lambda: None)
    assert not any(destination.exists() for _, destination in pairs)
    _assert_no_publish_temps(pairs[-1][1].parent)


@pytest.mark.parametrize("cleanup_call", [1, 2, 3, 4])
def test_linked_sibling_cleanup_fault_recovers_without_orphan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_call: int) -> None:
    pairs, payloads = _transaction_files(tmp_path); original = Path.unlink; calls = 0; injected = False
    def faulty(self: Path, *args, **kwargs):
        nonlocal calls, injected
        if self.name.endswith(".tmp"):
            calls += 1
            if calls == cleanup_call and not injected:
                injected = True; raise OSError("injected sibling cleanup fault")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", faulty)
    MEDIA.publish_transaction(pairs, lambda: None)
    assert injected and [destination.read_bytes() for _, destination in pairs] == payloads
    _assert_no_publish_temps(pairs[-1][1].parent)


def test_callback_foreign_replacement_blocks_sidecar_and_preserves_foreign_bytes(tmp_path: Path) -> None:
    pairs, _ = _transaction_files(tmp_path); first = pairs[0][1]; foreign = b"\x89PNG\r\n\x1a\nforeign-valid-magic"
    def adversarial_callback():
        first.unlink(); first.write_bytes(foreign)
    with pytest.raises(ValueError, match="ownership changed"): MEDIA.publish_transaction(pairs, adversarial_callback)
    assert first.read_bytes() == foreign
    assert not pairs[1][1].exists() and not pairs[-1][1].exists()
    _assert_no_publish_temps(pairs[-1][1].parent)


def test_atomic_transaction_validation_or_input_drift_has_no_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairs, _ = _transaction_files(tmp_path)
    with pytest.raises(ValueError, match="input drift"): MEDIA.publish_transaction(pairs, lambda: (_ for _ in ()).throw(ValueError("input drift")))
    assert not any(destination.exists() for _, destination in pairs)
    _assert_no_publish_temps(pairs[-1][1].parent)
    pairs, _ = _transaction_files(tmp_path / "copy"); original = MEDIA.file_sha256; source_calls = 0
    def drift(path: Path):
        nonlocal source_calls
        value = original(path)
        if path.name == "source-0.bin":
            source_calls += 1
            if source_calls == 2: return "0" * 64
        return value
    monkeypatch.setattr(MEDIA, "file_sha256", drift)
    with pytest.raises(ValueError, match="staged copy|source changed"): MEDIA.publish_transaction(pairs, lambda: None)
    assert not any(destination.exists() for _, destination in pairs)
    _assert_no_publish_temps(pairs[-1][1].parent)


def test_atomic_rollback_does_not_delete_replaced_foreign_bytes(tmp_path: Path) -> None:
    pairs, _ = _transaction_files(tmp_path); first = pairs[0][1]
    def replace_then_fail():
        first.unlink(); first.write_bytes(b"foreign-replacement"); raise ValueError("after replacement")
    with pytest.raises(ValueError, match="replacement"): MEDIA.publish_transaction(pairs, replace_then_fail)
    assert first.read_bytes() == b"foreign-replacement" and not pairs[-1][1].exists()
    _assert_no_publish_temps(pairs[-1][1].parent)


def test_atomic_preexisting_destination_bytes_are_preserved(tmp_path: Path) -> None:
    pairs, _ = _transaction_files(tmp_path); destination = pairs[0][1]; destination.parent.mkdir(); destination.write_bytes(b"existing")
    with pytest.raises(ValueError, match="overwrite"): MEDIA.publish_transaction(pairs, lambda: None)
    assert destination.read_bytes() == b"existing" and not pairs[-1][1].exists()
    _assert_no_publish_temps(pairs[-1][1].parent)


@pytest.mark.parametrize("phase", ["cpu", "final"])
def test_canonical_bundle_validates_when_present(phase: str) -> None:
    paths = MEDIA.phase_paths(phase)
    if paths["video"].exists(): assert load_validator().validate_bundle(phase)["status"] == "pass"
    else: assert not any(path.exists() for path in paths.values())


def test_validator_rejects_claim_and_governance_tamper(monkeypatch: pytest.MonkeyPatch) -> None:
    if not MEDIA.DEFAULT_VIDEO.exists(): return
    validator = load_validator(); original = validator.media.read_json
    for mutate, message in (
        (lambda value: value["claim_limits"].update(training_success_claimed=True), "claim limits"),
        (lambda value: value["governance"].update(learned=True), "governance"),
    ):
        tampered = copy.deepcopy(json.loads(MEDIA.DEFAULT_SIDECAR.read_text(encoding="utf-8"))); mutate(tampered)
        def fake_read(path: Path, _tampered=tampered):
            if path.resolve() == MEDIA.DEFAULT_SIDECAR.resolve(): return _tampered, b"tampered"
            return original(path)
        monkeypatch.setattr(validator.media, "read_json", fake_read)
        with pytest.raises(ValueError, match=message): validator.validate_bundle("cpu")
        monkeypatch.setattr(validator.media, "read_json", original)
