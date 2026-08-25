from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "record_g006_policy_tested", ROOT / "scripts" / "record_g006_policy.py"
)
assert SPEC is not None and SPEC.loader is not None
RECORDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECORDER)


def _configure(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> list[str]:
    argv = ["record_g006_policy.py", "--headless", "--video", *arguments]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", argv)
    RECORDER._configure_windows_headless_video()
    return sys.argv


def test_windows_capture_adds_required_kit_args(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _configure(monkeypatch)

    assert argv[-1] == "--kit_args=--/app/vulkan=false --/app/window/hideUi=true"


@pytest.mark.parametrize("separate_value", [False, True])
def test_windows_capture_merges_unrelated_kit_args(
    monkeypatch: pytest.MonkeyPatch, separate_value: bool
) -> None:
    existing = "--/renderer/multiGpu/enabled=false"
    arguments = ("--kit_args", existing) if separate_value else (f"--kit_args={existing}",)

    argv = _configure(monkeypatch, *arguments)
    merged = argv[argv.index("--kit_args") + 1] if separate_value else argv[-1].split("=", 1)[1]

    assert merged.split() == [existing, *RECORDER.REQUIRED_WINDOWS_KIT_ARGS]


@pytest.mark.parametrize(
    "conflict",
    ["--/app/vulkan=true", "--/app/window/hideUi=false"],
)
def test_windows_capture_rejects_conflicting_required_kit_args(
    monkeypatch: pytest.MonkeyPatch, conflict: str
) -> None:
    with pytest.raises(ValueError, match="conflicting --kit_args"):
        _configure(monkeypatch, f"--kit_args={conflict}")


def test_windows_capture_preserves_matching_required_kit_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = " ".join(RECORDER.REQUIRED_WINDOWS_KIT_ARGS)

    argv = _configure(monkeypatch, f"--kit_args={required}")

    assert argv[-1] == f"--kit_args={required}"


def test_non_capture_invocation_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = ["record_g006_policy.py", "--headless"]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", argv)

    RECORDER._configure_windows_headless_video()

    assert sys.argv == argv


def test_sha256_matches_known_content(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"isaac-walk-rl\n")

    assert RECORDER._sha256(sample) == "b21f0bb134758c09cd0cd24d6e3d70b920f0e08b41fedc09a4fd3c83e3c5cc07"
