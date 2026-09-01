from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "bootstrap_benchmark_g009.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bootstrap_benchmark_g009", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_registers_g009_before_running_official_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    upstream = tmp_path / "scripts" / "benchmarks" / "benchmark_rsl_rl.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# official fixture\n", encoding="utf-8")
    calls: list[object] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "EXPECTED_UPSTREAM_SHA256", module.file_sha256(upstream))
    monkeypatch.setattr(module, "validate_isaaclab_checkout", lambda root: calls.append(("checkout", root)))
    monkeypatch.setattr(module, "register_tasks", lambda: calls.append("register"))
    monkeypatch.setattr(
        module.runpy,
        "run_path",
        lambda path, run_name: calls.append((Path(path), run_name)),
    )

    module.main()

    assert calls == ["register", ("checkout", tmp_path), (upstream, "__main__")]


def test_bootstrap_fails_closed_when_official_benchmark_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "register_tasks", lambda: None)

    with pytest.raises(FileNotFoundError, match="official benchmark_rsl_rl.py"):
        module.main()


def test_bootstrap_rejects_unpinned_official_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    upstream = tmp_path / "scripts" / "benchmarks" / "benchmark_rsl_rl.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# changed upstream\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "register_tasks", lambda: None)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        module.main()
