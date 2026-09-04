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


def _configure_embedded_runner(module, monkeypatch, tmp_path, namespace):
    upstream = tmp_path / "scripts" / "benchmarks" / "benchmark_rsl_rl.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# official fixture\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "EXPECTED_UPSTREAM_SHA256", module.file_sha256(upstream))
    monkeypatch.setattr(module, "validate_isaaclab_checkout", lambda _root: None)
    monkeypatch.setattr(module, "register_tasks", lambda: None)
    run_names = []

    def load(path, run_name):
        assert Path(path) == upstream
        run_names.append(run_name)
        return namespace

    monkeypatch.setattr(module.runpy, "run_path", load)
    return run_names


def test_embedded_runner_calls_callback_before_close(monkeypatch, tmp_path):
    module = _load_module()
    events = []
    simulation_app = type("SimulationApp", (), {"close": lambda self: events.append("close")})()
    namespace = {
        "main": lambda: events.append("main"),
        "simulation_app": simulation_app,
    }
    run_names = _configure_embedded_runner(module, monkeypatch, tmp_path, namespace)

    module.run_with_before_close(lambda completed: events.append(("callback", completed)))

    assert run_names == ["g009_pinned_benchmark_runtime"]
    assert events == ["main", ("callback", True), "close"]


def test_embedded_runner_preserves_official_main_exception_after_callback_and_close(
    monkeypatch, tmp_path
):
    module = _load_module()
    events = []
    original = KeyboardInterrupt("official main failed")

    def fail_main():
        events.append("main")
        raise original

    simulation_app = type("SimulationApp", (), {"close": lambda self: events.append("close")})()
    namespace = {"main": fail_main, "simulation_app": simulation_app}
    _configure_embedded_runner(module, monkeypatch, tmp_path, namespace)

    with pytest.raises(KeyboardInterrupt) as raised:
        module.run_with_before_close(lambda completed: events.append(("callback", completed)))

    assert raised.value is original
    assert events == ["main", ("callback", False), "close"]


def test_embedded_runner_closes_once_when_callback_raises(monkeypatch, tmp_path):
    module = _load_module()
    events = []
    original = RuntimeError("callback failed")
    simulation_app = type("SimulationApp", (), {"close": lambda self: events.append("close")})()
    namespace = {
        "main": lambda: events.append("main"),
        "simulation_app": simulation_app,
    }
    _configure_embedded_runner(module, monkeypatch, tmp_path, namespace)

    def fail_callback(completed):
        events.append(("callback", completed))
        raise original

    with pytest.raises(RuntimeError) as raised:
        module.run_with_before_close(fail_callback)

    assert raised.value is original
    assert events == ["main", ("callback", True), "close"]


def test_embedded_runner_rethrows_close_only_base_exception_after_callback(
    monkeypatch, tmp_path
):
    module = _load_module()
    events = []
    original = SystemExit("close failed")

    def fail_close():
        events.append("close")
        raise original

    simulation_app = type("SimulationApp", (), {"close": lambda self: fail_close()})()
    namespace = {
        "main": lambda: events.append("main"),
        "simulation_app": simulation_app,
    }
    _configure_embedded_runner(module, monkeypatch, tmp_path, namespace)

    with pytest.raises(SystemExit) as raised:
        module.run_with_before_close(lambda completed: events.append(("callback", completed)))

    assert raised.value is original
    assert events == ["main", ("callback", True), "close"]


def test_embedded_runner_prioritizes_main_when_main_callback_and_close_all_fail(
    monkeypatch, tmp_path
):
    module = _load_module()
    events = []
    main_error = KeyboardInterrupt("main failed")
    callback_error = SystemExit("callback failed")
    close_error = RuntimeError("close failed")

    def fail_main():
        events.append("main")
        raise main_error

    def fail_callback(completed):
        events.append(("callback", completed))
        raise callback_error

    def fail_close():
        events.append("close")
        raise close_error

    simulation_app = type("SimulationApp", (), {"close": lambda self: fail_close()})()
    namespace = {"main": fail_main, "simulation_app": simulation_app}
    _configure_embedded_runner(module, monkeypatch, tmp_path, namespace)

    with pytest.raises(KeyboardInterrupt) as raised:
        module.run_with_before_close(fail_callback)

    assert raised.value is main_error
    assert raised.value is not callback_error
    assert raised.value is not close_error
    assert events == ["main", ("callback", False), "close"]


def test_embedded_runner_prioritizes_callback_over_close_after_main_succeeds(
    monkeypatch, tmp_path
):
    module = _load_module()
    events = []
    callback_error = KeyboardInterrupt("callback failed")
    close_error = SystemExit("close failed")

    def fail_callback(completed):
        events.append(("callback", completed))
        raise callback_error

    def fail_close():
        events.append("close")
        raise close_error

    simulation_app = type("SimulationApp", (), {"close": lambda self: fail_close()})()
    namespace = {
        "main": lambda: events.append("main"),
        "simulation_app": simulation_app,
    }
    _configure_embedded_runner(module, monkeypatch, tmp_path, namespace)

    with pytest.raises(KeyboardInterrupt) as raised:
        module.run_with_before_close(fail_callback)

    assert raised.value is callback_error
    assert raised.value is not close_error
    assert events == ["main", ("callback", True), "close"]
