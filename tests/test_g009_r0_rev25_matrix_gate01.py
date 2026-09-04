import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g009_rev25_summary", ROOT / "scripts/summarize_g009_r0_rev25_matrix_gate01.py"
)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


def _load_bootstrap_writer(monkeypatch):
    benchmark = types.ModuleType("bootstrap_benchmark_g009")
    benchmark.main = lambda: None
    matrix = types.ModuleType("isaac_walk_g009.matrix_gate01")
    matrix.MATRIX_CRITIC_OBSERVATION_DIM = 164
    matrix.MATRIX_OBSERVATION_DIM = 57
    matrix.MATRIX_POLICY_OBSERVATION_DIM = 140
    matrix.NOMINAL_BODY_WEIGHT_N = 147.33639000000002
    matrix.ORDERED_BODY_NAMES = ()
    matrix.ORDERED_BODY_NAMES_SHA256 = "0" * 64
    matrix.TERRAIN_FILTER_PATHS = ()
    matrix.reset_runtime_telemetry = lambda: None
    matrix.runtime_telemetry = lambda: {}
    package = types.ModuleType("isaac_walk_g009")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "bootstrap_benchmark_g009", benchmark)
    monkeypatch.setitem(sys.modules, "isaac_walk_g009", package)
    monkeypatch.setitem(sys.modules, "isaac_walk_g009.matrix_gate01", matrix)
    spec = importlib.util.spec_from_file_location(
        "g009_rev25_bootstrap_writer", ROOT / "scripts/bootstrap_matrix_gate01_g009.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preregistered_gate_manifest_matches_verifier_gate_names():
    prereg = json.loads(
        (ROOT / "configs/g009_r0_rev25_matrix_gate01.json").read_text(encoding="utf-8")
    )
    keys = sorted(prereg["pass_gates"])
    digest = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
    assert all(value is True for value in prereg["pass_gates"].values())
    assert digest == prereg["pass_gate_key_manifest_sha256"]
    source = (ROOT / "scripts/summarize_g009_r0_rev25_matrix_gate01.py").read_text(encoding="utf-8")
    assert set(keys) == {
        name for name in keys if f'"{name}"' in source
    }
    source_paths = sorted(SUMMARY.SOURCE_PATHS)
    assert prereg["source_binding_paths"] == source_paths
    assert prereg["source_binding_path_manifest_sha256"] == hashlib.sha256(
        json.dumps(source_paths, separators=(",", ":")).encode()
    ).hexdigest()
    assert "src/isaac_walk_g009/mdp/__init__.py" in source_paths
    assert "src/isaac_walk_g009/mdp/events.py" in source_paths


def test_actor_matrix_optimizer_evidence_requires_columns_83_140_and_step_20():
    moment = torch.zeros((512, 140))
    moment[:, 83:140] = 0.25
    evidence = SUMMARY.actor_matrix_optimizer_evidence(
        {"state": {7: {"step": torch.tensor(20.0), "exp_avg": moment}}}
    )
    assert evidence == [
        {
            "step": 20,
            "matrix_columns_nonzero": True,
            "matrix_columns_l2": evidence[0]["matrix_columns_l2"],
        }
    ]
    assert evidence[0]["matrix_columns_l2"] > 0.0


def test_actor_matrix_optimizer_evidence_rejects_zero_matrix_columns():
    evidence = SUMMARY.actor_matrix_optimizer_evidence(
        {"state": {7: {"step": 20, "exp_avg": torch.zeros((512, 140))}}}
    )
    assert evidence[0]["matrix_columns_nonzero"] is False
    assert evidence[0]["matrix_columns_l2"] == 0.0


def _telemetry_identity_fixture():
    prereg = json.loads(
        (ROOT / "configs/g009_r0_rev25_matrix_gate01.json").read_text(encoding="utf-8")
    )
    report = {"run_name": "g009_r0_rev25_matrix_gate01_retry01_s42"}
    telemetry = {
        "schema_version": "g009.r0.rev25.matrix_gate01_runtime.v1",
        "evidence_id": "G009-5-E018",
        "run_name": report["run_name"],
        "repository_commit": "a" * 40,
        "benchmark_completed": True,
        "terrain_filter_paths": prereg["terrain_filter_paths"],
        "matrix_observation_dimension": 57,
        "policy_observation_dimension": 140,
        "critic_observation_dimension": 164,
        "expected_policy_matrix_slice_from_term_order": [83, 140],
        "raw_authority_frame": "world",
        "policy_projection_frame": "base",
        "nominal_body_weight_n": 147.33639000000002,
        "bounding": "elementwise_tanh",
    }
    path = Path(f"{report['run_name']}.matrix_gate01.json")
    return prereg, report, telemetry, path


def test_matrix_telemetry_identity_binds_all_preregistered_fields():
    prereg, report, telemetry, path = _telemetry_identity_fixture()
    assert SUMMARY.validate_telemetry_identity(telemetry, prereg, report, path, "a" * 40)


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("schema_version", "forged"),
        ("evidence_id", "forged"),
        ("run_name", "forged"),
        ("terrain_filter_paths", ["/forged"]),
        ("raw_authority_frame", "body"),
        ("policy_projection_frame", "world"),
        ("nominal_body_weight_n", 1.0),
        ("bounding", "clamp"),
    ],
)
def test_matrix_telemetry_identity_rejects_field_forgery(field, forged):
    prereg, report, telemetry, path = _telemetry_identity_fixture()
    telemetry[field] = forged
    assert not SUMMARY.validate_telemetry_identity(telemetry, prereg, report, path, "a" * 40)


def _git_source_fixture(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "gate@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Gate Test"], cwd=tmp_path, check=True)
    paths = ("a.txt", "nested/b.txt")
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "nested/b.txt").write_text("beta\n", encoding="utf-8")
    prereg_path = tmp_path / "prereg.json"
    ordered = sorted(paths)
    prereg_path.write_text(
        json.dumps(
            {
                "source_binding_paths": ordered,
                "source_binding_path_manifest_sha256": hashlib.sha256(
                    json.dumps(ordered, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", *paths, "prereg.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    files = {path: hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() for path in paths}
    payload = "\n".join(f"{path}:{files[path]}" for path in ordered).encode()
    report = {
        "repository": {"commit": head, "dirty": False},
        "source_bundle": {
            "files": files,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "matches_repository_commit": True,
        },
    }
    monkeypatch.setattr(SUMMARY, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(SUMMARY, "PREREG_PATH", prereg_path)
    monkeypatch.setattr(SUMMARY, "SOURCE_PATHS", paths)
    return report


def test_validate_source_recomputes_git_blobs_and_rejects_forged_report(tmp_path, monkeypatch):
    report = _git_source_fixture(tmp_path, monkeypatch)
    assert SUMMARY.validate_source(report)
    report["source_bundle"]["files"]["a.txt"] = "0" * 64
    report["source_bundle"]["matches_repository_commit"] = True
    assert not SUMMARY.validate_source(report)


def test_validate_source_rejects_dirty_bound_path_even_with_pass_booleans(tmp_path, monkeypatch):
    report = _git_source_fixture(tmp_path, monkeypatch)
    (tmp_path / "a.txt").write_text("dirty\n", encoding="utf-8")
    report["repository"]["dirty"] = False
    report["source_bundle"]["matches_repository_commit"] = True
    assert not SUMMARY.validate_source(report)


def test_validate_source_does_not_trust_report_pass_booleans(tmp_path, monkeypatch):
    report = _git_source_fixture(tmp_path, monkeypatch)
    report["repository"]["dirty"] = True
    report["source_bundle"]["matches_repository_commit"] = False
    assert SUMMARY.validate_source(report)


def test_checkpoint_is_not_loaded_before_source_provenance_passes(tmp_path, monkeypatch):
    checkpoint = tmp_path / "malicious.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(
        SUMMARY.torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unsafe load attempted")),
    )
    assert SUMMARY.load_checkpoint_if_source_valid(checkpoint, False) == {}


def test_checkpoint_uses_weights_only_true(tmp_path, monkeypatch):
    checkpoint = tmp_path / "safe.pt"
    checkpoint.write_bytes(b"fixture")
    calls = []
    monkeypatch.setattr(
        SUMMARY.torch,
        "load",
        lambda path, **kwargs: calls.append((path, kwargs)) or {"model_state_dict": {}},
    )
    assert SUMMARY.load_checkpoint_if_source_valid(checkpoint, True) == {"model_state_dict": {}}
    assert calls == [(checkpoint, {"map_location": "cpu", "weights_only": True})]


def test_checkpoint_fixture_schema_loads_with_weights_only_true(tmp_path):
    checkpoint = tmp_path / "safe.pt"
    torch.save(
        {
            "model_state_dict": {"actor.0.weight": torch.ones((512, 140))},
            "optimizer_state_dict": {
                "state": {0: {"step": torch.tensor(20.0), "exp_avg": torch.ones((512, 140))}}
            },
        },
        checkpoint,
    )
    loaded = SUMMARY.load_checkpoint_if_source_valid(checkpoint, True)
    assert tuple(loaded["model_state_dict"]["actor.0.weight"].shape) == (512, 140)
    assert SUMMARY.actor_matrix_optimizer_evidence(loaded["optimizer_state_dict"])[0]["step"] == 20


def test_checkpoint_path_hash_and_tensorboard_are_bound_before_load(tmp_path):
    tensorboard = tmp_path / "run"
    tensorboard.mkdir()
    checkpoint = tensorboard / "model_0.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tensorboard / "events.out.tfevents.fixture").write_bytes(b"event")
    report = {
        "artifacts": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            "tensorboard_directory": str(tensorboard),
        }
    }
    assert SUMMARY.checkpoint_artifact_binding(report) == (checkpoint, tensorboard, True)
    report["artifacts"]["checkpoint_sha256"] = "0" * 64
    assert SUMMARY.checkpoint_artifact_binding(report)[2] is False


def test_checkpoint_outside_reported_tensorboard_is_rejected(tmp_path):
    tensorboard = tmp_path / "run"
    tensorboard.mkdir()
    (tensorboard / "events.out.tfevents.fixture").write_bytes(b"event")
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = {
        "artifacts": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            "tensorboard_directory": str(tensorboard),
        }
    }
    assert SUMMARY.checkpoint_artifact_binding(report)[2] is False


def test_canonical_output_is_no_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "canonical.json"
    output.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(SUMMARY, "DEFAULT_OUTPUT", output)
    with pytest.raises(ValueError, match="already exists"):
        SUMMARY.write_json(output, {"forged": True})
    assert output.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_canonical_output_is_published_once_from_complete_temp(tmp_path, monkeypatch):
    output = tmp_path / "canonical.json"
    monkeypatch.setattr(SUMMARY, "DEFAULT_OUTPUT", output)
    SUMMARY.write_json(output, {"passed": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"passed": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_canonical_output_race_is_atomic_no_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "canonical.json"
    monkeypatch.setattr(SUMMARY, "DEFAULT_OUTPUT", output)
    original_link = SUMMARY.os.link

    def create_competing_destination_then_link(source, destination):
        Path(destination).write_bytes(b"competing-writer\n")
        return original_link(source, destination)

    monkeypatch.setattr(SUMMARY.os, "link", create_competing_destination_then_link)
    with pytest.raises(FileExistsError, match="canonical output already exists"):
        SUMMARY.write_json(output, {"forged": True})
    assert output.read_bytes() == b"competing-writer\n"
    assert hashlib.sha256(output.read_bytes()).hexdigest() == hashlib.sha256(
        b"competing-writer\n"
    ).hexdigest()
    assert not list(tmp_path.glob("*.tmp"))


def test_matrix_telemetry_output_is_no_overwrite(tmp_path, monkeypatch):
    bootstrap = _load_bootstrap_writer(monkeypatch)
    output = tmp_path / "telemetry.json"
    output.write_bytes(b"existing\n")
    with pytest.raises(FileExistsError, match="matrix telemetry output already exists"):
        bootstrap.write_telemetry(output, {"forged": True})
    assert output.read_bytes() == b"existing\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_matrix_telemetry_is_published_once_from_complete_temp(tmp_path, monkeypatch):
    bootstrap = _load_bootstrap_writer(monkeypatch)
    output = tmp_path / "telemetry.json"
    bootstrap.write_telemetry(output, {"passed": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"passed": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_matrix_telemetry_race_is_atomic_no_overwrite(tmp_path, monkeypatch):
    bootstrap = _load_bootstrap_writer(monkeypatch)
    output = tmp_path / "telemetry.json"
    original_link = bootstrap.os.link

    def create_competing_destination_then_link(source, destination):
        Path(destination).write_bytes(b"competing-writer\n")
        return original_link(source, destination)

    monkeypatch.setattr(bootstrap.os, "link", create_competing_destination_then_link)
    with pytest.raises(FileExistsError, match="matrix telemetry output already exists"):
        bootstrap.write_telemetry(output, {"forged": True})
    assert output.read_bytes() == b"competing-writer\n"
    assert hashlib.sha256(output.read_bytes()).hexdigest() == hashlib.sha256(
        b"competing-writer\n"
    ).hexdigest()
    assert not list(tmp_path.glob("*.tmp"))
