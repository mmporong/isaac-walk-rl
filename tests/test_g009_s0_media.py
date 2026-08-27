from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_g009_s0_media.py"
SPEC = importlib.util.spec_from_file_location("build_g009_s0_media", SCRIPT)
assert SPEC and SPEC.loader
media = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(media)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profile_root = tmp_path / "profile"
    monkeypatch.setenv("USERPROFILE", str(profile_root))
    config_path = tmp_path / "repo" / "configs" / "g009_s0.json"
    checkpoint_payload = b"checkpoint"
    checkpoint_path = profile_root / "models" / "parent.pt"
    _write(checkpoint_path, checkpoint_payload)
    sequence = [
        {"name": "stand", "steps": 2, "command": [0.0, 0.0, 0.0]},
        {"name": "contour_left", "steps": 3, "command": [0.4, 0.0, 0.0]},
    ]
    profiles = [
        {"profile_id": "slope_05", "slope_deg": 5.0, "terrain_azimuth_deg": 0.0},
        {"profile_id": "slope_15", "slope_deg": 15.0, "terrain_azimuth_deg": 0.0},
        {"profile_id": "slope_25_stress", "slope_deg": 25.0, "terrain_azimuth_deg": 0.0},
    ]
    config = {
        "visual_protocol": {
            "profiles": profiles,
            "sequence": sequence,
            "seed": 20260828,
            "camera_contract": "same camera",
        },
        "terrain": {
            "ground_material": {
                "static_friction": 0.8,
                "dynamic_friction": 0.6,
                "friction_combine_mode": "multiply",
            }
        },
        "parent_checkpoint": {
            "path": "%USERPROFILE%\\models\\parent.pt",
            "sha256": _sha(checkpoint_payload),
        },
    }
    _write(config_path, (json.dumps(config) + "\n").encode())
    captures = []
    for index, profile in enumerate(profiles):
        video_payload = b"video" + bytes([index])
        video_path = profile_root / "IsaacLab" / "logs" / "visual_evidence" / "g009" / "S0" / f"{profile['profile_id']}.mp4"
        _write(video_path, video_payload)
        capture = {
            "schema_version": 1,
            "goal_id": "g009",
            "stage_id": "S0",
            "status": "complete",
            "source_commit": "b" * 40,
            "dirty_paths": [],
            "profile": {
                **profile,
                "seed": 20260828,
                "headless": True,
                "step_dt_s": 0.02,
                "total_steps": 5,
                "sequence": sequence,
                "camera": {"eye": [2, 2, 1], "lookat": [0, 0, 0]},
            },
            "config": {"path": "configs/g009_s0.json", "sha256": _sha(config_path.read_bytes())},
            "checkpoint": config["parent_checkpoint"],
            "physics_readback": {
                "slope_deg": profile["slope_deg"],
                "terrain_azimuth_deg": 0.0,
                "ground_material": config["terrain"]["ground_material"],
            },
            "metrics": {"frames": 5},
            "local_video": {
                "path": f"%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\S0\\{profile['profile_id']}.mp4",
                "sha256": _sha(video_payload),
                "bytes": len(video_payload),
                "git_policy": "local_only",
            },
            "record_source_sha256": "c" * 64,
        }
        capture_path = tmp_path / "repo" / "reports" / "runs" / f"{profile['profile_id']}.json"
        _write(capture_path, (json.dumps(capture) + "\n").encode())
        captures.append(capture_path)
    monkeypatch.setattr(media, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(
        media,
        "_git_commit_file_sha256",
        lambda _commit, path: {
            media.RECORD_SOURCE_PATH: "c" * 64,
            media.CONFIG_PATH: _sha(config_path.read_bytes()),
            media.BUILDER_SOURCE_PATH: media.file_sha256(media.SCRIPT if hasattr(media, "SCRIPT") else SCRIPT),
        }[path],
    )
    return config_path, captures


def test_capture_contract_accepts_exact_config_order_and_bindings(tmp_path: Path, monkeypatch) -> None:
    config_path, captures = _fixture(tmp_path, monkeypatch)
    config, reports, sources = media.validate_capture_reports(captures, config_path)
    assert config["visual_protocol"]["seed"] == 20260828
    assert [item["profile"]["profile_id"] for item in reports] == list(media.EXPECTED_PROFILE_IDS)
    assert len(sources) == 3 and all(path.is_file() for path in sources)


def test_capture_contract_accepts_float32_material_readback_only_within_tolerance(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, capture_paths = _fixture(tmp_path, monkeypatch)
    for path in capture_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["physics_readback"]["ground_material"].update(
            static_friction=0.800000011920929,
            dynamic_friction=0.6000000238418579,
        )
        path.write_text(json.dumps(report), encoding="utf-8")

    media.validate_capture_reports(capture_paths, config_path)

    report = json.loads(capture_paths[0].read_text(encoding="utf-8"))
    report["physics_readback"]["ground_material"]["static_friction"] = 0.80001
    capture_paths[0].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="ground_material.static_friction mismatch"):
        media.validate_capture_reports(capture_paths, config_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda reports: reports.reverse(), "profile order mismatch"),
        (lambda reports: reports[1]["profile"].update(camera={"eye": [9, 9, 9]}), "camera values do not match"),
        (lambda reports: reports[0].update(dirty_paths=["src/dirty.py"]), "dirty_paths must be empty"),
        (lambda reports: reports[2]["physics_readback"].update(slope_deg=24.0), "physics slope mismatch"),
        (lambda reports: reports[0]["local_video"].update(sha256="0" * 64), "local_video.sha256 mismatch"),
    ],
)
def test_capture_contract_fails_closed(tmp_path: Path, monkeypatch, mutation, message: str) -> None:
    config_path, capture_paths = _fixture(tmp_path, monkeypatch)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in capture_paths]
    mutation(reports)
    for path, report in zip(capture_paths, reports):
        path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        media.validate_capture_reports(capture_paths, config_path)


def test_media_magic_and_size_are_enforced(tmp_path: Path) -> None:
    gif = tmp_path / "ok.gif"
    png = tmp_path / "ok.png"
    gif.write_bytes(b"GIF89a")
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    media._validate_media(gif, "gif")
    media._validate_media(png, "png")
    gif.write_bytes(b"notgif")
    with pytest.raises(ValueError, match="GIF signature"):
        media._validate_media(gif, "gif")


def test_capture_contract_rejects_commit_tree_binding_mismatch(tmp_path: Path, monkeypatch) -> None:
    config_path, captures = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(media, "_git_commit_file_sha256", lambda _commit, _path: "0" * 64)
    with pytest.raises(ValueError, match="source_commit tree"):
        media.validate_capture_reports(captures, config_path)


def test_atomic_json_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    media._write_json_atomic(output, {"status": "complete"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "complete"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_publish_transaction_restores_all_existing_outputs_on_validation_failure(tmp_path: Path) -> None:
    pairs = []
    for index in range(5):
        staged = tmp_path / "staged" / f"{index}.bin"
        final = tmp_path / "final" / f"{index}.bin"
        _write(staged, f"new-{index}".encode())
        _write(final, f"old-{index}".encode())
        pairs.append((staged, final))

    with pytest.raises(ValueError, match="post-publish failure"):
        media._publish_transaction(pairs, lambda: (_ for _ in ()).throw(ValueError("post-publish failure")))

    assert [final.read_bytes() for _, final in pairs] == [f"old-{i}".encode() for i in range(5)]
    assert list((tmp_path / "final").glob("*.bak")) == []


def test_cli_defaults_are_canonical_and_requires_three_captures() -> None:
    args = media.parse_args(["--capture-reports", "a.json", "b.json", "c.json"])
    assert args.config == ROOT / media.CONFIG_PATH
    with pytest.raises(SystemExit):
        media.parse_args(["--capture-reports", "a.json", "b.json"])


def test_builder_declares_non_walk_scope_and_sidecar_validation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "QUALITATIVE PLAYBACK ONLY" in source
    assert "WALK success or policy qualification" in source
    assert "validate_sidecar(sidecar, REPO_ROOT, check_files=True)" in source
    assert "--rebuild-existing" in source
