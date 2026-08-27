from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_walk_g009.media_contract import (  # noqa: E402
    C0_EXECUTION_LOG_PATH,
    C0_REQUIRED_EVIDENCE,
    C0_VALIDATOR_JSON_PATH,
    MAX_PUBLIC_MEDIA_BYTES,
    STAGE_REGISTRY,
    canonical_json_sha256,
    count_g008_local_video_evidence,
    local_video_directory,
    public_media_directory,
    validate_c0_evidence,
    validate_contract,
    validate_repository_media_rules,
    validate_sidecar,
)


EXPECTED_STAGE_IDS = (
    "S0", "S1-low", "S1-high", "D0A", "D0B", "D0C", "D1", "S2",
    "S3-controlled", "S3-spatial", "R0", "F0A", "R0B", "R1", "R2",
    "R3-controlled", "R3-spatial", "D2", "F0B-TV", "R4", "I0",
    "F0B-FINAL", "I1", "D3",
)


def _artifact(kind: str, path: str, content: bytes, policy: str) -> dict[str, object]:
    return {
        "evidence_type": kind,
        "path": path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "git_policy": policy,
    }


def _write(root: Path, relative: str, content: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_registry_is_the_prd_single_source_of_truth() -> None:
    assert tuple(STAGE_REGISTRY) == EXPECTED_STAGE_IDS
    assert validate_contract() == []
    for stage_id, contract in STAGE_REGISTRY.items():
        assert local_video_directory(stage_id).endswith(f"\\{stage_id}")
        assert public_media_directory(stage_id) == f"docs/media/g009/{stage_id}"
        assert {"local_mp4", "public_gif", "public_png", "sidecar_json", "quantitative_report"} <= set(
            contract.required_evidence
        )


def test_c0_allows_only_governance_evidence() -> None:
    assert validate_c0_evidence(C0_REQUIRED_EVIDENCE) == []
    errors = validate_c0_evidence((*C0_REQUIRED_EVIDENCE, "local_mp4", "public_gif"))
    assert "C0: forbidden media evidence local_mp4" in errors
    assert "C0: forbidden media evidence public_gif" in errors
    assert C0_VALIDATOR_JSON_PATH == "reports/runs/g009_c0_media_contract.json"
    assert C0_EXECUTION_LOG_PATH == "reports/validation/g009_c0_media_contract.log"


def test_repository_rule_is_generic_and_g008_paths_remain_valid() -> None:
    assert validate_repository_media_rules(ROOT) == []
    assert count_g008_local_video_evidence(ROOT) > 0


def test_sidecar_binds_portable_paths_sizes_and_sha256(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    stage_id = "S0"
    source_commit = "b" * 40
    seed = 20260828
    physics_readback = {"static_friction": 0.8, "dynamic_friction": 0.6, "slope_deg": 15.0}
    physics_sha256 = canonical_json_sha256(physics_readback)
    checkpoint_path = "%USERPROFILE%\\checkpoints\\model.pt"
    config_path = "configs/g009_s0_fixture.json"
    checkpoint_payload = b"checkpoint"
    config_payload = b'{"stage":"S0"}\n'
    _write(tmp_path / "profile", "checkpoints/model.pt", checkpoint_payload)
    _write(tmp_path, config_path, config_payload)
    source_payloads = {
        "src/isaac_walk_g009/terrain.py": b"terrain\n",
        "src/isaac_walk_g009/support_plane.py": b"support\n",
        "scripts/validate_g009_s0.py": b"validator\n",
    }
    for relative, content in source_payloads.items():
        _write(tmp_path, relative, content)
    quantitative_report = {
        "schema_version": 1,
        "goal_id": "g009",
        "stage_id": "S0",
        "validator_id": "g009_s0_import_light_analytic_gate",
        "status": "pass",
        "aggregate": {"cell_count": 24, "pass_count": 24},
        "source_bindings": {
            "terrain_sha256": hashlib.sha256(source_payloads["src/isaac_walk_g009/terrain.py"]).hexdigest(),
            "support_plane_sha256": hashlib.sha256(source_payloads["src/isaac_walk_g009/support_plane.py"]).hexdigest(),
            "validator_sha256": hashlib.sha256(source_payloads["scripts/validate_g009_s0.py"]).hexdigest(),
        },
    }
    quantitative_payload = (json.dumps(quantitative_report) + "\n").encode()
    quantitative_sha = hashlib.sha256(quantitative_payload).hexdigest()
    media_binding = {
        "goal_id": "g009",
        "stage_id": "S0",
        "report_id": "g009_s0_visual_summary",
        "source_commit": source_commit,
        "seed": seed,
        "checkpoint_sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "physics_readback_sha256": physics_sha256,
        "quantitative_report_sha256": quantitative_sha,
    }
    visual_payload = (json.dumps({
        "schema_version": 1,
        "goal_id": "g009",
        "stage_id": "S0",
        "report_id": "g009_s0_visual_summary",
        "status": "complete",
        "media_binding": media_binding,
    }) + "\n").encode()
    payloads = {
        "local_mp4": b"mp4",
        "public_gif": b"GIF89a",
        "public_png": b"\x89PNG\r\n\x1a\n",
        "quantitative_report": quantitative_payload,
        "visual_summary": visual_payload,
    }
    paths = {
        "local_mp4": "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\S0\\capture.mp4",
        "public_gif": "docs/media/g009/S0/capture.gif",
        "public_png": "docs/media/g009/S0/contact.png",
        "quantitative_report": "reports/runs/g009_s0_analytic_validation.json",
        "visual_summary": "reports/runs/g009_s0_visual_summary.json",
    }
    for kind, portable in paths.items():
        if portable.startswith("%USERPROFILE%"):
            relative = portable[len("%USERPROFILE%\\") :]
            _write(tmp_path / "profile", relative, payloads[kind])
        else:
            _write(tmp_path, portable, payloads[kind])
    artifacts = [
        _artifact(
            kind,
            paths[kind],
            payloads[kind],
            "local_only" if kind == "local_mp4" else "git_public",
        )
        for kind in paths
    ]
    sidecar = {
        "schema_version": 1,
        "goal_id": "g009",
        "stage_id": stage_id,
        "status": "complete",
        "bindings": {
            "source_commit": source_commit,
            "seed": seed,
            "report_id": "g009_s0_import_light_analytic_gate",
            "checkpoint": {
                "path": checkpoint_path,
                "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
            },
            "config": {"path": config_path, "sha256": hashlib.sha256(config_payload).hexdigest()},
            "quantitative_report": {
                "path": paths["quantitative_report"],
                "sha256": hashlib.sha256(quantitative_payload).hexdigest(),
            },
            "visual_summary": {
                "path": paths["visual_summary"],
                "sha256": hashlib.sha256(visual_payload).hexdigest(),
            },
            "physics_readback_sha256": physics_sha256,
        },
        "physics_readback": physics_readback,
        "artifacts": artifacts,
    }
    assert validate_sidecar(sidecar, tmp_path) == []

    quantitative_file = tmp_path / paths["quantitative_report"]
    broken_quantitative = json.loads(quantitative_file.read_text(encoding="utf-8"))
    broken_quantitative["aggregate"]["pass_count"] = 23
    quantitative_file.write_text(json.dumps(broken_quantitative), encoding="utf-8")
    assert "S0 quantitative report aggregate.pass_count must be 24" in validate_sidecar(sidecar, tmp_path)
    quantitative_file.write_bytes(quantitative_payload)

    visual_file = tmp_path / paths["visual_summary"]
    broken_visual = json.loads(visual_file.read_text(encoding="utf-8"))
    broken_visual["media_binding"]["source_commit"] = "c" * 40
    visual_file.write_text(json.dumps(broken_visual), encoding="utf-8")
    assert "visual summary media_binding mismatch" in validate_sidecar(sidecar, tmp_path)
    visual_file.write_bytes(visual_payload)

    artifacts[1]["sha256"] = "0" * 64
    artifacts[2]["bytes"] = MAX_PUBLIC_MEDIA_BYTES + 1
    errors = validate_sidecar(sidecar, tmp_path)
    assert "public_gif: sha256 mismatch" in errors
    assert "public_png: exceeds 10 MiB" in errors
    assert "public_png: byte count mismatch" in errors


def test_sidecar_rejects_nonportable_and_wrong_stage_paths() -> None:
    required = STAGE_REGISTRY["S1-low"].required_evidence
    artifacts = []
    for kind in required:
        suffix = {"local_mp4": ".mp4", "public_gif": ".gif", "public_png": ".png"}.get(kind, ".json")
        if kind == "local_mp4":
            path = "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\S0\\bad.mp4"
            policy = "local_only"
        elif kind in {"public_gif", "public_png"}:
            path = f"../docs/media/g009/S1-low/bad{suffix}"
            policy = "git_public"
        else:
            path = f"reports/runs/bad{suffix}"
            policy = "git_public"
        artifacts.append({
            "evidence_type": kind,
            "path": path,
            "sha256": "a" * 64,
            "bytes": 1,
            "git_policy": policy,
        })
    physics = {"static_friction": 0.8}
    sidecar = {
        "schema_version": 1,
        "goal_id": "g009",
        "stage_id": "S1-low",
        "status": "complete",
        "bindings": {
            "source_commit": "b" * 40,
            "seed": 1,
            "report_id": "g009_s1_low_fixture",
            "checkpoint": {"path": "%USERPROFILE%\\checkpoints\\model.pt", "sha256": "a" * 64},
            "config": {"path": "configs/g009_s1_low.json", "sha256": "a" * 64},
            "quantitative_report": {"path": "reports/runs/g009_s1_low_fixture.json", "sha256": "a" * 64},
            "physics_readback_sha256": canonical_json_sha256(physics),
        },
        "physics_readback": physics,
        "artifacts": artifacts,
    }
    errors = validate_sidecar(sidecar, ROOT, check_files=False)
    assert any("local_mp4: invalid path" in error for error in errors)
    assert sum("invalid portable repo path" in error for error in errors) == 2


def test_sidecar_rejects_cross_stage_quantitative_report_binding() -> None:
    physics = {"static_friction": 0.8}
    sidecar = {
        "schema_version": 1,
        "goal_id": "g009",
        "stage_id": "S0",
        "status": "complete",
        "bindings": {
            "source_commit": "b" * 40,
            "seed": 1,
            "report_id": "g009_s0_fixture",
            "checkpoint": {"path": "%USERPROFILE%\\checkpoints\\model.pt", "sha256": "a" * 64},
            "config": {"path": "configs/g009_s0.json", "sha256": "a" * 64},
            "quantitative_report": {"path": "reports/runs/g008_unrelated.json", "sha256": "a" * 64},
            "physics_readback_sha256": canonical_json_sha256(physics),
        },
        "physics_readback": physics,
        "artifacts": [
            {
                "evidence_type": kind,
                "path": (
                    "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\S0\\capture.mp4"
                    if kind == "local_mp4"
                    else f"docs/media/g009/S0/capture.{('gif' if kind == 'public_gif' else 'png')}"
                    if kind in {"public_gif", "public_png"}
                    else "reports/runs/g008_unrelated.json"
                ),
                "sha256": "a" * 64,
                "bytes": 1,
                "git_policy": "local_only" if kind == "local_mp4" else "git_public",
            }
            for kind in ("local_mp4", "public_gif", "public_png", "quantitative_report")
        ],
    }
    errors = validate_sidecar(sidecar, ROOT, check_files=False)
    assert any("quantitative report path must start with g009_s0" in error for error in errors)


def test_cli_emits_json_and_nonzero_on_failure(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "validate_g009_media_contract.py"
    output = tmp_path / "contract.json"
    log = tmp_path / "contract.log"
    success = subprocess.run(
        [sys.executable, str(script), "--output", str(output), "--log", str(log)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert success.returncode == 0
    result = json.loads(success.stdout)
    assert result["status"] == "pass"
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert "status=pass" in log.read_text(encoding="utf-8")
    assert result["canonical_outputs"] == {
        "validator_json": C0_VALIDATOR_JSON_PATH,
        "execution_log": C0_EXECUTION_LOG_PATH,
    }
    assert result["g008_regression"]["path_contract_status"] == "pass"
    assert result["g008_regression"]["execution_status"] == "not_run"
    assert result["g008_regression"]["local_video_references_checked"] > 0
    assert result["rule_diff"]["g008_compatibility_preserved"] is True
    assert set(result["source_bindings"]) == {"agents_sha256", "contract_sha256", "validator_sha256"}

    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"stage_id":"S0","artifacts":[]}', encoding="utf-8")
    failure = subprocess.run(
        [sys.executable, str(script), "--sidecar", str(invalid), "--metadata-only", "--check-only"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert failure.returncode != 0
    result = json.loads(failure.stdout)
    assert result["status"] == "fail"
    assert any("missing required evidence" in error for error in result["errors"])


def test_cli_requires_receipt_outputs_or_explicit_check_only() -> None:
    script = ROOT / "scripts" / "validate_g009_media_contract.py"
    missing_outputs = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert missing_outputs.returncode != 0
    assert "requires both --output and --log" in missing_outputs.stderr

    check_only = subprocess.run(
        [sys.executable, str(script), "--check-only"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert check_only.returncode == 0
    assert json.loads(check_only.stdout)["mode"] == "check_only"
