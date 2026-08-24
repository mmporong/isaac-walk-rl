from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "g007_rbq_asset_manifest.json"
VALIDATOR_PATH = ROOT / "scripts" / "validate_rbq_assets.py"

SPEC = importlib.util.spec_from_file_location("validate_rbq_assets", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def write_manifest(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_cli(
    report: Path,
    mode: str,
    *,
    manifest: Path = MANIFEST_PATH,
    asset_root: Path | str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if asset_root is None:
        environment.pop("ISAAC_WALK_RBQ_ASSET_ROOT", None)
    else:
        environment["ISAAC_WALK_RBQ_ASSET_ROOT"] = str(asset_root)
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--manifest",
            str(manifest),
            "--report",
            str(report),
            mode,
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_expect_blocked_writes_deterministic_fail_closed_report(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    result_one = run_cli(first, "--expect-blocked")
    result_two = run_cli(second, "--expect-blocked")
    assert result_one.returncode == result_two.returncode == 0
    assert first.read_bytes() == second.read_bytes()

    report = json.loads(first.read_text(encoding="utf-8"))
    manifest = load_manifest()
    canonical_manifest = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert report["status"] == "blocked"
    assert report["classification"] == "external_custom_compatibility_spike"
    assert report["blocker"] == {
        "id": "RBQ-ASSET-LICENSE-001",
        "primary_blocker": "license_scope_unresolved",
    }
    assert report["primary_blocker"] == "license_scope_unresolved"
    assert report["converter_executed"] is False
    assert report["smoke_executed"] is False
    assert report["asset_bytes_present"] is False
    assert report["manifest_sha256"] == hashlib.sha256(canonical_manifest).hexdigest()
    assert report["validator_sha256"] == hashlib.sha256(VALIDATOR_PATH.read_bytes()).hexdigest()
    assert report["required_release_evidence"]["release_tag"] == "v1.20.0"
    assert len(report["required_release_evidence"]["pinned_blobs"]) == 8


def test_require_ready_writes_same_report_and_uses_dedicated_exit_code(tmp_path: Path) -> None:
    blocked_report = tmp_path / "blocked.json"
    ready_report = tmp_path / "ready.json"
    blocked = run_cli(blocked_report, "--expect-blocked")
    ready = run_cli(ready_report, "--require-ready")
    assert blocked.returncode == 0
    assert ready.returncode == 3
    assert blocked_report.read_bytes() == ready_report.read_bytes()
    assert "RBQ-ASSET-LICENSE-001" in ready.stderr


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("short_hash", "full lowercase 40-character SHA-1"),
        ("duplicate", "duplicate pinned blob path"),
        ("unsafe_path", "safe pinned repository path"),
        ("wrong_inventory", "inventory does not match"),
        ("license_overclaim", "overclaims"),
    ],
)
def test_invalid_manifest_semantics_are_rejected(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    manifest = copy.deepcopy(load_manifest())
    if mutation == "short_hash":
        manifest["pinned_blobs"][0]["git_blob_sha1"] = "a598350b"
    elif mutation == "duplicate":
        manifest["pinned_blobs"][1]["path"] = manifest["pinned_blobs"][0]["path"]
    elif mutation == "unsafe_path":
        manifest["pinned_blobs"][0]["path"] = "../rbq.urdf"
    elif mutation == "wrong_inventory":
        manifest["pinned_blobs"][0]["size_bytes"] += 1
    elif mutation == "license_overclaim":
        manifest["license_evidence"]["package_manifest"]["applies_to_asset_blobs"] = "confirmed"
        manifest["license_evidence"]["gate"] = "ready"
    manifest_path = tmp_path / f"{mutation}.json"
    write_manifest(manifest_path, manifest)

    result = run_cli(tmp_path / "report.json", "--expect-blocked", manifest=manifest_path)
    assert result.returncode == 2
    assert expected_error in result.stderr
    assert not (tmp_path / "report.json").exists()


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "malformed.json"
    manifest_path.write_text('{"schema_version": 1,', encoding="utf-8")
    result = run_cli(tmp_path / "report.json", "--expect-blocked", manifest=manifest_path)
    assert result.returncode == 2
    assert "malformed JSON at line" in result.stderr


@pytest.mark.parametrize(
    ("needle", "replacement", "object_path", "duplicate_key"),
    [
        (
            '"repository": "RainbowRobotics/RBQ",',
            '"repository": "RainbowRobotics/RBQ",\n    "repository": "RainbowRobotics/RBQ",',
            "$.source",
            "repository",
        ),
        (
            '"schema_version": 1,',
            '"schema_version": 1,\n  "schema_version": 2,',
            "$",
            "schema_version",
        ),
        (
            '"gate": "blocked",',
            '"gate": "blocked",\n    "gate": "ready",',
            "$.license_evidence",
            "gate",
        ),
        (
            '"converter_attempted": false,',
            '"converter_attempted": false,\n    "converter_attempted": true,',
            "$.compatibility",
            "converter_attempted",
        ),
    ],
)
def test_duplicate_json_keys_are_rejected_with_object_location_and_key(
    tmp_path: Path,
    needle: str,
    replacement: str,
    object_path: str,
    duplicate_key: str,
) -> None:
    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    assert raw.count(needle) == 1
    manifest_path = tmp_path / "duplicate.json"
    manifest_path.write_text(raw.replace(needle, replacement), encoding="utf-8")
    report_path = tmp_path / "report.json"
    result = run_cli(report_path, "--expect-blocked", manifest=manifest_path)
    assert result.returncode == 2
    assert f"duplicate JSON key at {object_path}" in result.stderr
    assert repr(duplicate_key) in result.stderr
    assert not report_path.exists()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("schema_version", "1"),
        ("schema_version", []),
        ("schema_version", {}),
        ("expected_count", True),
        ("expected_count", 8.0),
        ("expected_count", "8"),
        ("size_bytes", True),
        ("size_bytes", 19249.0),
        ("size_bytes", "19249"),
        ("converter_attempted", 0),
        ("converter_attempted", "false"),
        ("converter_attempted", []),
        ("smoke_attempted", 0),
        ("smoke_attempted", 1),
        ("smoke_attempted", {}),
    ],
)
def test_json_scalar_types_are_exact_and_bool_cannot_impersonate_int(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    manifest = copy.deepcopy(load_manifest())
    if field == "size_bytes":
        manifest["pinned_blobs"][0][field] = invalid_value
    elif field in {"converter_attempted", "smoke_attempted"}:
        manifest["compatibility"][field] = invalid_value
    else:
        manifest[field] = invalid_value
    manifest_path = tmp_path / f"invalid-{field}.json"
    write_manifest(manifest_path, manifest)
    report_path = tmp_path / "report.json"
    result = run_cli(report_path, "--expect-blocked", manifest=manifest_path)
    assert result.returncode == 2
    assert "must have JSON type" in result.stderr or "positive integer" in result.stderr
    assert not report_path.exists()


def test_cli_requires_exactly_one_gate_mode(tmp_path: Path) -> None:
    base = [
        sys.executable,
        str(VALIDATOR_PATH),
        "--report",
        str(tmp_path / "report.json"),
    ]
    neither = subprocess.run(base, cwd=ROOT, text=True, capture_output=True, check=False)
    both = subprocess.run(
        [*base, "--expect-blocked", "--require-ready"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert neither.returncode == 2
    assert both.returncode == 2
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize(
    "asset_root",
    [
        str(ROOT),
        str(ROOT / "non-existing" / "asset-root"),
        str(ROOT).swapcase(),
        str(ROOT).replace("\\", "/"),
        str(ROOT).replace("\\", "/", 1),
    ],
)
def test_repository_inside_asset_root_is_rejected_for_normalized_path_forms(
    tmp_path: Path,
    asset_root: str,
) -> None:
    report_path = tmp_path / "report.json"
    result = run_cli(report_path, "--expect-blocked", asset_root=asset_root)
    assert result.returncode == 2
    assert "outside the repository" in result.stderr
    assert not report_path.exists()


@pytest.mark.parametrize(
    "asset_root",
    [
        f"\\\\?\\{ROOT}",
        r"\\?\UNC\server\share\rbq",
        r"\\.\C:\rbq",
        r"\\server\share\rbq",
        "//?/C:/rbq",
        r"\\?/C:\rbq",
        r"\??\C:\rbq",
    ],
)
def test_windows_namespace_and_unc_asset_roots_fail_closed(
    tmp_path: Path,
    asset_root: str,
) -> None:
    report_path = tmp_path / "report.json"
    result = run_cli(report_path, "--expect-blocked", asset_root=asset_root)
    assert result.returncode == 2
    assert "UNC, device, or extended-length namespaces" in result.stderr
    assert not report_path.exists()


def test_outside_asset_root_is_presence_only_and_never_unblocks(tmp_path: Path) -> None:
    asset_root = tmp_path / "external-rbq"
    manifest = load_manifest()
    for blob in manifest["pinned_blobs"]:
        asset_path = asset_root.joinpath(*blob["path"].split("/"))
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        with asset_path.open("wb") as stream:
            stream.truncate(blob["size_bytes"])

    report_path = tmp_path / "presence-report.json"
    result = run_cli(report_path, "--expect-blocked", asset_root=asset_root)
    assert result.returncode == 0
    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["asset_bytes_present"] is True
    assert report["status"] == "blocked"
    assert report["converter_executed"] is False
    assert report["smoke_executed"] is False
    assert str(asset_root) not in report_text


def test_reparse_containment_is_rejected_before_presence_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = load_manifest()
    monkeypatch.setenv("ISAAC_WALK_RBQ_ASSET_ROOT", str(ROOT.parent / "external-rbq"))
    monkeypatch.setattr(VALIDATOR, "_is_reparse_point", lambda _path: True)
    with pytest.raises(VALIDATOR.ManifestError, match="reparse point"):
        VALIDATOR.detect_asset_bytes_present(manifest)


def test_report_contains_no_user_home_path_and_topology_stays_unknown(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    result = run_cli(report_path, "--expect-blocked")
    assert result.returncode == 0
    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert str(Path.home()).lower() not in report_text.lower()
    assert report["topology"] == {
        "status": "not_verified_without_asset_bytes",
        "link_count": None,
        "joint_count": None,
        "mesh_count": None,
    }


def test_repository_tracks_no_rbq_asset_or_generated_usd_files() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    tracked = [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]
    forbidden = {".urdf", ".stl", ".usd", ".usda", ".usdc"}
    assert [str(path) for path in tracked if path.suffix.lower() in forbidden] == []
