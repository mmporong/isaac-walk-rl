#!/usr/bin/env python3
"""Validate the pinned RBQ evidence and fail closed before asset conversion.

This gate intentionally does not parse or hash RBQ asset bytes.  Until the
license scope is resolved, even a locally supplied checkout can only contribute
an ``asset_bytes_present`` boolean to the compatibility-spike report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = 1
CLASSIFICATION = "external_custom_compatibility_spike"
BLOCKER_ID = "RBQ-ASSET-LICENSE-001"
PRIMARY_BLOCKER = "license_scope_unresolved"
ASSET_ROOT_ENV = "ISAAC_WALK_RBQ_ASSET_ROOT"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "g007_rbq_asset_manifest.json"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_SOURCE = {
    "repository": "RainbowRobotics/RBQ",
    "release_tag": "v1.20.0",
    "tag_object_sha1": "741ce5733dcd7c0babec663bb7e1afbc02a776ca",
    "source_commit_sha1": "68bc33b77719d357b4323fb88549efd905caf721",
    "asset_root_prefix": "rbq_sdk/ros2/src/rbq_description/",
}

EXPECTED_BLOBS = (
    (
        "robot_description",
        "rbq_sdk/ros2/src/rbq_description/urdf/rbq.urdf",
        19249,
        "a598350ba21dc521db7bb16cba199ef35507477e",
    ),
    (
        "package_manifest",
        "rbq_sdk/ros2/src/rbq_description/package.xml",
        909,
        "c631a432aa1e0083e14b60417bf5b6453552338e",
    ),
    (
        "mesh",
        "rbq_sdk/ros2/src/rbq_description/meshes/stl/calf.STL",
        71284,
        "253f2ee9e4bb67485223faa1951ad89f4442c183",
    ),
    (
        "mesh",
        "rbq_sdk/ros2/src/rbq_description/meshes/stl/hip2.stl",
        866584,
        "0078f9689f7fc31389b64a97b4b01e55a41b6b18",
    ),
    (
        "mesh",
        "rbq_sdk/ros2/src/rbq_description/meshes/stl/hip3.stl",
        818784,
        "8bc31dd3a7aed9926108477a03f200c52012339c",
    ),
    (
        "mesh",
        "rbq_sdk/ros2/src/rbq_description/meshes/stl/mid-360.stl",
        4193734,
        "ae42618a7b09f274f651156a203bfcc203abc354",
    ),
    (
        "mesh",
        "rbq_sdk/ros2/src/rbq_description/meshes/stl/thigh.stl",
        25784,
        "dbfa93ebe31302bb8fa7b383d4dda1b38c132fb6",
    ),
    (
        "mesh",
        "rbq_sdk/ros2/src/rbq_description/meshes/stl/trunk.stl",
        504284,
        "ddc44a3c5f7f41f42a63eb1a6981dfd1d58cd339",
    ),
)

EXPECTED_LICENSE = {
    "github_repository_detected_spdx": None,
    "github_repository_detection_interpretation": (
        "no repository-wide license detected; not proof of no license"
    ),
    "package_manifest": {
        "path": "rbq_sdk/ros2/src/rbq_description/package.xml",
        "declared_expression": "Apache-2.0",
        "scope": "package_manifest_declaration",
        "applies_to_asset_blobs": "unresolved",
    },
    "asset_adjacent_license_files": [],
    "redistribution": "unknown",
    "local_processing": "unknown",
    "gate": "blocked",
    "primary_blocker": PRIMARY_BLOCKER,
}

EXPECTED_ISAACLAB_SEARCHES = [
    {
        "ref": "v2.1.1",
        "commit": "90b79bb2d44feb8d833f260f2bf37da3487180ba",
        "rbq_matches": [],
    },
    {
        "ref": "v2.3.2",
        "commit": "37ddf626871758333d6ed89cf64ad702aef127d0",
        "rbq_matches": [],
    },
    {
        "ref": "main",
        "commit": "b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8",
        "rbq_matches": [],
    },
]

EXPECTED_TOPOLOGY = {
    "status": "not_verified_without_asset_bytes",
    "link_count": None,
    "joint_count": None,
    "mesh_count": None,
}

EXPECTED_COMPATIBILITY = {
    "status": "blocked_before_conversion",
    "converter_attempted": False,
    "smoke_attempted": False,
}


class ManifestError(ValueError):
    """Raised when the pinned manifest violates the fail-closed contract."""


class JsonObjectPairs(list[tuple[str, Any]]):
    """Distinguish JSON objects from arrays while preserving duplicate keys."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_path_key(parent: str, key: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{parent}.{key}"
    return f"{parent}[{json.dumps(key, ensure_ascii=False)}]"


def _strict_json_value(value: Any, path: str = "$") -> Any:
    if isinstance(value, JsonObjectPairs):
        result: dict[str, Any] = {}
        for key, child in value:
            if key in result:
                raise ManifestError(f"duplicate JSON key at {path}: {key!r}")
            result[key] = _strict_json_value(child, _json_path_key(path, key))
        return result
    if isinstance(value, list):
        return [_strict_json_value(child, f"{path}[{index}]") for index, child in enumerate(value)]
    return value


def strict_json_loads(text: str) -> Any:
    try:
        parsed = json.loads(text, object_pairs_hook=JsonObjectPairs)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    return _strict_json_value(parsed)


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ManifestError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _require_sha1(value: Any, label: str) -> None:
    if not isinstance(value, str) or SHA1_RE.fullmatch(value) is None:
        raise ManifestError(f"{label} must be a full lowercase 40-character SHA-1")


def _require_exact_type(value: Any, expected_type: type, label: str) -> None:
    if type(value) is not expected_type:
        raise ManifestError(f"{label} must have JSON type {expected_type.__name__}")


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ManifestError("manifest must be a JSON object")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "classification",
            "expected_count",
            "source",
            "pinned_blobs",
            "license_evidence",
            "official_isaaclab_searches",
            "topology",
            "compatibility",
        },
        "manifest",
    )
    _require_exact_type(manifest["schema_version"], int, "schema_version")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestError("schema_version must be 1")
    _require_exact_type(manifest["expected_count"], int, "expected_count")
    if manifest["expected_count"] != len(EXPECTED_BLOBS):
        raise ManifestError("expected_count must be 8")
    _require_exact_type(manifest["classification"], str, "classification")
    if manifest["classification"] != CLASSIFICATION:
        raise ManifestError(f"classification must be {CLASSIFICATION}")

    source = manifest["source"]
    if type(source) is not dict:
        raise ManifestError("source must be an object")
    _require_exact_keys(source, set(EXPECTED_SOURCE), "source")
    for key in ("tag_object_sha1", "source_commit_sha1"):
        _require_sha1(source.get(key), f"source.{key}")
    if source != EXPECTED_SOURCE:
        raise ManifestError("source does not match the pinned RBQ v1.20.0 evidence")

    blobs = manifest["pinned_blobs"]
    if type(blobs) is not list or len(blobs) != manifest["expected_count"]:
        raise ManifestError("pinned_blobs must contain exactly 8 entries")
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    actual_inventory: list[tuple[str, str, int, str]] = []
    for index, blob in enumerate(blobs):
        if type(blob) is not dict:
            raise ManifestError(f"pinned_blobs[{index}] must be an object")
        _require_exact_keys(
            blob,
            {"role", "path", "size_bytes", "git_type", "git_blob_sha1"},
            f"pinned_blobs[{index}]",
        )
        path = blob["path"]
        if type(path) is not str:
            raise ManifestError(f"pinned_blobs[{index}].path must be a string")
        pure_path = PurePosixPath(path)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or "\\" in path
            or not path.startswith(EXPECTED_SOURCE["asset_root_prefix"])
        ):
            raise ManifestError(f"pinned_blobs[{index}].path is not a safe pinned repository path")
        if path in seen_paths:
            raise ManifestError(f"duplicate pinned blob path: {path}")
        seen_paths.add(path)
        blob_sha1 = blob["git_blob_sha1"]
        _require_sha1(blob_sha1, f"pinned_blobs[{index}].git_blob_sha1")
        if blob_sha1 in seen_hashes:
            raise ManifestError(f"duplicate pinned blob SHA-1: {blob_sha1}")
        seen_hashes.add(blob_sha1)
        _require_exact_type(blob["git_type"], str, f"pinned_blobs[{index}].git_type")
        if blob["git_type"] != "blob":
            raise ManifestError(f"pinned_blobs[{index}].git_type must be blob")
        if type(blob["size_bytes"]) is not int or blob["size_bytes"] <= 0:
            raise ManifestError(f"pinned_blobs[{index}].size_bytes must be a positive integer")
        _require_exact_type(blob["role"], str, f"pinned_blobs[{index}].role")
        if blob["role"] not in {"robot_description", "package_manifest", "mesh"}:
            raise ManifestError(f"pinned_blobs[{index}].role is invalid")
        actual_inventory.append((blob["role"], path, blob["size_bytes"], blob_sha1))
    if tuple(actual_inventory) != EXPECTED_BLOBS:
        raise ManifestError("pinned blob role/path/size/SHA inventory does not match the contract")

    license_evidence = manifest["license_evidence"]
    if type(license_evidence) is not dict:
        raise ManifestError("license_evidence must be an object")
    _require_exact_keys(license_evidence, set(EXPECTED_LICENSE), "license_evidence")
    package_manifest = license_evidence.get("package_manifest")
    if type(package_manifest) is not dict:
        raise ManifestError("license_evidence.package_manifest must be an object")
    _require_exact_keys(
        package_manifest,
        set(EXPECTED_LICENSE["package_manifest"]),
        "license_evidence.package_manifest",
    )
    if license_evidence != EXPECTED_LICENSE:
        raise ManifestError(
            "license evidence overclaims the unresolved asset-blob scope or changes the blocked gate"
        )

    searches = manifest["official_isaaclab_searches"]
    if type(searches) is not list:
        raise ManifestError("official_isaaclab_searches must be a list")
    for index, search in enumerate(searches):
        if type(search) is not dict:
            raise ManifestError(f"official_isaaclab_searches[{index}] must be an object")
        _require_exact_keys(search, {"ref", "commit", "rbq_matches"}, f"official_isaaclab_searches[{index}]")
        _require_exact_type(search["ref"], str, f"official_isaaclab_searches[{index}].ref")
        _require_sha1(search.get("commit"), f"official_isaaclab_searches[{index}].commit")
        _require_exact_type(search["rbq_matches"], list, f"official_isaaclab_searches[{index}].rbq_matches")
    if searches != EXPECTED_ISAACLAB_SEARCHES:
        raise ManifestError("official Isaac Lab search evidence does not match the pinned refs")

    topology = manifest["topology"]
    if type(topology) is not dict or topology != EXPECTED_TOPOLOGY:
        raise ManifestError("topology must remain unverified with null counts before asset-byte review")
    compatibility = manifest["compatibility"]
    if type(compatibility) is not dict:
        raise ManifestError("compatibility must be an object")
    _require_exact_keys(compatibility, set(EXPECTED_COMPATIBILITY), "compatibility")
    _require_exact_type(compatibility["status"], str, "compatibility.status")
    _require_exact_type(compatibility["converter_attempted"], bool, "compatibility.converter_attempted")
    _require_exact_type(compatibility["smoke_attempted"], bool, "compatibility.smoke_attempted")
    if compatibility != EXPECTED_COMPATIBILITY:
        raise ManifestError("compatibility must remain blocked before conversion and smoke")
    return manifest


def load_and_validate_manifest(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = strict_json_loads(text)
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"cannot load manifest: {exc}") from exc
    return validate_manifest(value)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _path_has_reparse_below_root(root: Path, candidate: Path) -> bool:
    current = root
    if _is_reparse_point(current):
        return True
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return True
    for part in relative.parts:
        current = current / part
        if _is_reparse_point(current):
            return True
        if not current.exists():
            break
    return False


def _reject_windows_namespace(raw_root: str) -> str:
    candidate = raw_root.strip()
    normalized = candidate.replace("/", "\\")
    upper = normalized.upper()
    if upper.startswith("\\\\") or upper.startswith("\\??\\"):
        raise ManifestError(
            f"{ASSET_ROOT_ENV} must not use UNC, device, or extended-length namespaces"
        )
    return candidate


def detect_asset_bytes_present(manifest: dict[str, Any]) -> bool:
    raw_root = os.environ.get(ASSET_ROOT_ENV)
    if raw_root is None or not raw_root.strip():
        return False

    safe_root = _reject_windows_namespace(raw_root)
    lexical_root = Path(safe_root).expanduser().absolute()
    if _is_reparse_point(lexical_root):
        raise ManifestError(f"{ASSET_ROOT_ENV} must not be a reparse point")
    resolved_root = lexical_root.resolve(strict=False)
    resolved_repo = REPO_ROOT.resolve(strict=True)
    if _is_relative_to(resolved_root, resolved_repo):
        raise ManifestError(f"{ASSET_ROOT_ENV} must be outside the repository")

    all_present = True
    for blob in manifest["pinned_blobs"]:
        lexical_candidate = lexical_root.joinpath(*PurePosixPath(blob["path"]).parts)
        if _path_has_reparse_below_root(lexical_root, lexical_candidate):
            raise ManifestError(f"{ASSET_ROOT_ENV} asset containment must not cross a reparse point")
        resolved_candidate = lexical_candidate.resolve(strict=False)
        if not _is_relative_to(resolved_candidate, resolved_root):
            raise ManifestError(f"{ASSET_ROOT_ENV} asset containment escaped its root")
        try:
            metadata = resolved_candidate.stat()
        except OSError:
            all_present = False
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != blob["size_bytes"]:
            all_present = False
    return all_present


def build_report(manifest: dict[str, Any], asset_bytes_present: bool) -> dict[str, Any]:
    manifest_hash = sha256_bytes(canonical_json_bytes(manifest))
    validator_hash = sha256_bytes(Path(__file__).read_bytes())
    required_release_evidence = {
        **manifest["source"],
        "pinned_blobs": manifest["pinned_blobs"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "classification": CLASSIFICATION,
        "blocker": {
            "id": BLOCKER_ID,
            "primary_blocker": PRIMARY_BLOCKER,
        },
        "primary_blocker": PRIMARY_BLOCKER,
        "converter_executed": False,
        "smoke_executed": False,
        "topology": EXPECTED_TOPOLOGY,
        "asset_bytes_present": asset_bytes_present,
        "manifest_sha256": manifest_hash,
        "validator_sha256": validator_hash,
        "required_release_evidence": required_release_evidence,
    }


def write_report_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--expect-blocked", action="store_true")
    mode.add_argument("--require-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_and_validate_manifest(args.manifest)
        asset_bytes_present = detect_asset_bytes_present(manifest)
        report = build_report(manifest, asset_bytes_present)
        write_report_atomic(args.report, report)
    except ManifestError as exc:
        print(f"RBQ manifest validation failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"RBQ report write failed: {exc}", file=sys.stderr)
        return 2

    if args.require_ready:
        print(f"RBQ asset gate blocked: {BLOCKER_ID} ({PRIMARY_BLOCKER})", file=sys.stderr)
        return 3
    print(f"RBQ asset gate is correctly blocked: {BLOCKER_ID} ({PRIMARY_BLOCKER})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
