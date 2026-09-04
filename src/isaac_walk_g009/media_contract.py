from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping


MAX_PUBLIC_MEDIA_BYTES = 10 * 1024 * 1024
LOCAL_VIDEO_PREFIX = "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009"
PUBLIC_MEDIA_PREFIX = "docs/media/g009"
PUBLIC_REPORT_PREFIX = "reports/runs"
C0_VALIDATOR_JSON_PATH = "reports/runs/g009_c0_media_contract.json"
C0_EXECUTION_LOG_PATH = "reports/validation/g009_c0_media_contract.log"
G008_LOCAL_VIDEO_PREFIX = "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g008\\"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SOURCE_VIDEO_FPS = 30.0
TARGET_PUBLIC_GIF_FPS = 15.0
MIN_PUBLIC_GIF_FPS = 12.0
MAX_PUBLIC_GIF_FRAME_DURATION_MS = 84.0
GIF_COMPRESSION_ORDER = ("trim_duration", "reduce_resolution", "reduce_palette")
GIF_TEMPORAL_STRATEGIES: Mapping[str, str] = {
    "camera": "source_frame_sampling",
    "telemetry": "rendered_intermediate_frames",
}


@dataclass(frozen=True)
class StageEvidenceContract:
    stage_id: str
    required_evidence: tuple[str, ...]
    purpose: str


STANDARD_EVIDENCE = (
    "local_mp4",
    "public_gif",
    "public_png",
    "sidecar_json",
    "quantitative_report",
    "visual_summary",
)
DATA_COLLECTION_EVIDENCE = STANDARD_EVIDENCE + (
    "inventory_report",
    "representative_failure",
    "successful_snapshot",
)
SUPERVISOR_EVIDENCE = STANDARD_EVIDENCE + (
    "live_report",
    "snapshot_report",
)


def _stage(stage_id: str, required: tuple[str, ...], purpose: str) -> StageEvidenceContract:
    return StageEvidenceContract(stage_id, required, purpose)


# PRD의 G009 필수 동작 stage registry. C0는 동작 stage가 아니므로 별도 계약이다.
STAGE_REGISTRY: Mapping[str, StageEvidenceContract] = {
    item.stage_id: item
    for item in (
        _stage("S0", STANDARD_EVIDENCE, "analytic slope measurement and geometry"),
        _stage("S1-low", STANDARD_EVIDENCE, "low-slope WALK curriculum"),
        _stage("S1-high", STANDARD_EVIDENCE, "high-slope WALK curriculum"),
        _stage("D0A", STANDARD_EVIDENCE, "G006 regression"),
        _stage("D0B", STANDARD_EVIDENCE, "zero-shot slope transfer"),
        _stage("D0C", STANDARD_EVIDENCE, "slope-training delta"),
        _stage("D1", STANDARD_EVIDENCE, "cross-slope WALK evaluation"),
        _stage("S2", STANDARD_EVIDENCE, "residual-height curriculum"),
        _stage("S3-controlled", STANDARD_EVIDENCE, "controlled asymmetric friction"),
        _stage("S3-spatial", STANDARD_EVIDENCE, "spatial friction field"),
        _stage("R0", STANDARD_EVIDENCE, "flat-ground RECOVER baseline"),
        _stage("F0A", DATA_COLLECTION_EVIDENCE, "natural-fall dataset A collection"),
        _stage("R0B", STANDARD_EVIDENCE, "replay RECOVER baseline"),
        _stage("R1", STANDARD_EVIDENCE, "low-slope RECOVER curriculum"),
        _stage("R2", STANDARD_EVIDENCE, "high-slope RECOVER curriculum"),
        _stage("R3-controlled", STANDARD_EVIDENCE, "controlled-friction RECOVER"),
        _stage("R3-spatial", STANDARD_EVIDENCE, "spatial-friction RECOVER"),
        _stage("D2", STANDARD_EVIDENCE, "WALK-to-RECOVER transition evaluation"),
        _stage("F0B-TV", DATA_COLLECTION_EVIDENCE, "train-validation natural-fall dataset B"),
        _stage("R4", STANDARD_EVIDENCE, "final RECOVER curriculum"),
        _stage("I0", SUPERVISOR_EVIDENCE, "supervisor validation integration"),
        _stage("F0B-FINAL", DATA_COLLECTION_EVIDENCE, "sealed final natural-fall dataset"),
        _stage("I1", SUPERVISOR_EVIDENCE, "held-out supervisor integration"),
        _stage("D3", SUPERVISOR_EVIDENCE, "combined stress evaluation"),
    )
}

STAGE_REPORT_PREFIXES: Mapping[str, str] = {
    "S0": "g009_s0",
    "S1-low": "g009_s1_low",
    "S1-high": "g009_s1_high",
    "D0A": "g009_d0a_g006_regression",
    "D0B": "g009_d0b_zero_transfer",
    "D0C": "g009_d0c_slope_delta",
    "D1": "g009_d1_wrench",
    "S2": "g009_s2_geometry",
    "S3-controlled": "g009_s3_controlled",
    "S3-spatial": "g009_s3_spatial",
    "R0": "g009_r0_flat",
    "F0A": "g009_f0a_falls",
    "R0B": "g009_r0b_replay",
    "R1": "g009_r1_low",
    "R2": "g009_r2_high",
    "R3-controlled": "g009_r3_controlled",
    "R3-spatial": "g009_r3_spatial",
    "D2": "g009_d2_handoff",
    "F0B-TV": "g009_f0b_tv_falls",
    "R4": "g009_r4_natural_fall",
    "I0": "g009_i0_",
    "F0B-FINAL": "g009_f0b_final_falls",
    "I1": "g009_i1_",
    "D3": "g009_d3_stress",
}

C0_REQUIRED_EVIDENCE = ("rule_diff", "validator_json", "execution_log")
C0_FORBIDDEN_EVIDENCE = ("local_mp4", "public_gif", "public_png")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_gif_encoding_metadata(metadata: Any) -> list[str]:
    """Validate metadata produced from a future G009 GIF after encoding."""

    if not isinstance(metadata, Mapping):
        return ["gif_encoding must be an object"]

    errors: list[str] = []
    source_fps = metadata.get("source_video_fps")
    target_fps = metadata.get("target_gif_fps")
    actual_fps = metadata.get("actual_gif_fps")
    frame_count = metadata.get("gif_frame_count")
    duration_seconds = metadata.get("gif_duration_seconds")
    maximum_frame_duration_ms = metadata.get("maximum_frame_duration_ms")
    media_kind = metadata.get("media_kind")
    temporal_strategy = metadata.get("temporal_strategy")

    if not _finite_number(source_fps) or not math.isclose(float(source_fps), SOURCE_VIDEO_FPS):
        errors.append(f"source_video_fps must be {SOURCE_VIDEO_FPS:g}")
    if not _finite_number(target_fps) or not math.isclose(float(target_fps), TARGET_PUBLIC_GIF_FPS):
        errors.append(f"target_gif_fps must be {TARGET_PUBLIC_GIF_FPS:g}")
    if not _finite_number(actual_fps):
        errors.append("actual_gif_fps must be finite")
    elif not MIN_PUBLIC_GIF_FPS <= float(actual_fps) <= SOURCE_VIDEO_FPS:
        errors.append(
            f"actual_gif_fps must be between {MIN_PUBLIC_GIF_FPS:g} and {SOURCE_VIDEO_FPS:g}"
        )

    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 2:
        errors.append("gif_frame_count must be an integer greater than one")
    if not _finite_number(duration_seconds) or float(duration_seconds) <= 0.0:
        errors.append("gif_duration_seconds must be positive and finite")
    if (
        isinstance(frame_count, int)
        and not isinstance(frame_count, bool)
        and frame_count >= 2
        and _finite_number(duration_seconds)
        and float(duration_seconds) > 0.0
        and _finite_number(actual_fps)
    ):
        derived_fps = frame_count / float(duration_seconds)
        if not math.isclose(float(actual_fps), derived_fps, rel_tol=0.02, abs_tol=0.05):
            errors.append("actual_gif_fps does not match frame_count/duration")

    if not _finite_number(maximum_frame_duration_ms):
        errors.append("maximum_frame_duration_ms must be finite")
    elif not 0.0 < float(maximum_frame_duration_ms) <= MAX_PUBLIC_GIF_FRAME_DURATION_MS:
        errors.append(
            f"maximum_frame_duration_ms must be at most {MAX_PUBLIC_GIF_FRAME_DURATION_MS:g}"
        )

    if media_kind not in GIF_TEMPORAL_STRATEGIES:
        errors.append("media_kind must be camera or telemetry")
    elif temporal_strategy != GIF_TEMPORAL_STRATEGIES[media_kind]:
        errors.append(f"{media_kind} temporal_strategy must be {GIF_TEMPORAL_STRATEGIES[media_kind]}")

    compression_policy = metadata.get("compression_policy_order")
    if not isinstance(compression_policy, list) or tuple(compression_policy) != GIF_COMPRESSION_ORDER:
        errors.append(
            "compression_policy_order must be trim_duration, reduce_resolution, reduce_palette"
        )

    compression_steps = metadata.get("compression_steps_applied")
    if not isinstance(compression_steps, list) or tuple(compression_steps) != GIF_COMPRESSION_ORDER[
        : len(compression_steps)
    ]:
        errors.append("compression_steps_applied must be a prefix of compression_policy_order")

    for field in ("width", "height"):
        if not _positive_integer(metadata.get(field)):
            errors.append(f"{field} must be a positive integer")
    palette_colors = metadata.get("palette_colors")
    if not _positive_integer(palette_colors) or not 2 <= palette_colors <= 256:
        errors.append("palette_colors must be an integer from 2 through 256")
    byte_count = metadata.get("bytes")
    if not _positive_integer(byte_count):
        errors.append("bytes must be a positive integer")
    elif byte_count > MAX_PUBLIC_MEDIA_BYTES:
        errors.append("GIF exceeds 10 MiB")

    return errors


def inspect_gif_encoding(path: Path) -> dict[str, Any]:
    """Read the timing fields future builders must write to their sidecar."""

    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.format != "GIF":
                raise ValueError("file is not a GIF")
            width, height = image.size
            frame_count = image.n_frames
            frame_durations_ms: list[float] = []
            for frame_index in range(frame_count):
                image.seek(frame_index)
                duration_ms = image.info.get("duration")
                if not _finite_number(duration_ms) or float(duration_ms) <= 0.0:
                    raise ValueError(f"frame {frame_index} has no positive duration")
                frame_durations_ms.append(float(duration_ms))
    except OSError as exc:
        raise ValueError(f"timing inspection failed: {exc}") from exc

    total_duration_seconds = sum(frame_durations_ms) / 1000.0
    return {
        "actual_gif_fps": frame_count / total_duration_seconds,
        "gif_frame_count": frame_count,
        "gif_duration_seconds": total_duration_seconds,
        "maximum_frame_duration_ms": max(frame_durations_ms),
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
    }


def local_video_directory(stage_id: str) -> str:
    _require_known_stage(stage_id)
    return f"{LOCAL_VIDEO_PREFIX}\\{stage_id}"


def public_media_directory(stage_id: str) -> str:
    _require_known_stage(stage_id)
    return f"{PUBLIC_MEDIA_PREFIX}/{stage_id}"


def _require_known_stage(stage_id: str) -> None:
    if stage_id not in STAGE_REGISTRY:
        raise ValueError(f"unknown G009 stage: {stage_id}")


def _is_portable_repo_path(value: str, prefix: str) -> bool:
    if not value or "\\" in value or ":" in value or value.startswith(("/", "~", "%")):
        return False
    path = PurePosixPath(value)
    return ".." not in path.parts and (value == prefix or value.startswith(f"{prefix}/"))


def _is_local_video_path(value: str, stage_id: str) -> bool:
    expected = PureWindowsPath(local_video_directory(stage_id))
    path = PureWindowsPath(value)
    return (
        path.suffix.lower() == ".mp4"
        and path.parent == expected
        and ".." not in path.parts
        and str(path).startswith(f"{LOCAL_VIDEO_PREFIX}\\")
    )


def resolve_portable_path(value: str, repo_root: Path) -> Path:
    if value.startswith("%USERPROFILE%\\"):
        user_profile = os.environ.get("USERPROFILE")
        if not user_profile:
            raise ValueError("USERPROFILE is not set")
        return Path(user_profile) / Path(value[len("%USERPROFILE%\\") :])
    if not _is_portable_repo_path(value, value.split("/", 1)[0]):
        raise ValueError(f"not a portable path: {value}")
    return repo_root / PurePosixPath(value)


def validate_contract() -> list[str]:
    errors: list[str] = []
    if len(STAGE_REGISTRY) != 24:
        errors.append(f"stage registry count must be 24, got {len(STAGE_REGISTRY)}")
    if tuple(STAGE_REGISTRY) != tuple(item.stage_id for item in STAGE_REGISTRY.values()):
        errors.append("stage registry key/id mismatch")
    if tuple(STAGE_REPORT_PREFIXES) != tuple(STAGE_REGISTRY):
        errors.append("stage report prefix registry mismatch")
    if set(C0_REQUIRED_EVIDENCE) & set(C0_FORBIDDEN_EVIDENCE):
        errors.append("C0 required and forbidden evidence overlap")
    if not SOURCE_VIDEO_FPS >= TARGET_PUBLIC_GIF_FPS >= MIN_PUBLIC_GIF_FPS >= 12.0:
        errors.append("smooth GIF frame-rate contract is invalid")
    if GIF_COMPRESSION_ORDER != ("trim_duration", "reduce_resolution", "reduce_palette"):
        errors.append("smooth GIF compression order is invalid")
    if GIF_TEMPORAL_STRATEGIES != {
        "camera": "source_frame_sampling",
        "telemetry": "rendered_intermediate_frames",
    }:
        errors.append("smooth GIF temporal strategies are invalid")
    for stage_id, contract in STAGE_REGISTRY.items():
        if len(contract.required_evidence) != len(set(contract.required_evidence)):
            errors.append(f"{stage_id}: duplicate required evidence")
        for required in STANDARD_EVIDENCE:
            if required not in contract.required_evidence:
                errors.append(f"{stage_id}: missing {required}")
        if not local_video_directory(stage_id).startswith(f"{LOCAL_VIDEO_PREFIX}\\"):
            errors.append(f"{stage_id}: invalid local video directory")
        if not _is_portable_repo_path(public_media_directory(stage_id), PUBLIC_MEDIA_PREFIX):
            errors.append(f"{stage_id}: invalid public media directory")
    return errors


def _walk_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


def validate_repository_media_rules(repo_root: Path) -> list[str]:
    """Validate the generic goal path rule and the preserved G008 evidence paths."""
    errors: list[str] = []
    agents_path = repo_root / "AGENTS.md"
    if not agents_path.is_file():
        errors.append("repository AGENTS.md not found")
    else:
        rules = agents_path.read_text(encoding="utf-8")
        if "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\<goal_id>" not in rules:
            errors.append("AGENTS.md missing generic <goal_id> visual evidence path")
        if "기존 G008 증거" not in rules or "g008" not in rules:
            errors.append("AGENTS.md missing G008 compatibility rule")

    repository_mp4 = sorted(repo_root.rglob("*.mp4"))
    if repository_mp4:
        errors.append(f"repository contains MP4 files: {len(repository_mp4)}")

    runs_root = repo_root / PUBLIC_REPORT_PREFIX
    checked_g008_mp4 = 0
    for report_path in sorted(runs_root.glob("g008*.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"G008 report read failed: {report_path.name}: {exc}")
            continue
        for item in _walk_objects(report):
            path_value = item.get("path")
            if not isinstance(path_value, str) or not path_value.lower().endswith(".mp4"):
                continue
            checked_g008_mp4 += 1
            if item.get("git_policy") != "local_only":
                errors.append(f"{report_path.name}: G008 MP4 must be local_only")
            if not path_value.startswith(G008_LOCAL_VIDEO_PREFIX):
                errors.append(f"{report_path.name}: G008 MP4 path changed: {path_value}")
    if checked_g008_mp4 == 0:
        errors.append("no G008 local MP4 evidence paths were checked")
    return errors


def count_g008_local_video_evidence(repo_root: Path) -> int:
    """Count preserved G008 local-only MP4 references used by the C0 regression."""
    count = 0
    for report_path in sorted((repo_root / PUBLIC_REPORT_PREFIX).glob("g008*.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        for item in _walk_objects(report):
            path_value = item.get("path")
            if (
                isinstance(path_value, str)
                and path_value.lower().endswith(".mp4")
                and path_value.startswith(G008_LOCAL_VIDEO_PREFIX)
                and item.get("git_policy") == "local_only"
            ):
                count += 1
    return count


def _artifact_index(sidecar: Mapping[str, Any], errors: list[str]) -> dict[str, Mapping[str, Any]]:
    artifacts = sidecar.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("artifacts must be a list")
        return {}
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            errors.append(f"artifacts[{index}] must be an object")
            continue
        evidence_type = artifact.get("evidence_type")
        if not isinstance(evidence_type, str) or not evidence_type:
            errors.append(f"artifacts[{index}].evidence_type is required")
            continue
        if evidence_type in indexed:
            errors.append(f"duplicate evidence type: {evidence_type}")
            continue
        indexed[evidence_type] = artifact
    return indexed


def validate_sidecar(
    sidecar: Mapping[str, Any], repo_root: Path, *, check_files: bool = True
) -> list[str]:
    errors = validate_contract()
    stage_id = sidecar.get("stage_id")
    if not isinstance(stage_id, str) or stage_id not in STAGE_REGISTRY:
        errors.append(f"unknown sidecar stage_id: {stage_id!r}")
        return errors
    if sidecar.get("schema_version") != 1:
        errors.append("sidecar schema_version must be 1")
    if sidecar.get("goal_id") != "g009":
        errors.append("sidecar goal_id must be g009")
    if sidecar.get("status") != "complete":
        errors.append("sidecar status must be complete")

    bindings = sidecar.get("bindings")
    if not isinstance(bindings, Mapping):
        errors.append("bindings must be an object")
        bindings = {}
    source_commit = bindings.get("source_commit")
    seed = bindings.get("seed")
    report_id = bindings.get("report_id")
    if not isinstance(source_commit, str) or not GIT_COMMIT_RE.fullmatch(source_commit):
        errors.append("bindings.source_commit must be 40 lowercase hex characters")
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append("bindings.seed must be an integer")
    if not isinstance(report_id, str) or not report_id.startswith(STAGE_REPORT_PREFIXES[stage_id]):
        errors.append(f"bindings.report_id must start with {STAGE_REPORT_PREFIXES[stage_id]}")
    binding_artifacts: dict[str, Mapping[str, Any]] = {}
    for name in ("checkpoint", "config", "quantitative_report", "visual_summary"):
        value = bindings.get(name)
        if not isinstance(value, Mapping):
            errors.append(f"bindings.{name} must be an object")
            continue
        path_value = value.get("path")
        digest = value.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            errors.append(f"bindings.{name}.path is required")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            errors.append(f"bindings.{name}.sha256 must be 64 lowercase hex characters")
        binding_artifacts[name] = value
    physics_readback = sidecar.get("physics_readback")
    physics_sha256 = bindings.get("physics_readback_sha256")
    if not isinstance(physics_readback, Mapping):
        errors.append("physics_readback must be an object")
    elif not isinstance(physics_sha256, str) or not SHA256_RE.fullmatch(physics_sha256):
        errors.append("bindings.physics_readback_sha256 must be 64 lowercase hex characters")
    elif canonical_json_sha256(physics_readback) != physics_sha256:
        errors.append("physics_readback hash mismatch")
    artifacts = _artifact_index(sidecar, errors)
    for evidence_type in STAGE_REGISTRY[stage_id].required_evidence:
        # The validated document is the sidecar. Requiring it to hash itself would
        # create an impossible circular digest dependency.
        if evidence_type == "sidecar_json":
            continue
        if evidence_type not in artifacts:
            errors.append(f"{stage_id}: missing required evidence {evidence_type}")

    for evidence_type, artifact in artifacts.items():
        path_value = artifact.get("path")
        digest = artifact.get("sha256")
        byte_count = artifact.get("bytes")
        if not isinstance(path_value, str):
            errors.append(f"{evidence_type}: path is required")
            continue
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            errors.append(f"{evidence_type}: sha256 must be 64 lowercase hex characters")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            errors.append(f"{evidence_type}: bytes must be a non-negative integer")

        if evidence_type == "local_mp4":
            if artifact.get("git_policy") != "local_only":
                errors.append("local_mp4: git_policy must be local_only")
            if not _is_local_video_path(path_value, stage_id):
                errors.append(f"local_mp4: invalid path for {stage_id}: {path_value}")
        else:
            expected_prefix = (
                public_media_directory(stage_id)
                if evidence_type in {"public_gif", "public_png"}
                else PUBLIC_REPORT_PREFIX
            )
            if artifact.get("git_policy") != "git_public":
                errors.append(f"{evidence_type}: git_policy must be git_public")
            if not _is_portable_repo_path(path_value, expected_prefix):
                errors.append(f"{evidence_type}: invalid portable repo path: {path_value}")

        suffix = PureWindowsPath(path_value).suffix.lower()
        if evidence_type == "public_gif" and suffix != ".gif":
            errors.append("public_gif: path must end in .gif")
        if evidence_type == "public_png" and suffix != ".png":
            errors.append("public_png: path must end in .png")
        if evidence_type in {"public_gif", "public_png"} and isinstance(byte_count, int):
            if byte_count > MAX_PUBLIC_MEDIA_BYTES:
                errors.append(f"{evidence_type}: exceeds 10 MiB")

        if check_files:
            try:
                resolved = resolve_portable_path(path_value, repo_root)
            except ValueError as exc:
                errors.append(f"{evidence_type}: {exc}")
                continue
            if not resolved.is_file():
                errors.append(f"{evidence_type}: file not found: {path_value}")
                continue
            if evidence_type == "public_gif" and not resolved.read_bytes().startswith((b"GIF87a", b"GIF89a")):
                errors.append("public_gif: invalid GIF signature")
            if evidence_type == "public_png" and not resolved.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                errors.append("public_png: invalid PNG signature")
            actual_bytes = resolved.stat().st_size
            if byte_count != actual_bytes:
                errors.append(f"{evidence_type}: byte count mismatch")
            if isinstance(digest, str) and SHA256_RE.fullmatch(digest):
                if file_sha256(resolved) != digest:
                    errors.append(f"{evidence_type}: sha256 mismatch")

    quantitative_artifact = artifacts.get("quantitative_report")
    quantitative_binding = binding_artifacts.get("quantitative_report")
    if quantitative_artifact is not None and quantitative_binding is not None:
        if quantitative_artifact.get("path") != quantitative_binding.get("path"):
            errors.append("quantitative_report artifact/binding path mismatch")
        if quantitative_artifact.get("sha256") != quantitative_binding.get("sha256"):
            errors.append("quantitative_report artifact/binding sha256 mismatch")
        report_path_value = quantitative_binding.get("path")
        if isinstance(report_path_value, str):
            report_name = PurePosixPath(report_path_value).name
            if not report_name.startswith(STAGE_REPORT_PREFIXES[stage_id]):
                errors.append(f"quantitative report path must start with {STAGE_REPORT_PREFIXES[stage_id]}")

    visual_artifact = artifacts.get("visual_summary")
    visual_binding = binding_artifacts.get("visual_summary")
    if visual_artifact is not None and visual_binding is not None:
        if visual_artifact.get("path") != visual_binding.get("path"):
            errors.append("visual_summary artifact/binding path mismatch")
        if visual_artifact.get("sha256") != visual_binding.get("sha256"):
            errors.append("visual_summary artifact/binding sha256 mismatch")
        visual_path_value = visual_binding.get("path")
        if isinstance(visual_path_value, str):
            visual_name = PurePosixPath(visual_path_value).name
            if not visual_name.startswith(STAGE_REPORT_PREFIXES[stage_id]):
                errors.append(f"visual summary path must start with {STAGE_REPORT_PREFIXES[stage_id]}")

    for name in ("checkpoint", "config"):
        binding = binding_artifacts.get(name)
        if binding is None:
            continue
        path_value = binding.get("path")
        if not isinstance(path_value, str):
            continue
        if name == "checkpoint":
            if not path_value.startswith("%USERPROFILE%\\") or PureWindowsPath(path_value).suffix.lower() != ".pt":
                errors.append("bindings.checkpoint.path must be a portable %USERPROFILE% .pt path")
        elif not _is_portable_repo_path(path_value, "configs"):
            errors.append("bindings.config.path must be under configs/")
        if check_files:
            try:
                resolved = resolve_portable_path(path_value, repo_root)
            except ValueError as exc:
                errors.append(f"bindings.{name}: {exc}")
                continue
            if not resolved.is_file():
                errors.append(f"bindings.{name}: file not found: {path_value}")
            elif isinstance(binding.get("sha256"), str) and file_sha256(resolved) != binding.get("sha256"):
                errors.append(f"bindings.{name}: sha256 mismatch")

    if check_files and quantitative_binding is not None and isinstance(quantitative_binding.get("path"), str):
        try:
            report_path = resolve_portable_path(quantitative_binding["path"], repo_root)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"quantitative report binding read failed: {exc}")
        else:
            if report.get("goal_id") != "g009":
                errors.append("quantitative report goal_id mismatch")
            if report.get("stage_id") != stage_id:
                errors.append("quantitative report stage_id mismatch")
            if report.get("status") != "pass":
                errors.append("quantitative report status must be pass")
            if stage_id == "S0":
                if quantitative_binding.get("path") != "reports/runs/g009_s0_analytic_validation.json":
                    errors.append("S0 quantitative report must be g009_s0_analytic_validation.json")
                if report.get("validator_id") != report_id:
                    errors.append("quantitative report validator_id/report_id mismatch")
                aggregate = report.get("aggregate")
                if not isinstance(aggregate, Mapping):
                    errors.append("S0 quantitative report aggregate must be an object")
                else:
                    if aggregate.get("cell_count") != 24:
                        errors.append("S0 quantitative report aggregate.cell_count must be 24")
                    if aggregate.get("pass_count") != 24:
                        errors.append("S0 quantitative report aggregate.pass_count must be 24")
                source_bindings = report.get("source_bindings")
                expected_sources = {
                    "terrain_sha256": "src/isaac_walk_g009/terrain.py",
                    "support_plane_sha256": "src/isaac_walk_g009/support_plane.py",
                    "validator_sha256": "scripts/validate_g009_s0.py",
                }
                if not isinstance(source_bindings, Mapping):
                    errors.append("S0 quantitative report source_bindings must be an object")
                else:
                    for field, relative_path in expected_sources.items():
                        source_digest = source_bindings.get(field)
                        if not isinstance(source_digest, str) or not SHA256_RE.fullmatch(source_digest):
                            errors.append(f"S0 quantitative report source_bindings.{field} is invalid")
                            continue
                        source_path = repo_root / relative_path
                        if not source_path.is_file():
                            errors.append(f"S0 quantitative source not found: {relative_path}")
                        elif file_sha256(source_path) != source_digest:
                            errors.append(f"S0 quantitative source hash mismatch: {relative_path}")

    if check_files and visual_binding is not None and isinstance(visual_binding.get("path"), str):
        try:
            visual_path = resolve_portable_path(visual_binding["path"], repo_root)
            visual_report = json.loads(visual_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"visual summary binding read failed: {exc}")
        else:
            expected_visual_binding = {
                "goal_id": "g009",
                "stage_id": stage_id,
                "report_id": visual_report.get("report_id"),
                "source_commit": source_commit,
                "seed": seed,
                "checkpoint_sha256": binding_artifacts.get("checkpoint", {}).get("sha256"),
                "config_sha256": binding_artifacts.get("config", {}).get("sha256"),
                "physics_readback_sha256": physics_sha256,
                "quantitative_report_sha256": binding_artifacts.get("quantitative_report", {}).get("sha256"),
            }
            if visual_report.get("goal_id") != "g009" or visual_report.get("stage_id") != stage_id:
                errors.append("visual summary goal/stage mismatch")
            if visual_report.get("status") != "complete":
                errors.append("visual summary status must be complete")
            if visual_report.get("media_binding") != expected_visual_binding:
                errors.append("visual summary media_binding mismatch")
    return errors


def validate_c0_evidence(evidence_types: Iterable[str]) -> list[str]:
    provided = tuple(evidence_types)
    errors = [item for item in C0_REQUIRED_EVIDENCE if item not in provided]
    result = [f"C0: missing required evidence {item}" for item in errors]
    result.extend(
        f"C0: forbidden media evidence {item}"
        for item in C0_FORBIDDEN_EVIDENCE
        if item in provided
    )
    return result
