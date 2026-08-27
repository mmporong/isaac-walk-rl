#!/usr/bin/env python3
"""Build fail-closed G009 S0 slope playback evidence from three captures."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from isaac_walk_g009.media_contract import (  # noqa: E402
    MAX_PUBLIC_MEDIA_BYTES,
    canonical_json_sha256,
    file_sha256,
    validate_sidecar,
)


CONFIG_PATH = "configs/g009_s0.json"
QUANTITATIVE_PATH = "reports/runs/g009_s0_analytic_validation.json"
SUMMARY_PATH = "reports/runs/g009_s0_visual_summary.json"
SIDECAR_PATH = "reports/runs/g009_s0_visual_evidence.json"
PUBLIC_GIF_PATH = "docs/media/g009/S0/g009_s0_slopes.gif"
PUBLIC_PNG_PATH = "docs/media/g009/S0/g009_s0_slopes_contact_sheet.png"
LOCAL_MP4_PATH = "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\S0\\g009_s0_slopes.mp4"
REPORT_ID = "g009_s0_visual_summary"
RECORD_SOURCE_PATH = "scripts/record_g009_s0.py"
BUILDER_SOURCE_PATH = "scripts/build_g009_s0_media.py"
EXPECTED_PROFILE_IDS = ("slope_05", "slope_15", "slope_25_stress")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON read failed: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _portable_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside repository: {path}") from exc


def _resolve_portable(value: str) -> Path:
    if value.startswith("%USERPROFILE%\\"):
        profile = os.environ.get("USERPROFILE")
        if not profile:
            raise ValueError("USERPROFILE is not set")
        return Path(profile) / Path(value[len("%USERPROFILE%\\") :])
    _require("\\" not in value and ":" not in value, f"non-portable repository path: {value}")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts, f"non-portable repository path: {value}")
    return REPO_ROOT / path


def _is_local_s0_mp4(value: str) -> bool:
    path = PureWindowsPath(value)
    expected = PureWindowsPath("%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\S0")
    return path.parent == expected and path.suffix.lower() == ".mp4" and ".." not in path.parts


def _validate_file_binding(
    binding: Mapping[str, Any], *, name: str, expected_path: str | None = None
) -> Path:
    path_value = binding.get("path")
    digest = binding.get("sha256")
    _require(isinstance(path_value, str) and bool(path_value), f"{name}.path is required")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, f"{name}.sha256 is invalid")
    if expected_path is not None:
        _require(path_value == expected_path, f"{name}.path mismatch: expected={expected_path}, actual={path_value}")
    resolved = _resolve_portable(path_value)
    _require(resolved.is_file(), f"{name} file not found: {path_value}")
    _require(file_sha256(resolved) == digest, f"{name}.sha256 mismatch")
    return resolved


def _normalise_sequence(sequence: Any) -> Any:
    return json.loads(json.dumps(sequence, ensure_ascii=False, sort_keys=True))


def _git_commit_file_sha256(source_commit: str, repo_path: str) -> str:
    """Return the SHA-256 of a file exactly as stored in a commit tree."""
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    _require(commit_check.returncode == 0, f"source_commit not found in git object database: {source_commit}")
    blob = subprocess.run(
        ["git", "cat-file", "blob", f"{source_commit}:{repo_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    _require(blob.returncode == 0, f"commit tree is missing {repo_path}: {source_commit}")
    import hashlib

    return hashlib.sha256(blob.stdout).hexdigest()


def _validate_analytic_report(path: Path) -> tuple[dict[str, Any], str]:
    report = _read_json(path)
    for field, expected in (("goal_id", "g009"), ("stage_id", "S0"), ("status", "pass")):
        _require(report.get(field) == expected, f"analytic report {field} must be {expected!r}")
    aggregate = report.get("aggregate")
    _require(isinstance(aggregate, Mapping), "analytic report aggregate must be an object")
    _require(aggregate.get("cell_count") == 24, "analytic report cell_count must be 24")
    _require(aggregate.get("pass_count") == 24, "analytic report pass_count must be 24")
    expected_sources = {
        "terrain_sha256": "src/isaac_walk_g009/terrain.py",
        "support_plane_sha256": "src/isaac_walk_g009/support_plane.py",
        "validator_sha256": "scripts/validate_g009_s0.py",
    }
    bindings = report.get("source_bindings")
    _require(isinstance(bindings, Mapping), "analytic report source_bindings must be an object")
    for field, relative in expected_sources.items():
        source = REPO_ROOT / relative
        _require(source.is_file(), f"analytic source not found: {relative}")
        _require(bindings.get(field) == file_sha256(source), f"analytic source hash mismatch: {relative}")
    validator_id = report.get("validator_id")
    _require(isinstance(validator_id, str) and validator_id.startswith("g009_s0"), "analytic validator_id is invalid")
    return report, file_sha256(path)


def validate_capture_reports(
    capture_paths: Sequence[Path], config_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    """Validate all source bindings before FFmpeg is allowed to run."""
    _require(len(capture_paths) == 3, "exactly three capture reports are required")
    config = _read_json(config_path)
    config_portable = _portable_repo_path(config_path)
    _require(config_portable == CONFIG_PATH, f"config must be {CONFIG_PATH}")
    expected_profiles = config["visual_protocol"]["profiles"]
    _require(
        tuple(item.get("profile_id") for item in expected_profiles) == EXPECTED_PROFILE_IDS,
        "config visual profile order is not the S0 contract",
    )
    expected_sequence = _normalise_sequence(config["visual_protocol"]["sequence"])
    expected_seed = config["visual_protocol"]["seed"]
    expected_checkpoint = config["parent_checkpoint"]
    config_digest = file_sha256(config_path)

    captures: list[dict[str, Any]] = []
    sources: list[Path] = []
    for index, raw_path in enumerate(capture_paths):
        path = raw_path.resolve()
        _portable_repo_path(path)
        report = _read_json(path)
        prefix = f"capture[{index}]"
        for key, expected in (
            ("schema_version", 1),
            ("goal_id", "g009"),
            ("stage_id", "S0"),
            ("status", "complete"),
        ):
            _require(report.get(key) == expected, f"{prefix}.{key} must be {expected!r}")
        source_commit = report.get("source_commit")
        _require(
            isinstance(source_commit, str) and COMMIT_RE.fullmatch(source_commit) is not None,
            f"{prefix}.source_commit must be 40 lowercase hex",
        )
        _require(report.get("dirty_paths") == [], f"{prefix}.dirty_paths must be empty")
        record_sha = report.get("record_source_sha256")
        _require(
            isinstance(record_sha, str) and SHA256_RE.fullmatch(record_sha) is not None,
            f"{prefix}.record_source_sha256 is invalid",
        )
        _require(
            _git_commit_file_sha256(source_commit, RECORD_SOURCE_PATH) == record_sha,
            f"{prefix}.record_source_sha256 does not match source_commit tree",
        )

        profile = report.get("profile")
        _require(isinstance(profile, Mapping), f"{prefix}.profile must be an object")
        expected_profile = expected_profiles[index]
        _require(profile.get("profile_id") == expected_profile["profile_id"], f"{prefix} profile order mismatch")
        for field in ("slope_deg", "terrain_azimuth_deg"):
            _require(
                profile.get(field) == expected_profile[field],
                f"{prefix}.profile.{field} mismatch",
            )
        _require(profile.get("seed") == expected_seed, f"{prefix}.profile.seed mismatch")
        _require(profile.get("headless") is True, f"{prefix}.profile.headless must be true")
        _require(
            isinstance(profile.get("step_dt_s"), (int, float))
            and not isinstance(profile.get("step_dt_s"), bool)
            and profile["step_dt_s"] > 0,
            f"{prefix}.profile.step_dt_s must be positive",
        )
        _require(
            isinstance(profile.get("total_steps"), int)
            and not isinstance(profile.get("total_steps"), bool)
            and profile["total_steps"] == sum(item["steps"] for item in expected_sequence),
            f"{prefix}.profile.total_steps mismatch",
        )
        _require(
            _normalise_sequence(profile.get("sequence")) == expected_sequence,
            f"{prefix}.profile.sequence mismatch",
        )
        camera = profile.get("camera")
        _require(isinstance(camera, Mapping) and bool(camera), f"{prefix}.profile.camera is required")

        config_binding = report.get("config")
        checkpoint_binding = report.get("checkpoint")
        _require(isinstance(config_binding, Mapping), f"{prefix}.config must be an object")
        _require(isinstance(checkpoint_binding, Mapping), f"{prefix}.checkpoint must be an object")
        _require(config_binding.get("path") == CONFIG_PATH, f"{prefix}.config.path mismatch")
        _require(config_binding.get("sha256") == config_digest, f"{prefix}.config.sha256 mismatch")
        _require(
            _git_commit_file_sha256(source_commit, CONFIG_PATH) == config_binding.get("sha256"),
            f"{prefix}.config.sha256 does not match source_commit tree",
        )
        _require(checkpoint_binding.get("path") == expected_checkpoint["path"], f"{prefix}.checkpoint.path mismatch")
        _require(checkpoint_binding.get("sha256") == expected_checkpoint["sha256"], f"{prefix}.checkpoint.sha256 mismatch")
        _validate_file_binding(config_binding, name=f"{prefix}.config", expected_path=CONFIG_PATH)
        _validate_file_binding(
            checkpoint_binding,
            name=f"{prefix}.checkpoint",
            expected_path=expected_checkpoint["path"],
        )

        physics = report.get("physics_readback")
        _require(isinstance(physics, Mapping) and bool(physics), f"{prefix}.physics_readback is required")
        _require(physics.get("slope_deg") == expected_profile["slope_deg"], f"{prefix} physics slope mismatch")
        _require(
            physics.get("terrain_azimuth_deg") == expected_profile["terrain_azimuth_deg"],
            f"{prefix} physics azimuth mismatch",
        )
        material = physics.get("ground_material")
        _require(isinstance(material, Mapping), f"{prefix}.physics_readback.ground_material is required")
        for field, expected in config["terrain"]["ground_material"].items():
            _require(material.get(field) == expected, f"{prefix} ground_material.{field} mismatch")
        _require(isinstance(report.get("metrics"), Mapping), f"{prefix}.metrics must be an object")

        local_video = report.get("local_video")
        _require(isinstance(local_video, Mapping), f"{prefix}.local_video must be an object")
        local_path = local_video.get("path")
        _require(isinstance(local_path, str) and _is_local_s0_mp4(local_path), f"{prefix}.local_video.path is invalid")
        _require(local_video.get("git_policy") == "local_only", f"{prefix}.local_video must be local_only")
        source = _validate_file_binding(local_video, name=f"{prefix}.local_video")
        _require(local_video.get("bytes") == source.stat().st_size, f"{prefix}.local_video.bytes mismatch")

        captures.append(report)
        sources.append(source)

    equality_fields = {
        "source_commit": [capture["source_commit"] for capture in captures],
        "record_source_sha256": [capture["record_source_sha256"] for capture in captures],
        "camera": [capture["profile"]["camera"] for capture in captures],
        "step_dt_s": [capture["profile"]["step_dt_s"] for capture in captures],
        "total_steps": [capture["profile"]["total_steps"] for capture in captures],
        "sequence": [capture["profile"]["sequence"] for capture in captures],
    }
    for field, values in equality_fields.items():
        canonical = {canonical_json_sha256(value) for value in values}
        _require(len(canonical) == 1, f"capture {field} values do not match")
    _require(len(set(sources)) == 3, "capture local MP4 paths must be distinct")
    source_commit = captures[0]["source_commit"]
    _require(
        _git_commit_file_sha256(source_commit, BUILDER_SOURCE_PATH) == file_sha256(Path(__file__)),
        "builder source does not match capture source_commit tree",
    )
    return config, captures, sources


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout


def _ffprobe(path: Path, executable: str) -> dict[str, Any]:
    output = _run(
        [
            executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,r_frame_rate,pix_fmt",
            "-of",
            "json",
            str(path),
        ]
    )
    value = json.loads(output)
    _require(isinstance(value, dict) and "format" in value, f"ffprobe returned invalid metadata: {path}")
    return value


def _escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _build_composite(sources: Sequence[Path], destination: Path, ffmpeg: str) -> None:
    labels = ("5 deg SLOPE", "15 deg SLOPE", "25 deg STRESS")
    filters: list[str] = []
    for index, label in enumerate(labels):
        filters.append(
            f"[{index}:v]scale=480:360:force_original_aspect_ratio=decrease,"
            "pad=480:360:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
            "drawbox=x=0:y=0:w=iw:h=44:color=black@0.70:t=fill,"
            f"drawtext=text='{_escape_drawtext(label)}':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=10[p{index}]"
        )
    filters.extend(
        [
            "[p0][p1][p2]hstack=inputs=3[panels]",
            "[panels]pad=1440:430:0:35:black,"
            "drawtext=text='G009 S0 | matched camera, seed, commands and checkpoint':fontsize=25:fontcolor=white:x=(w-text_w)/2:y=6,"
            "drawtext=text='QUALITATIVE PLAYBACK ONLY - WALK success requires linked quantitative gates':fontsize=20:fontcolor=yellow:x=(w-text_w)/2:y=397[out]",
        ]
    )
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    for source in sources:
        command.extend(["-i", str(source)])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    _run(command)


def _build_gif(source: Path, destination: Path, ffmpeg: str) -> None:
    graph = (
        "fps=4,scale=960:-2:flags=lanczos,split[a][b];"
        "[b]palettegen=max_colors=64:stats_mode=diff[p];"
        "[a][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    _run([ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(source), "-filter_complex", graph, "-loop", "0", str(destination)])


def _build_contact_sheet(source: Path, destination: Path, ffmpeg: str) -> None:
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-ss",
            "2",
            "-i",
            str(source),
            "-ss",
            "7",
            "-i",
            str(source),
            "-filter_complex",
            "[0:v]scale=960:-2[a];[1:v]scale=960:-2[b];[a][b]vstack=inputs=2[out]",
            "-map",
            "[out]",
            "-frames:v",
            "1",
            str(destination),
        ]
    )


def _validate_media(path: Path, kind: str) -> None:
    _require(path.is_file() and path.stat().st_size > 0, f"missing or empty {kind}: {path}")
    header = path.read_bytes()[:8]
    if kind == "gif":
        _require(header.startswith(GIF_SIGNATURES), "invalid GIF signature")
    elif kind == "png":
        _require(header.startswith(PNG_SIGNATURE), "invalid PNG signature")
    if kind in {"gif", "png"}:
        _require(path.stat().st_size <= MAX_PUBLIC_MEDIA_BYTES, f"{kind} exceeds 10 MiB")


def _artifact(path: Path, portable: str, evidence_type: str, policy: str, ffprobe: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evidence_type": evidence_type,
        "path": portable,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "git_policy": policy,
    }
    if evidence_type in {"local_mp4", "public_gif", "public_png"}:
        result["ffprobe"] = _ffprobe(path, ffprobe)
    return result


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _publish_transaction(
    staged_to_final: Sequence[tuple[Path, Path]], validate_published: Callable[[], None]
) -> None:
    """Publish all outputs as one rollback-capable transaction."""
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for index, (staged, final) in enumerate(staged_to_final):
            _require(staged.is_file(), f"staged output missing: {staged}")
            final.parent.mkdir(parents=True, exist_ok=True)
            backup = final.with_name(f".{final.name}.{os.getpid()}.{index}.bak")
            backup.unlink(missing_ok=True)
            if final.exists():
                os.replace(final, backup)
                backups.append((final, backup))
            os.replace(staged, final)
            published.append(final)
        validate_published()
    except Exception:
        for final in reversed(published):
            final.unlink(missing_ok=True)
        for final, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, final)
        raise
    else:
        for _, backup in backups:
            backup.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    capture_paths = [path.resolve() for path in args.capture_reports]
    config, captures, sources = validate_capture_reports(capture_paths, config_path)
    quantitative_path = REPO_ROOT / QUANTITATIVE_PATH
    quantitative_report, quantitative_sha = _validate_analytic_report(quantitative_path)

    final_paths = {
        "local_mp4": _resolve_portable(LOCAL_MP4_PATH),
        "public_gif": REPO_ROOT / PUBLIC_GIF_PATH,
        "public_png": REPO_ROOT / PUBLIC_PNG_PATH,
        "summary": REPO_ROOT / SUMMARY_PATH,
        "sidecar": REPO_ROOT / SIDECAR_PATH,
    }
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not args.rebuild_existing:
        raise FileExistsError("refusing to overwrite existing outputs: " + ", ".join(map(str, existing)))
    for path in final_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="g009_s0_media_", dir=str(final_paths["local_mp4"].parent)) as temp_name:
        temp_root = Path(temp_name)
        staged_mp4 = temp_root / "composite.mp4"
        staged_gif = temp_root / "slopes.gif"
        staged_png = temp_root / "contact_sheet.png"
        staged_summary = temp_root / "visual_summary.json"
        staged_sidecar = temp_root / "visual_evidence.json"
        _build_composite(sources, staged_mp4, args.ffmpeg)
        _build_gif(staged_mp4, staged_gif, args.ffmpeg)
        _build_contact_sheet(staged_mp4, staged_png, args.ffmpeg)
        _validate_media(staged_mp4, "mp4")
        _validate_media(staged_gif, "gif")
        _validate_media(staged_png, "png")
        probes = {
            "source": [_ffprobe(path, args.ffprobe) for path in sources],
            "local_composite": _ffprobe(staged_mp4, args.ffprobe),
            "public_gif": _ffprobe(staged_gif, args.ffprobe),
            "public_png": _ffprobe(staged_png, args.ffprobe),
        }

        physics_readback = {
            "profiles": [
                {
                    "profile_id": capture["profile"]["profile_id"],
                    "readback": capture["physics_readback"],
                }
                for capture in captures
            ]
        }
        physics_sha = canonical_json_sha256(physics_readback)
        checkpoint = captures[0]["checkpoint"]
        source_commit = captures[0]["source_commit"]
        config_sha = file_sha256(config_path)
        media_binding = {
            "goal_id": "g009",
            "stage_id": "S0",
            "report_id": REPORT_ID,
            "source_commit": source_commit,
            "seed": config["visual_protocol"]["seed"],
            "checkpoint_sha256": checkpoint["sha256"],
            "config_sha256": config_sha,
            "physics_readback_sha256": physics_sha,
            "quantitative_report_sha256": quantitative_sha,
        }
        summary = {
            "schema_version": 1,
            "goal_id": "g009",
            "stage_id": "S0",
            "report_id": REPORT_ID,
            "status": "complete",
            "scope": {
                "claim": "S0 slope wiring and qualitative playback evidence",
                "not_claimed": "WALK success or policy qualification",
                "decision_source": "linked quantitative gates, not video-only judgement",
            },
            "media_binding": media_binding,
            "profile_order": list(EXPECTED_PROFILE_IDS),
            "matched_controls": {
                "source_commit": True,
                "clean_capture_worktree": True,
                "seed": True,
                "config": True,
                "checkpoint": True,
                "sequence": True,
                "camera": True,
                "step_dt_s": True,
            },
            "physics_readback": physics_readback,
            "metrics": [
                {"profile_id": capture["profile"]["profile_id"], "values": capture["metrics"]}
                for capture in captures
            ],
            "capture_reports": [
                {
                    "path": _portable_repo_path(path),
                    "sha256": file_sha256(path),
                    "record_source_sha256": capture["record_source_sha256"],
                    "local_video_sha256": capture["local_video"]["sha256"],
                    "ffprobe": probe,
                }
                for path, capture, probe in zip(capture_paths, captures, probes["source"], strict=True)
            ],
            "sources": {
                "builder": {"path": _portable_repo_path(Path(__file__)), "sha256": file_sha256(Path(__file__))},
                "config": {"path": CONFIG_PATH, "sha256": config_sha},
                "checkpoint": checkpoint,
                "quantitative_report": {"path": QUANTITATIVE_PATH, "sha256": quantitative_sha},
            },
            "composition": {
                "layout": "three synchronized horizontal panels in config order",
                "camera_contract": config["visual_protocol"]["camera_contract"],
                "ffmpeg_version": _run([args.ffmpeg, "-version"]).splitlines()[0],
                "ffprobe": probes,
            },
        }
        _write_json(staged_summary, summary)
        summary_sha = file_sha256(staged_summary)
        artifacts = [
            _artifact(staged_mp4, LOCAL_MP4_PATH, "local_mp4", "local_only", args.ffprobe),
            _artifact(staged_gif, PUBLIC_GIF_PATH, "public_gif", "git_public", args.ffprobe),
            _artifact(staged_png, PUBLIC_PNG_PATH, "public_png", "git_public", args.ffprobe),
            {
                "evidence_type": "quantitative_report",
                "path": QUANTITATIVE_PATH,
                "sha256": quantitative_sha,
                "bytes": quantitative_path.stat().st_size,
                "git_policy": "git_public",
            },
            {
                "evidence_type": "visual_summary",
                "path": SUMMARY_PATH,
                "sha256": summary_sha,
                "bytes": staged_summary.stat().st_size,
                "git_policy": "git_public",
            },
        ]
        sidecar = {
            "schema_version": 1,
            "goal_id": "g009",
            "stage_id": "S0",
            "status": "complete",
            "bindings": {
                "source_commit": source_commit,
                "seed": config["visual_protocol"]["seed"],
                "report_id": quantitative_report["validator_id"],
                "checkpoint": checkpoint,
                "config": {"path": CONFIG_PATH, "sha256": config_sha},
                "quantitative_report": {"path": QUANTITATIVE_PATH, "sha256": quantitative_sha},
                "visual_summary": {"path": SUMMARY_PATH, "sha256": summary_sha},
                "physics_readback_sha256": physics_sha,
            },
            "physics_readback": physics_readback,
            "scope": summary["scope"],
            "artifacts": artifacts,
            "capture_report_sha256": [item["sha256"] for item in summary["capture_reports"]],
            "builder_source_sha256": summary["sources"]["builder"]["sha256"],
        }
        metadata_errors = validate_sidecar(sidecar, REPO_ROOT, check_files=False)
        _require(not metadata_errors, "staged sidecar metadata failed validation: " + "; ".join(metadata_errors))
        _write_json(staged_sidecar, sidecar)
        sidecar_sha = file_sha256(staged_sidecar)

        def validate_published() -> None:
            errors = validate_sidecar(sidecar, REPO_ROOT, check_files=True)
            _require(not errors, "published sidecar failed validation: " + "; ".join(errors))
            _require(file_sha256(final_paths["sidecar"]) == sidecar_sha, "sidecar publish hash mismatch")

        _publish_transaction(
            (
                (staged_mp4, final_paths["local_mp4"]),
                (staged_gif, final_paths["public_gif"]),
                (staged_png, final_paths["public_png"]),
                (staged_summary, final_paths["summary"]),
                (staged_sidecar, final_paths["sidecar"]),
            ),
            validate_published,
        )
        return sidecar


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-reports", required=True, nargs=3, type=Path, metavar=("SLOPE_05", "SLOPE_15", "SLOPE_25_STRESS"))
    parser.add_argument("--config", type=Path, default=REPO_ROOT / CONFIG_PATH)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--rebuild-existing", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        sidecar = build(args)
    except (FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "sidecar": SIDECAR_PATH, "artifacts": len(sidecar["artifacts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
