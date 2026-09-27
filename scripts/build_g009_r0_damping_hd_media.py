"""Create measured 15fps diagnostic GIF/PNG from the local 1080p capture."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from record_g009_r0_diagnostic import file_sha256, portable_path, resolve_portable_path
from g009_r0_rev32 import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))
from isaac_walk_g009.media_contract import inspect_gif_encoding, validate_gif_encoding_metadata


def build(capture_path: Path):
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    if capture.get("diagnostic_only") is not True or capture.get("qualification_eligible") is not False:
        raise ValueError("diagnostic capture required")
    video = resolve_portable_path(capture["local_video"]["path"])
    if file_sha256(video) != capture["local_video"]["sha256"]:
        raise ValueError("local video hash mismatch")
    training_path = resolve_portable_path(capture["training_report"]["path"])
    if file_sha256(training_path) != capture["training_report"]["sha256"]:
        raise ValueError("quantitative training report hash mismatch")
    # Include the verified training-time gain evidence, even for the original
    # capture made before the sidecar field was added to the recorder.
    intervention_path = training_path.with_name(training_path.stem + "_intervention.json")
    intervention = json.loads(intervention_path.read_text(encoding="utf-8"))
    if intervention.get("actuator_readback_stable") is not True:
        raise ValueError("training gain readback was not stable")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(video),
    ], text=True))
    stream = probe["streams"][0]
    if (stream["width"], stream["height"], stream["avg_frame_rate"]) != (1920, 1080, "30/1"):
        raise ValueError("native1080p30fps required")
    if int(stream["nb_read_frames"]) != len(capture["recording"]["captured_control_steps"]):
        raise ValueError("capture frame count mismatch")
    directory = REPO_ROOT / "docs/media/g009/R0/diagnostic"
    directory.mkdir(parents=True, exist_ok=True)
    gif = directory / (capture_path.stem + ".gif")
    png = directory / (capture_path.stem + "_still.png")
    sidecar = REPO_ROOT / "reports/runs" / (capture_path.stem + "_media.json")
    if any(path.exists() for path in (gif, png, sidecar)):
        raise FileExistsError(capture_path.stem)
    duration = float(probe["format"]["duration"])
    # Try the full capture first; only then trim, resize, and reduce palette.
    attempts = [(duration, 1920, 256, []), (min(duration, 6), 1920, 256, ["trim_duration"]),
                (min(duration, 4), 1920, 256, ["trim_duration"]),
                (min(duration, 4), 960, 256, ["trim_duration", "reduce_resolution"]),
                (min(duration, 4), 960, 128, ["trim_duration", "reduce_resolution", "reduce_palette"])]
    for length, width, colors, steps in attempts:
        graph = (f"fps=15,scale={width}:-2:flags=lanczos,split[a][b];"
                 f"[a]palettegen=max_colors={colors}[p];[b][p]paletteuse=dither=bayer:bayer_scale=3")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                        "-t", str(length), "-filter_complex", graph, "-loop", "0", str(gif)], check=True)
        if gif.stat().st_size < 10 * 1024 * 1024:
            break
    metadata = inspect_gif_encoding(gif)
    metadata.update(source_video_fps=30, target_gif_fps=15, media_kind="camera",
                    temporal_strategy="source_frame_sampling", palette_colors=colors,
                    compression_policy_order=["trim_duration", "reduce_resolution", "reduce_palette"],
                    compression_steps_applied=steps)
    errors = validate_gif_encoding_metadata(metadata)
    if errors:
        raise ValueError(errors)
    # Last available state, not the automatic-reset frame.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", str(video),
                    "-vf", f"select=eq(n\\,{int(stream['nb_read_frames']) - 1})", "-frames:v", "1", str(png)], check=True)
    if png.stat().st_size >= 10 * 1024 * 1024:
        raise ValueError("PNG exceeds10MiB")
    result = {
        "schema_version": "g009.r0.damping.hd_media.v1", "status": "diagnostic_media_complete",
        "diagnostic_only": True, "qualification_eligible": False,
        "capture": {"path": portable_path(capture_path), "sha256": file_sha256(capture_path)},
        "quantitative_report": {"path": portable_path(training_path), "sha256": file_sha256(training_path),
                                "scope": "training safety telemetry, not recovery-rate evaluation"},
        "training_intervention": {"path": portable_path(intervention_path), "sha256": file_sha256(intervention_path),
                                  "protocol": intervention["protocol"]},
        "local_video": capture["local_video"], "source_video_probe": probe,
        "gif": {"path": portable_path(gif), "sha256": file_sha256(gif), "encoding": metadata},
        "png": {"path": portable_path(png), "sha256": file_sha256(png), "bytes": png.stat().st_size},
        "builder_source_sha256": file_sha256(Path(__file__)),
    }
    sidecar.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sidecar": str(sidecar), "gif_bytes": gif.stat().st_size,
                      "actual_gif_fps": metadata["actual_gif_fps"], "gif_duration": metadata["gif_duration_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    build(parser.parse_args().capture)
