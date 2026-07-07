"""Clip inventory: probe a directory of stimulus videos.

Reports how many clips there are, their durations, resolution, and whether each
has an audio / video track. Uses the ffmpeg binary bundled by ``imageio-ffmpeg``
so no system ffmpeg is required.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio_ffmpeg

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
# Audio-only stimuli (e.g. music tracks) are supported too; they yield audio
# features only.
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def _ffprobe_exe() -> str:
    """Locate an ffprobe binary.

    imageio-ffmpeg ships ffmpeg but not ffprobe. ffmpeg alone can report stream
    info via ``-i`` on stderr, so we parse that rather than requiring ffprobe.
    """
    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class ClipInfo:
    path: str
    name: str
    duration_s: float | None
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool
    has_video: bool
    audio_sample_rate: int | None
    error: str | None = None

    def as_row(self) -> dict:
        return asdict(self)


def _parse_ffmpeg_info(stderr: str) -> dict:
    """Extract duration / stream info from ffmpeg's ``-i`` stderr banner."""
    info: dict = {
        "duration_s": None,
        "width": None,
        "height": None,
        "fps": None,
        "has_audio": False,
        "has_video": False,
        "audio_sample_rate": None,
    }
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            token = line.split("Duration:")[1].split(",")[0].strip()
            if token and token != "N/A":
                h, m, s = token.split(":")
                info["duration_s"] = int(h) * 3600 + int(m) * 60 + float(s)
        elif line.startswith("Stream") and "Video:" in line:
            info["has_video"] = True
            for chunk in line.split(","):
                chunk = chunk.strip()
                if "x" in chunk and chunk.split(" ")[0].replace("x", "").isdigit():
                    dims = chunk.split(" ")[0]
                    try:
                        w, h = dims.split("x")
                        info["width"], info["height"] = int(w), int(h)
                    except ValueError:
                        pass
                if chunk.endswith("fps"):
                    try:
                        info["fps"] = float(chunk.replace("fps", "").strip())
                    except ValueError:
                        pass
        elif line.startswith("Stream") and "Audio:" in line:
            info["has_audio"] = True
            for chunk in line.split(","):
                chunk = chunk.strip()
                if chunk.endswith("Hz"):
                    try:
                        info["audio_sample_rate"] = int(chunk.replace("Hz", "").strip())
                    except ValueError:
                        pass
    return info


def probe_clip(path: str | Path) -> ClipInfo:
    path = Path(path)
    proc = subprocess.run(
        [_ffprobe_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    # ffmpeg exits non-zero when given no output, but still prints the banner.
    info = _parse_ffmpeg_info(proc.stderr)
    error = None
    if not info["has_video"] and not info["has_audio"]:
        error = "no readable audio/video stream"
    return ClipInfo(
        path=str(path),
        name=path.name,
        error=error,
        **info,
    )


def inventory(clips_dir: str | Path) -> list[ClipInfo]:
    """Probe every video file in ``clips_dir`` (non-recursive + recursive)."""
    clips_dir = Path(clips_dir)
    if not clips_dir.exists():
        raise FileNotFoundError(f"clips directory does not exist: {clips_dir}")
    files = sorted(
        p
        for p in clips_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in MEDIA_EXTENSIONS
        # Skip hidden files/dirs (scratch dirs, .venv, editor droppings).
        and not any(part.startswith(".") for part in p.relative_to(clips_dir).parts)
    )
    return [probe_clip(p) for p in files]


def inventory_to_json(clips_dir: str | Path) -> str:
    return json.dumps([c.as_row() for c in inventory(clips_dir)], indent=2)
