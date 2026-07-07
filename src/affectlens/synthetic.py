"""Generate a synthetic clips + ratings dataset to exercise the pipeline offline.

We build short videos whose low-level properties (brightness, motion, audio
loudness) vary over time on a known schedule, plus a matching transcript. We then
synthesize continuous ratings as noisy functions of those same latent drivers, so
a working pipeline should recover a positive correlation. This lets the whole
flow be run and tested with no external data and no model downloads.
"""

from __future__ import annotations

import contextlib
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def _write_video(path: Path, brightness: np.ndarray, motion: np.ndarray, fps: int, size: int = 64):
    """Write a silent mp4 whose per-frame luminance and motion follow the inputs."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    n = len(brightness)
    proc = subprocess.Popen(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}",
            "-r", str(fps), "-i", "pipe:0",
            "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    rng = np.random.default_rng(0)
    assert proc.stdin is not None
    for i in range(n):
        base = np.full((size, size, 3), int(np.clip(brightness[i], 0, 1) * 255), dtype=np.uint8)
        # Inject spatial noise proportional to the motion driver so frame-to-frame
        # differences track it.
        noise = (rng.standard_normal((size, size, 3)) * motion[i] * 120).astype(np.int16)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def _write_wav(path: Path, loudness: np.ndarray, dur_s: float, sr: int = 16000):
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    # Amplitude envelope interpolated from the per-bin loudness driver.
    env = np.interp(t, np.linspace(0, dur_s, len(loudness)), loudness)
    tone = np.sin(2 * np.pi * 220 * t) * env
    data = np.clip(tone, -1, 1)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _mux(video: Path, audio: Path, out: Path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-i", str(audio),
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
    )


def _make_clip(clips_dir: Path, name: str, seed: int, dur_s: float = 20.0, fps: int = 12):
    rng = np.random.default_rng(seed)
    n_frames = int(dur_s * fps)
    tt = np.linspace(0, 1, n_frames)
    # Latent drivers on the frame clock.
    brightness = 0.5 + 0.35 * np.sin(2 * np.pi * (1 + seed % 3) * tt + seed)
    motion = np.abs(0.3 + 0.3 * np.sin(2 * np.pi * (2 + seed % 2) * tt))
    # A mid-clip motion spike -- the "surprise" case that should survive binning.
    spike = int(n_frames * 0.6)
    motion[spike : spike + fps] += 0.8

    tmp = clips_dir / f".{name}"
    tmp.mkdir(exist_ok=True)
    vid = tmp / "v.mp4"
    aud = tmp / "a.wav"
    _write_video(vid, brightness, motion, fps)
    # Audio loudness driver sampled coarsely (per ~1s).
    loud = np.abs(0.2 + 0.6 * np.sin(2 * np.pi * (1.5 + seed % 2) * np.linspace(0, 1, int(dur_s))))
    _write_wav(aud, loud, dur_s)
    out = clips_dir / f"{name}.mp4"
    _mux(vid, aud, out)

    # Sidecar transcript so the semantic path has something to embed.
    words = ["happy", "tense", "calm", "sudden", "bright", "quiet", "loud", "still"]
    lines = []
    for k in range(int(dur_s // 4)):
        s, e = k * 4, k * 4 + 4
        w = " ".join(rng.choice(words, size=3))
        lines.append(f"{k+1}\n00:00:{s:02d},000 --> 00:00:{e:02d},000\n{w}\n")
    out.with_suffix(".srt").write_text("\n".join(lines), encoding="utf-8")

    return brightness, motion, loud, dur_s


def _make_ratings(csv_path: Path, drivers: dict, interval: float = 4.0):
    """Synthesize per-participant ratings as noisy functions of the drivers."""
    rows = []
    rng = np.random.default_rng(123)
    for clip, (brightness, motion, loud, dur_s) in drivers.items():
        times = np.arange(0, dur_s - interval + 1e-6, interval)
        for t in times:
            # Average each driver over the bin.
            fb = np.linspace(0, dur_s, len(brightness))
            fm = np.linspace(0, dur_s, len(motion))
            fl = np.linspace(0, dur_s, len(loud))
            b = np.interp([t + interval / 2], fb, brightness)[0]
            m = np.interp([t + interval / 2], fm, motion)[0]
            l = np.interp([t + interval / 2], fl, loud)[0]
            for p in range(4):
                rows.append(
                    {
                        "clip": clip,
                        "participant": f"p{p+1}",
                        "time": float(t),
                        # Rated dimensions are latent-driver-driven + rater noise.
                        "brightness": float(b + rng.normal(0, 0.05)),
                        "energy": float(0.6 * m + 0.4 * l + rng.normal(0, 0.05)),
                        "arousal": float(0.5 * l + 0.5 * m + rng.normal(0, 0.08)),
                    }
                )
    import csv

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


@contextlib.contextmanager
def build_synthetic_dataset(n_clips: int = 3):
    """Context manager yielding (clips_dir, ratings_csv_path) in a temp dir."""
    tmp = Path(tempfile.mkdtemp(prefix="affectlens_synth_"))
    clips_dir = tmp / "clips"
    clips_dir.mkdir()
    drivers = {}
    for i in range(n_clips):
        name = f"clip_{i+1:02d}"
        drivers[name] = _make_clip(clips_dir, name, seed=i + 1)
    ratings_csv = tmp / "ratings.csv"
    _make_ratings(ratings_csv, drivers)
    try:
        yield clips_dir, ratings_csv
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
