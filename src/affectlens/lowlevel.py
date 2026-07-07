"""Low-level physical feature extraction from a clip's video and audio.

These are the "luminance / color / loudness" family of regressors: a fast,
interpretable baseline that is known to explain real variance in perceptual and
early sensory responses. Each extractor returns a tidy DataFrame with a ``t``
(seconds) column plus one column per feature, sampled on its own clock;
``align.py`` bins these onto the rating grid.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

from .config import ExtractionConfig


# --------------------------------------------------------------------------- #
# Visual
# --------------------------------------------------------------------------- #
def _colorfulness(bgr: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colorfulness metric."""
    b, g, r = bgr[..., 0].astype(np.float32), bgr[..., 1].astype(np.float32), bgr[..., 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


def extract_visual(path: str | Path, config: ExtractionConfig | None = None) -> pd.DataFrame:
    """Per-sampled-frame visual features.

    Columns: t, luminance, contrast, colorfulness, saturation, edge_density,
    motion (mean absolute inter-frame difference of luminance).
    """
    config = config or ExtractionConfig()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    sample_fps = config.visual_sample_fps or native_fps or 8.0
    frame_step = max(1, int(round(native_fps / sample_fps))) if native_fps else 1

    rows: list[dict] = []
    prev_gray: np.ndarray | None = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step != 0:
            idx += 1
            continue
        t = idx / native_fps if native_fps else idx / sample_fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(frame, 100, 200)
        motion = (
            float(np.mean(np.abs(gray - prev_gray))) if prev_gray is not None else 0.0
        )
        rows.append(
            {
                "t": t,
                "luminance": float(gray.mean()),
                "contrast": float(gray.std()),
                "colorfulness": _colorfulness(frame),
                "saturation": float(hsv[..., 1].mean()) / 255.0,
                "edge_density": float((edges > 0).mean()),
                "motion": motion,
            }
        )
        prev_gray = gray
        idx += 1
    cap.release()

    if not rows:
        raise RuntimeError(f"no frames decoded from {path}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def _decode_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Decode a clip's audio to mono float32 in [-1, 1] via bundled ffmpeg.

    Returns an empty array when the clip has no audio track.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    import io

    with wave.open(io.BytesIO(proc.stdout), "rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
        width = wf.getsampwidth()
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    return data


def extract_audio(path: str | Path, config: ExtractionConfig | None = None) -> pd.DataFrame:
    """Framewise audio features.

    Columns: t, rms (loudness proxy), zcr (zero-crossing rate), spectral_centroid,
    spectral_flux. Returns an empty frame (with columns) when there is no audio.
    """
    config = config or ExtractionConfig()
    sr = config.audio_sample_rate
    y = _decode_audio(path, sr)
    cols = ["t", "rms", "zcr", "spectral_centroid", "spectral_flux"]
    if y.size == 0:
        return pd.DataFrame(columns=cols)

    frame = max(1, int(config.audio_frame_s * sr))
    hop = max(1, int(config.audio_hop_s * sr))
    window = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, d=1.0 / sr)

    rows: list[dict] = []
    prev_mag: np.ndarray | None = None
    for start in range(0, max(1, len(y) - frame + 1), hop):
        seg = y[start : start + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        win = seg * window
        rms = float(np.sqrt(np.mean(seg**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(seg))) > 0))
        mag = np.abs(np.fft.rfft(win))
        centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-9))
        flux = float(np.sqrt(np.sum((mag - prev_mag) ** 2))) if prev_mag is not None else 0.0
        rows.append(
            {
                "t": start / sr,
                "rms": rms,
                "zcr": zcr,
                "spectral_centroid": centroid,
                "spectral_flux": flux,
            }
        )
        prev_mag = mag
    return pd.DataFrame(rows, columns=cols)


def extract_lowlevel(path: str | Path, config: ExtractionConfig | None = None) -> dict[str, pd.DataFrame]:
    """Extract all enabled low-level families. Keys: 'visual', 'audio'.

    Only families whose stream is present are computed, so an audio-only clip
    (e.g. a music track) yields just 'audio', and a silent video yields just
    'visual'.
    """
    from . import clips as clips_mod

    config = config or ExtractionConfig()
    info = clips_mod.probe_clip(path)
    out: dict[str, pd.DataFrame] = {}
    if config.visual and info.has_video:
        out["visual"] = extract_visual(path, config)
    if config.audio and info.has_audio:
        out["audio"] = extract_audio(path, config)
    return out
