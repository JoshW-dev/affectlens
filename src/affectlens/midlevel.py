"""Mid-level features: the band between raw physical stats and dialogue meaning.

The low-level extractors (``lowlevel.py``) report physical signal statistics
(luminance, RMS level, mean frame-difference). The semantic path
(``highlevel.py``) reports what the *dialogue* means. In between sits a rich,
under-used band of **perceptual and social primitives** -- motion energy, pitch,
scene cuts, faces, voices -- that are neither raw pixels/samples nor full
meaning.

That mid band is worth its own tier because each feature tends to map onto a
*named, testable* brain system, which makes a feature-to-signal correlation
interpretable rather than diffuse:

    optical-flow motion   -> MT / V5, MST (looming drives arousal)
    pitch (F0) / voicing  -> Heschl's gyrus, right-STG prosody
    scene / shot cuts      -> event-segmentation (transients, hippocampal/PM)

The three implemented here are all pure numpy / OpenCV (no extra dependency) and
ride the decode passes ``lowlevel.py`` already makes, so they are close to free.
Each helper is small and returns plain floats; ``lowlevel.py`` calls them inside
its per-frame / per-window loops and the values flow through ``align.py`` like
any other feature.

Ideas for more (each is one extractor returning a ``t``-column DataFrame; heavier
ones would ship as optional extras, e.g. ``affectlens[faces]``):

  VISUAL
    - face presence / count / size            -> FFA / OFA / STS   (mediapipe / YuNet)
    - facial-motion dynamism (mouth, blinks)  -> posterior STS      (rides a face mesh)
    - scene / place category (indoor/outdoor) -> PPA / RSC / OPA    (Places365)
    - animacy occupancy (animate vs object)   -> ventral temporal   (a COCO detector)
  AUDIO
    - speech envelope / amplitude modulation  -> STG speech-tracking (numpy Hilbert)
    - voice-activity / speech presence         -> temporal voice areas (webrtcvad)
    - loudness transients (attack/startle)     -> auditory + amygdala
    - tempo / beat / onset density             -> auditory + SMA/basal ganglia
  SEMANTIC / CROSS-MODAL
    - word surprisal (LM -log p)               -> language network / N400 (distilgpt2)
    - topic / narrative-boundary segmentation  -> hippocampus / DMN (rides the Embedder)
    - dialogue sentiment / valence time course -> vmPFC / OFC       (a VAD lexicon)

See the README "Mid-level features" section for the full roadmap.
"""

from __future__ import annotations

import cv2
import numpy as np

# Optical flow is computed on a small copy of the frame -- dense flow does not
# need full resolution for a global motion-energy summary, and shrinking keeps
# it fast.
FLOW_WIDTH = 128


def to_flow_gray(gray_u8: np.ndarray) -> np.ndarray:
    """Downscale an 8-bit grayscale frame to the width optical flow runs at."""
    h, w = gray_u8.shape
    if w <= FLOW_WIDTH:
        return gray_u8
    return cv2.resize(gray_u8, (FLOW_WIDTH, max(1, round(h * FLOW_WIDTH / w))))


def optical_flow_features(prev_small: np.ndarray, cur_small: np.ndarray) -> tuple[float, float]:
    """Dense optical flow between two small grayscale frames.

    Returns ``(magnitude, looming)``:
      - **magnitude** -- mean flow speed, i.e. real motion energy (a truer motion
        signal than the mean absolute frame difference, which also fires on
        lighting changes and cuts).
      - **looming** -- mean radial (outward) flow component about the frame
        centre; positive when the scene expands toward the viewer (approach),
        negative on recede.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_small, cur_small, None,
        pyr_scale=0.5, levels=2, winsize=15, iterations=2,
        poly_n=5, poly_sigma=1.1, flags=0,
    )
    fx, fy = flow[..., 0], flow[..., 1]
    magnitude = float(np.sqrt(fx * fx + fy * fy).mean())

    h, w = prev_small.shape
    ys, xs = np.mgrid[0:h, 0:w]
    rx = xs - (w - 1) / 2.0
    ry = ys - (h - 1) / 2.0
    rnorm = np.sqrt(rx * rx + ry * ry) + 1e-6
    looming = float(((fx * rx + fy * ry) / rnorm).mean())
    return magnitude, looming


def scene_cut_score(prev_gray_u8: np.ndarray, cur_gray_u8: np.ndarray) -> float:
    """Shot-boundary score in [0, 1] from consecutive-frame histogram change.

    ~0 within a continuous shot, spiking toward 1 on a hard cut. Uses a
    luminance histogram (robust to motion/lighting, unlike a pixelwise diff).
    """
    h1 = cv2.calcHist([prev_gray_u8], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([cur_gray_u8], [0], None, [64], [0, 256])
    corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return float(min(1.0, max(0.0, 1.0 - corr)))


def pitch_from_spectrum(
    mag: np.ndarray, sr: int, frame: int, fmin: float = 75.0, fmax: float = 400.0
) -> tuple[float, float]:
    """Fundamental frequency (Hz) and voicing (0-1) from a window's spectrum.

    Reuses the magnitude spectrum ``lowlevel.extract_audio`` already computes:
    the autocorrelation is ``irfft(|X|^2)``, and its strongest peak in the
    human-voice lag range gives the pitch period. ``voicing`` is the normalised
    peak height -- periodicity strength -- so tonal/voiced sound scores high and
    noise/silence scores ~0. ``f0`` is 0 when the frame is not clearly voiced.
    """
    ac = np.fft.irfft(mag.astype(np.float64) ** 2, n=frame)
    ac0 = ac[0] + 1e-9
    lag_min = max(1, int(sr / fmax))
    lag_max = min(frame - 1, int(sr / fmin))
    if lag_max <= lag_min:
        return 0.0, 0.0
    peak = int(np.argmax(ac[lag_min:lag_max])) + lag_min
    voicing = float(min(1.0, max(0.0, ac[peak] / ac0)))
    f0 = float(sr / peak) if voicing > 0.3 else 0.0
    return f0, voicing
