"""Mid-level features: the band between raw physical stats and dialogue meaning.

The low-level extractors (``lowlevel.py``) report physical signal statistics
(luminance, RMS level, mean frame-difference). The semantic path
(``highlevel.py``) reports what the *dialogue* means. In between sits a rich,
under-used band of **perceptual and social primitives** -- motion structure,
pitch, scene cuts, spectral texture, colour opponency -- that are neither raw
pixels/samples nor full meaning.

That mid band is worth its own tier because each feature tends to map onto a
*named, testable* brain system, which makes a feature-to-signal correlation
interpretable rather than diffuse:

    optical-flow magnitude   -> MT / V5 motion-energy pooling
    optical-flow looming     -> MSTd radial-flow (approach / time-to-collision)
    optical-flow coherence   -> MT surround-antagonism vs MST / CSv egomotion
    spatial detail (SF)      -> V1 spatial-frequency channels
    chroma opponency         -> cone-opponent L-M / S axes (V1 -> V4 / VO glob cells)
    scene / shot cuts        -> hippocampal / posterior-medial event segmentation
    pitch (F0) / voicing     -> pitch region at the anterolateral Heschl's border
    loudness attack          -> brainstem acoustic-startle arc (PnC), amygdala-gated
    spectral flatness        -> non-primary auditory harmonicity (tone vs noise)

Two of these mappings are deliberately loose, and the README says so in the
open: "warm vs cool" is a perceptual grouping imposed on the cone-opponent axes,
not a canonical neural dimension; and spectral flatness has no dedicated cortical
region, standing in as an indirect proxy for harmonicity. Citations for every
mapping live in the README's References section.

Everything here is pure numpy / OpenCV (no extra dependency) and rides the
decode passes ``lowlevel.py`` already makes, so the whole tier is close to free:
each helper is small, takes something the decode loop already has in hand (a
grayscale frame, a BGR frame, an rfft magnitude spectrum, a flow field), and
returns plain floats that flow through ``align.py`` like any other feature.

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
    - tempo / beat / onset density             -> auditory + SMA / basal ganglia
  SEMANTIC / CROSS-MODAL
    - word surprisal (LM -log p)               -> language network / N400 (distilgpt2)
    - topic / narrative-boundary segmentation  -> hippocampus / DMN (rides the Embedder)
    - dialogue sentiment / valence time course -> vmPFC / OFC       (a VAD lexicon)

See the README "Mid-level features" section for the full roadmap and references.
"""

from __future__ import annotations

import cv2
import numpy as np

# Optical flow is computed on a small copy of the frame -- dense flow does not
# need full resolution for a global motion-energy summary, and shrinking keeps
# it fast.
FLOW_WIDTH = 128


# --------------------------------------------------------------------------- #
# Visual
# --------------------------------------------------------------------------- #
def to_flow_gray(gray_u8: np.ndarray) -> np.ndarray:
    """Downscale an 8-bit grayscale frame to the width optical flow runs at."""
    h, w = gray_u8.shape
    if w <= FLOW_WIDTH:
        return gray_u8
    return cv2.resize(gray_u8, (FLOW_WIDTH, max(1, round(h * FLOW_WIDTH / w))))


def optical_flow_features(
    prev_small: np.ndarray, cur_small: np.ndarray
) -> tuple[float, float, float]:
    """Dense optical flow between two small grayscale frames.

    Returns ``(magnitude, looming, coherence)`` -- three orthogonal summaries of
    the same Farneback flow field, a first-order decomposition of image motion:

      - **magnitude** -- mean flow speed, i.e. real motion energy (a truer motion
        signal than the mean absolute frame difference, which also fires on
        lighting changes and cuts).
      - **looming** -- mean radial (outward) flow about the frame centre;
        positive when the scene expands toward the viewer (approach), negative
        on recede.
      - **coherence** -- ``|mean vector| / mean speed`` in [0, 1]: ~1 when the
        whole field moves together (a camera pan, i.e. self-motion), ~0 when
        motion is scattered across independently moving parts (object motion, a
        crowd). Separates self-motion from object motion at equal speed.
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

    # A coherent field has |mean vector| ~ mean speed; scattered motion cancels
    # in the mean vector. Bounded to [0, 1] (|E[v]| <= E[|v|] by Jensen).
    mean_vec = float(np.hypot(fx.mean(), fy.mean()))
    coherence = float(min(1.0, mean_vec / (magnitude + 1e-6)))
    return magnitude, looming, coherence


def spatial_detail(gray: np.ndarray) -> float:
    """High-spatial-frequency energy: the variance of the Laplacian.

    The standard focus / acutance measure -- high for crisp, finely-textured
    frames, collapsing under defocus, fog, or motion blur. Expects ``gray`` in
    [0, 1] so the value is comparable across clips. Unlike ``edge_density`` (a
    thresholded Canny *count* that saturates on busy scenes) or ``contrast``
    (global luminance spread, dominated by coarse layout), this is a continuous
    fine-scale energy.
    """
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def chroma_opponency(bgr: np.ndarray) -> tuple[float, float]:
    """Signed position on the two cardinal chromatic axes, each roughly [-1, 1].

    Returns ``(chroma_rg, chroma_by)`` -- red-vs-green and blue-vs-yellow. This
    is the *sign* the (unsigned) colorfulness metric throws away: saturation and
    colorfulness say how vivid a frame is, this says which way it leans. A warm
    red scene and a cool blue scene of equal vividness look identical to
    saturation but opposite here (``chroma_by`` > 0 warm, < 0 cool).
    """
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    chroma_rg = float((r - g).mean()) / 255.0
    chroma_by = float((0.5 * (r + g) - b).mean()) / 255.0
    return chroma_rg, chroma_by


def scene_cut_score(prev_gray_u8: np.ndarray, cur_gray_u8: np.ndarray) -> float:
    """Shot-boundary score in [0, 1] from consecutive-frame histogram change.

    ~0 within a continuous shot, spiking toward 1 on a hard cut. Uses a
    luminance histogram (robust to motion/lighting, unlike a pixelwise diff).
    """
    h1 = cv2.calcHist([prev_gray_u8], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([cur_gray_u8], [0], None, [64], [0, 256])
    corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return float(min(1.0, max(0.0, 1.0 - corr)))


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
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


def spectral_flatness(mag: np.ndarray) -> float:
    """Wiener entropy (tonality) in [0, 1] from a window's magnitude spectrum.

    Geometric mean over arithmetic mean of the power spectrum: ~0 for tonal /
    harmonic sound (a sung note, a whistle), toward 1 for noise-like sound
    (fricatives, applause, static). An auditory *texture* measure with no
    pitch-range assumption -- complementary to ``voicing`` (band-limited
    periodicity) and ``spectral_centroid`` (energy location, not shape).
    """
    p = mag.astype(np.float64) ** 2
    if p.sum() < 1e-12:  # silence: guard against the misleading eps/eps -> 1
        return 0.0
    gm = np.exp(np.mean(np.log(p + 1e-12)))
    am = p.mean() + 1e-12
    return float(min(1.0, gm / am))


def loudness_attack(rms: float, prev_rms: float | None) -> float:
    """Half-wave-rectified positive rise in loudness (dB) between windows.

    Fires only on *rising* intensity -- a hit, a slam, a shout onset -- and is
    silent on decays, the onset/offset asymmetry the acoustic-startle reflex
    shows. Works in a log (dB-like) domain because startle scales with loudness
    *rise*, not raw amplitude difference. Distinct from ``spectral_flux``, which
    is symmetric and fires on any spectral change (including a constant-loudness
    timbre shift).
    """
    if prev_rms is None:
        return 0.0
    level = 20.0 * np.log10(rms + 1e-6)
    prev_level = 20.0 * np.log10(prev_rms + 1e-6)
    return float(max(0.0, level - prev_level))
