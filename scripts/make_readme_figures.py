#!/usr/bin/env python3
"""Regenerate the README figures in docs/images/ from the sample clips.

Prerequisites (from the repo root):

    pip install -e . matplotlib
    python scripts/fetch_samples.py
    affectlens extract --clips examples/samples --out out/

Then:

    python scripts/make_readme_figures.py

Figure 1 (features.png): real frames from Elephants Dream above four
higher-level feature time courses (motion energy, scene cuts, colour warmth,
sound onsets), each labelled with the brain system it probes — the hero.

Figure 2 (encoding.png): the `encode` workflow on a demo signal fabricated
from the clip's own loudness delayed by one bin, showing the lag scan
recovering the delay. Deterministic (fixed RNG seed).

Figure 3 (midlevel.png): the mid-level tier over the same clip — six
perceptual primitives (flow magnitude, flow coherence, scene cuts, colour
warmth, spectral flatness, and pitch F0 shaded by voicing), each labelled
with the brain system it is meant to probe.

Figure 4 (feature_matrix.png): every base feature over the clip as one
z-scored design matrix (a heatmap of 21 features x time), grouped by tier —
the whole feature space `extract` produces, at a glance.

Elephants Dream is (c) 2006 Blender Foundation / Netherlands Media Art
Institute, CC-BY-2.5 — the README credits it alongside the figures.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "examples" / "samples" / "elephants_dream.mp4"
FEATURES = ROOT / "out" / "elephants_dream__features.csv"
OUT_DIR = ROOT / "docs" / "images"

FRAME_TIMES_S = [60, 156, 252, 348, 444, 540]  # inside the film, before the credits
INTERVAL_S = 4.5


def grab_frame(cap: cv2.VideoCapture, t_s: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read frame at {t_s}s")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)


def make_features_figure(X: pd.DataFrame) -> None:
    cap = cv2.VideoCapture(str(CLIP))
    frames = [grab_frame(cap, t) for t in FRAME_TIMES_S]
    cap.release()

    # Higher-level features that visibly track the frames above: real motion
    # energy, shot boundaries, colour warmth, and sound onsets — each labelled
    # with the brain system it is meant to probe (see the mid-level tier).
    curves = [
        ("visual__flow_magnitude_mean", "motion energy\n(MT / V5)", "#c0504d"),
        ("visual__scene_cut_max", "scene cuts\n(hippocampus)", "#4472c4"),
        ("visual__chroma_by_mean", "colour warmth\n(V4 / VO)", "#c9820a"),
        ("audio__loudness_attack_mean", "sound onsets\n(startle)", "#2e8b7f"),
    ]

    fig = plt.figure(figsize=(12, 7.4))
    gs = fig.add_gridspec(
        len(curves) + 1, len(frames),
        height_ratios=[1.9] + [1] * len(curves), hspace=0.35, wspace=0.04,
    )

    for i, (t, frame) in enumerate(zip(FRAME_TIMES_S, frames, strict=False)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frame)
        ax.set_title(f"{t//60}:{t%60:02d}", fontsize=9, color="0.35", pad=3)
        ax.axis("off")

    t_min = X.index.to_numpy() / 60.0
    for row, (col, label, color) in enumerate(curves, start=1):
        ax = fig.add_subplot(gs[row, :])
        ax.plot(t_min, X[col], color=color, lw=1.0)
        ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", color="0.25")
        ax.set_yticks([])
        ax.margins(x=0.005)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("0.8")
        for t in FRAME_TIMES_S:
            ax.axvline(t / 60.0, color="0.75", lw=0.7, ls=":", zorder=0)
        if row < len(curves):
            ax.set_xticks([])
        else:
            ax.set_xlabel("time (minutes)", fontsize=9, color="0.25")
            ax.tick_params(labelsize=8, colors="0.35")

    fig.suptitle(
        "affectlens extract — one 11-minute clip in, aligned feature time courses out",
        fontsize=11, y=0.98,
    )
    fig.savefig(OUT_DIR / "features.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_encoding_figure(X: pd.DataFrame) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from affectlens import encoding

    rng = np.random.default_rng(7)
    drive = zscore(X["audio__rms_mean"]).to_numpy()
    signal = np.roll(drive, 1)  # the "recording" trails the stimulus by 1 bin
    signal[0] = 0.0
    signal = signal + rng.normal(0, 0.25, len(signal))

    lags = [0, 1, 2, 3]
    rs = [encoding.encode_signal(X, signal, lag_bins=lag).r for lag in lags]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 3.2), gridspec_kw={"width_ratios": [2.6, 1]}
    )

    window = slice(20, 80)  # a 4.5-minute stretch, for legibility
    t_min = X.index.to_numpy()[window] / 60.0
    ax1.plot(t_min, signal[window], color="0.3", lw=1.2, label="recorded signal  s(t)")
    ax1.plot(
        t_min, np.roll(drive, 1)[window], color="#4472c4", lw=1.2, ls="--",
        label="clip loudness, shifted 1 bin",
    )
    ax1.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    ax1.set_xlabel("time (minutes)", fontsize=9, color="0.25")
    ax1.set_yticks([])
    ax1.tick_params(labelsize=8, colors="0.35")
    for spine in ("top", "right", "left"):
        ax1.spines[spine].set_visible(False)
    ax1.set_title("the signal trails the stimulus…", fontsize=10, color="0.25", loc="left")

    bars = ax2.bar([str(l) for l in lags], rs, color=["0.8", "#4472c4", "0.8", "0.8"], width=0.6)
    best = int(np.argmax(rs))
    ax2.bar_label(bars, fmt="%.2f", fontsize=8, color="0.35", padding=2)
    ax2.set_xlabel("lag (bins)", fontsize=9, color="0.25")
    ax2.set_ylabel("held-out r", fontsize=9, color="0.25")
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(labelsize=8, colors="0.35")
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.set_title(f"…and the lag scan finds it (lag={lags[best]})", fontsize=10, color="0.25", loc="left")

    fig.suptitle("affectlens encode — which features the model leans on, and at what delay", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "encoding.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_midlevel_figure(X: pd.DataFrame) -> None:
    """The mid-level tier over the film: six perceptual primitives, each mapped
    to a named brain system, riding the same decode passes as the low-level set.
    """
    t_min = X.index.to_numpy() / 60.0
    fig, axes = plt.subplots(6, 1, figsize=(11, 7.4), sharex=True)

    # (column, label -> brain system, colour) for the five line panels.
    line_panels = [
        ("visual__flow_magnitude_mean", "motion energy\n(MT / V5)", "#c0504d"),
        ("visual__flow_coherence_mean", "flow coherence\n(MST / CSv)", "#e8862d"),
        ("visual__scene_cut_max", "scene cuts\n(hippocampus)", "#4472c4"),
        ("visual__chroma_by_mean", "colour warmth\n(V4 / VO)", "#7f6000"),
        ("audio__spectral_flatness_mean", "tonality: flatness\n(non-primary AC)", "#2e8b7f"),
    ]
    for ax, (col, label, color) in zip(axes[:5], line_panels, strict=True):
        ax.plot(t_min, X[col], color=color, lw=1.0)
        ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", color="0.25")
        ax.set_yticks([])

    f0 = X["audio__pitch_f0_mean"].to_numpy()
    voi = X["audio__voicing_mean"].to_numpy()
    sc = axes[5].scatter(t_min, f0, s=13, c=voi, cmap="viridis", vmin=0.0, vmax=float(voi.max() or 1.0))
    axes[5].set_ylabel("pitch F0 (Hz)\n(Heschl's)", fontsize=8, rotation=0, ha="right", va="center", color="0.25")
    axes[5].set_xlabel("time (minutes)", fontsize=9, color="0.25")
    fig.colorbar(sc, ax=axes[5], label="voicing", pad=0.01, fraction=0.05)

    axes[0].set_title(
        "affectlens mid-level tier — perceptual primitives over an 11-minute film, "
        "each mapped to a named brain system",
        loc="left", fontsize=11,
    )
    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("0.8")
        ax.tick_params(labelsize=8, colors="0.35")
        ax.margins(x=0.005)
    fig.savefig(OUT_DIR / "midlevel.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Every base (mean-aggregated) feature, grouped by tier, for the feature-matrix
# heatmap. scene_cut uses its _max column (the shot-boundary event, not its mean).
MATRIX_GROUPS = [
    ("Visual · low-level", [
        ("visual__luminance_mean", "luminance"),
        ("visual__contrast_mean", "contrast"),
        ("visual__colorfulness_mean", "colorfulness"),
        ("visual__saturation_mean", "saturation"),
        ("visual__edge_density_mean", "edge density"),
        ("visual__motion_mean", "motion"),
    ]),
    ("Visual · mid-level", [
        ("visual__flow_magnitude_mean", "flow magnitude"),
        ("visual__flow_looming_mean", "flow looming"),
        ("visual__flow_coherence_mean", "flow coherence"),
        ("visual__scene_cut_max", "scene cut"),
        ("visual__spatial_detail_mean", "spatial detail"),
        ("visual__chroma_rg_mean", "chroma R–G"),
        ("visual__chroma_by_mean", "chroma B–Y"),
    ]),
    ("Audio · low-level", [
        ("audio__rms_mean", "RMS loudness"),
        ("audio__zcr_mean", "zero-crossing rate"),
        ("audio__spectral_centroid_mean", "spectral centroid"),
        ("audio__spectral_flux_mean", "spectral flux"),
    ]),
    ("Audio · mid-level", [
        ("audio__pitch_f0_mean", "pitch F0"),
        ("audio__voicing_mean", "voicing"),
        ("audio__spectral_flatness_mean", "spectral flatness"),
        ("audio__loudness_attack_mean", "loudness attack"),
    ]),
]


def make_feature_matrix_figure(X: pd.DataFrame) -> None:
    """Every feature over time as one z-scored design matrix, grouped by tier —
    the whole feature space at a glance (what `extract` hands the encoding model).
    """
    t = X.index.to_numpy() / 60.0
    counts = [len(feats) for _, feats in MATRIX_GROUPS]
    fig, axes = plt.subplots(
        len(MATRIX_GROUPS), 1, sharex=True, figsize=(11, 8.8),
        gridspec_kw={"height_ratios": counts, "hspace": 0.22},
    )
    im = None
    for ax, (title, feats) in zip(axes, MATRIX_GROUPS, strict=True):
        names = [nm for _, nm in feats]
        rows = np.vstack([zscore(X[col]).to_numpy() for col, _ in feats])
        im = ax.imshow(
            rows, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
            interpolation="nearest",
            extent=[t[0], t[-1], len(names) - 0.5, -0.5],
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8, color="0.2")
        ax.set_title(title, loc="left", fontsize=9.5, color="0.25", pad=3)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel("time (minutes)", fontsize=9, color="0.25")
    axes[-1].tick_params(labelsize=8, colors="0.35", length=3)

    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
    cbar.set_label("z-scored value (per feature)", fontsize=8, color="0.3")
    cbar.ax.tick_params(labelsize=7, colors="0.4")
    fig.suptitle(
        "affectlens feature matrix — all 21 features (visual + audio, low- and "
        "mid-level) over an 11-minute film",
        fontsize=11.5, x=0.5, y=0.94,
    )
    fig.savefig(OUT_DIR / "feature_matrix.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    for path, hint in ((CLIP, "python scripts/fetch_samples.py"), (FEATURES, "affectlens extract --clips examples/samples --out out/")):
        if not path.exists():
            raise SystemExit(f"missing {path.relative_to(ROOT)} — run: {hint}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X = pd.read_csv(FEATURES, index_col=0)
    make_features_figure(X)
    make_encoding_figure(X)
    make_midlevel_figure(X)
    make_feature_matrix_figure(X)
    print(f"wrote {OUT_DIR / 'features.png'}")
    print(f"wrote {OUT_DIR / 'encoding.png'}")
    print(f"wrote {OUT_DIR / 'midlevel.png'}")
    print(f"wrote {OUT_DIR / 'feature_matrix.png'}")


if __name__ == "__main__":
    main()
