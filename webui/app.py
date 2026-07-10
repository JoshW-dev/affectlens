"""affectlens web UI — a point-and-click front end for the pipeline.

Run from the repo root:

    pip install -e ".[webui]"
    python scripts/fetch_samples.py      # optional: grab the demo clips
    streamlit run webui/app.py

No terminal needed after that — pick a clip, extract feature time courses, plot
them, and relate them to a recorded signal, all in the browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make the package importable when running from a checkout without installing.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from affectlens import clips as clips_mod  # noqa: E402
from affectlens import encoding  # noqa: E402
from affectlens import pipeline  # noqa: E402
from affectlens.config import ExtractionConfig  # noqa: E402

# Plain-English descriptions of each feature — used for the hover tooltips on the
# feature table and for the glossary.
FEATURE_HELP = {
    "luminance": "Mean pixel brightness (0-1) — bright vs. dark scenes.",
    "contrast": "Spread of brightness within the frame — flat vs. high-contrast.",
    "colorfulness": "How vivid the palette is.",
    "saturation": "Mean colour saturation.",
    "edge_density": "Fraction of pixels lying on an edge — visual busyness / detail.",
    "motion": "Mean absolute change between consecutive frames — how much is moving.",
    "flow_magnitude": "Optical-flow motion energy — how much is actually moving (truer than a frame difference).",
    "flow_looming": "Radial expansion of the optical flow — positive when the scene approaches the viewer.",
    "flow_coherence": "How global the motion is (0-1) — ~1 for a camera pan (self-motion), ~0 for scattered object motion.",
    "scene_cut": "Shot-boundary score — spikes at hard cuts.",
    "spatial_detail": "Fine-detail energy (variance of the Laplacian) — high for crisp frames, low under blur/defocus.",
    "chroma_rg": "Red-vs-green colour balance (signed) — which way the palette leans, not how vivid.",
    "chroma_by": "Blue-vs-yellow colour balance (signed) — positive = warm, negative = cool.",
    "rms": "Loudness (root-mean-square amplitude).",
    "zcr": "Zero-crossing rate — noisiness / high-frequency content (speech vs. tone).",
    "spectral_centroid": "Spectral 'brightness' — where the sound energy sits in frequency.",
    "spectral_flux": "How fast the spectrum changes — onsets and transitions.",
    "spectral_flatness": "Tone vs. noise (0-1) — ~0 for a musical note, →1 for hiss/applause/static.",
    "pitch_f0": "Fundamental frequency (pitch) of the audio, in Hz (0 when unvoiced).",
    "voicing": "Periodicity strength (0-1) — how voiced/tonal the sound is.",
    "loudness_attack": "Rise in loudness (dB) between windows — fires on hits/onsets, silent on decays.",
    "semantic": "Embedding of the dialogue text in a window (meaning, not pixels).",
}
_AGG = {"mean": "average over the bin", "std": "variation within the bin",
        "max": "peak within the bin"}


def feature_tooltip(col: str) -> str:
    """Turn a column like 'visual__luminance_mean' into a readable tooltip."""
    base = col.split("__", 1)[-1]
    for key, desc in FEATURE_HELP.items():
        if base.startswith(key):
            agg = base[len(key):].lstrip("_")
            return f"{desc}  [{_AGG.get(agg, agg)}]" if agg else desc
    return col


@st.cache_data(show_spinner=False)
def get_inventory(clips_dir: str):
    inv = clips_mod.inventory(Path(clips_dir))
    return [
        {"path": c.path, "clip": c.name,
         "duration_s": round(c.duration_s, 1) if c.duration_s else None,
         "fps": c.fps, "video": c.has_video, "audio": c.has_audio, "error": c.error}
        for c in inv
    ]


@st.cache_data(show_spinner="Extracting feature time courses…")
def extract(path: str, interval: float,
            visual: bool, audio: bool, use_semantic: bool) -> pd.DataFrame:
    cfg = ExtractionConfig(rating_interval_s=interval, visual=visual, audio=audio)
    # Grids from the decoded stream extent (survives bogus container headers) —
    # the same path extract_all/the CLI use.
    return pipeline.extract_clip_auto(path, cfg, use_semantic)


st.set_page_config(page_title="affectlens", layout="wide", page_icon="🎬")
st.title("🎬 affectlens")
st.caption("Extract time-varying features from video, audio, and music — then relate "
           "them to what people rated or what a signal recorded.")

with st.expander("What is this?"):
    st.markdown(
        "Point at a folder of clips. affectlens turns each clip into a **time × "
        "features** table (one row per time bin, one column per feature — luminance, "
        "motion, loudness, spectral shape, …). From there you can **plot** the feature "
        "time courses or **relate them to a recorded signal** (e.g. a brain channel) "
        "with a lag search. The same thing the CLI and the notebook do, without code."
    )

# ---------------------------------------------------------------- sidebar config
with st.sidebar:
    st.header("Setup")
    default_dir = str(ROOT / "examples" / "samples")
    clips_dir = st.text_input(
        "Clips folder", value=default_dir,
        help="A folder of video or audio files (.mp4, .mov, .wav, .mp3, …). "
             "The bundled demo clips live in examples/samples once you run "
             "`python scripts/fetch_samples.py`.")

    interval = st.slider(
        "Time-bin width (seconds)", min_value=0.5, max_value=10.0, value=4.5, step=0.5,
        help="Features are averaged into bins this wide. Smaller = finer temporal "
             "resolution — match it to your recording's sampling rate.")

    st.subheader("Feature families")
    use_visual = st.checkbox(
        "Visual", value=True,
        help="low-level: luminance, contrast, colorfulness, saturation, edge density, "
             "motion. mid-level: optical-flow magnitude/looming/coherence, scene cuts, "
             "spatial detail, colour opponency.")
    use_audio = st.checkbox(
        "Audio", value=True,
        help="low-level: loudness (RMS), zero-crossing rate, spectral centroid, spectral "
             "flux. mid-level: pitch + voicing, spectral flatness, loudness attack.")
    use_semantic = st.checkbox(
        "Semantic", value=False,
        help="Embeddings of dialogue text — needs a .srt/.vtt subtitle sidecar "
             "next to the clip.")

# ------------------------------------------------------------------ load clips
clips_path = Path(clips_dir)
if not clips_path.exists():
    st.warning(f"Folder not found: `{clips_dir}`. Point to a folder of clips, or run "
               "`python scripts/fetch_samples.py` to download the demo clips.")
    st.stop()

inv = [c for c in get_inventory(clips_dir) if c["error"] is None]
if not inv:
    st.warning(f"No readable clips in `{clips_dir}`.")
    st.stop()

tab_feat, tab_signal = st.tabs(["1 · Clips & features", "2 · Relate to a signal"])

# ============================================================ TAB 1: features
with tab_feat:
    st.subheader("Clips in this folder")
    st.dataframe(pd.DataFrame(inv)[["clip", "duration_s", "fps", "video", "audio"]],
                 hide_index=True, width="stretch")

    names = [c["clip"] for c in inv]
    picked = st.selectbox("Pick a clip to analyse", names)
    info = next(c for c in inv if c["clip"] == picked)

    X = extract(info["path"], interval, use_visual, use_audio, use_semantic)

    c1, c2, c3 = st.columns(3)
    c1.metric("Time bins", X.shape[0], help="rows — one per time bin")
    c2.metric("Features", X.shape[1], help="columns — one per feature")
    c3.metric("Bin width", f"{interval:g} s")

    st.markdown("**Feature matrix** — hover a column header for what it means.")
    st.dataframe(
        X.round(4), width="stretch", height=280,
        column_config={col: st.column_config.NumberColumn(col, help=feature_tooltip(col))
                       for col in X.columns})

    st.download_button(
        "⬇ Download this clip's features (CSV)",
        X.to_csv().encode("utf-8"),
        file_name=f"{picked.rsplit('.', 1)[0]}__features.csv", mime="text/csv")

    st.markdown("**Plot feature time courses** (x-axis = time in seconds):")
    default_plot = [c for c in ("visual__luminance_mean", "audio__rms_mean") if c in X.columns]
    chosen = st.multiselect("Features to plot", list(X.columns),
                            default=default_plot or list(X.columns[:2]))
    if chosen:
        st.line_chart(X[chosen])

    with st.expander("Feature glossary"):
        for key, desc in FEATURE_HELP.items():
            st.markdown(f"- **{key}** — {desc}")

# ============================================================ TAB 2: signal
with tab_signal:
    st.subheader("Relate the features to a recorded signal")
    st.markdown(
        "Upload a signal you recorded alongside the clip (an EEG band envelope, an "
        "fMRI ROI time course, pupil size, heart rate…) as a **CSV with a time column "
        "and a value column**. `encode` scans lags and fits a cross-validated model "
        "that reports how well the features predict it and which ones it leans on.")

    picked2 = st.selectbox("Clip", names, key="signal_clip")
    info2 = next(c for c in inv if c["clip"] == picked2)
    X2 = extract(info2["path"], interval, use_visual, use_audio, use_semantic)

    if len(X2) < 8:
        st.info(
            f"**‘{picked2}’ has only {len(X2)} time bins at a {interval:g}s bin width** — "
            "too few to fit an encoding model on a single clip. Lower the time-bin width "
            "in the sidebar (try 0.5–1s), or pick a longer clip. (For short stimuli, the "
            "usual approach is to concatenate features and signal across many clips.)")

    lags = st.multiselect("Lags to scan (in bins)", list(range(0, 6)), default=[0, 1, 2, 3],
                          help="A recorded response often trails the stimulus by a fixed "
                               "delay; scanning lags finds it. 1 bin = one time-bin width.")

    demo = st.toggle(
        "I don't have a signal yet — use a demo one",
        help="Builds a mock signal from this clip's own loudness, delayed by one bin, "
             "plus noise — a sanity check that the machinery recovers a known answer.")

    signal = None
    if demo:
        if "audio__rms_mean" not in X2.columns:
            st.info("The demo signal needs audio features — pick a clip with an audio track.")
        else:
            rng = np.random.default_rng(7)
            drive = X2["audio__rms_mean"].to_numpy()
            drive = (drive - drive.mean()) / (drive.std() + 1e-9)
            signal = np.roll(drive, 1)
            signal[0] = 0.0
            signal = signal + rng.normal(0, 0.25, len(signal))
            st.caption("Using a demo signal = this clip's loudness delayed 1 bin + noise.")
    else:
        up = st.file_uploader("Signal CSV", type=["csv"])
        if up is not None:
            sig = pd.read_csv(up)
            cols = list(sig.columns)
            a, b = st.columns(2)
            tcol = a.selectbox("Time column (seconds)", cols, index=0)
            vcol = b.selectbox("Value column", cols, index=min(1, len(cols) - 1))
            signal = encoding.bin_signal(
                sig[tcol].to_numpy(), sig[vcol].to_numpy(), X2.index.to_numpy(), interval_s=interval)

    if signal is not None and len(X2) >= 8 and st.button("Run encode", type="primary"):
        rs = {lag: encoding.encode_signal(X2, signal, lag_bins=lag).r for lag in sorted(lags)}
        best = max(rs, key=lambda k: (rs[k] if rs[k] == rs[k] else -1))
        enc = encoding.encode_signal(X2, signal, lag_bins=best)

        m1, m2 = st.columns(2)
        m1.metric("Best lag", f"{best} bins", help=f"= {best * interval:g} s")
        m2.metric("Held-out r at best lag", f"{enc.r:.3f}",
                  help="cross-validated correlation between predicted and recorded signal")

        st.markdown("**Held-out r by lag** — the scan; the tallest bar is the delay it found.")
        st.bar_chart(pd.Series(rs, name="held-out r").rename_axis("lag (bins)"))

        st.markdown("**Features the model leans on** (top |weight|):")
        st.dataframe(
            pd.DataFrame(enc.weights[:10], columns=["feature", "weight"]).round(3),
            hide_index=True, width="stretch",
            column_config={"feature": st.column_config.TextColumn(
                "feature", help="Interpret as an importance ranking, not a clean causal "
                                "attribution — ridge spreads weight across correlated features.")})
