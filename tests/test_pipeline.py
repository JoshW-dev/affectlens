"""End-to-end and unit tests. Run: python -m pytest (or python tests/test_pipeline.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the src-layout package importable without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from affectlens import align, encoding, highlevel, ratings
from affectlens.config import ExtractionConfig
from affectlens.ratings import RatingSchema
from affectlens.synthetic import build_synthetic_dataset


def test_ratings_wide_normalization():
    df = pd.DataFrame(
        {"time": [0, 4, 8], "brightness": [0.1, 0.5, 0.9], "energy": [0.2, 0.4, 0.6]}
    )
    long = ratings.normalize_frame(df, RatingSchema(), default_clip="c1")
    assert set(long["feature"]) == {"brightness", "energy"}
    assert len(long) == 6
    assert (long["t_end"] - long["t_start"]).round(1).eq(4.5).all()


def test_ratings_long_normalization_and_consensus():
    df = pd.DataFrame(
        {
            "clip": ["c1"] * 4,
            "participant": ["p1", "p2", "p1", "p2"],
            "time": [0, 0, 4, 4],
            "feature": ["energy", "energy", "energy", "energy"],
            "value": [0.2, 0.4, 0.6, 0.8],
        }
    )
    long = ratings.normalize_frame(df, RatingSchema())
    cons = ratings.consensus(long)
    row0 = cons[cons["t_start"] == 0].iloc[0]
    assert abs(row0["value"] - 0.3) < 1e-9
    assert row0["n_raters"] == 2


def test_align_binning_max_captures_spike():
    # A stream with a single spike inside the second bin; _max must catch it.
    stream = pd.DataFrame({"t": [0, 1, 2, 5, 6, 9], "motion": [0.1, 0.1, 0.1, 0.9, 0.1, 0.1]})
    edges = align.bin_edges_from_ratings(np.array([0.0, 4.0, 8.0]), interval_s=4.0)
    binned = align.aggregate_to_bins(stream, edges, ("mean", "max"), prefix="visual__")
    assert binned["visual__motion_max"].iloc[1] == 0.9
    assert binned["visual__motion_mean"].iloc[1] < 0.9


def test_semantic_stream_from_srt(tmp_path):
    srt = tmp_path / "clip.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:04,000\nhappy bright calm\n\n"
        "2\n00:00:04,000 --> 00:00:08,000\ntense sudden loud\n",
        encoding="utf-8",
    )
    (tmp_path / "clip.mp4").write_bytes(b"")  # only the stem is used to find the srt
    stream = highlevel.semantic_stream(tmp_path / "clip.mp4")
    assert not stream.empty
    assert "t" in stream.columns
    assert stream.shape[0] == 2


def test_encoding_recovers_and_ranks_signal_drivers():
    # An external signal that is a lagged, noisy combination of two features.
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {
            "visual__motion_mean": rng.random(n),
            "audio__rms_mean": rng.random(n),
            "noise": rng.random(n),
        }
    )
    true = 0.8 * X["visual__motion_mean"].to_numpy() + 0.5 * X["audio__rms_mean"].to_numpy()
    signal = np.empty(n)
    signal[1:] = true[:-1]  # signal lags the features by one bin
    signal[0] = np.nan
    signal += rng.normal(0, 0.03, n)

    enc = encoding.encode_signal(X, signal, lag_bins=1)
    assert enc.r > 0.6, f"encoding r too low: {enc.r}"
    # The pure-noise feature must not be the top driver.
    assert enc.weights[0][0] in ("visual__motion_mean", "audio__rms_mean")

    corr = encoding.correlate_signal(X, signal, lag_bins=[0, 1, 2])
    assert corr.iloc[0]["feature"] in ("visual__motion_mean", "audio__rms_mean")
    assert corr.iloc[0]["best_lag"] == 1


def test_end_to_end_pipeline_recovers_signal():
    from affectlens import pipeline

    with build_synthetic_dataset(n_clips=3) as (clips_dir, ratings_csv):
        per_clip, result = pipeline.run(
            clips_dir, ratings_csv, config=ExtractionConfig(), use_semantic=True
        )
    assert len(per_clip) == 3
    # Every clip should contribute aligned bins.
    assert all(len(cf.X) > 0 and len(cf.Y) > 0 for cf in per_clip)
    # The rated dimensions are noisy functions of the same drivers we extract,
    # so a working pipeline recovers a clearly positive mean correlation.
    assert result.mean_r > 0.3, f"mean_r too low: {result.mean_r}"


def test_midlevel_pitch_recovers_tone():
    from affectlens import midlevel

    sr, frame = 16000, int(0.05 * 16000)
    t = np.arange(frame) / sr
    tone = np.sin(2 * np.pi * 200 * t) * np.hanning(frame)
    f0, voicing = midlevel.pitch_from_spectrum(np.abs(np.fft.rfft(tone)), sr, frame)
    assert abs(f0 - 200) < 10  # recovers the fundamental
    assert voicing > 0.5

    noise = np.random.default_rng(0).normal(0, 1, frame) * np.hanning(frame)
    f0n, voicingn = midlevel.pitch_from_spectrum(np.abs(np.fft.rfft(noise)), sr, frame)
    assert f0n == 0.0  # unvoiced -> no pitch
    assert voicingn < 0.5


def test_midlevel_optical_flow_and_cut():
    from affectlens import midlevel

    img = (np.random.default_rng(1).random((72, 128)) * 255).astype(np.uint8)
    mag_shift, _ = midlevel.optical_flow_features(img, np.roll(img, 3, axis=1))
    mag_static, _ = midlevel.optical_flow_features(img, img)
    assert mag_shift > mag_static  # motion registers, static ~0
    assert mag_static < 0.1

    assert midlevel.scene_cut_score(img, img) < 0.05  # same frame -> no cut
    assert midlevel.scene_cut_score(img, 255 - img) > 0.5  # very different -> cut


def test_hashing_embedder_no_nan_on_cancelling_tokens():
    from affectlens.highlevel import HashingEmbedder

    # A small bucket count forces many tokens to collide; some buckets sum to 0
    # (opposite-sign hits cancel). The embedding must stay finite -- a 0 bucket
    # must not become NaN via log(0), which would later break the PCA SVD.
    emb = HashingEmbedder(dim=8)
    texts = [" ".join(f"tok{i}" for i in range(200)), "a b c a b c a"]
    v = emb.embed(texts)
    assert np.isfinite(v).all()


def test_extract_all_without_ratings():
    from affectlens import pipeline

    with build_synthetic_dataset(n_clips=2) as (clips_dir, _ratings_csv):
        per_clip = pipeline.extract_all(clips_dir, config=ExtractionConfig())
    assert len(per_clip) == 2
    for cf in per_clip:
        assert cf.Y is None
        # 20 s clips on the default 4.5 s grid -> 4 full bins.
        assert len(cf.X) >= 4
        assert any(c.startswith("visual__") for c in cf.X.columns)
        assert any(c.startswith("audio__") for c in cf.X.columns)


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            if "tmp_path" in inspect.signature(t).parameters:
                t(Path(tempfile.mkdtemp()))
            else:
                t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
