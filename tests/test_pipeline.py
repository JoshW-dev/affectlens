"""End-to-end and unit tests. Run: python -m pytest (or python tests/test_pipeline.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Make the src-layout package importable without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from affectlens import align, encoding, highlevel, lowlevel, ratings
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
    mag_shift, _, coh_shift = midlevel.optical_flow_features(img, np.roll(img, 3, axis=1))
    mag_static, _, _ = midlevel.optical_flow_features(img, img)
    assert mag_shift > mag_static  # motion registers, static ~0
    assert mag_static < 0.1
    # A uniform horizontal shift is coherent (whole field moves together).
    assert 0.0 <= coh_shift <= 1.0
    assert coh_shift > 0.5

    assert midlevel.scene_cut_score(img, img) < 0.05  # same frame -> no cut
    assert midlevel.scene_cut_score(img, 255 - img) > 0.5  # very different -> cut


def test_midlevel_flow_coherence_global_vs_local():
    from affectlens import midlevel

    rng = np.random.default_rng(2)
    base = (rng.random((80, 128)) * 255).astype(np.uint8)
    # Global pan: every pixel shifts the same way -> high coherence.
    _, _, coh_global = midlevel.optical_flow_features(base, np.roll(base, 2, axis=1))
    # Opposing halves shift in opposite directions -> they cancel in the mean
    # vector, so the field is much less globally coherent.
    scrambled = base.copy()
    h = base.shape[0] // 2
    scrambled[:h] = np.roll(base[:h], 3, axis=1)
    scrambled[h:] = np.roll(base[h:], -3, axis=1)
    _, _, coh_split = midlevel.optical_flow_features(base, scrambled)
    assert coh_global > coh_split


def test_midlevel_spatial_detail_and_chroma():
    from affectlens import midlevel

    rng = np.random.default_rng(3)
    detailed = rng.random((64, 64)).astype(np.float32)  # fine texture
    flat = np.full((64, 64), 0.5, np.float32)
    blurred = cv2.GaussianBlur(detailed, (0, 0), 3)
    assert midlevel.spatial_detail(flat) == 0.0  # no high-SF energy
    assert midlevel.spatial_detail(detailed) > midlevel.spatial_detail(blurred)

    # BGR frames: a red frame leans +rg and warm; a blue frame leans cool.
    red = np.zeros((8, 8, 3), np.uint8)
    red[..., 2] = 255
    blue = np.zeros((8, 8, 3), np.uint8)
    blue[..., 0] = 255
    gray = np.full((8, 8, 3), 128, np.uint8)
    rg_red, by_red = midlevel.chroma_opponency(red)
    _, by_blue = midlevel.chroma_opponency(blue)
    rg_gray, by_gray = midlevel.chroma_opponency(gray)
    assert rg_red > 0.5 and by_red > 0.0  # red-dominant, warm
    assert by_blue < 0.0  # blue-dominant, cool
    assert abs(rg_gray) < 1e-6 and abs(by_gray) < 1e-6  # neutral


def test_midlevel_flatness_and_attack():
    from affectlens import midlevel

    sr, frame = 16000, int(0.05 * 16000)
    t = np.arange(frame) / sr
    tone = np.sin(2 * np.pi * 220 * t) * np.hanning(frame)
    noise = np.random.default_rng(4).normal(0, 1, frame) * np.hanning(frame)
    flat_tone = midlevel.spectral_flatness(np.abs(np.fft.rfft(tone)))
    flat_noise = midlevel.spectral_flatness(np.abs(np.fft.rfft(noise)))
    assert flat_tone < flat_noise  # tonal is spectrally peaky, noise is flat
    assert 0.0 <= flat_tone <= 1.0 and 0.0 <= flat_noise <= 1.0
    assert midlevel.spectral_flatness(np.zeros(frame // 2 + 1)) == 0.0  # silence gate

    # loudness_attack fires on rises only, never on the first window or on decays.
    assert midlevel.loudness_attack(0.5, None) == 0.0  # first window
    assert midlevel.loudness_attack(0.5, 0.05) > 0.0  # rising -> positive dB
    assert midlevel.loudness_attack(0.05, 0.5) == 0.0  # falling -> rectified to 0


def test_midlevel_voice_band_ratio():
    from affectlens import midlevel

    sr, frame = 16000, int(0.05 * 16000)
    freqs = np.fft.rfftfreq(frame, d=1.0 / sr)
    t = np.arange(frame) / sr
    speech = np.sin(2 * np.pi * 1000 * t) * np.hanning(frame)  # in 300-3400 Hz band
    rumble = np.sin(2 * np.pi * 60 * t) * np.hanning(frame)    # below the band
    r_speech = midlevel.voice_band_ratio(np.abs(np.fft.rfft(speech)), freqs)
    r_rumble = midlevel.voice_band_ratio(np.abs(np.fft.rfft(rumble)), freqs)
    assert r_speech > 0.8  # nearly all energy in the speech band
    assert r_speech > r_rumble
    assert 0.0 <= r_rumble <= 1.0
    assert midlevel.voice_band_ratio(np.zeros(frame // 2 + 1), freqs) == 0.0  # silence


def test_midlevel_face_features_summary():
    from affectlens import midlevel

    # Pure summariser, no model needed: YuNet detect() returns None or Nx15
    # boxes (x, y, w, h, ...). Count and largest-box-area fraction.
    assert midlevel.face_features(None, 100, 100) == (0.0, 0.0)
    assert midlevel.face_features(np.zeros((0, 15)), 100, 100) == (0.0, 0.0)
    boxes = np.array([[10, 10, 20, 20] + [0.0] * 11, [0, 0, 50, 40] + [0.0] * 11])
    count, prom = midlevel.face_features(boxes, 100, 100)
    assert count == 2.0
    assert abs(prom - (50 * 40) / (100 * 100)) < 1e-9  # largest box area fraction
    assert 0.0 <= prom <= 1.0


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


def test_visual_decode_falls_back_when_opencv_truncates():
    """A clip OpenCV abandons partway must not silently lose its tail.

    Some variable-rate AVIs make OpenCV stop decoding early and report no error,
    which used to leave the rest of the clip with no frames and a block of NaN
    features once binned. Here the OpenCV path is truncated to half its frames,
    so ``extract_visual`` should notice that it decoded fewer than the container
    declares, re-decode with ffmpeg, and cover the whole clip again.
    """
    import warnings

    with build_synthetic_dataset(n_clips=1) as (clips_dir, _ratings_csv):
        clip = sorted(Path(clips_dir).glob("*.mp4"))[0]
        full = lowlevel.extract_visual(clip)

        real_iter = lowlevel._iter_frames_cv2

        def truncated(path):
            frames = list(real_iter(path))
            return iter(frames[: len(frames) // 2])

        lowlevel._iter_frames_cv2 = truncated
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                recovered = lowlevel.extract_visual(clip)
        finally:
            lowlevel._iter_frames_cv2 = real_iter

    # The tail is back: same coverage as an untruncated decode ...
    assert len(recovered) == len(full)
    assert recovered["t"].max() == full["t"].max()
    assert not recovered.isna().any().any()
    # ... and the recovery was announced rather than silently papered over.
    assert any(
        issubclass(w.category, RuntimeWarning) and "ffmpeg" in str(w.message) for w in caught
    )


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
