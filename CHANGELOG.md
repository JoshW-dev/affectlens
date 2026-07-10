# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

### Added
- Clip inventory (`affectlens inventory`) for video and audio-only files.
- Low-level visual features: luminance, contrast, colorfulness, saturation,
  edge density, motion.
- Low-level audio features: RMS loudness, zero-crossing rate, spectral centroid,
  spectral flux.
- Mid-level perceptual features (`midlevel.py`), each mapping onto a named brain
  system and computed inside the existing decode passes (no extra dependency):
  - optical-flow motion — `flow_magnitude` (energy, MT/V5), `flow_looming`
    (approach/recede, MSTd), `flow_coherence` (global self-motion vs local object
    motion, MST/CSv); gated by `ExtractionConfig.optical_flow`;
  - `scene_cut` — shot-boundary score (hippocampal event segmentation);
  - `spatial_detail` — high-spatial-frequency energy (V1 spatial-frequency channels);
  - `chroma_rg`, `chroma_by` — signed cone-opponent colour axes (V4/VO);
  - `pitch_f0`, `voicing` — fundamental frequency and periodicity (Heschl's gyrus);
  - `spectral_flatness` — tonal vs. noisy texture (non-primary auditory cortex);
  - `loudness_attack` — rectified loudness rise (brainstem acoustic-startle arc).
  `midlevel.py` documents a roadmap of further extractors mapped to their brain
  systems; the README carries the full tier table with references.
- High-level semantic features from dialogue, with swappable transcriber and
  embedder interfaces (offline hashing / subtitle defaults; Whisper and
  sentence-transformers as optional backends).
- Schema-flexible rating loader (wide/long, combined/per-participant, CSV/Excel)
  with per-participant consensus.
- Alignment of every feature stream onto a shared rating/feature time grid, with
  mean/std/max per bin to preserve within-bin events.
- Rating baseline (`affectlens baseline`): cross-validated Ridge with
  leave-one-clip-out folds, per-dimension Pearson r / R².
- Signal encoding (`affectlens encode`): correlate features against an external
  continuous signal with lag scanning, plus a cross-validated ridge encoding
  model (contiguous folds by default, to avoid temporal leakage) reporting
  held-out r and per-feature weights.
- `affectlens selftest`: generates synthetic clips and runs the full pipeline.
- Ratings-free extraction: `affectlens extract` works without `--ratings`,
  binning features on a duration-derived grid (`pipeline.extract_all`) — the
  workflow for relating features to an external signal with `encode`.
- Sample-clip manifest (`examples/samples.json`) and fetcher
  (`scripts/fetch_samples.py`): real public-domain / CC test clips are linked
  by URL and downloaded locally, never committed.
- README visuals: a pipeline diagram plus figures generated from the real
  sample footage (film frames over extracted feature time courses, an `encode`
  lag-scan demo, and the mid-level tier of perceptual primitives over the clip)
  — regenerate with `scripts/make_readme_figures.py`.

### Fixed
- `pipeline.extract_all` now derives each clip's bin grid from the decoded
  stream's true extent rather than the container-header duration. Some files
  (notably certain AVIs) report a bogus multi-hour duration from a malformed
  header, which previously produced a grid of thousands of empty bins.
