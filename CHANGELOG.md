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
  model reporting held-out r and per-feature weights.
- `affectlens selftest`: generates synthetic clips and runs the full pipeline.
- Ratings-free extraction: `affectlens extract` works without `--ratings`,
  binning features on a duration-derived grid (`pipeline.extract_all`) — the
  workflow for relating features to an external signal with `encode`.
- Sample-clip manifest (`examples/samples.json`) and fetcher
  (`scripts/fetch_samples.py`): real public-domain / CC test clips are linked
  by URL and downloaded locally, never committed.
- README visuals: a Mermaid pipeline diagram plus figures generated from the
  real sample footage (film frames over extracted feature time courses, and an
  `encode` lag-scan demo) — regenerate with `scripts/make_readme_figures.py`.

### Fixed
- `pipeline.extract_all` now derives each clip's bin grid from the decoded
  stream's true extent rather than the container-header duration. Some files
  (notably certain AVIs) report a bogus multi-hour duration from a malformed
  header, which previously produced a grid of thousands of empty bins.
