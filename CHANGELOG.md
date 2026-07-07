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
