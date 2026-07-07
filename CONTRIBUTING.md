# Contributing

Thanks for your interest in improving affectlens.

## Getting set up

```bash
pip install -e ".[dev]"
pytest
```

A working ffmpeg is provided by `imageio-ffmpeg`, so no system install is
needed to run the tests.

## Guidelines

- Keep feature extractors pure: given a clip path, return a tidy DataFrame with a
  `t` (seconds) column and one column per feature. Alignment onto the rating grid
  is handled separately in `align.py`.
- New high-level backends should implement the `Transcriber` / `Embedder`
  interfaces in `highlevel.py` rather than hard-wiring a model, so the pipeline
  keeps running offline by default.
- Add a test. The suite runs the whole pipeline on generated synthetic data, so
  prefer extending that over adding fixtures.
- Run `pytest` before opening a pull request.

## Scope

affectlens is a feature-extraction and modeling toolkit. It intentionally ships
no datasets and no pretrained weights — bring your own clips, ratings, and
signals.
