# Examples

- **`getting_started.ipynb`** — the guided tour, and the best place to start. It
  walks the whole pipeline on the public sample clips with a plot and a
  plain-English explanation at each step: inventory → extract feature time
  courses → relate them to a recorded signal (`encode`) → predict ratings
  (`baseline`). Ships with its outputs so you can read it without running
  anything.

  ```bash
  pip install -e ".[notebook]"
  python scripts/fetch_samples.py
  jupyter lab examples/getting_started.ipynb
  ```

- **`demo.py`** — runs the whole pipeline on generated synthetic clips (no
  external data): extracts features, reproduces synthetic ratings with the
  cross-validated baseline, then recovers a fabricated lagged "recorded signal"
  with the encoding model.

  ```bash
  python examples/demo.py
  ```

- **`samples.json`** — a manifest of sample clip URLs. Third-party clips are
  linked here, never committed. Fetch them into `examples/samples/` (gitignored)
  with:

  ```bash
  python scripts/fetch_samples.py
  affectlens inventory --clips examples/samples
  ```

  Direct media URLs need nothing extra; YouTube links require
  `pip install yt-dlp`. Each entry is `{"name", "url"}` with optional
  `"start"`/`"duration"` seconds to trim a section.

## Point-and-click web UI

Prefer buttons to code? A small [Streamlit](https://streamlit.io) app in
[`webui/app.py`](../webui/app.py) wraps the same pipeline: pick a clip, extract
feature time courses, plot them, and relate them to an uploaded signal — with a
hover tooltip explaining every feature.

```bash
pip install -e ".[webui]"
python scripts/fetch_samples.py       # optional: the demo clips
streamlit run webui/app.py
```

Bring your own clips and point any of these (CLI, notebook, or web UI) at them to
run the same flow on real data — see the top-level README quick start.
