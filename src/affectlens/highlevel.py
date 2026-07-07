"""High-level, human-centered feature extraction.

The most valuable part of the pipeline: moving past luminance/color to
regressors that track *meaning*. The approach is chunk-by-chunk semantic
embeddings of the spoken dialogue, so the design matrix carries what is being
said, not just how bright or loud the frame is. (Tonality and cadence live in
the audio features; on-screen text is a natural extension.)

Design: two small interfaces so the heavy, networked pieces are swappable.

  Transcriber  -- clip -> time-stamped dialogue segments
                  (real path: Whisper / whisperX; offline path: read a provided
                   .srt/.vtt/.csv transcript)
  Embedder     -- text -> vector
                  (real path: sentence-transformers or an embedding API;
                   offline default: hashed-bag TF-style vector, no network)

The offline defaults let the whole pipeline run and be tested end-to-end in a
sandbox with no model downloads. Swap in the networked implementations where the
environment allows; nothing downstream changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class Segment:
    t_start: float
    t_end: float
    text: str


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
class Transcriber(Protocol):
    def transcribe(self, clip_path: str | Path) -> list[Segment]: ...


def _parse_timestamp(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + s


class TranscriptFileTranscriber:
    """Offline transcriber: read an existing subtitle/transcript sidecar.

    Looks for ``<clip>.srt``, ``.vtt`` or ``.csv`` next to the clip (or takes an
    explicit path). CSV is expected to have t_start,t_end,text columns.
    """

    def __init__(self, transcript_path: str | Path | None = None):
        self.transcript_path = Path(transcript_path) if transcript_path else None

    def _find(self, clip_path: Path) -> Path | None:
        if self.transcript_path:
            return self.transcript_path
        for ext in (".srt", ".vtt", ".csv"):
            cand = clip_path.with_suffix(ext)
            if cand.exists():
                return cand
        return None

    def transcribe(self, clip_path: str | Path) -> list[Segment]:
        clip_path = Path(clip_path)
        src = self._find(clip_path)
        if src is None:
            return []
        text = src.read_text(encoding="utf-8", errors="ignore")
        if src.suffix.lower() == ".csv":
            df = pd.read_csv(src)
            return [
                Segment(float(r.t_start), float(r.t_end), str(r.text))
                for r in df.itertuples()
            ]
        return _parse_srt_vtt(text)


def _parse_srt_vtt(text: str) -> list[Segment]:
    segments: list[Segment] = []
    # Matches "00:00:01,000 --> 00:00:03,500" (srt) and vtt variants.
    pat = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*"
        r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
    )
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        m = pat.search(block)
        if not m:
            continue
        lines = [ln for ln in block.splitlines() if not pat.search(ln) and not ln.strip().isdigit()]
        content = " ".join(ln.strip() for ln in lines if ln.strip())
        if content:
            segments.append(Segment(_parse_timestamp(m.group(1)), _parse_timestamp(m.group(2)), content))
    return segments


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic, no-network sentence embedding via the hashing trick.

    A hashed bag-of-words with sublinear term weighting and L2 normalization.
    This is a stand-in for a real semantic model (e.g. sentence-transformers),
    good enough to exercise and test the pipeline offline. It is NOT a semantic
    model -- swap a real embedder in when the environment permits model
    downloads or API calls.
    """

    _token_re = re.compile(r"[a-z0-9']+")

    def __init__(self, dim: int = 256, seed: int = 17):
        self.dim = dim
        self.seed = seed

    def _tokens(self, text: str) -> list[str]:
        return self._token_re.findall(text.lower())

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[int, float] = {}
            for tok in self._tokens(text):
                h = hash((self.seed, tok))
                idx = h % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                counts[idx] = counts.get(idx, 0.0) + sign
            for idx, c in counts.items():
                out[i, idx] = np.sign(c) * (1.0 + np.log(abs(c)))
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


# --------------------------------------------------------------------------- #
# Semantic feature stream
# --------------------------------------------------------------------------- #
def semantic_stream(
    clip_path: str | Path,
    transcriber: Transcriber | None = None,
    embedder: Embedder | None = None,
    n_components: int | None = 16,
) -> pd.DataFrame:
    """Produce a per-segment semantic feature stream for a clip.

    Returns a frame with a ``t`` column (segment midpoint) and ``sem_*`` columns.
    Empty (no columns beyond none) when the clip has no available transcript, so
    the pipeline degrades gracefully to low-level-only features.

    ``n_components`` optionally reduces the raw embedding to its leading PCA
    components across the clip's segments, keeping the design matrix compact.
    """
    transcriber = transcriber or TranscriptFileTranscriber()
    embedder = embedder or HashingEmbedder()

    segments = transcriber.transcribe(clip_path)
    if not segments:
        return pd.DataFrame()

    vecs = embedder.embed([s.text for s in segments])
    if n_components is not None and vecs.shape[0] > 1 and vecs.shape[1] > n_components:
        vecs = _pca(vecs, n_components)

    mids = [(s.t_start + s.t_end) / 2.0 for s in segments]
    df = pd.DataFrame(vecs, columns=[f"sem_{i}" for i in range(vecs.shape[1])])
    df.insert(0, "t", mids)
    return df


def _pca(x: np.ndarray, k: int) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    # Economy SVD; columns of Vt are principal directions.
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:k].T
