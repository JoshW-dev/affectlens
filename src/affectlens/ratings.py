"""Load and normalize the behavioral rating files.

Rating files arrive in many shapes. Rather than hard-code columns, this module
normalizes whatever it is given into one canonical long-format table:

    clip | participant | t_start | t_end | feature | value

Everything downstream (alignment, baseline scoring) consumes that canonical
frame, so adapting to the real files means adjusting only the column mapping
here -- ideally just the ``RatingSchema`` passed in.

Supports CSV and Excel (.xlsx) inputs, both "wide" (one column per rated
feature) and "long" (a feature/value column pair) layouts, and either a single
combined file or one file per participant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_RATING_INTERVAL_S

# Column-name hints used by the auto-detector. Extend as we learn the real
# headers. Matching is case-insensitive and substring-based.
TIME_HINTS = ("time", "onset", "t_start", "start", "sec", "timestamp", "bin")
END_HINTS = ("t_end", "end", "offset")
PARTICIPANT_HINTS = ("participant", "subject", "subj", "rater", "sid", "id")
FEATURE_HINTS = ("feature", "dimension", "label", "item")
VALUE_HINTS = ("value", "rating", "score", "response")
CLIP_HINTS = ("clip", "movie", "stimulus", "video", "trial")

# Non-feature bookkeeping columns to exclude when treating a file as "wide".
_NON_FEATURE = set(TIME_HINTS) | set(END_HINTS) | set(PARTICIPANT_HINTS) | set(CLIP_HINTS)


@dataclass
class RatingSchema:
    """Explicit column mapping. Leave a field None to auto-detect it."""

    time_col: str | None = None
    end_col: str | None = None
    participant_col: str | None = None
    clip_col: str | None = None
    feature_col: str | None = None  # set for long-format files
    value_col: str | None = None  # set for long-format files
    feature_cols: list[str] | None = None  # set for wide-format files
    rating_interval_s: float = DEFAULT_RATING_INTERVAL_S


def _match(columns: list[str], hints: tuple[str, ...]) -> str | None:
    lowered = {c: c.lower() for c in columns}
    for hint in hints:
        for col, low in lowered.items():
            if low == hint:
                return col
    for hint in hints:
        for col, low in lowered.items():
            if hint in low:
                return col
    return None


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix in (".csv", ".txt", ".tsv"):
        sep = "\t" if suffix == ".tsv" else None
        return pd.read_csv(path, sep=sep, engine="python")
    raise ValueError(f"unsupported rating file type: {path.suffix}")


def _infer_participant_from_name(path: Path) -> str:
    m = re.search(r"(?:sub|subj|participant|p|s)[-_]?(\d+)", path.stem, re.IGNORECASE)
    return f"p{int(m.group(1))}" if m else path.stem


def normalize_frame(
    df: pd.DataFrame,
    schema: RatingSchema,
    *,
    default_clip: str | None = None,
    default_participant: str | None = None,
) -> pd.DataFrame:
    """Convert one raw rating table into the canonical long format."""
    cols = list(df.columns.astype(str))
    df = df.copy()
    df.columns = cols

    time_col = schema.time_col or _match(cols, TIME_HINTS)
    end_col = schema.end_col or _match(cols, END_HINTS)
    participant_col = schema.participant_col or _match(cols, PARTICIPANT_HINTS)
    clip_col = schema.clip_col or _match(cols, CLIP_HINTS)
    feature_col = schema.feature_col or _match(cols, FEATURE_HINTS)
    value_col = schema.value_col or _match(cols, VALUE_HINTS)

    if time_col is None:
        raise ValueError(
            f"could not find a time column in {cols!r}; set RatingSchema.time_col"
        )

    # Columns we carry through the melt as identifiers rather than features.
    id_cols = [c for c in (time_col, end_col, participant_col, clip_col) if c]

    if schema.feature_col and schema.value_col or (feature_col and value_col and not schema.feature_cols):
        # Long layout: an explicit feature/value column pair already exists.
        long = df[id_cols + [feature_col, value_col]].rename(
            columns={feature_col: "feature", value_col: "value"}
        )
    else:
        # Wide layout: one column per rated feature. Melt them into rows.
        if schema.feature_cols:
            feature_cols = list(schema.feature_cols)
        else:
            feature_cols = [
                c
                for c in cols
                if c not in id_cols
                and c.lower() not in _NON_FEATURE
                and pd.api.types.is_numeric_dtype(df[c])
            ]
        if not feature_cols:
            raise ValueError(
                f"no numeric feature columns detected in {cols!r}; "
                "pass RatingSchema.feature_cols or feature_col/value_col"
            )
        long = df.melt(
            id_vars=id_cols, value_vars=feature_cols, var_name="feature", value_name="value"
        )

    # Canonical rename of the identifier columns.
    rename = {time_col: "t_start"}
    if end_col:
        rename[end_col] = "t_end"
    if participant_col:
        rename[participant_col] = "participant"
    if clip_col:
        rename[clip_col] = "clip"
    long = long.rename(columns=rename)

    long["t_start"] = pd.to_numeric(long["t_start"], errors="coerce")
    long = long.dropna(subset=["t_start"])
    if "t_end" not in long:
        long["t_end"] = long["t_start"] + schema.rating_interval_s
    if "participant" not in long:
        long["participant"] = default_participant or "p1"
    if "clip" not in long:
        long["clip"] = default_clip or "clip"
    long["clip"] = long["clip"].fillna(default_clip or "clip")

    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    return long[["clip", "participant", "t_start", "t_end", "feature", "value"]].reset_index(drop=True)


def load_ratings(
    path: str | Path,
    schema: RatingSchema | None = None,
) -> pd.DataFrame:
    """Load rating file(s) into the canonical long format.

    ``path`` may be a single file or a directory. A directory is treated as one
    file per participant (participant id inferred from the filename when the
    file itself has no participant column).
    """
    schema = schema or RatingSchema()
    path = Path(path)

    if path.is_dir():
        frames = []
        for f in sorted(path.iterdir()):
            if f.suffix.lower() not in (".csv", ".tsv", ".txt", ".xlsx", ".xls"):
                continue
            raw = _read_any(f)
            frames.append(
                normalize_frame(
                    raw, schema, default_participant=_infer_participant_from_name(f)
                )
            )
        if not frames:
            raise FileNotFoundError(f"no rating files found in {path}")
        return pd.concat(frames, ignore_index=True)

    raw = _read_any(path)
    return normalize_frame(raw, schema)


def consensus(long: pd.DataFrame) -> pd.DataFrame:
    """Average across participants to a per-(clip, bin, feature) consensus target.

    Returns long format with an added ``n_raters`` column so we can see how many
    participants contributed to each cell.
    """
    grouped = (
        long.groupby(["clip", "t_start", "t_end", "feature"], as_index=False)
        .agg(value=("value", "mean"), n_raters=("participant", "nunique"))
    )
    return grouped


def to_target_matrix(consensus_long: pd.DataFrame, clip: str | None = None) -> pd.DataFrame:
    """Pivot consensus ratings to a bins x features matrix indexed by t_start."""
    df = consensus_long
    if clip is not None:
        df = df[df["clip"] == clip]
    wide = df.pivot_table(index="t_start", columns="feature", values="value", aggfunc="mean")
    wide = wide.sort_index()
    return wide
