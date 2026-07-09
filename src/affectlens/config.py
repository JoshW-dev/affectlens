"""Project-wide configuration and constants.

Features are aggregated onto a shared time grid -- the rating grid when ratings
are supplied, otherwise a grid derived from each clip's decoded duration -- so
the design matrix and target share a time axis. All values here are overridable
per run via :class:`ExtractionConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Default spacing of the human rating grid, in seconds. Continuous behavioral
# ratings are often collected every few seconds; override to match your data.
DEFAULT_RATING_INTERVAL_S: float = 4.5

# Visual sampling. We do not need every frame for slowly-varying low-level
# regressors; sampling a few frames per second keeps extraction fast while still
# capturing motion. Set to None to use the clip's native frame rate.
VISUAL_SAMPLE_FPS: float | None = 8.0

# Audio analysis frame/hop (seconds) for framewise loudness / spectral features.
AUDIO_FRAME_S: float = 0.05
AUDIO_HOP_S: float = 0.025
AUDIO_SAMPLE_RATE: int = 16000


@dataclass
class ExtractionConfig:
    """Knobs for a single extraction run."""

    rating_interval_s: float = DEFAULT_RATING_INTERVAL_S
    visual_sample_fps: float | None = VISUAL_SAMPLE_FPS
    audio_frame_s: float = AUDIO_FRAME_S
    audio_hop_s: float = AUDIO_HOP_S
    audio_sample_rate: int = AUDIO_SAMPLE_RATE
    # Which low-level feature families to compute.
    visual: bool = True
    audio: bool = True
    # Aggregation statistics applied within each rating bin.
    bin_aggregations: tuple[str, ...] = field(default_factory=lambda: ("mean", "std", "max"))
