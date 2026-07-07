"""affectlens: time-varying feature extraction from video/audio stimuli.

Extract low-level (visual/audio) and high-level (semantic dialogue) features
from video/audio clips and score them against continuous human ratings. Useful
for modeling how people respond to media moment-by-moment, and for correlating
feature time courses against any external signal (e.g. physiological or
neuroimaging recordings). See the README for the mental model.
"""

from .config import ExtractionConfig
from .ratings import RatingSchema

__all__ = ["ExtractionConfig", "RatingSchema", "__version__"]
__version__ = "0.1.0"
