"""horsetrack — track a horse's legs, joints and head in video (ridden or not)."""
from .layouts import AP10K, KeypointLayout, load_layout
from .pipeline import Config, Result, run

__all__ = ["AP10K", "KeypointLayout", "load_layout", "Config", "Result", "run"]
__version__ = "0.1.0"
