"""Compatibility exports for algorithms.motion."""

import sys

from .algorithms import motion as _implementation
from .algorithms.motion import (
    ANALYSIS_FIELDS,
    MotionConfig,
    analyze,
    analyze_detections,
    infer_events,
    robust_local_fit,
)

__all__ = [
    "ANALYSIS_FIELDS",
    "MotionConfig",
    "analyze",
    "analyze_detections",
    "infer_events",
    "robust_local_fit",
]
sys.modules[__name__] = _implementation
