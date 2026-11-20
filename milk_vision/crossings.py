"""Compatibility exports for algorithms.placement."""

import sys

from .algorithms import placement as _implementation
from .algorithms.placement import (
    CROSSING_CONFIDENCE,
    CROSSING_CONFIRM_SECONDS,
    EVIDENCE_GAP_SECONDS,
    GATE_HYSTERESIS,
    INNER,
    INWARD,
    OUTER,
    OUTWARD,
    TRACK_GAP_SECONDS,
    TRACKING_CONFIDENCE,
    CrossingDetector,
    _seconds_to_frames,
    analyze,
    calculate_net_delta,
)

__all__ = [
    "CROSSING_CONFIDENCE",
    "CROSSING_CONFIRM_SECONDS",
    "EVIDENCE_GAP_SECONDS",
    "GATE_HYSTERESIS",
    "INNER",
    "INWARD",
    "OUTER",
    "OUTWARD",
    "TRACKING_CONFIDENCE",
    "TRACK_GAP_SECONDS",
    "CrossingDetector",
    "_seconds_to_frames",
    "analyze",
    "calculate_net_delta",
]
sys.modules[__name__] = _implementation
