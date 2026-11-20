"""Evaluation records shared by runner and reports."""

from dataclasses import dataclass, field
from typing import Any

from ..types import Prediction

__all__ = ["EvaluationRow", "Prediction"]


@dataclass(frozen=True)
class EvaluationRow:
    video: str
    food: str
    scenario: str
    method: str
    expected_net: int
    expected_events: tuple[str, ...] | None
    predicted_net: int | None
    predicted_events: tuple[str, ...] | None
    seconds: float
    error: str | None
    details: dict[str, Any] = field(default_factory=dict)
