"""Plain Python data exchanged by analysis, inference, and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict


@dataclass(frozen=True)
class BoundingBox:
    # Pixel coordinates in Ultralytics' xyxy format.
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        # Crossing detection needs only the vertical midpoint of the box.
        return (self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class TrackedDetection:
    # One YOLO box with the tracker's persistent identity attached.
    class_name: str
    track_id: int
    frame_index: int
    box: BoundingBox
    confidence: float


@dataclass(frozen=True)
class CrossingEvent:
    # A confirmed transition from one side of the gate to the other.
    class_name: str
    track_id: int
    direction: str
    frame_index: int


@dataclass(frozen=True)
class ObservationResult:
    """State-machine details produced by one tracked detection observation."""

    observed_zone: str | None
    stable_zone_before: str | None
    stable_zone_after: str | None
    candidate_zone: str | None
    candidate_count: int
    event: CrossingEvent | None
    crossing_eligible: bool = True
    state_reset_reason: str | None = None


@dataclass(frozen=True)
class LocalFit:
    y: float
    slope: float
    residual: float


class MotionObservation(TypedDict):
    """The detector values required by motion analysis."""

    frame_index: int
    timestamp_seconds: float
    track_id: int | None
    confidence: float
    center_y_normalized: float


class MotionRow(MotionObservation, total=False):
    """An observation with optional box details and analysis diagnostics."""

    detection_index: int
    frame_width: int
    frame_height: int
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    segment_id: int
    raw_delta_y_normalized: float
    raw_delta_seconds: float
    fitted_y_normalized: float
    slope_per_second: float
    residual_normalized: float
    fit_status: str
    fitted_delta_y_normalized: float
    fitted_velocity_per_second: float
    motion: str
    event: str
    fit_reason: str
    window_observations: int
    window_span_seconds: float
    candidate_displacement_normalized: float
    candidate_duration_seconds: float
    candidate_hits: int
    candidate_direction: str
    logical_track_id: int
    tracking_status: str


class MotionEvent(TypedDict):
    track_id: int
    source_track_id: int | None
    segment_id: int
    frame_index: int
    timestamp_seconds: float
    direction: str
    displacement_normalized: float
    duration_seconds: float


class VideoMetadata(TypedDict, total=False):
    video: str
    model: str
    fps: float
    frame_count: int
    duration_seconds: float
    tracking_confidence: float


@dataclass(frozen=True)
class PlacementFrame:
    frame: Any
    frame_index: int
    height: int
    width: int
    detections: list[TrackedDetection]


@dataclass(frozen=True)
class PlacementResult:
    net_delta: int
    events: list[CrossingEvent]
    video_path: Path
    model_path: Path
    fps: float
    confidence: float
    tracking_confidence: float
    gate_y: float
    confirm_observations: int
    max_gap_frames: int
    max_evidence_gap_frames: int
    csv_path: Path | None = None
    video_debug_path: Path | None = None


@dataclass(frozen=True)
class MotionResult:
    rows: list[MotionRow]
    events: list[MotionEvent]
    metadata: VideoMetadata
    net_delta: int


@dataclass(frozen=True)
class RawDetection:
    box: BoundingBox
    confidence: float
    detection_index: int
    track_id: int | None = None


@dataclass(frozen=True)
class FrameObservations:
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    raw: tuple[RawDetection, ...]
    tracked: tuple[TrackedDetection, ...]


@dataclass(frozen=True)
class VideoObservations:
    metadata: VideoMetadata
    frames: tuple[FrameObservations, ...]

    def motion_rows(self) -> list[MotionRow]:
        """Return fresh raw rows; analysis may safely add its own diagnostics."""
        return [
            MotionRow(
                frame_index=f.frame_index,
                timestamp_seconds=f.timestamp_seconds,
                frame_width=f.width,
                frame_height=f.height,
                detection_index=d.detection_index,
                track_id=d.track_id,
                confidence=d.confidence,
                x1=d.box.x1,
                y1=d.box.y1,
                x2=d.box.x2,
                y2=d.box.y2,
                center_x=d.box.center_x,
                center_y=d.box.center_y,
                center_y_normalized=d.box.center_y / f.height,
            )
            for f in self.frames
            for d in f.raw
        ]


@dataclass(frozen=True)
class Prediction:
    net: int
    events: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)
