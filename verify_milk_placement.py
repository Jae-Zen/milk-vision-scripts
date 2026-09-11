#!/usr/bin/env python3
"""Count confident milk crossings, using weaker detections only for tracking."""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


# Resolve defaults relative to this source file rather than to the caller's current
# working directory.  This keeps the default model path stable when the script is
# launched from another directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "Models" / "milk-vision-v4s.pt"
TRACKER_CONFIG = Path(__file__).resolve().with_name("milk_botsort.yaml")
TRACKING_CONFIDENCE = 0.10
CROSSING_CONFIDENCE = 0.40

# Direction names used for emitted crossing events.  Image y-coordinates increase
# downward, so movement from the top of the image toward the fridge is "inward".
INWARD = "inward"
OUTWARD = "outward"

# Zone names on the two sides of the horizontal gate.
OUTER = "outer"
INNER = "inner"

# Durations are converted to frame counts using the video's FPS.
GATE_HYSTERESIS = 0.04
CROSSING_CONFIRM_SECONDS = 0.10
TRACK_GAP_SECONDS = 0.50  # Maximum gap without any matched milk detection.
EVIDENCE_GAP_SECONDS = 1.00  # Weak matches cannot preserve gate history forever.

DEBUG_CSV_FIELDS = (
    "frame_index",
    "timestamp_seconds",
    "detection_count",
    "detection_index",
    "tracked_detection",
    "class_name",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "center_y_normalized",
    "crossing_eligible",
    "observation_use",
    "state_reset_reason",
    "observed_zone",
    "stable_zone_before",
    "stable_zone_after",
    "candidate_zone",
    "candidate_count",
    "emitted_crossing_direction",
)


class VerificationError(RuntimeError):
    """An expected input or inference failure that should be shown to the user."""


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


@dataclass
class _TrackState:
    # The last fully confirmed side of the gate.
    stable_zone: str | None = None
    # A possible new side that has not yet received enough observations.
    candidate_zone: str | None = None
    candidate_count: int = 0
    # Used to discard stale identities after a sufficiently long detection gap.
    last_seen_frame: int = -1
    last_evidence_frame: int = -1


class CrossingDetector:
    """Turn tracked center movement through a hysteresis gate into crossings."""

    def __init__(
        self,
        gate_y: float,
        hysteresis: float,
        confirm_observations: int,
        max_gap_frames: int,
        confidence: float = CROSSING_CONFIDENCE,
        max_evidence_gap_frames: int | None = None,
    ) -> None:
        # Validate the geometry before deriving the two boundaries of the gate's
        # dead band.  Both boundaries must remain inside the normalized frame.
        if not 0.0 < gate_y < 1.0:
            raise ValueError("gate_y must be between 0 and 1")
        if hysteresis <= 0.0 or gate_y - hysteresis <= 0.0 or gate_y + hysteresis >= 1.0:
            raise ValueError("hysteresis must fit on both sides of gate_y")
        if confirm_observations < 1:
            raise ValueError("confirm_observations must be positive")
        if max_gap_frames < 1:
            raise ValueError("max_gap_frames must be positive")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if max_evidence_gap_frames is None:
            max_evidence_gap_frames = 2 * max_gap_frames
        if max_evidence_gap_frames < 1:
            raise ValueError("max_evidence_gap_frames must be positive")

        # Objects above outer_limit are definitely outside; objects below
        # inner_limit are definitely inside.  The space between is deliberately
        # undecided so small position jitter does not create repeated crossings.
        self.outer_limit = gate_y - hysteresis
        self.inner_limit = gate_y + hysteresis
        self.confirm_observations = confirm_observations
        self.max_gap_frames = max_gap_frames
        self.confidence = confidence
        self.max_evidence_gap_frames = max_evidence_gap_frames
        self._states: dict[tuple[str, int], _TrackState] = {}

    def observe(self, detection: TrackedDetection, frame_height: int) -> CrossingEvent | None:
        """Observe one detection and return only a newly confirmed crossing."""
        return self.observe_detailed(detection, frame_height).event

    def observe_detailed(
        self, detection: TrackedDetection, frame_height: int
    ) -> ObservationResult:
        """Observe one detection and return its complete gate-state transition."""
        if frame_height <= 0:
            raise ValueError("frame_height must be positive")
        if detection.class_name != "milk":
            raise ValueError("Only milk detections may enter the crossing detector")
        if not 0.0 <= detection.confidence <= 1.0:
            raise ValueError("detection confidence must be between 0 and 1")

        # Normalize the pixel y-coordinate so the same gate setting works across
        # videos with different resolutions.
        center_y = detection.box.center_y / frame_height
        crossing_eligible = detection.confidence >= self.confidence
        if not crossing_eligible:
            # A weak match refreshes liveness but supplies no gate-side evidence.
            observed_zone = None
        elif center_y <= self.outer_limit:
            observed_zone = OUTER
        elif center_y >= self.inner_limit:
            observed_zone = INNER
        else:
            observed_zone = None

        # Create empty state the first time this tracked object is observed.
        key = (detection.class_name, detection.track_id)
        state = self._states.setdefault(key, _TrackState())

        # Reusing an old state after a long disappearance could invent a crossing:
        # the object might have left and a tracker ID might have been recycled.
        state_reset_reason = self._expiration_reason(state, detection.frame_index)
        if state_reset_reason:
            state = _TrackState()
            self._states[key] = state

        stable_zone_before = state.stable_zone
        state.last_seen_frame = detection.frame_index
        if crossing_eligible:
            state.last_evidence_frame = detection.frame_index

        def result(event: CrossingEvent | None = None) -> ObservationResult:
            return ObservationResult(
                observed_zone=observed_zone,
                stable_zone_before=stable_zone_before,
                stable_zone_after=state.stable_zone,
                candidate_zone=state.candidate_zone,
                candidate_count=state.candidate_count,
                event=event,
                crossing_eligible=crossing_eligible,
                state_reset_reason=state_reset_reason,
            )

        # Weak matches and confident centers in the dead band do not change the
        # stable side or candidate count. Only confident side observations count.
        if observed_zone is None:
            return result()

        # Seeing the already-confirmed side cancels any tentative transition.
        if observed_zone == state.stable_zone:
            state.candidate_zone = None
            state.candidate_count = 0
            return result()

        # Accumulate observations of a possible new stable side.  These are
        # qualifying observations, not necessarily adjacent video frames.
        if observed_zone == state.candidate_zone:
            state.candidate_count += 1
        else:
            state.candidate_zone = observed_zone
            state.candidate_count = 1

        # Do not commit to a side until it has remained convincing long enough.
        if state.candidate_count < self.confirm_observations:
            return result()

        # Promote the candidate to the stable side and clear tentative state.
        previous_zone = state.stable_zone
        state.stable_zone = observed_zone
        state.candidate_zone = None
        state.candidate_count = 0

        # The very first confirmed side has no previous side, so it initializes
        # the track without emitting an event.  Only a full side-to-side change
        # produces a direction.
        if previous_zone == OUTER and observed_zone == INNER:
            direction = INWARD
        elif previous_zone == INNER and observed_zone == OUTER:
            direction = OUTWARD
        else:
            return result()

        event = CrossingEvent(
            class_name=detection.class_name,
            track_id=detection.track_id,
            direction=direction,
            frame_index=detection.frame_index,
        )
        return result(event)

    def expire(self, frame_index: int) -> None:
        # Remove tracks that have not appeared recently.  observe() performs the
        # same protection when a stale track suddenly reappears; this method also
        # prevents the state dictionary from growing forever.
        expired = [
            key
            for key, state in self._states.items()
            if self._expiration_reason(state, frame_index)
        ]
        for key in expired:
            del self._states[key]

    def _expiration_reason(self, state: _TrackState, frame_index: int) -> str | None:
        if state.last_seen_frame >= 0 and frame_index - state.last_seen_frame > self.max_gap_frames:
            return "tracking_gap"
        if (
            state.last_evidence_frame >= 0
            and frame_index - state.last_evidence_frame > self.max_evidence_gap_frames
        ):
            return "confident_evidence_gap"
        return None


def build_frame_debug_rows(
    frame_index: int,
    fps: float,
    frame_height: int,
    detections: Sequence[TrackedDetection],
    observations: Mapping[tuple[str, int], ObservationResult],
) -> list[dict[str, object]]:
    """Convert one frame's structured diagnostics into flat CSV rows."""
    timestamp_seconds = frame_index / fps
    if not detections:
        return [
            {
                "frame_index": frame_index,
                "timestamp_seconds": round(timestamp_seconds, 6),
                "detection_count": 0,
                "detection_index": "",
            }
        ]

    rows: list[dict[str, object]] = []
    for detection_index, detection in enumerate(detections):
        key = (detection.class_name, detection.track_id)
        observation = observations[key]
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": round(timestamp_seconds, 6),
                "detection_count": len(detections),
                "detection_index": detection_index,
                "tracked_detection": repr(detection),
                "class_name": detection.class_name,
                "track_id": detection.track_id,
                "confidence": round(detection.confidence, 6),
                "x1": round(detection.box.x1, 3),
                "y1": round(detection.box.y1, 3),
                "x2": round(detection.box.x2, 3),
                "y2": round(detection.box.y2, 3),
                "center_x": round(detection.box.center_x, 3),
                "center_y": round(detection.box.center_y, 3),
                "center_y_normalized": round(detection.box.center_y / frame_height, 6),
                "crossing_eligible": observation.crossing_eligible,
                "observation_use": (
                    "crossing_evidence" if observation.crossing_eligible else "tracking_only"
                ),
                "state_reset_reason": observation.state_reset_reason or "",
                "observed_zone": observation.observed_zone or "",
                "stable_zone_before": observation.stable_zone_before or "",
                "stable_zone_after": observation.stable_zone_after or "",
                "candidate_zone": observation.candidate_zone or "",
                "candidate_count": observation.candidate_count,
                "emitted_crossing_direction": (
                    observation.event.direction if observation.event else ""
                ),
            }
        )
    return rows


def _draw_debug_frame(
    cv2: Any,
    frame: Any,
    frame_index: int,
    gate_y: float,
    detections: Sequence[TrackedDetection],
    observations: Mapping[tuple[str, int], ObservationResult],
) -> Any:
    """Return a copy of the frame annotated with tracking and gate diagnostics."""
    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]

    gate_lines = (
        (gate_y - GATE_HYSTERESIS, (0, 180, 255), "outer limit"),
        (gate_y, (255, 255, 255), "gate"),
        (gate_y + GATE_HYSTERESIS, (255, 180, 0), "inner limit"),
    )
    for normalized_y, color, label in gate_lines:
        y = int(round(normalized_y * frame_height))
        cv2.line(annotated, (0, y), (frame_width - 1, y), color, 2)
        cv2.putText(
            annotated,
            label,
            (8, max(16, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    emitted_events: list[CrossingEvent] = []
    for detection in detections:
        key = (detection.class_name, detection.track_id)
        observation = observations[key]
        color = (70, 220, 70) if observation.crossing_eligible else (0, 180, 255)
        x1, y1, x2, y2 = (
            int(round(detection.box.x1)),
            int(round(detection.box.y1)),
            int(round(detection.box.x2)),
            int(round(detection.box.y2)),
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        first_line = (
            f"{detection.class_name} #{detection.track_id} "
            f"{detection.confidence:.2f}"
        )
        second_line = (
            f"obs={observation.observed_zone or '-'} "
            f"stable={observation.stable_zone_after or '-'} "
            f"cand={observation.candidate_zone or '-'}:{observation.candidate_count}"
        )
        third_line = "crossing evidence" if observation.crossing_eligible else "TRACKING ONLY"
        if observation.state_reset_reason:
            third_line += f" reset={observation.state_reset_reason}"

        text_y = max(18, y1 - 38)
        for line_index, text in enumerate((first_line, second_line, third_line)):
            cv2.putText(
                annotated,
                text,
                (max(0, x1), text_y + line_index * 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        if observation.event is not None:
            emitted_events.append(observation.event)

    cv2.putText(
        annotated,
        f"frame={frame_index} detections={len(detections)}",
        (8, frame_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for event_index, event in enumerate(emitted_events):
        cv2.putText(
            annotated,
            f"EVENT {event.class_name} #{event.track_id} {event.direction}",
            (8, 24 + event_index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _debug_artifact_paths(video_path: Path, debug_dir: Path | None = None) -> tuple[Path, Path]:
    script_dir = debug_dir.expanduser().resolve() if debug_dir else Path(__file__).resolve().parent
    return (
        script_dir / f"{video_path.stem}_tracking_debug.csv",
        script_dir / f"{video_path.stem}_annotated_debug.mp4",
    )


def print_crossing_events(
    *,
    stream: TextIO,
    milk_events: Sequence[CrossingEvent],
    label: str = "MILK EVENTS",
) -> None:
    """List confirmed crossings with their track IDs, directions, and frame indices."""
    print(f"{label} ({len(milk_events)})", file=stream)
    if not milk_events:
        print("  []", file=stream)
    for event in sorted(milk_events, key=lambda event: event.frame_index):
        print(f"  {event!r}", file=stream)


def print_debug_summary(
    *,
    stream: TextIO,
    video_path: Path,
    model_path: Path,
    fps: float,
    confidence: float,
    tracking_confidence: float,
    gate_y: float,
    confirm_observations: int,
    max_gap_frames: int,
    max_evidence_gap_frames: int,
    milk_events: Sequence[CrossingEvent],
    net_delta: int,
    csv_path: Path,
    video_debug_path: Path,
) -> None:
    """Write deterministic, human-readable run diagnostics to stderr."""
    print("DEBUG CONFIGURATION", file=stream)
    print(f"  video={video_path}", file=stream)
    print(f"  model={model_path}", file=stream)
    print(f"  tracker={TRACKER_CONFIG.name} fps={fps:.6f}", file=stream)
    print(
        f"  tracking_confidence={tracking_confidence:.3f} crossing_confidence={confidence:.3f}",
        file=stream,
    )
    print(
        f"  gate_y={gate_y:.3f} outer_limit={gate_y - GATE_HYSTERESIS:.3f} "
        f"inner_limit={gate_y + GATE_HYSTERESIS:.3f}",
        file=stream,
    )
    print(
        f"  confirm_observations={confirm_observations} "
        f"max_gap_frames={max_gap_frames} max_evidence_gap_frames={max_evidence_gap_frames}",
        file=stream,
    )

    print_crossing_events(stream=stream, milk_events=milk_events, label="DEBUG MILK EVENTS")

    print(f"DEBUG RESULT net_delta={net_delta}", file=stream)
    print(f"DEBUG CSV {csv_path}", file=stream)
    print(f"DEBUG VIDEO {video_debug_path}", file=stream)


def calculate_net_delta(milk_crossings: Iterable[CrossingEvent]) -> int:
    """Sum milk crossings directly, including an initial outward or final inward."""
    net_delta = 0
    for crossing in milk_crossings:
        if crossing.class_name != "milk":
            raise ValueError("Only milk crossing events may affect net_delta")
        if crossing.direction == INWARD:
            net_delta += 1
        elif crossing.direction == OUTWARD:
            net_delta -= 1
        else:
            raise ValueError(f"Unknown crossing direction: {crossing.direction}")
    return net_delta


def resolve_milk_class_id(names: Mapping[int, str]) -> int:
    """Require only a milk class; other checkpoint classes are unused."""
    # YOLO.names normalizes checkpoint class names to an {id: name} dictionary.
    normalized = {
        str(class_name).strip().casefold(): int(class_id)
        for class_id, class_name in names.items()
    }

    if "milk" not in normalized:
        raise VerificationError("Model must contain a class named 'milk'.")
    return normalized["milk"]


def _seconds_to_frames(seconds: float, fps: float, minimum: int = 1) -> int:
    # round() chooses the nearest whole frame; minimum prevents a positive timing
    # rule from disappearing entirely at low frame rates.
    return max(minimum, int(round(seconds * fps)))


def _extract_tracked_detections(
    result: Any,
    frame_index: int,
    milk_class_id: int,
) -> list[TrackedDetection]:
    """
        returns a list of TrackedDetection objects (each item detected in the frame)
    """
    boxes = result.boxes
    # Detections without tracker IDs cannot be followed across the gate, even if
    # YOLO produced a valid class and bounding box for them.
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []

    # Move tensor data to the CPU and turn it into ordinary Python values.
    track_ids = boxes.id.int().cpu().tolist()
    class_ids = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    coordinates = boxes.xyxy.cpu().tolist()
    detections: list[TrackedDetection] = []

    # strict=True treats inconsistent tensor lengths as an error instead of
    # silently dropping unmatched values.
    for track_id, class_id, detection_confidence, coordinates_for_box in zip(
        track_ids, class_ids, confidences, coordinates, strict=True
    ):
        # Keep weak milk matches for liveness; confidence gating happens later.
        if class_id != milk_class_id:
            continue

        # Store plain floats so downstream logic is independent of PyTorch.
        x1, y1, x2, y2 = (float(value) for value in coordinates_for_box)
        detections.append(
            TrackedDetection(
                class_name="milk",
                track_id=int(track_id),
                frame_index=frame_index,
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=float(detection_confidence),
            )
        )
    return detections


def verify_video(
    video_path: Path,
    model_path: Path,
    confidence: float = CROSSING_CONFIDENCE,
    gate_y: float = 0.27,
    device: str | None = None,
    debug: bool = False,
    tracking_confidence: float = TRACKING_CONFIDENCE,
    debug_dir: Path | None = None,
) -> int:
    """Run tracking and return the event-derived net milk change."""
    # Canonicalize paths before validating or reporting them.
    video_path = video_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()

    # Reject expected input problems with VerificationError so main() can report
    # them cleanly without placing anything on stdout.
    if not video_path.is_file():
        raise VerificationError(f"Video file does not exist: {video_path}")
    if not model_path.is_file():
        raise VerificationError(f"Model file does not exist: {model_path}")
    if not 0.0 <= confidence <= 1.0:
        raise VerificationError("Confidence must be between 0 and 1.")
    if not 0.0 < tracking_confidence <= confidence:
        raise VerificationError("Tracking confidence must be positive and no greater than crossing confidence.")
    if not 0.05 <= gate_y <= 0.95:
        raise VerificationError("Gate position must be between 0.05 and 0.95.")
    if not TRACKER_CONFIG.is_file():
        raise VerificationError(f"BoT-SORT configuration does not exist: {TRACKER_CONFIG}")

    # Delay heavyweight optional imports until verification is actually requested.
    try:
        import cv2  # type: ignore
        from ultralytics import YOLO
    except ImportError as exc:
        raise VerificationError(
            "OpenCV and Ultralytics are required. Install Scripts/requirements.txt."
        ) from exc

    # OpenCV supplies decoded frames and the FPS metadata used by every timing
    # threshold in the state machine.
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VerificationError(f"OpenCV could not open the video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise VerificationError("The video has invalid FPS metadata.")

    # Discover the milk class by name, without requiring any other model class.
    try:
        model = YOLO(str(model_path), task="segment")
        milk_class_id = resolve_milk_class_id(model.names)
    except VerificationError:
        capture.release()
        raise
    except Exception as exc:
        capture.release()
        raise VerificationError(f"Could not load the YOLO model: {exc}") from exc

    # Convert user-independent timing rules once so frame processing uses integer
    # comparisons.  Crossing confirmation deliberately requires at least 2 hits.
    confirm_observations = _seconds_to_frames(CROSSING_CONFIRM_SECONDS, fps, minimum=2)
    max_gap_frames = _seconds_to_frames(TRACK_GAP_SECONDS, fps)
    max_evidence_gap_frames = _seconds_to_frames(EVIDENCE_GAP_SECONDS, fps)
    crossing_detector = CrossingDetector(
        gate_y=gate_y,
        hysteresis=GATE_HYSTERESIS,
        confirm_observations=confirm_observations,
        max_gap_frames=max_gap_frames,
        confidence=confidence,
        max_evidence_gap_frames=max_evidence_gap_frames,
    )

    # Every confident milk crossing changes inventory independently.
    milk_events: list[CrossingEvent] = []
    frame_index = -1
    saw_frame = False

    debug_csv_path: Path | None = None
    debug_video_path: Path | None = None
    debug_csv_handle: TextIO | None = None
    debug_csv_writer: csv.DictWriter | None = None
    debug_video_writer: Any = None
    if debug:
        debug_csv_path, debug_video_path = _debug_artifact_paths(video_path, debug_dir)
        try:
            debug_csv_path.parent.mkdir(parents=True, exist_ok=True)
            debug_csv_handle = debug_csv_path.open("w", encoding="utf-8", newline="")
            debug_csv_writer = csv.DictWriter(
                debug_csv_handle, fieldnames=DEBUG_CSV_FIELDS, extrasaction="raise"
            )
            debug_csv_writer.writeheader()
        except OSError as exc:
            capture.release()
            if debug_csv_handle is not None:
                debug_csv_handle.close()
            raise VerificationError(f"Could not create debug CSV {debug_csv_path}: {exc}") from exc

    # persist=True is what allows BoT-SORT to retain identities across these
    # separate, one-frame model.track() calls.
    track_arguments: dict[str, Any] = {
        "persist": True,
        "tracker": str(TRACKER_CONFIG),
        "conf": tracking_confidence,
        "classes": [milk_class_id],
        "verbose": False,
    }
    if device is not None:
        track_arguments["device"] = device

    try:
        while True:
            ok, frame = capture.read()
            # OpenCV uses the same false result for normal EOF and decode failure.
            if not ok:
                break

            saw_frame = True
            frame_index += 1
            frame_height, frame_width = frame.shape[:2]
            if frame_height <= 0 or frame_width <= 0:
                raise VerificationError(f"Decoded frame {frame_index} has invalid dimensions.")

            # Ultralytics normally returns one result because one frame was passed.
            try:
                result = model.track(frame, **track_arguments)[0]
            except Exception as exc:
                raise VerificationError(
                    f"YOLO tracking failed at frame {frame_index}: {exc}"
                ) from exc

            detections = _extract_tracked_detections(
                result=result,
                frame_index=frame_index,
                milk_class_id=milk_class_id,
            )

            # Weak matches refresh track liveness only. The detector itself gates
            # all zone updates and event confirmation on crossing confidence.
            observations: dict[tuple[str, int], ObservationResult] = {}
            for detection in detections:
                observation = crossing_detector.observe_detailed(detection, frame_height)
                observations[(detection.class_name, detection.track_id)] = observation
                crossing = observation.event
                if crossing is None:
                    continue
                milk_events.append(crossing)

            if debug:
                assert debug_csv_writer is not None
                assert debug_csv_handle is not None
                assert debug_video_path is not None
                for row in build_frame_debug_rows(
                    frame_index=frame_index,
                    fps=fps,
                    frame_height=frame_height,
                    detections=detections,
                    observations=observations,
                ):
                    debug_csv_writer.writerow(row)
                if frame_index % max(1, int(round(fps))) == 0:
                    debug_csv_handle.flush()

                if debug_video_writer is None:
                    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                    debug_video_writer = cv2.VideoWriter(
                        str(debug_video_path),
                        fourcc,
                        fps,
                        (frame_width, frame_height),
                    )
                    if not debug_video_writer.isOpened():
                        debug_video_writer.release()
                        debug_video_writer = None
                        raise VerificationError(
                            f"OpenCV could not create debug video: {debug_video_path}"
                        )

                annotated = _draw_debug_frame(
                    cv2=cv2,
                    frame=frame,
                    frame_index=frame_index,
                    gate_y=gate_y,
                    detections=detections,
                    observations=observations,
                )
                debug_video_writer.write(annotated)

            # Discard state belonging to tracks absent beyond the allowed gap.
            crossing_detector.expire(frame_index)
    finally:
        # Release the native video handle on success and on every loop exception.
        capture.release()
        if debug_video_writer is not None:
            debug_video_writer.release()
        if debug_csv_handle is not None:
            debug_csv_handle.close()

    if not saw_frame:
        raise VerificationError(f"No frames could be decoded from the video: {video_path}")

    net_delta = calculate_net_delta(milk_events)

    if debug:
        assert debug_csv_path is not None
        assert debug_video_path is not None
        print_debug_summary(
            stream=sys.stderr,
            video_path=video_path,
            model_path=model_path,
            fps=fps,
            confidence=confidence,
            tracking_confidence=tracking_confidence,
            gate_y=gate_y,
            confirm_observations=confirm_observations,
            max_gap_frames=max_gap_frames,
            max_evidence_gap_frames=max_evidence_gap_frames,
            milk_events=milk_events,
            net_delta=net_delta,
            csv_path=debug_csv_path,
            video_debug_path=debug_video_path,
        )
    else:
        print_crossing_events(stream=sys.stderr, milk_events=milk_events)
    return net_delta


def _bounded_float(value: str, minimum: float, maximum: float, label: str) -> float:
    # argparse calls this adapter for --conf and --gate-y so malformed and
    # non-finite values receive normal command-line usage errors.
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be a number") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(
            f"{label} must be between {minimum:g} and {maximum:g}"
        )
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # Supplying argv makes the CLI independently testable; None means use sys.argv.
    parser = argparse.ArgumentParser(
        description="Print the signed net milk change found in a complete video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Input video path"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"YOLO model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--conf",
        type=lambda value: _bounded_float(value, 0.0, 1.0, "confidence"),
        default=CROSSING_CONFIDENCE,
        help="Minimum milk confidence for crossing evidence (default: 0.40)",
    )
    parser.add_argument(
        "--track-conf",
        type=lambda value: _bounded_float(value, 0.0, 1.0, "tracking confidence"),
        default=TRACKING_CONFIDENCE,
        help="YOLO threshold feeding BoT-SORT; must be positive and <= --conf (default: 0.10)",
    )
    parser.add_argument(
        "--gate-y",
        type=lambda value: _bounded_float(value, 0.05, 0.95, "gate position"),
        default=0.27,
        help="Normalized vertical top-gate position (default: 0.27)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, such as 0 or cpu (default: automatic)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write per-frame CSV/video diagnostics beside this script",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Optional output directory for --debug (default: beside this script)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # Third-party model code occasionally prints informational messages. Keep
        # stdout reserved for the promised single integer result.
        with contextlib.redirect_stdout(sys.stderr):
            net_delta = verify_video(
                video_path=args.video,
                model_path=args.model,
                confidence=args.conf,
                gate_y=args.gate_y,
                device=args.device,
                debug=args.debug,
                tracking_confidence=args.track_conf,
                debug_dir=args.debug_dir,
            )
    except VerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # This is the only successful stdout output, making the command easy for
    # another program to consume.
    print(net_delta)
    return 0


if __name__ == "__main__":
    # Convert main()'s integer return value into the process exit status.
    raise SystemExit(main())
