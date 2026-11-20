"""Pure gate-crossing analysis, independent of video and output libraries."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..types import (
    CrossingEvent,
    FrameObservations,
    ObservationResult,
    Prediction,
    TrackedDetection,
    VideoObservations,
)

TRACKING_CONFIDENCE = 0.10
CROSSING_CONFIDENCE = 0.40
INWARD = "inward"
OUTWARD = "outward"
OUTER = "outer"
INNER = "inner"
GATE_HYSTERESIS = 0.04
CROSSING_CONFIRM_SECONDS = 0.10
TRACK_GAP_SECONDS = 0.50  # Maximum gap without any matched milk detection.
EVIDENCE_GAP_SECONDS = 1.00  # Weak matches cannot preserve gate history forever.


def analyze(
    observations: VideoObservations,
    confidence: float = CROSSING_CONFIDENCE,
    gate_y: float = 0.27,
    *,
    on_frame: Callable[[FrameObservations, dict], None] | None = None,
) -> Prediction:
    """Replay tracked boxes, including empty frames, through the crossing rules."""
    if not 0 <= confidence <= 1 or not 0.05 <= gate_y <= 0.95:
        raise ValueError("Invalid placement confidence or gate position")
    fps = float(observations.metadata.get("fps", 0))
    detector = CrossingDetector(
        gate_y,
        GATE_HYSTERESIS,
        _seconds_to_frames(CROSSING_CONFIRM_SECONDS, fps, minimum=2),
        _seconds_to_frames(TRACK_GAP_SECONDS, fps),
        confidence,
        _seconds_to_frames(EVIDENCE_GAP_SECONDS, fps),
    )
    events = []
    for frame in observations.frames:
        details = {}
        for detection in frame.tracked:
            item = detector.observe_detailed(detection, frame.height)
            details[(detection.class_name, detection.track_id)] = item
            if item.event is not None:
                events.append(item.event)
        if on_frame is not None:
            on_frame(frame, details)
        detector.expire(frame.frame_index)
    from dataclasses import asdict

    return Prediction(
        calculate_net_delta(events),
        tuple("in" if e.direction == "inward" else "out" for e in events),
        {"events": [asdict(e) for e in events]},
    )


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
        if (
            hysteresis <= 0.0
            or gate_y - hysteresis <= 0.0
            or gate_y + hysteresis >= 1.0
        ):
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

    def observe(
        self, detection: TrackedDetection, frame_height: int
    ) -> CrossingEvent | None:
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
        if (
            state.last_seen_frame >= 0
            and frame_index - state.last_seen_frame > self.max_gap_frames
        ):
            return "tracking_gap"
        if (
            state.last_evidence_frame >= 0
            and frame_index - state.last_evidence_frame > self.max_evidence_gap_frames
        ):
            return "confident_evidence_gap"
        return None


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


def _seconds_to_frames(seconds: float, fps: float, minimum: int = 1) -> int:
    # round() chooses the nearest whole frame; minimum prevents a positive timing
    # rule from disappearing entirely at low frame rates.
    return max(minimum, round(seconds * fps))
