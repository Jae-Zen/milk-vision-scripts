"""Pure trajectory fitting and directional-event analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Sequence

from .types import LocalFit, MotionEvent, MotionRow

ANALYSIS_FIELDS = (
    "segment_id",
    "raw_delta_y_normalized",
    "raw_delta_seconds",
    "fitted_y_normalized",
    "slope_per_second",
    "residual_normalized",
    "fit_status",
    "fitted_delta_y_normalized",
    "fitted_velocity_per_second",
    "motion",
    "event",
    "fit_reason",
    "window_observations",
    "window_span_seconds",
    "candidate_displacement_normalized",
    "candidate_duration_seconds",
    "candidate_hits",
    "candidate_direction",
    "logical_track_id",
    "tracking_status",
)


@dataclass(frozen=True)
class MotionConfig:
    window_seconds: float = 0.60
    max_gap_seconds: float = 0.50
    min_speed: float = 0.04
    min_displacement: float = 0.08
    inward: str = "down"

    def validate(self) -> None:
        for name in (
            "window_seconds",
            "max_gap_seconds",
            "min_speed",
            "min_displacement",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.inward not in ("down", "up"):
            raise ValueError("inward must be down or up")


def robust_local_fit(
    times: Sequence[float], ys: Sequence[float], at: float, observed_y: float
) -> LocalFit | None:
    """Median pairwise slope + median intercept (univariate Theil-Sen).

    Center times for numerical stability. Bound work to 61 uniformly sampled
    points. The fit is descriptive: no inlier or residual threshold rejects it.
    """
    if len(times) != len(ys):
        raise ValueError("times and ys must have equal lengths")
    # Two time-distinct observations are the mathematical minimum for a slope.
    # There are deliberately no observation-count, time-span, inlier, or
    # residual acceptance thresholds.
    if len(times) < 2:
        return None
    indices = sorted({round(i * (len(times) - 1) / 60) for i in range(61)})
    slopes = [
        (ys[j] - ys[i]) / (times[j] - times[i])
        for k, i in enumerate(indices)
        for j in indices[k + 1 :]
        if times[j] > times[i]
    ]
    if not slopes:
        return None
    slope = median(slopes)
    y = median(
        value - slope * (time - at) for time, value in zip(times, ys, strict=True)
    )
    residual = observed_y - y
    return LocalFit(y, slope, residual)


def infer_events(rows: list[MotionRow], config: MotionConfig) -> list[MotionEvent]:
    """Confirm net directional movement from accepted fitted coordinates.

    Neutral steps preserve a pending candidate, while an opposite moving step
    replaces an unfinished candidate. A counted direction stays locked until a
    complete opposite event is emitted within the same segment.
    """
    events: list[MotionEvent] = []
    previous = None
    candidate = 0
    candidate_start_time = 0.0
    candidate_start_y = 0.0
    hits = 0
    last_emitted_sign = 0
    for row in rows:
        if row.get("fit_status") != "accepted":
            continue
        if previous is None:
            previous = row
            continue
        dt = row["timestamp_seconds"] - previous["timestamp_seconds"]
        if dt > config.max_gap_seconds:
            candidate, hits = 0, 0
            previous = row
            continue
        # Accepted fits always have coordinates; raw observations may not.
        assert "fitted_y_normalized" in row
        assert "fitted_y_normalized" in previous
        delta = row["fitted_y_normalized"] - previous["fitted_y_normalized"]
        velocity = delta / dt
        row["fitted_delta_y_normalized"] = delta
        row["fitted_velocity_per_second"] = velocity
        sign = (
            1
            if velocity >= config.min_speed
            else -1
            if velocity <= -config.min_speed
            else 0
        )
        row["motion"] = (
            "down" if sign > 0 else "up" if sign < 0 else "stationary_or_uncertain"
        )
        if sign != 0:
            if sign != candidate:
                candidate, hits = sign, 0
                candidate_start_time = previous["timestamp_seconds"]
                candidate_start_y = previous["fitted_y_normalized"]
            hits += 1
            # Net displacement from the candidate anchor avoids inflating travel
            # by summing coordinate fluctuations. Neutral steps preserve state.
            displacement = row["fitted_y_normalized"] - candidate_start_y
            duration = row["timestamp_seconds"] - candidate_start_time
            row.update(
                candidate_displacement_normalized=displacement,
                candidate_duration_seconds=duration,
                candidate_hits=hits,
                candidate_direction="down" if candidate > 0 else "up",
            )
            if (
                sign != last_emitted_sign
                and sign * displacement >= config.min_displacement
            ):
                direction = (
                    "inward" if (sign > 0) == (config.inward == "down") else "outward"
                )
                assert "logical_track_id" in row and "segment_id" in row
                event: MotionEvent = {
                    "track_id": row["logical_track_id"],
                    "source_track_id": row["track_id"],
                    "segment_id": row["segment_id"],
                    "frame_index": row["frame_index"],
                    "timestamp_seconds": row["timestamp_seconds"],
                    "direction": direction,
                    "displacement_normalized": displacement,
                    "duration_seconds": duration,
                }
                events.append(event)
                row["event"] = direction
                last_emitted_sign = sign
        elif candidate:
            # Record that an existing candidate survived this neutral observation.
            displacement = row["fitted_y_normalized"] - candidate_start_y
            row.update(
                candidate_displacement_normalized=displacement,
                candidate_duration_seconds=row["timestamp_seconds"]
                - candidate_start_time,
                candidate_hits=hits,
                candidate_direction="down" if candidate > 0 else "up",
            )
        previous = row
    return events


def analyze_detections(
    rows: list[MotionRow], config: MotionConfig
) -> list[MotionEvent]:
    """Analyze every detector result as one trajectory, split at evidence gaps.

    Tracker IDs remain diagnostic only. When a frame has multiple milk boxes,
    the highest-confidence box supplies its single trajectory coordinate.
    """
    config.validate()
    for row in rows:
        # Re-analysis must not retain previous fits, events or diagnostics.
        for field in ANALYSIS_FIELDS:
            row.pop(field, None)
        row["logical_track_id"] = 0
        row["tracking_status"] = (
            "tracked" if row["track_id"] is not None else "untracked_used"
        )
    frame_groups: dict[int, list[MotionRow]] = {}
    for row in rows:
        frame_groups.setdefault(row["frame_index"], []).append(row)
    trajectory = []
    for frame_index in sorted(frame_groups):
        candidates = frame_groups[frame_index]
        chosen = max(
            candidates,
            key=lambda row: (row["confidence"], -row.get("detection_index", 0)),
        )
        trajectory.append(chosen)
        for row in candidates:
            if row is not chosen:
                row["fit_status"] = "same_frame_duplicate"
                row["fit_reason"] = "lower_confidence_box_at_same_timestamp"
    segments: list[list[MotionRow]] = []
    for row in trajectory:
        if (
            not segments
            or row["timestamp_seconds"] - segments[-1][-1]["timestamp_seconds"]
            > config.max_gap_seconds
        ):
            segments.append([])
        if segments[-1]:
            previous = segments[-1][-1]
            if row["timestamp_seconds"] <= previous["timestamp_seconds"]:
                raise ValueError("Logical trajectory timestamps must increase")
            row["raw_delta_y_normalized"] = (
                row["center_y_normalized"] - previous["center_y_normalized"]
            )
            row["raw_delta_seconds"] = (
                row["timestamp_seconds"] - previous["timestamp_seconds"]
            )
        segments[-1].append(row)
    events = []
    for segment_id, segment in enumerate(segments):
        left, right = 0, 0
        half_window = config.window_seconds / 2
        for row in segment:
            row["segment_id"] = segment_id
            time = row["timestamp_seconds"]
            while (
                left < len(segment)
                and segment[left]["timestamp_seconds"] < time - half_window
            ):
                left += 1
            while (
                right < len(segment)
                and segment[right]["timestamp_seconds"] <= time + half_window
            ):
                right += 1
            window = segment[left:right]
            span = window[-1]["timestamp_seconds"] - window[0]["timestamp_seconds"]
            row.update(window_observations=len(window), window_span_seconds=span)
            fit = robust_local_fit(
                [r["timestamp_seconds"] for r in window],
                [r["center_y_normalized"] for r in window],
                time,
                row["center_y_normalized"],
            )
            row["fit_status"] = "insufficient_evidence"
            row["fit_reason"] = "one_observation_cannot_define_slope"
            if fit is not None:
                row.update(
                    fitted_y_normalized=fit.y,
                    slope_per_second=fit.slope,
                    residual_normalized=fit.residual,
                    fit_reason="accepted",
                    fit_status="accepted",
                )
        events.extend(infer_events(segment, config))
    return sorted(events, key=lambda event: event["frame_index"])
