"""Diagnostic serialization, plotting, and streaming annotated video."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import SCRIPTS_DIR
from .crossings import GATE_HYSTERESIS
from .errors import VerificationError
from .motion import MotionConfig
from .types import (
    CrossingEvent,
    MotionEvent,
    MotionResult,
    MotionRow,
    ObservationResult,
    PlacementFrame,
    TrackedDetection,
)

DEBUG_CSV_FIELDS: tuple[str, ...] = (
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

CSV_FIELDS: tuple[str, ...] = (
    "frame_index",
    "timestamp_seconds",
    "detection_index",
    "track_id",
    "confidence",
    "frame_width",
    "frame_height",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "center_y_normalized",
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


class PlacementDiagnostics:
    """Write one frame at a time and release native/file handles on every exit."""

    def __init__(
        self, video_path: Path, debug_dir: Path | None, fps: float, gate_y: float
    ) -> None:
        self.csv_path, self.video_path = _debug_artifact_paths(video_path, debug_dir)
        self.fps = fps
        self.gate_y = gate_y
        self.handle = None
        self.video_writer = None

    def __enter__(self) -> PlacementDiagnostics:
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.csv_path.open("w", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(
                self.handle, fieldnames=DEBUG_CSV_FIELDS, extrasaction="raise"
            )
            self.writer.writeheader()
        except OSError as exc:
            self.__exit__(None, None, None)
            raise VerificationError(
                f"Could not create debug CSV {self.csv_path}: {exc}"
            ) from exc
        return self

    def write_frame(
        self,
        item: PlacementFrame,
        observations: Mapping[tuple[str, int], ObservationResult],
    ) -> None:
        import cv2

        if self.handle is None:
            raise VerificationError(
                "Open PlacementDiagnostics with a with block before writing frames."
            )
        self.writer.writerows(
            build_frame_debug_rows(
                item.frame_index,
                self.fps,
                item.height,
                item.detections,
                observations,
            )
        )
        if item.frame_index % max(1, int(round(self.fps))) == 0:
            self.handle.flush()
        if self.video_writer is None:
            self.video_writer = cv2.VideoWriter(
                str(self.video_path),
                cv2.VideoWriter.fourcc(*"mp4v"),
                self.fps,
                (item.width, item.height),
            )
            if not self.video_writer.isOpened():
                raise VerificationError(
                    f"OpenCV could not create debug video: {self.video_path}"
                )
        annotated = _draw_debug_frame(
            cv2,
            item.frame,
            item.frame_index,
            self.gate_y,
            item.detections,
            observations,
        )
        self.video_writer.write(annotated)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
        if self.handle is not None:
            self.handle.close()


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
                    "crossing_evidence"
                    if observation.crossing_eligible
                    else "tracking_only"
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
            f"{detection.class_name} #{detection.track_id} {detection.confidence:.2f}"
        )
        second_line = (
            f"obs={observation.observed_zone or '-'} "
            f"stable={observation.stable_zone_after or '-'} "
            f"cand={observation.candidate_zone or '-'}:{observation.candidate_count}"
        )
        third_line = (
            "crossing evidence" if observation.crossing_eligible else "TRACKING ONLY"
        )
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


def _debug_artifact_paths(
    video_path: Path, debug_dir: Path | None = None
) -> tuple[Path, Path]:
    script_dir = debug_dir.expanduser().resolve() if debug_dir else SCRIPTS_DIR
    return (
        script_dir / f"{video_path.stem}_tracking_debug.csv",
        script_dir / f"{video_path.stem}_annotated_debug.mp4",
    )


def plot_motion(
    rows: list[MotionRow], events: list[MotionEvent], path: Path, config: MotionConfig
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise VerificationError("Matplotlib is required for the motion plot.") from exc
    fig, (position, velocity) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, layout="constrained"
    )
    try:
        styles = {
            ("accepted", "tracked"): ("#17834c", "o", "Accepted: tracker ID present"),
            ("accepted", "untracked_used"): ("#777777", "s", "Accepted: no tracker ID"),
            ("insufficient_evidence", "tracked"): (
                "#8b46b0",
                "X",
                "Isolated tracked observation",
            ),
            ("insufficient_evidence", "untracked_used"): (
                "#c74040",
                "X",
                "Isolated untracked observation",
            ),
            ("same_frame_duplicate", "tracked"): (
                "#c46a1a",
                "D",
                "Unused same-frame tracked box",
            ),
            ("same_frame_duplicate", "untracked_used"): (
                "#c46a1a",
                "D",
                "Unused same-frame untracked box",
            ),
        }
        for (fit_status, tracking_status), (color, marker, label) in styles.items():
            selected = [
                r
                for r in rows
                if r.get("fit_status") == fit_status
                and r.get("tracking_status") == tracking_status
            ]
            if selected:
                position.scatter(
                    [r["timestamp_seconds"] for r in selected],
                    [r["center_y_normalized"] for r in selected],
                    s=30,
                    c=color,
                    marker=marker,
                    alpha=0.85,
                    label=f"{label} ({len(selected)})",
                    zorder=3,
                )
        groups: dict[int, list[MotionRow]] = {}
        for row in rows:
            if "segment_id" in row:
                groups.setdefault(row["segment_id"], []).append(row)
        for segment_id, group in groups.items():
            times = [r["timestamp_seconds"] for r in group]
            fitted = []
            for row in group:
                y = math.nan
                if row.get("fit_status") == "accepted":
                    assert "fitted_y_normalized" in row
                    y = row["fitted_y_normalized"]
                fitted.append(y)
            (line,) = position.plot(
                times,
                fitted,
                linewidth=1.8,
                label=f"Accepted fit: logical trajectory, segment {segment_id}"
                if len(groups) <= 12
                else "_nolegend_",
            )
            velocity.plot(
                times,
                [r.get("fitted_velocity_per_second", math.nan) for r in group],
                color=line.get_color(),
            )
        for event in events:
            color = "#198754" if event["direction"] == "inward" else "#be4b27"
            position.axvline(
                event["timestamp_seconds"], color=color, alpha=0.65, linestyle=":"
            )
            position.text(
                event["timestamp_seconds"],
                0.02,
                event["direction"],
                rotation=90,
                transform=position.get_xaxis_transform(),
                color=color,
                va="bottom",
            )
        position.set(
            title="Milk motion: one logical trajectory and robust rolling fits (no spatial gate)",
            ylabel="Center y / frame height",
        )
        position.invert_yaxis()
        if rows:
            position.legend(fontsize=8, loc="best")
        velocity.axhline(0, color="#555555", linewidth=0.8)
        velocity.axhline(
            config.min_speed, color="#999999", linestyle="--", linewidth=0.8
        )
        velocity.axhline(
            -config.min_speed, color="#999999", linestyle="--", linewidth=0.8
        )
        velocity.set(
            xlabel="Video time (seconds)",
            ylabel="Fitted delta y / delta t\n(frame heights / second)",
        )
        for axis in (position, velocity):
            axis.grid(alpha=0.15)
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def write_outputs(
    result: MotionResult,
    config: MotionConfig,
    output_dir: Path,
    stem: str,
    status: Callable[[str], None] = lambda message: None,
) -> None:
    rows, events = result.rows, result.events
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}_motion.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        **result.metadata,
        "config": asdict(config),
        "detection_count": len(rows),
        "events": events,
        "net_delta": result.net_delta,
        "interpretation": "Directional motion episodes; not verified fridge crossings or shelf placements.",
    }
    json_path = output_dir / f"{stem}_motion.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_path = output_dir / f"{stem}_motion.png"
    plot_motion(rows, events, plot_path, config)
    for label, path in (
        ("Coordinates CSV", csv_path),
        ("Event report", json_path),
        ("Trajectory plot", plot_path),
    ):
        status(f"{label}: {path}")
