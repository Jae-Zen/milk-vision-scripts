#!/usr/bin/env python3
"""Experimental boundary-free milk direction estimation from robust local fits.

This is an offline motion estimator, not proof of a fridge crossing or placement.
With --debug, all post-NMS milk detections are saved, including boxes without a confirmed ID.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import median
import sys
import time
from typing import Any, Sequence

from verify_milk_placement import PROJECT_ROOT, TRACKER_CONFIG, VerificationError, resolve_milk_class_id


DEFAULT_MODEL = PROJECT_ROOT / "Models" / "milk-vision-v4n.pt"


def status(message: str) -> None:
    """Flush status immediately, including when launched with redirected output."""
    print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class MotionConfig:
    confidence: float = 0.40
    window_seconds: float = 0.60
    adaptive_window: bool = True
    max_gap_seconds: float = 0.50
    min_observations: int = 5
    min_fit_seconds: float = 0.12
    residual_threshold: float = 0.025
    min_inlier_fraction: float = 0.65
    min_speed: float = 0.04
    min_displacement: float = 0.08
    confirm_seconds: float = 0.15
    inward: str = "down"

    def validate(self) -> None:
        for name in ("window_seconds", "max_gap_seconds", "min_fit_seconds",
                     "residual_threshold", "min_speed", "min_displacement", "confirm_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.confidence <= 1 or not 0 < self.min_inlier_fraction <= 1:
            raise ValueError("confidence and min_inlier_fraction must be in (0, 1]")
        if self.min_observations < 3:
            raise ValueError("min_observations must be at least 3")
        if self.min_fit_seconds > self.window_seconds:
            raise ValueError("min_fit_seconds must not exceed window_seconds")
        if self.inward not in ("down", "up"):
            raise ValueError("inward must be down or up")


@dataclass(frozen=True)
class LocalFit:
    y: float
    slope: float
    inlier_fraction: float
    residual: float
    accepted: bool
    reason: str
    inlier_count: int
    inlier_span_seconds: float


def robust_local_fit(times: Sequence[float], ys: Sequence[float], at: float,
                     observed_y: float, config: MotionConfig) -> LocalFit | None:
    """Median pairwise slope + median intercept (univariate Theil-Sen).

    Center times for numerical stability. Bound work to 61 uniformly sampled
    points; score residuals against every observation in the local window.
    """
    if len(times) != len(ys):
        raise ValueError("times and ys must have equal lengths")
    if len(times) < config.min_observations or times[-1] - times[0] < config.min_fit_seconds:
        return None
    indices = sorted({round(i * (len(times) - 1) / 60) for i in range(61)})
    slopes = [(ys[j] - ys[i]) / (times[j] - times[i])
              for k, i in enumerate(indices) for j in indices[k + 1:]
              if times[j] > times[i]]
    if not slopes:
        return None
    slope = median(slopes)
    y = median([value - slope * (time - at) for time, value in zip(times, ys)])
    residuals = [abs(value - (y + slope * (time - at))) for time, value in zip(times, ys)]
    inliers = [i for i, residual in enumerate(residuals) if residual <= config.residual_threshold]
    fraction = len(inliers) / len(times)
    residual = observed_y - y
    # Require temporal support from inliers, not merely the outlier endpoints.
    span = times[inliers[-1]] - times[inliers[0]] if inliers else 0.0
    failures = []
    if len(inliers) < config.min_observations:
        failures.append("too_few_inliers")
    if fraction < config.min_inlier_fraction:
        failures.append("low_inlier_fraction")
    if span < config.min_fit_seconds:
        failures.append("short_inlier_span")
    if abs(residual) > config.residual_threshold:
        failures.append("observation_outlier")
    return LocalFit(y, slope, fraction, residual, not failures,
                    ";".join(failures) or "accepted", len(inliers), span)


def supported_window(segment: list[dict[str, Any]], left: int, right: int,
                     at: float, config: MotionConfig) -> tuple[list[dict[str, Any]], bool]:
    """Extend sparse centered support toward nearest observations, within full width.

    Only insufficient count/span triggers adaptation, never a rejected regression.
    Preserve segment boundaries and the same maximum total window duration.
    """
    initial = (left, right)
    if config.adaptive_window:
        while (right - left < config.min_observations or
               segment[right - 1]["timestamp_seconds"] - segment[left]["timestamp_seconds"] < config.min_fit_seconds):
            choices = []
            if left > 0 and segment[right - 1]["timestamp_seconds"] - segment[left - 1]["timestamp_seconds"] <= config.window_seconds:
                choices.append((abs(segment[left - 1]["timestamp_seconds"] - at), left - 1, right))
            if right < len(segment) and segment[right]["timestamp_seconds"] - segment[left]["timestamp_seconds"] <= config.window_seconds:
                choices.append((abs(segment[right]["timestamp_seconds"] - at), left, right + 1))
            if not choices:
                break
            _, left, right = min(choices)
    return segment[left:right], initial != (left, right)


def infer_events(rows: list[dict[str, Any]], config: MotionConfig) -> list[dict[str, Any]]:
    """Confirm net directional movement from accepted fitted coordinates.

    Neutral steps preserve a pending candidate, while an opposite moving step
    replaces an unfinished candidate. A counted direction stays locked until a
    complete opposite event is emitted within the same segment.
    """
    events: list[dict[str, Any]] = []
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
        delta = row["fitted_y_normalized"] - previous["fitted_y_normalized"]
        velocity = delta / dt
        row["fitted_delta_y_normalized"] = delta
        row["fitted_velocity_per_second"] = velocity
        # Fit slope corroborates the finite difference. Moving window membership
        # can otherwise move the fitted point despite little underlying motion.
        sign = 1 if velocity >= config.min_speed else -1 if velocity <= -config.min_speed else 0
        if sign * row["slope_per_second"] < config.min_speed:
            sign = 0
        row["motion"] = "down" if sign > 0 else "up" if sign < 0 else "stationary_or_uncertain"
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
            row.update(candidate_displacement_normalized=displacement,
                       candidate_duration_seconds=duration, candidate_hits=hits,
                       candidate_direction="down" if candidate > 0 else "up")
            if (sign != last_emitted_sign and hits >= 3
                    and sign * displacement >= config.min_displacement
                    and duration >= config.confirm_seconds):
                direction = "inward" if (sign > 0) == (config.inward == "down") else "outward"
                event = {"track_id": row["track_id"], "segment_id": row["segment_id"],
                         "frame_index": row["frame_index"], "timestamp_seconds": row["timestamp_seconds"],
                         "direction": direction, "displacement_normalized": displacement,
                         "duration_seconds": duration}
                events.append(event)
                row["event"] = direction
                last_emitted_sign = sign
        elif candidate:
            # Record that an existing candidate survived this neutral observation.
            displacement = row["fitted_y_normalized"] - candidate_start_y
            row.update(candidate_displacement_normalized=displacement,
                       candidate_duration_seconds=row["timestamp_seconds"] - candidate_start_time,
                       candidate_hits=hits,
                       candidate_direction="down" if candidate > 0 else "up")
        previous = row
    return events


def analyze_detections(rows: list[dict[str, Any]], config: MotionConfig) -> list[dict[str, Any]]:
    """Analyze each confident track separately, splitting at evidence gaps."""
    config.validate()
    tracks: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        # Re-analysis must not retain previous fits, events or diagnostics.
        for field in CSV_FIELDS[14:]:
            row.pop(field, None)
        if row["track_id"] is None:
            row["fit_status"] = "untracked"
            row["fit_reason"] = "no_confirmed_track_id"
        elif row["confidence"] < config.confidence:
            row["fit_status"] = "low_confidence"
            row["fit_reason"] = "below_confidence_threshold"
        else:
            tracks.setdefault(row["track_id"], []).append(row)
    events = []
    for track_rows in tracks.values():
        track_rows.sort(key=lambda row: row["frame_index"])
        segments: list[list[dict[str, Any]]] = []
        for row in track_rows:
            if not segments or row["timestamp_seconds"] - segments[-1][-1]["timestamp_seconds"] > config.max_gap_seconds:
                segments.append([])
            if segments[-1]:
                previous = segments[-1][-1]
                if row["timestamp_seconds"] <= previous["timestamp_seconds"]:
                    raise ValueError("Each track must have at most one detection per frame")
                row["raw_delta_y_normalized"] = row["center_y_normalized"] - previous["center_y_normalized"]
                row["raw_delta_seconds"] = row["timestamp_seconds"] - previous["timestamp_seconds"]
            segments[-1].append(row)
        for segment_id, segment in enumerate(segments):
            left, right = 0, 0
            half_window = config.window_seconds / 2
            for row in segment:
                row["segment_id"] = segment_id
                time = row["timestamp_seconds"]
                while left < len(segment) and segment[left]["timestamp_seconds"] < time - half_window:
                    left += 1
                while right < len(segment) and segment[right]["timestamp_seconds"] <= time + half_window:
                    right += 1
                window, adapted = supported_window(segment, left, right, time, config)
                span = window[-1]["timestamp_seconds"] - window[0]["timestamp_seconds"]
                row.update(window_observations=len(window), window_span_seconds=span,
                           window_adapted=adapted)
                fit = robust_local_fit([r["timestamp_seconds"] for r in window],
                                       [r["center_y_normalized"] for r in window],
                                       time, row["center_y_normalized"], config)
                row["fit_status"] = "insufficient_evidence"
                row["fit_reason"] = ";".join(reason for failed, reason in (
                    (len(window) < config.min_observations, "too_few_observations"),
                    (span < config.min_fit_seconds, "short_window_span")) if failed)
                if fit is not None:
                    row.update(fitted_y_normalized=fit.y, slope_per_second=fit.slope,
                               inlier_fraction=fit.inlier_fraction, residual_normalized=fit.residual,
                               fit_reason=fit.reason, inlier_count=fit.inlier_count,
                               inlier_span_seconds=fit.inlier_span_seconds,
                               fit_status="accepted" if fit.accepted else "rejected")
            events.extend(infer_events(segment, config))
    return sorted(events, key=lambda event: (event["frame_index"], event["track_id"]))


def detection_rows(result: Any, tracks: Any, frame_index: int, fps: float) -> list[dict[str, Any]]:
    """Join tracker identities to original detections using its detection index.

    Ultralytics' tracker output is [x1, y1, x2, y2, id, score, class, source_index].
    Save the original detector box, not the Kalman-filtered tracker box.
    """
    ids = {int(track[-1]): int(track[4]) for track in tracks}
    height, width = result.orig_shape
    boxes = result.boxes.cpu().numpy()
    rows = []
    for index, (xyxy, confidence) in enumerate(zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True)):
        x1, y1, x2, y2 = map(float, xyxy)
        rows.append({"frame_index": frame_index, "timestamp_seconds": frame_index / fps,
                     "detection_index": index, "track_id": ids.get(index), "confidence": float(confidence),
                     "frame_width": width, "frame_height": height,
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "center_x": (x1 + x2) / 2, "center_y": (y1 + y2) / 2,
                     "center_y_normalized": (y1 + y2) / (2 * height)})
    return rows


def collect_detections(video_path: Path, model_path: Path, tracking_confidence: float,
                       device: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for path in (video_path, model_path, TRACKER_CONFIG):
        if not path.is_file():
            raise VerificationError(f"File does not exist: {path}")
    try:
        import cv2
        from ultralytics import YOLO
        from ultralytics.engine.results import Results
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import YAML, IterableSimpleNamespace
    except ImportError as exc:
        raise VerificationError("Install Scripts/requirements.txt to run video inference.") from exc
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VerificationError(f"Could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise VerificationError("Video FPS must be finite and positive")
        reported_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = int(reported_frames) if math.isfinite(reported_frames) and reported_frames > 0 else 0
        status(f"Loading model: {model_path}")
        model = YOLO(str(model_path))
        milk_id = resolve_milk_class_id(model.names)
        tracker = BOTSORT(args=IterableSimpleNamespace(**YAML.load(str(TRACKER_CONFIG))))
        arguments = {"conf": tracking_confidence, "classes": [milk_id], "verbose": False}
        if device is not None:
            arguments["device"] = device
        rows: list[dict[str, Any]] = []
        frame_count = 0
        started = last_progress = time.monotonic()
        status(f"Processing video ({fps:.2f} FPS). First-frame setup may take a moment...")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            predictions = model.predict(frame, stream=False, **arguments)
            if not isinstance(predictions, list) or not predictions:
                raise VerificationError("Expected a non-empty list of prediction results")
            result = predictions[0]
            if not isinstance(result, Results) or result.boxes is None:
                raise VerificationError("The model must return object detection boxes")
            # Explicit tracker update preserves all raw detections and advances
            # its clock even on empty frames. model.track() can filter boxes.
            tracks = tracker.update(result.boxes.cpu().numpy(), frame)
            rows.extend(detection_rows(result, tracks, frame_count, fps))
            frame_count += 1
            now = time.monotonic()
            if frame_count == 1 or now - last_progress >= 2.0:
                elapsed = now - started
                progress = (f"{frame_count}/{total_frames} frames ({min(100.0, 100 * frame_count / total_frames):.1f}%)"
                            if total_frames else f"{frame_count} frames")
                status(f"Processed {progress} | milk detections: {len(rows)} | elapsed: {elapsed:.1f}s")
                last_progress = now
        if not frame_count:
            raise VerificationError("Video yielded no decoded frames")
        status(f"Video processing finished: {frame_count} frames, {len(rows)} milk detections.")
        return rows, {"video": str(video_path), "model": str(model_path), "fps": fps,
                      "frame_count": frame_count, "duration_seconds": frame_count / fps}
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"Video inference failed: {exc}") from exc
    finally:
        capture.release()


CSV_FIELDS: tuple[str, ...] = ("frame_index", "timestamp_seconds", "detection_index", "track_id", "confidence",
              "frame_width", "frame_height", "x1", "y1", "x2", "y2", "center_x", "center_y",
              "center_y_normalized", "segment_id", "raw_delta_y_normalized", "raw_delta_seconds",
              "fitted_y_normalized", "slope_per_second", "inlier_fraction", "residual_normalized",
              "fit_status", "fitted_delta_y_normalized", "fitted_velocity_per_second", "motion", "event",
              "fit_reason", "window_observations", "window_span_seconds", "window_adapted",
              "inlier_count", "inlier_span_seconds", "candidate_displacement_normalized",
              "candidate_duration_seconds", "candidate_hits", "candidate_direction")


def plot_motion(rows: list[dict[str, Any]], events: list[dict[str, Any]], path: Path,
                config: MotionConfig) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise VerificationError("Matplotlib is required for the motion plot.") from exc
    fig, (position, velocity) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout="constrained")
    try:
        styles = {"accepted": ("#17834c", "o", "Accepted observation (event checks separate)"),
                  "untracked": ("#777777", "s", "Untracked: no confirmed ID"),
                  "low_confidence": ("#bb8700", "v", "Below fit confidence threshold"),
                  "insufficient_evidence": ("#8b46b0", "X", "Insufficient observations / time span"),
                  "rejected": ("#c74040", "x", "Rejected fit / observation")}
        for fit_status, (color, marker, label) in styles.items():
            selected = [r for r in rows if r["fit_status"] == fit_status]
            if selected:
                position.scatter([r["timestamp_seconds"] for r in selected],
                                 [r["center_y_normalized"] for r in selected],
                                 s=30, c=color, marker=marker, alpha=0.85, label=f"{label} ({len(selected)})", zorder=3)
        adapted = [r for r in rows if r.get("window_adapted")]
        if adapted:
            position.scatter([r["timestamp_seconds"] for r in adapted],
                             [r["center_y_normalized"] for r in adapted],
                             s=85, facecolors="none", edgecolors="#1673b1", label="Adapted window", zorder=4)
        groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in rows:
            if "segment_id" in row:
                groups.setdefault((row["track_id"], row["segment_id"]), []).append(row)
        for (track_id, segment_id), group in groups.items():
            times = [r["timestamp_seconds"] for r in group]
            fitted = [r["fitted_y_normalized"] if r["fit_status"] == "accepted" else math.nan for r in group]
            line, = position.plot(times, fitted, linewidth=1.8,
                                  label=f"Accepted fit: track {track_id}, segment {segment_id}" if len(groups) <= 12 else "_nolegend_")
            velocity.plot(times, [r.get("fitted_velocity_per_second", math.nan) for r in group], color=line.get_color())
        for event in events:
            color = "#198754" if event["direction"] == "inward" else "#be4b27"
            position.axvline(event["timestamp_seconds"], color=color, alpha=0.65, linestyle=":")
            position.text(event["timestamp_seconds"], 0.02, event["direction"], rotation=90,
                          transform=position.get_xaxis_transform(), color=color, va="bottom")
        position.set(title="Milk motion: raw centers and robust rolling fits (no spatial gate)",
                     ylabel="Center y / frame height")
        position.invert_yaxis()
        if rows:
            position.legend(fontsize=8, loc="best")
        velocity.axhline(0, color="#555555", linewidth=0.8)
        velocity.axhline(config.min_speed, color="#999999", linestyle="--", linewidth=0.8)
        velocity.axhline(-config.min_speed, color="#999999", linestyle="--", linewidth=0.8)
        velocity.set(xlabel="Video time (seconds)", ylabel="Fitted delta y / delta t\n(frame heights / second)")
        for axis in (position, velocity):
            axis.grid(alpha=0.15)
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def write_outputs(rows: list[dict[str, Any]], events: list[dict[str, Any]], metadata: dict[str, Any],
                  config: MotionConfig, output_dir: Path, stem: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}_motion.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    net_delta = sum(1 if event["direction"] == "inward" else -1 for event in events)
    report = {**metadata, "config": asdict(config), "detection_count": len(rows), "events": events,
              "net_delta": net_delta,
              "interpretation": "Directional motion episodes; not verified fridge crossings or shelf placements."}
    json_path = output_dir / f"{stem}_motion.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_path = output_dir / f"{stem}_motion.png"
    plot_motion(rows, events, plot_path, config)
    for label, path in (("Coordinates CSV", csv_path), ("Event report", json_path), ("Trajectory plot", plot_path)):
        status(f"{label}: {path}")
    return net_delta


def print_summary(rows: list[dict[str, Any]], events: list[dict[str, Any]], net_delta: int) -> None:
    accepted = sum(row.get("fit_status") == "accepted" for row in rows)
    if not rows:
        status("No milk detections were found.")
    elif not accepted:
        status("Milk was detected, but no observations met the tracking/confidence/regression requirements.")
    elif not events:
        status("Milk was detected and fitted, but no movement met the event thresholds.")
    status(f"Complete: {len(rows)} detections, {accepted} accepted fits, {len(events)} motion events, net change {net_delta:+d}.")
    print(f"MOTION EVENTS ({len(events)}): {json.dumps(events)}", file=sys.stderr)
    print("Experimental motion estimate; y-direction alone does not verify a fridge crossing.", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--debug", action="store_true", help="Save debugging CSV, JSON and PNG files")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "debug" / "milk_motion",
                        help="Output folder (only used with --debug)")
    parser.add_argument("--track-conf", type=float, default=0.10)
    parser.add_argument("--conf", type=float, default=0.40, help="Minimum confidence for regression")
    parser.add_argument("--inward", choices=("down", "up"), default="down", help="Image direction into the fridge")
    for name, default, help_text in (
        ("window-seconds", 0.60, "Full width of centered local regression window"),
        ("max-gap-seconds", 0.50, "Split a track after this gap in confident observations"),
        ("min-fit-seconds", 0.12, "Minimum time spanned by fit inliers"),
        ("residual-threshold", 0.025, "Maximum fit residual in frame heights"),
        ("min-inlier-fraction", 0.65, "Required fraction of window observations agreeing with fit"),
        ("min-speed", 0.04, "Minimum speed in frame heights per second"),
        ("min-displacement", 0.08, "Minimum sustained travel in frame heights"),
        ("confirm-seconds", 0.15, "Minimum sustained movement duration"),
    ):
        parser.add_argument(f"--{name}", type=float, default=default, help=help_text)
    parser.add_argument("--min-observations", type=int, default=5)
    parser.add_argument("--adaptive-window", action=argparse.BooleanOptionalAction, default=True,
                        help="Extend sparse centered windows within --window-seconds total span (default: enabled)")
    args = parser.parse_args(argv)
    args.config = MotionConfig(confidence=args.conf, **{name: getattr(args, name)
                               for name in MotionConfig.__dataclass_fields__ if name != "confidence"})
    try:
        args.config.validate()
        if not 0 < args.track_conf <= args.conf:
            raise ValueError("Require 0 < --track-conf <= --conf <= 1")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        status(f"Starting milk motion analysis: {args.video.expanduser().resolve()}")
        if args.debug:
            status(f"Results will be saved to: {args.output_dir.expanduser().resolve()}")
        status("Loading inference dependencies...")
        with contextlib.redirect_stdout(sys.stderr):
            rows, metadata = collect_detections(args.video.expanduser().resolve(), args.model.expanduser().resolve(),
                                                args.track_conf, args.device)
            metadata["tracking_confidence"] = args.track_conf
            status("Fitting milk trajectories and finding motion events...")
            events = analyze_detections(rows, args.config)
            net_delta = sum(1 if event["direction"] == "inward" else -1 for event in events)
            if args.debug:
                status("Saving coordinate CSV, event report, and trajectory plot...")
                write_outputs(rows, events, metadata, args.config,
                              args.output_dir.expanduser().resolve(), args.video.stem)
            print_summary(rows, events, net_delta)
    except (VerificationError, OSError, ValueError) as exc:
        status(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        status("Cancelled.")
        return 130
    print(net_delta, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
