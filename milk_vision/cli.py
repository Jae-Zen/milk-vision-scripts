"""Parse one command, run its handler, and report the result."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .batch import find_videos, run_batch
from .config import MOTION_MODEL, PLACEMENT_MODEL, PROJECT_ROOT, TRACKER_CONFIG
from .crossings import CROSSING_CONFIDENCE, GATE_HYSTERESIS, TRACKING_CONFIDENCE
from .errors import VerificationError
from .motion import MotionConfig
from .pipeline import verify_motion_video, verify_placement_video
from .types import CrossingEvent, MotionEvent, MotionRow, PlacementResult


def status(message: str) -> None:
    """Flush status immediately, including when launched with redirected output."""
    print(message, file=sys.stderr, flush=True)


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


def print_placement_result(
    result: PlacementResult, *, stream: TextIO | None = None
) -> None:
    """Write deterministic, human-readable run diagnostics to stderr."""
    if stream is None:
        stream = sys.stderr
    if stream is None:
        # Python may have no stderr stream when running without a console.
        return
    if result.csv_path is None:
        print_crossing_events(stream=stream, milk_events=result.events)
        return
    print("DEBUG CONFIGURATION", file=stream)
    print(f"  video={result.video_path}", file=stream)
    print(f"  model={result.model_path}", file=stream)
    print(f"  tracker={TRACKER_CONFIG.name} fps={result.fps:.6f}", file=stream)
    print(
        f"  tracking_confidence={result.tracking_confidence:.3f} crossing_confidence={result.confidence:.3f}",
        file=stream,
    )
    print(
        f"  gate_y={result.gate_y:.3f} outer_limit={result.gate_y - GATE_HYSTERESIS:.3f} "
        f"inner_limit={result.gate_y + GATE_HYSTERESIS:.3f}",
        file=stream,
    )
    print(
        f"  confirm_observations={result.confirm_observations} "
        f"max_gap_frames={result.max_gap_frames} max_evidence_gap_frames={result.max_evidence_gap_frames}",
        file=stream,
    )

    print_crossing_events(
        stream=stream, milk_events=result.events, label="DEBUG MILK EVENTS"
    )

    print(f"DEBUG RESULT net_delta={result.net_delta}", file=stream)
    print(f"DEBUG CSV {result.csv_path}", file=stream)
    print(f"DEBUG VIDEO {result.video_debug_path}", file=stream)


def print_motion_summary(
    rows: list[MotionRow], events: list[MotionEvent], net_delta: int
) -> None:
    accepted = sum(row.get("fit_status") == "accepted" for row in rows)
    if not rows:
        status("No milk detections were found.")
    elif not accepted:
        status("Milk was detected, but no observations could form a local fit.")
    elif not events:
        status(
            "Milk was detected and fitted, but no movement met the event thresholds."
        )
    status(
        f"Complete: {len(rows)} detections, {accepted} accepted fits, {len(events)} motion events, net change {net_delta:+d}."
    )
    print(f"MOTION EVENTS ({len(events)}): {json.dumps(events)}", file=sys.stderr)
    print(
        "Experimental motion estimate; y-direction alone does not verify a fridge crossing.",
        file=sys.stderr,
    )


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
    """Parse and validate the complete command once."""
    parser = argparse.ArgumentParser(
        prog=f"python -m {__package__}",
        description="Verify milk crossings or estimate milk motion.",
        allow_abbrev=False,
    )
    methods = parser.add_subparsers(dest="method", required=True)
    defaults = MotionConfig()
    for method, model in (("placement", PLACEMENT_MODEL), ("motion", MOTION_MODEL)):
        command = methods.add_parser(method, allow_abbrev=False)
        inputs = command.add_mutually_exclusive_group(required=True)
        inputs.add_argument("--video", type=Path, help="Input video path")
        inputs.add_argument("--folder", type=Path, help="Input video folder")
        command.add_argument(
            "--model",
            type=Path,
            default=model,
            help=f"YOLO checkpoint (default: {model})",
        )
        command.add_argument("--cache-dir", type=Path, help="Reuse shared observations")
        command.add_argument("--device", help="Ultralytics device, such as cpu or 0")
        command.add_argument(
            "--debug", action="store_true", help="Save diagnostic files"
        )
        command.add_argument("--track-conf", type=float, default=TRACKING_CONFIDENCE)
        if method == "placement":
            command.add_argument(
                "--conf",
                default=CROSSING_CONFIDENCE,
                type=lambda value: _bounded_float(value, 0, 1, "confidence"),
                help="Minimum milk confidence for crossing evidence (default: 0.40)",
            )
            command.add_argument(
                "--gate-y",
                default=0.27,
                type=lambda value: _bounded_float(value, 0.05, 0.95, "gate position"),
                help="Normalized vertical gate position (default: 0.27)",
            )
            command.add_argument(
                "--debug-dir",
                type=Path,
                help="Diagnostic directory (default: Scripts directory)",
            )
        else:
            command.add_argument(
                "--output-dir",
                type=Path,
                default=PROJECT_ROOT / "debug" / "milk_motion",
                help="Diagnostic directory (only used with --debug)",
            )
            command.add_argument(
                "--inward", choices=("down", "up"), default=defaults.inward
            )
            for name, value, help_text in (
                (
                    "window-seconds",
                    defaults.window_seconds,
                    "Full local regression window width",
                ),
                (
                    "max-gap-seconds",
                    defaults.max_gap_seconds,
                    "Split trajectory after this gap",
                ),
                (
                    "min-speed",
                    defaults.min_speed,
                    "Minimum speed in frame heights per second",
                ),
                (
                    "min-displacement",
                    defaults.min_displacement,
                    "Minimum travel in frame heights",
                ),
            ):
                command.add_argument(
                    f"--{name}", type=float, default=value, help=help_text
                )
    args = parser.parse_args(argv)
    try:
        if args.method == "placement":
            if not 0 < args.track_conf <= args.conf:
                raise ValueError(
                    "Tracking confidence must be positive and no greater than crossing confidence."
                )
        else:
            args.config = MotionConfig(
                window_seconds=args.window_seconds,
                max_gap_seconds=args.max_gap_seconds,
                min_speed=args.min_speed,
                min_displacement=args.min_displacement,
                inward=args.inward,
            )
            args.config.validate()
            if not 0 < args.track_conf <= 1:
                raise ValueError("--track-conf must be in (0, 1]")
        if args.folder is not None:
            args.folder = args.folder.expanduser().resolve()
            args.videos = find_videos(args.folder, debug=args.debug)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run_placement(args: argparse.Namespace) -> int:
    result = verify_placement_video(
        video_path=args.video,
        model_path=args.model,
        confidence=args.conf,
        gate_y=args.gate_y,
        device=args.device,
        debug=args.debug,
        tracking_confidence=args.track_conf,
        debug_dir=args.debug_dir,
        **({"cache_dir": args.cache_dir} if args.cache_dir is not None else {}),
    )
    print_placement_result(result)
    return result.net_delta


def run_motion(args: argparse.Namespace) -> int:
    status(f"Starting milk motion analysis: {args.video.expanduser().resolve()}")
    if args.debug:
        status(f"Results will be saved to: {args.output_dir.expanduser().resolve()}")
    status("Loading inference dependencies...")
    result = verify_motion_video(
        args.video,
        args.model,
        args.config,
        args.track_conf,
        args.device,
        args.debug,
        args.output_dir,
        status=status,
        **({"cache_dir": args.cache_dir} if args.cache_dir is not None else {}),
    )
    print_motion_summary(result.rows, result.events, result.net_delta)
    return result.net_delta


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.folder is not None:
            return run_batch(args)
        with contextlib.redirect_stdout(sys.stderr):
            if args.method == "placement":
                net_delta = run_placement(args)
            else:
                net_delta = run_motion(args)
    except (VerificationError, OSError, ValueError) as exc:
        status(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        status("Cancelled.")
        return 130
    print(net_delta, flush=True)
    return 0
