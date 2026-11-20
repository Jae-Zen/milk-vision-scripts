"""Coordinate inference, pure analysis, and optional diagnostic output."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

from . import diagnostics, inference
from .config import PROJECT_ROOT
from .crossings import (
    CROSSING_CONFIDENCE,
    CROSSING_CONFIRM_SECONDS,
    EVIDENCE_GAP_SECONDS,
    TRACK_GAP_SECONDS,
    TRACKING_CONFIDENCE,
    _seconds_to_frames,
    calculate_net_delta,
)
from .errors import VerificationError
from .motion import MotionConfig, analyze_detections
from .types import CrossingEvent, MotionResult, PlacementResult


def verify_placement_video(
    video_path: Path,
    model_path: Path,
    confidence: float = CROSSING_CONFIDENCE,
    gate_y: float = 0.27,
    device: str | None = None,
    debug: bool = False,
    tracking_confidence: float = TRACKING_CONFIDENCE,
    debug_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> PlacementResult:
    """Return crossings and run details without printing a console summary."""
    video_path = video_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    if not 0 <= confidence <= 1:
        raise VerificationError("Confidence must be between 0 and 1.")
    if not 0 < tracking_confidence <= confidence:
        raise VerificationError(
            "Tracking confidence must be positive and no greater than crossing confidence."
        )
    if not 0.05 <= gate_y <= 0.95:
        raise VerificationError("Gate position must be between 0.05 and 0.95.")

    with ExitStack() as stack:
        video = stack.enter_context(
            inference.PlacementVideo(
                video_path,
                model_path,
                tracking_confidence,
                device,
                load_frames=debug,
                cache_dir=cache_dir,
            )
        )
        confirm = _seconds_to_frames(CROSSING_CONFIRM_SECONDS, video.fps, minimum=2)
        gap = _seconds_to_frames(TRACK_GAP_SECONDS, video.fps)
        evidence_gap = _seconds_to_frames(EVIDENCE_GAP_SECONDS, video.fps)
        writer = None
        if debug:
            writer = stack.enter_context(
                diagnostics.PlacementDiagnostics(
                    video_path, debug_dir, video.fps, gate_y
                )
            )
        from .algorithms.placement import analyze

        frames = iter(video)

        def render(frame, details):
            if writer is not None:
                item = next(frames)
                if item.frame_index != frame.frame_index:
                    raise VerificationError(
                        "Diagnostic frame does not match observations"
                    )
                writer.write_frame(item, details)

        prediction = analyze(
            video.observations, confidence, gate_y, on_frame=render if debug else None
        )
        events = [CrossingEvent(**event) for event in prediction.details["events"]]

    return PlacementResult(
        net_delta=calculate_net_delta(events),
        events=events,
        video_path=video_path,
        model_path=model_path,
        fps=video.fps,
        confidence=confidence,
        tracking_confidence=tracking_confidence,
        gate_y=gate_y,
        confirm_observations=confirm,
        max_gap_frames=gap,
        max_evidence_gap_frames=evidence_gap,
        csv_path=writer.csv_path if writer else None,
        video_debug_path=writer.video_path if writer else None,
    )


def verify_motion_video(
    video_path: Path,
    model_path: Path,
    config: MotionConfig,
    tracking_confidence: float = 0.10,
    device: str | None = None,
    debug: bool = False,
    output_dir: Path | None = None,
    *,
    status: Callable[[str], None] = lambda message: None,
    cache_dir: Path | None = None,
) -> MotionResult:
    """Collect raw positions, analyze the trajectory, and optionally save it."""
    video_path = video_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    rows, metadata = inference.collect_detections(
        video_path,
        model_path,
        tracking_confidence,
        device,
        status=status,
        **({"cache_dir": cache_dir} if cache_dir is not None else {}),
    )
    metadata["tracking_confidence"] = tracking_confidence
    status("Fitting milk trajectories and finding motion events...")
    events = analyze_detections(rows, config)
    net_delta = sum(1 if event["direction"] == "inward" else -1 for event in events)
    result = MotionResult(rows, events, metadata, net_delta)
    if debug:
        status("Saving coordinate CSV, event report, and trajectory plot...")
        directory = (
            output_dir
            if output_dir is not None
            else PROJECT_ROOT / "debug" / "milk_motion"
        )
        diagnostics.write_outputs(
            result,
            config,
            directory.expanduser().resolve(),
            video_path.stem,
            status=status,
        )
    return result
