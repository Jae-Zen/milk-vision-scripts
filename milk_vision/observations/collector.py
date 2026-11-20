"""One detector/tracker pass, retaining raw and tracker-produced boxes."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

from ..config import TRACKER_CONFIG, resolve_milk_class_id
from ..errors import VerificationError
from ..types import (
    BoundingBox,
    FrameObservations,
    RawDetection,
    TrackedDetection,
    VideoObservations,
)

COLLECTOR_VERSION = 2


def clip_tracked_box(box: BoundingBox, width: int, height: int) -> BoundingBox:
    """Match Results.update clipping before placement consumes tracker boxes."""
    return BoundingBox(
        max(0.0, min(float(width), box.x1)),
        max(0.0, min(float(height), box.y1)),
        max(0.0, min(float(width), box.x2)),
        max(0.0, min(float(height), box.y2)),
    )


def collect_video(
    video_path: Path,
    model_path: Path,
    confidence: float = 0.10,
    device: str | None = None,
    status: Callable[[str], None] = lambda message: None,
) -> VideoObservations:
    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise VerificationError("Detector confidence must be finite and in (0, 1]")
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
        raise VerificationError(
            "Install Scripts/requirements.txt to collect observations"
        ) from exc
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VerificationError(f"Could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise VerificationError("Video FPS must be finite and positive")
        model = YOLO(str(model_path))
        milk = resolve_milk_class_id(model.names)
        tracker = BOTSORT(
            args=IterableSimpleNamespace(**YAML.load(str(TRACKER_CONFIG)))
        )
        arguments = {"conf": confidence, "classes": [milk], "verbose": False}
        if device is not None:
            arguments["device"] = device
        frames = []
        started = progress = time.monotonic()
        while True:
            ok, image = capture.read()
            if not ok:
                break
            index = len(frames)
            height, width = image.shape[:2]
            predictions = model.predict(image, stream=False, **arguments)
            if not isinstance(predictions, list) or not predictions:
                raise VerificationError(
                    "Expected a non-empty list of prediction results"
                )
            result = predictions[0]
            if not isinstance(result, Results) or result.boxes is None:
                raise VerificationError("YOLO must return detection boxes")
            boxes = result.boxes.cpu().numpy()
            tracks = tracker.update(boxes, image)
            associations = {int(t[-1]): int(t[4]) for t in tracks}
            if any(i < 0 or i >= len(boxes) for i in associations):
                raise VerificationError(
                    "Tracker returned an invalid detector association"
                )
            raw = tuple(
                RawDetection(
                    BoundingBox(*map(float, xyxy)), float(score), i, associations.get(i)
                )
                for i, (xyxy, score) in enumerate(
                    zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True)
                )
            )
            tracked = tuple(
                TrackedDetection(
                    "milk",
                    int(t[4]),
                    index,
                    clip_tracked_box(BoundingBox(*map(float, t[:4])), width, height),
                    float(t[5]),
                )
                for t in tracks
            )
            frames.append(
                FrameObservations(index, index / fps, width, height, raw, tracked)
            )
            if index == 0 or time.monotonic() - progress >= 2:
                progress = time.monotonic()
                status(
                    f"Collected {len(frames)} frames | {sum(len(f.raw) for f in frames)} detections | {progress - started:.1f}s"
                )
        if not frames:
            raise VerificationError(
                "No frames could be decoded: video yielded no decoded frames"
            )
        return VideoObservations(
            {
                "video": str(video_path),
                "model": str(model_path),
                "fps": fps,
                "frame_count": len(frames),
                "duration_seconds": len(frames) / fps,
                "tracking_confidence": confidence,
            },
            tuple(frames),
        )
    except VerificationError:
        raise
    except (ValueError, TypeError, RuntimeError, IndexError) as exc:
        raise VerificationError(f"Video inference failed: {exc}") from exc
    finally:
        capture.release()
