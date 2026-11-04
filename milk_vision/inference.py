"""Lazy video/model adapters; preserve each algorithm's tracker semantics."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .config import TRACKER_CONFIG, resolve_milk_class_id
from .errors import VerificationError
from .types import (
    BoundingBox,
    MotionRow,
    PlacementFrame,
    TrackedDetection,
    VideoMetadata,
)


class PlacementVideo:
    """Own a capture and persistent tracker for exactly one placement video."""

    def __init__(
        self,
        video_path: Path,
        model_path: Path,
        tracking_confidence: float,
        device: str | None,
    ) -> None:
        self.video_path = video_path
        self.model_path = model_path
        self.tracking_confidence = tracking_confidence
        self.device = device
        self.capture = None

    def __enter__(self) -> PlacementVideo:
        for path, label in (
            (self.video_path, "Video file"),
            (self.model_path, "Model file"),
            (TRACKER_CONFIG, "BoT-SORT configuration"),
        ):
            if not path.is_file():
                raise VerificationError(f"{label} does not exist: {path}")
        try:
            import cv2
            from ultralytics import YOLO
        except ImportError as exc:
            raise VerificationError(
                "OpenCV and Ultralytics are required. Install Scripts/requirements.txt."
            ) from exc
        self.capture = cv2.VideoCapture(str(self.video_path))
        try:
            if not self.capture.isOpened():
                raise VerificationError(
                    f"OpenCV could not open the video: {self.video_path}"
                )
            self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise VerificationError("The video has invalid FPS metadata.")
            try:
                self.model = YOLO(str(self.model_path), task="segment")
                self.milk_class_id = resolve_milk_class_id(self.model.names)
            except VerificationError:
                raise
            except Exception as exc:
                raise VerificationError(
                    f"Could not load the YOLO model: {exc}"
                ) from exc
            self.track_arguments = {
                "persist": True,
                "tracker": str(TRACKER_CONFIG),
                "conf": self.tracking_confidence,
                "classes": [self.milk_class_id],
                "verbose": False,
            }
            if self.device is not None:
                self.track_arguments["device"] = self.device
            return self
        except BaseException:
            self.capture.release()
            raise

    def __iter__(self) -> Iterator[PlacementFrame]:
        if self.capture is None:
            raise VerificationError(
                "Open PlacementVideo with a with block before reading frames."
            )
        frame_index = -1
        while True:
            ok, frame = self.capture.read()
            # OpenCV cannot distinguish normal EOF from a decode failure here.
            if not ok:
                break
            frame_index += 1
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                raise VerificationError(
                    f"Decoded frame {frame_index} has invalid dimensions."
                )
            try:
                result = self.model.track(frame, **self.track_arguments)[0]
            except Exception as exc:
                raise VerificationError(
                    f"YOLO tracking failed at frame {frame_index}: {exc}"
                ) from exc
            yield PlacementFrame(
                frame,
                frame_index,
                height,
                width,
                _extract_tracked_detections(result, frame_index, self.milk_class_id),
            )
        if frame_index < 0:
            raise VerificationError(
                f"No frames could be decoded from the video: {self.video_path}"
            )

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.capture is not None:
            self.capture.release()


def _extract_tracked_detections(
    result: Any,
    frame_index: int,
    milk_class_id: int,
) -> list[TrackedDetection]:
    """Convert matched milk boxes into plain Python detections."""
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


def detection_rows(
    result: Any, tracks: Any, frame_index: int, fps: float
) -> list[MotionRow]:
    """Join tracker identities to original detections using its detection index.

    Ultralytics' tracker output is [x1, y1, x2, y2, id, score, class, source_index].
    Save the original detector box, not the Kalman-filtered tracker box.
    """
    ids = {int(track[-1]): int(track[4]) for track in tracks}
    height, width = result.orig_shape
    boxes = result.boxes.cpu().numpy()
    rows = []
    for index, (xyxy, confidence) in enumerate(
        zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True)
    ):
        x1, y1, x2, y2 = map(float, xyxy)
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / fps,
                "detection_index": index,
                "track_id": ids.get(index),
                "confidence": float(confidence),
                "frame_width": width,
                "frame_height": height,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
                "center_y_normalized": (y1 + y2) / (2 * height),
            }
        )
    return rows


def collect_detections(
    video_path: Path,
    model_path: Path,
    tracking_confidence: float,
    device: str | None,
    status: Callable[[str], None] = lambda message: None,
) -> tuple[list[MotionRow], VideoMetadata]:
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
            "Install Scripts/requirements.txt to run video inference."
        ) from exc
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VerificationError(f"Could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise VerificationError("Video FPS must be finite and positive")
        reported_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = (
            int(reported_frames)
            if math.isfinite(reported_frames) and reported_frames > 0
            else 0
        )
        status(f"Loading model: {model_path}")
        model = YOLO(str(model_path))
        milk_id = resolve_milk_class_id(model.names)
        tracker = BOTSORT(
            args=IterableSimpleNamespace(**YAML.load(str(TRACKER_CONFIG)))
        )
        arguments = {
            "conf": tracking_confidence,
            "classes": [milk_id],
            "verbose": False,
        }
        if device is not None:
            arguments["device"] = device
        rows: list[MotionRow] = []
        frame_count = 0
        started = last_progress = time.monotonic()
        status(
            f"Processing video ({fps:.2f} FPS). First-frame setup may take a moment..."
        )
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            predictions = model.predict(frame, stream=False, **arguments)
            if not isinstance(predictions, list) or not predictions:
                raise VerificationError(
                    "Expected a non-empty list of prediction results"
                )
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
                progress = (
                    f"{frame_count}/{total_frames} frames ({min(100.0, 100 * frame_count / total_frames):.1f}%)"
                    if total_frames
                    else f"{frame_count} frames"
                )
                status(
                    f"Processed {progress} | milk detections: {len(rows)} | elapsed: {elapsed:.1f}s"
                )
                last_progress = now
        if not frame_count:
            raise VerificationError("Video yielded no decoded frames")
        status(
            f"Video processing finished: {frame_count} frames, {len(rows)} milk detections."
        )
        return rows, {
            "video": str(video_path),
            "model": str(model_path),
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps,
        }
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"Video inference failed: {exc}") from exc
    finally:
        capture.release()
