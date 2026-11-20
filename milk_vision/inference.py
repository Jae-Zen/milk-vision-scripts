"""Lazy video/model adapters; preserve each algorithm's tracker semantics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import VerificationError
from .types import (
    BoundingBox,
    MotionRow,
    PlacementFrame,
    TrackedDetection,
    VideoMetadata,
)


class PlacementVideo:
    """Compatibility adapter over the shared observations; pixels are debug-only."""

    def __init__(
        self,
        video_path: Path,
        model_path: Path,
        tracking_confidence: float,
        device: str | None,
        *,
        load_frames: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        self.video_path, self.model_path = video_path, model_path
        self.tracking_confidence, self.device = tracking_confidence, device
        self.load_frames, self.cache_dir = load_frames, cache_dir
        self.capture = None

    def __enter__(self):
        from .observations.storage import get_observations

        self.observations, self.collection_details = get_observations(
            self.video_path,
            self.model_path,
            self.tracking_confidence,
            self.device,
            self.cache_dir,
        )
        self.fps = float(self.observations.metadata.get("fps", 0))
        if self.load_frames:
            import cv2

            self.capture = cv2.VideoCapture(str(self.video_path))
            if not self.capture.isOpened():
                self.capture.release()
                raise VerificationError(
                    f"Could not reopen video for diagnostics: {self.video_path}"
                )
        return self

    def __iter__(self):
        for item in self.observations.frames:
            image = None
            if self.capture is not None:
                ok, image = self.capture.read()
                if not ok:
                    raise VerificationError(
                        "Debug video ended before the observation frames"
                    )
            yield PlacementFrame(
                image, item.frame_index, item.height, item.width, list(item.tracked)
            )

    def __exit__(self, exc_type, exc_value, traceback):
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
    *,
    cache_dir: Path | None = None,
) -> tuple[list[MotionRow], VideoMetadata]:
    """Compatibility projection of the single shared collector."""
    from .observations.collector import collect_video
    from .observations.storage import get_observations

    if cache_dir is None:
        observations = collect_video(
            video_path, model_path, tracking_confidence, device, status
        )
    else:
        observations, _ = get_observations(
            video_path, model_path, tracking_confidence, device, cache_dir
        )
    return observations.motion_rows(), observations.metadata.copy()
