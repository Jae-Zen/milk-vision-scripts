"""Render YOLO segmentation predictions onto a video.

The output is named ``<input-name>_segmented.mp4`` and is always written beside
this script. Each prediction is shown with a translucent segmentation mask, a
high-contrast polygon outline, its class name, and its confidence score.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MASK_COLORS = [
    (50, 220, 50),
    (255, 170, 40),
    (60, 90, 240),
    (220, 80, 220),
    (40, 210, 230),
    (230, 210, 70),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a video showing segmentation masks, outlines, class names, "
            "and confidence scores from a YOLO segmentation model."
        )
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        type=Path,
        help="Path to the input video.",
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        type=Path,
        help="Path to a trained YOLO segmentation model (.pt).",
    )
    parser.add_argument(
        "--video",
        dest="video_option",
        type=Path,
        default=None,
        help="Input video path (alternative to the first positional argument).",
    )
    parser.add_argument(
        "--model",
        dest="model_option",
        type=Path,
        default=None,
        help="Model path (alternative to the second positional argument).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum prediction confidence from 0 to 1 (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold used during prediction (default: 0.7).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example '0' for GPU or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=3,
        help="Segmentation outline width in pixels (default: 3).",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.30,
        help="Mask opacity from 0 to 1 (default: 0.30).",
    )
    return parser.parse_args()


def resolve_and_validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.video_path is not None and args.video_option is not None:
        raise ValueError("Supply the video either positionally or with --video, not both.")
    if args.model_path is not None and args.model_option is not None:
        raise ValueError("Supply the model either positionally or with --model, not both.")

    raw_video = args.video_option or args.video_path
    raw_model = args.model_option or args.model_path
    if raw_video is None:
        raise ValueError("An input video path is required.")
    if raw_model is None:
        raise ValueError("A vision model path is required.")

    video_path = raw_video.expanduser().resolve()
    model_path = raw_model.expanduser().resolve()
    if not video_path.is_file():
        raise ValueError(f"Input video does not exist: {video_path}")
    if not model_path.is_file():
        raise ValueError(f"Vision model does not exist: {model_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than zero.")
    if args.line_width <= 0:
        raise ValueError("--line-width must be greater than zero.")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be between 0 and 1.")
    return video_path, model_path


def class_name(names: Any, class_id: int) -> str:
    """Return a readable class name for Ultralytics list or dict metadata."""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def draw_caption(
    cv2: Any,
    image: Any,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a label against a solid class-colored background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(image.shape[:2]) / 1000.0)
    thickness = max(1, round(scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x = max(0, min(origin[0], image.shape[1] - text_width - 7))
    y = max(text_height + 6, min(origin[1], image.shape[0] - baseline - 3))
    cv2.rectangle(
        image,
        (x, y - text_height - 6),
        (x + text_width + 7, y + baseline + 3),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        text,
        (x + 3, y - 3),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def annotate_result(
    cv2: Any,
    np: Any,
    frame: Any,
    result: Any,
    line_width: int,
    mask_alpha: float,
) -> tuple[Any, int]:
    """Draw one prediction result and return the frame and instance count."""
    polygons = [] if result.masks is None else result.masks.xy
    if not polygons:
        return frame, 0
    if result.boxes is None:
        raise RuntimeError("The model returned masks without matching boxes.")

    class_ids = [int(value) for value in result.boxes.cls.cpu().tolist()]
    confidences = [float(value) for value in result.boxes.conf.cpu().tolist()]
    if len(polygons) != len(class_ids) or len(polygons) != len(confidences):
        raise RuntimeError(
            "The model returned different numbers of masks, classes, and confidences."
        )

    contours: list[tuple[Any, int, float]] = []
    overlay = frame.copy()
    for polygon, predicted_class, confidence in zip(
        polygons, class_ids, confidences, strict=True
    ):
        if polygon is None or len(polygon) < 3:
            continue
        contour = np.rint(polygon).astype(np.int32).reshape((-1, 1, 2))
        color = MASK_COLORS[predicted_class % len(MASK_COLORS)]
        cv2.fillPoly(overlay, [contour], color, lineType=cv2.LINE_AA)
        contours.append((contour, predicted_class, confidence))

    annotated = cv2.addWeighted(overlay, mask_alpha, frame, 1.0 - mask_alpha, 0)
    for contour, predicted_class, confidence in contours:
        color = MASK_COLORS[predicted_class % len(MASK_COLORS)]
        cv2.polylines(
            annotated,
            [contour],
            isClosed=True,
            color=color,
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )
        points = contour.reshape((-1, 2))
        topmost = points[points[:, 1].argmin()]
        draw_caption(
            cv2,
            annotated,
            f"{class_name(result.names, predicted_class)} {confidence:.2f}",
            (int(topmost[0]), int(topmost[1]) - 5),
            color,
        )
    return annotated, len(contours)


def process_video(
    cv2: Any,
    np: Any,
    model: Any,
    video_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> tuple[int, int]:
    """Run inference over every input frame and write the annotated video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open input video: {video_path}")

    writer = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Input video has an invalid frame rate: {fps}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Input video has invalid dimensions: {width}x{height}")

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise OSError(f"Could not create output video: {output_path}")

        predict_options = {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "retina_masks": True,
            "verbose": False,
        }
        if args.device is not None:
            predict_options["device"] = args.device

        processed_frames = 0
        total_instances = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = model.predict(source=frame, **predict_options)
            if len(results) != 1:
                raise RuntimeError(
                    f"Model returned {len(results)} results for one video frame."
                )
            result = results[0]
            if result.boxes is not None and len(result.boxes) and result.masks is None:
                raise RuntimeError(
                    "The model produced detections without segmentation masks. "
                    "Use a YOLO segmentation model."
                )

            annotated, instance_count = annotate_result(
                cv2,
                np,
                frame,
                result,
                args.line_width,
                args.mask_alpha,
            )
            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height))
            writer.write(annotated)
            processed_frames += 1
            total_instances += instance_count

            if processed_frames == 1 or processed_frames % 30 == 0:
                total_text = str(frame_count) if frame_count > 0 else "?"
                print(f"Processed {processed_frames}/{total_text} frames", end="\r")

        if processed_frames == 0:
            raise ValueError("The input video contains no readable frames.")
        print(" " * 60, end="\r")
        return processed_frames, total_instances
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main() -> int:
    args = parse_args()
    try:
        video_path, model_path = resolve_and_validate_args(args)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError:
        requirements_path = SCRIPT_DIR / "requirements.txt"
        print(
            "Error: required packages are not installed. Run: "
            f'python -m pip install -r "{requirements_path}"',
            file=sys.stderr,
        )
        return 2

    output_path = SCRIPT_DIR / f"{video_path.stem}_segmented.mp4"
    try:
        print(f"Loading segmentation model: {model_path}")
        model = YOLO(str(model_path))
        if getattr(model, "task", None) != "segment":
            raise ValueError(
                f"Model task is {getattr(model, 'task', None)!r}; a segmentation "
                "model is required."
            )
        print(f"Writing labeled video to: {output_path}")
        frames, instances = process_video(
            cv2, np, model, video_path, output_path, args
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("Finished.")
    print(f"Frames processed:       {frames}")
    print(f"Predicted instances:    {instances}")
    print(f"Labeled video:          {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
