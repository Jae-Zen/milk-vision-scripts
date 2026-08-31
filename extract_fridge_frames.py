#!/usr/bin/env python3
"""Interactively extract selected video intervals as JPEG frames."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def natural_key(path: Path) -> list[object]:
    """Sort vid_9 before vid_10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def parse_seconds(value: str) -> float:
    cleaned = value.strip().lower()
    if cleaned.endswith("s"):
        cleaned = cleaned[:-1].strip()
    try:
        seconds = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Invalid time: {value!r}") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Time must be a non-negative number.")
    return seconds


def ask_time_ranges(duration: float) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []

    while True:
        start_text = input(
            "  Start time (e.g. 4.14s; 'n' = next video; 'q' = quit): "
        ).strip()
        if start_text.lower() == "q":
            raise KeyboardInterrupt
        if start_text.lower() in {"n", "next"}:
            return ranges

        end_text = input("  End time in seconds (e.g. 10.24s): ").strip()
        if end_text.lower() == "q":
            raise KeyboardInterrupt

        try:
            start = parse_seconds(start_text)
            end = parse_seconds(end_text)
            if end < start:
                raise ValueError("End time must be greater than or equal to start time.")
            if start > duration:
                raise ValueError(f"Start time exceeds the video duration ({duration:.2f}s).")
            if end > duration + 0.02:
                raise ValueError(f"End time exceeds the video duration ({duration:.2f}s).")
            ranges.append((start, min(end, duration)))
            print(
                f"  Added range {len(ranges)}: "
                f"{start:.2f}s to {min(end, duration):.2f}s"
            )
        except ValueError as exc:
            print(f"  {exc} Please try again.\n")


def merge_time_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Combine overlapping selections to avoid writing duplicate frames."""
    if not ranges:
        return []

    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def require_opencv() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError:
        print(
            "OpenCV is required. Install it with:\n\n"
            "    python -m pip install opencv-python\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return cv2


def ask_video_directory() -> Path:
    while True:
        entered = input("Enter the folder containing your videos: ").strip().strip('"')
        if not entered:
            print("Please enter a folder path.\n")
            continue

        video_dir = Path(entered).expanduser().resolve()
        if video_dir.is_dir():
            return video_dir
        print(f"Folder does not exist: {video_dir}\n")


def inspect_video(cv2: Any, video_path: Path) -> tuple[Any, float, int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("The video has invalid FPS or frame-count metadata.")

    # The last frame is at (frame_count - 1) / fps; frame_count / fps is the
    # conventional displayed duration and is easier for choosing an endpoint.
    duration = frame_count / fps
    return capture, fps, frame_count, duration


def extract_range(
    cv2: Any,
    capture: Any,
    video_path: Path,
    output_dir: Path,
    fps: float,
    frame_count: int,
    start_seconds: float,
    end_seconds: float,
    jpeg_quality: int,
) -> int:
    # Include each frame whose timestamp lies in the closed interval [start, end].
    start_frame = max(0, math.ceil(start_seconds * fps - 1e-9))
    end_frame = min(frame_count - 1, math.floor(end_seconds * fps + 1e-9))
    if start_frame > end_frame:
        print("  No video frame falls inside that interval; nothing extracted.")
        return 0

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    expected = end_frame - start_frame + 1

    for frame_number in range(start_frame, end_frame + 1):
        ok, frame = capture.read()
        if not ok:
            print(f"\n  Warning: decoding stopped at source frame {frame_number}.")
            break

        output_path = output_dir / f"{video_path.stem}_frame_{frame_number:08d}.jpg"
        saved = cv2.imwrite(
            str(output_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not saved:
            raise RuntimeError(f"Could not write {output_path}")

        written += 1
        if written % 100 == 0 or written == expected:
            print(f"\r  Extracting: {written}/{expected} frames", end="", flush=True)

    if written:
        print()
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask for one or more time ranges per video and extract the selected frames."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Folder containing videos; if omitted, the script asks for it",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        choices=range(1, 101),
        metavar="1-100",
        help="JPEG quality (default: 95)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video_dir = (
        args.video_dir.expanduser().resolve()
        if args.video_dir is not None
        else ask_video_directory()
    )
    output_dir = Path(__file__).resolve().parent / "fridge items"

    if not video_dir.is_dir():
        print(f"Video folder does not exist: {video_dir}", file=sys.stderr)
        return 1

    cv2 = require_opencv()

    videos = sorted(
        (path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=natural_key,
    )
    if not videos:
        print(f"No supported video files found in: {video_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_jpegs = sum(1 for _ in output_dir.glob("*.jpg"))

    print(f"Found {len(videos)} videos in: {video_dir}")
    print(f"Frames will be saved to: {output_dir}")
    if existing_jpegs:
        print(
            f"Note: the output folder already contains {existing_jpegs} JPEG(s). "
            "Matching filenames will be overwritten; other files will be left unchanged."
        )
    print(
        "Add as many start/end ranges as you need for each video.\n"
        "Enter 'n' at a start-time prompt to continue to the next video, "
        "or 'q' to stop.\n"
    )

    total_written = 0
    processed = 0

    try:
        for index, video_path in enumerate(videos, start=1):
            print(f"[{index}/{len(videos)}] {video_path.name}")
            try:
                capture, fps, frame_count, duration = inspect_video(cv2, video_path)
            except RuntimeError as exc:
                print(f"  Skipped: {exc}\n")
                continue

            print(f"  Duration: {duration:.2f}s | FPS: {fps:.3f} | Frames: {frame_count}")
            try:
                selected_ranges = ask_time_ranges(duration)
                if not selected_ranges:
                    print("  No ranges selected; skipped.\n")
                    continue

                merged_ranges = merge_time_ranges(selected_ranges)
                video_written = 0
                if len(merged_ranges) < len(selected_ranges):
                    print(
                        f"  Combined {len(selected_ranges)} overlapping selections "
                        f"into {len(merged_ranges)} range(s)."
                    )

                for range_index, (start, end) in enumerate(merged_ranges, start=1):
                    print(
                        f"  Range {range_index}/{len(merged_ranges)}: "
                        f"{start:.2f}s to {end:.2f}s"
                    )
                    video_written += extract_range(
                        cv2,
                        capture,
                        video_path,
                        output_dir,
                        fps,
                        frame_count,
                        start,
                        end,
                        args.jpeg_quality,
                    )

                total_written += video_written
                processed += 1
                print(f"  Saved {video_written} JPEG frame(s) for this video.\n")
            finally:
                capture.release()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped by user. Frames already extracted have been kept.")

    print(f"\nDone. Saved {total_written} frame(s) from {processed} video(s).")
    print(f"Output folder: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
