"""Run the milk-placement verifier for every video in a folder."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = SCRIPT_DIR / "verify_milk_placement.py"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".wmv"}


def natural_key(path: Path) -> list[object]:
    """Return a case-insensitive key that sorts vid_9 before vid_10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def find_videos(folder: Path) -> list[Path]:
    """Find supported video files directly inside *folder*, sorted by filename."""
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=natural_key,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run verify_milk_placement.py for every video in a folder."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Folder containing the videos to verify",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    folder = args.folder.expanduser().resolve()

    if not folder.is_dir():
        print(f"Video folder does not exist: {folder}", file=sys.stderr)
        return 1

    videos = find_videos(folder)
    if not videos:
        print(f"No supported video files found in: {folder}", file=sys.stderr)
        return 1

    print(f"Found {len(videos)} video(s) in: {folder}")
    failed = 0

    for index, video_path in enumerate(videos, start=1):
        print(f"\n[{index}/{len(videos)}] Running: {video_path.name}")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_NAME), "--video", str(video_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Success! Output:")
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        else:
            failed += 1
            print(f"Error processing {video_path.name}:", file=sys.stderr)
            print(
                result.stderr,
                file=sys.stderr,
                end="" if result.stderr.endswith("\n") else "\n",
            )

    if failed:
        print(f"\nCompleted with {failed} failed video(s).", file=sys.stderr)
        return 1

    print(f"\nCompleted all {len(videos)} video(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
