"""Shared sequential subprocess execution for both folder commands."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from .config import SCRIPTS_DIR

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".wmv"}


def natural_key(path: Path) -> tuple:
    """Sort numbered names naturally, with a deterministic case-sensitive tie."""
    parts = tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )
    return parts, path.name


def find_videos(folder: Path, *, debug: bool = False) -> list[Path]:
    """Discover and validate batch inputs before any child process starts."""
    if not folder.is_dir():
        raise ValueError(f"Video folder does not exist: {folder}")
    try:
        videos = sorted(
            (
                p
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
            ),
            key=natural_key,
        )
    except OSError as exc:
        raise ValueError(f"Could not read video folder {folder}: {exc}") from exc
    if not videos:
        raise ValueError(f"No supported videos found in: {folder}")
    if debug:
        stems = set()
        for video in videos:
            stem = video.stem.casefold()
            if stem in stems:
                raise ValueError(
                    f"Duplicate video stem would overwrite debug artifacts: {video.stem}"
                )
            stems.add(stem)
    return videos


def _child_options(args: argparse.Namespace) -> list[str]:
    """Forward verifier settings, excluding batch inputs and derived values."""
    options = []
    for name, value in vars(args).items():
        if name in {"method", "video", "folder", "videos", "config"} or value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                options.append(flag)
        else:
            options.extend([flag, str(value)])
    return options


def run_batch(args: argparse.Namespace) -> int:
    """Run validated videos with the settings parsed by the CLI."""
    videos = args.videos
    options = _child_options(args)
    # Keep the caller's working directory for relative input/output paths while
    # making the package importable in every fresh child process.
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(SCRIPTS_DIR), environment.get("PYTHONPATH")))
    )
    failed = 0
    print(f"Found {len(videos)} video(s) in: {args.folder}", flush=True)
    for index, video in enumerate(videos, 1):
        print(f"\n[{index}/{len(videos)}] {video.name}", flush=True)
        try:
            # Inherit both streams so even successful stderr diagnostics stay visible.
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "milk_vision",
                    args.method,
                    *options,
                    "--video",
                    str(video),
                ],
                check=False,
                env=environment,
            )
        except OSError as exc:
            print(
                f"Error starting verifier for {video.name}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            failed += 1
            continue
        if result.returncode in (130, -2, 0xC000013A, -1073741510):
            return 130
        if result.returncode != 0:
            print(
                f"Error processing {video.name} (exit {result.returncode})",
                file=sys.stderr,
                flush=True,
            )
            failed += 1
    print(f"\nCompleted {len(videos)} videos; {failed} failed.", flush=True)
    return int(failed > 0)
