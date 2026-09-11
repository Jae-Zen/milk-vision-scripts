"""Run motion verification on each video in a folder; forward extra CLI options."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    args, options = parser.parse_known_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"Video folder does not exist: {folder}")
    extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".wmv"}
    videos = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions),
                    key=lambda p: p.name.lower())
    if not videos:
        parser.error(f"No supported videos found in: {folder}")
    script = Path(__file__).with_name("verify_milk_motion.py")
    failed = 0
    for index, video in enumerate(videos, 1):
        print(f"\n[{index}/{len(videos)}] {video.name}", flush=True)
        result = subprocess.run([sys.executable, str(script), *options, "--video", str(video)])
        if result.returncode == 130:
            return 130
        failed += result.returncode != 0
    print(f"\nCompleted {len(videos)} videos; {failed} failed.")
    return int(failed > 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
