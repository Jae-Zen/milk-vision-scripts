import subprocess
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = SCRIPT_DIR / "verify_milk_placement.py"
VIDEO_FOLDER = SCRIPT_DIR / "../Dataset/videos"

count = 1
for video_path in VIDEO_FOLDER.iterdir():
    if count == 50: 
        break
    if video_path.is_file():
        print(f"....running for: {video_path.name} ")

        result = subprocess.run(
            [sys.executable, SCRIPT_NAME, "--video", video_path],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("Success! Output:")
            print(result.stdout)
        else:
            print(f"Error, processing {video_path.name}: ")
            print(result.stderr)

        count += 1
