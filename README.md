# Milk vision scripts

Verify milk additions and removals in fridge videos, or extract video frames.

- **Placement** counts confirmed milk crossings through a horizontal gate.
- **Motion** estimates directional movement from one logical trajectory.
- **Frame extraction** saves selected video intervals as JPEG images.

Motion is experimental: vertical movement alone does not prove a fridge crossing
or shelf placement.

## Project layout

Clone this repository as `Scripts` inside a parent project directory:

```bash
git clone https://github.com/Jae-Zen/milk-vision-scripts.git Scripts
```

```text
Yolo/
  Scripts/                  # This repository
    milk_vision/            # Verification and diagnostics
    tools/                  # Frame extraction
    tests/                  # Automated checks and fixtures
    docs/                   # Detailed guides
  Models/
    milk-vision-v4s.pt       # Placement default
    milk-vision-v4n.pt       # Motion default
  Dataset/
    videos/                 # Your input videos
```

Supply model checkpoints and input videos separately. Each checkpoint needs a
class named `milk`; other classes are ignored.

## Installation

Use Python 3.10 or newer. Run these PowerShell commands from the parent `Yolo`
directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\Scripts\requirements.txt
```

The requirements install Ultralytics and the tracker assignment dependency.
Ultralytics brings PyTorch, NumPy, OpenCV, and plotting dependencies.
All commands below use the activated environment.

## Quick start

Run commands from `Yolo`. Quote paths containing spaces.

### Verify placement

```powershell
python -m Scripts.milk_vision placement --video ".\Dataset\videos\clip.mp4"
```

### Estimate motion

```powershell
python -m Scripts.milk_vision motion --video ".\Dataset\videos\clip.mp4"
```

For either method, replace `--video PATH` with `--folder PATH` to process a folder.
Use `--model PATH` for another checkpoint or `--device cpu` for CPU inference.

### Extract frames

```powershell
python -m Scripts.tools.extract_fridge_frames --video-dir ".\Dataset\videos"
```

Enter start/end times for each video, then `n` to extract the selected ranges.
Enter `q` to stop; pending selections for the current video are discarded.

## Results and debugging

Single-video verification prints one signed integer on stdout: inward events
minus outward events. Diagnostics go to stderr. Folder runs also print progress
and a summary, so their stdout is not a single integer.

| Command | Default files |
| --- | --- |
| Placement | None; add `--debug` for CSV and annotated video in `Scripts/`. |
| Motion | None; add `--debug` for CSV, JSON, and PNG in `debug/milk_motion/` under `Yolo`. |
| Frame extraction | JPEG images in `Scripts/fridge items/`. |

Use `--debug-dir PATH` to redirect placement diagnostics and `--output-dir PATH`
to redirect motion diagnostics. Default model/output paths follow the project
layout; explicit relative paths follow your terminal directory.

For current CLI options, append `--help` to any command.

## Detailed documentation

- [Placement](docs/placement.md): gate rules, confidence thresholds, timeouts, and diagnostics.
- [Motion](docs/motion.md): regression, event rules, outputs, and limitations.
- [Frame extraction](docs/frame-extraction.md): time selection, filenames, and output behavior.
- [Development](docs/development.md): package layout, folder execution, tests, and troubleshooting.
