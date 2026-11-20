# Milk vision scripts

Verify milk additions and removals in fridge videos, or extract video frames.

- **Placement** counts confirmed milk crossings through a horizontal gate.
- **Motion** estimates directional movement from one logical trajectory.
- **Jev** groups directional episodes using timestamps, normalized y and confidence.
- **Observation collection** saves reusable detector/tracker evidence for all three methods.
- **Frame extraction** saves selected video intervals as JPEG images.

Motion is experimental: vertical movement alone does not prove a fridge crossing
or shelf placement.

## Labelled-video evaluation

Compare placement, motion, and Jev on all annotated milk clips and random-object negatives:

```powershell
python -m Scripts.tools.evaluate --methods placement motion jev --include-negatives --device cpu
```

This uses one shared v4s extraction per video and reuses completed observations and
predictions on subsequent runs. Jev calls the TypeSafe API; set `TYPESAFE_API_KEY`
in your environment or the ignored `Scripts/.env`. The historical default command
without `--methods` still runs placement and motion.

See [Shared observations](docs/observations.md) for collection and cache commands.
For GPU testing, use the [existing Colab notebook workflow](docs/colab.md):
select a published Git ref, tick placement/motion/Jev, and run fresh predictions
or resume completed work while reusing observations.
See [Development and operations](docs/development.md#labelled-video-evaluation)
for annotation rules, filtering, commands, and score interpretation.

## Project layout

Clone this repository as `Scripts` inside a parent project directory:

```bash
git clone https://github.com/Jae-Zen/milk-vision-scripts.git Scripts
```

```text
Yolo/
  Scripts/                  # This repository
    milk_vision/            # observations/, algorithms/, evaluation/, CLI and diagnostics
    tools/                  # Command entry points and frame extraction
    tests/                  # Automated checks and fixtures
    docs/                   # Detailed guides
  Models/
    milk-vision-v4s.pt       # Placement default
    milk-vision-v4n.pt       # Motion default
  cache/observations/       # Reusable raw detections and tracking results
  cache/predictions/        # Versioned algorithm results
  debug/evaluation/         # Per-run comparison reports
  Dataset/
    videos/                 # Your input videos
```

Supply model checkpoints and input videos separately. Each checkpoint needs a
class named `milk`; other classes are ignored.

## Installation

Use Python 3.11 or newer. Run these PowerShell commands from the parent `Yolo`
directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\Scripts\requirements.txt
```

The requirements install Ultralytics, the tracker assignment dependency, and
python-dotenv for local Jev credential loading.
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
- [Jev](docs/jev-experiment.md): episode framework, input fields, API behavior, and historical experiments.
- [Shared observations](docs/observations.md): collection, cache contents, compatibility, and refresh.
- [Colab lab](docs/colab.md): published revisions, algorithm selection, GPU setup, and experiment history.
- [Frame extraction](docs/frame-extraction.md): time selection, filenames, and output behavior.
- [Development](docs/development.md): package layout, folder execution, tests, and troubleshooting.
