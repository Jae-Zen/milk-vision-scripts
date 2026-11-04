# Development and operations

[Quick start](../README.md) · [Placement](placement.md) · [Motion](motion.md) · [Frame extraction](frame-extraction.md) · [Development](development.md)

Commands assume the terminal is in the parent `Yolo` directory unless stated otherwise.

## Implementation layout

Verification starts through the package's `__main__.py`; the four old wrapper
scripts have been removed. No package installation is required. From the parent
`Yolo` directory, use `python -m Scripts.milk_vision`; from inside `Scripts`, use
`python -m milk_vision` instead. These are the same package entry point:

```powershell
# From Yolo:
python -m Scripts.milk_vision placement --video ".\Dataset\videos\clip.mp4"
python -m Scripts.milk_vision motion --folder ".\Dataset\videos"

# Alternatively, from Yolo\Scripts:
python -m milk_vision placement --video "..\Dataset\videos\clip.mp4"
python -m milk_vision motion --folder "..\Dataset\videos"
```

Choose exactly one of `--video` and `--folder`. Relative paths are resolved from
the terminal's current directory. The model and default diagnostic locations
remain unchanged. Implementation lives in the `milk_vision` package:

| Module | Responsibility |
| --- | --- |
| `cli.py` | Argument definitions, console summaries, and exit codes. |
| `batch.py` | Video discovery and sequential child processes for both methods. |
| `pipeline.py` | Coordinate inference, analysis, and optional diagnostics; return structured results. |
| `inference.py` | Lazy model/video loading, detection extraction, and separate tracking adapters. |
| `crossings.py` | Pure crossing state machine, thresholds, and signed counts. |
| `motion.py` | Pure regression, trajectory segmentation, and motion-event rules. |
| `diagnostics.py` | Streaming placement CSV/video and motion CSV/JSON/plots. |
| `types.py` | Detection/event/result dataclasses and typed motion dictionaries. |
| `config.py`, `errors.py` | Stable project paths, model-class validation, and expected errors. |
| `milk_botsort.yaml` | Shared tracker configuration, moved into the package. |

The CLI parses the complete command once using argparse subcommands. It passes
the parsed settings directly to `run_placement`, `run_motion`, or `run_batch`;
handlers do not parse arguments again. Single-video errors, cancellation, and
signed-integer output are handled in one place in `main`.

The CLI calls the pipelines; the pipelines connect video inference to the pure
algorithms and optional diagnostic writers. Algorithms do not parse arguments,
print results, write files, or import OpenCV/Ultralytics. Plotting and inference
dependencies are loaded only when needed. Placement diagnostics are written one
frame at a time rather than buffering a video in memory.

Placement still uses persistent tracker identities and confirmed gate crossings.
Motion still uses raw detections and one logical trajectory with its existing
regression/event rules. Refactoring does not establish real-world model accuracy.
Python callers should import definitions from their owning package modules.

## Folder verification commands

Both folder commands share discovery, argument validation, and sequential
subprocess execution:

```powershell
python -m Scripts.milk_vision placement --folder "D:\fridge videos"
python -m Scripts.milk_vision motion --folder "D:\fridge videos"
```

Use `--folder` instead of `--video`. Each runner accepts its single-video
verifier's other options, including `--model`, `--device`, `--track-conf`, and
`--debug`. Placement also accepts gate/confidence settings and `--debug-dir`;
motion accepts motion thresholds, `--inward`, and `--output-dir`. Invalid options
and thresholds are rejected before any video is started.

```powershell
python -m Scripts.milk_vision placement `
  --folder "D:\fridge videos" `
  --model "D:\models\milk-vision-custom.pt" `
  --debug `
  --debug-dir ".\debug\milk_placement"

python -m Scripts.milk_vision motion `
  --folder "D:\fridge videos" `
  --debug `
  --output-dir ".\debug\milk_motion" `
  --device cpu
```

- Supported extensions are `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.webm`, and
  `.wmv`, matched case-insensitively. Subfolders and non-video files are ignored.
- Both runners use case-insensitive natural filename order: `vid_2.mp4` precedes
  `vid_10.mp4`. There is no video-count limit.
- Each video runs in a fresh process using the same Python interpreter and loads
  its own model. No tracking state is shared between videos.
- The console shows each filename before processing, live verifier output on
  stdout and stderr (including successful diagnostics), and a final failure count.
  The batch console is human-readable; it is not a single integer result.
- Video failures and process-launch errors are reported without stopping the
  remaining videos. Cancellation stops the batch.
- Exit codes are `0` for complete success, `1` for processing failures, `2` for
  invalid arguments or missing/empty input folders, and `130` for cancellation.
- Debug output is off by default. With `--debug`, duplicate filename stems are
  rejected case-insensitively before processing: `clip.mp4` and `clip.mov` would
  otherwise overwrite each other's artifacts. Existing artifacts from earlier
  runs can still be overwritten. Neither runner writes an aggregate report.
- Model defaults remain `Models/milk-vision-v4s.pt` for placement and
  `Models/milk-vision-v4n.pt` for motion. Model loading errors are processing
  failures, reported separately by each child verifier.

Show the complete options for either method:

```powershell
python -m Scripts.milk_vision placement --help
python -m Scripts.milk_vision motion --help
```

## Tests

```powershell
python -m unittest discover -s Scripts/tests -v
```

For syntax, lint, and type validation, check the entire package. Runtime tests
alone do not check type annotations. With the optional development tools
installed (`python -m pip install ruff pyright`), run from the parent project:

```powershell
python -m compileall -q Scripts/tools Scripts/milk_vision Scripts/tests
python -m ruff check --select F,I,B,SIM Scripts/tools Scripts/milk_vision Scripts/tests
python -m pyright --pythonpath .venv/Scripts/python.exe Scripts
```

Tests cover frame-extractor paths and command entry points, signed milk counting,
confidence gating, weak-match liveness, evidence timeouts, debug CSV/CLI output,
and mocked inference pipelines. A BoT-SORT test exercises real weak-score matching
on an existing track when the inference dependencies are installed. No test
requires a trained model file. The real tracker test is skipped if its imports
are unavailable. The suite does not test actual model accuracy or visually
render/inspect debug videos.

## Troubleshooting

### `Ultralytics is not installed`

Activate the environment used during installation, then run:

```powershell
python -m pip install -r .\Scripts\requirements.txt
```

### CUDA out-of-memory error

Force CPU inference for either verification method:

```powershell
python -m Scripts.milk_vision placement --video ".\Dataset\videos\clip.mp4" --device cpu
python -m Scripts.milk_vision motion --video ".\Dataset\videos\clip.mp4" --device cpu
```
