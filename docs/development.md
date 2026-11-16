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

Dataset evaluation lives in the sibling `tools` package:

| Module | Responsibility |
| --- | --- |
| `tools/dataset.py` | Load and validate annotations and video coverage. |
| `tools/evaluate.py` | Call the milk pipelines, compare labelled outcomes, and report scores. |

`tests/test_batch.py` checks the folder runner. `tests/test_video_evaluation.py`
checks annotation loading and evaluation using temporary fixtures and mocked predictions.

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

## Labelled-video evaluation

Labelled videos live directly in `food/net -1`, `food/net 0`, or `food/net 1`
under `Dataset/videos`. The number is the expected signed inventory change for
that food. Entering adds one and leaving subtracts one.

`dataset.py` loads `annotations.json` and validates video paths, complete
coverage, duplicate keys, and event totals. `evaluate.py` uses those labels to
compare placement and motion predictions with the expected outcomes.

Each annotation is keyed by its video path relative to `Dataset/videos` inside
the document's `videos` object. The document uses `schema_version: 1` and an
`excluded` list for videos without net labels. Two long cheese recordings for
frame extraction are excluded. An annotation record contains:

```json
{
  "scenario": "add_and_remove_milk",
  "description": "Milk is added and then removed.",
  "expected_events": ["in", "out"]
}
```

Scenario labels describe exactly what happened. Remove cheese expects `["out"]`,
add milk expects `["in"]`, and remove/add expects `["out", "in"]`.
Repositioning/no change and random-object milk negatives expect `[]`. Event sums
must match the net folder. New videos need matching annotation entries, and
renaming a video requires updating its key. Unknown sequences can use `null`,
which excludes them from event metrics. Current clips have explicit event labels.

Validate the dataset or run an evaluation from `Yolo`:

```powershell
python -m Scripts.tools.dataset
python -m Scripts.tools.evaluate --list
python -m Scripts.tools.evaluate --scenario add_milk --limit 1
python -m Scripts.tools.evaluate --include-negatives
```

From inside `Yolo/Scripts`, use `python -m tools.dataset` or
`python -m tools.evaluate` with the same options. With the project's virtual
environment activated, `python` selects its interpreter. The terminal directory
determines whether the module name starts with `Scripts.tools` or `tools`.

The dataset command validates by default and also accepts the existing `--check`
spelling. Evaluation performs the same validation before loading models.

### Command syntax

```powershell
.\.venv\Scripts\python.exe -m Scripts.tools.evaluate --scenario add_milk --limit 1
```

| Part | Meaning |
| --- | --- |
| `.\` | Start the executable path from the terminal's current directory, `Yolo`. |
| `.venv\Scripts\python.exe` | Use Python inside this project's virtual environment. |
| `-m` | Find and execute the module named next. |
| `Scripts.tools.evaluate` | Execute `Scripts/tools/evaluate.py`, whose main guard calls `main()`. |
| `--scenario add_milk` | Select annotations with `"scenario": "add_milk"`. |
| `--limit 1` | Select at most one matching video before running algorithms. |

The argument parser receives `--scenario add_milk --limit 1`. Both algorithms
run by default, so this command produces two predictions for one clip. Use
`--method placement` or `--method motion` to choose one. The evaluator has its
own main guard and does not execute `milk_vision/__main__.py`.

### Reports and scores

Evaluation is explicitly invoked and runs sequentially. The default selection is
133 milk clips. `--include-negatives` adds 36 `others/net 0` clips, for 169 initial
clips total. Cheese annotations are retained for future cheese algorithms.
Both milk algorithms use `Models/milk-vision-v4s.pt` by default. `--model` changes
the shared checkpoint. Single-video verifier defaults are independent of these
evaluation defaults.

JSON reports go under `debug/evaluation/` outside the dataset. They record model,
manifest, and algorithm-code hashes, settings, per-video predictions/errors, and
overall, scenario, and expected-net score breakdowns. `--output` selects a new
report path. Existing reports are not overwritten. Other options include
`--device`, `--conf`, `--track-conf`, `--gate-y`, and motion thresholds.
Use `--help` for the complete list.

- Net accuracy is the fraction of attempted videos with the correct final count.
  Processing failures count against this score.
- `mean_absolute_error_successful` averages count error over successfully
  processed videos. Processing failures are reported separately.
- Event-sequence accuracy requires the complete ordered sequence to match.
  Predicting `[]` for `["in", "out"]` gets the zero net count right but the
  event sequence wrong.
- Event precision, recall, and F1 use ordered sequence alignment, without timestamp
  matching. Unknown event annotations do not contribute to these metrics, and
  empty precision/recall denominators are reported as `null`.

The terminal prints each clip's result followed by a JSON score summary.
Dataset scores alone do not establish accuracy on unseen recordings.

Exit codes are `0` for all comparisons matching, `1` for mismatches or processing
failures, `2` for invalid input, and `130` for cancellation. A partial report is
saved after cancellation. Fast unit tests use temporary fixtures and mocked
predictions. Run real-video evaluations explicitly using the commands above.

## Tests

| File | What it checks |
| --- | --- |
| `tests/test_verify_milk_placement.py` | Gate crossings, jitter, confidence thresholds, tracker gaps/IDs, signed counts, detection extraction, placement diagnostics, and CLI validation. Includes a real BoT-SORT matching check when dependencies are available. |
| `tests/test_verify_milk_motion.py` | Trajectory fitting, direction changes, pauses, gaps, outliers, confidence selection, raw detections, motion events, diagnostics, and CLI thresholds. |
| `tests/test_batch.py` | Folder discovery, natural filename sorting, processing every video, and forwarding the selected model to child commands. |
| `tests/test_milk_vision_package.py` | Integration between CLI, batch, pipelines, inference, and diagnostics. Covers failures, cancellation, live output, stable paths/schemas, lazy imports, and command entry points. |
| `tests/test_tools_paths.py` | Tool defaults and module commands from the project root or another working directory. |
| `tests/test_video_evaluation.py` | Annotation validation, food/scenario selection, net/event scoring, processing failures, cancellation, and report generation using temporary files and mocked predictions. |
| `tests/fixtures/milk_motion_204912.json` | Recorded detector observations used by the motion regression test. This is input data loaded by a test. |

Python's `unittest` runner discovers the `test_*.py` files and executes their test
methods. Developers and coding agents run relevant tests after changing code and
before reporting completion. Local hooks or CI can invoke the same runner when
configured. The command below runs the complete suite from `Yolo`:

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
