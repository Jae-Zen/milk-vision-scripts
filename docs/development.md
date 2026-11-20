# Development and operations

[Quick start](../README.md) · [Placement](placement.md) · [Motion](motion.md) · [Frame extraction](frame-extraction.md) · [Development](development.md) · [Observations](observations.md) · [Jev](jev-experiment.md)

Commands assume the terminal is in the parent `Yolo` directory unless stated otherwise.

## Implementation layout

Use `python -m Scripts.milk_vision` from `Yolo`, or `python -m milk_vision`
from inside `Scripts`. Existing `Scripts.tools.evaluate` / `tools.evaluate`
and dataset command entry points remain available without installing the package.

| Component | Responsibility |
| --- | --- |
| `observations/collector.py` | One YOLO predict and BoT-SORT update per decoded frame; preserve raw and tracked boxes and their association. |
| `observations/storage.py` | Versioned, validated, atomic observation storage and fresh-process extraction. |
| `observations/milk_botsort.yaml` | Canonical tracker configuration. The old root YAML remains a compatibility asset. |
| `algorithms/placement.py` | Tracked-box gate crossing rules and replay, including empty frames. |
| `algorithms/motion.py` | Raw-coordinate fitting and directional event rules; fresh working rows per analysis. |
| `algorithms/jev.py` | Three-field input, frozen five-choice prompt, API call, response validation, episode counts and net calculation. |
| `evaluation/dataset.py` | Annotation validation and dataset records. |
| `evaluation/runner.py`, `reporting.py`, `types.py` | Method selection, resumable execution, metrics and evaluation records. |
| `types.py` | Shared frames/detections and the common prediction contract. |
| `pipeline.py`, `inference.py` | CLI orchestration and compatibility adapters over shared observations. |
| `cli.py`, `batch.py` | Single-video/folder arguments, output and sequential child processes. |
| `diagnostics.py` | Coordinate plots/exports and frame-by-frame annotated video rendering. |
| `tools/*.py` | Compatibility commands and frame extraction; algorithm logic lives in the package. |

The root `crossings.py` and `motion.py` are compatibility imports for their new
algorithm modules. `tools.dataset` and `tools.evaluate` similarly forward to the
new evaluation modules. Existing imported entry points remain available.

The shared collector includes every decoded frame, even with no detections.
Placement reads tracker-produced boxes and IDs; motion and Jev read original
post-NMS detector boxes. Algorithms do not read videos or initialize YOLO.
Jev alone makes remote API calls, and owns all request formatting and instructions.
Each algorithm uses fresh working data, so motion diagnostics cannot alter Jev input.
Jev events have no invented timestamps. Ground truth enters only evaluation.

See [Shared observations](observations.md) for artifact contents, compatibility,
refresh, and collection commands; [Jev](jev-experiment.md) for prompt and API details.
Debug placement video reopens and decodes the source video for pixels only; it
joins them to shared frame indices without rerunning the detector or tracker.
No raw pixel arrays are retained in observation caches.

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
compare placement, motion and Jev predictions with the expected outcomes.

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

### Three-method shared-observation comparison

For the same comparison on a hosted GPU, see [Colab setup and resume](colab.md).

```powershell
.\.venv\Scripts\python.exe -m Scripts.tools.evaluate `
  --methods placement motion jev --include-negatives --device cpu `
  --model .\Models\milk-vision-v4s.pt `
  --output .\debug\evaluation\comparison.json
```

The current milk scope is 133 milk clips plus 36 optional negative clips. The
55 cheese-labelled clips are not part of milk evaluation. `--method both`
remains the historical default (placement and motion); `--method jev` selects
Jev alone. `--methods` accepts an explicit list and is mutually exclusive with
`--method`. All selected methods share the explicitly selected detector and
extraction settings; the evaluation default is v4s, while standalone motion
retains its historical v4n default.

Evaluation caches observations under `Yolo/cache/observations` by default.
`--cache-dir` overrides it. Fresh extraction runs sequentially in a fresh process
per video. `--refresh-observations` creates a new completed artifact version.
Predictions are separately versioned under the cache directory's sibling
`predictions/` directory. Identity includes observation content, algorithm code,
settings and, for Jev, its exact request and model. `--rerun` forces new prediction
versions without discarding previous results. Repeating with a new `--output`
resumes compatible predictions; output reports cannot overwrite earlier reports.
Labels are rescored from the current annotations rather than stored in prediction
caches. Changing a checkpoint, extraction settings, prompt or algorithm naturally
invalidates the corresponding reuse. Cache hits can be replayed without inference;
Jev authentication is needed only when an actual uncached request is sent.

Reports are atomically updated after every completed method result, including
failures. Ctrl+C saves a partial report; a hard interruption leaves the most recent
completed report and caches. Reports include algorithm-only and collection/load
seconds, detection counts, cache hits, observation paths/hashes, and Jev confidence,
probabilities, request/response, actual model and token usage. Collection timing is
shared evidence metadata repeated per method; do not sum it across methods.

Per-class/scenario metrics include an explicit split between empty and nonempty
detection inputs. The accuracy denominator includes failed selected comparisons;
failures are not silently dropped. Event scores compare inventory-event annotations,
not independently labelled visible-motion intervals. `none` on empty evidence can
be correct for observed motion while still missing an inventory event. Zero-net
agreement alone does not establish correct episode grouping.

`--jev-model` defaults to `jev-1.14.0`, and `--api-timeout` defaults to 45 seconds.
There are no automatic API retries, long-context activation, truncation, smoothing
or temporal subsampling of Jev inputs. Context-limit, timeout, network or malformed
response errors are reported as failures. An explicit retry may still be billable
if the previous timed-out request reached TypeSafe.

### Command syntax

```powershell
.\.venv\Scripts\python.exe -m Scripts.tools.evaluate --scenario add_milk --limit 1
```

| Part | Meaning |
| --- | --- |
| `.\` | Start the executable path from the terminal's current directory, `Yolo`. |
| `.venv\Scripts\python.exe` | Use Python inside this project's virtual environment. |
| `-m` | Find and execute the module named next. |
| `Scripts.tools.evaluate` | Execute the compatibility entry point, which calls `milk_vision.evaluation.runner.main()`. |
| `--scenario add_milk` | Select annotations with `"scenario": "add_milk"`. |
| `--limit 1` | Select at most one matching video before running algorithms. |

The argument parser receives `--scenario add_milk --limit 1`. Placement and motion
run by default, so this command produces two predictions for one clip. Use
`--method placement`, `--method motion`, or `--method jev` to choose one. The evaluator has its
own main guard and does not execute `milk_vision/__main__.py`.

### Reports and scores

Evaluation is explicitly invoked and runs sequentially. The default selection is
133 milk clips. `--include-negatives` adds 36 `others/net 0` clips, for 169 initial
clips total. Cheese annotations are retained for future cheese algorithms.
All selected methods use `Models/milk-vision-v4s.pt` by default. `--model` changes
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
| `tests/test_shared_observations.py` | Cache integrity/invalidation, empty frames, independent algorithm inputs, shared collection, prediction resume, method selection. |
| `tests/test_jev_experiment.py` | Three-field isolation, five episode choices, Python net calculation, transport and saved-result validation. |
| `tests/test_colab_workflow.py` | Notebook helpers: all algorithm combinations, run modes, filters, source compatibility, edited-worktree protection and experiment history; no inference or API calls. |
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
