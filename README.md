# Milk vision scripts

This repository contains a frame-extraction utility in `tools` and the `milk_vision` package:

- `tools/extract_fridge_frames.py` interactively extracts selected time ranges from videos as JPEG frames.
- `milk_vision placement` prints signed net milk change from milk-only crossings.
- `milk_vision motion` estimates signed net milk motion from tracked vertical movement.
- Both methods accept `--video` for one video or `--folder` for sequential folder verification.

The `milk_vision` package generates its own diagnostics with `--debug` through
`milk_vision/diagnostics.py`. Placement writes CSV and annotated video; motion
writes CSV, JSON, and plots. These workflows do not require a separate viewer
or labeling script.

This repository supplies scripts, tests, documentation, and configuration, not
trained models or input datasets. To preserve the default relative paths, clone
it as `Scripts` inside your project directory:

```bash
git clone https://github.com/Jae-Zen/milk-vision-scripts.git Scripts
```

The commands below assume the terminal is in the parent project directory:

```text
Yolo/
  Scripts/                 # This Git repository
    tools/                 # Frame-extraction utility
    milk_vision/           # Verification and diagnostics
    tests/                 # Automated checks and fixtures
  Models/
    milk-vision-v4s.pt       # Placement default; supply separately
    milk-vision-v4n.pt       # Motion default; supply separately
  Dataset/
    videos/                # Supply input videos separately
```

Paths containing spaces must be enclosed in quotes.

Default model and diagnostic paths are resolved relative to the project layout,
independent of the terminal directory. Explicit relative paths are resolved from
the terminal's current directory. Placement uses **v4s**, while motion uses **v4n**.
The frame extractor saves images to `Scripts\fridge items`.

The Git ignore rules cover weights, several video formats, ZIPs, resume state,
and diagnostics, but do not exclude every generated image/label directory.
Review `git status` before committing generated data or previews.

## Installation

The scripts use Python 3.10+ features. Use an environment with the dependencies
installed; the local project uses `.venv`. If creating a fresh environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
python -m pip install --upgrade pip
python -m pip install -r Scripts/requirements.txt
```

`requirements.txt` specifies `ultralytics>=8.4.121,<9` and `lap>=0.5.12,<1`.
Ultralytics' dependencies include PyTorch, NumPy, and OpenCV.

The commands below use `python` from the active environment. Without activation,
replace `python` with `.\.venv\Scripts\python.exe` in PowerShell.

## Frame-extraction tool

The frame extractor lives in `Scripts/tools/`. Run it as a file using the commands
below, or as a module from inside `Scripts`:

```powershell
python -m tools.extract_fridge_frames --help
```

From the parent `Yolo` directory, use
`python -m Scripts.tools.extract_fridge_frames` instead.

## `tools/extract_fridge_frames.py`

### Purpose

This script scans a folder for MP4, MOV, AVI, MKV, M4V, and WebM videos. For each video it displays duration, frame rate, and frame count, then asks for one or more start/end time ranges. It merges overlapping ranges and writes every frame within each selected closed interval as a JPEG.

Only direct child files are scanned, in natural filename order (`vid_9` before
`vid_10`). Frame numbers are zero-based; range selection uses `frame_number / FPS`
as the timestamp.

Frame names preserve the video name and original frame number:

```text
vid_12_frame_00000149.jpg
```

The fixed output folder remains directly inside `Scripts`, outside `tools/`:

```text
Scripts\fridge items
```

Matching JPEG names are overwritten. Other existing files are left unchanged.
Videos sharing the same filename stem also share output frame names, so use unique
stems. There is no output-directory CLI option.

### Commands

Let the script ask for the video folder:

```powershell
python .\Scripts\tools\extract_fridge_frames.py
```

Supply the exact video folder immediately:

```powershell
python .\Scripts\tools\extract_fridge_frames.py --video-dir "D:\fridge videos"
```

Change JPEG quality from its default of `95`:

```powershell
python .\Scripts\tools\extract_fridge_frames.py `
  --video-dir "D:\fridge videos" `
  --jpeg-quality 90
```

At each start-time prompt:

- Enter a number such as `4.14` or `4.14s`, then enter the end time.
- Enter `n` to finish selecting ranges, extract them, and continue to the next video.
- Enter `q` to stop the script while retaining frames already written. Ranges
  selected for the current video are not extracted until you enter `n`, so
  quitting during selection discards those pending selections.

`--jpeg-quality` accepts integers from `1` to `100`.

Show all arguments with:

```powershell
python .\Scripts\tools\extract_fridge_frames.py --help
```

## Milk placement verification

### Purpose

Track **milk only** and print its signed net movement across a horizontal gate:

```text
net_delta = inward milk crossings - outward milk crossings
```

Adding one milk prints `1`, removing one prints `-1`, and removing then returning
milk prints `0`. Each crossing contributes independently: an initial outward event
or a final inward event is counted even without a paired opposite-direction event.
There are no hand detections, hand/milk associations, payload windows, or paired
hand transactions in this verifier. Other utilities in this repository can still
use the model's hand class.

This measures movement across the gate, not whether milk was released onto a shelf.
A temporarily invisible milk item is not automatically treated as removed.
Crossing position uses the vertical midpoint of the tracked **bounding box**,
not the segmentation contour. The return value is a signed integer, not the older
`True`/`False` test for exactly one placement.

### Commands

From the parent project directory, with the default `Models/milk-vision-v4s.pt` available:

```powershell
python -m Scripts.milk_vision placement `
  --video ".\Dataset\videos\milk_added.mp4"
```

Successful stdout is exactly one integer line. Every successful run also lists
confirmed milk crossings on stderr, sorted by `frame_index`, including each
event's track ID and direction. An empty list is shown if no crossings occur.
Frame indices are zero-based. No debug files are created without `--debug`.
Specify inference and evidence thresholds independently:

```powershell
python -m Scripts.milk_vision placement `
  --video ".\Dataset\videos\vid_11.mp4" `
  --track-conf 0.10 `
  --conf 0.40 `
  --gate-y 0.20
```

To use a different checkpoint or CPU inference:

```powershell
python -m Scripts.milk_vision placement `
  --video ".\Dataset\videos\vid_11.mp4" `
  --model ".\Models\milk-vision-v2.pt" `
  --device cpu
```

### Two confidence thresholds

- `--track-conf 0.10` controls which YOLO milk detections reach BoT-SORT.
- `--conf 0.40` controls which tracked detections can supply crossing evidence.
- A matched detection below `--conf` supplies no side/crossing evidence. It
  refreshes track liveness, but gate history can still expire as described below.
- Milk boxes without a confirmed tracker ID cannot supply crossing observations.
- The checkpoint needs only a class named `milk` (case-insensitive); the inference
  class filter prevents other classes from reaching BoT-SORT.

The bundled `milk_vision/milk_botsort.yaml` uses high/low matching thresholds of `0.25` and
`0.10`, a `new_track_thresh` of `0.40`, and a 30-frame lost-track buffer. It
disables global motion compensation because the camera is fixed and disables
appearance re-identification because this is a milk-only crossing task.
The new-track threshold is a separate tracker setting: changing `--conf` changes
crossing acceptance, not the YAML's new-track threshold. A weak match can maintain
an established ID, but does not create a new tentative track below `0.40`.

The script requires `0 < --track-conf <= --conf <= 1`. Raising `--track-conf` above
the low-confidence interval removes BoT-SORT's opportunity to use weak matches.

### Gate confirmation and timeouts

With the default gate at `0.20` of image height and hysteresis `0.04`, confident
centers with normalized y `<= 0.16` are outside and those with y `>= 0.24` are
inside (image y increases downward). The band between them is undecided.

At approximately 30 FPS, the script requires three qualifying confident
observations to establish a side, then three on the opposite side to emit a
crossing. These observations need not be adjacent video frames. Weak matches and
dead-band observations do not advance or cancel the candidate count. Observing
the already-confirmed side with sufficient confidence cancels a tentative change.

Track liveness and confident evidence have separate timeouts:

| Constant | Default | Effect |
| --- | --- | --- |
| `CROSSING_CONFIRM_SECONDS` | `0.10` | Required confident observations, converted to frames with a minimum of two. |
| `TRACK_GAP_SECONDS` | `0.50` | Reset gate history after this gap without any matched milk detection. |
| `EVIDENCE_GAP_SECONDS` | `1.00` | Reset gate history after this gap without a confident milk observation, even if weak matches continue. |

At 30 FPS those limits are 15 and 30 frames; a reset occurs when the elapsed frame
gap is **greater than** the limit. A confident observation inside the dead band
refreshes the evidence timer, but contributes no side confirmation. After a reset,
the next confirmed side initializes history; it does not invent a crossing.
BoT-SORT has its own separate
30-frame lost-ID buffer.

### Debugging

```powershell
python -m Scripts.milk_vision placement `
  --video ".\Dataset\videos\vid_11.mp4" `
  --debug `
  --debug-dir ".\Scripts\diagnostics"
```

Stdout stays a single integer. Debug mode adds configuration and net delta to the
crossing events already shown on stderr, without duplicating those events.
`--debug-dir` is optional; without it files are written beside this
script. Repeating a run overwrites the same debug filenames in that directory:

```text
vid_11_tracking_debug.csv
vid_11_annotated_debug.mp4
```

The CSV contains one row per matched milk track and placeholder rows for frames
with none. It records confidence, boxes, centers, `crossing_eligible`,
`observation_use` (`crossing_evidence` or `tracking_only`), gate states, candidate
counts, reset reasons when observed, and crossing events. Weak detections have no
`observed_zone`; their coordinates remain available for diagnosis.

The debug video draws confident milk tracks in green and weak tracks in orange
with a `TRACKING ONLY` label. It also shows the gate, hysteresis boundaries, track
IDs, confidence, state, and emitted milk events. These are **bounding boxes, not
segmentation masks or polygon outlines**. The video uses the input FPS and decoded
frame dimensions, does not copy audio, and does not open a viewer window. All
annotation and writing is implemented in `milk_vision/diagnostics.py`.

An empty debug row does **not** prove YOLO saw nothing: raw detections rejected
before confirmed tracker output are not present in this CSV.

### Limitations

- Use a fixed camera with a gate that milk actually passes through.
- An item needs confident observations on both sides under one track ID before
  that crossing can be counted.
- Occlusion, duplicate milk detections, identity switches, and poor gate placement
  can still produce missed or extra events.
- Counting milk crossings does not use hand evidence to suppress false positives.
- OpenCV read failure after at least one decoded frame is treated as end-of-video;
  the verifier does not prove that every expected frame was decoded.
- Validate thresholds on labeled videos containing placements, removals, returns,
  and no-change cases, rather than tuning to a single clip.

Input/inference failures caught by the verifier produce an `ERROR:` message on
stderr and exit code `1`, rather than being reported as zero inventory change.
This includes missing paths, videos that cannot open or yield no frames, invalid
FPS, and a missing milk class or tracker configuration. Argument-parser errors
(such as a missing `--video` or an out-of-range CLI value) print usage and exit
with code `2`. Neither is a successful integer result.

### Arguments

| Argument | Required/default | Function |
| --- | --- | --- |
| `--video PATH` | Required | Input video. |
| `--model PATH` | `Models/milk-vision-v4s.pt` | Checkpoint containing a milk class. |
| `--conf FLOAT` | `0.40` | Minimum confidence for crossing observations. |
| `--track-conf FLOAT` | `0.10` | Lower detector threshold feeding BoT-SORT. |
| `--gate-y FLOAT` | `0.20` | Normalized gate position, from `0.05` to `0.95`. |
| `--device VALUE` | Automatic | Ultralytics device such as `0` or `cpu`. |
| `--debug` | Off | Write CSV and annotated video. |
| `--debug-dir PATH` | Beside script | Output directory for `--debug`. |

Show the argument help with `python -m Scripts.milk_vision placement --help`.

## Milk motion estimation (experimental, no boundary lines)

Estimate milk motion from per-object y-coordinate histories. This separate script
does not change the gate-based verifier. Its output is a **motion-derived estimate**:
vertical movement cannot by itself establish that an item crossed into the fridge
or was released onto a shelf.

```powershell
.\.venv\Scripts\python.exe -m Scripts.milk_vision motion `
  --video ".\Dataset\videos\add_milk\milk_added.mp4"
```

The default checkpoint is `Models/milk-vision-v4n.pt`. Override it with `--model`;
use `--device cpu` for CPU inference. Run `--help` for every option. This uses the
existing dependencies from `Scripts/requirements.txt`. Matplotlib is only needed
when `--debug` is enabled; ordinary runs create no CSV, JSON or PNG files.

### Coordinate storage and regression

1. Run YOLO's milk-only detector on every decoded frame at `--track-conf 0.10`,
   then feed its boxes into the existing BoT-SORT configuration.
2. Collect **all post-NMS detector boxes** in memory, including weak detections and boxes that
   have no confirmed tracker ID. Retain `y1`, `y2`, midpoint y in pixels, normalized
   midpoint y, confidence, time, frame index, and any matched track ID. These are
   the original detector coordinates, not the tracker's Kalman-filtered boxes.
3. Treat every detector result admitted by `--track-conf 0.10` as one logical
   milk trajectory. Tracker IDs remain in debug output but do not control grouping
   or eligibility; detections without IDs are used normally. Split the trajectory
   when observations are more than `--max-gap-seconds 0.50` apart. This mode
   uses the highest-confidence box when a frame contains multiple milk detections,
   because simultaneous boxes cannot define one unambiguous y coordinate. The
   unused boxes remain labelled in debug output.
4. Fit a local line `y(t) = a + b*t` in a centered `--window-seconds 0.60` window.
   Use a univariate Theil-Sen estimate: median pairwise slope and median intercept.
   At most 61 uniformly spaced observations determine the slope. There are no
   observation-count, fitting-duration, inlier, or residual acceptance thresholds.
   Two time-distinct observations are the mathematical minimum needed to define a
   slope; an isolated observation cannot contribute fitted motion. Residual is
   retained as diagnostic information only.
5. Difference successive accepted fitted centers:
   `delta_y = fitted_y_now - fitted_y_previous`,
   `velocity = delta_y / elapsed_seconds`. Classify movement from this velocity
   alone when its magnitude is at least `--min-speed 0.04` frame heights/second.
   The local slope remains diagnostic and does not gate movement. A moving step
   starts or continues a directional candidate; neutral steps retain that candidate,
   and clear opposite motion replaces an unfinished candidate. Candidate displacement
   is the net fitted position change from its starting coordinate rather than a sum
   of individual steps. Confirm an
   event as soon as its net travel reaches `--min-displacement 0.08` frame
   heights. There is no minimum moving-step count or candidate duration. There is
   also no fixed y boundary.

Image y increases downward. The default `--inward down` maps positive motion to
entering and negative motion to leaving, following the existing verifier's camera
convention. Use `--inward up` if the camera geometry is reversed.

One continuous direction is counted once per logical-trajectory segment. A pause preserves a
pending movement but does not re-arm a direction already counted. Only a complete
opposite event re-arms it; an incomplete opposite candidate is discarded if the
original direction resumes. A sufficiently long evidence gap starts a new segment.
A single global best-fit line would hide
reversals; the local fits preserve sufficiently long entering/leaving episodes.
Centered windows use future observations, so this implementation analyzes the
complete video before emitting events. It is not a zero-latency live detector.

### Outputs

Successful stdout is one integer: inferred inward episodes minus outward episodes.
Stderr immediately shows startup and model-loading messages, then frame progress
approximately every two seconds (after the current frame finishes). A completion
summary distinguishes no detections, insufficient fit evidence, and no qualifying
motion. Stderr also lists motion events. CPU inference can take longer than the
video's playback duration. No output files are needed to run the algorithm.

Add `--debug` to save the diagnostic files after processing the whole video:

```powershell
python -m Scripts.milk_vision motion `
  --video ".\Dataset\videos\add_milk\milk_added.mp4" `
  --debug `
  --output-dir ".\debug\milk_motion"
```

`--output-dir` defaults to `debug/milk_motion` under the project directory and is
only used with `--debug`. Debug mode prints the artifact paths on stderr and
writes these three files; repeating the same video stem in the same directory
overwrites them. The script does not open a video or plot window.

- `<video_stem>_motion.csv`: raw detections, fit status, fitted coordinates,
  residuals, local slopes, finite differences, velocities, and events. Includes
  window count/span and directional candidate travel/duration/hit count.
- `<video_stem>_motion.json`: video metadata, thresholds, events, and net estimate.
- `<video_stem>_motion.png`: every saved raw y position labelled by status:
  accepted with a tracker ID (green circles), accepted without a tracker ID (grey
  squares), an unused same-frame box (orange diamond), or a mathematically
  unfittable isolated observation (cross).
  Coloured trajectories show accepted fits;
  vertical event markers identify counted movements. Accepted fits still need
  to pass the event checks. The lower panel shows fitted velocity over time.

CSV rows exist only for detections; missing frame indices are not interpolated.
Raw differences are between consecutive selected detections in the same logical
segment; fitted differences are between accepted fits. Each uses its actual time
gap. Every coordinate admitted by `--track-conf` is fitted regardless of its
confidence or tracker-ID status. A rejected-fit category is not used: constructed
fits are not rejected by statistical quality checks.
The y axis on the position plot increases downward like the video. Dashed lines on
the velocity plot show speed thresholds, not spatial boundaries.

### Feasibility and limits

This is useful when entry and exit have consistent vertical motion under a fixed
camera. Robust regression reduces isolated coordinate spikes; it cannot reject a
false object that forms a coherent moving track. Bounding-box size changes,
occlusion, rotation, camera motion, and moving milk already inside the fridge can
still cause missed or extra counts. Tracker identity changes do not split motion,
but multiple milk detections in one frame are reduced to the highest-confidence
box and may select the wrong carton.
Long detection gaps can split one physical movement into multiple counted segments.
Fast turns can be smoothed away; small windows preserve turns but suppress less
noise, while larger windows need more sustained movement. Constant-FPS timestamps
come from frame index / OpenCV FPS; variable-frame-rate timing is approximate.
OpenCV cannot reliably distinguish EOF from a midstream decode failure.

Treat all defaults as starting points, not calibrated accuracy claims. Evaluate
entering, leaving, round trips, stationary milk, repositioning, and non-milk
videos, and inspect event directions/times as well as net totals: two wrong events
can cancel to a correct zero. The existing gate verifier may be more selective
when a fixed physical entrance corresponds well to its boundary.

Initial spot checks with `milk-vision-v4n.pt` and the default motion thresholds
produced the following results. These are individual clips, not accuracy estimates:

| Folder / clip | Inferred events | Net |
| --- | --- | --- |
| `add_milk/milk_added.mp4` | inward | `+1` |
| `remove_milk/video_20260904_203519.mp4` | outward | `-1` |
| `add_and_remove_milk/video_20260904_204755.mp4` | inward, outward | `0` |
| `reposition_milk_or_do_nothing/video_20260904_205222.mp4` | inward, outward | `0` |

The last row demonstrates why net totals alone are insufficient: the no-change
category still produced two directional events. Artifacts from these runs are
under `debug/milk_motion/smoke`. The complete repository test suite passed 70
tests, including 14 tests for motion regression, filtering, gaps, reversals,
independent tracks, coordinate retention, argument validation, and artifacts.

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
