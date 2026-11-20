# Milk motion estimation

[Quick start](../README.md) · [Placement](placement.md) · [Motion](motion.md) · [Frame extraction](frame-extraction.md) · [Development](development.md) · [Observations](observations.md) · [Jev](jev-experiment.md)

Commands assume the terminal is in the parent `Yolo` directory unless stated otherwise.

Estimate milk motion from one logical y-coordinate trajectory. This command uses
a separate algorithm from placement. Its output is a **motion-derived estimate**:
vertical movement cannot by itself establish that an item crossed into the fridge
or was released onto a shelf.

```powershell
.\.venv\Scripts\python.exe -m Scripts.milk_vision motion `
  --video ".\Dataset\videos\milk\net 1\milk_added.mp4"
```

The default checkpoint is `Models/milk-vision-v4n.pt`. Override it with `--model`;
use `--device cpu` for CPU inference. Run `--help` for every option. This uses the
existing dependencies from `Scripts/requirements.txt`. Matplotlib is only needed
when `--debug` is enabled; ordinary runs create no CSV, JSON or PNG files.

## Coordinate storage and regression

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

## Outputs

Successful stdout is one integer: inferred inward episodes minus outward episodes.
Stderr immediately shows startup and model-loading messages, then frame progress
approximately every two seconds (after the current frame finishes). A completion
summary distinguishes no detections, insufficient fit evidence, and no qualifying
motion. Stderr also lists motion events. CPU inference can take longer than the
video's playback duration. No output files are needed to run the algorithm.

Add `--debug` to save the diagnostic files after processing the whole video:

```powershell
python -m Scripts.milk_vision motion `
  --video ".\Dataset\videos\milk\net 1\milk_added.mp4" `
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
gap. Each selected highest-confidence box is eligible for fitting regardless of
tracker-ID status; other same-frame boxes remain diagnostic only. Constructed
fits are not rejected by statistical quality checks.
The y axis on the position plot increases downward like the video. Dashed lines on
the velocity plot show speed thresholds, not spatial boundaries.

## Feasibility and limits

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

Model weights, videos, and diagnostic artifacts are supplied separately and are
not included in this repository. Use the commands above to evaluate the current
implementation on your own labeled clips; passing automated tests does not
establish model accuracy.

## Full command reference

Use `--help` for current arguments and defaults:

```powershell
python -m Scripts.milk_vision motion --help
```

For multiple videos, replace `--video` with `--folder` and see
[folder execution](development.md#folder-verification-commands).

## Shared observations

Motion uses the raw-detection projection of the shared collector, while placement
uses its tracked boxes. The algorithm receives fresh rows so fitted values and
candidate diagnostics never modify shared evidence. `--cache-dir .\cache\observations`
enables reuse for standalone commands; uncached ordinary runs retain their existing
file behavior. See [Shared observations](observations.md) and the
[three-method evaluator](development.md#three-method-shared-observation-comparison).
