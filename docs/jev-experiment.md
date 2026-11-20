# Jev trajectory experiment

Run commands from the parent `Yolo` directory with its existing `.venv`.
The current dataset evaluator asks Jev to choose one of five directional episode
sequences; Python calculates the net change. Start with
[shared dataset evaluation](#shared-collection-and-dataset-evaluation), or use the
[Colab lab](colab.md) for GPU testing. Jev does not replace the motion command or
train a new model.

The standalone commands below preserve the earlier net-classification experiment.
For the current episode framework, add `--framework episodes` when preparing a
case; standalone preparation otherwise defaults to `net`.

## Prepare locally

```powershell
.\.venv\Scripts\python.exe -m Scripts.tools.jev_experiment prepare `
  --video ".\Dataset\videos\milk\net 1\example.mp4" `
  --expected-net 1 --device cpu `
  --output ".\debug\jev\cases\example.json"
```

Replace the example path with an actual labelled video. Alternatively, reuse
an existing motion debug CSV with `--csv path\to\clip_motion.csv` instead of
`--video`. CSV provenance is marked unknown: use fresh video extraction for a
controlled comparison if the detector/settings of the old CSV are uncertain.
Preparation never calls the API. It saves a local envelope containing the exact
`request`, its hash, source hash, optional label, detector provenance, and a
recomputed motion baseline using the same detections and default thresholds.
Outputs refuse to overwrite earlier files.

The request contains only raw timestamps, normalized bounding-box midpoint y,
and detection confidence, plus fixed task instructions. No filenames, labels,
scenarios, tracker IDs, derived features, or baseline results are sent. All
detections admitted by YOLO's default 0.10 confidence cutoff remain present,
including simultaneous boxes and detections without tracker IDs. No temporal
subsampling, interpolation, or smoothing is applied to Jev's input. Detection
gaps remain visible through timestamps; clip end time and trailing empty frames
are unavailable with this three-field representation. No detections produces
an empty list; the model must still choose, so inspect confidence and errors.
Detector confidence and Jev decision confidence are different measurements.

Fresh extraction uses the existing raw-detection collector and milk-vision-v4n
checkpoint, including its tracker internally, but excludes tracker data from the
request. `--detector`, `--track-conf`, and `--device` override extraction settings.
`--inward up` reverses camera convention for both Jev and the local baseline.

## Call Jev

Create a key at <https://console.typesafe.ai/keys>. Set `TYPESAFE_API_KEY` in the
local environment; do not put it in a request file, source code, or Git. The
runner uses Python's standard-library HTTPS client, so no extra SDK is required.

```powershell
$cases = (Get-ChildItem .\debug\jev\cases\*.json).FullName
.\.venv\Scripts\python.exe -m Scripts.tools.jev_experiment run @cases `
  --output-dir .\debug\jev\results
```

Each case sends one request to `https://api.typesafe.ai/v1/systemone`. Only the
envelope's `request` is transmitted. The default pinned model is `jev-1.14.0`;
change `--model` during preparation to evaluate another version. Preserve the
original cases/results and use new directories for reruns. The 45-second HTTP
timeout can be changed with `--timeout`. Failures stop the run with a nonzero
exit; prior successes remain saved. There are no automatic retries or silent
truncation. Supply only remaining case paths to continue a partially finished
batch. A timeout may still have consumed a billable request server-side.

The normal model context limit applies. If TypeSafe rejects a long request,
record the failure rather than shortening it silently. Request byte count is
reported at preparation time; it is not a token count. Long-context mode is not
enabled by this runner. Save the returned model version and token usage when
comparing experiments; a syntactically valid decision does not prove accuracy.

## Compare results

```powershell
$results = (Get-ChildItem .\debug\jev\results\*.result.json).FullName
.\.venv\Scripts\python.exe -m Scripts.tools.jev_experiment report @results
```

The report gives per-class totals, correct counts, and confusion counts for Jev
and the motion baseline, including baseline outputs outside the three classes.
Results retain full probabilities, Jev confidence, latency, response model,
usage, and the exact input. Unlabelled cases are excluded from accuracy counts.
Do not convert uncertainty to a zero prediction. API failures are incomplete
cases, not correct classifications; compare completed counts with planned cases.

Start with a development batch of five clips per class, covering different
scenarios. The first locally prepared cases may be a smaller smoke test. These
are development examples, not a held-out accuracy claim. Fix instructions and
format using development clips, then freeze them before evaluating untouched
recordings. Keep related takes/sessions together across splits, and never send
the held-out labels or scenario descriptions. The current 133-clip milk dataset
has 84 zeros, so an always-zero classifier scores about 63%; use per-class
results, not overall accuracy alone.

API references: [Choice](https://docs.typesafe.ai/primitives/choice),
[HTTP API](https://docs.typesafe.ai/api/system-one),
[models and context limits](https://docs.typesafe.ai/models).

## Five-choice episode experiment

Use exactly these choices: `none`, `inward`, `outward`, `inward → outward`,
and `outward → inward`. Jev groups directional movement into episodes; Python
counts the selected inward and outward episodes and subtracts the counts.
The prompt contains no inventory reasoning, labels, or net values.

Reuse an existing case's exact full-precision observations without rerunning YOLO:

```powershell
.\.venv\Scripts\python.exe -m Scripts.tools.jev_experiment reframe `
  --case .\debug\jev\cases\pilot_02.json `
  --output .\debug\jev\episodes_v1\cases\pilot_02.json
```

Then use `run` and `report` as above with the new directories. Fresh video/CSV
preparation also accepts `--framework episodes`; the default remains `net` to
preserve the first experiment. Reframing preserves the model, camera convention,
source provenance, labels, and baseline while retaining the original case intact.
Results additionally save `predicted_episodes`, `inward_count`, and `outward_count`.
Optional local `expected_episodes` lists enable exact ordered-sequence comparison
in `report`; these labels are never sent to Jev. For the paired 15-video pilot,
these lists come from the dataset's `expected_events` (`in`/`out` mapped to
`inward`/`outward`). Those annotations describe labelled events, not independently
annotated visible trajectory intervals; interpret episode agreement accordingly.
Matching a zero net does not establish whether `none` or either round trip is right.

## Shared collection and dataset evaluation

All Jev algorithm decisions now live in `milk_vision/algorithms/jev.py`: selection
of fields, formatting, instructions, choices, HTTP transport, validation and the
Python count-to-net conversion. The experiment CLI delegates to that module.
The five-choice prompt is unchanged from the successful episode pilot.
`python-dotenv` loads `Scripts/.env` as a fallback; an existing environment value
wins. Keys are never persisted in requests, caches or reports.

For the complete supported dataset, use:

```powershell
.\.venv\Scripts\python.exe -m Scripts.tools.evaluate `
  --methods placement motion jev --include-negatives --device cpu `
  --model .\Models\milk-vision-v4s.pt
```

This selects 169 milk/negative clips and shares detector evidence with the other
methods. Jev still receives only timestamp, normalized y and confidence. Tracker
IDs, boxes, source paths, ground truth and baseline predictions are not transmitted.
The local cache also retains empty frames and video duration for other consumers;
these extra fields do not enter the frozen Jev request. No detections produces an
empty list, and Jev can select `none`; classify any inventory disagreement separately
from the observed-motion task. Existing inventory annotations do not establish
independent motion grouping accuracy. Jev supplies episode order, not frame bounds.

Dataset evaluation reuses observations and completed compatible predictions.
See [Shared observations](observations.md) and
[Development](development.md#three-method-shared-observation-comparison) for refresh,
resume, failure accounting, timing and credential behavior. Old experiment case
files are retained for historical replay; they lack complete per-frame/raw-tracked
information and cannot be promoted into full shared observation artifacts.
