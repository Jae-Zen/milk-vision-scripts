# Shared video observations

[Quick start](../README.md) · [Development](development.md) · [Placement](placement.md) · [Motion](motion.md) · [Jev](jev-experiment.md)

Run commands from `Yolo` with its existing local `.venv` (Python 3.11+).
Observation collection is independent of algorithm thresholds, Jev prompts and
ground truth. One detector/tracker extraction supplies all three methods.

## Collect once

```powershell
.\.venv\Scripts\python.exe -m Scripts.tools.collect_detections `
  --include-negatives --device cpu --model .\Models\milk-vision-v4s.pt
```

The command validates annotation coverage and selects 133 milk clips plus 36
other-object negatives when `--include-negatives` is present. Cheese labels are
outside milk evaluation. Use `--scenario`, `--limit`, or `--list` to inspect a
smaller selection. `--root` and `--manifest` select another labelled dataset.
The default detector confidence cutoff is `--track-conf 0.10`.

Extraction is sequential, with a fresh subprocess/model/tracker per video.
Progress is printed during collection. Completed compatible videos are reused
after interruption. A collection report under `debug/collection` records errors,
artifact paths, detection counts and elapsed times; `--output` chooses a new
report path. Return codes are 0 for success, 1 for processing failures, 2 for
invalid input and 130 for cancellation.

## What is saved

Each completed JSON artifact contains a schema version, completion flag,
extraction identity, data hash, metadata and every decoded frame:

- Metadata: source/model paths, FPS, decoded frame count, duration and confidence cutoff.
- Frame: zero-based index, timestamp (index/FPS), dimensions and detection tuples.
- Raw detection: original post-NMS box, detector confidence, source detection index
  and optional matched tracker ID. Untracked and simultaneous boxes are preserved.
- Tracked detection: tracker-produced box clipped to image bounds exactly as
  Ultralytics `Results.update` does, plus confidence, ID, class and frame index.
  Its association to the raw detection is preserved through the source index/ID.
- Empty frames contain empty detection lists, rather than disappearing.

Frames/pixels are not cached. Algorithm fits, predicted events, labels, scenario
descriptions and API credentials do not belong in these artifacts.

Placement reads tracked boxes and IDs; motion reads original detector boxes;
Jev receives only `[timestamp_seconds, normalized_y, detection_confidence]`.
Algorithm-specific rows are fresh projections. Neither motion analysis nor Jev
formatting modifies the common observations. Jev does not receive the metadata,
frame dimensions, identities or empty-frame records under the frozen prompt.

## Compatibility and refresh

The default cache is `Yolo/cache/observations`. Override it with `--cache-dir`.
Identity includes video/checkpoint content hashes, tracker YAML, detector cutoff,
device selection, collector source/schema version, relevant installed dependency
versions and CPU thread environment settings. Compatible artifacts are validated
before loading. A changed identity gets a distinct directory. Corrupt or incomplete
artifacts raise errors; they are not silently treated as zero detections.

`--refresh-observations` creates a new version even when identity matches. Earlier
completed versions remain available. Completed writes are atomic; interrupted
workers cannot publish a partially written successful artifact. Temporary pending
files are not considered cache hits. Existing compatible artifacts require no
YOLO inference; dependency identity still uses the configured local environment.

Successfully decoding a video with no milk detections is valid empty evidence.
Failure to open/decode any frames is an error. OpenCV cannot reliably distinguish
normal end-of-file from every midstream decode failure; decoded counts and source
fingerprints are retained. Timestamps remain approximate for variable-FPS videos.

## Reuse and inspect

```powershell
# All methods reuse the same observation artifact with matched settings.
.\.venv\Scripts\python.exe -m Scripts.tools.evaluate `
  --methods placement motion jev --include-negatives --device cpu

# Single-video motion/placement can opt into the same cache.
.\.venv\Scripts\python.exe -m Scripts.milk_vision motion `
  --video ".\Dataset\videos\milk\net 1\milk_added.mp4" `
  --model .\Models\milk-vision-v4s.pt --device cpu `
  --cache-dir .\cache\observations --debug
```

Standalone commands retain their existing checkpoints unless explicitly changed.
Both the model and extraction settings must match for a cache hit. Standalone
commands without `--cache-dir` retain their ordinary no-cache behavior.
Motion debug graphs/CSV use the loaded raw rows. Placement debug video decodes
the original pixels and joins them to saved frame indices, without rerunning YOLO.

Evaluation stores prediction versions separately under the cache's sibling
`predictions` directory. Repeating an evaluation with a new report path reuses
compatible results; `--rerun` requests fresh predictions. The report includes the
exact observation artifact path/hash. Prediction compatibility currently includes
the combined implementation hashes and settings for all algorithms, so changing
one algorithm can invalidate other methods' predictions too. Observation reuse
has its own extraction identity and is unaffected by algorithm-only changes.
In [Colab](colab.md), **Fresh predictions** passes `--rerun`; **Resume** permits
compatible prediction reuse. Neither mode forces fresh YOLO collection.
See [evaluation details](development.md#three-method-shared-observation-comparison)
for failures, API limits and the distinction between inventory and motion scores.
