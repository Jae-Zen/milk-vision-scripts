# Milk placement verification

[Quick start](../README.md) · [Placement](placement.md) · [Motion](motion.md) · [Frame extraction](frame-extraction.md) · [Development](development.md) · [Observations](observations.md) · [Jev](jev-experiment.md)

Commands assume the terminal is in the parent `Yolo` directory unless stated otherwise.

## Purpose

Track **milk only** and print its signed net movement across a horizontal gate:

```text
net_delta = inward milk crossings - outward milk crossings
```

Adding one milk prints `1`, removing one prints `-1`, and removing then returning
milk prints `0`. Each crossing contributes independently: an initial outward event
or a final inward event is counted even without a paired opposite-direction event.
There are no hand detections, hand/milk associations, payload windows, or paired
hand transactions in this verifier. The remaining verification commands filter
inference to the model's milk class.

This measures movement across the gate, not whether milk was released onto a shelf.
A temporarily invisible milk item is not automatically treated as removed.
Crossing position uses the vertical midpoint of the tracked **bounding box**,
not the segmentation contour. The return value is a signed integer, not the older
`True`/`False` test for exactly one placement.

## Commands

From the parent project directory, with the default `Models/milk-vision-v4s.pt` available:

```powershell
python -m Scripts.milk_vision placement `
  --video ".\Dataset\videos\milk\net 1\milk_added.mp4"
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

## Two confidence thresholds

- `--track-conf 0.10` controls which YOLO milk detections reach BoT-SORT.
- `--conf 0.40` controls which tracked detections can supply crossing evidence.
- A matched detection below `--conf` supplies no side/crossing evidence. It
  refreshes track liveness, but gate history can still expire as described below.
- Milk boxes without a confirmed tracker ID cannot supply crossing observations.
- The checkpoint needs only a class named `milk` (case-insensitive); the inference
  class filter prevents other classes from reaching BoT-SORT.

The bundled `milk_vision/observations/milk_botsort.yaml` uses high/low matching thresholds of `0.25` and
`0.10`, a `new_track_thresh` of `0.40`, and a 30-frame lost-track buffer. It
disables global motion compensation because the camera is fixed and disables
appearance re-identification because this is a milk-only crossing task.
The new-track threshold is a separate tracker setting: changing `--conf` changes
crossing acceptance, not the YAML's new-track threshold. A weak match can maintain
an established ID, but does not create a new tentative track below `0.40`.

The script requires `0 < --track-conf <= --conf <= 1`. Raising `--track-conf` above
the low-confidence interval removes BoT-SORT's opportunity to use weak matches.

## Gate confirmation and timeouts

With the default gate at `0.27` of image height and hysteresis `0.04`, confident
centers with normalized y `<= 0.23` are outside and those with y `>= 0.31` are
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

## Debugging

```powershell
python -m Scripts.milk_vision placement `
  --video ".\Dataset\videos\vid_11.mp4" `
  --debug `
  --debug-dir ".\Scripts\diagnostics"
```

Stdout stays a single integer. Debug mode adds configuration and net delta to the
crossing events already shown on stderr, without duplicating those events.
`--debug-dir` is optional; without it files are written directly inside `Scripts`,
outside the `milk_vision` package. Repeating a run overwrites the same debug filenames:

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

## Limitations

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
(such as omitting both `--video` and `--folder`, or an out-of-range CLI value)
print usage and exit with code `2`. Neither is a successful integer result.

## Full command reference

Use `--help` for current arguments and defaults:

```powershell
python -m Scripts.milk_vision placement --help
```

For multiple videos, replace `--video` with `--folder` and see
[folder execution](development.md#folder-verification-commands).

## Shared observations

Placement now consumes tracked boxes from the same collector used by motion and
Jev. Empty frames are retained for expiry, and original raw boxes are saved
separately for the other algorithms. Crossing thresholds and confidence gating
remain unchanged. Add `--cache-dir .\cache\observations` to reuse observations
for a single-video/folder command; without it ordinary runs do not create caches.
Debug video rendering reopens source frames and never reruns YOLO for the overlay.
See [Shared observations](observations.md) and [evaluation](development.md#three-method-shared-observation-comparison).
