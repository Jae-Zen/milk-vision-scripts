# Milk vision scripts

This folder contains six Python utilities:

- `auto_label_milk_dataset.py` uses the trained milk/hand segmentation model to create a resumable, upload-ready Ultralytics dataset and ZIP archive.
- `auto_label_milk_images.py` creates YOLO instance-segmentation labels and review images using a text-prompted YOLOE model.
- `extract_fridge_frames.py` interactively extracts selected time ranges from videos as JPEG frames.
- `yolo_segment.py` displays segmentation predictions from an existing segmentation model on images, videos, or cameras.
- `verify_milk_placement.py` prints signed net milk change from milk-only crossings.
- `run_multiple_verify_milk_placement.py` sequentially runs the verifier on every video in a folder.

The standalone `auto_label_milk_video.py` has been removed. The verifier writes
its own diagnostic video with `--debug`; it does not depend on a separate labeling
script. For segmentation-mask viewing or recording, use `yolo_segment.py`.

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
  Models/
    milk-vision-v3.pt       # Verifier default; supply separately
    milk-vision-v2.pt       # Dataset-labeling default; supply separately
    yoloe-26l-seg.pt         # Optional YOLOE checkpoint used in examples
  Dataset/
    videos/                # Supply input videos separately
    photos/                # Optional input-image folder
    photos_extracted_vid2-50/ # Dataset-labeling default input, if available
  sample_img/              # Optional images used in viewer/YOLOE examples
```

Paths containing spaces must be enclosed in quotes.

Default paths in the dataset labeler and verifier are resolved relative to their
script locations. Explicit relative paths are resolved from the terminal's
current directory. The dataset labeler still defaults to **v2**; changing the
verifier's default to **v3** did not change the other scripts.

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
Ultralytics' dependencies include PyTorch, NumPy, and OpenCV. GUI viewing needs
a working OpenCV display environment. The optional Raspberry Pi camera branch
also needs `picamera2`, which is not installed by this requirements file.

The commands below use `python` from the active environment. Without activation,
replace `python` with `.\.venv\Scripts\python.exe` in PowerShell.

## `auto_label_milk_dataset.py`

### Purpose

This script labels the extracted fridge frames with the closed-set
`Models\milk-vision-v2.pt` segmentation model. The required class mapping is fixed
and validated before dataset output is created; names and IDs must match exactly:

```text
0: milk
1: hand
```

It copies every supported source image unchanged into a deterministic random
train/validation split (approximately 90/10), writes matching normalized YOLO
polygon labels, creates an
empty label for an image with no detections, and packages the completed dataset as
a ZIP in Ultralytics dataset layout. It does not upload anything automatically
or create local mask previews.

Only images directly inside `--source` are scanned (BMP, JPEG, PNG, TIFF, WebP).
Subfolders are ignored, and duplicate filename stems are rejected regardless of
case. Validation size is rounded and clamped to leave at least one image in each
split when there are two or more images. A single image goes into training only.

Run it from the project root with all defaults, **only if the default source
folder and model exist**:

```powershell
.\.venv\Scripts\python.exe .\Scripts\auto_label_milk_dataset.py
```

The defaults are:

```text
source: Dataset\photos_extracted_vid2-50
model:  Models\milk-vision-v2.pt
output: Scripts\milk_vision_v2_ultralytics_dataset
conf:   0.25
iou:    0.70
imgsz:  640
seed:   42
split:  approximately 90% train / 10% validation
```

The default image folder is not created by this labeler. In particular,
`extract_fridge_frames.py` writes to `Scripts\fridge items`, not to
`Dataset\photos_extracted_vid2-50`. Point the labeler at the folder you actually
have, for example:

```powershell
python .\Scripts\auto_label_milk_dataset.py --source ".\Dataset\photos"
```

With the default output path, a successful run creates this archive for upload:

```text
Scripts\milk_vision_v2_ultralytics_dataset.zip
```

The archive contains:

```text
data.yaml
train\images
train\labels
val\images
val\labels
```

Interrupted runs retain completed image/label pairs. Run the same command again to
resume. The sibling `<output-name>.state.json` records source/model paths, the
model SHA-256 hash, source filenames/sizes/modification times, split, and inference
settings. Source image contents are not individually hashed. Pairs with a missing
or wrong-size image copy, or missing/invalid label text, are regenerated; copied
image contents are not compared byte-for-byte.

If these recorded settings or inputs change, use a new `--output` directory or
deliberately restart with `--overwrite`:

```powershell
.\.venv\Scripts\python.exe .\Scripts\auto_label_milk_dataset.py --overwrite
```

**Warning:** `--overwrite` deletes the entire selected output directory, its
sibling ZIP, and its sibling state file before starting again. Keep unrelated
files out of that directory, and repeat any `--source`, `--model`, or `--output`
overrides when restarting.

`--batch` and `--device` may be changed while resuming because they are not part of
the saved compatibility settings; bit-identical inference across devices is not
guaranteed.

Override performance or inference settings when needed:

```powershell
.\.venv\Scripts\python.exe .\Scripts\auto_label_milk_dataset.py `
  --device cpu `
  --batch 4 `
  --conf 0.25 `
  --imgsz 640
```

### Arguments

| Argument | Default | Function |
| --- | --- | --- |
| `--source PATH` | `Dataset\photos_extracted_vid2-50` | Existing folder of input images. |
| `--model PATH` | `Models\milk-vision-v2.pt` | Existing segmentation checkpoint with exactly `0: milk`, `1: hand`. |
| `--output PATH` | `Scripts\milk_vision_v2_ultralytics_dataset` | Dataset directory; ZIP and state file are siblings. |
| `--conf FLOAT` | `0.25` | Minimum prediction confidence from `0` to `1`. |
| `--iou FLOAT` | `0.70` | Prediction IoU threshold from `0` to `1`. |
| `--imgsz INTEGER` | `640` | Positive inference image size. |
| `--device VALUE` | Automatic | Inference device, such as `0` or `cpu`. |
| `--batch INTEGER` | `8` | Positive number of images per inference batch. |
| `--seed INTEGER` | `42` | Deterministic split seed. |
| `--val-fraction FLOAT` | `0.10` | Validation fraction, strictly between `0` and `1`. |
| `--overwrite` | Off | Delete existing output/state/archive and restart. |

Use `python .\Scripts\auto_label_milk_dataset.py --help` to see the CLI help.

These are model-generated labels, not reviewed ground truth. The random split is
by **image**, not source video, so neighboring frames from one video can appear
in both splits; this is not a held-out-video evaluation split.

## `auto_label_milk_images.py`

### Purpose

This is the batch auto-labeling script. It:

1. Reads every supported image directly inside one input folder.
2. Loads a YOLOE segmentation checkpoint.
3. Converts one or more text prompts per item class into YOLOE embeddings.
4. Segments matching objects.
5. Writes one YOLO segmentation `.txt` file per image.
6. Writes a review copy of every image with class-colored polygon outlines, item names, and confidence values.

Supported input formats are BMP, JPEG, PNG, TIFF, and WebP. Subfolders are not searched.

Duplicate filename stems are rejected regardless of case. Avoid source images
named `classes.jpg`, `classes.png`, etc.: their label would collide with the
`classes.txt` metadata file. This utility does not create a train/validation split,
`data.yaml`, or a ZIP; use `auto_label_milk_dataset.py` for the closed-set dataset
workflow.

Each repeated `--class` creates a distinct YOLO training class. Multiple phrases within that `--class` are aliases for the same item. Each label row has standard YOLO segmentation form:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Polygon coordinates are normalized from `0` to `1`. An image with no accepted detection receives an empty `.txt` file, which is a valid negative training sample. The labels folder also receives `classes.txt`, which records the numeric ID assigned to every configured item.

### Classes and alternative prompts

Repeat `--class` once for every different type of item. Within each occurrence:

- The first quoted value is the output class name and is also used as a YOLOE prompt.
- Every following quoted value is an alternative prompt for that same item.
- Class IDs follow command order, starting at `0`.

For example, this configuration creates three training classes:

```powershell
python .\Scripts\auto_label_milk_images.py `
  --source ".\sample_img" `
  --model ".\Models\yoloe-26l-seg.pt" `
  --class "milk container" "plastic milk jug" "milk bottle" `
  --class "juice carton" "juice box" "drink carton" `
  --class "yogurt cup" "yogurt tub" "small yogurt container" `
  --conf 0.10 `
  --imgsz 960 `
  --batch 1
```

The resulting mapping is:

```text
0: milk container
1: juice carton
2: yogurt cup
```

Therefore, detections produced by `"milk container"`, `"plastic milk jug"`, or `"milk bottle"` all become class `0`. Juice prompts become class `1`, and yogurt prompts become class `2`.

Do not assign the same prompt phrase to two different classes because the output class would be ambiguous. The script detects and rejects that configuration.

For backward compatibility, `--prompts` still defines alternative phrases for one class only:

```powershell
python .\Scripts\auto_label_milk_images.py `
  --source ".\sample_img" `
  --prompts "plastic milk jug" "milk bottle" "milk container"
```

This legacy form maps every supplied prompt to class `0`. Do not combine `--prompts` with `--class`. If neither option is supplied, the default is one class named `milk carton`.

### Input folder

If `--source` is omitted, the script asks you to enter or paste the exact image folder:

```powershell
python .\Scripts\auto_label_milk_images.py
```

To avoid the interactive question, pass the exact folder on the command line:

```powershell
python .\Scripts\auto_label_milk_images.py --source "D:\datasets\fridge frames"
```

### Default output locations

Unless overridden, output is always placed beside the Python script, independent of the terminal's current directory:

```text
Scripts\segmentation_labels
Scripts\segmentation_previews
```

Use different locations with `--labels-dir` and `--previews-dir`:

```powershell
python .\Scripts\auto_label_milk_images.py `
  --source "D:\datasets\fridge frames" `
  --labels-dir "D:\datasets\milk-yolo\labels" `
  --previews-dir "D:\datasets\milk-yolo\previews"
```

Explicit relative output paths are resolved from the terminal's current directory. Existing files with matching names are overwritten; unrelated files are retained.


If the model name is given without a local path, Ultralytics downloads an official missing checkpoint automatically:

```powershell
python .\Scripts\auto_label_milk_images.py `
  --source ".\sample_img" `
  --model yoloe-26x-seg.pt
```

### Arguments

| Argument | Default | Function |
| --- | --- | --- |
| `--source PATH` | Interactive question | Exact folder containing the input images. |
| `--model PATH_OR_NAME` | `yoloe-26n-seg.pt` | Local checkpoint path or official Ultralytics checkpoint name. |
| `--labels-dir PATH` | `Scripts\segmentation_labels` | Destination for YOLO polygon `.txt` files. |
| `--previews-dir PATH` | `Scripts\segmentation_previews` | Destination for outlined review images. |
| `--class NAME [ALIAS ...]` | None | Defines one output item and its alternative prompts. Repeat for multiple item classes. |
| `--prompts TEXT [TEXT ...]` | None | Backward-compatible single-class aliases; all are mapped to class `0`. |
| `--conf FLOAT` | `0.25` | Minimum accepted confidence from `0` to `1`. Lower values increase recall and false positives. |
| `--iou FLOAT` | `0.7` | Intersection-over-union threshold used during prediction. |
| `--imgsz INTEGER` | `640` | Inference image size. Larger sizes can help small objects but use more memory and time. |
| `--device VALUE` | Automatic | PyTorch device, such as `0` for the first CUDA GPU or `cpu`. |
| `--batch INTEGER` | `8` | Images processed together. Reduce to `1` if GPU memory is insufficient. |
| `--line-width INTEGER` | `3` | Polygon-outline width in review images. |

Show the built-in argument help with:

```powershell
python .\Scripts\auto_label_milk_images.py --help
```

## `extract_fridge_frames.py`

### Purpose

This script scans a folder for MP4, MOV, AVI, MKV, M4V, and WebM videos. For each video it displays duration, frame rate, and frame count, then asks for one or more start/end time ranges. It merges overlapping ranges and writes every frame within each selected closed interval as a JPEG.

Only direct child files are scanned, in natural filename order (`vid_9` before
`vid_10`). Frame numbers are zero-based; range selection uses `frame_number / FPS`
as the timestamp.

Frame names preserve the video name and original frame number:

```text
vid_12_frame_00000149.jpg
```

The fixed output folder is beside the script:

```text
Scripts\fridge items
```

Matching JPEG names are overwritten. Other existing files are left unchanged.
Videos sharing the same filename stem also share output frame names, so use unique
stems. There is no output-directory CLI option.

### Commands

Let the script ask for the video folder:

```powershell
python .\Scripts\extract_fridge_frames.py
```

Supply the exact video folder immediately:

```powershell
python .\Scripts\extract_fridge_frames.py --video-dir "D:\fridge videos"
```

Change JPEG quality from its default of `95`:

```powershell
python .\Scripts\extract_fridge_frames.py `
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
python .\Scripts\extract_fridge_frames.py --help
```

## `yolo_segment.py`

### Purpose

This is an interactive viewer for an existing YOLO segmentation model. It accepts one image, a folder of images, a video, a USB camera, or a Raspberry Pi camera. It overlays translucent masks, polygon outlines, class names, confidences, and an object count in an OpenCV window.

Image inputs support `.jpg`, `.jpeg`, `.png`, and `.bmp` with lowercase or fully
uppercase extensions. Video inputs support lowercase `.avi`, `.mov`, `.mp4`,
`.mkv`, and `.wmv`. Folder scanning is non-recursive and unsorted. This viewer
does not have the same format list as the image labelers or frame extractor.

This viewer does **not** create YOLO training label files. Use `auto_label_milk_images.py` for label export. It also does not configure YOLOE text prompts, so it is best suited to an already trained, closed-set segmentation checkpoint such as `Models\milk-vision-v2.pt`.

### Required arguments

- `--model PATH` is the exact segmentation checkpoint path.
- `--source VALUE` is an image path, image-folder path, video path, `usb0`, `usb1`, and so on, or `picamera0`.

### Commands

View one image:

```powershell
python .\Scripts\yolo_segment.py `
  --model ".\Models\milk-vision-v2.pt" `
  --source ".\sample_img\vid_20_frame_00000133.jpg" `
  --thresh 0.40
```

View every supported image directly inside a folder:

```powershell
python .\Scripts\yolo_segment.py `
  --model ".\Models\milk-vision-v2.pt" `
  --source ".\sample_img" `
  --thresh 0.30
```

View a video resized to 1280 by 720:

```powershell
python .\Scripts\yolo_segment.py `
  --model ".\Models\milk-vision-v2.pt" `
  --source "D:\fridge videos\vid_1.mp4" `
  --resolution 1280x720
```

Use the first USB camera:

```powershell
python .\Scripts\yolo_segment.py `
  --model ".\Models\milk-vision-v2.pt" `
  --source usb0 `
  --resolution 1280x720
```

Record a video or USB-camera session as `demo1.avi` in the terminal's current directory:

```powershell
python .\Scripts\yolo_segment.py `
  --model ".\Models\milk-vision-v2.pt" `
  --source usb0 `
  --resolution 1280x720 `
  --record
```

`--record` requires `--resolution` and works only with video or USB-camera sources.
The recording uses MJPG at a fixed **30 FPS**, regardless of input FPS, so source
videos with another frame rate will not retain their original playback timing.
It does not copy audio, opens a preview window, and overwrites `demo1.avi`.
For a segmentation-labeled copy of an existing video, use the same `--record`
command with `--source` set to the video path instead of `usb0`.

### Viewer-specific caveats

- `--thresh` is only a display filter applied after model inference (strictly
  greater than the supplied value). It is not passed to YOLO as `conf`, so lowering
  it cannot recover predictions already filtered out by the model call.
- The Pi camera branch requires `--resolution` in the current implementation,
  plus `picamera2` and compatible hardware. Although a `picameraN` index is parsed,
  the code constructs `Picamera2()` without using that index. Pi recording is not
  supported by `--record`.

### Window controls

- `Q`: quit.
- `S`: pause; press another key to resume.
- `P`: save the currently displayed frame as `capture.png` in the terminal's current directory.
- Image and image-folder modes wait for a key press before advancing to the next image.

### Arguments

| Argument | Required/default | Function |
| --- | --- | --- |
| `--model PATH` | Required | Segmentation checkpoint to load. |
| `--source VALUE` | Required | Image, folder, video, USB camera, or Pi camera source. |
| `--thresh FLOAT` | `0.5` | Minimum confidence for displaying a returned prediction. |
| `--resolution WIDTHxHEIGHT` | Original resolution except Pi branch | Display/inference resize, such as `1280x720`; required for recording and Pi input. |
| `--record` | Off | Record video/USB results to `demo1.avi`. |

Show all arguments with:

```powershell
python .\Scripts\yolo_segment.py --help
```

## `verify_milk_placement.py`

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

From the parent project directory, with the default `Models/milk-vision-v3.pt` available:

```powershell
python .\Scripts\verify_milk_placement.py `
  --video ".\Dataset\videos\milk_added.mp4"
```

Successful stdout is exactly one integer line. Every successful run also lists
confirmed milk crossings on stderr, sorted by `frame_index`, including each
event's track ID and direction. An empty list is shown if no crossings occur.
Frame indices are zero-based. No debug files are created without `--debug`.
Specify inference and evidence thresholds independently:

```powershell
python .\Scripts\verify_milk_placement.py `
  --video ".\Dataset\videos\vid_11.mp4" `
  --track-conf 0.10 `
  --conf 0.40 `
  --gate-y 0.20
```

To use a different checkpoint or CPU inference:

```powershell
python .\Scripts\verify_milk_placement.py `
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

The bundled `milk_botsort.yaml` uses high/low matching thresholds of `0.25` and
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
python .\Scripts\verify_milk_placement.py `
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
annotation and writing is implemented inside `verify_milk_placement.py`.

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
| `--model PATH` | `Models/milk-vision-v3.pt` | Checkpoint containing a milk class. |
| `--conf FLOAT` | `0.40` | Minimum confidence for crossing observations. |
| `--track-conf FLOAT` | `0.10` | Lower detector threshold feeding BoT-SORT. |
| `--gate-y FLOAT` | `0.20` | Normalized gate position, from `0.05` to `0.95`. |
| `--device VALUE` | Automatic | Ultralytics device such as `0` or `cpu`. |
| `--debug` | Off | Write CSV and annotated video. |
| `--debug-dir PATH` | Beside script | Output directory for `--debug`. |

Show the argument help with `python .\Scripts\verify_milk_placement.py --help`.

## `run_multiple_verify_milk_placement.py`

### Purpose

This script processes a folder of videos. Unlike the single-video verifier, its
input argument is `--folder`, not `--video`.

### Command

```powershell
python .\Scripts\run_multiple_verify_milk_placement.py `
  --folder "D:\fridge videos"
```

### Argument

| Argument | Required/default | Function |
| --- | --- | --- |
| `--folder PATH` | Required | Existing folder containing all videos to process. |

### Processing behavior

- Every supported video directly inside `--folder` is processed; there is no
  49-video, 50-video, or other count limit.
- Videos are processed sequentially in case-insensitive natural filename order,
  so `vid_2.mp4` comes before `vid_10.mp4`.
- Supported extensions are `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.webm`, and
  `.wmv`. Non-video files and subfolders are ignored.
- If one video fails, the script reports it and continues through every remaining
  video. Afterward, it returns a non-zero exit status if any video failed.

For each file, the wrapper launches a separate verifier process using the same
Python interpreter. It passes one file at a time to the single-video verifier,
whose input necessarily remains `--video`. That is an internal implementation
detail; when running this multiple-video script, use only `--folder`. The
single-video verifier's defaults apply, including `Models\milk-vision-v3.pt`, and
debug output is **off**. The wrapper does not write an aggregate report.

Show the wrapper's argument help with:

```powershell
python .\Scripts\run_multiple_verify_milk_placement.py --help
```

## Tests

```powershell
python -m unittest discover -s Scripts/tests -v
```

Tests cover dataset discovery/splitting, polygon formatting, class validation,
resume compatibility, ZIP layout, signed milk counting, confidence gating,
weak-match liveness, evidence timeouts, debug CSV/CLI output, and a mocked
inference pipeline. A BoT-SORT test exercises real weak-score matching on an
existing track when the inference dependencies are installed. No test requires a
trained model file. Dataset tests require NumPy; the real tracker test is skipped
if its imports are unavailable. The suite does not test actual model accuracy or
visually render/inspect debug videos.

## Optional YOLOE text encoder: `mobileclip2_b.ts`

If a YOLOE run downloads `mobileclip2_b.ts`, it is a TorchScript text encoder,
not TypeScript source. The installed Ultralytics MobileCLIP2 loader uses it to
convert text prompts into embeddings; it is separate from the YOLOE `.pt` model.
It is not a supplied repository file, and it need not already exist in the project
root. Its location depends on asset lookup/download behavior.

Removing a downloaded copy can require another download on a later text-prompted
run. The trained closed-set milk models used by the dataset labeler and verifier
do not need this text encoder for their normal inference workflow.

## Troubleshooting

### `Ultralytics is not installed`

Activate the environment used during installation, then run:

```powershell
python -m pip install -r .\Scripts\requirements.txt
```

### CUDA out-of-memory error

Reduce the batch and/or image size:

```powershell
python .\Scripts\auto_label_milk_images.py `
  --source ".\sample_img" `
  --batch 1 `
  --imgsz 640
```

Alternatively force CPU inference:

```powershell
python .\Scripts\auto_label_milk_images.py `
  --source ".\sample_img" `
  --device cpu `
  --batch 1
```

### Very few milk-container detections

Use visually accurate prompt nouns such as `"plastic milk jug"` and `"milk bottle"`, try `--conf 0.10`, then compare a larger YOLOE checkpoint. Always inspect the preview images before using generated labels for training.
