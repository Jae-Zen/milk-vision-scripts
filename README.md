# YOLO fridge-image scripts

This folder contains seven Python utilities:

- `auto_label_milk_dataset.py` uses the trained milk/hand segmentation model to create a resumable, upload-ready Ultralytics dataset and ZIP archive.
- `auto_label_milk_images.py` creates YOLO instance-segmentation labels and review images using a text-prompted YOLOE model.
- `auto_label_milk_video.py` writes a labeled copy of a video using an existing YOLO segmentation model.
- `extract_fridge_frames.py` interactively extracts selected time ranges from videos as JPEG frames.
- `yolo_segment.py` displays segmentation predictions from an existing segmentation model on images, videos, or cameras.
- `verify_milk_placement.py` prints signed net milk change from milk-only crossings.
- `run_multiple_verify_milk_placement.py` runs the verifier over the local video directory.

This repository contains only the scripts, tests, and configuration. Models,
datasets, generated debug media, and labeling state are not uploaded. To preserve
the default relative paths, clone it as `Scripts` inside your project directory:

```bash
git clone https://github.com/Jae-Zen/milk-vision-scripts.git Scripts
```

The commands below assume the terminal is in the parent project directory:

```text
Yolo/
  Scripts/                 # This Git repository
  Models/milk-vision-v2.pt  # Supply separately
  Dataset/videos/          # Supply separately
```

Paths containing spaces must be enclosed in quotes.

## Installation

```bash
python -m pip install --upgrade pip
python -m pip install -r Scripts/requirements.txt
```

`requirements.txt` installs Ultralytics 8.x and its tracker assignment solver. Ultralytics also supplies the other packages needed by these scripts, including PyTorch, NumPy, and OpenCV.

## `auto_label_milk_dataset.py`

### Purpose

This script labels the extracted fridge frames with the closed-set
`models\milk-vision-v2.pt` segmentation model. The required class mapping is fixed
and validated before any output is created:

```text
0: milk
1: hand
```

It copies every source image unchanged into a deterministic random 90/10
train/validation split, writes matching normalized YOLO polygon labels, creates an
empty label for an image with no detections, and packages the completed dataset as
a ZIP for Ultralytics Platform upload. It does not create local mask previews.

Run it from the project root with all defaults:

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
split:  90% train / 10% validation
```

After a successful run, upload this archive to Ultralytics:

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
resume. The resume state binds the exact source inventory, model hash, split, and
inference settings. If any setting or input changes, deliberately discard the old
dataset and start again with `--overwrite`:

```powershell
.\.venv\Scripts\python.exe .\Scripts\auto_label_milk_dataset.py --overwrite
```

`--batch` and `--device` may be changed while resuming because they control how
inference executes rather than which predictions are accepted.

Override performance or inference settings when needed:

```powershell
.\.venv\Scripts\python.exe .\Scripts\auto_label_milk_dataset.py `
  --device cpu `
  --batch 4 `
  --conf 0.25 `
  --imgsz 640
```

Available path and split overrides are `--source`, `--model`, `--output`,
`--seed`, and `--val-fraction`. Run with `--help` for the complete argument list.

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
python .\scripts\auto_label_milk_images.py `
  --source ".\sample_img" `
  --model ".\models\yoloe-26l-seg.pt" `
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
python .\scripts\auto_label_milk_images.py `
  --source ".\sample_img" `
  --prompts "plastic milk jug" "milk bottle" "milk container"
```

This legacy form maps every supplied prompt to class `0`. Do not combine `--prompts` with `--class`. If neither option is supplied, the default is one class named `milk carton`.

### Input folder

If `--source` is omitted, the script asks you to enter or paste the exact image folder:

```powershell
python .\scripts\auto_label_milk_images.py
```

To avoid the interactive question, pass the exact folder on the command line:

```powershell
python .\scripts\auto_label_milk_images.py --source "D:\datasets\fridge frames"
```

### Default output locations

Unless overridden, output is always placed beside the Python script, independent of the terminal's current directory:

```text
scripts\segmentation_labels
scripts\segmentation_previews
```

Use different locations with `--labels-dir` and `--previews-dir`:

```powershell
python .\scripts\auto_label_milk_images.py `
  --source "D:\datasets\fridge frames" `
  --labels-dir "D:\datasets\milk-yolo\labels" `
  --previews-dir "D:\datasets\milk-yolo\previews"
```

Explicit relative output paths are resolved from the terminal's current directory. Existing files with matching names are overwritten; unrelated files are retained.


If the model name is given without a local path, Ultralytics downloads an official missing checkpoint automatically:

```powershell
python .\scripts\auto_label_milk_images.py `
  --source ".\sample_img" `
  --model yoloe-26x-seg.pt
```

### Arguments

| Argument | Default | Function |
| --- | --- | --- |
| `--source PATH` | Interactive question | Exact folder containing the input images. |
| `--model PATH_OR_NAME` | `yoloe-26n-seg.pt` | Local checkpoint path or official Ultralytics checkpoint name. |
| `--labels-dir PATH` | `scripts\segmentation_labels` | Destination for YOLO polygon `.txt` files. |
| `--previews-dir PATH` | `scripts\segmentation_previews` | Destination for outlined review images. |
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
python .\scripts\auto_label_milk_images.py --help
```

## `auto_label_milk_video.py`

### Purpose

This script accepts one video and one trained YOLO segmentation model. It writes
an MP4 copy beside the script with translucent masks, polygon outlines, class
names, and confidence values. The output name is
`<input-name>_segmented.mp4`. OpenCV video output does not copy the input audio.

Pass the video and model paths positionally:

```powershell
python .\scripts\auto_label_milk_video.py `
  ".\Dataset\videos\vid_1.mp4" `
  ".\Models\milk-vision-v2.pt"
```

The equivalent named form is:

```powershell
python .\scripts\auto_label_milk_video.py `
  --video ".\Dataset\videos\vid_1.mp4" `
  --model ".\Models\milk-vision-v2.pt" `
  --conf 0.30 `
  --line-width 4 `
  --mask-alpha 0.25
```

| Argument | Default | Function |
| --- | --- | --- |
| `VIDEO_PATH` / `--video PATH` | Required | Input video file. |
| `MODEL_PATH` / `--model PATH` | Required | Trained YOLO segmentation checkpoint. |
| `--conf FLOAT` | `0.25` | Minimum displayed prediction confidence. |
| `--iou FLOAT` | `0.7` | Intersection-over-union threshold used during prediction. |
| `--imgsz INTEGER` | `640` | Inference image size in pixels. |
| `--device VALUE` | Automatic | PyTorch device, such as `0` or `cpu`. |
| `--line-width INTEGER` | `3` | Segmentation polygon outline width. |
| `--mask-alpha FLOAT` | `0.30` | Translucent mask opacity from `0` to `1`. |

Show all arguments with:

```powershell
python .\scripts\auto_label_milk_video.py --help
```

## `extract_fridge_frames.py`

### Purpose

This script scans a folder for MP4, MOV, AVI, MKV, M4V, and WebM videos. For each video it displays duration, frame rate, and frame count, then asks for one or more start/end time ranges. It merges overlapping ranges and writes every frame within each selected closed interval as a JPEG.

Frame names preserve the video name and original frame number:

```text
vid_12_frame_00000149.jpg
```

The fixed output folder is beside the script:

```text
scripts\fridge items
```

Matching JPEG names are overwritten. Other existing files are left unchanged.

### Commands

Let the script ask for the video folder:

```powershell
python .\scripts\extract_fridge_frames.py
```

Supply the exact video folder immediately:

```powershell
python .\scripts\extract_fridge_frames.py --video-dir "D:\fridge videos"
```

Change JPEG quality from its default of `95`:

```powershell
python .\scripts\extract_fridge_frames.py `
  --video-dir "D:\fridge videos" `
  --jpeg-quality 90
```

At each start-time prompt:

- Enter a number such as `4.14` or `4.14s`, then enter the end time.
- Enter `n` to finish the current video and continue to the next one.
- Enter `q` to stop the script while retaining frames already written.

Show all arguments with:

```powershell
python .\scripts\extract_fridge_frames.py --help
```

## `yolo_segment.py`

### Purpose

This is an interactive viewer for an existing YOLO segmentation model. It accepts one image, a folder of images, a video, a USB camera, or a Raspberry Pi camera. It overlays translucent masks, polygon outlines, class names, confidences, and an object count in an OpenCV window.

This viewer does **not** create YOLO training label files. Use `auto_label_milk_images.py` for label export. It also does not configure YOLOE text prompts, so it is best suited to an already trained, closed-set segmentation checkpoint such as `models\milk-vision-v2.pt`.

### Required arguments

- `--model PATH` is the exact segmentation checkpoint path.
- `--source VALUE` is an image path, image-folder path, video path, `usb0`, `usb1`, and so on, or `picamera0`.

### Commands

View one image:

```powershell
python .\scripts\yolo_segment.py `
  --model ".\models\milk-vision-v2.pt" `
  --source ".\sample_img\vid_20_frame_00000133.jpg" `
  --thresh 0.40
```

View every supported image directly inside a folder:

```powershell
python .\scripts\yolo_segment.py `
  --model ".\models\milk-vision-v2.pt" `
  --source ".\sample_img" `
  --thresh 0.30
```

View a video resized to 1280 by 720:

```powershell
python .\scripts\yolo_segment.py `
  --model ".\models\milk-vision-v2.pt" `
  --source "D:\fridge videos\vid_1.mp4" `
  --resolution 1280x720
```

Use the first USB camera:

```powershell
python .\scripts\yolo_segment.py `
  --model ".\models\milk-vision-v2.pt" `
  --source usb0 `
  --resolution 1280x720
```

Record a video or USB-camera session as `demo1.avi` in the terminal's current directory:

```powershell
python .\scripts\yolo_segment.py `
  --model ".\models\milk-vision-v2.pt" `
  --source usb0 `
  --resolution 1280x720 `
  --record
```

`--record` requires `--resolution` and works only with video or USB-camera sources.

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
| `--resolution WIDTHxHEIGHT` | Original resolution | Display/inference resize, such as `1280x720`. |
| `--record` | Off | Record video/USB results to `demo1.avi`. |

Show all arguments with:

```powershell
python .\scripts\yolo_segment.py --help
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

### Commands

From the parent project directory, with `Models/milk-vision-v2.pt` available:

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

### Two confidence thresholds

- `--track-conf 0.10` controls which YOLO milk detections reach BoT-SORT.
- `--conf 0.40` controls which tracked detections can supply crossing evidence.
- A matched detection below `--conf` updates only track liveness, not the stable
  zone, candidate zone/count, or net delta.
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
centers above the outer boundary are outside and those below the inner boundary
are inside. The band between them is undecided.

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

At 30 FPS those gaps are 15 and 30 frames. After a reset, the next confirmed side
initializes history; it does not invent a crossing. BoT-SORT has its own separate
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
IDs, confidence, state, and emitted milk events.

An empty debug row does **not** prove YOLO saw nothing: raw detections rejected
before confirmed tracker output are not present in this CSV.

### Limitations

- Use a fixed camera with a gate that milk actually passes through.
- An item needs confident observations on both sides under one track ID before
  that crossing can be counted.
- Occlusion, duplicate milk detections, identity switches, and poor gate placement
  can still produce missed or extra events.
- Counting milk crossings does not use hand evidence to suppress false positives.
- Validate thresholds on labeled videos containing placements, removals, returns,
  and no-change cases, rather than tuning to a single clip.

Invalid paths, unreadable videos, invalid FPS, incompatible models, and invalid
thresholds produce an `ERROR:` message on stderr and a nonzero exit code rather
than being reported as zero inventory change.

### Arguments

| Argument | Required/default | Function |
| --- | --- | --- |
| `--video PATH` | Required | Input video. |
| `--model PATH` | `Models/milk-vision-v2.pt` | Checkpoint containing a milk class. |
| `--conf FLOAT` | `0.40` | Minimum confidence for crossing observations. |
| `--track-conf FLOAT` | `0.10` | Lower detector threshold feeding BoT-SORT. |
| `--gate-y FLOAT` | `0.20` | Normalized gate position, from `0.05` to `0.95`. |
| `--device VALUE` | Automatic | Ultralytics device such as `0` or `cpu`. |
| `--debug` | Off | Write CSV and annotated video. |
| `--debug-dir PATH` | Beside script | Output directory for `--debug`. |

### Tests

```powershell
python -m unittest discover -s Scripts/tests -v
```

Tests cover signed milk counting, confidence gating, weak-match liveness,
evidence timeouts, debug output, and the inference pipeline. An additional
BoT-SORT test exercises real low-score ID recovery when the inference
dependencies are installed. The core logic tests do not require a model file.

## `mobileclip2_b.ts`

`mobileclip2_b.ts` in the project root is not TypeScript code. Here, `.ts` means a TorchScript model. Ultralytics downloads this pretrained MobileCLIP2-B text encoder when YOLOE needs to turn prompts such as `"plastic milk jug"` into numerical embeddings for text-prompted detection and segmentation.

The local file is approximately 254 MB. It is separate from the YOLOE `.pt` checkpoints because the YOLOE model consumes the prompt embeddings produced by this encoder.

You may delete it if disk space matters, but the next text-prompted YOLOE run will need it and Ultralytics will normally download it again. Keep it if you plan to continue testing `auto_label_milk_images.py`, especially if downloads are slow or the machine may be offline. A normal closed-set model such as `milk-vision-v2.pt` does not need this text encoder for inference.

## Troubleshooting

### `Ultralytics is not installed`

Activate the environment used during installation, then run:

```powershell
python -m pip install -r .\scripts\requirements.txt
```

### CUDA out-of-memory error

Reduce the batch and/or image size:

```powershell
python .\scripts\auto_label_milk_images.py `
  --source ".\sample_img" `
  --batch 1 `
  --imgsz 640
```

Alternatively force CPU inference:

```powershell
python .\scripts\auto_label_milk_images.py `
  --source ".\sample_img" `
  --device cpu `
  --batch 1
```

### Very few milk-container detections

Use visually accurate prompt nouns such as `"plastic milk jug"` and `"milk bottle"`, try `--conf 0.10`, then compare a larger YOLOE checkpoint. Always inspect the preview images before using generated labels for training.
