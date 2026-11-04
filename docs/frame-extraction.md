# Frame extraction

[Quick start](../README.md) · [Placement](placement.md) · [Motion](motion.md) · [Frame extraction](frame-extraction.md) · [Development](development.md)

Commands assume the terminal is in the parent `Yolo` directory unless stated otherwise.

## Purpose

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

## Commands

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

## Module entry points

From the parent `Yolo` directory:

```powershell
python -m Scripts.tools.extract_fridge_frames --help
```

From inside `Scripts`:

```powershell
python -m tools.extract_fridge_frames --help
```
