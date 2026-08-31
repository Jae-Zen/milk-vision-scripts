"""Create multi-class YOLO segmentation labels and review images with YOLOE.

The input folder can be supplied with ``--source`` or entered interactively.
Each item class can have multiple text prompts. The script uses ``milk carton``
as its single default class when no class configuration is supplied and creates:

* ``segmentation_labels`` beside this script: one YOLO polygon label per image
* ``segmentation_previews`` beside this script: outlined review images

The official checkpoint is downloaded automatically by Ultralytics the first
time the script is run if ``yoloe-26n-seg.pt`` is not already available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
PREVIEW_COLORS = [
    (50, 220, 50),
    (255, 170, 40),
    (60, 90, 240),
    (220, 80, 220),
    (40, 210, 230),
    (230, 210, 70),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label one or more item classes with YOLOE segmentation."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Exact folder containing input images; if omitted, you will be prompted.",
    )
    parser.add_argument(
        "--model",
        default="yoloe-26n-seg.pt",
        help="YOLOE segmentation checkpoint (default: yoloe-26n-seg.pt).",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help=(
            "Folder for YOLO segmentation text files (default: "
            "segmentation_labels beside this script)."
        ),
    )
    parser.add_argument(
        "--previews-dir",
        type=Path,
        default=None,
        help=(
            "Folder for outlined review images (default: segmentation_previews "
            "beside this script)."
        ),
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        metavar="TEXT",
        help=(
            "Legacy single-class mode: all prompts describe class 0. Cannot be "
            "combined with --class (default: 'milk carton')."
        ),
    )
    parser.add_argument(
        "--class",
        dest="class_groups",
        action="append",
        nargs="+",
        default=None,
        metavar="TEXT",
        help=(
            "Define one item class. The first value is its output name and a "
            "prompt; remaining values are alternative prompts. Repeat --class "
            "for additional items."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence from 0 to 1 (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold used during prediction (default: 0.7).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example '0' for GPU or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Images per prediction batch (default: 8). Reduce if GPU memory is low.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=3,
        help="Preview outline width in pixels (default: 3).",
    )
    return parser.parse_args()


def build_class_schema(
    class_groups: list[list[str]] | None,
    legacy_prompts: list[str] | None,
) -> tuple[list[str], list[str], list[int]]:
    """Return output names, unique model prompts, and prompt-to-class IDs."""
    if class_groups and legacy_prompts:
        raise ValueError("Use either --class or --prompts, not both.")

    groups = class_groups if class_groups else [legacy_prompts or ["milk carton"]]
    class_names: list[str] = []
    model_prompts: list[str] = []
    prompt_class_ids: list[int] = []
    seen_class_names: set[str] = set()
    seen_prompts: dict[str, int] = {}

    for class_id, raw_group in enumerate(groups):
        values = [value.strip() for value in raw_group if value.strip()]
        if not values:
            raise ValueError("Each --class must contain a non-empty class name.")

        class_name = values[0]
        class_key = class_name.casefold()
        if class_key in seen_class_names:
            raise ValueError(f"Duplicate class name: {class_name!r}")
        seen_class_names.add(class_key)
        class_names.append(class_name)

        # The class name itself is always a prompt. Repeated aliases within the
        # same class are ignored, while a prompt shared by two classes is
        # rejected because its prediction would have an ambiguous output ID.
        for prompt in values:
            prompt_key = prompt.casefold()
            if prompt_key in seen_prompts:
                existing_class_id = seen_prompts[prompt_key]
                if existing_class_id != class_id:
                    raise ValueError(
                        f"Prompt {prompt!r} is assigned to both "
                        f"{class_names[existing_class_id]!r} and {class_name!r}."
                    )
                continue
            seen_prompts[prompt_key] = class_id
            model_prompts.append(prompt)
            prompt_class_ids.append(class_id)

    return class_names, model_prompts, prompt_class_ids


def ask_image_directory() -> Path:
    """Ask until the user enters an existing image folder."""
    while True:
        entered = (
            input("Enter the exact folder containing the images: ")
            .strip()
            .strip("\"'")
        )
        if not entered:
            print("Please enter a folder path.\n")
            continue

        source = Path(entered).expanduser().resolve()
        if source.is_dir():
            return source
        print(f"Folder does not exist: {source}\n")


def resolve_paths(args: argparse.Namespace) -> None:
    """Resolve user paths and apply script-relative output defaults."""
    args.source = (
        args.source.expanduser().resolve()
        if args.source is not None
        else ask_image_directory()
    )
    args.labels_dir = (
        args.labels_dir.expanduser().resolve()
        if args.labels_dir is not None
        else SCRIPT_DIR / "segmentation_labels"
    )
    args.previews_dir = (
        args.previews_dir.expanduser().resolve()
        if args.previews_dir is not None
        else SCRIPT_DIR / "segmentation_previews"
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.source.is_dir():
        raise ValueError(f"Source folder does not exist: {args.source}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than zero.")
    if args.batch <= 0:
        raise ValueError("--batch must be greater than zero.")
    if args.line_width <= 0:
        raise ValueError("--line-width must be greater than zero.")


def find_images(source: Path) -> list[Path]:
    images = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    # YOLO label filenames are stem-based, so two images such as item.jpg and
    # item.png would otherwise overwrite the same item.txt file.
    stems: dict[str, Path] = {}
    for image in images:
        key = image.stem.casefold()
        if key in stems:
            raise ValueError(
                "Two input images have the same filename stem and would share a "
                f"label file: {stems[key].name} and {image.name}"
            )
        stems[key] = image
    return images


def result_class_ids(result, prompt_class_ids: list[int]) -> list[int]:
    """Translate YOLOE prompt indices into the requested training class IDs."""
    if result.boxes is None:
        return []

    class_ids: list[int] = []
    for prompt_index_value in result.boxes.cls.cpu().tolist():
        prompt_index = int(prompt_index_value)
        if not 0 <= prompt_index < len(prompt_class_ids):
            raise RuntimeError(f"Model returned unknown prompt index {prompt_index}.")
        class_ids.append(prompt_class_ids[prompt_index])
    return class_ids


def write_yolo_label(label_path: Path, polygons: list, class_ids: list[int]) -> list[int]:
    """Write normalized polygons with their requested YOLO training class IDs."""
    if len(polygons) != len(class_ids):
        raise RuntimeError(
            f"Model returned {len(polygons)} masks but {len(class_ids)} class IDs."
        )

    rows: list[str] = []
    written_class_ids: list[int] = []
    for polygon, class_id in zip(polygons, class_ids, strict=True):
        if polygon is None or len(polygon) < 3:
            continue
        # xyn should already be normalized; clipping also protects against tiny
        # floating-point excursions outside the image bounds.
        values = polygon.reshape(-1).clip(0.0, 1.0)
        coordinates = " ".join(f"{float(value):.6f}" for value in values)
        rows.append(f"{class_id} {coordinates}")
        written_class_ids.append(class_id)

    contents = "\n".join(rows)
    if contents:
        contents += "\n"
    label_path.write_text(contents, encoding="utf-8")
    return written_class_ids


def draw_caption(image, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(origin[0], image.shape[1] - width - 5))
    y = max(height + 5, min(origin[1], image.shape[0] - baseline - 2))
    cv2.rectangle(image, (x, y - height - 5), (x + width + 5, y + baseline + 2), color, -1)
    cv2.putText(image, text, (x + 2, y - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def save_preview(
    result,
    preview_path: Path,
    line_width: int,
    class_ids: list[int],
    class_names: list[str],
) -> None:
    import cv2
    import numpy as np

    preview = result.orig_img.copy()
    polygons = [] if result.masks is None else result.masks.xy
    confidences = [] if result.boxes is None else result.boxes.conf.cpu().tolist()

    if len(polygons) != len(class_ids):
        raise RuntimeError(
            f"Model returned {len(polygons)} masks but {len(class_ids)} class IDs."
        )

    for index, polygon in enumerate(polygons):
        if polygon is None or len(polygon) < 3:
            continue
        class_id = class_ids[index]
        outline_color = PREVIEW_COLORS[class_id % len(PREVIEW_COLORS)]
        contour = np.rint(polygon).astype(np.int32)
        cv2.polylines(
            preview,
            [contour],
            isClosed=True,
            color=outline_color,
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )
        confidence = confidences[index] if index < len(confidences) else 0.0
        topmost = contour[contour[:, 1].argmin()]
        draw_caption(
            preview,
            f"{class_names[class_id]} {confidence:.2f}",
            (int(topmost[0]), int(topmost[1]) - 4),
            outline_color,
        )

    if not polygons:
        draw_caption(preview, "No configured items detected", (8, 28), (80, 180, 255))

    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f"Could not write preview image: {preview_path}")


def batched(items: list[Path], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> int:
    args = parse_args()
    try:
        resolve_paths(args)
        validate_args(args)
        images = find_images(args.source)
        class_names, model_prompts, prompt_class_ids = build_class_schema(
            args.class_groups, args.prompts
        )
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled; no images were processed.")
        return 130
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    if not images:
        print(f"Error: no supported images found in {args.source}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLOE
    except ImportError:
        requirements_path = SCRIPT_DIR / "requirements.txt"
        print(
            "Error: Ultralytics is not installed. Run: "
            f'python -m pip install -r "{requirements_path}"',
            file=sys.stderr,
        )
        return 2

    args.labels_dir.mkdir(parents=True, exist_ok=True)
    args.previews_dir.mkdir(parents=True, exist_ok=True)
    class_map = "\n".join(
        f"{class_id}: {class_name}"
        for class_id, class_name in enumerate(class_names)
    )
    (args.labels_dir / "classes.txt").write_text(class_map + "\n", encoding="utf-8")

    print(f"Loading {args.model} ...")
    model = YOLOE(args.model)
    model.set_classes(model_prompts)
    print("Training classes:")
    for class_id, class_name in enumerate(class_names):
        prompts = [
            prompt
            for prompt, mapped_id in zip(model_prompts, prompt_class_ids, strict=True)
            if mapped_id == class_id
        ]
        print(f"  {class_id}: {class_name} <- {', '.join(prompts)}")

    processed = 0
    instances = 0
    images_with_detections = 0
    class_totals = [0] * len(class_names)
    predict_options = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "retina_masks": True,
        "agnostic_nms": True,
        "verbose": False,
    }
    if args.device is not None:
        predict_options["device"] = args.device

    for image_batch in batched(images, args.batch):
        results = model.predict(
            source=[str(image) for image in image_batch],
            batch=len(image_batch),
            **predict_options,
        )

        if len(results) != len(image_batch):
            raise RuntimeError(
                f"Model returned {len(results)} results for {len(image_batch)} images."
            )

        for image_path, result in zip(image_batch, results, strict=True):
            normalized_polygons = [] if result.masks is None else result.masks.xyn
            class_ids = result_class_ids(result, prompt_class_ids)
            written_class_ids = write_yolo_label(
                args.labels_dir / f"{image_path.stem}.txt",
                normalized_polygons,
                class_ids,
            )
            save_preview(
                result,
                args.previews_dir / image_path.name,
                args.line_width,
                class_ids,
                class_names,
            )

            count = len(written_class_ids)
            processed += 1
            instances += count
            images_with_detections += int(count > 0)
            for class_id in written_class_ids:
                class_totals[class_id] += 1
            print(
                f"[{processed:>{len(str(len(images)))}}/{len(images)}] "
                f"{image_path.name}: {count} item(s)"
            )

    print("\nFinished.")
    print(f"Images processed:       {processed}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Item instances:         {instances}")
    for class_id, (class_name, total) in enumerate(
        zip(class_names, class_totals, strict=True)
    ):
        print(f"  Class {class_id} ({class_name}): {total}")
    print(f"YOLO labels:            {args.labels_dir.resolve()}")
    print(f"Review images:          {args.previews_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
