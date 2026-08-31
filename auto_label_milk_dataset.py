"""Build an upload-ready Ultralytics segmentation dataset from fridge frames.

The default checkpoint is a closed-set segmentation model with the fixed class
mapping ``0: milk`` and ``1: hand``. Source images are copied unchanged into a
deterministic train/validation split, while predictions are written as normalized
YOLO polygons. A completed run also creates a ZIP archive ready for upload to the
Ultralytics Platform.

Runs are resumable. A state file beside the output directory records the model,
source inventory, split, and inference settings. If any of those inputs change,
use ``--overwrite`` to start a clean run rather than mixing incompatible labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE = PROJECT_DIR / "Dataset" / "photos_extracted_vid2-50"
DEFAULT_MODEL = PROJECT_DIR / "Models" / "milk-vision-v2.pt"
DEFAULT_OUTPUT = SCRIPT_DIR / "milk_vision_v2_ultralytics_dataset"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EXPECTED_CLASSES = {0: "milk", 1: "hand"}
STATE_VERSION = 1


@dataclass(frozen=True)
class DatasetItem:
    source: Path
    split: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-label fridge images with milk-vision-v2 and build an "
            "Ultralytics segmentation dataset plus upload ZIP."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Folder containing input images (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Closed-set segmentation checkpoint (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Dataset output folder (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum prediction confidence from 0 to 1 (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="IoU threshold used during prediction (default: 0.70).",
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
        help="Ultralytics device such as '0' or 'cpu' (default: automatic).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Images per prediction batch (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random split seed (default: 42).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Fraction of images assigned to validation (default: 0.10).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete this script's existing output/state/archive and start clean.",
    )
    return parser.parse_args(argv)


def resolve_args(args: argparse.Namespace) -> None:
    args.source = args.source.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    args.output = args.output.expanduser().resolve()


def validate_args(args: argparse.Namespace) -> None:
    if not args.source.is_dir():
        raise ValueError(f"Source folder does not exist: {args.source}")
    if not args.model.is_file():
        raise ValueError(f"Model file does not exist: {args.model}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than zero.")
    if args.batch <= 0:
        raise ValueError("--batch must be greater than zero.")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be greater than 0 and less than 1.")
    validate_safe_output(args.output, args.source, args.model)


def validate_safe_output(output: Path, source: Path, model: Path) -> None:
    """Reject broad or overlapping targets before an optional recursive delete."""
    protected = {Path(output.anchor), Path.home().resolve(), SCRIPT_DIR, PROJECT_DIR}
    if output in protected or output == output.parent:
        raise ValueError(f"Unsafe output folder: {output}")

    relevant_paths = (source, model, Path(__file__).resolve())
    for relevant in relevant_paths:
        if output == relevant or output in relevant.parents or relevant in output.parents:
            raise ValueError(
                f"Output folder must not overlap source, model, or script paths: {output}"
            )


def archive_path_for(output: Path) -> Path:
    return output.with_name(f"{output.name}.zip")


def state_path_for(output: Path) -> Path:
    return output.with_name(f"{output.name}.state.json")


def find_images(source: Path) -> list[Path]:
    images = sorted(
        (
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
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


def split_images(
    images: Sequence[Path], val_fraction: float, seed: int
) -> list[DatasetItem]:
    if not images:
        raise ValueError("At least one image is required.")
    if len(images) == 1:
        return [DatasetItem(source=images[0], split="train")]
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    val_count = round(len(shuffled) * val_fraction)
    val_count = max(1, min(len(shuffled) - 1, val_count))
    validation_names = {path.name.casefold() for path in shuffled[:val_count]}
    return [
        DatasetItem(
            source=image,
            split="val" if image.name.casefold() in validation_names else "train",
        )
        for image in images
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_state(args: argparse.Namespace, items: Sequence[DatasetItem]) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "source": str(args.source),
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "settings": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "classes": [EXPECTED_CLASSES[index] for index in sorted(EXPECTED_CLASSES)],
        },
        "images": [
            {
                "name": item.source.name,
                "size": item.source.stat().st_size,
                "mtime_ns": item.source.stat().st_mtime_ns,
                "split": item.split,
            }
            for item in items
        ],
    }


def atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(contents, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def safe_remove_outputs(output: Path, source: Path, model: Path) -> None:
    validate_safe_output(output, source, model)
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Output path exists but is not a folder: {output}")
        shutil.rmtree(output)
    archive_path_for(output).unlink(missing_ok=True)
    state_path_for(output).unlink(missing_ok=True)


def initialize_or_resume(
    args: argparse.Namespace, expected_state: dict[str, Any]
) -> bool:
    """Prepare output state and return True when resuming an existing run."""
    output = args.output
    archive = archive_path_for(output)
    state_path = state_path_for(output)

    if args.overwrite:
        safe_remove_outputs(output, args.source, args.model)

    if state_path.exists():
        try:
            current_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Could not read resume state {state_path}; use --overwrite: {error}"
            ) from error
        if current_state != expected_state:
            raise ValueError(
                "Existing resume state does not match the current source, model, "
                "split, or inference settings. Use --overwrite to start clean."
            )
        return True

    if output.exists() or archive.exists():
        raise ValueError(
            "Output or archive already exists without matching resume state. "
            "Use --overwrite to start clean."
        )

    atomic_write_text(
        state_path,
        json.dumps(expected_state, indent=2, sort_keys=True) + "\n",
    )
    return False


def dataset_yaml() -> str:
    return (
        "path: .\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: milk\n"
        "  1: hand\n"
    )


def prepare_dataset_directories(output: Path) -> None:
    for split in ("train", "val"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)
    atomic_write_text(output / "data.yaml", dataset_yaml())


def normalized_model_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        try:
            return {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError(f"Model contains invalid class metadata: {names!r}") from error
    if isinstance(names, (list, tuple)):
        return {index: str(name) for index, name in enumerate(names)}
    raise ValueError(f"Model contains invalid class metadata: {names!r}")


def validate_model(model: Any) -> None:
    if getattr(model, "task", None) != "segment":
        raise ValueError(
            f"Model task must be 'segment', not {getattr(model, 'task', None)!r}."
        )
    names = normalized_model_names(getattr(model, "names", None))
    if names != EXPECTED_CLASSES:
        raise ValueError(
            "Model classes must be exactly {0: 'milk', 1: 'hand'}; "
            f"found {names!r}."
        )


def format_yolo_label(
    polygons: Sequence[Any], class_ids: Sequence[int]
) -> tuple[str, list[int]]:
    """Serialize valid normalized polygons and return the classes written."""
    if len(polygons) != len(class_ids):
        raise RuntimeError(
            f"Model returned {len(polygons)} masks but {len(class_ids)} class IDs."
        )

    import numpy as np

    rows: list[str] = []
    written_class_ids: list[int] = []
    for polygon, class_id in zip(polygons, class_ids, strict=True):
        if class_id not in EXPECTED_CLASSES:
            raise RuntimeError(f"Model returned unexpected class index {class_id}.")
        if polygon is None:
            continue
        values = np.asarray(polygon, dtype=float).reshape(-1)
        if values.size < 6 or values.size % 2 != 0 or not np.isfinite(values).all():
            continue
        values = values.clip(0.0, 1.0)
        coordinates = " ".join(f"{float(value):.6f}" for value in values)
        rows.append(f"{class_id} {coordinates}")
        written_class_ids.append(class_id)

    contents = "\n".join(rows)
    if contents:
        contents += "\n"
    return contents, written_class_ids


def result_annotations(result: Any) -> tuple[Sequence[Any], list[int]]:
    polygons = [] if result.masks is None else result.masks.xyn
    class_ids = (
        []
        if result.boxes is None
        else [int(value) for value in result.boxes.cls.cpu().tolist()]
    )
    if len(polygons) != len(class_ids):
        raise RuntimeError(
            "The model returned different numbers of segmentation masks and classes."
        )
    return polygons, class_ids


def destination_paths(output: Path, item: DatasetItem) -> tuple[Path, Path]:
    return (
        output / item.split / "images" / item.source.name,
        output / item.split / "labels" / f"{item.source.stem}.txt",
    )


def read_label_counts(label_path: Path) -> list[int]:
    counts = [0] * len(EXPECTED_CLASSES)
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except (ValueError, IndexError) as error:
            raise ValueError(
                f"Invalid YOLO label at {label_path}:{line_number}"
            ) from error
        if class_id not in EXPECTED_CLASSES:
            raise ValueError(
                f"Unexpected class {class_id} at {label_path}:{line_number}"
            )
        if (
            len(coordinates) < 6
            or len(coordinates) % 2 != 0
            or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates)
        ):
            raise ValueError(
                f"Invalid polygon coordinates at {label_path}:{line_number}"
            )
        counts[class_id] += 1
    return counts


def completed_item_counts(output: Path, item: DatasetItem) -> list[int] | None:
    image_path, label_path = destination_paths(output, item)
    if (
        not image_path.is_file()
        or image_path.stat().st_size != item.source.stat().st_size
        or not label_path.is_file()
    ):
        return None
    try:
        return read_label_counts(label_path)
    except (OSError, ValueError):
        return None


def batched(items: Sequence[DatasetItem], size: int) -> Iterable[Sequence[DatasetItem]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def verify_complete_dataset(
    output: Path, items: Sequence[DatasetItem]
) -> tuple[list[int], int]:
    class_totals = [0] * len(EXPECTED_CLASSES)
    negatives = 0
    for item in items:
        counts = completed_item_counts(output, item)
        if counts is None:
            raise RuntimeError(f"Dataset item is incomplete: {item.source.name}")
        for class_id, count in enumerate(counts):
            class_totals[class_id] += count
        negatives += int(sum(counts) == 0)
    return class_totals, negatives


def create_archive(output: Path) -> Path:
    archive = archive_path_for(output)
    temporary = archive.with_name(f".{archive.name}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as bundle:
            bundle.write(output / "data.yaml", "data.yaml")
            for split in ("train", "val"):
                for category in ("images", "labels"):
                    folder = output / split / category
                    bundle.writestr(f"{split}/{category}/", "")
                    files = sorted(
                        (path for path in folder.iterdir() if path.is_file()),
                        key=lambda path: path.name.casefold(),
                    )
                    for path in files:
                        bundle.write(path, f"{split}/{category}/{path.name}")
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def run(args: argparse.Namespace) -> None:
    resolve_args(args)
    validate_args(args)
    images = find_images(args.source)
    if not images:
        raise ValueError(f"No supported images found in {args.source}")
    items = split_images(images, args.val_fraction, args.seed)

    try:
        from ultralytics import YOLO
    except ImportError as error:
        requirements = SCRIPT_DIR / "requirements.txt"
        raise RuntimeError(
            f'Ultralytics is not installed. Run: python -m pip install -r "{requirements}"'
        ) from error

    print(f"Loading segmentation model: {args.model}")
    model = YOLO(str(args.model))
    validate_model(model)

    expected_state = build_state(args, items)
    resumed = initialize_or_resume(args, expected_state)
    prepare_dataset_directories(args.output)

    train_count = sum(item.split == "train" for item in items)
    val_count = len(items) - train_count
    print("Classes: 0=milk, 1=hand")
    print(f"Images: {len(items)} ({train_count} train, {val_count} val)")
    print("Resuming matching output." if resumed else "Starting a new output dataset.")

    pending: list[DatasetItem] = []
    skipped = 0
    for item in items:
        if completed_item_counts(args.output, item) is None:
            pending.append(item)
        else:
            skipped += 1

    print(f"Already complete: {skipped}; remaining: {len(pending)}")
    if pending:
        # Never leave an older archive looking current while output is incomplete.
        archive_path_for(args.output).unlink(missing_ok=True)
    predict_options: dict[str, Any] = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "retina_masks": True,
        "verbose": False,
    }
    if args.device is not None:
        predict_options["device"] = args.device

    newly_processed = 0
    for item_batch in batched(pending, args.batch):
        results = model.predict(
            source=[str(item.source) for item in item_batch],
            batch=len(item_batch),
            **predict_options,
        )
        if len(results) != len(item_batch):
            raise RuntimeError(
                f"Model returned {len(results)} results for {len(item_batch)} images."
            )

        for item, result in zip(item_batch, results, strict=True):
            polygons, class_ids = result_annotations(result)
            label_contents, written_class_ids = format_yolo_label(polygons, class_ids)
            image_destination, label_destination = destination_paths(args.output, item)
            atomic_copy(item.source, image_destination)
            atomic_write_text(label_destination, label_contents)
            newly_processed += 1
            print(
                f"[{skipped + newly_processed:>{len(str(len(items)))}}/{len(items)}] "
                f"{item.source.name}: {len(written_class_ids)} instance(s)"
            )

    class_totals, negatives = verify_complete_dataset(args.output, items)
    archive = create_archive(args.output)

    print("\nFinished.")
    print(f"Images:                 {len(items)}")
    print(f"Newly processed:        {newly_processed}")
    print(f"Resumed/skipped:        {skipped}")
    print(f"Negative images:        {negatives}")
    for class_id, class_name in EXPECTED_CLASSES.items():
        print(f"Class {class_id} ({class_name}): {class_totals[class_id]}")
    print(f"Dataset folder:         {args.output}")
    print(f"Upload ZIP:             {archive}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nCancelled. Completed items were preserved for resume.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
