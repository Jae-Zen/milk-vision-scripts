"""Load and validate the labelled-video dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..batch import VIDEO_EXTENSIONS
from ..config import PROJECT_ROOT

DEFAULT_ROOT = PROJECT_ROOT / "Dataset" / "videos"
NET_FOLDERS = {"net -1": -1, "net 0": 0, "net 1": 1}


@dataclass(frozen=True)
class VideoCase:
    path: Path
    relative_path: str
    food: str
    expected_net: int
    scenario: str
    description: str
    expected_events: tuple[str, ...] | None


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate annotation key: {key}")
        result[key] = value
    return result


def _video_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    path = (root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.is_relative_to(root)
    ):
        raise ValueError(f"Video path must stay inside the dataset: {name}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video path: {name}")
    return path


def read_annotations(
    root: Path, manifest: Path | None = None
) -> tuple[list[VideoCase], list[str]]:
    """Validate annotations, layout, event sums, files, and complete coverage."""
    root = root.expanduser().resolve()
    manifest = manifest or root / "annotations.json"
    document = json.loads(
        manifest.read_text(encoding="utf-8"), object_pairs_hook=_pairs
    )
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Expected an annotations document with schema_version 1")
    records = document.get("videos")
    excluded = document.get("excluded", [])
    if not isinstance(records, dict) or not isinstance(excluded, list):
        raise TypeError("videos must be an object and excluded must be a list")
    cases = []
    seen = set()
    for name, record in records.items():
        path = _video_path(root, name)
        parts = PurePosixPath(name).parts
        if len(parts) != 3 or parts[1] not in NET_FOLDERS:
            raise ValueError(f"Expected food/net N/video layout: {name}")
        if not path.is_file() or name.casefold() in seen:
            raise ValueError(f"Missing or case-colliding video: {name}")
        seen.add(name.casefold())
        if not isinstance(record, dict):
            raise TypeError(f"Invalid annotation for {name}")
        for field in ("scenario", "description"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Missing {field} for {name}")
        events = record.get("expected_events")
        if events is not None:
            if not isinstance(events, list) or any(
                e not in ("in", "out") for e in events
            ):
                raise ValueError(
                    f"expected_events must be null or a list of in/out: {name}"
                )
            if (
                sum(1 if event == "in" else -1 for event in events)
                != NET_FOLDERS[parts[1]]
            ):
                raise ValueError(f"Event sequence disagrees with net folder: {name}")
        cases.append(
            VideoCase(
                path,
                name,
                parts[0],
                NET_FOLDERS[parts[1]],
                record["scenario"],
                record["description"],
                None if events is None else tuple(events),
            )
        )
    for name in excluded:
        if not isinstance(name, str):
            raise TypeError("Excluded paths must be strings")
        path = _video_path(root, name)
        if not path.is_file() or name.casefold() in seen:
            raise ValueError(f"Missing or duplicate excluded video: {name}")
        seen.add(name.casefold())
    actual = {
        p.relative_to(root).as_posix().casefold()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    }
    if actual != seen:
        raise ValueError(f"Annotation coverage mismatch: {sorted(actual ^ seen)}")
    return sorted(cases, key=lambda case: case.relative_path.casefold()), excluded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate annotations and video coverage (the default action)",
    )
    args = parser.parse_args(argv)
    try:
        cases, excluded = read_annotations(args.root)
        summary = {"videos": len(cases), "excluded": len(excluded)}
        print(json.dumps(summary, indent=2))
        return 0
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
