"""Versioned, content-checked observation storage with atomic completed writes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, SCRIPTS_DIR, TRACKER_CONFIG
from ..types import (
    BoundingBox,
    FrameObservations,
    RawDetection,
    TrackedDetection,
    VideoObservations,
)
from .collector import COLLECTOR_VERSION, collect_video

SCHEMA_VERSION = 1
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "observations"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(video: Path, model: Path, confidence: float, device: str | None) -> dict:
    # Ultralytics applies this default on import. Apply it before fingerprinting
    # so the parent and fresh worker inherit the same effective setting.
    if not os.environ.get("OMP_NUM_THREADS"):
        os.environ["OMP_NUM_THREADS"] = "1"
    return {
        "video_sha256": file_hash(video),
        "model_sha256": file_hash(model),
        "tracker_sha256": file_hash(TRACKER_CONFIG),
        "confidence": confidence,
        "device": device,
        "schema": SCHEMA_VERSION,
        "collector": COLLECTOR_VERSION,
        "collector_sha256": file_hash(Path(__file__).with_name("collector.py")),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("ultralytics", "torch", "opencv-python", "numpy", "lap")
        },
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save(path: Path, observations: VideoObservations, fingerprint: dict) -> None:
    data = asdict(observations)
    atomic_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "identity": fingerprint,
            "data_sha256": digest(data),
            "data": data,
        },
    )


def load(path: Path, fingerprint: dict | None = None) -> VideoObservations:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if (
        envelope.get("schema_version") != SCHEMA_VERSION
        or envelope.get("complete") is not True
    ):
        raise ValueError(f"Incomplete or unsupported observation artifact: {path}")
    if fingerprint is not None and envelope["identity"] != fingerprint:
        raise ValueError(f"Stale observation artifact: {path}")
    data = envelope["data"]
    if digest(data) != envelope["data_sha256"]:
        raise ValueError(f"Observation content hash mismatch: {path}")
    metadata = data["metadata"]
    fps = metadata["fps"]
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid observation FPS")
    frames = []
    for index, f in enumerate(data["frames"]):
        if (
            f["frame_index"] != index
            or f["width"] <= 0
            or f["height"] <= 0
            or not math.isclose(f["timestamp_seconds"], index / fps)
        ):
            raise ValueError("Invalid observation frame sequence")
        raw = tuple(
            RawDetection(
                BoundingBox(**d["box"]),
                d["confidence"],
                d["detection_index"],
                d["track_id"],
            )
            for d in f["raw"]
        )
        tracked = tuple(
            TrackedDetection(
                d["class_name"],
                d["track_id"],
                d["frame_index"],
                BoundingBox(**d["box"]),
                d["confidence"],
            )
            for d in f["tracked"]
        )
        for detection in (*raw, *tracked):
            if not 0 <= detection.confidence <= 1 or not all(
                math.isfinite(v) for v in asdict(detection.box).values()
            ):
                raise ValueError("Invalid detection values")
        if [d.detection_index for d in raw] != list(range(len(raw))) or any(
            d.frame_index != index for d in tracked
        ):
            raise ValueError("Invalid detection indices")
        frames.append(
            FrameObservations(
                index, f["timestamp_seconds"], f["width"], f["height"], raw, tracked
            )
        )
    if not frames or len(frames) != metadata["frame_count"]:
        raise ValueError("Incomplete observation frame count")
    return VideoObservations(metadata, tuple(frames))


def get_observations(
    video: Path,
    model: Path,
    confidence: float = 0.10,
    device: str | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
    worker: bool = False,
) -> tuple[VideoObservations, dict]:
    video, model = video.expanduser().resolve(), model.expanduser().resolve()
    started = time.perf_counter()
    if cache_dir is None:
        return collect_video(video, model, confidence, device), {
            "cache_hit": False,
            "extraction_seconds": time.perf_counter() - started,
        }
    fingerprint = identity(video, model, confidence, device)
    directory = cache_dir / digest(fingerprint)
    artifacts = sorted(directory.glob("*.json"))
    if artifacts and not refresh:
        path = artifacts[-1]
        observations = load(path, fingerprint)
        hit = True
    else:
        path = directory / f"{time.time_ns()}.json"
        if worker:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(SCRIPTS_DIR), environment.get("PYTHONPATH")))
            )
            command = [
                sys.executable,
                "-m",
                "milk_vision.observations.storage",
                str(video),
                str(model),
                str(path),
                str(confidence),
            ]
            if device is not None:
                command.append(device)
            process = subprocess.Popen(command, env=environment)
            try:
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
            if code:
                raise RuntimeError(
                    f"Observation worker failed with exit {code}: {video}"
                )
            observations = load(path, fingerprint)
        else:
            observations = collect_video(video, model, confidence, device)
            save(path, observations, fingerprint)
        hit = False
    return observations, {
        "cache_hit": hit,
        "artifact": str(path.resolve()),
        "observation_sha256": file_hash(path),
        "extraction_seconds": time.perf_counter() - started,
        "detection_count": sum(len(f.raw) for f in observations.frames),
    }


if __name__ == "__main__":
    video, model, output = map(Path, sys.argv[1:4])
    confidence = float(sys.argv[4])
    device = sys.argv[5] if len(sys.argv) > 5 else None
    fingerprint = identity(video, model, confidence, device)
    save(
        output,
        collect_video(
            video,
            model,
            confidence,
            device,
            status=lambda message: print(message, file=sys.stderr),
        ),
        fingerprint,
    )
