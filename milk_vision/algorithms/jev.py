"""Jev algorithm: input, prompt, API call, five episode choices, and net calculation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import SCRIPTS_DIR
from ..types import Prediction, VideoObservations

CHOICES = {"added": 1, "unchanged": 0, "removed": -1}
EPISODES = {
    "none": (),
    "inward": ("inward",),
    "outward": ("outward",),
    "inward → outward": ("inward", "outward"),
    "outward → inward": ("outward", "inward"),
}
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
FIELDS = ("timestamp_seconds", "center_y_normalized", "confidence")


def encode(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an earlier experiment.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def make_request(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    inward: str = "down",
    framework: str = "net",
) -> dict:
    observations = []
    for index, row in enumerate(rows):
        values = [float(row[field]) for field in FIELDS]
        timestamp, y, confidence = values
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite observation at row {index + 1}")
        if timestamp < 0 or not 0 <= y <= 1 or not 0 <= confidence <= 1:
            raise ValueError(f"Out-of-range observation at row {index + 1}")
        observations.append(values)
    observations.sort(key=lambda values: values[0])
    direction = "Increasing" if inward == "down" else "Decreasing"
    if framework == "episodes":
        outward = "decreasing" if inward == "down" else "increasing"
        return {
            "model": model,
            "state": {
                "context": (
                    "Each row is [timestamp_seconds, normalized_y, detection_confidence]. "
                    "Y is the milk bounding-box midpoint divided by image height: "
                    "0 is top and 1 is bottom. Confidence is YOLO's milk detection confidence. "
                    "Rows are in time order; equal timestamps mean separate detections."
                ),
                "observations": observations,
            },
            "questions": {
                "episodes": {
                    "type": "choice",
                    "instructions": (
                        "Identify the sequence of directional motion episodes. "
                        f"{direction} y is inward; {outward} y is outward. "
                        "Group consecutive movements in the same direction into one episode. "
                        "A pause followed by the same direction remains one episode. "
                        "A clear reversal starts the opposite episode. Ignore small detection jitter. "
                        "A detection gap alone does not start a new episode. "
                        "Choose none if there is no detected directional motion."
                    ),
                    "criteria": {
                        "none": "No directional motion episodes.",
                        "inward": "One inward episode only.",
                        "outward": "One outward episode only.",
                        "inward → outward": "One inward episode followed by one outward episode.",
                        "outward → inward": "One outward episode followed by one inward episode.",
                    },
                }
            },
        }
    return {
        "model": model,
        "state": {
            "context": (
                "Milk detections from one complete fixed-camera video. Each row is "
                "[timestamp_seconds, y_normalized, detection_confidence]. Time is "
                "seconds from video start. Y is the bounding-box vertical midpoint "
                "divided by image height: 0 is top, 1 is bottom. "
                f"{direction} y corresponds to the camera's inward direction. "
                "Rows with equal timestamps are separate detections and may be "
                "different cartons; object identities are unavailable. Gaps indicate "
                "missing detections, not proven removal. Confidence measures milk "
                "detection confidence, not direction or inventory-change confidence. "
                "There are no interpolated, smoothed, or subsampled observations. "
                "The video's true net inventory change is one of -1, 0, +1."
            ),
            "observations": observations,
        },
        "questions": {
            "net_change": {
                "type": "choice",
                "instructions": (
                    "Infer the final milk inventory minus the initial milk inventory "
                    "from the complete sequence. Consider reversals and detection "
                    "uncertainty. Vertical movement alone does not prove placement. "
                    "Select the most likely net change even when evidence is ambiguous."
                ),
                "criteria": {
                    "added": "Final milk inventory is one greater than initial: net +1.",
                    "unchanged": "Final milk inventory equals initial: net 0, including round trips.",
                    "removed": "Final milk inventory is one less than initial: net -1.",
                },
            }
        },
    }


def validate_response(response: dict, question: str = "net_change") -> tuple[int, dict]:
    if not isinstance(response.get("model"), str) or not response["model"]:
        raise ValueError("Response is missing its model version")
    choices = EPISODES if question == "episodes" else CHOICES
    answer = response["answers"][question]
    if answer.get("type") != "choice" or answer["choice"] not in choices:
        raise ValueError("Unexpected Jev answer type or choice")
    probabilities = answer["probabilities"]
    if set(probabilities) != set(choices):
        raise ValueError("Response must include every choice probability")
    values = [*probabilities.values(), answer["confidence"]]
    if any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or not 0 <= v <= 1
        for v in values
    ):
        raise ValueError(
            "Response probabilities/confidence must be finite and in [0, 1]"
        )
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.001):
        raise ValueError("Response probabilities do not sum to one")
    if probabilities[answer["choice"]] < max(probabilities.values()):
        raise ValueError("Selected choice is not a highest-probability option")
    if question == "episodes":
        sequence = EPISODES[answer["choice"]]
        prediction = sequence.count("inward") - sequence.count("outward")
    else:
        prediction = CHOICES[answer["choice"]]
    return prediction, answer


def api_key() -> str:
    from dotenv import load_dotenv

    load_dotenv(SCRIPTS_DIR / ".env", override=False)
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise ValueError("Set TYPESAFE_API_KEY in the environment or Scripts/.env")
    return key


def call(request: dict, timeout: float = 45) -> dict:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("API timeout must be finite and positive")
    http = urllib.request.Request(
        ENDPOINT,
        data=encode(request),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(http, timeout=timeout) as stream:
            return json.load(stream)
    except urllib.error.HTTPError as exc:
        raise ValueError(
            f"Jev HTTP {exc.code}; inspect account, rate and context limits. No automatic retries."
        ) from None


def predict(
    observations: VideoObservations,
    model: str = "jev-1.14.0",
    inward: str = "down",
    timeout: float = 45,
) -> Prediction:
    request = make_request(observations.motion_rows(), model, inward, "episodes")
    started = time.perf_counter()
    response = call(request, timeout)
    net, answer = validate_response(response, "episodes")
    episodes = EPISODES[answer["choice"]]
    return Prediction(
        net,
        tuple("in" if e == "inward" else "out" for e in episodes),
        {
            "request": request,
            "response": response,
            "request_sha256": digest(request),
            "jev_confidence": answer["confidence"],
            "probabilities": answer["probabilities"],
            "inward_count": episodes.count("inward"),
            "outward_count": episodes.count("outward"),
            "api_seconds": time.perf_counter() - started,
        },
    )
