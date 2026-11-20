"""Prepare three-field milk trajectories and evaluate them with Jev."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ == "tools":
    from milk_vision.config import MOTION_MODEL
    from milk_vision.inference import collect_detections
    from milk_vision.motion import MotionConfig, analyze_detections
    from milk_vision.types import MotionRow
else:
    from ..milk_vision.config import MOTION_MODEL
    from ..milk_vision.inference import collect_detections
    from ..milk_vision.motion import MotionConfig, analyze_detections
    from ..milk_vision.types import MotionRow

if __package__ == "tools":
    from milk_vision.algorithms import jev as algorithm
else:
    from ..milk_vision.algorithms import jev as algorithm

urllib = algorithm.urllib
CHOICES = algorithm.CHOICES
EPISODES = algorithm.EPISODES
ENDPOINT = algorithm.ENDPOINT
FIELDS = algorithm.FIELDS
encode = algorithm.encode
digest = algorithm.digest
write_json = algorithm.write_json
make_request = algorithm.make_request
validate_response = algorithm.validate_response


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    config = MotionConfig(inward=args.inward)
    if args.csv:
        with args.csv.open(encoding="utf-8-sig", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
        source = args.csv
        provenance = {
            "kind": "existing_csv",
            "detector_settings": "unknown; inspect original run",
        }
    else:
        rows, metadata = collect_detections(
            args.video.resolve(),
            args.detector.resolve(),
            args.track_conf,
            args.device,
            status=lambda message: print(message, file=sys.stderr),
        )
        source = args.video
        provenance = {
            "kind": "fresh_video",
            "metadata": metadata,
            "track_conf": args.track_conf,
        }
    request = make_request(rows, args.model, args.inward, args.framework)
    # Recompute the baseline from the same raw values; never send it to Jev.
    frame_indices = {
        timestamp: index
        for index, timestamp in enumerate(
            sorted({values[0] for values in request["state"]["observations"]})
        )
    }
    baseline_rows: list[MotionRow] = [
        {
            "frame_index": frame_indices[values[0]],
            "timestamp_seconds": values[0],
            "center_y_normalized": values[1],
            "confidence": values[2],
            "track_id": None,
        }
        for values in request["state"]["observations"]
    ]
    events = analyze_detections(baseline_rows, config)
    baseline = sum(1 if event["direction"] == "inward" else -1 for event in events)
    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    case = {
        "schema_version": 1,
        "purpose": "development",
        "source": str(source.resolve()),
        "source_sha256": source_hash,
        "provenance": provenance,
        "expected_net": args.expected_net,
        "baseline_net": baseline,
        "baseline_config": vars(config),
        "request": request,
        "request_sha256": digest(request),
    }
    write_json(args.output, case)
    print(
        json.dumps(
            {
                "prepared": str(args.output),
                "observations": len(rows),
                "request_bytes": len(encode(request)),
                "baseline_net": baseline,
            }
        )
    )


def reframe(args: argparse.Namespace) -> None:
    """Reuse saved detections without rerunning YOLO or altering the first run."""
    case = json.loads(args.case.read_text(encoding="utf-8"))
    if digest(case["request"]) != case["request_sha256"]:
        raise ValueError("Original request hash mismatch")
    observations = case["request"]["state"]["observations"]
    rows: list[dict[str, Any]] = [
        dict(zip(FIELDS, values, strict=True)) for values in observations
    ]
    case["parent_request_sha256"] = case["request_sha256"]
    case["request"] = make_request(
        rows, case["request"]["model"], case["baseline_config"]["inward"], "episodes"
    )
    case["request_sha256"] = digest(case["request"])
    case["framework"] = "episodes"
    write_json(args.output, case)
    print(json.dumps({"prepared": str(args.output), "observations": len(rows)}))


def run(args: argparse.Namespace) -> None:
    algorithm.api_key()
    for path in args.cases:
        case = json.loads(path.read_text(encoding="utf-8"))
        request = case["request"]
        if digest(request) != case["request_sha256"]:
            raise ValueError(f"Request hash mismatch: {path}; prepare a new case")
        output = args.output_dir / f"{path.stem}-{digest(request)[:12]}.result.json"
        if output.exists():
            raise ValueError(
                f"Result already exists: {output}; use another output directory"
            )
        started = time.monotonic()
        # Only the explicit request is transmitted, never local labels.
        response = algorithm.call(request, args.timeout)
        questions = set(request["questions"])
        if questions not in ({"net_change"}, {"episodes"}):
            raise ValueError("Expected one net_change or episodes question")
        question = next(iter(questions))
        prediction, answer = validate_response(response, question)
        result = copy.deepcopy(case)
        if question == "episodes":
            sequence = EPISODES[answer["choice"]]
            result.update(
                {
                    "predicted_episodes": list(sequence),
                    "inward_count": sequence.count("inward"),
                    "outward_count": sequence.count("outward"),
                }
            )
        result.update(
            {
                "predicted_net": prediction,
                "jev_confidence": answer["confidence"],
                "probabilities": answer["probabilities"],
                "response": response,
                "completed_at": datetime.now(UTC).isoformat(),
                "latency_seconds": time.monotonic() - started,
            }
        )
        write_json(output, result)
        print(
            json.dumps(
                {
                    "result": str(output),
                    "predicted_net": prediction,
                    "jev_confidence": answer["confidence"],
                }
            )
        )


def report(args: argparse.Namespace) -> None:
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    identities = [row["source_sha256"] for row in results]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate sources: report one prediction per video")
    labelled = [row for row in results if row["expected_net"] is not None]
    summary: dict[str, Any] = {"videos": len(results), "labelled": len(labelled)}
    for field in ("predicted_net", "baseline_net"):
        summary[field] = {
            str(label): {
                "total": sum(row["expected_net"] == label for row in labelled),
                "correct": sum(
                    row["expected_net"] == label and row[field] == label
                    for row in labelled
                ),
                "predictions": {
                    str(pred): sum(
                        row["expected_net"] == label and row[field] == pred
                        for row in labelled
                    )
                    for pred in (-1, 0, 1)
                },
                "outside_three_classes": sum(
                    row["expected_net"] == label and row[field] not in (-1, 0, 1)
                    for row in labelled
                ),
            }
            for label in (-1, 0, 1)
        }
    summary["jev_models"] = sorted({row["response"]["model"] for row in results})
    summary["usage"] = {
        name: sum(row["response"]["usage"][name] for row in results)
        for name in ("input_tokens", "output_tokens")
    }
    episodes = [
        row
        for row in results
        if "predicted_episodes" in row and row.get("expected_episodes") is not None
    ]
    if episodes:
        summary["episode_sequences"] = {
            "labelled": len(episodes),
            "correct": sum(
                row["predicted_episodes"] == row["expected_episodes"]
                for row in episodes
            ),
        }
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="Save a local case; no API call"
    )
    inputs = prepare_parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--csv", type=Path)
    inputs.add_argument("--video", type=Path)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--expected-net", type=int, choices=(-1, 0, 1))
    prepare_parser.add_argument("--model", default="jev-1.14.0")
    prepare_parser.add_argument(
        "--framework", choices=("net", "episodes"), default="net"
    )
    prepare_parser.add_argument("--inward", choices=("up", "down"), default="down")
    prepare_parser.add_argument("--detector", type=Path, default=MOTION_MODEL)
    prepare_parser.add_argument("--track-conf", type=float, default=0.10)
    prepare_parser.add_argument("--device")
    prepare_parser.set_defaults(handler=prepare)
    reframe_parser = commands.add_parser(
        "reframe", help="Reuse detections with five episode choices"
    )
    reframe_parser.add_argument("--case", type=Path, required=True)
    reframe_parser.add_argument("--output", type=Path, required=True)
    reframe_parser.set_defaults(handler=reframe)
    run_parser = commands.add_parser("run", help="Send prepared requests to TypeSafe")
    run_parser.add_argument("cases", nargs="+", type=Path)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--timeout", type=float, default=45)
    run_parser.set_defaults(handler=run)
    report_parser = commands.add_parser(
        "report", help="Compare saved Jev and motion predictions"
    )
    report_parser.add_argument("results", nargs="+", type=Path)
    report_parser.set_defaults(handler=report)
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "track_conf") and (
            not math.isfinite(args.track_conf) or not 0 < args.track_conf <= 1
        ):
            raise ValueError("--track-conf must be finite and in (0, 1]")
        if hasattr(args, "timeout") and (
            not math.isfinite(args.timeout) or args.timeout <= 0
        ):
            raise ValueError("--timeout must be finite and positive")
        args.handler(args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
