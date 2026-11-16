"""Explicit, sequential real-video evaluation, separate from fast unit tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .dataset import DEFAULT_ROOT, VideoCase, read_annotations

if __package__ == "tools":
    from milk_vision.config import PLACEMENT_MODEL, PROJECT_ROOT, SCRIPTS_DIR
    from milk_vision.crossings import CROSSING_CONFIDENCE, TRACKING_CONFIDENCE
    from milk_vision.motion import MotionConfig
else:
    from ..milk_vision.config import PLACEMENT_MODEL, PROJECT_ROOT, SCRIPTS_DIR
    from ..milk_vision.crossings import CROSSING_CONFIDENCE, TRACKING_CONFIDENCE
    from ..milk_vision.motion import MotionConfig


@dataclass(frozen=True)
class Prediction:
    net: int
    events: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationRow:
    video: str
    food: str
    scenario: str
    method: str
    expected_net: int
    expected_events: tuple[str, ...] | None
    predicted_net: int | None
    predicted_events: tuple[str, ...] | None
    seconds: float
    error: str | None


def select_cases(
    cases: list[VideoCase],
    *,
    include_negatives: bool = False,
    scenario: str | None = None,
    limit: int | None = None,
) -> list[VideoCase]:
    selected = [
        case
        for case in cases
        if (case.food == "milk" or (include_negatives and case.food == "others"))
        and (scenario is None or case.scenario == scenario)
    ]
    return selected if limit is None else selected[:limit]


def iter_results(
    cases: list[VideoCase],
    methods: Sequence[str],
    predict: Callable[[str, VideoCase], Prediction],
) -> Iterator[EvaluationRow]:
    for case in cases:
        for method in methods:
            start = time.perf_counter()
            prediction = None
            error = None
            try:
                prediction = predict(method, case)
                if any(event not in ("in", "out") for event in prediction.events):
                    raise ValueError("Unknown predicted event direction")
                if prediction.net != sum(
                    1 if event == "in" else -1 for event in prediction.events
                ):
                    raise ValueError("Predicted net disagrees with predicted events")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                prediction = None
            yield EvaluationRow(
                case.relative_path,
                case.food,
                case.scenario,
                method,
                case.expected_net,
                case.expected_events,
                prediction.net if prediction is not None else None,
                prediction.events if prediction is not None else None,
                time.perf_counter() - start,
                error,
            )


def _ordered_matches(expected: tuple[str, ...], actual: tuple[str, ...]) -> int:
    """Longest common subsequence: count matches while respecting event order."""
    previous = [0] * (len(actual) + 1)
    for event in expected:
        current = [0]
        for index, predicted in enumerate(actual):
            current.append(
                previous[index] + 1
                if event == predicted
                else max(previous[index + 1], current[-1])
            )
        previous = current
    return previous[-1]


def metrics(rows: list[EvaluationRow]) -> dict:
    successful = [row for row in rows if row.error is None]
    correct = sum(row.predicted_net == row.expected_net for row in successful)
    annotated = [row for row in rows if row.expected_events is not None]
    matches = expected_count = predicted_count = exact = 0
    for row in annotated:
        expected = row.expected_events
        assert expected is not None
        actual = row.predicted_events or ()
        matches += _ordered_matches(expected, actual)
        expected_count += len(expected)
        predicted_count += len(actual)
        exact += row.error is None and expected == actual
    return {
        "videos": len(rows),
        "successful": len(successful),
        "processing_failures": len(rows) - len(successful),
        "correct_net": correct,
        "net_accuracy": correct / len(rows) if rows else None,
        "mean_absolute_error_successful": sum(
            abs(row.predicted_net - row.expected_net)
            for row in successful
            if row.predicted_net is not None
        )
        / len(successful)
        if successful
        else None,
        "event_annotated_videos": len(annotated),
        "event_sequence_accuracy": exact / len(annotated) if annotated else None,
        "event_precision": matches / predicted_count if predicted_count else None,
        "event_recall": matches / expected_count if expected_count else None,
        "event_f1": 2 * matches / (expected_count + predicted_count)
        if expected_count + predicted_count
        else None,
    }


def summarize(rows: list[EvaluationRow]) -> dict:
    result = {}
    for method in sorted({row.method for row in rows}):
        selected = [row for row in rows if row.method == method]
        scenarios = sorted({f"{row.food}/{row.scenario}" for row in selected})
        result[method] = {
            "overall": metrics(selected),
            "by_scenario": {
                scenario: metrics(
                    [
                        row
                        for row in selected
                        if f"{row.food}/{row.scenario}" == scenario
                    ]
                )
                for scenario in scenarios
            },
            "by_expected_net": {
                str(net): metrics([row for row in selected if row.expected_net == net])
                for net in sorted({row.expected_net for row in selected})
            },
        }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _predictor(
    args: argparse.Namespace, config: MotionConfig
) -> Callable[[str, VideoCase], Prediction]:
    # Import inference only for an actual model run, never annotation checks/help.
    if __package__ == "tools":
        from milk_vision.pipeline import verify_motion_video, verify_placement_video
    else:
        from ..milk_vision.pipeline import verify_motion_video, verify_placement_video

    def predict(method: str, case: VideoCase) -> Prediction:
        if method == "placement":
            result = verify_placement_video(
                case.path,
                args.model,
                confidence=args.conf,
                gate_y=args.gate_y,
                tracking_confidence=args.track_conf,
                device=args.device,
            )
            directions = [event.direction for event in result.events]
            net = result.net_delta
        else:
            motion = verify_motion_video(
                case.path,
                args.model,
                config,
                tracking_confidence=args.track_conf,
                device=args.device,
            )
            directions = [event["direction"] for event in motion.events]
            net = motion.net_delta
        mapping = {"inward": "in", "outward": "out"}
        return Prediction(net, tuple(mapping[direction] for direction in directions))

    return predict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--method", choices=("both", "placement", "motion"), default="both"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PLACEMENT_MODEL,
        help="Same checkpoint for both algorithms",
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="Also evaluate others/net 0 as milk-negative clips",
    )
    parser.add_argument("--scenario")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output", type=Path, help="JSON report; existing files are not overwritten"
    )
    parser.add_argument(
        "--list", action="store_true", help="List selected clips without loading models"
    )
    parser.add_argument("--conf", type=float, default=CROSSING_CONFIDENCE)
    parser.add_argument("--track-conf", type=float, default=TRACKING_CONFIDENCE)
    parser.add_argument("--gate-y", type=float, default=0.27)
    defaults = MotionConfig()
    parser.add_argument("--inward", choices=("up", "down"), default=defaults.inward)
    for field in ("window_seconds", "max_gap_seconds", "min_speed", "min_displacement"):
        parser.add_argument(
            "--" + field.replace("_", "-"), type=float, default=getattr(defaults, field)
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.root = args.root.expanduser().resolve()
        manifest = (
            (args.manifest or args.root / "annotations.json").expanduser().resolve()
        )
        cases, _ = read_annotations(args.root, manifest)
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        cases = select_cases(
            cases,
            include_negatives=args.include_negatives,
            scenario=args.scenario,
            limit=args.limit,
        )
        if not cases:
            raise ValueError(
                "No matching milk/negative clips; check the scenario labels"
            )
        if args.list:
            print("\n".join(case.relative_path for case in cases))
            return 0
        if not (math.isfinite(args.conf) and 0 < args.track_conf <= args.conf <= 1):
            raise ValueError("Require 0 < --track-conf <= --conf <= 1")
        if not 0.05 <= args.gate_y <= 0.95:
            raise ValueError("--gate-y must be between 0.05 and 0.95")
        config = MotionConfig(
            window_seconds=args.window_seconds,
            max_gap_seconds=args.max_gap_seconds,
            min_speed=args.min_speed,
            min_displacement=args.min_displacement,
            inward=args.inward,
        )
        config.validate()
        args.model = args.model.expanduser().resolve()
        if not args.model.is_file():
            raise ValueError(f"Model does not exist: {args.model}")
        output = (
            (
                args.output
                or PROJECT_ROOT / "debug" / "evaluation" / f"{time.time_ns()}.json"
            )
            .expanduser()
            .resolve()
        )
        if output.exists() or output.suffix.lower() != ".json":
            raise ValueError("--output must name a new .json file")
        # Never let an output path overwrite source data or annotations.
        if output.is_relative_to(args.root):
            raise ValueError("Write evaluation reports outside the video dataset")
        output.parent.mkdir(parents=True, exist_ok=True)
        methods = ("placement", "motion") if args.method == "both" else (args.method,)
        run = {
            "schema_version": 1,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(args.root),
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "model": str(args.model),
            "model_sha256": _sha256(args.model),
            "implementation_sha256": {
                name: _sha256(SCRIPTS_DIR / "milk_vision" / name)
                for name in ("crossings.py", "motion.py", "inference.py", "pipeline.py")
            },
            "python": platform.python_version(),
            "methods": methods,
            "selected_videos": len(cases),
            "selection": {
                key: getattr(args, key)
                for key in ("include_negatives", "scenario", "limit")
            },
            "settings": {
                "device": args.device,
                "conf": args.conf,
                "track_conf": args.track_conf,
                "gate_y": args.gate_y,
                "motion": asdict(config),
            },
        }
        rows = []
        cancelled = False
        try:
            for row in iter_results(cases, methods, _predictor(args, config)):
                rows.append(row)
                print(
                    f"[{len(rows)}/{len(cases) * len(methods)}] {row.method} {row.video}: expected {row.expected_net:+d}, predicted {row.predicted_net}, error={row.error}",
                    flush=True,
                )
        except KeyboardInterrupt:
            cancelled = True
        run.update(
            cancelled=cancelled,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            summary=summarize(rows),
            results=[asdict(row) for row in rows],
        )
        with output.open("x", encoding="utf-8") as handle:
            json.dump(run, handle, indent=2)
            handle.write("\n")
        print(json.dumps(run["summary"], indent=2))
        print(f"Report: {output}", flush=True)
        if cancelled:
            return 130
        return (
            1
            if any(
                row.error
                or row.predicted_net != row.expected_net
                or (
                    row.expected_events is not None
                    and row.expected_events != row.predicted_events
                )
                for row in rows
            )
            else 0
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
