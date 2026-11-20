"""Explicit, sequential real-video evaluation, separate from fast unit tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..algorithms.motion import MotionConfig
from ..algorithms.placement import CROSSING_CONFIDENCE, TRACKING_CONFIDENCE
from ..config import PLACEMENT_MODEL, PROJECT_ROOT
from . import reporting
from .dataset import DEFAULT_ROOT, VideoCase, read_annotations
from .reporting import implementation_hashes, summarize

metrics = reporting.metrics
from .types import EvaluationRow, Prediction


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
            except (
                ValueError,
                TypeError,
                KeyError,
                OSError,
                RuntimeError,
                ImportError,
            ) as exc:
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
                prediction.details if prediction is not None else {},
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _predictor(
    args: argparse.Namespace, config: MotionConfig
) -> Callable[[str, VideoCase], Prediction]:
    from ..algorithms import jev, motion, placement
    from ..observations.storage import atomic_json, digest, get_observations
    from .reporting import implementation_hashes

    current_path = None
    observations = None
    collection = {}
    implementations = implementation_hashes()

    def predict(method: str, case: VideoCase) -> Prediction:
        nonlocal current_path, observations, collection
        if current_path != case.path:
            observations, collection = get_observations(
                case.path,
                args.model,
                args.track_conf,
                args.device,
                args.cache_dir,
                args.refresh_observations,
                worker=True,
            )
            current_path = case.path
        assert observations is not None
        settings = {
            "method": method,
            "observation_sha256": collection["observation_sha256"],
            "implementation": implementations,
            "motion": asdict(config),
            "conf": args.conf,
            "gate_y": args.gate_y,
            "jev_model": args.jev_model,
        }
        # Include the exact request in Jev's identity; labels never enter the request.
        if method == "jev":
            settings["request"] = jev.make_request(
                observations.motion_rows(), args.jev_model, args.inward, "episodes"
            )
        directory = args.cache_dir.parent / "predictions" / method / digest(settings)
        cached = sorted(directory.glob("*.json"))
        if cached and not args.rerun:
            record = json.loads(cached[-1].read_text(encoding="utf-8"))
            if record["identity"] != settings or record["prediction_sha256"] != digest(
                record["prediction"]
            ):
                raise ValueError("Prediction cache identity/content mismatch")
            value = record["prediction"]
            prediction = Prediction(
                value["net"], tuple(value["events"]), value["details"]
            )
            return Prediction(
                prediction.net,
                prediction.events,
                {
                    **prediction.details,
                    "collection": collection,
                    "prediction_cache_hit": True,
                },
            )
        started = time.perf_counter()
        if method == "placement":
            prediction = placement.analyze(observations, args.conf, args.gate_y)
        elif method == "motion":
            prediction = motion.analyze(observations, config)
        else:
            prediction = jev.predict(
                observations, args.jev_model, args.inward, args.api_timeout
            )
        prediction = Prediction(
            prediction.net,
            prediction.events,
            {
                **prediction.details,
                "collection": collection,
                "algorithm_seconds": time.perf_counter() - started,
                "prediction_cache_hit": False,
            },
        )
        payload = asdict(prediction)
        atomic_json(
            directory / f"{time.time_ns()}.json",
            {
                "identity": settings,
                "prediction_sha256": digest(payload),
                "prediction": payload,
            },
        )
        return prediction

    return predict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--method", choices=("both", "placement", "motion", "jev"))
    selection.add_argument(
        "--methods", nargs="+", choices=("placement", "motion", "jev")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "cache" / "observations"
    )
    parser.add_argument("--refresh-observations", action="store_true")
    parser.add_argument(
        "--rerun", action="store_true", help="Create fresh prediction versions"
    )
    parser.add_argument("--jev-model", default="jev-1.14.0")
    parser.add_argument("--api-timeout", type=float, default=45)

    parser.add_argument(
        "--model",
        type=Path,
        default=PLACEMENT_MODEL,
        help="Same checkpoint for all selected algorithms",
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
    args.method = args.method or "both"
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
        methods = (
            tuple(dict.fromkeys(args.methods))
            if args.methods
            else (("placement", "motion") if args.method == "both" else (args.method,))
        )
        args.cache_dir = args.cache_dir.expanduser().resolve()
        if not math.isfinite(args.api_timeout) or args.api_timeout <= 0:
            raise ValueError("--api-timeout must be finite and positive")
        run = {
            "schema_version": 1,
            "started_utc": datetime.now(UTC).isoformat(),
            "root": str(args.root),
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "model": str(args.model),
            "model_sha256": _sha256(args.model),
            "implementation_sha256": implementation_hashes(),
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
                "jev_model": args.jev_model,
                "cache_dir": str(args.cache_dir),
            },
        }
        rows = []
        cancelled = False
        try:
            for row in iter_results(cases, methods, _predictor(args, config)):
                rows.append(row)
                # Save after every result, including errors, for crash/cancellation recovery.
                from ..observations.storage import atomic_json

                run.update(
                    cancelled=False,
                    completed_utc=datetime.now(UTC).isoformat(),
                    summary=summarize(rows),
                    results=[asdict(item) for item in rows],
                )
                atomic_json(output, run)
                print(
                    f"[{len(rows)}/{len(cases) * len(methods)}] {row.method} {row.video}: expected {row.expected_net:+d}, predicted {row.predicted_net}, error={row.error}",
                    flush=True,
                )
        except KeyboardInterrupt:
            cancelled = True
        run.update(
            cancelled=cancelled,
            completed_utc=datetime.now(UTC).isoformat(),
            summary=summarize(rows),
            results=[asdict(row) for row in rows],
        )
        from ..observations.storage import atomic_json

        atomic_json(output, run)
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
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
