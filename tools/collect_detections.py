"""Collect reusable observations for the annotated milk/negative dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

if __package__ == "tools":
    from milk_vision.config import PLACEMENT_MODEL, PROJECT_ROOT
    from milk_vision.evaluation.dataset import DEFAULT_ROOT, read_annotations
    from milk_vision.evaluation.runner import select_cases
    from milk_vision.observations.storage import (
        DEFAULT_CACHE,
        atomic_json,
        get_observations,
    )
else:
    from ..milk_vision.config import PLACEMENT_MODEL, PROJECT_ROOT
    from ..milk_vision.evaluation.dataset import DEFAULT_ROOT, read_annotations
    from ..milk_vision.evaluation.runner import select_cases
    from ..milk_vision.observations.storage import (
        DEFAULT_CACHE,
        atomic_json,
        get_observations,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", type=Path, default=PLACEMENT_MODEL)
    parser.add_argument("--device")
    parser.add_argument("--track-conf", type=float, default=0.10)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh-observations", action="store_true")
    parser.add_argument("--include-negatives", action="store_true")
    parser.add_argument("--scenario")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        cases, _ = read_annotations(args.root, args.manifest)
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        cases = select_cases(
            cases,
            include_negatives=args.include_negatives,
            scenario=args.scenario,
            limit=args.limit,
        )
        if not cases:
            raise ValueError("No selected cases")
        if args.list:
            print("\n".join(c.relative_path for c in cases))
            return 0
        if not 0 < args.track_conf <= 1:
            raise ValueError("--track-conf must be in (0, 1]")
        output = (
            args.output
            or PROJECT_ROOT / "debug" / "collection" / f"{time.time_ns()}.json"
        )
        if output.exists() or output.resolve().is_relative_to(args.root.resolve()):
            raise ValueError("Output must be a new file outside the dataset")
        report = {"selected": len(cases), "results": [], "cancelled": False}
        try:
            for index, case in enumerate(cases, 1):
                print(f"[{index}/{len(cases)}] {case.relative_path}", flush=True)
                item = {"video": case.relative_path}
                try:
                    _, details = get_observations(
                        case.path,
                        args.model,
                        args.track_conf,
                        args.device,
                        args.cache_dir,
                        args.refresh_observations,
                        worker=True,
                    )
                    item.update(details)
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    OSError,
                    RuntimeError,
                    ImportError,
                ) as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                report["results"].append(item)
                atomic_json(output, report)
        except KeyboardInterrupt:
            report["cancelled"] = True
            atomic_json(output, report)
            return 130
        print(json.dumps({"completed": len(report["results"]), "report": str(output)}))
        return int(any("error" in r for r in report["results"]))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
