"""Dataset comparison metrics; labels describe inventory events."""

import hashlib

from ..config import SCRIPTS_DIR
from .types import EvaluationRow


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
            "empty_detections": metrics(
                [
                    row
                    for row in selected
                    if row.details.get("collection", {}).get("detection_count") == 0
                ]
            ),
            "with_detections": metrics(
                [
                    row
                    for row in selected
                    if row.details.get("collection", {}).get("detection_count", 0) > 0
                ]
            ),
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


def implementation_hashes() -> dict[str, str]:
    root = SCRIPTS_DIR / "milk_vision"
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("algorithms", "observations")
        for path in sorted((root / folder).glob("*.py"))
    } | {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("types.py", "config.py")
    }
