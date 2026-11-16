"""Fast annotation/evaluation tests; no model loading or real-video inference."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import dataset, evaluate


def write_dataset(root: Path) -> None:
    records = {
        "milk/net 0/a.mp4": {
            "scenario": "add_and_remove_milk",
            "description": "Milk enters then leaves.",
            "expected_events": ["in", "out"],
        },
        "milk/net 1/b.mp4": {
            "scenario": "add_milk",
            "description": "Milk enters.",
            "expected_events": ["in"],
        },
        "others/net 0/c.mp4": {
            "scenario": "add_random_object",
            "description": "Another object enters.",
            "expected_events": [],
        },
    }
    excluded = ["cheese/long_for_frames/d.mp4"]
    for name in [*records, *excluded]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake video")
    (root / "annotations.json").write_text(
        json.dumps({"schema_version": 1, "videos": records, "excluded": excluded}),
        encoding="utf-8",
    )


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "videos"
        self.root.mkdir()

    def test_validation_command_defaults_to_checking_the_current_dataset(self):
        write_dataset(self.root)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(0, dataset.main(["--root", str(self.root)]))
        self.assertEqual({"videos": 3, "excluded": 1}, json.loads(output.getvalue()))
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            dataset.main(["--root", str(self.root), "--apply"])
        self.assertEqual(2, raised.exception.code)

    def test_missing_or_unannotated_video_fails_validation(self):
        write_dataset(self.root)
        extra = self.root / "milk/net 0/extra.mp4"
        extra.write_bytes(b"fake video")
        with self.assertRaisesRegex(ValueError, "coverage"):
            dataset.read_annotations(self.root)
        extra.unlink()
        (self.root / "milk/net 0/a.mp4").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            dataset.read_annotations(self.root)

    def test_event_annotations_must_agree_with_folder_net(self):
        write_dataset(self.root)
        path = self.root / "annotations.json"
        document = json.loads(path.read_text())
        document["videos"]["milk/net 0/a.mp4"]["expected_events"] = ["in"]
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            dataset.read_annotations(self.root)
        document["videos"]["milk/net 0/a.mp4"]["expected_events"] = ["in", "out"]
        path.write_text(json.dumps(document))
        cases, _ = dataset.read_annotations(self.root)
        self.assertEqual(
            ("in", "out"),
            next(case.expected_events for case in cases if case.path.name == "a.mp4"),
        )

    def test_duplicate_json_keys_and_outside_paths_are_rejected(self):
        path = self.root / "annotations.json"
        path.write_text('{"schema_version":1,"videos":{},"videos":{}}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            dataset.read_annotations(self.root)
        path.write_text(
            json.dumps({"schema_version": 1, "videos": {"../outside.mp4": {}}})
        )
        with self.assertRaisesRegex(ValueError, "inside"):
            dataset.read_annotations(self.root)


class EvaluationTests(unittest.TestCase):
    def case(
        self,
        name="milk/net 0/clip.mp4",
        expected=0,
        events=None,
        food="milk",
    ):
        return dataset.VideoCase(
            Path(name), name, food, expected, "scenario", "description", events
        )

    def test_zero_net_without_detected_events_is_not_event_success(self):
        cases = [self.case(events=("in", "out"))]
        rows = list(
            evaluate.iter_results(
                cases, ("placement",), lambda *_: evaluate.Prediction(0, ())
            )
        )
        metrics = evaluate.metrics(rows)
        self.assertEqual(1, metrics["net_accuracy"])
        self.assertEqual(0, metrics["event_sequence_accuracy"])
        self.assertEqual(0, metrics["event_recall"])
        self.assertEqual(0, metrics["event_f1"])

    def test_event_order_matters(self):
        rows = list(
            evaluate.iter_results(
                [self.case(events=("in", "out"))],
                ("motion",),
                lambda *_: evaluate.Prediction(0, ("out", "in")),
            )
        )
        metrics = evaluate.metrics(rows)
        self.assertEqual(0, metrics["event_sequence_accuracy"])
        self.assertEqual(0.5, metrics["event_precision"])

    def test_unannotated_events_are_not_treated_as_empty_sequences(self):
        rows = list(
            evaluate.iter_results(
                [self.case()], ("motion",), lambda *_: evaluate.Prediction(0, ())
            )
        )
        self.assertIsNone(evaluate.metrics(rows)["event_sequence_accuracy"])

    def test_failure_continues_and_cannot_inflate_net_accuracy(self):
        def predict(method, case):
            if case.path.name == "bad.mp4":
                raise RuntimeError("bad video")
            return evaluate.Prediction(1, ("in",))

        cases = [
            self.case("milk/net 1/bad.mp4", 1),
            self.case("milk/net 1/good.mp4", 1),
        ]
        rows = list(evaluate.iter_results(cases, ("placement", "motion"), predict))
        self.assertEqual(4, len(rows))
        summary = evaluate.summarize(rows)
        self.assertEqual(0.5, summary["placement"]["overall"]["net_accuracy"])
        self.assertEqual(1, summary["motion"]["overall"]["processing_failures"])
        self.assertEqual(
            0, summary["motion"]["overall"]["mean_absolute_error_successful"]
        )

    def test_inconsistent_predictions_are_processing_failures(self):
        rows = list(
            evaluate.iter_results(
                [self.case()], ("motion",), lambda *_: evaluate.Prediction(1, ())
            )
        )
        self.assertIn("disagrees", rows[0].error or "")

    def test_food_selection_and_negatives(self):
        milk = self.case()
        other = self.case("others/net 0/other.mp4", food="others")
        cheese = self.case("cheese/net 1/cheese.mp4", expected=1, food="cheese")
        self.assertEqual([milk], evaluate.select_cases([milk, other, cheese]))
        self.assertEqual(
            [milk, other],
            evaluate.select_cases([milk, other, cheese], include_negatives=True),
        )

    def test_cancelled_inference_propagates_for_partial_reporting(self):
        with self.assertRaises(KeyboardInterrupt):
            list(
                evaluate.iter_results(
                    [self.case()], ("motion",), mock.Mock(side_effect=KeyboardInterrupt)
                )
            )

    def test_cli_report_both_methods_and_cancellation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "videos"
            write_dataset(root)
            model = base / "model.pt"
            model.write_bytes(b"fake checkpoint")
            output = base / "report.json"
            args = [
                "--root",
                str(root),
                "--scenario",
                "add_and_remove_milk",
                "--model",
                str(model),
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(
                    evaluate,
                    "_predictor",
                    return_value=lambda *_: evaluate.Prediction(0, ("in", "out")),
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(0, evaluate.main(args))
            report = json.loads(output.read_text())
            self.assertEqual(2, len(report["results"]))
            self.assertEqual({"motion", "placement"}, set(report["summary"]))
            self.assertEqual(str(model.resolve()), report["model"])
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                evaluate.main(args)
            self.assertEqual(2, raised.exception.code)
            args[-1] = str(base / "cancelled.json")
            with (
                mock.patch.object(
                    evaluate,
                    "_predictor",
                    return_value=mock.Mock(side_effect=KeyboardInterrupt),
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(130, evaluate.main(args))
            self.assertTrue(
                json.loads((base / "cancelled.json").read_text())["cancelled"]
            )


if __name__ == "__main__":
    unittest.main()
