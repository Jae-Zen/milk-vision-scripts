"""Check experiment payload isolation, data integrity, and API response handling."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import jev_experiment as jev


def response() -> dict:
    return {
        "model": "jev-1.14.0",
        "answers": {
            "net_change": {
                "type": "choice",
                "choice": "added",
                "confidence": 0.8,
                "probabilities": {"added": 0.9, "unchanged": 0.08, "removed": 0.02},
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


class JevExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / "secret_net_1.csv"
        self.csv.write_text(
            "timestamp_seconds,center_y_normalized,confidence,event,track_id\n"
            "2,0.5,0.9,inward,42\n1,0.2,0.8,,42\n1,0.3,0.7,,43\n",
            encoding="utf-8",
        )
        self.case = self.root / "case.json"
        self.output = self.root / "results"

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            jev.main(
                [
                    "prepare",
                    "--csv",
                    str(self.csv),
                    "--expected-net",
                    "1",
                    "--output",
                    str(self.case),
                ]
            )
        return json.loads(self.case.read_text(encoding="utf-8"))

    def test_payload_contains_only_three_fields_and_preserves_simultaneous_detections(
        self,
    ):
        case = self.prepare()
        self.assertEqual(
            [[1, 0.2, 0.8], [1, 0.3, 0.7], [2, 0.5, 0.9]],
            case["request"]["state"]["observations"],
        )
        payload = jev.encode(case["request"]).decode()
        for excluded in (
            "secret_net_1",
            "expected_net",
            "baseline_net",
            "track_id",
            "inward,42",
        ):
            self.assertNotIn(excluded, payload)
        self.assertEqual(1, case["expected_net"])
        self.assertEqual(jev.digest(case["request"]), case["request_sha256"])

    def test_invalid_coordinates_fail_without_writing_a_case(self):
        for field, value in (
            ("timestamp_seconds", -1),
            ("center_y_normalized", 1.1),
            ("confidence", float("nan")),
            ("confidence", float("inf")),
        ):
            row = {
                "timestamp_seconds": 0,
                "center_y_normalized": 0.5,
                "confidence": 0.9,
            }
            row[field] = value
            with self.assertRaises(ValueError):
                jev.make_request([row], "jev-1.14.0")
        self.assertEqual(
            [], jev.make_request([], "jev-1.14.0")["state"]["observations"]
        )

    def test_request_sent_without_local_labels_and_result_retains_full_response(self):
        case = self.prepare()
        body = response()
        with (
            mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}),
            mock.patch.object(jev.urllib.request, "urlopen") as call,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            call.return_value.__enter__.return_value = io.StringIO(json.dumps(body))
            jev.main(["run", str(self.case), "--output-dir", str(self.output)])
        sent = json.loads(call.call_args.args[0].data)
        self.assertEqual(case["request"], sent)
        result = json.loads(
            next(self.output.glob("*.result.json")).read_text(encoding="utf-8")
        )
        self.assertEqual(1, result["predicted_net"])
        self.assertEqual(body, result["response"])
        self.assertNotIn("test-key", json.dumps(result))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            jev.main(["report", str(next(self.output.glob("*.result.json")))])
        self.assertEqual(
            1, json.loads(output.getvalue())["predicted_net"]["1"]["correct"]
        )

    def test_no_key_or_modified_request_never_calls_api(self):
        case = self.prepare()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("dotenv.load_dotenv"),
            mock.patch.object(jev.urllib.request, "urlopen") as call,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            jev.main(["run", str(self.case), "--output-dir", str(self.output)])
        call.assert_not_called()
        case["request"]["state"]["observations"] = []
        self.case.write_text(json.dumps(case))
        with (
            mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}),
            mock.patch.object(jev.urllib.request, "urlopen") as call,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            jev.main(["run", str(self.case), "--output-dir", str(self.output)])
        call.assert_not_called()

    def test_invalid_response_is_rejected(self):
        for invalid in (float("nan"), -0.1, 1.1, True):
            body = response()
            body["answers"]["net_change"]["confidence"] = invalid
            with self.assertRaises(ValueError):
                jev.validate_response(body)
        body = response()
        body["answers"]["net_change"]["probabilities"]["added"] = 0.1
        with self.assertRaises(ValueError):
            jev.validate_response(body)

    def test_episode_answers_are_counted_in_python_for_all_five_choices(self):
        expected = {
            "none": 0,
            "inward": 1,
            "outward": -1,
            "inward → outward": 0,
            "outward → inward": 0,
        }
        for choice, net in expected.items():
            body = response()
            body["answers"] = {
                "episodes": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 1.0,
                    "probabilities": {name: float(name == choice) for name in expected},
                }
            }
            prediction, _ = jev.validate_response(body, "episodes")
            self.assertEqual(net, prediction)

    def test_reframe_preserves_detections_and_original_case_and_runs_episode_question(
        self,
    ):
        original = self.prepare()
        before = self.case.read_bytes()
        episode_case = self.root / "episodes.json"
        with contextlib.redirect_stdout(io.StringIO()):
            jev.main(
                ["reframe", "--case", str(self.case), "--output", str(episode_case)]
            )
        updated = json.loads(episode_case.read_text(encoding="utf-8"))
        self.assertEqual(before, self.case.read_bytes())
        self.assertEqual(
            original["request"]["state"]["observations"],
            updated["request"]["state"]["observations"],
        )
        self.assertEqual(
            set(jev.EPISODES),
            set(updated["request"]["questions"]["episodes"]["criteria"]),
        )
        self.assertNotIn("inventory", json.dumps(updated["request"]))
        self.assertEqual(original["request_sha256"], updated["parent_request_sha256"])
        updated["expected_episodes"] = ["outward", "inward"]
        episode_case.write_text(json.dumps(updated))
        body = response()
        body["answers"] = {
            "episodes": {
                "type": "choice",
                "choice": "outward → inward",
                "confidence": 1.0,
                "probabilities": {
                    name: float(name == "outward → inward") for name in jev.EPISODES
                },
            }
        }
        with (
            mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}),
            mock.patch.object(jev.urllib.request, "urlopen") as call,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            call.return_value.__enter__.return_value = io.StringIO(json.dumps(body))
            jev.main(["run", str(episode_case), "--output-dir", str(self.output)])
        saved = next(self.output.glob("*.result.json"))
        result = json.loads(saved.read_text(encoding="utf-8"))
        self.assertEqual(["outward", "inward"], result["predicted_episodes"])
        self.assertEqual(
            (1, 1, 0),
            (result["inward_count"], result["outward_count"], result["predicted_net"]),
        )
        self.assertNotIn("expected_episodes", json.loads(call.call_args.args[0].data))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            jev.main(["report", str(saved)])
        self.assertEqual(
            {"labelled": 1, "correct": 1},
            json.loads(output.getvalue())["episode_sequences"],
        )


if __name__ == "__main__":
    unittest.main()
