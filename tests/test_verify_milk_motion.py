from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_milk_motion as motion


def samples(ys, track_id=1, start_frame=0, confidence=0.9, fps=30):
    return [{"frame_index": start_frame + i, "timestamp_seconds": (start_frame + i) / fps,
             "track_id": track_id, "confidence": confidence, "center_y_normalized": y}
            for i, y in enumerate(ys)]


class MotionTests(unittest.TestCase):
    def analyze(self, rows, **kwargs):
        return motion.analyze_detections(rows, motion.MotionConfig(**kwargs))

    def test_direction_and_one_count_for_long_continuous_movement(self):
        rows = samples([0.1 + i * 0.006 for i in range(100)])
        events = self.analyze(rows)
        self.assertEqual(["inward"], [e["direction"] for e in events])
        self.assertAlmostEqual(0.18, rows[20]["slope_per_second"])
        self.assertAlmostEqual(0.006, rows[20]["fitted_delta_y_normalized"])

    def test_up_and_camera_mapping(self):
        for inward, expected in [("down", "outward"), ("up", "inward")]:
            events = self.analyze(samples([0.8 - i * 0.01 for i in range(45)]), inward=inward)
            self.assertEqual([expected], [e["direction"] for e in events])

    def test_round_trip_preserves_reversal(self):
        ys = [0.1 + i * 0.01 for i in range(45)] + [0.55 - i * 0.01 for i in range(46)]
        self.assertEqual(["inward", "outward"], [e["direction"] for e in self.analyze(samples(ys))])

    def test_stationary_jitter_and_isolated_false_positive(self):
        ys = [0.5 + (0.002 if i % 2 else -0.002) for i in range(100)]
        ys[40] = 0.95
        rows = samples(ys)
        self.assertEqual([], self.analyze(rows))
        self.assertEqual("rejected", rows[40]["fit_status"])

    def test_moving_track_survives_an_outlier(self):
        ys = [0.1 + i * 0.01 for i in range(50)]
        ys[20] = 0.95
        rows = samples(ys)
        self.assertEqual(["inward"], [e["direction"] for e in self.analyze(rows)])
        self.assertEqual("rejected", rows[20]["fit_status"])

    def test_separate_objects_keep_opposite_directions(self):
        rows = samples([0.1 + i * 0.01 for i in range(45)], 1)
        rows += samples([0.8 - i * 0.01 for i in range(45)], 2)
        self.assertEqual({(1, "inward"), (2, "outward")},
                         {(e["track_id"], e["direction"]) for e in self.analyze(rows)})

    def test_gaps_do_not_create_movement_between_stationary_positions(self):
        rows = samples([0.1] * 20) + samples([0.8] * 20, start_frame=80)
        self.assertEqual([], self.analyze(rows))
        self.assertNotEqual(rows[0]["segment_id"], rows[-1]["segment_id"])

    def test_weak_matches_do_not_bridge_evidence_gap(self):
        rows = samples([0.1] * 20)
        rows += samples([0.5] * 60, start_frame=20, confidence=0.2)
        rows += samples([0.8] * 20, start_frame=80)
        self.assertEqual([], self.analyze(rows))
        self.assertEqual("low_confidence", rows[40]["fit_status"])
        self.assertNotEqual(rows[0]["segment_id"], rows[-1]["segment_id"])

    def test_short_and_untracked_detections_are_not_motion_evidence(self):
        rows = samples([0.1, 0.5, 0.8]) + samples([0.1 + i * 0.01 for i in range(50)], None)
        self.assertEqual([], self.analyze(rows))
        self.assertEqual("untracked", rows[-1]["fit_status"])

    def test_irregular_sampling_uses_time(self):
        rows = samples([0.1 + i * 0.005 for i in range(90)])
        rows = [row for i, row in enumerate(rows) if i % 3 != 1]
        self.assertEqual(["inward"], [e["direction"] for e in self.analyze(rows)])
        self.assertAlmostEqual(0.15, rows[20]["fitted_velocity_per_second"])

    def test_pause_does_not_recount_same_direction(self):
        ys = [0.1 + i * 0.005 for i in range(50)] + [0.35] * 60
        ys += [0.35 + i * 0.005 for i in range(50)]
        self.assertEqual(["inward"], [e["direction"] for e in self.analyze(samples(ys))])

    def test_empty_input_produces_zero_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(0, motion.write_outputs([], [], {}, motion.MotionConfig(), Path(directory), "empty"))
            report = json.loads((Path(directory) / "empty_motion.json").read_text())
            self.assertEqual([], report["events"])
            self.assertTrue((Path(directory) / "empty_motion.png").is_file())

    def test_invalid_cli_thresholds(self):
        for option, value in [("--window-seconds", "nan"), ("--min-speed", "0"),
                              ("--min-observations", "2"), ("--track-conf", "0.9")]:
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                motion.parse_args(["--video", "test.mp4", option, value])
            self.assertEqual(2, raised.exception.code)

    def test_raw_boxes_are_preserved_including_untracked_detection(self):
        import numpy as np
        from ultralytics.engine.results import Results
        result = Results(np.zeros((100, 200, 3), dtype=np.uint8), "test", {0: "milk"},
                         boxes=np.array([[10, 10, 30, 30, 0.9, 0], [80, 60, 100, 80, 0.2, 0]]))
        tracks = np.array([[11, 11, 31, 31, 7, 0.9, 0, 0]])
        rows = motion.detection_rows(result, tracks, 3, 30)
        self.assertEqual(2, len(rows))
        self.assertEqual(7, rows[0]["track_id"])
        self.assertEqual(20, rows[0]["center_y"])
        self.assertIsNone(rows[1]["track_id"])


if __name__ == "__main__":
    unittest.main()
