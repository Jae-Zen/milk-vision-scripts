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


def samples(ys, track_id: int | None = 1, start_frame=0, confidence=0.9, fps=30):
    return [{"frame_index": start_frame + i, "timestamp_seconds": (start_frame + i) / fps,
             "track_id": track_id, "confidence": confidence, "center_y_normalized": y}
            for i, y in enumerate(ys)]


def accepted_fits(ys, slopes=None, step=0.1):
    """Minimal accepted-fit rows for testing event state independently of fitting."""
    slopes = slopes or [0.0] + [(ys[i] - ys[i - 1]) / step for i in range(1, len(ys))]
    return [{"frame_index": i, "timestamp_seconds": i * step, "track_id": 1,
             "segment_id": 0, "fit_status": "accepted", "fitted_y_normalized": y,
             "slope_per_second": slopes[i]} for i, y in enumerate(ys)]


class MotionTests(unittest.TestCase):
    def analyze(self, rows, **kwargs):
        return motion.analyze_detections(rows, motion.MotionConfig(**kwargs))

    def test_sparse_recorded_add_and_remove(self):
        rows = json.loads((Path(__file__).parent / "fixtures" / "milk_motion_204912.json").read_text())
        events = self.analyze(rows)
        self.assertEqual(["inward", "outward"], [e["direction"] for e in events])
        self.assertEqual([152, 266], [e["frame_index"] for e in events])
        for frame in (140, 141):
            row = next(r for r in rows if r["frame_index"] == frame)
            self.assertEqual("accepted", row["fit_status"])
            self.assertEqual(2, row["window_observations"])

    def test_one_observation_cannot_define_a_slope(self):
        rows = samples([0.1])
        self.assertEqual([], self.analyze(rows))
        self.assertEqual("insufficient_evidence", rows[0]["fit_status"])
        self.assertEqual("one_observation_cannot_define_slope", rows[0]["fit_reason"])

    def test_two_observations_are_enough_for_a_fit(self):
        rows = samples([0.1, 0.2], fps=10)
        self.assertEqual(["inward"], [e["direction"] for e in self.analyze(rows)])
        self.assertTrue(all(r["fit_status"] == "accepted" for r in rows))

    def test_neutral_step_preserves_unfinished_candidate(self):
        rows = accepted_fits([0.50, 0.48, 0.46, 0.46, 0.44, 0.42])
        events = motion.infer_events(rows, motion.MotionConfig())
        self.assertEqual(["outward"], [event["direction"] for event in events])
        self.assertEqual("stationary_or_uncertain", rows[3]["motion"])
        self.assertEqual("up", rows[3]["candidate_direction"])
        self.assertEqual(2, rows[3]["candidate_hits"])

    def test_opposite_motion_replaces_unfinished_candidate(self):
        rows = accepted_fits([0.50, 0.48, 0.46, 0.48, 0.50, 0.52, 0.54])
        events = motion.infer_events(rows, motion.MotionConfig())
        self.assertEqual(["inward"], [event["direction"] for event in events])
        self.assertAlmostEqual(0.08, events[0]["displacement_normalized"])

    def test_same_direction_stays_locked_after_incomplete_opposite_motion(self):
        rows = accepted_fits([0.10, 0.13, 0.16, 0.19, 0.17, 0.15,
                              0.18, 0.21, 0.24, 0.27])
        events = motion.infer_events(rows, motion.MotionConfig())
        self.assertEqual(["inward"], [event["direction"] for event in events])

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
        self.assertEqual("accepted", rows[40]["fit_status"])

    def test_moving_track_survives_an_outlier(self):
        ys = [0.1 + i * 0.01 for i in range(50)]
        ys[20] = 0.95
        rows = samples(ys)
        self.assertEqual(["inward"], [e["direction"] for e in self.analyze(rows)])
        self.assertEqual("accepted", rows[20]["fit_status"])

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

    def test_untracked_detections_are_not_motion_evidence(self):
        rows = samples([0.1 + i * 0.01 for i in range(50)], None)
        self.assertEqual([], self.analyze(rows))
        self.assertTrue(all(row["fit_status"] == "untracked" for row in rows))

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
                              ("--track-conf", "0.9")]:
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                motion.parse_args(["--video", "test.mp4", option, value])
            self.assertEqual(2, raised.exception.code)

    def test_raw_boxes_are_preserved_including_untracked_detection(self):
        import numpy as np
        import torch
        from ultralytics.engine.results import Results
        result = Results(np.zeros((100, 200, 3), dtype=np.uint8), "test", {0: "milk"},
                         boxes=torch.tensor([[10, 10, 30, 30, 0.9, 0], [80, 60, 100, 80, 0.2, 0]]))
        tracks = np.array([[11, 11, 31, 31, 7, 0.9, 0, 0]])
        rows = motion.detection_rows(result, tracks, 3, 30)
        self.assertEqual(2, len(rows))
        self.assertEqual(7, rows[0]["track_id"])
        self.assertEqual(20, rows[0]["center_y"])
        self.assertIsNone(rows[1]["track_id"])


if __name__ == "__main__":
    unittest.main()
