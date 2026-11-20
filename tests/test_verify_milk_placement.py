from __future__ import annotations

import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
from milk_vision import (
    cli,
    config,
    crossings,
    diagnostics,
    errors,
    inference,
    pipeline,
    types,
)


def detection(track_id, frame, center_y, confidence=0.9, class_name="milk"):
    return types.TrackedDetection(
        class_name,
        track_id,
        frame,
        types.BoundingBox(100, center_y - 10, 140, center_y + 10),
        confidence,
    )


class CrossingDetectorTests(unittest.TestCase):
    def make_detector(self, max_gap_frames=5, **kwargs):
        return crossings.CrossingDetector(
            gate_y=0.20,
            hysteresis=0.04,
            confirm_observations=2,
            max_gap_frames=max_gap_frames,
            **kwargs,
        )

    def test_hysteresis_confirms_once_and_ignores_jitter(self):
        gate = self.make_detector()
        events = [
            event
            for frame, y in enumerate([100, 110, 190, 205, 290, 300, 310])
            if (event := gate.observe(detection(1, frame, y), 1000)) is not None
        ]
        self.assertEqual([types.CrossingEvent("milk", 1, crossings.INWARD, 5)], events)

    def test_round_trip_emits_inward_then_outward(self):
        gate = self.make_detector()
        events = [
            event
            for frame, y in enumerate([100, 100, 300, 300, 100, 100])
            if (event := gate.observe(detection(1, frame, y), 1000)) is not None
        ]
        self.assertEqual(
            [crossings.INWARD, crossings.OUTWARD], [e.direction for e in events]
        )
        self.assertEqual(0, crossings.calculate_net_delta(events))

    def test_brief_gap_preserves_state(self):
        gate = self.make_detector()
        for frame, y in [(0, 100), (1, 100), (4, 300)]:
            self.assertIsNone(gate.observe(detection(1, frame, y), 1000))
        event = gate.observe(detection(1, 5, 300), 1000)
        assert event is not None
        self.assertEqual(crossings.INWARD, event.direction)

    def test_long_gap_resets_before_observing(self):
        gate = self.make_detector(max_gap_frames=2)
        gate.observe(detection(1, 0, 100), 1000)
        gate.observe(detection(1, 1, 100), 1000)
        result = gate.observe_detailed(detection(1, 10, 300), 1000)
        self.assertEqual("tracking_gap", result.state_reset_reason)
        self.assertIsNone(result.stable_zone_before)
        self.assertIsNone(gate.observe(detection(1, 11, 300), 1000))

    def test_expire_discards_missing_track(self):
        gate = self.make_detector(max_gap_frames=2)
        gate.observe(detection(1, 0, 100), 1000)
        gate.observe(detection(1, 1, 100), 1000)
        gate.expire(4)
        self.assertNotIn(("milk", 1), gate._states)

    def test_track_id_change_does_not_inherit_side(self):
        gate = self.make_detector()
        gate.observe(detection(1, 0, 100), 1000)
        gate.observe(detection(1, 1, 100), 1000)
        self.assertIsNone(gate.observe(detection(2, 2, 300), 1000))
        self.assertIsNone(gate.observe(detection(2, 3, 300), 1000))

    def test_initial_inner_side_is_not_an_inward_crossing(self):
        gate = self.make_detector()
        for frame, y in enumerate([200, 220, 300, 310, 320]):
            self.assertIsNone(gate.observe(detection(1, frame, y), 1000))
        self.assertEqual(crossings.INNER, gate._states[("milk", 1)].stable_zone)

    def test_low_confidence_never_initializes_a_side(self):
        gate = self.make_detector()
        for frame, y in enumerate([100, 100, 300, 300]):
            result = gate.observe_detailed(detection(1, frame, y, 0.3999), 1000)
            self.assertFalse(result.crossing_eligible)
            self.assertIsNone(result.observed_zone)
            self.assertIsNone(result.stable_zone_after)
            self.assertEqual(0, result.candidate_count)
            self.assertIsNone(result.event)

    def test_exact_confidence_threshold_is_accepted(self):
        gate = self.make_detector()
        result = None
        for frame, y in enumerate([100, 100, 300, 300]):
            result = gate.observe_detailed(detection(1, frame, y, 0.40), 1000)
            self.assertTrue(result.crossing_eligible)
        assert result is not None
        assert result.event is not None
        self.assertEqual(crossings.INWARD, result.event.direction)

    def test_weak_observations_do_not_increment_or_cancel_candidates(self):
        gate = self.make_detector()
        gate.observe(detection(1, 0, 100), 1000)
        gate.observe(detection(1, 1, 100), 1000)
        candidate = gate.observe_detailed(detection(1, 2, 300), 1000)
        self.assertEqual(1, candidate.candidate_count)
        for frame, y in [(3, 300), (4, 100)]:
            weak = gate.observe_detailed(detection(1, frame, y, 0.18), 1000)
            self.assertEqual(crossings.OUTER, weak.stable_zone_after)
            self.assertEqual(crossings.INNER, weak.candidate_zone)
            self.assertEqual(1, weak.candidate_count)
            self.assertIsNone(weak.event)
        event = gate.observe(detection(1, 5, 300), 1000)
        assert event is not None
        self.assertEqual(crossings.INWARD, event.direction)

    def test_weak_matches_keep_track_alive_across_short_confidence_gap(self):
        gate = self.make_detector(max_gap_frames=2, max_evidence_gap_frames=6)
        event = None
        for frame, y, conf in [
            (0, 100, 0.9),
            (1, 100, 0.9),
            (2, 300, 0.18),
            (3, 300, 0.18),
            (4, 300, 0.18),
            (5, 300, 0.9),
            (6, 300, 0.9),
        ]:
            event = gate.observe(detection(1, frame, y, conf), 1000)
            gate.expire(frame)
            if frame < 6:
                self.assertIsNone(event)
        assert event is not None
        self.assertEqual(crossings.INWARD, event.direction)

    def test_prolonged_weak_matches_reset_confident_history(self):
        gate = self.make_detector(max_gap_frames=2, max_evidence_gap_frames=4)
        gate.observe(detection(1, 0, 100), 1000)
        gate.observe(detection(1, 1, 100), 1000)
        result = None
        for frame in range(2, 7):
            result = gate.observe_detailed(detection(1, frame, 300, 0.18), 1000)
            gate.expire(frame)
            self.assertIsNone(result.event)
        assert result is not None
        self.assertEqual("confident_evidence_gap", result.state_reset_reason)
        self.assertIsNone(result.stable_zone_after)
        self.assertIsNone(gate.observe(detection(1, 7, 300), 1000))
        self.assertIsNone(gate.observe(detection(1, 8, 300), 1000))

    def test_confident_return_to_stable_side_cancels_candidate(self):
        gate = self.make_detector()
        result = None
        for frame, y in enumerate([100, 100, 300, 100]):
            result = gate.observe_detailed(detection(1, frame, y), 1000)
        assert result is not None
        self.assertEqual(0, result.candidate_count)
        self.assertIsNone(result.candidate_zone)

    def test_three_qualifying_observations_per_side(self):
        gate = crossings.CrossingDetector(0.2, 0.04, 3, 15)
        result = None
        for frame, y in enumerate([100, 100, 100, 300, 200, 300, 200, 300]):
            result = gate.observe_detailed(detection(1, frame, y), 1000)
            if frame < 7:
                self.assertIsNone(result.event)
        assert result is not None
        assert result.event is not None
        self.assertEqual(7, result.event.frame_index)

    def test_non_milk_and_invalid_confidence_are_rejected(self):
        gate = self.make_detector()
        with self.assertRaises(ValueError):
            gate.observe(detection(1, 0, 100, class_name="hand"), 1000)
        for conf in [-0.1, 1.1, float("nan")]:
            with self.subTest(conf=conf), self.assertRaises(ValueError):
                gate.observe(detection(1, 0, 100, conf), 1000)


class NetDeltaTests(unittest.TestCase):
    def event(self, direction, frame=0, track=1):
        return types.CrossingEvent("milk", track, direction, frame)

    def test_empty_is_zero(self):
        self.assertEqual(0, crossings.calculate_net_delta([]))

    def test_single_inward_counts_without_a_later_exit(self):
        self.assertEqual(
            1, crossings.calculate_net_delta([self.event(crossings.INWARD)])
        )

    def test_initial_outward_counts_without_a_prior_entry(self):
        self.assertEqual(
            -1, crossings.calculate_net_delta([self.event(crossings.OUTWARD)])
        )

    def test_vid_11_removal_then_replacement_with_different_ids_is_zero(self):
        events = [
            self.event(crossings.OUTWARD, 141, 9),
            self.event(crossings.INWARD, 198, 13),
        ]
        self.assertEqual(0, crossings.calculate_net_delta(events))

    def test_two_independent_inward_events_count_twice(self):
        self.assertEqual(
            2,
            crossings.calculate_net_delta(
                [
                    self.event(crossings.INWARD, 10, 1),
                    self.event(crossings.INWARD, 11, 2),
                ]
            ),
        )

    def test_invalid_direction_and_non_milk_are_rejected(self):
        for event in [
            self.event("sideways"),
            types.CrossingEvent("hand", 1, crossings.INWARD, 0),
        ]:
            with self.subTest(event=event), self.assertRaises(ValueError):
                crossings.calculate_net_delta([event])


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, detections, with_ids=True):
        self.id = FakeTensor([d.track_id for d in detections]) if with_ids else None
        self.cls = FakeTensor([4 if d.class_name == "milk" else 7 for d in detections])
        self.conf = FakeTensor([d.confidence for d in detections])
        self.xyxy = FakeTensor(
            [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in detections]
        )
        self.count = len(detections)

    def cpu(self):
        return self

    def numpy(self):
        return self

    def __len__(self):
        return self.count


class ExtractionAndDiagnosticsTests(unittest.TestCase):
    def test_extraction_keeps_weak_milk_for_liveness_but_excludes_hand(self):
        milk = detection(1, 0, 100, 0.18)
        result = SimpleNamespace(
            boxes=FakeBoxes([milk, detection(2, 0, 100, class_name="hand")])
        )
        self.assertEqual([milk], inference._extract_tracked_detections(result, 0, 4))

    def test_unconfirmed_or_missing_boxes_produce_no_observations(self):
        for boxes in [
            None,
            FakeBoxes([]),
            FakeBoxes([detection(1, 0, 100)], with_ids=False),
        ]:
            with self.subTest(boxes=boxes):
                self.assertEqual(
                    [],
                    inference._extract_tracked_detections(
                        SimpleNamespace(boxes=boxes), 0, 4
                    ),
                )

    def test_debug_csv_explicitly_marks_tracking_only(self):
        gate = crossings.CrossingDetector(0.2, 0.04, 3, 15)
        milk = detection(1, 10, 100, 0.18)
        observation = gate.observe_detailed(milk, 1000)
        rows = diagnostics.build_frame_debug_rows(
            10, 30, 1000, [milk], {("milk", 1): observation}
        )
        self.assertFalse(rows[0]["crossing_eligible"])
        self.assertEqual("tracking_only", rows[0]["observation_use"])
        self.assertEqual("", rows[0]["observed_zone"])
        self.assertEqual("", rows[0]["emitted_crossing_direction"])
        self.assertNotIn("associated_track_ids", rows[0])
        self.assertEqual(set(diagnostics.DEBUG_CSV_FIELDS), set(rows[0]))

    def test_empty_debug_frame_is_preserved(self):
        row = diagnostics.build_frame_debug_rows(11, 20, 1000, [], {})[0]
        self.assertEqual(0, row["detection_count"])
        self.assertEqual(0.55, row["timestamp_seconds"])

    def test_debug_summary_only_contains_sorted_milk_events(self):
        output = io.StringIO()
        result = types.PlacementResult(
            net_delta=0,
            events=[
                types.CrossingEvent("milk", 2, crossings.INWARD, 198),
                types.CrossingEvent("milk", 1, crossings.OUTWARD, 141),
            ],
            video_path=Path("video.mp4"),
            model_path=Path("model.pt"),
            fps=30,
            confidence=0.4,
            tracking_confidence=0.1,
            gate_y=0.2,
            confirm_observations=3,
            max_gap_frames=15,
            max_evidence_gap_frames=30,
            csv_path=Path("debug.csv"),
            video_debug_path=Path("debug.mp4"),
        )
        cli.print_placement_result(result, stream=output)
        summary = output.getvalue()
        self.assertLess(
            summary.index("frame_index=141"), summary.index("frame_index=198")
        )
        self.assertIn("tracking_confidence=0.100 crossing_confidence=0.400", summary)
        self.assertIn("DEBUG RESULT net_delta=0", summary)
        self.assertNotIn("HAND", summary)


class VerifyVideoIntegrationTests(unittest.TestCase):
    def run_video(
        self, rows, *, debug=False, fps=30.0, names=None, stderr=None, **kwargs
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, model_path = root / "input.mp4", root / "model.pt"
            video.touch()
            model_path.touch()
            frame = SimpleNamespace(shape=(1000, 1000, 3))
            capture = mock.Mock()
            capture.isOpened.return_value = True
            capture.get.return_value = fps
            capture.read.side_effect = (
                [(True, frame) for _ in rows] + [(False, None)]
            ) * 2
            cv2 = mock.Mock()
            cv2.VideoCapture.return_value = capture
            cv2.VideoWriter.return_value.isOpened.return_value = True
            model = mock.Mock()
            model.names = {4: "milk"} if names is None else names
            model.predict.side_effect = [
                [SimpleNamespace(boxes=FakeBoxes(ds))] for ds in rows
            ]
            tracker = mock.Mock()
            tracker.update.side_effect = [
                [
                    [
                        d.box.x1,
                        d.box.y1,
                        d.box.x2,
                        d.box.y2,
                        d.track_id,
                        d.confidence,
                        4,
                        i,
                    ]
                    for i, d in enumerate(ds)
                ]
                for ds in rows
            ]
            yolo = mock.Mock(return_value=model)
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "cv2": cv2,
                        "ultralytics": SimpleNamespace(YOLO=yolo),
                        "ultralytics.engine.results": SimpleNamespace(
                            Results=SimpleNamespace
                        ),
                        "ultralytics.trackers.bot_sort": SimpleNamespace(
                            BOTSORT=mock.Mock(return_value=tracker)
                        ),
                        "ultralytics.utils": SimpleNamespace(
                            YAML=SimpleNamespace(load=lambda _: {}),
                            IterableSimpleNamespace=SimpleNamespace,
                        ),
                    },
                ),
                mock.patch.object(diagnostics, "_draw_debug_frame", return_value=frame),
                contextlib.redirect_stderr(
                    stderr if stderr is not None else io.StringIO()
                ),
            ):
                try:
                    result = pipeline.verify_placement_video(
                        video,
                        model_path,
                        debug=debug,
                        debug_dir=root / "debug",
                        **kwargs,
                    )
                    cli.print_placement_result(result)
                    net = result.net_delta
                finally:
                    self.assertTrue(capture.release.called)
            debug_rows = []
            if debug:
                with (root / "debug" / "input_tracking_debug.csv").open(
                    newline=""
                ) as handle:
                    debug_rows = list(csv.DictReader(handle))
                self.assertTrue(cv2.VideoWriter.return_value.release.called)
            return net, model, debug_rows

    def test_milk_only_pipeline_uses_two_thresholds(self):
        rows = [
            [detection(1, i, y, conf)]
            for i, (y, conf) in enumerate(
                [(100, 0.9)] * 3 + [(500, 0.18)] * 3 + [(500, 0.9)] * 3
            )
        ]
        net, model, debug_rows = self.run_video(rows, debug=True)
        self.assertEqual(1, net)
        for call in model.predict.call_args_list:
            self.assertEqual(0.1, call.kwargs["conf"])
            self.assertEqual([4], call.kwargs["classes"])
            self.assertFalse(call.kwargs["stream"])
        for row in debug_rows[3:6]:
            self.assertEqual("False", row["crossing_eligible"])
            self.assertEqual("tracking_only", row["observation_use"])
            self.assertEqual("", row["emitted_crossing_direction"])
            self.assertEqual("0", row["candidate_count"])
        event_rows = [r for r in debug_rows if r["emitted_crossing_direction"]]
        self.assertEqual(["8"], [r["frame_index"] for r in event_rows])

    def test_initial_outward_is_counted_without_a_hand_class(self):
        rows = [[detection(1, i, y)] for i, y in enumerate([500] * 3 + [100] * 3)]
        self.assertEqual(-1, self.run_video(rows)[0])

    def test_crossing_events_print_once_with_frames_in_both_modes(self):
        rows = [[detection(1, i, y)] for i, y in enumerate([500] * 3 + [100] * 3)]
        for debug in (False, True):
            with self.subTest(debug=debug):
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    net, _, _ = self.run_video(rows, debug=debug, stderr=stderr)
                self.assertEqual(-1, net)
                self.assertEqual("", stdout.getvalue())
                self.assertEqual(1, stderr.getvalue().count("CrossingEvent("))
                self.assertIn("direction='outward', frame_index=5", stderr.getvalue())
                self.assertEqual(debug, "DEBUG CONFIGURATION" in stderr.getvalue())

    def test_no_crossings_prints_an_empty_list_without_debug(self):
        stderr = io.StringIO()
        net, _, _ = self.run_video([[], []], stderr=stderr)
        self.assertEqual(0, net)
        self.assertEqual("MILK EVENTS (0)\n  []\n", stderr.getvalue())

    def test_weak_crossing_alone_cannot_change_inventory(self):
        rows = [
            [detection(1, i, y, conf)]
            for i, (y, conf) in enumerate(
                [(100, 0.9)] * 3 + [(500, 0.18)] * 3 + [(100, 0.9)] * 3
            )
        ]
        self.assertEqual(0, self.run_video(rows)[0])

    def test_no_frames_is_an_error_not_zero(self):
        with self.assertRaisesRegex(errors.VerificationError, "No frames"):
            self.run_video([])

    def test_missing_milk_class_and_invalid_fps_release_capture(self):
        with self.assertRaisesRegex(errors.VerificationError, "milk"):
            self.run_video([], names={7: "hand"})
        with self.assertRaisesRegex(errors.VerificationError, "FPS"):
            self.run_video([], fps=float("nan"))


class BoTSORTConfidenceTests(unittest.TestCase):
    def test_actual_tracker_uses_weak_matches_but_does_not_start_weak_tracks(self):
        try:
            import numpy as np
            import yaml
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.bot_sort import BOTSORT
        except ImportError:
            self.skipTest("Optional inference dependencies are not installed")
        with config.TRACKER_CONFIG.open() as handle:
            tracker_config = SimpleNamespace(**yaml.safe_load(handle))
        self.assertEqual(0.4, tracker_config.new_track_thresh)
        self.assertEqual(0.25, tracker_config.track_high_thresh)
        self.assertEqual(0.1, tracker_config.track_low_thresh)
        tracker = BOTSORT(tracker_config)

        def update(rows):
            return tracker.update(
                Boxes(np.asarray(rows, dtype=np.float32), orig_shape=(1000, 1000))
            )

        strong = update([[100, 100, 200, 200, 0.9, 4]])
        track_id = strong[0, 4]
        weak = update([[100, 100, 200, 200, 0.18, 4], [600, 600, 700, 700, 0.3, 4]])
        self.assertEqual(1, len(weak))
        self.assertEqual(track_id, weak[0, 4])
        self.assertAlmostEqual(0.18, weak[0, 5], places=5)
        strong_again = update([[100, 100, 200, 200, 0.9, 4]])
        self.assertEqual(track_id, strong_again[0, 4])


class ValidationTests(unittest.TestCase):
    def test_only_milk_class_is_required(self):
        self.assertEqual(4, config.resolve_milk_class_id({4: " Milk "}))
        self.assertEqual(4, config.resolve_milk_class_id({4: "Milk", 7: "HAND"}))
        with self.assertRaises(errors.VerificationError):
            config.resolve_milk_class_id({7: "hand"})

    def test_defaults_and_overrides(self):
        args = cli.parse_args(["placement", "--video", "unused.mp4"])
        self.assertEqual(0.4, args.conf)
        self.assertEqual(0.1, args.track_conf)
        args = cli.parse_args(
            [
                "placement",
                "--video",
                "unused.mp4",
                "--conf",
                ".6",
                "--track-conf",
                ".12",
            ]
        )
        self.assertEqual((0.6, 0.12), (args.conf, args.track_conf))

    def test_invalid_threshold_relationship_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dummy"
            path.touch()
            for conf in [0, 0.5, float("nan")]:
                with (
                    self.subTest(conf=conf),
                    self.assertRaisesRegex(
                        errors.VerificationError, "Tracking confidence"
                    ),
                ):
                    pipeline.verify_placement_video(
                        path, path, confidence=0.4, tracking_confidence=conf
                    )

    def test_frame_conversion_requires_three_hits_at_this_video_fps(self):
        self.assertEqual(
            3, crossings._seconds_to_frames(0.1, 30.015571988475223, minimum=2)
        )
        self.assertEqual(2, crossings._seconds_to_frames(0.1, 5, minimum=2))

    def test_invalid_video_keeps_stdout_empty(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(
                [
                    "placement",
                    "--video",
                    str(SCRIPTS_DIR / "definitely_missing_video.mp4"),
                ]
            )
        self.assertEqual(1, code)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("ERROR:", stderr.getvalue())

    def test_cli_preserves_signed_integer_output(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(
                cli,
                "verify_placement_video",
                return_value=SimpleNamespace(net_delta=-1, events=[], csv_path=None),
            ) as verify,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = cli.main(["placement", "--video", "unused.mp4"])
        self.assertEqual(0, code)
        self.assertEqual("-1\n", stdout.getvalue())
        self.assertEqual("MILK EVENTS (0)\n  []\n", stderr.getvalue())
        self.assertFalse(verify.call_args.kwargs["debug"])
        self.assertEqual(0.4, verify.call_args.kwargs["confidence"])
        self.assertEqual(0.1, verify.call_args.kwargs["tracking_confidence"])

    def test_debug_cli_keeps_summary_off_stdout(self):
        def fake_verify(**kwargs):
            self.assertTrue(kwargs["debug"])
            self.assertEqual(Path("diagnostics"), kwargs["debug_dir"])
            return types.PlacementResult(
                0,
                [],
                Path("unused.mp4"),
                config.PLACEMENT_MODEL,
                30.0,
                0.4,
                0.1,
                0.27,
                3,
                15,
                30,
                Path("diagnostics/unused_tracking_debug.csv"),
                Path("diagnostics/unused_annotated_debug.mp4"),
            )

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(cli, "verify_placement_video", side_effect=fake_verify),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = cli.main(
                [
                    "placement",
                    "--video",
                    "unused.mp4",
                    "--debug",
                    "--debug-dir",
                    "diagnostics",
                ]
            )
        self.assertEqual(0, code)
        self.assertEqual("0\n", stdout.getvalue())
        self.assertIn("DEBUG RESULT net_delta=0", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
