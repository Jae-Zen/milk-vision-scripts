"""Shared evidence integrity, algorithm isolation, and resumable comparisons."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milk_vision.algorithms import jev, motion, placement
from milk_vision.evaluation.dataset import VideoCase
from milk_vision.evaluation.runner import _parser, _predictor
from milk_vision.observations import storage
from milk_vision.observations.collector import clip_tracked_box
from milk_vision.types import (
    BoundingBox,
    FrameObservations,
    RawDetection,
    TrackedDetection,
    VideoObservations,
)


def observations() -> VideoObservations:
    frames = []
    for index in range(12):
        raw = ()
        tracked = ()
        if index < 9:
            y = 10 if index < 4 else 60
            box = BoundingBox(0, y - 2, 10, y + 2)
            raw = (RawDetection(box, 0.9, 0, 1),)
            tracked = (TrackedDetection("milk", 1, index, box, 0.9),)
        frames.append(FrameObservations(index, index / 10, 100, 100, raw, tracked))
    return VideoObservations(
        {"fps": 10, "frame_count": 12, "duration_seconds": 1.2}, tuple(frames)
    )


class SharedObservationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_tracker_box_clipping_matches_ultralytics_without_changing_raw_box(self):
        import numpy as np
        import torch
        from ultralytics.engine.results import Results

        raw = BoundingBox(-1, -2, 120, 130)
        result = Results(
            orig_img=np.zeros((100, 100, 3), dtype=np.uint8),
            path="frame.jpg",
            names={0: "milk"},
        )
        result.update(boxes=torch.tensor([[-1, -2, 120, 130, 1, 0.9, 0]]))
        assert result.boxes is not None
        expected = BoundingBox(*result.boxes.xyxy[0].tolist())
        self.assertEqual(expected, clip_tracked_box(raw, 100, 100))
        self.assertEqual(BoundingBox(-1, -2, 120, 130), raw)

    def test_round_trip_preserves_empty_frames_and_both_coordinate_representations(
        self,
    ):
        original = observations()
        path = self.root / "observations.json"
        storage.save(path, original, {"video": "hash"})
        loaded = storage.load(path, {"video": "hash"})
        self.assertEqual(original, loaded)
        self.assertEqual((), loaded.frames[-1].raw)
        self.assertEqual((), loaded.frames[-1].tracked)
        self.assertEqual(9, len(loaded.motion_rows()))
        self.assertEqual([], list(self.root.glob(".pending-*")))

    def test_incomplete_corrupt_stale_and_noncontiguous_artifacts_are_rejected(self):
        path = self.root / "observations.json"
        storage.save(path, observations(), {"version": 1})
        original = json.loads(path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "Stale"):
            storage.load(path, {"version": 2})
        for change in ("complete", "hash", "frames"):
            value = json.loads(json.dumps(original))
            if change == "complete":
                value["complete"] = False
            elif change == "hash":
                value["data"]["frames"][0]["width"] = 200
            else:
                value["data"]["frames"].pop()
                value["data_sha256"] = storage.digest(value["data"])
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                storage.load(path)

    def test_motion_cannot_modify_shared_inputs_and_placement_uses_tracked_boxes(self):
        original = observations()
        raw = original.motion_rows()
        request = jev.make_request(raw, "jev-1.14.0", framework="episodes")
        self.assertEqual(1, placement.analyze(original).net)
        motion.analyze(original, motion.MotionConfig())
        self.assertEqual(raw, original.motion_rows())
        self.assertEqual(
            request,
            jev.make_request(
                original.motion_rows(), "jev-1.14.0", framework="episodes"
            ),
        )
        shifted = VideoObservations(
            original.metadata,
            tuple(
                FrameObservations(
                    f.frame_index, f.timestamp_seconds, f.width, f.height, (), f.tracked
                )
                for f in original.frames
            ),
        )
        self.assertEqual(1, placement.analyze(shifted).net)
        self.assertEqual([], shifted.motion_rows())

    def test_empty_detection_collection_is_successfully_cached_and_refresh_is_versioned(
        self,
    ):
        empty = VideoObservations(
            {"fps": 10, "frame_count": 1, "duration_seconds": 0.1},
            (FrameObservations(0, 0, 100, 100, (), ()),),
        )
        with (
            mock.patch.object(storage, "identity", return_value={"test": 1}),
            mock.patch.object(storage, "collect_video", return_value=empty) as collect,
        ):
            first, a = storage.get_observations(
                self.root / "v", self.root / "m", cache_dir=self.root / "cache"
            )
            second, b = storage.get_observations(
                self.root / "v", self.root / "m", cache_dir=self.root / "cache"
            )
            _, c = storage.get_observations(
                self.root / "v",
                self.root / "m",
                cache_dir=self.root / "cache",
                refresh=True,
            )
        self.assertEqual(first, second)
        self.assertEqual(2, collect.call_count)
        self.assertFalse(a["cache_hit"])
        self.assertTrue(b["cache_hit"])
        self.assertEqual(0, b["detection_count"])
        self.assertNotEqual(a["artifact"], c["artifact"])
        self.assertTrue(Path(a["artifact"]).is_file())

    def test_extraction_failure_creates_no_completed_artifact(self):
        with (
            mock.patch.object(storage, "identity", return_value={"test": 1}),
            mock.patch.object(
                storage, "collect_video", side_effect=RuntimeError("failed")
            ),
            self.assertRaises(RuntimeError),
        ):
            storage.get_observations(
                self.root / "v", self.root / "m", cache_dir=self.root / "cache"
            )
        self.assertEqual([], list((self.root / "cache").rglob("*.json")))

    def test_model_and_confidence_changes_invalidate_identity(self):
        video, model = self.root / "v", self.root / "m"
        video.write_bytes(b"video")
        model.write_bytes(b"model")
        with mock.patch.object(
            storage.importlib.metadata, "version", return_value="test"
        ):
            first = storage.identity(video, model, 0.1, "cpu")
            self.assertNotEqual(first, storage.identity(video, model, 0.2, "cpu"))
            model.write_bytes(b"new model")
            self.assertNotEqual(first, storage.identity(video, model, 0.1, "cpu"))

    def test_all_methods_reuse_one_collection_and_prediction_cache_resumes(self):
        obs = observations()
        args = Namespace(
            model=self.root / "model",
            track_conf=0.1,
            device="cpu",
            cache_dir=self.root / "observations",
            refresh_observations=False,
            rerun=False,
            jev_model="jev-1.14.0",
            inward="down",
            conf=0.4,
            gate_y=0.27,
            api_timeout=45,
        )
        case = VideoCase(
            self.root / "video",
            "milk/net 1/v.mp4",
            "milk",
            1,
            "add_milk",
            "Label stays local",
            ("in",),
        )
        details = {
            "observation_sha256": "test",
            "detection_count": 9,
            "extraction_seconds": 1,
        }
        body = {
            "model": "jev-1.14.0",
            "answers": {
                "episodes": {
                    "type": "choice",
                    "choice": "inward",
                    "confidence": 1,
                    "probabilities": {k: float(k == "inward") for k in jev.EPISODES},
                }
            },
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        with (
            mock.patch.object(
                storage, "get_observations", return_value=(obs, details)
            ) as collect,
            mock.patch.object(jev, "call", return_value=body) as call,
        ):
            predict = _predictor(args, motion.MotionConfig())
            for method in ("placement", "motion", "jev"):
                self.assertEqual(1, predict(method, case).net)
            self.assertEqual(1, collect.call_count)
            self.assertNotIn("Label stays local", json.dumps(call.call_args.args[0]))
            self.assertTrue(
                _predictor(args, motion.MotionConfig())("jev", case).details[
                    "prediction_cache_hit"
                ]
            )
            self.assertEqual(1, call.call_count)

    def test_method_interfaces_preserve_both_and_reject_conflicting_flags(self):
        self.assertIsNone(_parser().parse_args([]).method)
        self.assertEqual(
            ["motion", "jev"],
            _parser().parse_args(["--methods", "motion", "jev"]).methods,
        )
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            _parser().parse_args(["--method", "both", "--methods", "jev"])


if __name__ == "__main__":
    unittest.main()
