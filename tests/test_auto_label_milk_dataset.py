from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import auto_label_milk_dataset as labeller


class DiscoveryAndSplitTests(unittest.TestCase):
    def test_duplicate_case_insensitive_stems_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "frame.jpg").write_bytes(b"one")
            (source / "FRAME.png").write_bytes(b"two")

            with self.assertRaisesRegex(ValueError, "same filename stem"):
                labeller.find_images(source)

    def test_split_is_deterministic_and_has_expected_90_10_counts(self):
        images = [Path(f"image_{index:04}.jpg") for index in range(2622)]

        first = labeller.split_images(images, val_fraction=0.10, seed=42)
        second = labeller.split_images(images, val_fraction=0.10, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(2360, sum(item.split == "train" for item in first))
        self.assertEqual(262, sum(item.split == "val" for item in first))

    def test_one_image_smoke_source_is_assigned_to_train(self):
        image = Path("only.jpg")
        self.assertEqual(
            [labeller.DatasetItem(source=image, split="train")],
            labeller.split_images([image], val_fraction=0.10, seed=42),
        )


class LabelFormattingTests(unittest.TestCase):
    def test_polygons_are_clipped_and_invalid_polygons_are_skipped(self):
        polygons = [
            np.array([[-0.1, 0.25], [0.5, 1.1], [1.0, 0.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            np.array([[0.1, np.nan], [0.2, 0.3], [0.4, 0.5]]),
        ]

        contents, class_ids = labeller.format_yolo_label(polygons, [0, 1, 1])

        self.assertEqual([0], class_ids)
        self.assertEqual(
            "0 0.000000 0.250000 0.500000 1.000000 1.000000 0.000000\n",
            contents,
        )

    def test_no_predictions_produce_an_empty_label(self):
        contents, class_ids = labeller.format_yolo_label([], [])

        self.assertEqual("", contents)
        self.assertEqual([], class_ids)

    def test_unexpected_class_is_rejected(self):
        polygon = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        with self.assertRaisesRegex(RuntimeError, "unexpected class index 2"):
            labeller.format_yolo_label([polygon], [2])


class ModelValidationTests(unittest.TestCase):
    def test_expected_segmentation_classes_are_accepted(self):
        model = SimpleNamespace(task="segment", names={0: "milk", 1: "hand"})
        labeller.validate_model(model)

    def test_wrong_class_order_is_rejected(self):
        model = SimpleNamespace(task="segment", names={0: "hand", 1: "milk"})

        with self.assertRaisesRegex(ValueError, "must be exactly"):
            labeller.validate_model(model)

    def test_non_segmentation_model_is_rejected(self):
        model = SimpleNamespace(task="detect", names={0: "milk", 1: "hand"})

        with self.assertRaisesRegex(ValueError, "task must be 'segment'"):
            labeller.validate_model(model)


class ResumeAndArchiveTests(unittest.TestCase):
    def make_args(self, root: Path) -> argparse.Namespace:
        source = root / "source"
        source.mkdir()
        model = root / "model.pt"
        model.write_bytes(b"model")
        return argparse.Namespace(
            source=source,
            model=model,
            output=root / "output",
            conf=0.25,
            iou=0.70,
            imgsz=640,
            device=None,
            batch=8,
            seed=42,
            val_fraction=0.10,
            overwrite=False,
        )

    def test_matching_state_resumes_and_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)
            state = {"version": 1, "value": "same"}

            self.assertFalse(labeller.initialize_or_resume(args, state))
            self.assertTrue(labeller.initialize_or_resume(args, state))
            with self.assertRaisesRegex(ValueError, "does not match"):
                labeller.initialize_or_resume(args, {"version": 1, "value": "new"})

    def test_built_state_round_trips_and_allows_batch_or_device_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)
            image = args.source / "frame.jpg"
            image.write_bytes(b"image")
            items = [labeller.DatasetItem(image, "train")]

            first = labeller.build_state(args, items)
            args.batch = 1
            args.device = "cpu"
            second = labeller.build_state(args, items)

            self.assertEqual(first, second)
            self.assertEqual(first, json.loads(json.dumps(first)))

    def test_archive_contains_only_upload_dataset_at_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            labeller.prepare_dataset_directories(output)
            for split, filename in (("train", "train.jpg"), ("val", "val.jpg")):
                (output / split / "images" / filename).write_bytes(b"image")
                (output / split / "labels" / f"{Path(filename).stem}.txt").write_text(
                    "0 0.1 0.1 0.2 0.2 0.3 0.3\n", encoding="utf-8"
                )
            labeller.state_path_for(output).write_text(
                json.dumps({"not": "uploaded"}), encoding="utf-8"
            )

            archive = labeller.create_archive(output)

            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(
                    {
                        "data.yaml",
                        "train/images/",
                        "train/images/train.jpg",
                        "train/labels/",
                        "train/labels/train.txt",
                        "val/images/",
                        "val/images/val.jpg",
                        "val/labels/",
                        "val/labels/val.txt",
                    },
                    set(bundle.namelist()),
                )


if __name__ == "__main__":
    unittest.main()
