from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from milk_vision import (
    batch,
    cli,
    config,
    diagnostics,
    errors,
    inference,
    motion,
    types,
)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        for name in ("vid_10.mp4", "vid_2.MOV", "notes.txt"):
            (self.folder / name).touch()
        (self.folder / "nested.mp4").mkdir()

    def run_batch(self, method, extra=(), **run_kwargs):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(batch.subprocess, "run", **run_kwargs) as run,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = cli.main([method, "--folder", str(self.folder), *extra])
        return code, run, stdout.getvalue(), stderr.getvalue()

    def test_both_methods_use_natural_order_and_inherit_streams(self):
        for method in ("placement", "motion"):
            with self.subTest(method=method):
                code, run, stdout, _ = self.run_batch(
                    method,
                    return_value=SimpleNamespace(returncode=0),
                )
                self.assertEqual(0, code)
                names = []
                for call in run.call_args_list:
                    command = call.args[0]
                    self.assertEqual(sys.executable, command[0])
                    self.assertEqual(["-m", "milk_vision", method], command[1:4])
                    self.assertEqual({"env", "check"}, set(call.kwargs))
                    self.assertFalse(call.kwargs["check"])
                    self.assertEqual(
                        str(SCRIPTS_DIR),
                        call.kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0],
                    )
                    names.append(Path(command[command.index("--video") + 1]).name)
                self.assertEqual(["vid_2.MOV", "vid_10.mp4"], names)
                self.assertIn("Completed 2 videos; 0 failed.", stdout)

    def test_forwards_method_specific_options(self):
        for method, extra in (
            (
                "placement",
                ["--debug-dir", "debug output", "--conf", ".6", "--gate-y", ".3"],
            ),
            (
                "motion",
                ["--output-dir", "debug output", "--min-speed", ".2", "--inward", "up"],
            ),
        ):
            with self.subTest(method=method):
                code, run, _, _ = self.run_batch(
                    method,
                    ["--debug", "--device", "cpu", *extra],
                    return_value=SimpleNamespace(returncode=0),
                )
                self.assertEqual(0, code)
                for call in run.call_args_list:
                    command = call.args[0]
                    self.assertIn("--debug", command)
                    self.assertEqual("cpu", command[command.index("--device") + 1])
                    for flag, value in zip(extra[::2], extra[1::2], strict=True):
                        actual = command[command.index(flag) + 1]
                        if flag in {"--conf", "--gate-y", "--min-speed"}:
                            self.assertEqual(float(value), float(actual))
                        else:
                            self.assertEqual(value, actual)

    def test_failures_and_launch_errors_do_not_stop_remaining_videos(self):
        for method in ("placement", "motion"):
            for first in (SimpleNamespace(returncode=1), OSError("launch failed")):
                with self.subTest(method=method, first=first):
                    code, run, stdout, stderr = self.run_batch(
                        method,
                        side_effect=[first, SimpleNamespace(returncode=0)],
                    )
                    self.assertEqual(1, code)
                    self.assertEqual(2, run.call_count)
                    self.assertIn("1 failed", stdout)
                    self.assertIn("vid_2.MOV", stderr)

    def test_interrupt_stops_before_next_video(self):
        for method in ("placement", "motion"):
            for interruption in (
                KeyboardInterrupt(),
                SimpleNamespace(returncode=130),
                SimpleNamespace(returncode=-2),
                SimpleNamespace(returncode=-1073741510),
            ):
                with self.subTest(method=method, interruption=interruption):
                    code, run, _, _ = self.run_batch(method, side_effect=[interruption])
                    self.assertEqual(130, code)
                    self.assertEqual(1, run.call_count)

    def test_invalid_arguments_fail_before_launching(self):
        for method, extras in (
            (
                "placement",
                [
                    ["--conf", "nan"],
                    ["--track-conf", ".9"],
                    ["--gate-y", "0"],
                    ["--video", "other.mp4"],
                    ["--unknown"],
                ],
            ),
            (
                "motion",
                [
                    ["--min-speed", "0"],
                    ["--window-seconds", "nan"],
                    ["--track-conf", "nan"],
                    ["--video", "other.mp4"],
                    ["--unknown"],
                ],
            ),
        ):
            for extra in extras:
                with self.subTest(method=method, extra=extra):
                    with (
                        mock.patch.object(batch.subprocess, "run") as run,
                        contextlib.redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit) as raised,
                    ):
                        cli.main([method, "--folder", str(self.folder), *extra])
                    self.assertEqual(2, raised.exception.code)
                    run.assert_not_called()

    def test_missing_and_empty_folders_are_usage_errors(self):
        empty = self.folder / "empty"
        empty.mkdir()
        for method in ("placement", "motion"):
            for folder in (empty, self.folder / "missing"):
                with self.subTest(method=method, folder=folder):
                    with (
                        contextlib.redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit) as raised,
                    ):
                        cli.main([method, "--folder", str(folder)])
                    self.assertEqual(2, raised.exception.code)

    def test_duplicate_stems_rejected_only_when_debugging(self):
        (self.folder / "VID_2.mp4").touch()
        for method in ("placement", "motion"):
            with self.subTest(method=method):
                with (
                    mock.patch.object(batch.subprocess, "run") as run,
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    cli.main([method, "--folder", str(self.folder), "--debug"])
                self.assertEqual(2, raised.exception.code)
                run.assert_not_called()
                code, run, _, _ = self.run_batch(
                    method, return_value=SimpleNamespace(returncode=0)
                )
                self.assertEqual(0, code)
                self.assertEqual(3, run.call_count)

    def test_real_child_output_is_live_and_successful_diagnostics_are_visible(self):
        # A harmless fake verifier tests actual stream inheritance and banner order.
        fake_package = self.folder / "milk_vision"
        fake_package.mkdir()
        (fake_package / "__init__.py").touch()
        fake = fake_package / "__main__.py"
        fake.write_text(
            "import sys\nprint('CHILD DIAGNOSTIC', file=sys.stderr, flush=True)\nprint('-1', flush=True)\n"
        )
        source = (
            f"import sys; sys.path.insert(0, {str(SCRIPTS_DIR)!r}); "
            "from pathlib import Path; from milk_vision import batch, cli; "
            f"batch.SCRIPTS_DIR = Path({str(self.folder)!r}); "
            f"sys.exit(cli.main(['motion', '--folder', {str(self.folder)!r}]))"
        )
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=self.folder,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stdout)
        self.assertEqual(2, result.stdout.count("CHILD DIAGNOSTIC"))
        self.assertEqual(2, result.stdout.count("\n-1\n"))
        self.assertLess(
            result.stdout.index("[1/2]"), result.stdout.index("CHILD DIAGNOSTIC")
        )


class PackageBoundaryTests(unittest.TestCase):
    def test_placement_summary_handles_unavailable_stderr(self):
        result = types.PlacementResult(
            0,
            [],
            Path("clip.mp4"),
            config.PLACEMENT_MODEL,
            30,
            0.4,
            0.1,
            0.27,
            3,
            15,
            30,
        )
        with (
            mock.patch.object(sys, "stderr", None),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            cli.print_placement_result(result)
        self.assertEqual("", stdout.getvalue())

    def test_each_command_parses_arguments_once(self):
        original_parse = argparse.ArgumentParser.parse_args
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "clip.mp4").touch()
            for method in ("placement", "motion"):
                for flag, value, handler in (
                    ("--video", "clip.mp4", f"run_{method}"),
                    ("--folder", folder, "run_batch"),
                ):
                    with self.subTest(method=method, flag=flag):
                        with (
                            mock.patch.object(
                                argparse.ArgumentParser,
                                "parse_args",
                                autospec=True,
                                side_effect=original_parse,
                            ) as parse,
                            mock.patch.object(cli, handler, return_value=0),
                            contextlib.redirect_stdout(io.StringIO()),
                        ):
                            self.assertEqual(0, cli.main([method, flag, value]))
                        self.assertEqual(1, parse.call_count)

    def test_shared_error_and_cancellation_handling(self):
        for method in ("placement", "motion"):
            for failure, expected in (
                (errors.VerificationError("failed"), 1),
                (KeyboardInterrupt(), 130),
            ):
                with self.subTest(method=method, failure=type(failure)):
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with (
                        mock.patch.object(cli, f"run_{method}", side_effect=failure),
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(stderr),
                    ):
                        self.assertEqual(
                            expected, cli.main([method, "--video", "clip.mp4"])
                        )
                    self.assertEqual("", stdout.getvalue())
                    self.assertIn(
                        "ERROR:" if expected == 1 else "Cancelled.", stderr.getvalue()
                    )

    def test_default_paths_remain_at_original_locations(self):
        placement = cli.parse_args(["placement", "--video", "unused.mp4"])
        motion_args = cli.parse_args(["motion", "--video", "unused.mp4"])
        self.assertEqual(
            SCRIPTS_DIR.parent / "Models/milk-vision-v4s.pt", placement.model
        )
        self.assertEqual(
            SCRIPTS_DIR.parent / "Models/milk-vision-v4n.pt", motion_args.model
        )
        self.assertEqual(
            SCRIPTS_DIR.parent / "debug/milk_motion", motion_args.output_dir
        )
        self.assertEqual(
            (
                SCRIPTS_DIR / "clip_tracking_debug.csv",
                SCRIPTS_DIR / "clip_annotated_debug.mp4",
            ),
            diagnostics._debug_artifact_paths(Path("clip.mp4")),
        )
        self.assertTrue(config.TRACKER_CONFIG.is_file())
        self.assertEqual("observations", config.TRACKER_CONFIG.parent.name)

    def test_analysis_imports_without_site_packages(self):
        source = f"import sys; sys.path.insert(0, {str(SCRIPTS_DIR)!r}); from milk_vision import crossings, motion"
        result = subprocess.run(
            [sys.executable, "-S", "-c", source],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_package_commands_work_from_another_directory_without_site_packages(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SCRIPTS_DIR)
        with tempfile.TemporaryDirectory() as cwd:
            for method in ("placement", "motion"):
                for is_batch in (False, True):
                    with self.subTest(method=method, is_batch=is_batch):
                        command = [sys.executable, "-S", "-m", "milk_vision", method]
                        help_result = subprocess.run(
                            [*command, "--help"],
                            cwd=cwd,
                            env=environment,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        self.assertEqual(0, help_result.returncode, help_result.stderr)
                        for option in ("--model", "--debug", "--video", "--folder"):
                            self.assertIn(option, help_result.stdout)
                        flag = "--folder" if is_batch else "--video"
                        result = subprocess.run(
                            [*command, flag, "missing"],
                            cwd=cwd,
                            env=environment,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        self.assertEqual(
                            2 if is_batch else 1, result.returncode, result.stderr
                        )
                        self.assertEqual("", result.stdout)

    def test_package_runs_from_parent_project_without_installation(self):
        result = subprocess.run(
            [sys.executable, "-S", "-m", "Scripts.milk_vision", "motion", "--help"],
            cwd=SCRIPTS_DIR.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--folder", result.stdout)

    def test_dispatcher_passes_parsed_arguments_to_handlers(self):
        for method, target in (
            ("placement", "run_placement"),
            ("motion", "run_motion"),
        ):
            with self.subTest(method=method):
                parsed = cli.parse_args([method, "--video", "clip.mp4"])
                with (
                    mock.patch.object(cli, "parse_args", return_value=parsed) as parse,
                    mock.patch.object(cli, target, return_value=-1) as handler,
                    contextlib.redirect_stdout(io.StringIO()) as stdout,
                ):
                    self.assertEqual(0, cli.main([method, "--video", "clip.mp4"]))
                parse.assert_called_once_with([method, "--video", "clip.mp4"])
                handler.assert_called_once_with(parsed)
                self.assertEqual("-1\n", stdout.getvalue())
                with tempfile.TemporaryDirectory() as folder:
                    (Path(folder) / "clip.mp4").touch()
                    with mock.patch.object(cli, "run_batch", return_value=1) as handler:
                        self.assertEqual(1, cli.main([method, "--folder", folder]))
                    args = handler.call_args.args[0]
                    self.assertEqual(method, args.method)
                    self.assertEqual([Path(folder) / "clip.mp4"], args.videos)

    def test_dispatcher_rejects_missing_method_or_conflicting_inputs(self):
        for arguments in (
            [],
            ["unknown"],
            ["motion"],
            ["placement", "--video", "clip.mp4", "--folder", "videos"],
        ):
            with self.subTest(arguments=arguments):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    cli.main(arguments)
                self.assertEqual(2, raised.exception.code)

    def test_reanalysis_is_independent_of_csv_column_order(self):
        rows: list[types.MotionRow] = [
            {
                "frame_index": i,
                "timestamp_seconds": i / 10,
                "track_id": 1,
                "confidence": 0.9,
                "center_y_normalized": i / 10,
            }
            for i in range(6)
        ]
        expected = motion.analyze_detections(rows, motion.MotionConfig())
        with mock.patch.object(
            diagnostics, "CSV_FIELDS", tuple(reversed(diagnostics.CSV_FIELDS))
        ):
            self.assertEqual(
                expected, motion.analyze_detections(rows, motion.MotionConfig())
            )
        rows[0]["event"] = "stale"
        motion.analyze_detections(rows, motion.MotionConfig())
        self.assertNotIn("event", rows[0])

    def test_motion_outputs_keep_schema_and_metadata(self):
        rows: list[types.MotionRow] = [
            {
                "frame_index": i,
                "timestamp_seconds": i / 10,
                "track_id": 1,
                "confidence": 0.9,
                "center_y_normalized": i / 10,
            }
            for i in range(6)
        ]
        settings = motion.MotionConfig()
        events = motion.analyze_detections(rows, settings)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.object(diagnostics, "plot_motion"):
                diagnostics.write_outputs(
                    types.MotionResult(rows, events, {"fps": 10}, 1),
                    settings,
                    output,
                    "clip",
                )
            with (output / "clip_motion.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(list(diagnostics.CSV_FIELDS), reader.fieldnames)
                self.assertEqual(6, len(list(reader)))
            report = json.loads((output / "clip_motion.json").read_text())
            self.assertEqual(
                {
                    "fps",
                    "config",
                    "detection_count",
                    "events",
                    "net_delta",
                    "interpretation",
                },
                set(report),
            )
            self.assertEqual(events, report["events"])
            self.assertEqual(1, report["net_delta"])

    def test_motion_pipeline_and_cli_keep_signed_result_separate_from_diagnostics(self):
        rows = [
            {
                "frame_index": i,
                "timestamp_seconds": i / 10,
                "track_id": 1,
                "confidence": 0.9,
                "center_y_normalized": 0.9 - i / 10,
            }
            for i in range(6)
        ]
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(
                inference, "collect_detections", return_value=(rows, {"fps": 10})
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = cli.main(["motion", "--video", "unused.mp4"])
        self.assertEqual(0, code)
        self.assertEqual("-1\n", stdout.getvalue())
        self.assertIn('"direction": "outward"', stderr.getvalue())

    def test_diagnostic_video_failure_closes_open_files_and_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = diagnostics.PlacementDiagnostics(
                Path("clip.mp4"), Path(directory), 30, 0.27
            )
            cv2 = mock.Mock()
            cv2.VideoWriter.return_value.isOpened.return_value = False
            item = types.PlacementFrame(None, 0, 100, 100, [])
            with (
                mock.patch.dict(sys.modules, {"cv2": cv2}),
                self.assertRaisesRegex(
                    errors.VerificationError, "could not create debug video"
                ),
                writer,
            ):
                writer.write_frame(item, {})
            assert writer.handle is not None
            self.assertTrue(writer.handle.closed)
            cv2.VideoWriter.return_value.release.assert_called_once()


class MotionInferenceTests(unittest.TestCase):
    def run_inference(self, predictions):
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30
        frames = [SimpleNamespace(shape=(100, 100, 3)) for _ in predictions]
        capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = capture
        model = mock.Mock()
        model.names = {0: "milk"}
        model.predict.side_effect = predictions
        tracker = mock.Mock()
        tracker.update.return_value = []
        modules = {
            "cv2": cv2,
            "ultralytics": SimpleNamespace(YOLO=mock.Mock(return_value=model)),
            "ultralytics.engine.results": SimpleNamespace(Results=SimpleNamespace),
            "ultralytics.trackers.bot_sort": SimpleNamespace(
                BOTSORT=mock.Mock(return_value=tracker)
            ),
            "ultralytics.utils": SimpleNamespace(
                YAML=SimpleNamespace(load=lambda path: {}),
                IterableSimpleNamespace=SimpleNamespace,
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            model_path = Path(directory) / "model.pt"
            video.touch()
            model_path.touch()
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(
                    inference, "detection_rows", return_value=[]
                ) as extract,
            ):
                try:
                    result = inference.collect_detections(video, model_path, 0.1, "cpu")
                finally:
                    capture.release.assert_called_once()
        return result, model, tracker, extract, frames

    def test_updates_tracker_for_every_frame_including_empty_detections(self):
        class EmptyBoxes:
            xyxy = SimpleNamespace(tolist=list)
            conf = SimpleNamespace(tolist=list)

            def cpu(self):
                return self

            def numpy(self):
                return self

            def __len__(self):
                return 0

        empty, detected = EmptyBoxes(), EmptyBoxes()
        predictions = [
            [SimpleNamespace(boxes=empty)],
            [SimpleNamespace(boxes=detected)],
        ]
        (rows, metadata), model, tracker, _extract, frames = self.run_inference(
            predictions
        )
        self.assertEqual([], rows)
        assert "frame_count" in metadata
        self.assertEqual(2, metadata["frame_count"])
        self.assertEqual(2, tracker.update.call_count)
        self.assertEqual(
            (empty.cpu().numpy(), frames[0]), tracker.update.call_args_list[0].args
        )
        self.assertEqual(
            (detected.cpu().numpy(), frames[1]), tracker.update.call_args_list[1].args
        )
        self.assertEqual(
            frames, [call.args[0] for call in model.predict.call_args_list]
        )
        for call in model.predict.call_args_list:
            self.assertEqual(0.1, call.kwargs["conf"])
            self.assertEqual([0], call.kwargs["classes"])
            self.assertEqual("cpu", call.kwargs["device"])

    def test_releases_capture_on_model_failure(self):
        with self.assertRaisesRegex(
            errors.VerificationError, "Video inference failed: model failed"
        ):
            self.run_inference([RuntimeError("model failed")])

    def test_zero_decoded_frames_remains_an_error(self):
        with self.assertRaisesRegex(errors.VerificationError, "no decoded frames"):
            self.run_inference([])


if __name__ == "__main__":
    unittest.main()
