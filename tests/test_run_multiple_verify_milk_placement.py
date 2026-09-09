from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import run_multiple_verify_milk_placement as runner


class VideoDiscoveryTests(unittest.TestCase):
    def test_finds_every_video_in_natural_filename_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ("vid_10.mp4", "notes.txt", "vid_2.MOV", "vid_1.avi"):
                (folder / name).touch()
            (folder / "nested.mp4").mkdir()

            self.assertEqual(
                ["vid_1.avi", "vid_2.MOV", "vid_10.mp4"],
                [path.name for path in runner.find_videos(folder)],
            )


class MainTests(unittest.TestCase):
    def test_processes_all_videos_without_a_count_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for index in range(55):
                (folder / f"video_{index}.mp4").touch()

            completed = mock.Mock(returncode=0, stdout="0\n", stderr="")
            with mock.patch.object(
                runner.subprocess, "run", return_value=completed
            ) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = runner.main(["--folder", str(folder)])

            self.assertEqual(0, code)
            self.assertEqual(55, run.call_count)
            processed = [Path(call.args[0][-1]).name for call in run.call_args_list]
            self.assertEqual(
                [f"video_{index}.mp4" for index in range(55)], processed
            )


if __name__ == "__main__":
    unittest.main()
