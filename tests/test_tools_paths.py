from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from tools import extract_fridge_frames as extractor


class ToolLocationTests(unittest.TestCase):
    def test_defaults_keep_existing_project_locations(self):
        self.assertEqual(SCRIPTS_DIR / "fridge items", extractor.DEFAULT_OUTPUT)
        self.assertEqual(SCRIPTS_DIR, extractor.SCRIPTS_DIR)

    def test_file_and_module_commands_work_from_another_directory(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SCRIPTS_DIR)
        with tempfile.TemporaryDirectory() as temporary:
            for entry in (
                [str(SCRIPTS_DIR / "tools/extract_fridge_frames.py")],
                ["-m", "tools.extract_fridge_frames"],
            ):
                with self.subTest(entry=entry):
                    result = subprocess.run(
                        [sys.executable, *entry, "--help"],
                        cwd=temporary,
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_module_commands_work_from_parent_project(self):
        result = subprocess.run(
            [sys.executable, "-m", "Scripts.tools.extract_fridge_frames", "--help"],
            cwd=SCRIPTS_DIR.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
