"""Exercise notebook helper functions without Drive, GPU, Git or API execution."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks" / "evaluate_milk_colab.ipynb"
)


def notebook_helpers():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "subprocess": subprocess,
        "sys": sys,
        "hashlib": hashlib,
        "json": json,
    }
    functions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            functions.extend(
                node for node in tree.body if isinstance(node, ast.FunctionDef)
            )
    for node in functions:
        compiled = compile(
            ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"
        )
        code = next(
            value for value in compiled.co_consts if isinstance(value, types.CodeType)
        )
        defaults = (
            tuple(ast.literal_eval(value) for value in node.args.defaults) or None
        )
        namespace[node.name] = types.FunctionType(code, namespace, node.name, defaults)
    return notebook, namespace


class ColabWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.notebook, self.helpers = notebook_helpers()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def command(self, methods, **kwargs):
        return self.helpers["build_evaluation_command"](
            "python",
            "/videos",
            "/model.pt",
            "/cache",
            "/report.json",
            methods,
            **kwargs,
        )

    def test_all_seven_checkbox_combinations_and_empty_rejection(self):
        for flags in itertools.product((False, True), repeat=3):
            expected = [
                name
                for name, enabled in zip(("placement", "motion", "jev"), flags, strict=True)
                if enabled
            ]
            if not expected:
                with self.assertRaises(ValueError):
                    self.helpers["choose_methods"](*flags)
                continue
            methods = self.helpers["choose_methods"](*flags)
            self.assertEqual(expected, methods)
            command = self.command(methods)
            start = command.index("--methods") + 1
            self.assertEqual(expected, command[start : command.index("--model")])

    def test_fresh_resume_filters_and_invalid_limits(self):
        self.assertIn("--rerun", self.command(["jev"]))
        self.assertNotIn("--rerun", self.command(["jev"], mode="Resume"))
        command = self.command(
            ["motion"], include_negatives=False, scenario=" add_milk ", limit=2
        )
        self.assertNotIn("--include-negatives", command)
        self.assertEqual("add_milk", command[command.index("--scenario") + 1])
        self.assertEqual("2", command[command.index("--limit") + 1])
        for limit in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                self.command(["motion"], limit=limit)
        with self.assertRaises(ValueError):
            self.command(["unknown"])

    def test_scope_preflight_only_lists_and_never_predicts(self):
        process = mock.Mock()
        process.check_output.return_value = "milk/net 1/a.mp4\n"
        with mock.patch.dict(self.helpers, {"subprocess": process}):
            self.assertEqual(
                ["milk/net 1/a.mp4"],
                self.helpers["select_scope"](self.command(["jev"]), self.root),
            )
        command = process.check_output.call_args.args[0]
        self.assertIn("--list", command)
        self.assertNotIn("--rerun", command)
        self.assertNotIn("--output", command)
        process.run.assert_not_called()

    def test_revision_refresh_reuses_checkout_and_refuses_dirty_edits(self):
        repo, code = self.root / "repo", self.root / "code"
        repo.mkdir()
        sha = "a" * 40
        checkout = code / sha / "Scripts"
        checkout.mkdir(parents=True)
        marker = checkout / "my_edit.py"
        marker.write_text("preserve me", encoding="utf-8")

        def git_output(_repo, *args):
            if args == ("remote", "get-url", "origin"):
                return "https://example/repo.git"
            if args == ("status", "--porcelain"):
                return " M my_edit.py"
            return sha

        process = mock.Mock()
        with (
            mock.patch.dict(
                self.helpers, {"subprocess": process, "git_output": git_output}
            ),
            self.assertRaisesRegex(RuntimeError, "Runtime checkout has edits"),
        ):
            self.helpers["fetch_revision"](
                "https://example/repo.git", "main", repo, code
            )
        self.assertEqual("preserve me", marker.read_text(encoding="utf-8"))
        self.assertEqual(
            ["git", "-C", str(repo), "fetch", "origin", "main"],
            process.run.call_args.args[0],
        )
        with self.assertRaises(ValueError):
            self.helpers["fetch_revision"]("url", "--bad", repo, code)

    def test_old_published_revision_fails_with_actionable_error(self):
        process = mock.Mock()
        with mock.patch.dict(self.helpers, {"subprocess": process}):
            with self.assertRaisesRegex(
                RuntimeError, "Publish the current local implementation"
            ):
                self.helpers["require_shared_evaluator"](self.root)
            process.check_output.assert_not_called()
            for name in (
                "milk_vision/algorithms/jev.py",
                "milk_vision/observations/storage.py",
                "tools/evaluate.py",
            ):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            process.check_output.return_value = (
                "--methods --cache-dir --rerun --jev-model"
            )
            self.helpers["require_shared_evaluator"](self.root)
            process.check_output.reset_mock()
            self.helpers["require_shared_evaluator"](self.root, check_cli=False)
            process.check_output.assert_not_called()

    def test_history_distinguishes_complete_partial_and_failed_runs(self):
        for name, count, error in (
            ("complete", 2, None),
            ("partial", 1, None),
            ("failed", 2, "network"),
        ):
            directory = self.root / name
            directory.mkdir()
            metadata = {
                "label": name,
                "requested_ref": "main",
                "commit": "abc",
                "selected_videos": 1,
                "methods": ["placement", "jev"],
                "scope_sha256": "data",
                "settings_sha256": "settings",
                "gpu": "T4",
            }
            (directory / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            results = [
                {
                    "video": "a.mp4",
                    "error": error,
                    "details": {"collection": {"observation_sha256": "obs"}},
                }
                for _ in range(count)
            ]
            report = {"results": results, "summary": {}, "cancelled": False}
            (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
        rows = self.helpers["history_rows"](self.root)
        self.assertEqual(6, len(rows))
        states = {row["run"]: row["status"] for row in rows}
        self.assertEqual(
            {
                "complete": "complete",
                "partial": "partial",
                "failed": "processing failures",
            },
            states,
        )

    def test_notebook_controls_are_in_run_cell_and_no_outputs_are_exported(self):
        source = "".join(self.notebook["cells"][6]["source"])
        self.assertEqual(
            3, source.count('type:"boolean"}') - 1
        )  # fourth checkbox selects negatives
        self.assertIn(
            "METHODS = choose_methods(RUN_PLACEMENT, RUN_MOTION, RUN_JEV)", source
        )
        self.assertIn('RUN_MODE = "Fresh predictions"', source)
        self.assertNotIn(
            "rmtree(SCRIPTS)", "".join(self.notebook["cells"][3]["source"])
        )
        self.assertTrue(all(not cell.get("outputs") for cell in self.notebook["cells"]))


if __name__ == "__main__":
    unittest.main()
