"""Tests for ``scripts.record_loc``."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_loc as rl  # noqa: E402


class TestRecordLoc(unittest.TestCase):
    def _make_tree(self, root: str) -> None:
        os.makedirs(os.path.join(root, "pkg"))
        os.makedirs(os.path.join(root, "pkg", "cpp"))
        os.makedirs(os.path.join(root, "build"))
        os.makedirs(os.path.join(root, ".git"))
        os.makedirs(os.path.join(root, "__pycache__"))
        with open(os.path.join(root, "pkg", "a.py"), "w", encoding="utf-8") as f:
            f.write("a = 1\nb = 2\nc = 3\n")
        with open(os.path.join(root, "pkg", "b.py"), "w", encoding="utf-8") as f:
            f.write("x = 1\n")
        with open(
            os.path.join(root, "pkg", "cpp", "m.cpp"), "w", encoding="utf-8"
        ) as f:
            f.write("int main() {\n  return 0;\n}\n")
        with open(os.path.join(root, "pkg", "cpp", "m.h"), "w", encoding="utf-8") as f:
            f.write("#pragma once\n")
        # Files that must be ignored: build artefacts and unrelated
        # extensions.
        with open(
            os.path.join(root, "build", "ignored.py"), "w", encoding="utf-8"
        ) as f:
            f.write("should not be counted\n")
        with open(
            os.path.join(root, "__pycache__", "ignored.py"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("should not be counted\n")
        with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as f:
            f.write("doc\n")

    def test_count_source_tree_counts_python_and_cpp(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_tree(tmp)
            totals = rl.count_source_tree(tmp)
            self.assertEqual(
                totals["Python"],
                {"files": 2, "lines": 4, "code_lines": 4, "comment_lines": 0},
            )
            self.assertEqual(
                totals["C++"],
                {"files": 2, "lines": 4, "code_lines": 4, "comment_lines": 0},
            )

    def test_count_source_tree_distinguishes_code_and_comment_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "src"))
            with open(os.path.join(tmp, "src", "a.py"), "w", encoding="utf-8") as f:
                # 7 lines: 2 comment, 1 blank, 2 code, 1 code-with-trailing-comment,
                # 1 code line with ``#`` inside a string (must be code, not comment).
                f.write(
                    "# header comment\n"
                    "# another comment\n"
                    "\n"
                    "x = 1\n"
                    "y = 2  # trailing comment\n"
                    "z = '# not a comment'\n"
                    "print(x + y)\n"
                )
            with open(os.path.join(tmp, "src", "a.cpp"), "w", encoding="utf-8") as f:
                # 8 lines: 1 block-comment opening line, 1 inside block comment,
                # 1 block-comment closing line, 1 code line, 1 line comment,
                # 1 code-with-trailing-comment, 1 blank, 1 code with ``//`` in string.
                f.write(
                    "/* multi\n"
                    " * line\n"
                    " */\n"
                    "int x = 1;\n"
                    "// pure comment\n"
                    "int y = 2; // trailing\n"
                    "\n"
                    'const char* s = "http://example";\n'
                )
            totals = rl.count_source_tree(tmp)
            self.assertEqual(
                totals["Python"],
                {"files": 1, "lines": 7, "code_lines": 4, "comment_lines": 2},
            )
            self.assertEqual(
                totals["C++"],
                {"files": 1, "lines": 8, "code_lines": 3, "comment_lines": 4},
            )

    def test_append_rows_writes_header_then_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_tree(tmp)
            out = os.path.join(tmp, "out", "loc.csv")
            totals = rl.count_source_tree(tmp)
            rl.append_rows(out, totals, date_iso="2024-01-02T03:04:05Z", commit="abc")
            rl.append_rows(out, totals, date_iso="2024-01-03T03:04:05Z", commit="def")
            with open(out, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                set(rows[0].keys()),
                set(rl.CSV_FIELDS),
            )
            languages = sorted({r["language"] for r in rows})
            self.assertEqual(languages, ["C++", "Python"])
            self.assertEqual(rows[0]["commit"], "abc")
            self.assertEqual(rows[-1]["commit"], "def")
            self.assertEqual(rows[0]["files"], "2")
            self.assertEqual(rows[0]["lines"], "4")
            self.assertEqual(rows[0]["code_lines"], "4")
            self.assertEqual(rows[0]["comment_lines"], "0")

    def test_append_rows_migrates_legacy_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_tree(tmp)
            out = os.path.join(tmp, "loc.csv")
            # Simulate a CSV produced by a previous version of the script
            # with only the original five columns.
            with open(out, "w", encoding="utf-8", newline="") as f:
                f.write("date,commit,language,files,lines\n")
                f.write("2024-01-01T00:00:00Z,old,Python,5,42\n")
            totals = rl.count_source_tree(tmp)
            rl.append_rows(out, totals, date_iso="2024-02-01T00:00:00Z", commit="new")
            with open(out, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            # Header is rewritten to include the new columns.
            self.assertEqual(set(rows[0].keys()), set(rl.CSV_FIELDS))
            # Legacy row is preserved with empty back-filled values.
            legacy = next(r for r in rows if r["commit"] == "old")
            self.assertEqual(legacy["lines"], "42")
            self.assertEqual(legacy["code_lines"], "")
            self.assertEqual(legacy["comment_lines"], "")
            # New rows carry the additional columns.
            for r in rows:
                if r["commit"] == "new":
                    self.assertEqual(r["code_lines"], "4")

    def test_main_records_rows_with_explicit_commit_and_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_tree(tmp)
            out = os.path.join(tmp, "loc.csv")
            rc = rl.main(
                [
                    "--source-dir",
                    tmp,
                    "--output",
                    out,
                    "--commit",
                    "deadbeef",
                    "--date",
                    "2024-05-06T07:08:09Z",
                ]
            )
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["commit"] for r in rows}, {"deadbeef"})
            self.assertEqual({r["date"] for r in rows}, {"2024-05-06T07:08:09Z"})

    def test_main_rejects_missing_source_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "loc.csv")
            missing = os.path.join(tmp, "does-not-exist")
            rc = rl.main(
                [
                    "--source-dir",
                    missing,
                    "--output",
                    out,
                    "--commit",
                    "x",
                    "--date",
                    "2024-01-01T00:00:00Z",
                ]
            )
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
