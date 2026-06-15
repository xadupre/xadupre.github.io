"""Tests for ``scripts.record_so_sizes``."""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_so_sizes as rss  # noqa: E402


def _make_wheel(members: dict[str, bytes]) -> bytes:
    """Build an in-memory ``.whl`` (a zip archive) containing ``members``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _make_artifact_zip(members: dict[str, bytes]) -> bytes:
    """Build an in-memory artifact zip archive containing ``members``.

    ``actions/upload-artifact`` stores ``.whl`` files (which are themselves
    zips) without further compression, so ``ZIP_STORED`` is used here to
    faithfully reproduce that layout.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestIsSharedLibrary(unittest.TestCase):
    def test_recognises_shared_library_suffixes(self):
        for name in (
            "_onnxpykernels.cpython-312-x86_64-linux-gnu.so",
            "liblib_onnx_lib.so",
            "_module.pyd",
            "libfoo.dylib",
            "liblib_onnx_lib.so.1",
            "liblib_onnx_lib.so.1.2",
        ):
            self.assertTrue(rss._is_shared_library(name), name)

    def test_rejects_non_shared_libraries(self):
        for name in (
            "onnx_light-0.1-cp312-cp312-linux_x86_64.whl",
            "README.txt",
            "module.py",
            "notes.solution",
        ):
            self.assertFalse(rss._is_shared_library(name), name)


class TestExtractSharedLibrarySizes(unittest.TestCase):
    def test_extracts_shared_libraries_from_wheel(self):
        wheel = _make_wheel(
            {
                "onnx_light/onnx_py/_onnxpykernels.cpython-312-x86_64-linux-gnu.so": b"k" * 256,
                "onnx_light/onnx_py/liblib_onnx_lib.so": b"l" * 512,
                "onnx_light/__init__.py": b"# python",
                "onnx_light-0.1.dist-info/RECORD": b"",
            }
        )
        artifact = _make_artifact_zip(
            {"onnx_light-0.1-cp312-cp312-linux_x86_64.whl": wheel}
        )
        result = rss.extract_shared_library_sizes(artifact)
        self.assertEqual(
            result,
            [
                ("_onnxpykernels.cpython-312-x86_64-linux-gnu.so", 256),
                ("liblib_onnx_lib.so", 512),
            ],
        )

    def test_extracts_shared_libraries_stored_directly(self):
        # Some artifacts upload the shared libraries directly rather than
        # wrapped inside a wheel.
        artifact = _make_artifact_zip(
            {
                "build/_onnxpyoptim.cpython-312-x86_64-linux-gnu.so": b"o" * 32,
                "build/notes.txt": b"ignore me",
            }
        )
        self.assertEqual(
            rss.extract_shared_library_sizes(artifact),
            [("_onnxpyoptim.cpython-312-x86_64-linux-gnu.so", 32)],
        )

    def test_keeps_largest_size_for_duplicate_names(self):
        wheel_small = _make_wheel({"lib/_mod.so": b"a" * 10})
        wheel_large = _make_wheel({"lib/_mod.so": b"a" * 99})
        artifact = _make_artifact_zip(
            {
                "onnx_light-0.1-cp312-cp312-linux_x86_64.whl": wheel_small,
                "onnx_light-0.1-cp313-cp313-linux_x86_64.whl": wheel_large,
            }
        )
        self.assertEqual(
            rss.extract_shared_library_sizes(artifact), [("_mod.so", 99)]
        )

    def test_ignores_artifacts_without_shared_libraries(self):
        artifact = _make_artifact_zip({"README.txt": b"nothing here"})
        self.assertEqual(rss.extract_shared_library_sizes(artifact), [])


class TestReadExistingCommits(unittest.TestCase):
    def test_returns_recorded_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "so_sizes.csv")
            rss.append_rows(
                path,
                [
                    {"date": "d", "commit": "aaa", "size": "1", "name": "x.so"},
                    {"date": "d", "commit": "aaa", "size": "2", "name": "y.so"},
                    {"date": "d", "commit": "bbb", "size": "3", "name": "x.so"},
                ],
            )
            self.assertEqual(rss.read_existing_commits(path), {"aaa", "bbb"})

    def test_returns_empty_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                rss.read_existing_commits(os.path.join(tmp, "missing.csv")), set()
            )


class TestAppendRows(unittest.TestCase):
    def test_writes_header_then_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "onnx-light", "so_sizes.csv")
            self.assertEqual(
                rss.append_rows(
                    path,
                    [{"date": "d", "commit": "c", "size": "1", "name": "x.so"}],
                ),
                1,
            )
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            self.assertTrue(content.startswith("date,commit,size,name"))
            self.assertEqual(
                rss.append_rows(
                    path,
                    [{"date": "d2", "commit": "c2", "size": "2", "name": "y.so"}],
                ),
                1,
            )
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            rows = list(csv.DictReader(io.StringIO(content)))
            self.assertEqual([r["commit"] for r in rows], ["c", "c2"])
            # Header must only appear once even after the second append.
            self.assertEqual(content.count("date,commit,size,name"), 1)

    def test_creates_header_only_file_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "onnx-light", "so_sizes.csv")
            self.assertEqual(rss.append_rows(path, []), 0)
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(fh.read().strip(), "date,commit,size,name")


class TestProcessRun(unittest.TestCase):
    def test_skips_unfinished(self):
        run = {"id": 1, "status": "in_progress", "head_sha": "c"}
        with mock.patch.object(rss, "list_run_artifacts") as la:
            self.assertEqual(rss.process_run(run, "o/r", None), [])
            la.assert_not_called()

    def test_collects_rows_from_failed_run_with_artifacts(self):
        run = {
            "id": 7,
            "status": "completed",
            "conclusion": "failure",
            "created_at": "2024-05-02T10:00:00Z",
            "head_sha": "cafef00d",
        }
        artifact = {
            "id": 200,
            "name": "wheels-linux-x86_64",
            "expired": False,
            "archive_download_url": "https://api/artifact/200",
        }
        wheel = _make_wheel(
            {"onnx_light/onnx_py/_onnxpykernels.cpython-312.so": b"z" * 64}
        )
        zip_bytes = _make_artifact_zip(
            {"onnx_light-0.1-cp312-cp312-linux_x86_64.whl": wheel}
        )
        with mock.patch.object(
            rss, "list_run_artifacts", return_value=[artifact]
        ), mock.patch.object(rss, "_download", return_value=zip_bytes):
            rows = rss.process_run(run, "o/r", None)
        self.assertEqual(
            rows,
            [
                {
                    "date": "2024-05-02T10:00:00Z",
                    "commit": "cafef00d",
                    "size": "64",
                    "name": "_onnxpykernels.cpython-312.so",
                }
            ],
        )

    def test_skips_expired_artifacts(self):
        run = {
            "id": 9,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2024-05-01T10:00:00Z",
            "head_sha": "deadbeef",
        }
        wheel = _make_wheel({"lib/_mod.so": b"x" * 100})
        artifact_ok = {
            "id": 100,
            "name": "wheels",
            "expired": False,
            "archive_download_url": "https://api/artifact/100",
        }
        artifact_expired = {
            "id": 101,
            "name": "wheels-expired",
            "expired": True,
            "archive_download_url": "https://api/artifact/101",
        }
        zip_ok = _make_artifact_zip(
            {"onnx_light-0.1-cp312-cp312-linux_x86_64.whl": wheel}
        )
        with mock.patch.object(
            rss,
            "list_run_artifacts",
            return_value=[artifact_ok, artifact_expired],
        ), mock.patch.object(
            rss, "_download", return_value=zip_ok
        ) as dl:
            rows = rss.process_run(run, "o/r", None)
        self.assertEqual(dl.call_count, 1)
        self.assertEqual(
            rows,
            [
                {
                    "date": "2024-05-01T10:00:00Z",
                    "commit": "deadbeef",
                    "size": "100",
                    "name": "_mod.so",
                }
            ],
        )

    def test_skips_invalid_zip(self):
        run = {
            "id": 7,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2024-05-01T10:00:00Z",
            "head_sha": "c",
        }
        with mock.patch.object(
            rss,
            "list_run_artifacts",
            return_value=[
                {
                    "id": 1,
                    "name": "bad",
                    "expired": False,
                    "archive_download_url": "https://x",
                }
            ],
        ), mock.patch.object(rss, "_download", return_value=b"not a zip"):
            self.assertEqual(rss.process_run(run, "o/r", None), [])


class TestProcessRepo(unittest.TestCase):
    def test_skips_commits_already_recorded(self):
        run_old = {
            "id": 1,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2024-05-01T10:00:00Z",
            "head_sha": "old",
        }
        run_new = {
            "id": 2,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2024-05-02T10:00:00Z",
            "head_sha": "new",
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "onnx-light", "so_sizes.csv")
            # Seed the cache so the "old" commit is considered already recorded
            # (e.g. by the inline workflow step).
            rss.append_rows(
                csv_path,
                [
                    {
                        "date": "2024-05-01T10:00:00Z",
                        "commit": "old",
                        "size": "10",
                        "name": "_mod.so",
                    }
                ],
            )
            with mock.patch.object(
                rss, "iter_workflow_runs", return_value=iter([run_new, run_old])
            ), mock.patch.object(
                rss,
                "process_run",
                side_effect=lambda run, repo, token: [
                    {
                        "date": run["created_at"],
                        "commit": run["head_sha"],
                        "size": "42",
                        "name": "_new.so",
                    }
                ],
            ) as pr:
                added = rss.process_repo(
                    "xadupre/onnx-light",
                    "build_release.yml",
                    tmp,
                    months=6,
                    token=None,
                )
            self.assertEqual(added, 1)
            # Only the "new" commit should have been processed.
            self.assertEqual(pr.call_count, 1)
            self.assertEqual(pr.call_args[0][0]["id"], 2)
            with open(csv_path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["commit"] for r in rows], ["old", "new"])

    def test_does_not_persist_commits_without_libraries(self):
        run = {
            "id": 3,
            "status": "completed",
            "conclusion": "cancelled",
            "created_at": "2024-05-03T10:00:00Z",
            "head_sha": "empty",
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "onnx-light", "so_sizes.csv")
            with mock.patch.object(
                rss, "iter_workflow_runs", return_value=iter([run])
            ), mock.patch.object(rss, "process_run", return_value=[]):
                added = rss.process_repo(
                    "xadupre/onnx-light",
                    "build_release.yml",
                    tmp,
                    months=6,
                    token=None,
                )
            self.assertEqual(added, 0)
            # A header-only file is created and no data row is written, so the
            # commit can be retried on a future run once artifacts appear.
            self.assertEqual(rss.read_existing_commits(csv_path), set())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
