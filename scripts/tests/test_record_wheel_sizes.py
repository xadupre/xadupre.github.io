"""Tests for ``scripts.record_wheel_sizes``."""

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

import record_wheel_sizes as rws  # noqa: E402


def _make_artifact_zip(wheels: dict[str, bytes]) -> bytes:
    """Build an in-memory zip archive containing ``wheels`` as ``.whl`` files.

    The artifact zips produced by ``actions/upload-artifact`` store ``.whl``
    files (which are themselves zips) without further compression, so we
    use ``ZIP_STORED`` here to faithfully reproduce their layout.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in wheels.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _make_wheel(members: dict[str, bytes]) -> bytes:
    """Build an in-memory ``.whl`` (a zip archive) containing ``members``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestRecordWheelSizes(unittest.TestCase):
    def test_extract_wheel_sizes_only_returns_whl_files(self):
        zip_bytes = _make_artifact_zip(
            {
                "onnx_light-0.1-cp312-cp312-linux_x86_64.whl": b"x" * 1024,
                "onnx_light-0.1-cp313-cp313-linux_x86_64.whl": b"y" * 2048,
                "README.txt": b"ignored",
            }
        )
        result = sorted(rws.extract_wheel_sizes(zip_bytes))
        self.assertEqual(
            result,
            [
                ("onnx_light-0.1-cp312-cp312-linux_x86_64.whl", 1024),
                ("onnx_light-0.1-cp313-cp313-linux_x86_64.whl", 2048),
            ],
        )

    def test_extract_wheel_sizes_strips_directory_prefixes(self):
        # ``actions/upload-artifact`` preserves the path passed to ``path:``
        # (e.g. ``./wheelhouse/*.whl`` keeps the ``wheelhouse/`` prefix on
        # some runner versions). The recorder must record the wheel name
        # rather than the path-with-prefix.
        zip_bytes = _make_artifact_zip(
            {"wheelhouse/onnx_light-0.1-cp312-cp312-win_amd64.whl": b"z" * 16}
        )
        self.assertEqual(
            rws.extract_wheel_sizes(zip_bytes),
            [("onnx_light-0.1-cp312-cp312-win_amd64.whl", 16)],
        )

    def test_append_rows_writes_header_then_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "wheel_sizes.csv")
            rows1 = [
                {
                    "date": "2024-05-01T10:00:00Z",
                    "commit": "abc1234",
                    "run_id": "1",
                    "size": "1024",
                    "name": "a.whl",
                }
            ]
            self.assertEqual(rws.append_rows(path, rows1), 1)
            rows2 = [
                {
                    "date": "2024-05-02T10:00:00Z",
                    "commit": "def5678",
                    "run_id": "2",
                    "size": "2048",
                    "name": "b.whl",
                }
            ]
            self.assertEqual(rws.append_rows(path, rows2), 1)
            with open(path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["run_id"] for r in rows], ["1", "2"])
            self.assertEqual([r["name"] for r in rows], ["a.whl", "b.whl"])

    def test_append_rows_creates_header_only_file_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "wheel_sizes.csv")
            self.assertEqual(rws.append_rows(path, []), 0)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                header = fh.readline().strip()
            self.assertEqual(header, ",".join(rws.CSV_FIELDS))

    def test_read_existing_returns_seen_run_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wheel_sizes.csv")
            rws.append_rows(
                path,
                [
                    {
                        "date": "2024-05-01T10:00:00Z",
                        "commit": "c1",
                        "run_id": "10",
                        "size": "1",
                        "name": "a.whl",
                    },
                    {
                        "date": "2024-05-01T10:00:00Z",
                        "commit": "c1",
                        "run_id": "10",
                        "size": "2",
                        "name": "b.whl",
                    },
                    {
                        "date": "2024-05-02T10:00:00Z",
                        "commit": "c2",
                        "run_id": "11",
                        "size": "3",
                        "name": "c.whl",
                    },
                ],
            )
            self.assertEqual(rws.read_existing(path), {"10", "11"})

    def test_read_existing_returns_empty_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                rws.read_existing(os.path.join(tmp, "missing.csv")), set()
            )

    def test_process_run_skips_unfinished(self):
        for run in (
            {"id": 1, "status": "in_progress", "conclusion": None},
            {"id": 2, "status": "queued", "conclusion": None},
        ):
            with mock.patch.object(rws, "list_run_artifacts") as m:
                self.assertEqual(rws.process_run(run, "o/r", None), ([], []))
                m.assert_not_called()

    def test_process_run_collects_rows_from_failed_run_with_artifacts(self):
        # A completed run whose conclusion is not "success" can still have
        # uploaded wheel artifacts before a later step failed; those wheels
        # must be recorded so the dashboard does not silently hide them.
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
        zip_bytes = _make_artifact_zip(
            {"onnx_light-0.1-cp312-cp312-linux_x86_64.whl": b"z" * 64}
        )
        with mock.patch.object(
            rws, "list_run_artifacts", return_value=[artifact]
        ), mock.patch.object(
            rws, "_download", return_value=zip_bytes
        ):
            wheel_rows, so_rows = rws.process_run(run, "o/r", None)
        self.assertEqual(
            wheel_rows,
            [
                {
                    "date": "2024-05-02T10:00:00Z",
                    "commit": "cafef00d",
                    "run_id": "7",
                    "size": "64",
                    "name": "onnx_light-0.1-cp312-cp312-linux_x86_64.whl",
                }
            ],
        )
        # The wheel contains no shared library, so no so rows are produced.
        self.assertEqual(so_rows, [])

    def test_process_run_records_shared_libraries_from_wheels(self):
        # The shared libraries shipped inside the wheel are recorded as so
        # rows in the same single download pass as the wheel sizes.
        run = {
            "id": 8,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2024-05-02T10:00:00Z",
            "head_sha": "cafef00d",
        }
        wheel = _make_wheel(
            {
                "onnx_light/onnx_py/_onnxpykernels.cpython-312.so": b"k" * 256,
                "onnx_light/onnx_py/liblib_onnx_lib.so": b"l" * 512,
                "onnx_light/__init__.py": b"# python",
            }
        )
        artifact = {
            "id": 200,
            "name": "wheels-linux-x86_64",
            "expired": False,
            "archive_download_url": "https://api/artifact/200",
        }
        zip_bytes = _make_artifact_zip(
            {"onnx_light-0.1-cp312-cp312-linux_x86_64.whl": wheel}
        )
        with mock.patch.object(
            rws, "list_run_artifacts", return_value=[artifact]
        ), mock.patch.object(rws, "_download", return_value=zip_bytes):
            wheel_rows, so_rows = rws.process_run(run, "o/r", None)
        self.assertEqual(len(wheel_rows), 1)
        self.assertEqual(
            so_rows,
            [
                {
                    "date": "2024-05-02T10:00:00Z",
                    "commit": "cafef00d",
                    "size": "256",
                    "name": "_onnxpykernels.cpython-312.so",
                },
                {
                    "date": "2024-05-02T10:00:00Z",
                    "commit": "cafef00d",
                    "size": "512",
                    "name": "liblib_onnx_lib.so",
                },
            ],
        )

    def test_process_run_collects_rows_from_each_artifact(self):
        run = {
            "id": 42,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2024-05-01T10:00:00Z",
            "head_sha": "deadbeef",
        }
        artifact_linux = {
            "id": 100,
            "name": "wheels-linux-x86_64",
            "expired": False,
            "archive_download_url": "https://api/artifact/100",
        }
        artifact_win = {
            "id": 101,
            "name": "wheels-windows-AMD64",
            "expired": False,
            "archive_download_url": "https://api/artifact/101",
        }
        artifact_expired = {
            "id": 102,
            "name": "wheels-expired",
            "expired": True,
            "archive_download_url": "https://api/artifact/102",
        }
        zip_linux = _make_artifact_zip(
            {"onnx_light-0.1-cp312-cp312-linux_x86_64.whl": b"x" * 100}
        )
        zip_win = _make_artifact_zip(
            {"onnx_light-0.1-cp312-cp312-win_amd64.whl": b"y" * 200}
        )
        downloads = {
            "https://api/artifact/100": zip_linux,
            "https://api/artifact/101": zip_win,
        }
        with mock.patch.object(
            rws,
            "list_run_artifacts",
            return_value=[artifact_linux, artifact_win, artifact_expired],
        ), mock.patch.object(
            rws, "_download", side_effect=lambda url, token: downloads[url]
        ) as dl:
            rows, _ = rws.process_run(run, "o/r", None)
        # Expired artifact must not have been downloaded.
        self.assertEqual(dl.call_count, 2)
        self.assertEqual(len(rows), 2)
        names = sorted(r["name"] for r in rows)
        self.assertEqual(
            names,
            [
                "onnx_light-0.1-cp312-cp312-linux_x86_64.whl",
                "onnx_light-0.1-cp312-cp312-win_amd64.whl",
            ],
        )
        for r in rows:
            self.assertEqual(r["run_id"], "42")
            self.assertEqual(r["commit"], "deadbeef")
            self.assertEqual(r["date"], "2024-05-01T10:00:00Z")
            self.assertIn(r["size"], {"100", "200"})

    def test_process_run_skips_invalid_zip(self):
        run = {
            "id": 7,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2024-05-01T10:00:00Z",
            "head_sha": "c",
        }
        with mock.patch.object(
            rws,
            "list_run_artifacts",
            return_value=[
                {
                    "id": 1,
                    "name": "bad",
                    "expired": False,
                    "archive_download_url": "https://x",
                }
            ],
        ), mock.patch.object(rws, "_download", return_value=b"not a zip"):
            self.assertEqual(rws.process_run(run, "o/r", None), ([], []))

    def test_process_repo_skips_runs_already_recorded(self):
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
            csv_path = os.path.join(tmp, "onnx-light", "wheel_sizes.csv")
            # Seed the cache so run id 1 is considered already processed.
            rws.append_rows(
                csv_path,
                [
                    {
                        "date": "2024-05-01T10:00:00Z",
                        "commit": "old",
                        "run_id": "1",
                        "size": "10",
                        "name": "old.whl",
                    }
                ],
            )
            with mock.patch.object(
                rws, "iter_workflow_runs", return_value=iter([run_new, run_old])
            ), mock.patch.object(
                rws,
                "process_run",
                side_effect=lambda run, repo, token: (
                    [
                        {
                            "date": run["created_at"],
                            "commit": run["head_sha"],
                            "run_id": str(run["id"]),
                            "size": "42",
                            "name": "new.whl",
                        }
                    ],
                    [],
                ),
            ) as pr:
                added_wheels, added_so = rws.process_repo(
                    "xadupre/onnx-light",
                    "build_release.yml",
                    tmp,
                    months=6,
                    token=None,
                )
            self.assertEqual(added_wheels, 1)
            self.assertEqual(added_so, 0)
            # Both runs are inspected (run 1's commit "old" has no so rows yet),
            # but only the new run id produces a new wheel row.
            self.assertEqual(pr.call_count, 2)
            with open(csv_path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["run_id"] for r in rows], ["1", "2"])


class TestSharedLibrarySizes(unittest.TestCase):
    def test_is_shared_library_recognises_suffixes(self):
        for name in (
            "_onnxpykernels.cpython-312-x86_64-linux-gnu.so",
            "liblib_onnx_lib.so",
            "_module.pyd",
            "libfoo.dylib",
            "liblib_onnx_lib.so.1",
            "liblib_onnx_lib.so.1.2",
        ):
            self.assertTrue(rws._is_shared_library(name), name)

    def test_is_shared_library_rejects_non_libraries(self):
        for name in (
            "onnx_light-0.1-cp312-cp312-linux_x86_64.whl",
            "README.txt",
            "module.py",
            "notes.solution",
        ):
            self.assertFalse(rws._is_shared_library(name), name)

    def test_extract_shared_library_sizes_from_wheel(self):
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
        self.assertEqual(
            rws.extract_shared_library_sizes(artifact),
            [
                ("_onnxpykernels.cpython-312-x86_64-linux-gnu.so", 256),
                ("liblib_onnx_lib.so", 512),
            ],
        )

    def test_extract_shared_library_sizes_stored_directly(self):
        # Some artifacts upload the shared libraries directly rather than
        # wrapped inside a wheel.
        artifact = _make_artifact_zip(
            {
                "build/_onnxpyoptim.cpython-312-x86_64-linux-gnu.so": b"o" * 32,
                "build/notes.txt": b"ignore me",
            }
        )
        self.assertEqual(
            rws.extract_shared_library_sizes(artifact),
            [("_onnxpyoptim.cpython-312-x86_64-linux-gnu.so", 32)],
        )

    def test_extract_shared_library_sizes_keeps_largest_duplicate(self):
        wheel_small = _make_wheel({"lib/_mod.so": b"a" * 10})
        wheel_large = _make_wheel({"lib/_mod.so": b"a" * 99})
        artifact = _make_artifact_zip(
            {
                "onnx_light-0.1-cp312-cp312-linux_x86_64.whl": wheel_small,
                "onnx_light-0.1-cp313-cp313-linux_x86_64.whl": wheel_large,
            }
        )
        self.assertEqual(
            rws.extract_shared_library_sizes(artifact), [("_mod.so", 99)]
        )

    def test_extract_shared_library_sizes_ignores_artifacts_without_libraries(self):
        artifact = _make_artifact_zip({"README.txt": b"nothing here"})
        self.assertEqual(rws.extract_shared_library_sizes(artifact), [])

    def test_read_existing_commits_returns_recorded_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "so_sizes.csv")
            rws.append_rows(
                path,
                [
                    {"date": "d", "commit": "aaa", "size": "1", "name": "x.so"},
                    {"date": "d", "commit": "aaa", "size": "2", "name": "y.so"},
                    {"date": "d", "commit": "bbb", "size": "3", "name": "x.so"},
                ],
                rws.SO_CSV_FIELDS,
            )
            self.assertEqual(rws.read_existing_commits(path), {"aaa", "bbb"})

    def test_read_existing_commits_returns_empty_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                rws.read_existing_commits(os.path.join(tmp, "missing.csv")), set()
            )

    def test_append_rows_with_so_fields_writes_so_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "onnx-light", "so_sizes.csv")
            self.assertEqual(
                rws.append_rows(
                    path,
                    [{"date": "d", "commit": "c", "size": "1", "name": "x.so"}],
                    rws.SO_CSV_FIELDS,
                ),
                1,
            )
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            self.assertTrue(content.startswith("date,commit,size,name"))
            self.assertEqual(content.count("date,commit,size,name"), 1)

    def test_process_repo_backfills_both_series(self):
        run = {
            "id": 5,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2024-05-02T10:00:00Z",
            "head_sha": "feedface",
        }
        wheel = _make_wheel(
            {"onnx_light/onnx_py/_mod.cpython-312.so": b"x" * 100}
        )
        artifact = {
            "id": 100,
            "name": "wheels",
            "expired": False,
            "archive_download_url": "https://api/artifact/100",
        }
        zip_bytes = _make_artifact_zip(
            {"onnx_light-0.1-cp312-cp312-linux_x86_64.whl": wheel}
        )
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                rws, "iter_workflow_runs", return_value=iter([run])
            ), mock.patch.object(
                rws, "list_run_artifacts", return_value=[artifact]
            ), mock.patch.object(rws, "_download", return_value=zip_bytes):
                added_wheels, added_so = rws.process_repo(
                    "xadupre/onnx-light",
                    "build_release.yml",
                    tmp,
                    months=6,
                    token=None,
                )
            self.assertEqual((added_wheels, added_so), (1, 1))
            so_path = os.path.join(tmp, "onnx-light", "so_sizes.csv")
            with open(so_path, encoding="utf-8") as fh:
                so_rows = list(csv.DictReader(fh))
            self.assertEqual(
                so_rows,
                [
                    {
                        "date": "2024-05-02T10:00:00Z",
                        "commit": "feedface",
                        "size": "100",
                        "name": "_mod.cpython-312.so",
                    }
                ],
            )

    def test_process_repo_skips_commit_already_recorded_for_so(self):
        # A commit already snapshotted by the inline workflow step must not be
        # recorded a second time in so_sizes.csv, even when its run is new.
        run = {
            "id": 6,
            "status": "completed",
            "conclusion": "success",
            "created_at": "2024-05-02T10:00:00Z",
            "head_sha": "feedface",
        }
        wheel = _make_wheel(
            {"onnx_light/onnx_py/_mod.cpython-312.so": b"x" * 100}
        )
        artifact = {
            "id": 100,
            "name": "wheels",
            "expired": False,
            "archive_download_url": "https://api/artifact/100",
        }
        zip_bytes = _make_artifact_zip(
            {"onnx_light-0.1-cp312-cp312-linux_x86_64.whl": wheel}
        )
        with tempfile.TemporaryDirectory() as tmp:
            so_path = os.path.join(tmp, "onnx-light", "so_sizes.csv")
            rws.append_rows(
                so_path,
                [
                    {
                        "date": "2024-05-02T10:00:00Z",
                        "commit": "feedface",
                        "size": "100",
                        "name": "_mod.cpython-312.so",
                    }
                ],
                rws.SO_CSV_FIELDS,
            )
            with mock.patch.object(
                rws, "iter_workflow_runs", return_value=iter([run])
            ), mock.patch.object(
                rws, "list_run_artifacts", return_value=[artifact]
            ), mock.patch.object(rws, "_download", return_value=zip_bytes):
                added_wheels, added_so = rws.process_repo(
                    "xadupre/onnx-light",
                    "build_release.yml",
                    tmp,
                    months=6,
                    token=None,
                )
            # The wheel row is new, but the commit's so rows are already present.
            self.assertEqual((added_wheels, added_so), (1, 0))
            with open(so_path, encoding="utf-8") as fh:
                so_rows = list(csv.DictReader(fh))
            self.assertEqual(len(so_rows), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
