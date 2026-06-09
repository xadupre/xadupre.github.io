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

    def test_process_run_skips_unfinished_or_failed(self):
        for run in (
            {"id": 1, "status": "in_progress", "conclusion": None},
            {"id": 2, "status": "completed", "conclusion": "failure"},
            {"id": 3, "status": "completed", "conclusion": "cancelled"},
        ):
            with mock.patch.object(rws, "list_run_artifacts") as m:
                self.assertEqual(rws.process_run(run, "o/r", None), [])
                m.assert_not_called()

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
            rows = rws.process_run(run, "o/r", None)
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
            self.assertEqual(rws.process_run(run, "o/r", None), [])

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
                side_effect=lambda run, repo, token: [
                    {
                        "date": run["created_at"],
                        "commit": run["head_sha"],
                        "run_id": str(run["id"]),
                        "size": "42",
                        "name": "new.whl",
                    }
                ],
            ) as pr:
                added = rws.process_repo(
                    "xadupre/onnx-light",
                    "build_release.yml",
                    tmp,
                    months=6,
                    token=None,
                )
            self.assertEqual(added, 1)
            # Only run 2 (the new one) should have been processed.
            self.assertEqual(pr.call_count, 1)
            self.assertEqual(pr.call_args[0][0]["id"], 2)
            with open(csv_path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["run_id"] for r in rows], ["1", "2"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
