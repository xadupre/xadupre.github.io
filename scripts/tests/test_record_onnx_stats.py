"""Tests for ``scripts.record_onnx_stats``."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_stats as ros  # noqa: E402

try:
    import onnx  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class TestRecordOnnxStats(unittest.TestCase):
    def test_python_version_from_filename(self):
        self.assertEqual(
            ros._python_version_from_filename(
                "onnx-1.21.0-cp312-abi3-manylinux_2_27_x86_64.whl"
            ),
            (3, 12),
        )
        self.assertEqual(
            ros._python_version_from_filename(
                "onnx-1.21.0-cp39-cp39-manylinux_2_17_x86_64.whl"
            ),
            (3, 9),
        )
        self.assertIsNone(ros._python_version_from_filename("onnx-1.21.0.tar.gz"))

    def test_pick_latest_linux_wheel(self):
        files = [
            {
                "packagetype": "sdist",
                "filename": "onnx-1.21.0.tar.gz",
                "size": 1,
            },
            {
                "packagetype": "bdist_wheel",
                "filename": "onnx-1.21.0-cp310-cp310-manylinux_2_27_x86_64.whl",
                "size": 100,
            },
            {
                "packagetype": "bdist_wheel",
                "filename": "onnx-1.21.0-cp312-abi3-manylinux_2_27_x86_64.whl",
                "size": 200,
            },
            {
                "packagetype": "bdist_wheel",
                "filename": "onnx-1.21.0-cp314-cp314t-manylinux_2_27_x86_64.whl",
                "size": 300,
            },
            {
                "packagetype": "bdist_wheel",
                "filename": "onnx-1.21.0-cp312-cp312-win_amd64.whl",
                "size": 999,
            },
        ]
        wheel = ros.pick_latest_linux_wheel(files)
        self.assertIsNotNone(wheel)
        self.assertEqual(wheel["size"], 300)
        self.assertIn("cp314", wheel["filename"])

    def test_pick_latest_linux_wheel_no_match(self):
        files = [
            {
                "packagetype": "bdist_wheel",
                "filename": "onnx-1.21.0-cp312-cp312-win_amd64.whl",
                "size": 1,
            },
        ]
        self.assertIsNone(ros.pick_latest_linux_wheel(files))

    def test_latest_release_files_prefers_urls(self):
        # ``urls`` holds the files of the version reported in ``info.version``
        # and must be used in preference to the ``releases`` mapping.
        metadata = {
            "info": {"version": "1.22.0"},
            "urls": [{"filename": "from-urls.whl"}],
            "releases": {"1.22.0": [{"filename": "from-releases.whl"}]},
        }
        files = ros.latest_release_files(metadata)
        self.assertEqual(files, [{"filename": "from-urls.whl"}])

    def test_latest_release_files_falls_back_to_releases(self):
        # When ``urls`` is missing or empty (e.g. an older cached payload or a
        # normalised ``info.version`` mismatch), fall back to the ``releases``
        # entry for the reported version.
        metadata = {
            "info": {"version": "1.22.0"},
            "urls": [],
            "releases": {"1.22.0": [{"filename": "from-releases.whl"}]},
        }
        files = ros.latest_release_files(metadata)
        self.assertEqual(files, [{"filename": "from-releases.whl"}])

    def test_latest_release_files_empty_when_nothing_available(self):
        metadata = {"info": {"version": "1.22.0"}}
        self.assertEqual(ros.latest_release_files(metadata), [])

    def test_append_row_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "onnx", "stats.csv")
            row = {field: f"v-{field}" for field in ros.CSV_FIELDS}
            ros.append_row(csv_path, row)
            ros.append_row(csv_path, row)
            with open(csv_path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["filename"], "v-filename")

    @unittest.skipUnless(HAS_ONNX, "requires the onnx package")
    def test_count_supported_types_excludes_undefined(self):
        # The actual count depends on the installed onnx version, but it must
        # be at least the number of types present in onnx 1.0 (UNDEFINED +
        # 16 known types -> at least 16 after excluding UNDEFINED).
        n = ros.count_supported_types()
        self.assertGreaterEqual(n, 16)

    @unittest.skipUnless(HAS_ONNX, "requires the onnx package")
    def test_count_node_test_cases_positive(self):
        # The installed onnx package ships hundreds of node test cases.
        self.assertGreater(ros.count_node_test_cases(), 0)

    def test_count_node_test_cases_uses_onnx_light_catalog(self):
        # ``onnx-weekly`` no longer bundles ``onnx/backend/test/data/node``, so
        # the count is taken from the onnx-light backend test catalog, keeping
        # only ``kind == "node"`` cases.
        import types

        cases = {
            "test_abs": types.SimpleNamespace(kind="node"),
            "test_add": types.SimpleNamespace(kind="node"),
            "test_cc_shape": types.SimpleNamespace(kind="model"),
            "test_simple": types.SimpleNamespace(kind="simple"),
        }
        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda include_big=False: cases
        parents = [
            ("onnx_light", types.ModuleType("onnx_light")),
            ("onnx_light.onnx_lib", types.ModuleType("onnx_light.onnx_lib")),
            (
                "onnx_light.onnx_lib.backend",
                types.ModuleType("onnx_light.onnx_lib.backend"),
            ),
            (
                "onnx_light.onnx_lib.backend.test",
                types.ModuleType("onnx_light.onnx_lib.backend.test"),
            ),
            ("onnx_light.onnx_lib.backend.test.case", fake_module),
        ]
        saved = {name: sys.modules.get(name) for name, _ in parents}
        try:
            for name, mod in parents:
                sys.modules[name] = mod
            self.assertEqual(ros.count_node_test_cases(), 2)
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod


if __name__ == "__main__":
    unittest.main()
