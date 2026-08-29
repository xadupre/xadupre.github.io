"""Tests for the onnx-light-cpu backend coverage dashboard."""

from __future__ import annotations

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(
    REPO_ROOT, "dashboard", "onnx-light-cpu", "backend-end-coverage.html"
)
INDEX = os.path.join(REPO_ROOT, "index.html")
WORKFLOW = os.path.join(
    REPO_ROOT, ".github", "workflows", "record_onnx_backend_test_coverage.yml"
)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class TestOnnxLightCpuBackendEndCoverageDashboard(unittest.TestCase):
    def test_page_uses_complete_backend_coverage_data(self):
        text = _read(PAGE)
        self.assertIn(
            'const JSON_URL = '
            '"../../cache_data/onnx-light/backend_test_coverage.json";',
            text,
        )
        for field in (
            "onnx_light_cpu_elapsed_s",
            "payload.totals",
            '"tests"',
            '"passed"',
            '"failed"',
            '"pass ratio"',
        ):
            self.assertIn(field, text)

    def test_page_documents_cpu_kernel_registration(self):
        text = _read(PAGE)
        self.assertIn("onnx_light_cpu.register_kernels()", text)
        self.assertIn("onnx-light-cpu", text)

    def test_page_filters_cpu_results_by_status(self):
        text = _read(PAGE)
        self.assertIn('id="statusFilter"', text)
        self.assertIn('<option value="">all tests</option>', text)
        self.assertIn('<option value="failing">failing tests</option>', text)
        self.assertIn('<option value="succeeding">succeeding tests</option>', text)
        self.assertIn("state.statusFilter", text)

    def test_page_can_hide_succeeded_results(self):
        text = _read(PAGE)
        self.assertIn('id="hideSucceeded"', text)
        self.assertIn("hide succeeded tests", text)
        self.assertIn("hideSucceeded: false", text)
        self.assertIn("state.hideSucceeded && row.onnx_light_cpu", text)

    def test_page_shortens_missing_kernel_error(self):
        text = _read(PAGE)
        self.assertIn('return "no kernel";', text)

    def test_page_has_last_updated_footer(self):
        text = _read(PAGE)
        self.assertIn(
            'data-source="../../cache_data/onnx-light/backend_test_coverage.json"',
            text,
        )
        self.assertIn('<script src="../../assets/last-updated.js">', text)

    def test_index_links_page(self):
        self.assertIn(
            'href="dashboard/onnx-light-cpu/backend-end-coverage.html"', _read(INDEX)
        )

    def test_recording_workflow_links_page_and_data(self):
        text = _read(WORKFLOW)
        self.assertIn("name: DATA onnx-light backend test coverage", text)
        self.assertIn("dashboard/onnx-light-cpu/backend-end-coverage.html", text)
        self.assertIn("cache_data/onnx-light/backend_test_coverage.json", text)
        self.assertIn("ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON", text)


if __name__ == "__main__":
    unittest.main()
