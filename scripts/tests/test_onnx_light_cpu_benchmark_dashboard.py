"""Tests for the onnx-light benchmark using onnx-light-cpu kernels."""

from __future__ import annotations

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(REPO_ROOT, "dashboard", "onnx-light-cpu", "benchmark.html")
INDEX = os.path.join(REPO_ROOT, "index.html")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class TestOnnxLightCpuBenchmarkDashboard(unittest.TestCase):
    def test_page_uses_onnx_light_benchmark_data(self):
        text = _read(PAGE)
        self.assertIn(
            'const JSON_URL = "../../cache_data/onnx-light/benchmark.json";', text
        )
        for field in (
            "onnx_light_cpu_avg_ms",
            "onnx_light_cpu_success",
            "speedup_cpu",
            "avg_speedup_cpu",
            "avg_speedup_weighted_cpu",
            "speedup_sum_latency_cpu",
        ):
            self.assertIn(field, text)

    def test_page_documents_cpu_kernel_registration(self):
        text = _read(PAGE)
        self.assertIn("onnx_light_cpu.register_kernels()", text)
        self.assertIn("onnx-light-cpu", text)

    def test_page_has_last_updated_footer(self):
        text = _read(PAGE)
        self.assertIn(
            'data-source="../../cache_data/onnx-light/benchmark.json"', text
        )
        self.assertIn('<script src="../../assets/last-updated.js">', text)

    def test_index_links_page(self):
        self.assertIn(
            'href="dashboard/onnx-light-cpu/benchmark.html"', _read(INDEX)
        )


if __name__ == "__main__":
    unittest.main()
