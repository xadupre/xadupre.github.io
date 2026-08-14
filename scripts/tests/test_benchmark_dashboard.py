"""Tests for the onnx-light benchmark dashboard speed-up colouring."""

from __future__ import annotations

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(REPO_ROOT, "dashboard", "onnx-light", "benchmark.html")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class TestBenchmarkDashboard(unittest.TestCase):
    def test_page_exists(self):
        self.assertTrue(os.path.isfile(PAGE), f"missing page: {PAGE}")

    def test_speedup_near_one_uses_distinct_color(self):
        text = _read(PAGE)
        # A speed-up in [0.9, 1[ is coloured amber, neither red nor green.
        self.assertIn("function speedupCellClass(v)", text)
        self.assertIn('if (v >= 0.9 && v < 1) return "speedup-close";', text)
        self.assertIn(
            "table.benchmark td.speedup-close { color: #d29922; font-weight: bold; }",
            text,
        )
        # Summary cards for the average speed-up also use the amber shade.
        self.assertIn(".summary-card.close .card-value { color: #d29922; }", text)
        self.assertIn(
            'avgSpeedup >= 1 ? "faster" : (avgSpeedup >= 0.9 ? "close" : "slower")',
            text,
        )
        # And the legend documents the new colour.
        self.assertIn('class="swatch close"', text)


if __name__ == "__main__":
    unittest.main()
