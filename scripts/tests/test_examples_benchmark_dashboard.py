"""Tests for the onnx-light-cpu examples benchmark dashboard and wiring."""

from __future__ import annotations

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(
    REPO_ROOT, "dashboard", "onnx-light-cpu", "examples-benchmark.html"
)
INDEX = os.path.join(REPO_ROOT, "index.html")
WORKFLOW = os.path.join(
    REPO_ROOT, ".github", "workflows", "record_onnx_light_cpu_examples_benchmark.yml"
)
SCRIPT = os.path.join(
    REPO_ROOT, "scripts", "record_onnx_light_cpu_examples_benchmark.py"
)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class TestExamplesBenchmarkDashboard(unittest.TestCase):
    def test_page_exists(self):
        self.assertTrue(os.path.isfile(PAGE), f"missing page: {PAGE}")

    def test_page_loads_the_expected_json(self):
        text = _read(PAGE)
        self.assertIn(
            'const JSON_URL = "../../cache_data/onnx-light-cpu/examples_benchmark.json";',
            text,
        )

    def test_page_renders_examples_and_speedup(self):
        text = _read(PAGE)
        self.assertIn("function renderExample(example, cats)", text)
        self.assertIn("payload.examples", text)
        self.assertIn("speedup_cpu", text)
        # The three backends are labelled for the table header.
        for backend in ("numpy", "onnx-light-cpu", "onnxruntime"):
            self.assertIn(backend, text)

    def test_operator_tables_are_folded_with_summary_rows(self):
        text = _read(PAGE)
        self.assertIn('document.createElement("details")', text)
        self.assertIn('document.createElement("summary")', text)
        self.assertIn('summaryRow.className = "operator-summary"', text)
        self.assertIn('details.className = "operator-details"', text)
        self.assertNotIn("panel.open = true", text)
        for label in ("input types", "average speed-up", "best speed-up", "worst speed-up"):
            self.assertIn(label, text)

    def test_page_has_footer_pointing_at_cache(self):
        text = _read(PAGE)
        self.assertIn(
            'data-source="../../cache_data/onnx-light-cpu/examples_benchmark.json"',
            text,
        )
        self.assertIn('<script src="../../assets/last-updated.js">', text)

    def test_speedup_near_one_uses_neutral_color(self):
        text = _read(PAGE)
        # Speed-ups in [0.9, 1[ are neither red nor green but a distinct color.
        self.assertIn('if (value >= 0.9 && value < 1) return "neutral";', text)
        self.assertIn(".operator-summary .neutral { color: #bc8cff; }", text)
        self.assertIn(
            "table.benchmark td.neutral { color: #bc8cff; font-weight: bold; }",
            text,
        )
        self.assertIn(".legend .swatch.neutral { background: #bc8cff; }", text)
        self.assertIn('class="swatch neutral"', text)

    def test_page_classifies_operators_into_small_mid_big(self):
        text = _read(PAGE)
        self.assertIn("function classifyExample(example)", text)
        # SMALL/MID/BIG derive from the speed-up on the first/last/middle sizes.
        self.assertIn("cats.small = first !== null && first > 1;", text)
        self.assertIn("cats.big = last !== null && last > 1;", text)
        self.assertIn(
            "cats.mid = minIdx > 0 && minIdx < rows.length - 1;", text
        )

    def test_page_has_category_checkbox_column(self):
        text = _read(PAGE)
        self.assertIn("categories", text)
        self.assertIn('catValue.className = "summary-value category-boxes";', text)
        self.assertIn('[["small", "SMALL"], ["mid", "MID"], ["big", "BIG"]]', text)
        # The per-operator category boxes are read-only indicators.
        self.assertIn("input.disabled = true;", text)

    def test_page_has_filter_tool(self):
        text = _read(PAGE)
        self.assertIn('id="filterPanel"', text)
        self.assertIn("function setupFilter(rendered)", text)
        for value in ("small", "mid", "big"):
            self.assertIn(
                f'<input type="checkbox" class="filter-check" value="{value}" />',
                text,
            )
        self.assertIn('id="filterCount"', text)
        self.assertIn("operators shown", text)


class TestIndexWiring(unittest.TestCase):
    def test_index_links_dashboard(self):
        text = _read(INDEX)
        self.assertIn(
            'href="dashboard/onnx-light-cpu/examples-benchmark.html"', text
        )
        # The doc-link label / word must match (checked generically elsewhere).
        self.assertIn('data-word="BENCH"', text)

    def test_index_has_workflow_badge(self):
        text = _read(INDEX)
        self.assertIn("record_onnx_light_cpu_examples_benchmark.yml", text)


class TestWorkflow(unittest.TestCase):
    def test_workflow_exists(self):
        self.assertTrue(os.path.isfile(WORKFLOW), f"missing workflow: {WORKFLOW}")

    def test_workflow_builds_from_source_and_runs_script(self):
        text = _read(WORKFLOW)
        self.assertIn("name: DATA onnx-light-cpu benchmark", text)
        # Both dependencies are built from source, as required by the issue.
        self.assertIn("repository: xadupre/onnx-light", text)
        self.assertIn("repository: xadupre/onnx-light-cpu", text)
        self.assertIn("ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON", text)
        self.assertIn(
            "python -u scripts/record_onnx_light_cpu_examples_benchmark.py", text
        )
        self.assertIn(
            "cache_data/onnx-light-cpu/examples_benchmark.json", text
        )


class TestScriptCLI(unittest.TestCase):
    def test_script_is_executable_module(self):
        # A trivial import-time smoke test to make sure the module has no syntax
        # errors and exposes its public entry points.
        import importlib

        module = importlib.import_module("record_onnx_light_cpu_examples_benchmark")
        for name in ("build_payload", "write_payload", "main", "parse_args"):
            self.assertTrue(hasattr(module, name), f"missing {name}")


if __name__ == "__main__":
    unittest.main()
