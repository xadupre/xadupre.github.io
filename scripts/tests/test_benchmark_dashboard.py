"""Tests for the onnx-light benchmark dashboard speed-up colouring."""

from __future__ import annotations

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(REPO_ROOT, "dashboard", "onnx-light", "benchmark.html")
METHODOLOGY = os.path.join(REPO_ROOT, "dashboard", "benchmark-methodology.html")
WORKFLOW = os.path.join(
    REPO_ROOT, ".github", "workflows", "record_onnx_light_benchmark.yml"
)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class TestBenchmarkDashboard(unittest.TestCase):
    def test_page_exists(self):
        self.assertTrue(os.path.isfile(PAGE), f"missing page: {PAGE}")

    def test_workflow_builds_benchmark_backends_from_source(self):
        text = _read(WORKFLOW)
        self.assertIn("repository: xadupre/onnx-light", text)
        self.assertIn("cmake -S ./onnx-light -B ./onnx-light/build", text)
        self.assertIn("pip install ./onnx-light", text)
        self.assertIn("repository: xadupre/onnx-light-cpu", text)
        self.assertIn("ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON", text)
        self.assertIn("--no-deps", text)
        self.assertNotIn("pip install onnx-light", text)
        self.assertNotIn("pip install onnx-light-cpu", text)

    def test_page_links_benchmark_methodology(self):
        text = _read(PAGE)
        self.assertIn('href="../benchmark-methodology.html"', text)
        methodology = _read(METHODOLOGY)
        self.assertIn("separate competing runtimes into global phases", methodology)
        self.assertIn("two warm-up repetitions", methodology)
        self.assertIn("ten measured repetitions per logical CPU", methodology)
        self.assertIn("after one second of cumulative execution", methodology)

    def test_page_shows_repeat_time_limit(self):
        text = _read(PAGE)
        self.assertIn('id="maxRepeatTimeLabel"', text)
        self.assertIn("payload.max_repeat_time_s", text)

    def test_input_type_column_present(self):
        text = _read(PAGE)
        # The table exposes a sortable "input type" column bound to the
        # ``input_type`` field recorded per test.
        self.assertIn('data-key="input_type"', text)
        self.assertIn("input type", text)
        # Rows render the recorded input type inside the new cell.
        self.assertIn("r.input_type", text)

    def test_input_type_selector_present(self):
        text = _read(PAGE)
        # A dropdown selector lets the user filter rows by their input type.
        self.assertIn('id="inputTypeFilter"', text)
        self.assertIn("renderInputTypeFilter", text)
        # The selector filters rows through the shared rowMatches predicate.
        self.assertIn("state.inputTypeFilter", text)
        self.assertIn('(row.input_type || "") !== state.inputTypeFilter', text)

    def test_table_fits_without_horizontal_scrolling(self):
        text = _read(PAGE)
        # A fixed layout with explicit column widths keeps the table within
        # the page width so it does not require horizontal scrolling.
        self.assertIn("table-layout: fixed;", text)
        self.assertIn("<colgroup>", text)

    def test_rows_show_the_benchmark_date(self):
        text = _read(PAGE)
        self.assertIn('<th class="center">date</th>', text)
        self.assertIn("state.benchmarkDate = payload.date;", text)
        self.assertIn(
            "dateTd.textContent = formatBenchmarkDate(state.benchmarkDate);", text
        )
        self.assertIn('slice(0, 16) + " UTC"', text)

    def test_weight_and_three_speedup_averages_are_present(self):
        text = _read(PAGE)
        self.assertIn('data-key="cost_n"', text)
        self.assertIn("rowWeight(r)", text)
        self.assertIn("weighted avg speedup (ort / light)", text)
        self.assertIn("sum latency speedup (ort / light)", text)
        self.assertIn("sum(onnxruntime latency) / sum(onnx-light latency)", text)
        self.assertIn('"Attention", "Gemm", "MatMul"', text)

    def test_weight_is_scaled_down_and_floored_to_one(self):
        text = _read(PAGE)
        # Raw symbolic weights are divided by 1_000_000 and floored to a
        # minimum of 1 so that huge (quadratic) costs stay readable.
        self.assertIn("Math.max(Math.floor(weight / 1000000), 1)", text)

    def test_weight_is_capped_at_64(self):
        text = _read(PAGE)
        # Weights are clipped to 64 so quadratic kernels (Gemm, MatMul, ...)
        # do not dwarf every other operator in the weighted average.
        self.assertIn("const MAX_WEIGHT = 64;", text)
        self.assertIn(
            "return Math.min(Math.max(Math.floor(weight / 1000000), 1), MAX_WEIGHT);",
            text,
        )

    def test_size_reshape_and_training_operators_have_unit_weight(self):
        text = _read(PAGE)
        # Size/Reshape (metadata-only) and ai.onnx.training optimizer
        # operators are pinned to a weight of 1 to reduce their importance.
        self.assertIn("UNIT_WEIGHT_OPERATORS", text)
        self.assertIn('"Size", "Reshape"', text)
        self.assertIn('"Adagrad", "Adam", "Momentum", "Gradient"', text)
        self.assertIn(
            "if (operators.some(op => UNIT_WEIGHT_OPERATORS.has(op))) return 1;",
            text,
        )

    def test_speedup_near_one_uses_distinct_color(self):
        text = _read(PAGE)
        # A speed-up in [0.9, 1[ is coloured purple, neither red nor green,
        # and distinct from the orange used for onnx-light times/light-only rows.
        self.assertIn("function speedupCellClass(v)", text)
        self.assertIn('if (v >= 0.9 && v < 1) return "speedup-close";', text)
        self.assertIn(
            "table.benchmark td.speedup-close { color: #bc8cff; font-weight: bold; }",
            text,
        )
        # Summary cards for the average speed-up also use the purple shade.
        self.assertIn(".summary-card.close .card-value { color: #bc8cff; }", text)
        self.assertIn(
            'avgSpeedup >= 1 ? "faster" : (avgSpeedup >= 0.9 ? "close" : "slower")',
            text,
        )
        self.assertIn(
            '"sum(onnxruntime latency) / sum(onnx-light-cpu latency)"', text
        )
        # And the legend documents the new colour.
        self.assertIn('class="swatch close"', text)
        self.assertIn("within 10% (speedup in [0.9, 1[)", text)


if __name__ == "__main__":
    unittest.main()
