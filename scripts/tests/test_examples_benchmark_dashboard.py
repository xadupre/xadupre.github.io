"""Tests for the onnx-light-cpu examples benchmark dashboard and wiring."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(REPO_ROOT, "dashboard", "onnx-light-cpu", "cpu-benchmark.html")
METHODOLOGY = os.path.join(REPO_ROOT, "dashboard", "benchmark-methodology.html")
INDEX = os.path.join(REPO_ROOT, "index.html")
WORKFLOW = os.path.join(
    REPO_ROOT, ".github", "workflows", "record_onnx_light_cpu_examples_benchmark.yml"
)
SCRIPT = os.path.join(REPO_ROOT, "scripts", "record_onnx_light_cpu_benchmark.py")
sys.path.insert(0, os.path.dirname(SCRIPT))


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class TestExamplesBenchmarkDashboard(unittest.TestCase):
    def test_page_exists(self):
        self.assertTrue(os.path.isfile(PAGE), f"missing page: {PAGE}")

    def test_page_links_benchmark_methodology(self):
        text = _read(PAGE)
        self.assertIn('href="../benchmark-methodology.html"', text)
        self.assertTrue(os.path.isfile(METHODOLOGY))

    def test_page_loads_the_expected_json(self):
        text = _read(PAGE)
        self.assertIn(
            'const JSON_URL = "../../cache_data/onnx-light-cpu/examples_benchmark.json";',
            text,
        )

    def test_page_renders_examples_and_speedup(self):
        text = _read(PAGE)
        self.assertIn("function renderExample(example, cats, benchmarkDate)", text)
        self.assertIn("payload.examples", text)
        self.assertIn("speedup_cpu", text)
        self.assertIn(
            "sum(onnxruntime latency) / sum(onnx-light-cpu latency)", text
        )
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
        self.assertIn('id="operatorHeader"', text)
        self.assertIn(
            'document.getElementById("operatorHeader").style.display = "grid";',
            text,
        )
        for label in (
            "input types",
            "average speed-up",
            "best speed-up",
            "worst speed-up",
        ):
            self.assertEqual(text.count(f"<span>{label}</span>"), 1)
        self.assertNotIn('cellLabel.className = "summary-label";', text)
        self.assertNotIn('["inputs", String(summary.inputs', text)
        self.assertIn("(example.rows || []).map(firstInputType).filter(Boolean)", text)
        self.assertIn('["input types", inputTypes || "—", ""]', text)

    def test_expanded_table_shows_the_first_input_tensor_shape(self):
        text = _read(PAGE)
        # The summary groups rows by first-input type, while the expanded table
        # shows the first tensor's type and dimensions (e.g. "float32[16x16]").
        self.assertIn("function firstInputTensor(row)", text)
        self.assertIn("function firstInputType(row)", text)
        self.assertIn('typeTh.textContent = "first input tensor";', text)
        self.assertNotIn('sizeTh.textContent = "inputs";', text)
        self.assertIn("? row.inputs", text)
        self.assertIn("? row.input_type", text)
        self.assertIn('return raw.split(",")[0].trim();', text)
        self.assertIn('firstInputTensor(row).split("[")[0].trim()', text)
        self.assertIn("code.textContent = inputTensor;", text)

    def test_expanded_table_shows_test_names(self):
        text = _read(PAGE)
        self.assertIn('nameTh.textContent = "test name";', text)
        self.assertIn("if (row.test_name)", text)
        self.assertIn("code.textContent = row.test_name;", text)

    def test_summary_and_rows_show_the_benchmark_date(self):
        text = _read(PAGE)
        self.assertEqual(text.count("<span>date</span>"), 1)
        self.assertIn('["date", formatBenchmarkDate(benchmarkDate), ""]', text)
        self.assertIn('dateTh.textContent = "date";', text)
        self.assertIn("dateTd.textContent = formatBenchmarkDate(benchmarkDate);", text)
        self.assertIn("renderExample(ex, cats, payload.date)", text)
        self.assertIn("return date.toISOString().slice(0, 10);", text)
        self.assertIn("? formatBenchmarkDate(payload.date)", text)

    def test_operator_type_summary_shows_its_machine(self):
        text = _read(PAGE)
        self.assertIn("<span>machine</span>", text)
        self.assertIn('["machine", example.machine || "not recorded", ""]', text)

    def test_page_omits_observations_without_a_machine(self):
        text = _read(PAGE)
        self.assertIn("payload.examples.filter(example => example.machine)", text)

    def test_expanded_table_puts_speedup_immediately_after_date(self):
        text = _read(PAGE)
        header_date = text.index('dateTh.textContent = "date";')
        header_speedup = text.index('spTh.textContent = "speed-up (cpu)";')
        header_backends = text.index("backends.forEach(b => {", header_speedup)
        self.assertLess(header_date, header_speedup)
        self.assertLess(header_speedup, header_backends)

        row_date = text.index(
            "dateTd.textContent = formatBenchmarkDate(benchmarkDate);"
        )
        row_speedup = text.index(
            "spTd.textContent = fmtSpeedup(row.speedup_cpu);", row_date
        )
        row_backends = text.index("backends.forEach(b => {", row_speedup)
        self.assertLess(row_date, row_speedup)
        self.assertLess(row_speedup, row_backends)

    def test_expanded_tables_are_sortable(self):
        text = _read(PAGE)
        self.assertIn("function makeSortable(table)", text)
        self.assertIn('header.addEventListener("click", sort);', text)
        self.assertIn('header.setAttribute("aria-sort"', text)
        self.assertIn("makeSortable(table);", text)
        self.assertIn('th[aria-sort="ascending"]::after', text)
        self.assertIn('th[aria-sort="descending"]::after', text)

    def test_panels_are_sorted_by_operator(self):
        text = _read(PAGE)
        self.assertIn("const sorted = examples.slice().sort(", text)
        self.assertIn("return opA.localeCompare(opB)", text)
        self.assertIn("sorted.forEach(ex =>", text)

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
        self.assertIn("cats.mid = minIdx > 0 && minIdx < rows.length - 1;", text)

    def test_page_has_category_checkbox_column(self):
        text = _read(PAGE)
        self.assertIn("categories", text)
        self.assertIn('catCell.className = "summary-categories summary-value";', text)
        self.assertIn('catValue.className = "category-boxes";', text)
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

    def test_page_filters_operators_by_input_type(self):
        text = _read(PAGE)
        self.assertIn('id="typeFilter"', text)
        self.assertIn('<option value="">all input types</option>', text)
        self.assertIn(
            "const inputTypes = [...new Set(rendered.flatMap(item => item.inputTypes))]",
            text,
        )
        self.assertIn(
            'const typeMatch = inputType === "" || panelInputTypes.includes(inputType);',
            text,
        )
        self.assertIn('typeSelect.addEventListener("change", apply);', text)

    def test_page_filters_operators_with_slow_tests(self):
        text = _read(PAGE)
        self.assertIn('id="slowFilter"', text)
        self.assertIn("slow tests only (speed-up &lt; 1)", text)
        self.assertIn("row.speedup_cpu < 1", text)
        self.assertIn("const slowMatch = !slowCheck.checked || hasSlow;", text)
        self.assertIn('slowCheck.addEventListener("change", apply);', text)

    def test_operator_rows_are_compact(self):
        text = _read(PAGE)
        self.assertIn("margin-bottom: 0.25em;", text)
        self.assertIn("padding: 0.35em 1em;", text)
        self.assertIn(".operator-summary {\n  cursor: pointer;\n  font-size: 0.72em;", text)


class TestIndexWiring(unittest.TestCase):
    def test_index_links_dashboard(self):
        text = _read(INDEX)
        self.assertIn('href="dashboard/onnx-light-cpu/cpu-benchmark.html"', text)
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
        self.assertIn("timeout-minutes: 240", text)
        # Both dependencies are built from source, as required by the issue.
        self.assertIn("repository: xadupre/onnx-light", text)
        self.assertIn("repository: xadupre/onnx-light-cpu", text)
        self.assertIn("cmake -S ./onnx-light -B ./onnx-light/build", text)
        self.assertIn("pip install ./onnx-light", text)
        self.assertIn("ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON", text)
        self.assertIn("--no-deps", text)
        self.assertNotIn("pip install onnx-light", text)
        self.assertNotIn("pip install onnx-light-cpu", text)
        self.assertIn("python -u scripts/record_onnx_light_cpu_benchmark.py", text)
        self.assertIn('type: choice', text)
        self.assertIn('--type "${{ needs.select-type.outputs.type }}"', text)
        self.assertIn("cache_data/onnx-light-cpu/examples_benchmark.json", text)


class TestScriptCLI(unittest.TestCase):
    def test_script_is_executable_module(self):
        # A trivial import-time smoke test to make sure the module has no syntax
        # errors and exposes its public entry points.
        import importlib

        module = importlib.import_module("record_onnx_light_cpu_benchmark")
        for name in ("build_payload", "write_payload", "main", "parse_args"):
            self.assertTrue(hasattr(module, name), f"missing {name}")

    def test_page_shows_repeat_time_limit(self):
        text = _read(PAGE)
        self.assertIn('id="maxRepeatTimeLabel"', text)
        self.assertIn("payload.max_repeat_time_s", text)


if __name__ == "__main__":
    unittest.main()
