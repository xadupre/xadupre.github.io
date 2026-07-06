import os
import unittest


HERE = os.path.dirname(__file__)


class TestSchemaComparisonDashboard(unittest.TestCase):
    def test_dashboard_supports_expanded_backend_tests_tab(self):
        root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(root, "dashboard", "onnx-light", "schema-comparison.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn('id="reportTabs"', content)
        self.assertIn("collectBackendKeys(rows, payload.totals || {});", content)
        self.assertIn('["expanded", "expanded tests"]', content)
        self.assertIn('"backend tests (expanded)"', content)
        self.assertIn("const prefix = `${side}_`;", content)
        self.assertIn("key.startsWith(prefix)", content)
        self.assertIn("state.backendKeys.onnx.main ||", content)
        self.assertIn("state.backendKeys.onnx_light.main ||", content)
        self.assertIn('const show = state.reportMode === "expanded";', content)
        self.assertIn('const showExpanded = state.reportMode === "expanded";', content)
        self.assertNotIn(
            'state.hasExpandedBackend && state.reportMode === "expanded"',
            content,
        )

    def test_get_total_backend_value_returns_null_when_no_key(self):
        # When a side has no expanded key registered (e.g. onnx_light before
        # the expanded column is populated), getTotalBackendValue must return
        # null rather than 0.  The renderTotals function renders null as an
        # empty cell, which avoids showing a misleading "0" in the totals
        # table.
        root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(root, "dashboard", "onnx-light", "schema-comparison.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        # The function must return null immediately when no key is found
        self.assertIn("if (!key) {", content)
        self.assertIn("return null;", content)


if __name__ == "__main__":
    unittest.main()
