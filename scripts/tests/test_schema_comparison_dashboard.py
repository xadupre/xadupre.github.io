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


if __name__ == "__main__":
    unittest.main()
