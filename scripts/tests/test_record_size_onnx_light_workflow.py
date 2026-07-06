import os
import unittest


HERE = os.path.dirname(__file__)


class TestRecordSizeOnnxLightWorkflow(unittest.TestCase):
    def test_schema_comparison_keeps_backend_counts_from_compute_schema_comparison(self):
        # The schema-comparison snapshot must preserve per-operator backend
        # test counts returned by ``compute_schema_comparison()``. Replacing
        # them with ``collect_snippets()`` undercounts ONNX tests because
        # many generated node-test directories are paired variants.
        root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(root, ".github", "workflows", "record_size_onnx_light.yml")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("comparison = compute_schema_comparison()", content)
        self.assertNotIn(
            "from onnx.backend.test.case import collect_snippets as _collect_snippets",
            content,
        )
        self.assertNotIn("_row.onnx_backend_tests = len(_snippets[_row.name])", content)

    def test_schema_comparison_optionally_exports_expanded_backend_totals(self):
        root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(root, ".github", "workflows", "record_size_onnx_light.yml")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("onnx_backend_tests_expanded", content)
        self.assertIn("onnx_light_backend_tests_expanded", content)
        self.assertIn("_optional_total(", content)

    def test_schema_comparison_reclassifies_expanded_backend_tests(self):
        root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(root, ".github", "workflows", "record_size_onnx_light.yml")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn('if "_expanded" not in test.name:', content)
        self.assertIn('if "_expanded" not in name:', content)
        self.assertIn('row["onnx_backend_tests_expanded"] = onnx_exp', content)
        self.assertIn('row["onnx_light_backend_tests_expanded"] = light_exp', content)
        self.assertIn('int(row.get("onnx_backend_tests", 0)) - onnx_exp', content)
        self.assertIn('int(row.get("onnx_light_backend_tests", 0)) - light_exp', content)


if __name__ == "__main__":
    unittest.main()
