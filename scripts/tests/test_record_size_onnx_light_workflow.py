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


if __name__ == "__main__":
    unittest.main()
