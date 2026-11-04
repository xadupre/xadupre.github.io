import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "build_yet_another_onnx_builder_docs.yml"


class TestBuildYobxDocsWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        cls.steps = workflow["jobs"]["build-and-publish"]["steps"]
        cls.prepare = next(
            step for step in cls.steps if step["name"] == "Prepare TensorFlow example"
        )

    def prepare_example(self, source):
        with tempfile.TemporaryDirectory() as temporary:
            example = Path(temporary) / "docs/examples/tensorflow/plot_tensorflow_to_onnx.py"
            example.parent.mkdir(parents=True)
            example.write_text(source, encoding="utf-8")
            for _ in range(2):
                subprocess.run(
                    ["bash", "-e", "-c", self.prepare["run"]],
                    cwd=temporary,
                    check=True,
                )
            return example.read_text(encoding="utf-8")

    def test_only_dense_operator_assertion_changes(self):
        source = (
            'op_types = [n.op_type for n in onx_mlp.graph.node]\n'
            'assert "MatMul" in op_types\n'
            'assert "Relu" in op_types\n'
            'np.testing.assert_allclose(expected_mlp, result_mlp, atol=1e-2)\n'
        )
        prepared = self.prepare_example(source)
        assertion = 'assert {"MatMul", "Gemm"}.intersection(op_types), op_types'
        self.assertEqual(
            prepared, source.replace('assert "MatMul" in op_types', assertion)
        )
        for operators in (["MatMul", "Add", "Relu"], ["Gemm", "Relu"]):
            with self.subTest(operators=operators):
                exec(assertion, {"op_types": operators})
        for operators in ([], ["Add", "Relu"], ["Relu"]):
            with self.subTest(operators=operators), self.assertRaises(AssertionError):
                exec(assertion, {"op_types": operators})

    def test_upstream_fix_is_preserved(self):
        source = 'assert "MatMul" in op_types or "Gemm" in op_types\n'
        self.assertEqual(self.prepare_example(source), source)

    def test_preparation_precedes_both_gallery_builds(self):
        build = next(step for step in self.steps if step["name"] == "Build documentation")
        self.assertLess(self.steps.index(self.prepare), self.steps.index(build))
        self.assertEqual(self.prepare["working-directory"], build["working-directory"])
        self.assertEqual(build["run"].count("sphinx-build"), 2)
        self.assertNotIn("|| true", build["run"])
        self.assertNotIn("only_warn_on_example_error", build["run"])


if __name__ == "__main__":
    unittest.main()
