import os
import unittest
import yaml

HERE = os.path.dirname(__file__)


class TestBuildMbextDocsWorkflow(unittest.TestCase):
    def _read_workflow(self):
        root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(root, ".github", "workflows", "build_mbext_docs.yml")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_workflow_exists(self):
        root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(root, ".github", "workflows", "build_mbext_docs.yml")
        self.assertTrue(os.path.isfile(path))

    def test_workflow_checks_out_mbext(self):
        content = self._read_workflow()
        self.assertIn("repository: xadupre/mbext", content)

    def test_workflow_installs_doc_dependencies(self):
        content = self._read_workflow()
        # The gallery example imports torch/transformers/onnxruntime which live
        # in the ``dev`` extra, and Sphinx/furo which live in the ``doc`` extra.
        self.assertIn('pip install -e ".[dev,doc]"', content)

    def test_dependency_resolution_uses_cpu_torch_index(self):
        workflow = yaml.safe_load(self._read_workflow())
        steps = workflow["jobs"]["build-and-publish"]["steps"]
        resolve = next(
            step for step in steps
            if step["name"] == "Resolve onnx-light release dependencies"
        )
        self.assertEqual(
            resolve["env"]["PIP_EXTRA_INDEX_URL"],
            "https://download.pytorch.org/whl/cpu",
        )
        self.assertNotIn("PIP_INDEX_URL", resolve["env"])
        self.assertNotIn("--pre", resolve["run"])

    def test_workflow_builds_and_publishes_docs(self):
        content = self._read_workflow()
        self.assertIn("sphinx-build -b html docs docs/_build/html", content)
        self.assertIn("site/docs/mbext", content)
        # GitHub Pages must not run Jekyll so that ``_static``/``_sources`` are served.
        self.assertIn("touch site/docs/mbext/.nojekyll", content)


if __name__ == "__main__":
    unittest.main()
