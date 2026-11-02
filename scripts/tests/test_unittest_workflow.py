from pathlib import Path
import unittest

import yaml


class TestUnittestWorkflow(unittest.TestCase):
    def test_builder_checkout_uses_explicit_ref(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load(
            (root / ".github/workflows/unittest.yml").read_text(encoding="utf-8")
        )
        steps = workflow["jobs"]["unittest"]["steps"]
        checkout = next(
            step for step in steps
            if step.get("with", {}).get("repository") == "xadupre/yet-another-onnx-builder"
        )
        self.assertEqual(checkout["with"].get("ref"), "main")

    def test_release_resolution_precedes_builder_install(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load(
            (root / ".github/workflows/unittest.yml").read_text(encoding="utf-8")
        )
        steps = workflow["jobs"]["unittest"]["steps"]
        resolve_index = next(
            i for i, step in enumerate(steps)
            if step["name"] == "Resolve onnx-light release dependencies"
        )
        install_index = next(
            i for i, step in enumerate(steps)
            if step["name"] == "Install test dependencies"
        )
        self.assertLess(resolve_index, install_index)
        self.assertIn(
            '--project "$GITHUB_WORKSPACE/yet-another-onnx-builder[torch,transformers,onnxscript]"',
            steps[resolve_index]["run"],
        )
        step = next(step for step in steps if step["name"] == "Install test dependencies")
        commands = step["run"].splitlines()
        builder = 'pip install -e "./yet-another-onnx-builder[torch,transformers,onnxscript]"'
        self.assertIn(builder, commands)
        self.assertNotIn("releases/download", step["run"])


if __name__ == "__main__":
    unittest.main()
