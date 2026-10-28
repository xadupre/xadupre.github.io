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

    def test_onnx_light_wheel_precedes_builder_install(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load(
            (root / ".github/workflows/unittest.yml").read_text(encoding="utf-8")
        )
        steps = workflow["jobs"]["unittest"]["steps"]
        step = next(step for step in steps if step["name"] == "Install test dependencies")
        commands = step["run"].splitlines()
        wheel = (
            'pip install "https://github.com/xadupre/onnx-light/releases/download/'
            "0.2.1/onnx_light-0.2.1-cp312-cp312-manylinux_2_27_x86_64."
            'manylinux_2_28_x86_64.whl"'
        )
        self.assertIn(wheel, commands)
        builder = 'pip install -e "./yet-another-onnx-builder[torch,transformers,onnxscript]"'
        self.assertLess(commands.index(wheel), commands.index(builder))


if __name__ == "__main__":
    unittest.main()
