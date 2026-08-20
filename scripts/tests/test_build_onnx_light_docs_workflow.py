import os
import unittest

HERE = os.path.dirname(__file__)


class TestBuildOnnxLightDocsWorkflow(unittest.TestCase):
    def test_sphinx_build_runs_serially(self):
        root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(root, ".github", "workflows", "build_onnx_light_docs.yml")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        step = content.split("- name: Build documentation", 1)[1].split(
            "\n\n", 1
        )[0]
        self.assertIn("run: python -m sphinx docs dist/html", step)
        self.assertNotIn(" -j ", step)


if __name__ == "__main__":
    unittest.main()
