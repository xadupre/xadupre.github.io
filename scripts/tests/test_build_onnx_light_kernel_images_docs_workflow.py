import os
import unittest
import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(
    ROOT, ".github", "workflows", "build_onnx_light_kernel_images_docs.yml"
)


class TestBuildOnnxLightKernelImagesDocsWorkflow(unittest.TestCase):
    def test_installs_resolved_runtime_before_native_build(self):
        with open(WORKFLOW, encoding="utf-8") as fh:
            content = fh.read()
        workflow = yaml.safe_load(content)
        steps = workflow["jobs"]["build-and-publish"]["steps"]
        names = [step["name"] for step in steps]
        resolve = names.index("Resolve onnx-light release dependencies")
        install = names.index("Install resolved onnx-light runtime for the native build")
        build = names.index("Build and install onnx-light-kernel-images with doc dependencies")
        self.assertLess(resolve, install)
        self.assertLess(install, build)
        self.assertIn("--no-index --no-deps", steps[install]["run"])
        self.assertIn('--find-links "$RUNNER_TEMP/onnx-light-wheels" onnx-light', steps[install]["run"])
        self.assertIn('pip install ".[docs]" -v', steps[build]["run"])
        self.assertIn("OnnxLightPython.cmake", steps[build]["run"])
        self.assertNotIn("gh release download", content)
        self.assertNotIn("SDK", steps[build]["run"])


if __name__ == "__main__":
    unittest.main()
