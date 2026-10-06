import os
import unittest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(
    ROOT, ".github", "workflows", "build_onnx_light_kernel_images_docs.yml"
)


class TestBuildOnnxLightKernelImagesDocsWorkflow(unittest.TestCase):
    def test_installs_full_wheel_from_release(self):
        with open(WORKFLOW, encoding="utf-8") as fh:
            content = fh.read()
        step = content.split(
            "- name: Install onnx-light from the latest full release wheel", 1
        )[1].split("\n      - name:", 1)[0]
        self.assertIn("gh release download", step)
        self.assertIn("! -name '*reduced*'", step)
        self.assertNotIn("gh run download", step)


if __name__ == "__main__":
    unittest.main()
