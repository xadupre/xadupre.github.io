import os
import unittest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "build_onnx_light_cpu_docs.yml")


class TestBuildOnnxLightCpuDocsWorkflow(unittest.TestCase):
    def test_build_parallelism_is_memory_safe(self):
        with open(WORKFLOW, encoding="utf-8") as fh:
            content = fh.read()
        step = content.split("- name: Configure build parallelism", 1)[1].split(
            "\n      - name:", 1
        )[0]
        self.assertIn('echo "CMAKE_BUILD_PARALLEL_LEVEL=2" >> "$GITHUB_ENV"', step)

    def test_build_uses_one_onnx_light_runtime(self):
        with open(WORKFLOW, encoding="utf-8") as fh:
            content = fh.read()
        onnx_light_step = content.split(
            "- name: Build and install onnx-light from source", 1
        )[1].split("\n      - name:", 1)[0]
        cpu_step = content.split(
            "- name: Build and install onnx-light-cpu with doc dependencies", 1
        )[1].split("\n      - name:", 1)[0]
        self.assertIn("-C wheel.py-api=cp312", onnx_light_step)
        self.assertIn("get_cpp_build_info", onnx_light_step)
        self.assertNotIn("CMAKE_PREFIX_PATH", onnx_light_step)
        self.assertIn(
            "python setup.py build_ext --inplace --onnx-light-source",
            cpu_step,
        )
        self.assertNotIn('-e ".[docs]"', cpu_step)


if __name__ == "__main__":
    unittest.main()
