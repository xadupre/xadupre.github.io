import os
import unittest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(
    ROOT, ".github", "workflows", "build_onnx_light_cpu_docs.yml"
)


class TestBuildOnnxLightCpuDocsWorkflow(unittest.TestCase):
    def test_python_build_uses_sccache(self):
        with open(WORKFLOW, encoding="utf-8") as fh:
            content = fh.read()
        step = content.split("- name: Build and install onnx-light from source", 1)[
            1
        ].split("\n      - name:", 1)[0]
        self.assertIn(
            'CMAKE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=sccache '
            '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache" \\',
            step,
        )


if __name__ == "__main__":
    unittest.main()
