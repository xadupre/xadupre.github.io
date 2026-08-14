import os
import unittest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(
    ROOT, ".github", "workflows", "record_size_onnx_light_cpu.yml"
)


class TestRecordSizeOnnxLightCpuWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(WORKFLOW, encoding="utf-8") as fh:
            cls.content = fh.read()

    def test_builds_with_onnx_light_integration(self):
        # The recorded binaries must reflect the released build, which links
        # the onnx-light C++ integration (mirrors build_onnx_light_cpu_docs.yml).
        self.assertIn(
            "cmake.define.ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON", self.content
        )

    def test_records_shared_library_sizes(self):
        self.assertIn("cache_data/onnx-light-cpu/so_sizes.csv", self.content)
        self.assertIn("onnx_light_cpu/onnx_py", self.content)

    def test_records_wheel_sizes(self):
        self.assertIn("cache_data/onnx-light-cpu/wheel_sizes.csv", self.content)
        self.assertIn("python -m pip wheel .", self.content)

    def test_records_lines_of_code(self):
        self.assertIn("scripts/record_loc.py", self.content)
        self.assertIn("cache_data/onnx-light-cpu/loc.csv", self.content)

    def test_only_owner_or_schedule_runs(self):
        self.assertIn(
            "github.event_name == 'schedule' || "
            "github.actor == github.repository_owner",
            self.content,
        )


if __name__ == "__main__":
    unittest.main()
