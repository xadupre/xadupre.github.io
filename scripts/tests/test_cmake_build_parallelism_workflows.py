import os
import unittest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOWS = (
    "record_onnx_backend_test_coverage.yml",
    "build_onnx_light_cpu_docs.yml",
    "record_onnx_light_benchmark.yml",
    "record_onnx_light_cpu_examples_benchmark.yml",
    "record_size_onnx_light_cpu.yml",
)


class TestCMakeBuildParallelismWorkflows(unittest.TestCase):
    def test_builds_use_explicit_memory_safe_parallelism(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                path = os.path.join(ROOT, ".github", "workflows", name)
                with open(path, encoding="utf-8") as fh:
                    content = fh.read()
                limit = (
                    'echo "CMAKE_BUILD_PARALLEL_LEVEL=2" >> "$GITHUB_ENV"'
                )
                self.assertIn(limit, content)
                self.assertLess(
                    content.index(limit), content.index("cmake --build")
                )
                builds = [
                    line.strip()
                    for line in content.splitlines()
                    if line.strip().startswith("cmake --build ")
                ]
                self.assertTrue(builds)
                for command in builds:
                    # Bare --parallel overrides the environment limit and
                    # lets Unix Makefiles launch an unlimited number of jobs.
                    self.assertIn(
                        '--parallel "${CMAKE_BUILD_PARALLEL_LEVEL}"', command
                    )


if __name__ == "__main__":
    unittest.main()
