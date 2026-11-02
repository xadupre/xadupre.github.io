from pathlib import Path
import re
import unittest

import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
NATIVE_WORKFLOWS = (
    "build_locodellm_docs.yml",
    "build_mbext_docs.yml",
    "build_onnx_light_cpu_docs.yml",
    "build_onnx_light_docs.yml",
    "build_onnx_light_kernel_images_docs.yml",
    "build_onnx_light_ort_docs.yml",
    "build_yet_another_onnx_builder_docs.yml",
    "build_yet_another_onnxruntime_extensions_docs.yml",
    "record_onnx_backend_node_coverage.yml",
    "record_onnx_backend_test_coverage.yml",
    "record_onnx_constant_info_coverage.yml",
    "record_onnx_inplace_reuse_coverage.yml",
    "record_onnx_light_benchmark.yml",
    "record_onnx_light_cpu_examples_benchmark.yml",
    "record_onnx_release_after_coverage.yml",
    "record_onnx_shape_inference_coverage.yml",
    "record_onnx_shape_tag_coverage.yml",
    "record_onnx_time.yml",
    "record_size_onnx_light.yml",
    "record_size_onnx_light_cpu.yml",
    "record_torch_coverage.yml",
    "record_yobx_model_validate.yml",
    "record_yobx_sklearn_coverage.yml",
    "unittest.yml",
)


class TestCMakeVersionWorkflows(unittest.TestCase):
    def test_native_builds_provision_cmake_before_dependencies(self):
        for name in NATIVE_WORKFLOWS:
            workflow = yaml.safe_load(
                (WORKFLOWS / name).read_text(encoding="utf-8")
            )
            for job_name, job in workflow["jobs"].items():
                if not any(
                    step.get("uses", "").startswith("actions/setup-python@")
                    for step in job["steps"]
                ):
                    continue
                with self.subTest(workflow=name, job=job_name):
                    steps = job["steps"]
                    setup_index = next(
                        i for i, step in enumerate(steps)
                        if step.get("name") == "Set up CMake"
                    )
                    python_index = next(
                        i for i, step in enumerate(steps)
                        if step.get("uses", "").startswith("actions/setup-python@")
                    )
                    self.assertLess(python_index, setup_index)
                    setup = steps[setup_index]
                    self.assertNotIn("if", setup)
                    self.assertNotIn("continue-on-error", setup)
                    self.assertNotIn("working-directory", setup)
                    commands = setup["run"].splitlines()
                    install = commands[0]
                    self.assertIn("python -m pip install --upgrade", install)
                    version = re.search(r'"cmake>=(\d+)\.(\d+)(?:\.(\d+))?"', install)
                    self.assertIsNotNone(version)
                    self.assertGreaterEqual(
                        tuple(int(part or 0) for part in version.groups()),
                        (4, 0, 0),
                    )
                    self.assertIn(
                        'cmake_bin="$(python -c \'import cmake; print(cmake.CMAKE_BIN_DIR)\')"',
                        commands,
                    )
                    self.assertIn('echo "$cmake_bin" >> "$GITHUB_PATH"', commands)
                    self.assertIn('export PATH="$cmake_bin:$PATH"', commands)
                    self.assertEqual(commands[-1], "cmake --version")
                    for step in steps[:setup_index]:
                        run = step.get("run", "")
                        self.assertNotRegex(
                            run,
                            r"\bpip\s+(?:install|wheel)\b|"
                            r"\bcmake\s|setup\.py\s+build|python\s+-m\s+build",
                        )

    def test_explicit_cmake_and_source_builds_are_covered(self):
        for path in sorted(WORKFLOWS.glob("*.yml")):
            workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
            for job_name, job in workflow["jobs"].items():
                for step in job.get("steps", []):
                    run = step.get("run", "")
                    if re.search(
                        r"\bcmake\s+(?:-S|--build|--install)\b|"
                        r"setup\.py\s+build_ext|"
                        r"bash\s+examples/.*build\.sh|"
                        r"pip\s+(?:install|wheel)\s+(?:-e\s+)?[\"']?"
                        r"(?:\./)?onnx-light",
                        run,
                    ):
                        with self.subTest(workflow=path.name, job=job_name):
                            self.assertIn(path.name, NATIVE_WORKFLOWS)
                            self.assertTrue(
                                any(
                                    item.get("name") == "Set up CMake"
                                    for item in job["steps"]
                                )
                            )


if __name__ == "__main__":
    unittest.main()
