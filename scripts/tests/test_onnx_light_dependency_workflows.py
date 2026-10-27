import unittest
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


class TestOnnxLightDependencyWorkflows(unittest.TestCase):
    def test_release_wheels_are_available_to_runtime_and_isolated_builds(self):
        for name in (
            "record_torch_coverage.yml",
            "record_yobx_sklearn_coverage.yml",
            "record_yobx_model_validate.yml",
            "build_yet_another_onnx_builder_docs.yml",
            "build_onnx_light_kernel_images_docs.yml",
        ):
            with self.subTest(workflow=name):
                workflow = yaml.safe_load(
                    (WORKFLOWS / name).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    workflow["env"]["PIP_FIND_LINKS"],
                    "https://github.com/xadupre/onnx-light/releases/expanded_assets/0.2.0",
                )
                for job in workflow["jobs"].values():
                    self.assertNotIn("PIP_FIND_LINKS", job.get("env", {}))
                    for step in job["steps"]:
                        self.assertNotIn("PIP_FIND_LINKS", step.get("env", {}))

    def test_coverage_extensions_use_matching_python_abis(self):
        workflow = yaml.safe_load(
            (WORKFLOWS / "record_onnx_backend_test_coverage.yml").read_text(
                encoding="utf-8"
            )
        )
        steps = {step["name"]: step for step in workflow["jobs"]["record"]["steps"]}
        self.assertIn(
            "-C cmake.define.ONNX_LIGHT_PYTHON_STABLE_ABI=OFF",
            steps["Build and install onnx-light from source"]["run"],
        )
        self.assertIn(
            "-C cmake.define.ONNX_LIGHT_CPU_PYTHON_STABLE_ABI=OFF",
            steps["Build and install onnx-light-cpu"]["run"],
        )
        self.assertIn("Verify onnx-light-cpu backend", steps)


if __name__ == "__main__":
    unittest.main()
