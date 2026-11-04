import unittest
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"

PROJECTS = {
    "build_locodellm_docs.yml": ["locodellm[docs]"],
    "build_mbext_docs.yml": ["mbext[dev,doc]"],
    "build_onnx_light_kernel_images_docs.yml": ["onnx-light-kernel-images[docs]"],
    "build_yet_another_onnx_builder_docs.yml": [
        "yet-another-onnx-builder[category_encoders,docs,imblearn,lightgbm,op-db,polars,sklearn,sksurv,spox,statsmodels,tensorflow,torch,transformers,xgboost]"
    ],
    "record_onnx_backend_node_coverage.yml": [
        "onnx-light", "yet-another-onnx-builder"
    ],
    "record_torch_coverage.yml": [
        "yet-another-onnx-builder[torch,transformers,onnxscript]"
    ],
    "record_yobx_model_validate.yml": [
        "yet-another-onnx-builder[torch,transformers,onnxscript]"
    ],
    "record_yobx_sklearn_coverage.yml": [
        "yet-another-onnx-builder[category_encoders,imblearn,lightgbm,sklearn,sksurv,statsmodels,xgboost]"
    ],
    "unittest.yml": ["yet-another-onnx-builder[torch,transformers,onnxscript]"],
}


class TestOnnxLightDependencyWorkflows(unittest.TestCase):
    def test_locodellm_uses_python_without_stale_runtime_wheel_pins(self):
        workflow = yaml.safe_load(
            (WORKFLOWS / "build_locodellm_docs.yml").read_text(encoding="utf-8")
        )
        steps = workflow["jobs"]["build-and-publish"]["steps"]
        setup = next(step for step in steps if step["name"] == "Set up Python")
        self.assertEqual(setup["with"]["python-version"], "3.14")

    def test_resolver_precedes_dependency_installs_with_matching_extras(self):
        for name, projects in PROJECTS.items():
            with self.subTest(workflow=name):
                workflow = yaml.safe_load(
                    (WORKFLOWS / name).read_text(encoding="utf-8")
                )
                steps = next(iter(workflow["jobs"].values()))["steps"]
                resolve_index = next(
                    i for i, step in enumerate(steps)
                    if step["name"] == "Resolve onnx-light release dependencies"
                )
                resolve = steps[resolve_index]
                site_checkout = next(
                    step for step in steps
                    if step["name"] == "Checkout xadupre.github.io"
                )
                site_path = site_checkout.get("with", {}).get("path", "")
                prefix = f"{site_path}/" if site_path else ""
                self.assertIn(
                    f'python "$GITHUB_WORKSPACE/{prefix}scripts/'
                    'resolve_onnx_light_dependencies.py"',
                    resolve["run"],
                )
                self.assertEqual(resolve["env"]["GH_TOKEN"], "${{ secrets.GITHUB_TOKEN }}")
                self.assertIn("python -m pip install --upgrade pip packaging", resolve["run"])
                self.assertIn('--wheelhouse "$RUNNER_TEMP/onnx-light-wheels"', resolve["run"])
                self.assertEqual(resolve["run"].count("--project "), len(projects))
                for project in projects:
                    self.assertIn(f'--project "$GITHUB_WORKSPACE/{project}"', resolve["run"])
                cmake_index = next(i for i, step in enumerate(steps) if step["name"] == "Set up CMake")
                self.assertLess(cmake_index, resolve_index)
                for step in steps[:resolve_index]:
                    if step["name"] != "Set up CMake":
                        self.assertNotIn("pip install", step.get("run", ""))
                for project in projects:
                    checkout_index = next(
                        i for i, step in enumerate(steps)
                        if step.get("with", {}).get("path") == project.split("[")[0]
                    )
                    self.assertLess(checkout_index, resolve_index)
                self.assertNotIn("working-directory", resolve)

    def test_all_workflows_avoid_independent_release_selection(self):
        integrated = set()
        for path in WORKFLOWS.glob("*.yml"):
            with self.subTest(workflow=path.name):
                content = path.read_text(encoding="utf-8")
                self.assertNotRegex(
                    content,
                    r"https://github\.com/xadupre/onnx-light[^/\s]*/releases/",
                )
                self.assertNotIn("gh release download", content)
                workflow = yaml.safe_load(content)
                self.assertNotIn("PIP_FIND_LINKS", workflow.get("env", {}))
                for job in workflow["jobs"].values():
                    self.assertNotIn("PIP_FIND_LINKS", job.get("env", {}))
                    for step in job.get("steps", []):
                        self.assertNotIn("PIP_FIND_LINKS", step.get("env", {}))
                        if "resolve_onnx_light_dependencies.py" in step.get("run", ""):
                            integrated.add(path.name)
        self.assertEqual(integrated, set(PROJECTS))

    def test_resolver_exports_runtime_and_isolated_build_constraints(self):
        helper = WORKFLOWS.parents[1] / "scripts" / "resolve_onnx_light_dependencies.py"
        content = helper.read_text(encoding="utf-8")
        for variable in ("GITHUB_ENV", "PIP_FIND_LINKS", "PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT"):
            self.assertIn(variable, content)

    def test_source_integration_workflows_do_not_preinstall_release_runtimes(self):
        for name in (
            "build_onnx_light_cpu_docs.yml",
            "record_size_onnx_light_cpu.yml",
            "record_onnx_backend_test_coverage.yml",
            "record_onnx_light_benchmark.yml",
            "record_onnx_light_cpu_examples_benchmark.yml",
        ):
            with self.subTest(workflow=name):
                content = (WORKFLOWS / name).read_text(encoding="utf-8")
                self.assertNotIn("--find-links", content)
                self.assertNotIn("releases/download", content)
                self.assertIn("onnx-light from source", content)

    def test_docs_without_ecosystem_requirements_remain_docs_only(self):
        for name in ("build_onnx_light_viewer_docs.yml", "build_onnx_light_ort_docs.yml"):
            with self.subTest(workflow=name):
                content = (WORKFLOWS / name).read_text(encoding="utf-8")
                self.assertIn("python -m pip install -r docs/requirements.txt", content)
                self.assertNotIn("resolve_onnx_light_dependencies.py", content)

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
