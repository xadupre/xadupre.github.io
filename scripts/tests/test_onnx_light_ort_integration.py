"""Tests for onnx-light-ort documentation and GitHub statistics wiring."""

from pathlib import Path
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import record_build_durations
import record_pr_stats


class TestOnnxLightOrtIntegration(unittest.TestCase):
    def test_collectors_track_repository(self):
        for recorder in (record_build_durations, record_pr_stats):
            with self.subTest(recorder=recorder.__name__):
                self.assertIn("xadupre/onnx-light-ort", recorder.DEFAULT_REPOS)

    def test_homepage_links(self):
        text = (ROOT / "index.html").read_text(encoding="utf-8")
        for target in (
            "https://github.com/xadupre/onnx-light-ort",
            "docs/onnx-light-ort/index.html",
            "dashboard/onnx-light-ort/pr-stats.html",
            "dashboard/onnx-light-ort/build-durations.html",
        ):
            with self.subTest(target=target):
                self.assertIn(f'href="{target}"', text)
        self.assertIn("build_onnx_light_ort_docs.yml/badge.svg", text)
        self.assertIn("main/docs/_static/onnx-light-ort.svg", text)

    def test_dashboards_use_project_cache(self):
        for page, source in (
            ("pr-stats", "pr_stats.csv"),
            ("build-durations", "jobs/index.json"),
        ):
            with self.subTest(page=page):
                text = (
                    ROOT / "dashboard" / "onnx-light-ort" / f"{page}.html"
                ).read_text(encoding="utf-8")
                self.assertIn('const REPO = "onnx-light-ort";', text)
                self.assertIn(
                    f'data-source="../../cache_data/onnx-light-ort/{source}"',
                    text,
                )
                self.assertIn('src="../../assets/chart-loader.js"', text)
                self.assertIn('src="../../assets/last-updated.js"', text)
                self.assertIn("r.status === 404", text)
                self.assertNotIn("onnx-light-cpu", text)
                if page == "pr-stats":
                    self.assertIn(
                        'const CSV_URL = "../../cache_data/onnx-light-ort/pr_stats.csv";',
                        text,
                    )
                else:
                    self.assertIn(
                        'const JOBS_DIR = "../../cache_data/" + REPO + "/jobs/";',
                        text,
                    )
                    self.assertIn('const INDEX_URL = JOBS_DIR + "index.json";', text)

    def test_documentation_workflow(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/build_onnx_light_ort_docs.yml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(workflow["name"], "DOC onnx-light-ort")
        triggers = workflow.get("on", workflow.get(True))
        self.assertEqual(
            triggers["workflow_dispatch"]["inputs"]["ref"]["default"], "main"
        )
        self.assertTrue(triggers["schedule"])
        job = workflow["jobs"]["build-and-publish"]
        self.assertIn("github.repository_owner", job["if"])
        self.assertIn("github.event_name == 'schedule'", job["if"])
        self.assertIn("github.actor == 'github-actions[bot]'", job["if"])
        steps = {step["name"]: step for step in job["steps"]}
        checkout = steps["Checkout onnx-light-ort"]
        self.assertEqual(checkout["with"]["repository"], "xadupre/onnx-light-ort")
        self.assertEqual(checkout["with"]["ref"], "${{ inputs.ref || 'main' }}")
        self.assertEqual(
            steps["Install documentation dependencies"]["run"],
            "python -m pip install -r docs/requirements.txt",
        )
        self.assertEqual(
            steps["Build documentation"]["run"],
            "python -m sphinx -b html -W --keep-going docs build/docs",
        )
        self.assertEqual(
            steps["Build documentation"]["working-directory"], "onnx-light-ort"
        )
        copy = steps["Copy documentation into site/docs/onnx-light-ort"]["run"]
        self.assertIn(
            "cp -r onnx-light-ort/build/docs/. site/docs/onnx-light-ort/", copy
        )
        self.assertIn("touch site/docs/onnx-light-ort/.nojekyll", copy)
        publish = steps["Commit and push updated documentation"]
        self.assertEqual(publish["working-directory"], "site")
        self.assertIn("git add docs/onnx-light-ort", publish["run"])
        self.assertIn('git rebase -X ours "origin/${branch}"', publish["run"])

    def test_documentation_gallery_native_dependencies(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/build_onnx_light_ort_docs.yml").read_text(
                encoding="utf-8"
            )
        )
        steps = workflow["jobs"]["build-and-publish"]["steps"]
        names = [step["name"] for step in steps]
        by_name = {step["name"]: step for step in steps}
        checkout = by_name["Checkout onnx-light"]
        self.assertEqual(checkout["with"]["repository"], "xadupre/onnx-light")
        self.assertEqual(checkout["with"]["ref"], "main")
        self.assertEqual(checkout["with"]["path"], "onnx-light")
        self.assertFalse(checkout["with"]["persist-credentials"])
        build = by_name["Build native libraries for the gallery"]
        self.assertEqual(build["working-directory"], "onnx-light-ort")
        for command in (
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release",
            '-DFETCHCONTENT_SOURCE_DIR_ONNX_LIGHT="$GITHUB_WORKSPACE/onnx-light"',
            "-DONNX_LIGHT_ORT_BUILD_TESTS=OFF",
            "-DONNX_LIGHT_ORT_BUILD_PYTHON=ON",
            'cmake --build build --parallel "$(nproc)"',
        ):
            with self.subTest(command=command):
                self.assertIn(command, build["run"])
        self.assertLess(
            names.index("Checkout onnx-light"),
            names.index("Build native libraries for the gallery"),
        )
        self.assertLess(
            names.index("Install documentation dependencies"),
            names.index("Build native libraries for the gallery"),
        )
        self.assertLess(
            names.index("Build native libraries for the gallery"),
            names.index("Build documentation"),
        )
        env = by_name["Build documentation"]["env"]
        build_dir = "${{ github.workspace }}/onnx-light-ort/build"
        self.assertEqual(env["PYTHONPATH"], f"{build_dir}/python")
        library_dir = f"{build_dir}/onnx_light_ort"
        for variable, relative_path in (
            (
                "ONNX_LIGHT_ORT_PLUGIN_LIBRARY",
                "custom_ops/libonnx_light_ort_custom_ops.so",
            ),
            (
                "ONNX_LIGHT_ORT_HOST_LIBRARY",
                "bridge_ops_ort_called_by_light/"
                "libonnx_light_ort_bridge_ops_ort_called_by_light.so",
            ),
            (
                "ONNX_LIGHT_ORT_INCOMING_CUSTOM_OP_LIBRARY",
                "bridge_ops_ort_called_by_light/"
                "libonnx_light_ort_ort_custom_ops.so",
            ),
        ):
            with self.subTest(variable=variable):
                self.assertEqual(env[variable], f"{library_dir}/{relative_path}")


if __name__ == "__main__":
    unittest.main()
