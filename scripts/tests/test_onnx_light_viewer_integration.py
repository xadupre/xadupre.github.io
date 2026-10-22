"""Tests for onnx-light-viewer documentation and GitHub statistics wiring."""

from pathlib import Path
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import record_build_durations
import record_pr_stats


class TestOnnxLightViewerIntegration(unittest.TestCase):
    def test_collectors_track_repository(self):
        for recorder in (record_build_durations, record_pr_stats):
            with self.subTest(recorder=recorder.__name__):
                self.assertIn("xadupre/onnx-light-viewer", recorder.DEFAULT_REPOS)

    def test_homepage_links(self):
        text = (ROOT / "index.html").read_text(encoding="utf-8")
        for target in (
            "https://github.com/xadupre/onnx-light-viewer",
            "docs/onnx-light-viewer/index.html",
            "dashboard/onnx-light-viewer/pr-stats.html",
            "dashboard/onnx-light-viewer/build-durations.html",
        ):
            with self.subTest(target=target):
                self.assertIn(f'href="{target}"', text)
        self.assertIn("build_onnx_light_viewer_docs.yml/badge.svg", text)
        self.assertIn("onnx-light-viewer/main/media/logo.svg", text)

    def test_dashboards_use_project_cache(self):
        for page, source in (
            ("pr-stats", "pr_stats.csv"),
            ("build-durations", "jobs/index.json"),
        ):
            with self.subTest(page=page):
                text = (
                    ROOT / "dashboard" / "onnx-light-viewer" / f"{page}.html"
                ).read_text(encoding="utf-8")
                self.assertIn('const REPO = "onnx-light-viewer";', text)
                self.assertIn(
                    f'data-source="../../cache_data/onnx-light-viewer/{source}"',
                    text,
                )
                self.assertIn('src="../../assets/chart-loader.js"', text)
                self.assertIn('src="../../assets/last-updated.js"', text)
                self.assertIn("r.status === 404", text)
                self.assertNotIn("onnx-light-cpu", text)
                self.assertNotIn("onnx-light-ort", text)
                if page == "pr-stats":
                    self.assertIn(
                        'const CSV_URL = "../../cache_data/onnx-light-viewer/pr_stats.csv";',
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
            (ROOT / ".github/workflows/build_onnx_light_viewer_docs.yml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(workflow["name"], "DOC onnx-light-viewer")
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
        checkout = steps["Checkout onnx-light-viewer"]
        self.assertEqual(checkout["with"]["repository"], "xadupre/onnx-light-viewer")
        self.assertEqual(checkout["with"]["path"], "onnx-light-viewer")
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
            steps["Build documentation"]["working-directory"], "onnx-light-viewer"
        )
        copy = steps["Copy documentation into site/docs/onnx-light-viewer"]["run"]
        self.assertIn(
            "cp -r onnx-light-viewer/build/docs/. site/docs/onnx-light-viewer/", copy
        )
        self.assertIn("touch site/docs/onnx-light-viewer/.nojekyll", copy)
        publish = steps["Commit and push updated documentation"]
        self.assertEqual(publish["working-directory"], "site")
        self.assertIn("git add docs/onnx-light-viewer", publish["run"])
        self.assertIn('git rebase -X ours "origin/${branch}"', publish["run"])


if __name__ == "__main__":
    unittest.main()
