"""Tests for arranged-agents documentation and pull request statistics wiring."""

from pathlib import Path
import re
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import record_pr_stats


class TestArrangedAgentsIntegration(unittest.TestCase):
    def test_collector_tracks_repository(self):
        self.assertIn("xadupre/arranged-agents", record_pr_stats.DEFAULT_REPOS)
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/record_pr_stats.yml").read_text(
                encoding="utf-8"
            )
        )
        steps = {step["name"]: step for step in workflow["jobs"]["record"]["steps"]}
        self.assertEqual(
            re.findall(r"--repo\s+(\S+)", steps["Record daily PR activity"]["run"]),
            list(record_pr_stats.DEFAULT_REPOS),
        )

    def test_homepage_links(self):
        text = (ROOT / "index.html").read_text(encoding="utf-8")
        for target in (
            "https://github.com/xadupre/arranged-agents",
            "docs/arranged-agents/index.html",
            "dashboard/arranged-agents/pr-stats.html",
        ):
            with self.subTest(target=target):
                self.assertIn(f'href="{target}"', text)
        self.assertIn("build_arranged_agents_docs.yml/badge.svg", text)
        self.assertIn(
            "Calls the locally installed Copilot CLI and presents its answers "
            "as structured Markdown.",
            text,
        )
        self.assertNotIn("arranged-agents/build-durations.html", text)

    def test_dashboard_uses_project_cache_and_shared_scripts(self):
        text = (
            ROOT / "dashboard/arranged-agents/pr-stats.html"
        ).read_text(encoding="utf-8")
        self.assertIn('const REPO = "arranged-agents";', text)
        self.assertIn(
            'const CSV_URL = "../../cache_data/arranged-agents/pr_stats.csv";',
            text,
        )
        self.assertIn(
            'data-source="../../cache_data/arranged-agents/pr_stats.csv"', text
        )
        self.assertIn('data-pr-activity="xadupre/arranged-agents"', text)
        self.assertIn('href="../../assets/pr-activity.css"', text)
        for script in (
            "chart-loader.js",
            "pr-review-stats.js",
            "pr-activity.js",
            "last-updated.js",
        ):
            with self.subTest(script=script):
                self.assertIn(f'src="../../assets/{script}"', text)
        self.assertIn("r.status === 404", text)
        self.assertIn("rows.length === 0", text)
        self.assertNotIn("onnx-light-viewer", text)

    def test_dashboard_preserves_existing_behavior(self):
        reference = (
            ROOT / "dashboard/onnx-light-viewer/pr-stats.html"
        ).read_text(encoding="utf-8")
        text = (
            ROOT / "dashboard/arranged-agents/pr-stats.html"
        ).read_text(encoding="utf-8")
        self.assertEqual(text, reference.replace("onnx-light-viewer", "arranged-agents"))

    def test_documentation_workflow(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/build_arranged_agents_docs.yml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(workflow["name"], "DOC arranged-agents")
        triggers = workflow.get("on", workflow.get(True))
        self.assertEqual(
            triggers["workflow_dispatch"]["inputs"]["ref"]["default"], "main"
        )
        self.assertEqual(
            triggers["schedule"], [{"cron": "47 9 1-7,15-21 * 1"}]
        )
        self.assertEqual(workflow["permissions"], {"contents": "write"})
        self.assertEqual(workflow["concurrency"]["group"], "${{ github.workflow }}")
        self.assertTrue(workflow["concurrency"]["cancel-in-progress"])
        job = workflow["jobs"]["build-and-publish"]
        self.assertIn("github.repository_owner", job["if"])
        self.assertIn("github.event_name == 'schedule'", job["if"])
        self.assertIn("github.actor == 'github-actions[bot]'", job["if"])
        steps = {step["name"]: step for step in job["steps"]}
        site = steps["Checkout xadupre.github.io"]
        self.assertEqual(site["uses"], "actions/checkout@v7")
        self.assertEqual(site["with"]["path"], "site")
        self.assertEqual(site["with"]["fetch-depth"], 0)
        self.assertEqual(
            site["with"]["token"], "${{ secrets.BOT_TOKEN || secrets.GITHUB_TOKEN }}"
        )
        checkout = steps["Checkout arranged-agents"]
        self.assertEqual(checkout["uses"], "actions/checkout@v7")
        self.assertEqual(checkout["with"]["repository"], "xadupre/arranged-agents")
        self.assertEqual(checkout["with"]["path"], "arranged-agents")
        self.assertEqual(checkout["with"]["ref"], "${{ inputs.ref || 'main' }}")
        self.assertEqual(steps["Set up Python"]["uses"], "actions/setup-python@v7")
        self.assertEqual(steps["Set up Python"]["with"]["python-version"], "3.12")
        for name, command in (
            (
                "Install documentation dependencies",
                "python -m pip install -r docs/requirements.txt",
            ),
            (
                "Build documentation",
                "python -m sphinx -b html -W --keep-going docs build/docs",
            ),
        ):
            with self.subTest(step=name):
                self.assertEqual(steps[name]["run"], command)
                self.assertEqual(steps[name]["working-directory"], "arranged-agents")
        copy = steps["Copy documentation into site/docs/arranged-agents"]["run"]
        for command in (
            "rm -rf site/docs/arranged-agents",
            "mkdir -p site/docs/arranged-agents",
            "cp -r arranged-agents/build/docs/. site/docs/arranged-agents/",
            "touch site/docs/arranged-agents/.nojekyll",
        ):
            self.assertIn(command, copy)
        publish = steps["Commit and push updated documentation"]
        self.assertEqual(publish["working-directory"], "site")
        for command in (
            "git add docs/arranged-agents",
            "git diff --cached --quiet",
            'git -C ../arranged-agents rev-parse HEAD',
            'git commit -m "Update arranged-agents documentation',
            "attempts=5",
            'git push origin "HEAD:${branch}"',
            "error: 403|Permission to .* denied",
            'git fetch origin "${branch}"',
            'git rebase -X ours "origin/${branch}"',
        ):
            self.assertIn(command, publish["run"])


if __name__ == "__main__":
    unittest.main()
