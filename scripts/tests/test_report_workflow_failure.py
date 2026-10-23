import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "report_workflow_failure.yml"


class TestReportWorkflowFailure(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        self.job = self.workflow["jobs"]["report"]
        self.step = self.job["steps"][0]

    def test_trigger_and_guard(self):
        trigger = self.workflow.get("on", self.workflow.get(True))
        self.assertEqual(
            trigger,
            {"workflow_run": {"workflows": ["*"], "types": ["completed"]}},
        )
        guard = self.job["if"]
        self.assertIn("github.event.workflow_run.conclusion == 'failure'", guard)
        self.assertIn("github.event.workflow_run.conclusion == 'timed_out'", guard)
        self.assertIn(
            "github.event.workflow_run.head_repository.full_name == github.repository",
            guard,
        )
        self.assertIn("startsWith(github.event.workflow_run.name, 'DATA ')", guard)
        self.assertIn("startsWith(github.event.workflow_run.name, 'DOC ')", guard)

    def test_permissions_and_concurrency(self):
        self.assertEqual(self.workflow["permissions"], {"issues": "write"})
        self.assertEqual(
            self.workflow["concurrency"],
            {
                "group": "report-workflow-failure-${{ github.event.workflow_run.workflow_id }}",
                "cancel-in-progress": False,
            },
        )
        self.assertEqual(len(self.job["steps"]), 1)
        self.assertNotIn("uses", self.step)
        self.assertNotIn("${{", self.step["run"])
        self.assertEqual(
            self.step["env"]["GH_TOKEN"], "${{ secrets.GITHUB_TOKEN }}"
        )

    def _run_report(self, issues="", api_failure=False, workflow_name="DOC onnx-light"):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            gh = directory / "gh"
            gh.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['CALLS'], 'a', encoding='utf-8') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "if sys.argv[1] == 'api':\n"
                "    print(os.environ['ISSUES'])\n"
                "    sys.exit(int(os.environ['API_FAILURE']))\n",
                encoding="utf-8",
            )
            gh.chmod(0o755)
            calls = directory / "calls.jsonl"
            environment = {
                **os.environ,
                "PATH": f"{directory}{os.pathsep}{os.environ['PATH']}",
                "CALLS": str(calls),
                "ISSUES": issues,
                "API_FAILURE": str(int(api_failure)),
                "GITHUB_REPOSITORY": "xadupre/xadupre.github.io",
                "GH_TOKEN": "",
                "WORKFLOW_NAME": workflow_name,
                "RUN_URL": "https://github.com/xadupre/xadupre.github.io/actions/runs/123",
                "RUN_BRANCH": "main",
                "RUN_SHA": "abc123",
                "RUN_ATTEMPT": "2",
            }
            result = subprocess.run(
                ["bash", "-c", self.step["run"]],
                env=environment,
                capture_output=True,
                text=True,
            )
            recorded = [
                json.loads(line)
                for line in calls.read_text(encoding="utf-8").splitlines()
            ]
            return result, recorded

    def test_creates_issue_with_run_details(self):
        for name in ("DOC onnx-light", "DATA onnx-light-cpu benchmark"):
            with self.subTest(name=name):
                result, calls = self._run_report(
                    issues="42\tUnrelated issue", workflow_name=name
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--paginate", calls[0])
                self.assertIn("state=open&per_page=100", calls[0][2])
                self.assertEqual(calls[1][:2], ["issue", "create"])
                self.assertEqual(
                    calls[1][calls[1].index("--title") + 1], f"Workflow failed: {name}"
                )
                body = calls[1][calls[1].index("--body") + 1]
                for value in (name, "/actions/runs/123", "main", "abc123", "Attempt: 2"):
                    self.assertIn(value, body)

    def test_comments_on_existing_issue_instead_of_creating_duplicate(self):
        result, calls = self._run_report(
            issues="41\tUnrelated issue\n42\tWorkflow failed: DOC onnx-light"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][:3], ["issue", "comment", "42"])

    def test_api_failure_does_not_create_duplicate(self):
        result, calls = self._run_report(api_failure=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)

    def test_metadata_is_not_executed_by_shell(self):
        result, calls = self._run_report(workflow_name="DOC $(exit 99) `exit 98` 'test'")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("$(exit 99) `exit 98` 'test'", calls[1][-1])


if __name__ == "__main__":
    unittest.main()
