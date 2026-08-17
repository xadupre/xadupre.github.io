import os
import unittest

import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(
    ROOT, ".github", "workflows", "record_build_durations_onnx_light_cpu.yml"
)


class TestRecordBuildDurationsOnnxLightCpuWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(WORKFLOW, encoding="utf-8") as fh:
            cls.content = fh.read()
        cls.data = yaml.safe_load(cls.content)

    def test_workflow_exists(self):
        self.assertTrue(os.path.isfile(WORKFLOW))

    def test_is_a_data_workflow(self):
        # The ``Run all DATA and DOC`` dispatcher selects workflows whose name
        # starts with ``DATA `` (or ``DOC ``); keep the prefix so this
        # gathering action is launched alongside the others.
        self.assertTrue(self.data.get("name", "").startswith("DATA "))

    def test_scopes_the_fetch_to_onnx_light_cpu(self):
        # The whole point of this dedicated workflow is to gather onnx-light-cpu
        # data on its own, without competing with the huge ``onnx/onnx`` fetch
        # of the shared ``DATA build durations`` workflow.
        self.assertIn("scripts/record_build_durations.py", self.content)
        self.assertIn("--repo xadupre/onnx-light-cpu", self.content)

    def test_commits_the_cache_data(self):
        self.assertIn("git add cache_data", self.content)

    def test_only_owner_or_schedule_runs(self):
        guard = self.data["jobs"]["record"]["if"]
        self.assertIn("github.event_name == 'schedule'", guard)
        self.assertIn("github.actor == github.repository_owner", guard)

    def test_accepts_dispatcher_actor(self):
        # Runs dispatched via the built-in ``GITHUB_TOKEN`` report the
        # ``github-actions[bot]`` actor; the owner-only guard must accept it so
        # that ``Run all DATA and DOC`` does not silently skip this workflow.
        guard = self.data["jobs"]["record"]["if"]
        self.assertIn("github.actor == 'github-actions[bot]'", guard)


if __name__ == "__main__":
    unittest.main()
