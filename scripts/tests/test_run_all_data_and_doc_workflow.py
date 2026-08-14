import glob
import os
import unittest

import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")
DISPATCHER = os.path.join(WORKFLOWS, "run_all_data_and_doc.yml")


class TestRunAllDataAndDocWorkflow(unittest.TestCase):
    def _read(self, path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_workflow_exists(self):
        self.assertTrue(os.path.isfile(DISPATCHER))

    def test_dispatch_uses_github_token(self):
        # Dispatching workflows needs ``actions: write`` which the built-in
        # ``GITHUB_TOKEN`` has. ``BOT_TOKEN`` only carries ``contents: write``
        # for git pushes and returns HTTP 403 for the dispatch API, so it must
        # not be used here.
        content = self._read(DISPATCHER)
        self.assertIn("GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}", content)
        self.assertNotIn("secrets.BOT_TOKEN", content)

    def test_dispatch_has_actions_write_permission(self):
        data = yaml.safe_load(self._read(DISPATCHER))
        self.assertEqual(data.get("permissions", {}).get("actions"), "write")

    def test_dispatch_is_owner_guarded(self):
        data = yaml.safe_load(self._read(DISPATCHER))
        guard = data["jobs"]["dispatch"]["if"]
        self.assertIn("github.actor == github.repository_owner", guard)

    def _data_and_doc_workflows(self):
        for path in sorted(glob.glob(os.path.join(WORKFLOWS, "*.yml"))):
            data = yaml.safe_load(self._read(path))
            name = data.get("name", "")
            if name.startswith("DATA ") or name.startswith("DOC "):
                yield path, data

    def test_children_accept_dispatcher_actor(self):
        # Every DATA/DOC workflow the dispatcher can trigger must allow the
        # ``github-actions[bot]`` actor, since dispatching via ``GITHUB_TOKEN``
        # reports that actor. Otherwise the owner-only guard would skip them.
        found = 0
        for path, data in self._data_and_doc_workflows():
            found += 1
            for job_name, job in data.get("jobs", {}).items():
                guard = job.get("if")
                if guard is not None and "github.repository_owner" in guard:
                    self.assertIn(
                        "github.actor == 'github-actions[bot]'",
                        guard,
                        msg=f"{os.path.basename(path)}:{job_name} rejects dispatcher",
                    )
        self.assertGreater(found, 0)


if __name__ == "__main__":
    unittest.main()
