import json
import os
import subprocess
import sys
import tempfile
import unittest

import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import generate_workflow_manifest as generator


class TestGenerateWorkflowManifest(unittest.TestCase):
    def test_manifest_matches_workflow_files(self):
        manifest = generator.build_manifest()
        paths = {item["path"] for item in manifest}
        expected = {
            f".github/workflows/{name}"
            for name in os.listdir(os.path.join(ROOT, ".github", "workflows"))
            if name.endswith(".yml")
        }
        self.assertEqual(paths, expected)
        self.assertEqual(len(paths), len(manifest))
        self.assertTrue(all(item["name"] for item in manifest))

    def test_crons_match_workflow_definitions(self):
        for item in generator.build_manifest():
            path = os.path.join(ROOT, item["path"])
            with open(path, encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
            triggers = data.get("on", data.get(True, {})) or {}
            schedules = (
                triggers.get("schedule", []) if isinstance(triggers, dict) else []
            )
            self.assertEqual(
                item["crons"], [schedule["cron"] for schedule in schedules]
            )

    def test_cli_writes_current_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "manifest.json")
            subprocess.run(
                [sys.executable, generator.__file__, "--output", output],
                check=True,
                cwd=ROOT,
            )
            with open(output, encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), generator.build_manifest())

    def test_committed_manifest_is_current(self):
        path = os.path.join(ROOT, "assets", "workflow-manifest.json")
        with open(path, encoding="utf-8") as stream:
            self.assertEqual(json.load(stream), generator.build_manifest())


if __name__ == "__main__":
    unittest.main()
