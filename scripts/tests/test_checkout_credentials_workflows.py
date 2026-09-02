import os
import unittest

import yaml


HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")


class TestCheckoutCredentialsWorkflows(unittest.TestCase):
    def test_pushing_jobs_use_checkout_v5_credentials(self):
        pushing_jobs = 0
        for name in os.listdir(WORKFLOWS):
            if not name.endswith(".yml"):
                continue
            with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as stream:
                workflow = yaml.safe_load(stream)
            for job in workflow.get("jobs", {}).values():
                steps = job.get("steps", [])
                if not any(
                    "git push origin" in step.get("run", "") for step in steps
                ):
                    continue
                pushing_jobs += 1
                checkout = [
                    step
                    for step in steps
                    if step.get("name") == "Checkout xadupre.github.io"
                ]
                self.assertEqual(len(checkout), 1, name)
                self.assertEqual(checkout[0].get("uses"), "actions/checkout@v5", name)
        self.assertGreater(pushing_jobs, 0)


if __name__ == "__main__":
    unittest.main()
