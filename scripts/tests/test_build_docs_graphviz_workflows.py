import os
import unittest

HERE = os.path.dirname(__file__)
WORKFLOWS = os.path.join(os.path.dirname(os.path.dirname(HERE)), ".github", "workflows")


class TestBuildDocsGraphvizWorkflows(unittest.TestCase):
    def _workflows(self):
        return sorted(
            os.path.join(WORKFLOWS, name)
            for name in os.listdir(WORKFLOWS)
            if name.startswith("build_") and name.endswith("_docs.yml")
        )

    def test_no_setup_graphviz_action(self):
        # ``ts-graphviz/setup-graphviz`` runs ``apt-get update`` without any
        # timeout, which made the documentation workflows hang for hours.
        for path in self._workflows():
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            self.assertNotIn("uses: ts-graphviz/setup-graphviz", content, path)

    def test_graphviz_installed_with_apt(self):
        for path in self._workflows():
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            if "Install graphviz" not in content:
                continue
            self.assertIn("sudo apt-get install -y graphviz", content, path)
            self.assertIn("timeout-minutes: 10", content, path)


if __name__ == "__main__":
    unittest.main()
