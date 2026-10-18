import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

HERE = os.path.dirname(__file__)


class TestBuildOnnxLightDocsWorkflow(unittest.TestCase):
    def _steps(self):
        root = Path(HERE).resolve().parents[1]
        path = root / ".github" / "workflows" / "build_onnx_light_docs.yml"
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        return workflow["jobs"]["build-and-publish"]["steps"]

    def test_breathe_supports_directory_qualified_files(self):
        step = next(
            step
            for step in self._steps()
            if step["name"] == "Install onnx-light with doc dependencies"
        )
        self.assertIn('"breathe==4.36.0"', step["run"])

    def test_legacy_simple_span_path(self):
        step = next(
            step
            for step in self._steps()
            if step["name"] == "Fix legacy simple_span documentation path"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "docs/api/cpp/common/simple_span.rst"
            path.parent.mkdir(parents=True)
            path.write_text(
                "simple_span\n===========\n\n"
                ".. doxygenfile:: common/simple_span.h\n"
                "   :project: onnx-light\n",
                encoding="utf-8",
            )
            for _ in range(2):
                subprocess.run(["bash", "-e", "-c", step["run"]], cwd=temporary, check=True)
                self.assertEqual(
                    path.read_text(encoding="utf-8"),
                    "simple_span\n===========\n\n"
                    ".. doxygenfile:: onnx_proto/simple_span.h\n"
                    "   :project: onnx-light\n",
                )

    def test_unresolved_simple_span_is_not_published(self):
        steps = self._steps()
        name = "Check simple_span C++ documentation"
        step = next(step for step in steps if step["name"] == name)
        names = [step["name"] for step in steps]
        self.assertLess(names.index("Build documentation"), names.index(name))
        self.assertLess(
            names.index(name),
            names.index("Copy documentation into site/docs/onnx-light"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = [
                Path(temporary) / f"dist/html/api/cpp/{directory}/simple_span.html"
                for directory in ("common", "onnx_proto")
            ]
            for path in paths:
                path.parent.mkdir(parents=True)
                path.write_text("<h1>simple_span</h1>", encoding="utf-8")

            def check():
                return subprocess.run(
                    ["bash", "-e", "-c", step["run"]],
                    cwd=temporary,
                    capture_output=True,
                    text=True,
                )

            self.assertEqual(check().returncode, 0)
            for path in paths:
                with self.subTest(page=path.parent.name):
                    path.write_text(
                        '<p>doxygenfile: Cannot find file "simple_span.h</p>',
                        encoding="utf-8",
                    )
                    result = check()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Unresolved C++ documentation", result.stderr)
                    path.write_text("", encoding="utf-8")
                    self.assertNotEqual(check().returncode, 0)
                    path.unlink()
                    self.assertNotEqual(check().returncode, 0)
                    path.write_text("<h1>simple_span</h1>", encoding="utf-8")

    def test_sphinx_build_runs_serially(self):
        root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(root, ".github", "workflows", "build_onnx_light_docs.yml")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        step = content.split("- name: Build documentation", 1)[1].split(
            "\n      - name:", 1
        )[0]
        self.assertIn("run: python -m sphinx docs dist/html", step)
        self.assertNotIn(" -j ", step)


if __name__ == "__main__":
    unittest.main()
