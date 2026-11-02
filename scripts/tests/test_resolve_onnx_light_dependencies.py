import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

from packaging.requirements import Requirement

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resolve_onnx_light_dependencies as resolver


BACKEND = '''
import pathlib
import subprocess
import sysconfig
import tomllib
import zipfile

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]
    name = project["name"]
    native = pathlib.Path("native.c")
    files = {"fixture.py": "VALUE = 1"}
    if native.exists():
        extension = "_fixture" + sysconfig.get_config_var("EXT_SUFFIX")
        subprocess.run(["cc", "-shared", "-fPIC",
                        "-I" + sysconfig.get_path("include"),
                        str(native), "-o", extension], check=True)
        files[extension] = pathlib.Path(extension).read_bytes()
    filename = name + "-1.0-py3-none-any.whl"
    dist = name + "-1.0.dist-info/"
    files[dist + "METADATA"] = (
        "Metadata-Version: 2.1\\nName: " + name + "\\nVersion: 1.0\\n"
        + "".join("Requires-Dist: " + req + "\\n" for req in project.get("dependencies", []))
    )
    files[dist + "WHEEL"] = "Wheel-Version: 1.0\\nGenerator: fixture\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n"
    files[dist + "RECORD"] = "".join(key + ",,\\n" for key in files)
    with zipfile.ZipFile(pathlib.Path(wheel_directory) / filename, "w") as archive:
        for key, value in files.items():
            archive.writestr(key, value)
    return filename

def get_requires_for_build_wheel(config_settings=None):
    requirements = pathlib.Path("hook-requirements.txt")
    return requirements.read_text().splitlines() if requirements.exists() else []
'''


def wheel(directory, name, version, requirements=()):
    normalized = name.replace("-", "_")
    path = directory / f"{normalized}-{version}-py3-none-any.whl"
    dist = f"{normalized}-{version}.dist-info/"
    files = {
        dist + "METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {req}\n" for req in requirements)
        ),
        dist + "WHEEL": (
            "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
        "release_backend.py": BACKEND,
    }
    files[dist + "RECORD"] = "".join(f"{key},,\n" for key in files)
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in files.items():
            archive.writestr(key, value)
    return path


class TestRequirements(unittest.TestCase):
    def test_release_discovery_paginates_and_filters_platform_and_reduced_wheels(self):
        requests = []

        def release(names, draft=False):
            return {
                "draft": draft,
                "assets": [
                    {"name": name, "browser_download_url": f"https://github.com/wheels/{name}"}
                    for name in names
                ],
            }

        def pages(url):
            requests.append(url)
            if url.endswith("&page=1"):
                return [
                    release([
                        "onnx_light-2.0-py3-none-any.whl",
                        "onnx_light-2.0-0reduced-py3-none-any.whl",
                        "onnx_light-2.0-cp399-cp399-win_amd64.whl",
                        "onnx_light_cpu-2.0-py3-none-any.whl",
                    ]),
                    *[release([], draft=True) for _ in range(99)],
                ]
            return [release(["onnx_light-1.0-py3-none-any.whl"])]

        links = resolver.release_wheels("onnx-light", read_page=pages)
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            [link.rsplit("/", 1)[-1] for link in links],
            ["onnx_light-2.0-py3-none-any.whl", "onnx_light-1.0-py3-none-any.whl"],
        )

    def test_markers_use_target_interpreter_and_selected_extras(self):
        self.assertTrue(resolver.applicable(Requirement('onnx-light; python_version >= "3"'), set()))
        self.assertFalse(resolver.applicable(Requirement('onnx-light; sys_platform == "no-such-os"'), {"docs"}))
        self.assertTrue(resolver.applicable(Requirement('onnx-light; extra == "docs"'), {"docs"}))
        self.assertFalse(resolver.applicable(Requirement('onnx-light; extra == "unused"'), {"docs"}))

    def test_collects_runtime_build_and_recursive_selected_extras(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "pyproject.toml").write_text(
                '[project]\nname="fixture"\nversion="1.0"\n'
                'dependencies=["onnx-light>=1,<3"]\n'
                '[project.optional-dependencies]\n'
                'docs=["fixture[native]", "onnx-light-viewer==2"]\n'
                'native=["onnx-light-cpu==1; python_version >= \'3\'"]\n'
                'unused=["onnx-light-ort==99"]\n'
                '[build-system]\nrequires=["onnx-light==2"]\n',
                encoding="utf-8",
            )
            _, extras, requirements = resolver.project_requirements(f"{path}[docs]")
            self.assertEqual(extras, {"docs", "native"})
            self.assertEqual(len(requirements), 4)
            self.assertTrue(any("build-system.requires" in source for source, _ in requirements))
            self.assertFalse(any("ort" in text for _, text in requirements))

    def test_unknown_selected_extra_preserves_pips_warn_and_ignore_behavior(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "pyproject.toml").write_text(
                '[project]\nname="fixture"\nversion="1.0"\n',
                encoding="utf-8",
            )
            _, extras, requirements = resolver.project_requirements(f"{path}[absent]")
            self.assertEqual(extras, {"absent"})
            self.assertEqual(requirements, [])

    def test_constraints_preserve_direct_url_but_remove_evaluated_markers(self):
        self.assertEqual(
            resolver.constraint('onnx-light[foo] @ https://example.org/a.whl ; python_version >= "3"'),
            "onnx-light @ https://example.org/a.whl",
        )
        self.assertEqual(resolver.constraint("onnx-light>=1,<3"), "onnx-light<3,>=1")

    def test_vcs_download_url_uses_resolved_commit_and_subdirectory(self):
        self.assertEqual(
            resolver.resolved_download_url({
                "url": "https://github.com/xadupre/project.git",
                "vcs_info": {"vcs": "git", "commit_id": "abc123"},
                "subdirectory": "packages/model",
            }),
            "git+https://github.com/xadupre/project.git@abc123#subdirectory=packages/model",
        )

    def test_isolated_build_requirements_preserve_direct_urls_without_pinning_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / "pip-build-env-fixture" / "overlay"
            direct = environment / "onnx_light-1.0.dist-info"
            direct.mkdir(parents=True)
            (direct / "METADATA").write_text("Name: onnx-light\nVersion: 1.0\n")
            (direct / "direct_url.json").write_text(json.dumps({
                "url": "https://github.com/xadupre/project.git",
                "vcs_info": {"vcs": "git", "commit_id": "abc123"},
                "subdirectory": "packages/model",
            }))
            index = environment / "onnx_light_cpu-3.0.dist-info"
            index.mkdir()
            (index / "METADATA").write_text("Name: onnx-light-cpu\nVersion: 3.0\n")
            self.assertEqual(
                resolver.isolated_build_requirements(Path(temporary)),
                [
                    "onnx-light @ git+https://github.com/xadupre/project.git"
                    "@abc123#subdirectory=packages/model",
                    "onnx-light-cpu",
                ],
            )

    def test_ecosystem_self_extras_are_expanded_not_forced_as_release_requirements(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "pyproject.toml").write_text(
                '[project]\nname="onnx-light-kernel-images"\nversion="1.0"\n'
                '[project.optional-dependencies]\n'
                'docs=["onnx-light-kernel-images[native_support]"]\n'
                'native_support=["onnx-light==2.0"]\n',
                encoding="utf-8",
            )
            _, extras, requirements = resolver.project_requirements(f"{path}[docs]")
            self.assertEqual(extras, {"docs", "native-support"})
            self.assertEqual([text for _, text in requirements], ["onnx-light==2.0"])


class TestOfflineResolution(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.project = self.root / "project"
        self.project.mkdir()
        self.house = self.root / "wheelhouse"
        self.environment = self.root / "github-env"
        # A real offline pip subprocess is used, not a mocked resolver.
        self.saved = os.environ.copy()
        self.addCleanup(self.restore_environment)
        for key in ("PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT", "PIP_FIND_LINKS"):
            os.environ.pop(key, None)
        os.environ["PIP_NO_INDEX"] = "1"
        os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        os.environ["GITHUB_ENV"] = str(self.environment)

    def restore_environment(self):
        os.environ.clear()
        os.environ.update(self.saved)

    def project_file(self, runtime, build="onnx-light==2.0"):
        (self.project / "pyproject.toml").write_text(
            '[project]\nname="fixture"\nversion="1.0"\n'
            f"dependencies={json.dumps(runtime)}\n"
            '[build-system]\n'
            f"requires={json.dumps([build])}\n"
            'build-backend="release_backend"\n',
            encoding="utf-8",
        )

    def prepare(self):
        links = [path.as_uri() for path in self.assets.glob("*.whl")]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            resolver.prepare([str(self.project)], self.house, links=links)
        return output.getvalue()

    def exported_env(self):
        env = os.environ.copy()
        env.update(line.split("=", 1) for line in self.environment.read_text().splitlines())
        return env

    def test_range_selects_matching_release_and_transitive_package(self):
        wheel(self.assets, "onnx-light", "1.0")
        wheel(self.assets, "onnx-light", "2.0", ["onnx-light-cpu>=1,<2"])
        wheel(self.assets, "onnx-light", "3.0")
        wheel(self.assets, "onnx-light-cpu", "1.5")
        wheel(self.assets, "onnx-light-cpu", "2.0")
        self.project_file(["onnx-light>=1,<3"])
        output = self.prepare()
        constraints = (self.house / "constraints.txt").read_text()
        self.assertIn("onnx-light==2.0", constraints)
        self.assertIn("onnx-light-cpu==1.5", constraints)
        self.assertIn("build-system.requires", output)
        self.assertIn("Requires-Dist: onnx-light-cpu", output)
        self.assertIn("Selected onnx-light==2.0", output)
        self.assertFalse(list(self.house.glob("*3.0*.whl")))
        env = self.exported_env()
        self.assertEqual(env["PIP_CONSTRAINT"], env["PIP_BUILD_CONSTRAINT"])
        # The discovery HTML must not expose unselected releases afterwards.
        self.assertNotIn("releases.html", env["PIP_FIND_LINKS"])
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
             str(self.project)], env=env, check=True, capture_output=True, text=True,
        )

    def test_direct_wheel_url_and_inactive_runtime_marker(self):
        release = wheel(self.assets, "onnx-light", "2.0")
        requirement = f"onnx-light @ {release.as_uri()}"
        self.project_file([requirement, 'onnx-light==99; python_version < "2"'], requirement)
        self.prepare()
        self.assertTrue(list(self.house.glob("onnx_light-2.0*.whl")))

    def test_previous_runtime_and_build_constraints_are_merged_once(self):
        wheel(self.assets, "onnx-light", "2.0")
        self.project_file(["onnx-light>=1"])
        previous = self.root / "previous-constraints.txt"
        previous.write_text("onnx-light<3", encoding="utf-8")
        os.environ["PIP_CONSTRAINT"] = str(previous)
        os.environ["PIP_BUILD_CONSTRAINT"] = str(previous)
        self.prepare()
        constraints = (self.house / "constraints.txt").read_text()
        self.assertEqual(constraints.count("onnx-light<3"), 1)

    def test_runtime_only_root_with_transitive_source_build_dependency(self):
        wheel(self.assets, "onnx-light", "2.0")
        dependency = self.root / "source_dependency"
        dependency.mkdir()
        (dependency / "pyproject.toml").write_text(
            '[project]\nname="source_dependency"\nversion="1.0"\n'
            '[build-system]\nrequires=["onnx-light==2.0"]\n'
            'build-backend="release_backend"\n',
            encoding="utf-8",
        )
        archive = self.root / "source_dependency-1.0.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(dependency, arcname="source_dependency-1.0")
        (self.project / "release_backend.py").write_text(BACKEND, encoding="utf-8")
        (self.project / "pyproject.toml").write_text(
            '[project]\nname="fixture"\nversion="1.0"\n'
            f'dependencies=["source_dependency @ {archive.as_uri()}"]\n'
            '[build-system]\nrequires=[]\nbuild-backend="release_backend"\n'
            'backend-path=["."]\n',
            encoding="utf-8",
        )
        output = self.prepare()
        self.assertIn("build-system.requires", output)
        self.assertIn("onnx-light==2.0", (self.house / "constraints.txt").read_text())
        self.assertTrue(list(self.house.glob("onnx_light-2.0*.whl")))
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
             "--no-cache-dir", str(self.project)],
            env=self.exported_env(), check=True, capture_output=True, text=True,
        )

    def test_dynamic_backend_hook_dependency_is_included_in_resolution(self):
        wheel(self.assets, "onnx-light", "2.0")
        wheel(self.assets, "onnx-light-cpu", "2.0")
        self.project_file(["onnx-light>=2"])
        (self.project / "hook-requirements.txt").write_text(
            "onnx-light-cpu==2.0\n",
            encoding="utf-8",
        )
        output = self.prepare()
        self.assertIn("onnx-light-cpu==2.0", output)
        self.assertIn("onnx-light-cpu==2.0", (self.house / "constraints.txt").read_text())
        self.assertTrue(list(self.house.glob("onnx_light_cpu-2.0*.whl")))

    def test_dynamic_hook_version_is_resolved_with_runtime_requirements(self):
        wheel(self.assets, "onnx-light", "2.0")
        wheel(self.assets, "onnx-light", "3.0")
        wheel(self.assets, "onnx-light-cpu", "2.0", ["onnx-light==2.0"])
        self.project_file(["onnx-light-cpu==2.0"])
        (self.project / "hook-requirements.txt").write_text(
            "onnx-light>=1\n",
            encoding="utf-8",
        )
        self.prepare()
        constraints = (self.house / "constraints.txt").read_text()
        self.assertIn("onnx-light==2.0", constraints)
        self.assertNotIn("onnx-light==3.0", constraints)

    def test_conflicting_runtime_and_build_requirements_fail_with_source(self):
        wheel(self.assets, "onnx-light", "1.0")
        wheel(self.assets, "onnx-light", "2.0")
        self.project_file(["onnx-light==1.0"])
        with self.assertRaisesRegex(RuntimeError, r"Cannot resolve.*Python") as caught:
            self.prepare()
        self.assertIn("build-system.requires", str(caught.exception))
        self.assertIn("project.dependencies", str(caught.exception))
        self.assertFalse(self.environment.exists())

    def test_unavailable_release_fails_instead_of_using_latest(self):
        wheel(self.assets, "onnx-light", "3.0")
        self.project_file(["onnx-light==2.0"])
        with self.assertRaisesRegex(RuntimeError, "onnx-light==2.0"):
            self.prepare()

    @unittest.skipUnless(shutil.which("cc"), "C compiler required")
    def test_isolated_native_build_uses_prepared_release_without_preinstall(self):
        wheel(self.assets, "onnx-light", "2.0")
        self.project_file(["onnx-light==2.0"])
        (self.project / "native.c").write_text(
            '#include <Python.h>\n'
            'static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_fixture", NULL, -1, NULL};\n'
            'PyMODINIT_FUNC PyInit__fixture(void) { return PyModule_Create(&module); }\n',
            encoding="utf-8",
        )
        self.prepare()
        result = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir",
             str(self.root / "built"), str(self.project)],
            env=self.exported_env(), check=True, capture_output=True, text=True,
        )
        self.assertIn("Successfully built fixture", result.stdout)
        built = next((self.root / "built").glob("fixture*.whl"))
        with zipfile.ZipFile(built) as archive:
            self.assertTrue(any(name.endswith(".so") for name in archive.namelist()))


if __name__ == "__main__":
    unittest.main()
