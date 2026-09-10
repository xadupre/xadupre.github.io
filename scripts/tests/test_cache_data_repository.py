import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
HELPER = ROOT / "scripts" / "commit_cache_data.sh"


class TestCacheDataWorkflowWiring(unittest.TestCase):
    def test_cache_data_is_ignored_and_untracked(self):
        ignore_lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("cache_data/", ignore_lines)
        tracked = subprocess.run(
            ["git", "ls-files", "cache_data"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(tracked.stdout, "")

    def test_all_cache_data_workflows_use_separate_checkout_and_helper(self):
        workflows = []
        for path in sorted(WORKFLOWS.glob("*.yml")):
            text = path.read_text(encoding="utf-8")
            if "cache_data" not in text:
                continue
            workflows.append(path.name)
            workflow = yaml.safe_load(text)
            for job in workflow.get("jobs", {}).values():
                steps = job.get("steps", [])
                if not any(
                    "commit_cache_data.sh" in step.get("run", "") for step in steps
                ):
                    continue
                data_checkouts = [
                    step
                    for step in steps
                    if (step.get("with") or {}).get("repository")
                    == "xadupre/cache_data"
                ]
                self.assertEqual(len(data_checkouts), 1, path.name)
                checkout = data_checkouts[0]
                site_checkout = any(
                    step.get("name") == "Checkout xadupre.github.io"
                    and (step.get("with") or {}).get("path") == "site"
                    for step in steps
                )
                data_path = "site/cache_data" if site_checkout else "cache_data"
                self.assertEqual(checkout.get("uses"), "actions/checkout@v6")
                self.assertEqual((checkout.get("with") or {}).get("path"), data_path)
                self.assertEqual((checkout.get("with") or {}).get("ref"), "main")
                self.assertEqual((checkout.get("with") or {}).get("fetch-depth"), 0)
                self.assertEqual(
                    (checkout.get("with") or {}).get("token"),
                    "${{ secrets.BOT_TOKEN }}",
                )

                helper_steps = [
                    step
                    for step in steps
                    if "commit_cache_data.sh" in step.get("run", "")
                ]
                self.assertEqual(len(helper_steps), 1, path.name)
                helper = "site/scripts" if site_checkout else "scripts"
                self.assertIn(
                    f"bash {helper}/commit_cache_data.sh {data_path}",
                    helper_steps[0]["run"],
                )
                self.assertIsNone(helper_steps[0].get("working-directory"))

        self.assertEqual(len(workflows), 26)

    def test_site_documentation_commits_do_not_stage_cache_data(self):
        for name in ("build_onnx_light_docs.yml", "record_size_onnx_light.yml"):
            workflow = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
            steps = next(iter(workflow["jobs"].values()))["steps"]
            data_index = next(
                i
                for i, step in enumerate(steps)
                if step.get("name") == "Commit and push cache data"
            )
            docs_index = next(
                i
                for i, step in enumerate(steps)
                if step.get("name") == "Commit and push updated documentation"
            )
            self.assertLess(data_index, docs_index, name)
            docs_run = steps[docs_index]["run"]
            self.assertIn("git add docs/onnx-light", docs_run)
            self.assertNotIn("cache_data", docs_run)

    def test_no_workflow_contains_a_legacy_cache_commit(self):
        for path in WORKFLOWS.glob("*.yml"):
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"git add [^\n]*cache_data", path.name)


class TestCommitCacheDataHelper(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(
            tempfile.mkdtemp(prefix=".cache-data-test-", dir=ROOT)
        )
        self.addCleanup(shutil.rmtree, self.workspace, True)
        self.remote = self.workspace / "remote.git"
        self._git("init", "--bare", "--initial-branch=main", str(self.remote))
        seed = self.workspace / "seed"
        self._git("clone", str(self.remote), str(seed))
        self._configure_identity(seed)
        (seed / "data.txt").write_text("base\n", encoding="utf-8")
        self._git("-C", str(seed), "add", "data.txt")
        self._git("-C", str(seed), "commit", "-m", "Initial data")
        self._git("-C", str(seed), "push", "origin", "main")

    def _git(self, *args, check=True):
        return subprocess.run(
            ["git", *args],
            check=check,
            capture_output=True,
            text=True,
        )

    def _configure_identity(self, repo):
        self._git("-C", str(repo), "config", "user.name", "Test User")
        self._git("-C", str(repo), "config", "user.email", "test@example.com")

    def _clone_data(self, name="data"):
        repo = self.workspace / name
        self._git("clone", str(self.remote), str(repo))
        self._git(
            "-C",
            str(repo),
            "remote",
            "set-url",
            "origin",
            "https://github.com/xadupre/cache_data",
        )
        self._git(
            "-C",
            str(repo),
            "config",
            f"url.file://{self.remote.resolve()}.insteadOf",
            "https://github.com/xadupre/cache_data",
        )
        return repo

    def _run_helper(self, repo, message="Update test data", extra_env=None):
        env = os.environ.copy()
        env["CACHE_DATA_RETRY_DELAY_SECONDS"] = "0"
        env["GIT_ALLOW_PROTOCOL"] = "file:https"
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(HELPER), str(repo), message],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_rejects_wrong_repository_and_subdirectory(self):
        repo = self._clone_data()
        child = repo / "child"
        child.mkdir()
        result = self._run_helper(child)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkout root", result.stderr)

        self._git(
            "-C",
            str(repo),
            "config",
            "remote.origin.url",
            "https://github.com/xadupre/xadupre.github.io",
        )
        result = self._run_helper(repo)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to commit unexpected repository", result.stderr)

    def test_exits_successfully_when_unchanged(self):
        repo = self._clone_data()
        result = self._run_helper(repo)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No cache data changes to commit.", result.stdout)

    def test_fails_fast_on_permission_error(self):
        repo = self._clone_data()
        (repo / "new-data.txt").write_text("generated\n", encoding="utf-8")
        fake_bin = self.workspace / "fake-bin"
        fake_bin.mkdir()
        attempts = self.workspace / "push-attempts"
        real_git = shutil.which("git")
        self.assertIsNotNone(real_git)
        wrapper = fake_bin / "git"
        wrapper.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "for arg in \"$@\"; do\n"
            "  if [[ $arg == push ]]; then\n"
            f"    echo attempt >> {attempts!s}\n"
            "    echo 'remote: Permission to xadupre/cache_data.git denied to bot.' >&2\n"
            "    echo 'fatal: unable to access: The requested URL returned error: 403' >&2\n"
            "    exit 128\n"
            "  fi\n"
            "done\n"
            f"exec {real_git} \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)

        result = self._run_helper(
            repo,
            extra_env={"PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("permission denied", result.stderr)
        self.assertEqual(attempts.read_text(encoding="utf-8").splitlines(), ["attempt"])

    def test_rebases_on_concurrent_update_and_preserves_generated_data(self):
        repo = self._clone_data()
        (repo / "data.txt").write_text("generated\n", encoding="utf-8")

        hook = repo / ".git" / "hooks" / "pre-push"
        marker = self.workspace / "concurrent-push-complete"
        race = self.workspace / "race"
        hook.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"if [[ ! -e {marker!s} ]]; then\n"
            f"  touch {marker!s}\n"
            f"  git clone {self.remote!s} {race!s}\n"
            f"  git -C {race!s} config user.name 'Concurrent User'\n"
            f"  git -C {race!s} config user.email concurrent@example.com\n"
            f"  printf 'remote\\n' > {race!s}/data.txt\n"
            f"  git -C {race!s} add data.txt\n"
            f"  git -C {race!s} commit -m 'Concurrent update'\n"
            f"  git -C {race!s} push origin main\n"
            "fi\n",
            encoding="utf-8",
        )
        hook.chmod(0o755)

        result = self._run_helper(repo)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rebasing on origin/main", result.stdout)

        published = self.workspace / "published"
        self._git("clone", str(self.remote), str(published))
        self.assertEqual(
            (published / "data.txt").read_text(encoding="utf-8"),
            "generated\n",
        )
        log = self._git("-C", str(published), "log", "--format=%s").stdout
        self.assertIn("Update test data", log)
        self.assertIn("Concurrent update", log)
        author = self._git(
            "-C", str(published), "log", "-1", "--format=%an <%ae>"
        ).stdout.strip()
        self.assertEqual(
            author,
            "github-actions[bot] "
            "<41898282+github-actions[bot]@users.noreply.github.com>",
        )


if __name__ == "__main__":
    unittest.main()
