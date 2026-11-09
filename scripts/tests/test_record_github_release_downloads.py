"""Exercise release download collection against a real local HTTP API."""

import datetime as dt
import gzip
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import record_github_release_downloads as downloads

ROOT = Path(__file__).resolve().parents[2]


class TestReleaseDownloads(unittest.TestCase):
    def read_project(self, cache_dir, repository):
        directory = Path(cache_dir) / "github-release-downloads" / repository
        data = json.loads((directory / "latest.json").read_text())
        data["history"] = [
            json.loads((directory / f"{date}.json").read_text())
            for date in data["dates"]
        ]
        return data

    def setUp(self):
        self.count = 3
        self.asset_count = 100
        self.fail_assets = False
        self.requests = []
        case = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlsplit(self.path)
                page = int(parse_qs(parsed.query)["page"][0])
                case.requests.append((parsed.path, page))
                if parsed.path == "/users/xadupre/repos":
                    if page == 1:
                        payload = [
                            {
                                "full_name": f"xadupre/project-{i}",
                                "name": f"project-{i}",
                                "private": False,
                            }
                            for i in range(94)
                        ] + [
                            {
                                "full_name": "xadupre/onnx-light-cpu",
                                "name": "onnx-light-cpu",
                                "private": False,
                            },
                            {
                                "full_name": "xadupre/onnx-light-private",
                                "name": "onnx-light-private",
                                "private": True,
                            },
                            {
                                "full_name": "xadupre/onnx-asv-benchmark",
                                "name": "onnx-asv-benchmark",
                                "private": False,
                            },
                            {
                                "full_name": "xadupre/onnx-diagnostic",
                                "name": "onnx-diagnostic",
                                "private": False,
                            },
                            {
                                "full_name": "xadupre/yet-another-tool",
                                "name": "yet-another-tool",
                                "private": False,
                            },
                            {
                                "full_name": "xadupre/onnx-lightish",
                                "name": "onnx-lightish",
                                "private": False,
                            },
                        ]
                    else:
                        payload = [
                            {
                                "full_name": "xadupre/onnx-light-com-microsoft",
                                "name": "onnx-light-com-microsoft",
                                "private": False,
                            },
                            {
                                "full_name": "xadupre/onnx-light-viewer",
                                "name": "onnx-light-viewer",
                                "private": False,
                            },
                            {
                                "full_name": "xadupre/arranged-agents",
                                "name": "arranged-agents",
                                "private": False,
                            },
                        ]
                elif parsed.path.endswith("/releases/1/assets"):
                    if case.fail_assets:
                        self.send_error(503, "asset API unavailable")
                        return
                    payload = (
                        [
                            {
                                "id": n,
                                "name": f"wheel-{n}.whl",
                                "state": "uploaded",
                                "download_count": case.count,
                            }
                            for n in range(100)
                        ]
                        if page == 1
                        else [
                            {
                                "id": 101,
                                "name": "checksums.txt",
                                "state": "uploaded",
                                "download_count": 2,
                            },
                            {
                                "id": 102,
                                "name": "pending.whl",
                                "state": "starter",
                                "download_count": 99,
                            },
                        ]
                    )
                elif (
                    parsed.path.endswith("/releases/2/assets")
                    or parsed.path == "/repos/xadupre/onnx-empty/releases"
                ):
                    payload = []
                elif parsed.path.endswith("/releases"):
                    if page == 1:
                        payload = [
                            {
                                "id": 1,
                                "tag_name": "v1/preview",
                                "draft": False,
                                "prerelease": True,
                                "published_at": "2026-01-01T00:00:00Z",
                                "assets": [
                                    {
                                        "id": n,
                                        "name": f"wheel-{n}.whl",
                                        "state": "uploaded",
                                        "download_count": case.count,
                                    }
                                    for n in range(case.asset_count)
                                ],
                            },
                        ] + [{"id": n, "draft": True} for n in range(3, 102)]
                    else:
                        payload = [
                            {
                                "id": 2,
                                "tag_name": "v0",
                                "draft": False,
                                "prerelease": False,
                                "published_at": "2025-01-01T00:00:00Z",
                                "assets": [],
                            },
                        ]
                else:
                    self.send_error(404)
                    return
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.original_api = downloads.API
        downloads.API = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.cleanup_server)

    def cleanup_server(self):
        downloads.API = self.original_api
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def test_discovery_includes_new_repositories_and_home_upstreams(self):
        repositories = downloads.discover_repositories("")
        self.assertEqual(
            repositories,
            [
                "ggml-org/llama.cpp",
                "microsoft/onnxruntime",
                "microsoft/onnxruntime-extensions",
                "microsoft/onnxruntime-genai",
                "onnx/onnx",
                "vllm-project/vllm",
                "xadupre/arranged-agents",
                "xadupre/onnx-light",
                "xadupre/onnx-light-cpu",
                "xadupre/onnx-light-kernel-images",
                "xadupre/onnx-light-ort",
                "xadupre/onnx-light-viewer",
                "xadupre/yet-another-onnx-builder",
                "xadupre/yet-another-onnxruntime-extensions",
                "xadupre/yet-another-tool",
            ],
        )
        self.assertIn(("/users/xadupre/repos", 2), self.requests)

    def test_home_links_cannot_reintroduce_excluded_owner_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "index.html"
            home.write_text(
                '<a href="https://github.com/xadupre/onnx-asv-benchmark">excluded</a>'
                '<a href="https://github.com/xadupre/onnxruntime">excluded</a>'
                '<a href="https://github.com/xadupre/arranged-agents-extra">excluded</a>'
                '<a href="https://github.com/xadupre/onnx-light-com-microsoft">excluded</a>'
                '<a href="https://github.com/xadupre/arranged-agents">included</a>'
                '<a href="https://github.com/xadupre/onnx-light-viewer">included</a>'
                '<a href="https://github.com/xadupre/yet-another-project">included</a>',
                encoding="utf-8",
            )
            repositories = downloads.discover_repositories("", home)
            self.assertNotIn("xadupre/onnx-asv-benchmark", repositories)
            self.assertNotIn("xadupre/onnxruntime", repositories)
            self.assertNotIn("xadupre/arranged-agents-extra", repositories)
            self.assertNotIn("xadupre/onnx-light-com-microsoft", repositories)
            self.assertIn("xadupre/arranged-agents", repositories)
            self.assertIn("xadupre/onnx-light-viewer", repositories)
            self.assertIn("xadupre/yet-another-project", repositories)
            self.assertIn("microsoft/onnxruntime-genai", repositories)
            self.assertIn("microsoft/onnxruntime-extensions", repositories)
            self.assertIn("ggml-org/llama.cpp", repositories)
            self.assertIn("vllm-project/vllm", repositories)

    def test_all_release_and_asset_pages_previews_and_empty_repositories(self):
        result = downloads.collect_repository("xadupre/onnx-light-cpu", "")
        self.assertEqual(result["total"], 302)
        self.assertEqual([r["tag"] for r in result["releases"]], ["v0", "v1/preview"])
        self.assertTrue(result["releases"][1]["prerelease"])
        self.assertEqual(len(result["releases"][1]["assets"]), 101)
        self.assertEqual(result["releases"][1]["assets"][-1]["name"], "checksums.txt")
        self.assertIn(("/repos/xadupre/onnx-light-cpu/releases", 2), self.requests)
        self.assertIn(
            ("/repos/xadupre/onnx-light-cpu/releases/1/assets", 2), self.requests
        )
        empty = downloads.collect_repository("xadupre/onnx-empty", "")
        self.assertEqual(empty["total"], 0)
        self.assertEqual(empty["releases"], [])

    def test_embedded_assets_avoid_per_release_requests(self):
        self.asset_count = 2
        self.fail_assets = True
        result = downloads.collect_repository("ggml-org/llama.cpp", "")
        self.assertEqual(result["total"], 6)
        self.assertEqual(len(result["releases"][1]["assets"]), 2)
        self.assertEqual(
            self.requests,
            [
                ("/repos/ggml-org/llama.cpp/releases", 1),
                ("/repos/ggml-org/llama.cpp/releases", 2),
            ],
        )

    def test_llm_repositories_record_release_downloads_and_history(self):
        repositories = ["ggml-org/llama.cpp", "vllm-project/vllm"]
        with tempfile.TemporaryDirectory() as tmp:
            result = downloads.record_downloads(tmp, repositories=repositories)
            self.assertEqual(
                [item["repository"] for item in result["repositories"]], repositories
            )
            self.assertEqual(
                result["history"][0]["totals"],
                {repository: 302 for repository in repositories},
            )
            for repository in repositories:
                project = self.read_project(tmp, repository)
                self.assertEqual(project["history"][0]["assets"][repository]["0"], 3)
                self.assertEqual(
                    [item["repository"] for item in project["repositories"]],
                    [repository],
                )
                self.assertEqual(set(project["history"][0]["assets"]), {repository})
                self.assertIn((f"/repos/{repository}/releases", 2), self.requests)
            self.assertNotIn("assets", result["history"][0])
            self.assertEqual(
                json.loads(
                    (Path(tmp) / "github-release-downloads/index.json").read_text()
                ),
                result,
            )

    def test_daily_upsert_and_history_survives_counter_decrease(self):
        with tempfile.TemporaryDirectory() as tmp:
            repositories = ["xadupre/onnx-light-cpu", "xadupre/onnx-empty"]
            now = dt.datetime(2026, 10, 20, 6, tzinfo=dt.UTC)
            first = downloads.record_downloads(tmp, repositories=repositories, now=now)
            path = (
                Path(tmp)
                / "github-release-downloads/xadupre/onnx-light-cpu/2026-10-20.json"
            )
            first_bytes = path.read_bytes()
            repeated = downloads.record_downloads(
                tmp, repositories=repositories, now=now
            )
            self.assertEqual(first, repeated)
            self.assertEqual(path.read_bytes(), first_bytes)
            self.count = 4
            updated = downloads.record_downloads(
                tmp, repositories=repositories, now=now
            )
            self.assertEqual(len(updated["history"]), 1)
            self.assertEqual(
                updated["history"][0]["totals"]["xadupre/onnx-light-cpu"], 402
            )
            self.assertEqual(
                self.read_project(tmp, repositories[0])["history"][0]["assets"][
                    "xadupre/onnx-light-cpu"
                ]["0"],
                4,
            )
            self.assertEqual(
                self.read_project(tmp, repositories[1])["history"][0]["assets"][
                    "xadupre/onnx-empty"
                ],
                {},
            )
            previous_day_bytes = path.read_bytes()
            previous_day_mtime = path.stat().st_mtime_ns
            self.count = 1
            next_day = downloads.record_downloads(
                tmp, repositories=repositories, now=now + dt.timedelta(days=1)
            )
            self.assertEqual(len(next_day["history"]), 2)
            self.assertEqual(
                [
                    entry["totals"]["xadupre/onnx-light-cpu"]
                    for entry in next_day["history"]
                ],
                [402, 102],
            )
            self.assertEqual(
                self.read_project(tmp, repositories[0])["history"][-1]["totals"],
                {"xadupre/onnx-light-cpu": 102},
            )
            self.assertEqual(path.read_bytes(), previous_day_bytes)
            self.assertEqual(path.stat().st_mtime_ns, previous_day_mtime)
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            self.assertFalse(list(Path(tmp).rglob("*.gz")))

    def test_legacy_history_retains_totals_without_inventing_asset_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "github-release-downloads.json"
            legacy = {"date": "2026-10-20", "totals": {"xadupre/onnx-light-cpu": 99}}
            path.write_text(json.dumps({"history": [legacy]}), encoding="utf-8")
            result = downloads.record_downloads(
                tmp,
                repositories=["xadupre/onnx-light-cpu"],
                now=dt.datetime(2026, 10, 21, tzinfo=dt.UTC),
            )
            self.assertEqual(result["history"][0], legacy)
            project = self.read_project(tmp, "xadupre/onnx-light-cpu")
            self.assertEqual(project["history"][0], legacy)
            counts = project["history"][1]["assets"]["xadupre/onnx-light-cpu"]
            self.assertEqual(counts["0"], 3)
            self.assertEqual(counts["101"], 2)
            self.assertNotIn("102", counts)
            self.assertEqual(sum(counts.values()), 302)
            self.assertFalse(path.exists())

    def test_daily_files_preserve_large_legacy_asset_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "github-release-downloads.json"
            history = [
                {
                    "date": (dt.date(2026, 1, 1) + dt.timedelta(days=day)).isoformat(),
                    "totals": {"ggml-org/llama.cpp": 1000 * day},
                    "assets": {
                        "ggml-org/llama.cpp": {
                            str(100000000 + asset): day for asset in range(1000)
                        }
                    },
                }
                for day in range(60)
            ]
            path.write_text(
                json.dumps({"history": history}, indent=2), encoding="utf-8"
            )
            result = downloads.record_downloads(
                tmp,
                repositories=["ggml-org/llama.cpp"],
                now=dt.datetime(2026, 10, 21, tzinfo=dt.UTC),
            )
            project = self.read_project(tmp, "ggml-org/llama.cpp")
            self.assertEqual(project["history"][:-1], history)
            directory = Path(tmp) / "github-release-downloads/ggml-org/llama.cpp"
            self.assertEqual(len(list(directory.glob("2026-*.json"))), 61)
            self.assertFalse(list(Path(tmp).rglob("*.gz")))
            self.assertFalse(path.exists())
            # A stale legacy file must not override the migrated history.
            path.write_text('{"history":[]}', encoding="utf-8")
            self.assertEqual(
                downloads.record_downloads(
                    tmp,
                    repositories=["ggml-org/llama.cpp"],
                    now=dt.datetime(2026, 10, 21, tzinfo=dt.UTC),
                ),
                result,
            )
            self.assertFalse(path.exists())

    def test_failed_staged_write_preserves_legacy_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "github-release-downloads.json"
            original = '{"history":[]}'
            path.write_text(original, encoding="utf-8")
            temporary = Path(tmp) / "github-release-downloads/index.json.tmp"
            temporary.mkdir(parents=True)
            with self.assertRaises(IsADirectoryError):
                downloads.record_downloads(tmp, repositories=["xadupre/onnx-empty"])
            self.assertEqual(path.read_text(), original)
            self.assertFalse(
                (
                    Path(tmp)
                    / "github-release-downloads/xadupre/onnx-empty/latest.json"
                ).exists()
            )

    def test_migration_splits_all_projects_and_preserves_unselected_history(self):
        projects = ["xadupre/onnx-light-cpu", "other/onnx-light-cpu"]
        history = [
            {"date": "2026-10-19", "totals": {projects[0]: 10, projects[1]: 20}},
            {
                "date": "2026-10-20",
                "totals": {projects[0]: 11, projects[1]: 19},
                "assets": {
                    projects[0]: {"101": 11},
                    projects[1]: {"101": 19, "deleted": 0},
                    "retired/project": {"removed": 8},
                },
            },
        ]
        legacy = {
            "updated_at": "2026-10-20T06:00:00Z",
            "repositories": [
                {"repository": repo, "total": total, "releases": []}
                for repo, total in zip(projects, [11, 19])
            ],
            "history": history,
        }
        for compressed in (False, True):
            with (
                self.subTest(compressed=compressed),
                tempfile.TemporaryDirectory() as tmp,
            ):
                path = Path(tmp) / (
                    "github-release-downloads.json.gz"
                    if compressed
                    else "github-release-downloads.json"
                )
                content = json.dumps(legacy).encode()
                path.write_bytes(gzip.compress(content) if compressed else content)
                now = dt.datetime(2026, 10, 21, tzinfo=dt.UTC)
                index = downloads.record_downloads(
                    tmp, repositories=[projects[0]], now=now
                )
                self.assertFalse(path.exists())
                for repo in projects:
                    project = self.read_project(tmp, repo)
                    for row, original in zip(project["history"], history):
                        self.assertEqual(
                            row["totals"], {repo: original["totals"][repo]}
                        )
                        self.assertEqual(
                            row.get("assets"),
                            {repo: original["assets"][repo]}
                            if "assets" in original
                            else None,
                        )
                archived_path = Path(tmp) / (
                    "github-release-downloads/other/onnx-light-cpu/2026-10-20.json"
                )
                archived_bytes = archived_path.read_bytes()
                self.assertEqual(
                    self.read_project(tmp, "retired/project")["history"][0]["assets"],
                    {"retired/project": {"removed": 8}},
                )
                self.assertTrue(all("assets" not in row for row in index["history"]))
                downloads.record_downloads(tmp, repositories=[projects[0]], now=now)
                self.assertEqual(archived_path.read_bytes(), archived_bytes)
                restored = downloads.record_downloads(
                    tmp, repositories=projects, now=now + dt.timedelta(days=1)
                )
                self.assertEqual(len(restored["repositories"]), 2)
                self.assertEqual(
                    self.read_project(tmp, projects[1])["history"][1]["assets"],
                    {projects[1]: {"101": 19, "deleted": 0}},
                )
                self.assertFalse(list(Path(tmp).rglob("*.gz")))

    def test_missing_project_history_does_not_replace_published_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            downloads.record_downloads(tmp, repositories=["xadupre/onnx-empty"])
            index_path = Path(tmp) / "github-release-downloads/index.json"
            original = index_path.read_bytes()
            project_path = (
                Path(tmp) / "github-release-downloads/xadupre/onnx-empty/latest.json"
            )
            project_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing release download"):
                downloads.record_downloads(tmp, repositories=["xadupre/onnx-empty"])
            self.assertEqual(index_path.read_bytes(), original)

    def test_project_gzip_migration_includes_archives_and_preserves_every_day(self):
        repository = "xadupre/onnx-light-cpu"
        archived = "retired/project"
        row = {
            "date": "2026-10-19",
            "totals": {repository: 5},
            "assets": {repository: {"42": 0, "43": 5}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "github-release-downloads"
            directory.mkdir()
            (directory / "index.json").write_text(
                json.dumps(
                    {
                        "updated_at": "2026-10-19T06:00:00Z",
                        "repositories": [
                            {
                                "repository": repository,
                                "total": 5,
                                "file": f"{repository}.json.gz",
                            },
                        ],
                        "history": [{"date": row["date"], "totals": row["totals"]}],
                    }
                )
            )
            for name in (repository, archived):
                source = directory / f"{name}.json.gz"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(
                    gzip.compress(
                        json.dumps(
                            {
                                "updated_at": "2026-10-19T06:00:00Z",
                                "repositories": [
                                    {"repository": name, "total": 5, "releases": []}
                                ],
                                "history": [
                                    {
                                        "date": row["date"],
                                        "totals": {name: 5},
                                        "assets": {name: {"42": 0, "43": 5}},
                                    }
                                ],
                            }
                        ).encode()
                    )
                )
            original_index = (directory / "index.json").read_bytes()
            blocked_index = directory / "index.json.tmp"
            blocked_index.mkdir()
            with self.assertRaises(IsADirectoryError):
                downloads.record_downloads(
                    tmp,
                    repositories=[repository],
                    now=dt.datetime(2026, 10, 20, tzinfo=dt.UTC),
                )
            self.assertEqual((directory / "index.json").read_bytes(), original_index)
            self.assertEqual(len(list(directory.glob("*/*.json.gz"))), 2)
            self.assertFalse((directory / repository / "latest.json").exists())
            blocked_index.rmdir()
            downloads.record_downloads(
                tmp,
                repositories=[repository],
                now=dt.datetime(2026, 10, 20, tzinfo=dt.UTC),
            )
            self.assertEqual(self.read_project(tmp, repository)["history"][0], row)
            self.assertEqual(
                self.read_project(tmp, archived)["history"][0]["assets"],
                {archived: {"42": 0, "43": 5}},
            )
            self.assertFalse(list(directory.rglob("*.gz")))
            historical = directory / repository / "2026-10-19.json"
            original_mtime = historical.stat().st_mtime_ns
            downloads.record_downloads(
                tmp,
                repositories=[repository],
                now=dt.datetime(2026, 10, 21, tzinfo=dt.UTC),
            )
            self.assertEqual(historical.stat().st_mtime_ns, original_mtime)
            self.assertEqual(
                self.read_project(tmp, repository)["dates"],
                ["2026-10-19", "2026-10-20", "2026-10-21"],
            )

    def test_failed_collection_never_replaces_existing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "github-release-downloads.json"
            path.write_text("existing data", encoding="utf-8")
            self.fail_assets = True
            with self.assertRaisesRegex(HTTPError, "503"):
                downloads.record_downloads(
                    tmp, repositories=["xadupre/onnx-empty", "xadupre/onnx-light-cpu"]
                )
            self.assertEqual(path.read_text(), "existing data")
            with self.assertRaisesRegex(RuntimeError, "No ONNX repositories"):
                downloads.record_downloads(tmp, repositories=[])
            self.assertEqual(path.read_text(), "existing data")

    def test_failed_collection_preserves_compressed_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "github-release-downloads.json.gz"
            original = gzip.compress(b'{"history":[]}', mtime=0)
            path.write_bytes(original)
            self.fail_assets = True
            with self.assertRaisesRegex(HTTPError, "503"):
                downloads.record_downloads(tmp, repositories=["xadupre/onnx-light-cpu"])
            self.assertEqual(path.read_bytes(), original)

    def test_cli_explicit_repositories(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                downloads.main(["--cache-dir", tmp, "--repo", "xadupre/onnx-empty"]), 0
            )
            data = self.read_project(tmp, "xadupre/onnx-empty")
            self.assertEqual(
                data["repositories"],
                [
                    {"repository": "xadupre/onnx-empty", "total": 0, "releases": []},
                ],
            )
            self.assertNotIn(("/users/xadupre/repos", 1), self.requests)


class TestDownloadIntegration(unittest.TestCase):
    def test_daily_workflow_and_page_are_connected(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/record_github_release_downloads.yml").read_text()
        )
        triggers = workflow.get("on", workflow.get(True))
        self.assertEqual(triggers["schedule"], [{"cron": "41 6 * * *"}])
        self.assertIn("workflow_dispatch", triggers)
        self.assertTrue(workflow["name"].startswith("DATA "))
        steps = workflow["jobs"]["record"]["steps"]
        self.assertTrue(
            any(
                step.get("run")
                == "python -u scripts/record_github_release_downloads.py"
                for step in steps
            )
        )
        page = (ROOT / "dashboard/github-release-downloads.html").read_text()
        self.assertIn("../assets/github-release-downloads.js", page)
        self.assertIn("../cache_data/github-release-downloads/index.json", page)
        self.assertIn("GitHub-generated source archives", page)
        self.assertIn(
            "dashboard/github-release-downloads.html", (ROOT / "index.html").read_text()
        )

    def test_browser_logic(self):
        subprocess.run(
            [
                "node",
                "--test",
                str(ROOT / "scripts/tests/github_release_downloads.test.cjs"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
