"""Tests for ``scripts.record_torch_coverage``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_torch_coverage as rtc  # noqa: E402


class TestRecordTorchCoverage(unittest.TestCase):
    def test_stringify_error_truncates_and_takes_first_line(self):
        self.assertEqual(rtc._stringify_error(None), "")
        self.assertEqual(rtc._stringify_error("boom"), "boom")
        self.assertEqual(rtc._stringify_error("boom\nrest"), "boom")
        long = "x" * 500
        out = rtc._stringify_error(long)
        self.assertTrue(out.endswith("..."))
        self.assertEqual(len(out), 400)

    def test_normalise_result_picks_subset(self):
        norm = rtc._normalise_result(
            {
                "name": "AtenA",
                "exporter": "yobx-new-tracing",
                "dynamic": True,
                "success": True,
                "abs": 1e-7,
                "rel": 2e-7,
                "extra": "ignored",
            }
        )
        self.assertEqual(norm["name"], "AtenA")
        self.assertEqual(norm["exporter"], "yobx-new-tracing")
        self.assertEqual(norm["dynamic"], 1)
        self.assertEqual(norm["success"], 1)
        self.assertEqual(norm["error_step"], "")
        self.assertEqual(norm["error"], "")
        self.assertAlmostEqual(norm["abs"], 1e-7)
        self.assertAlmostEqual(norm["rel"], 2e-7)
        self.assertNotIn("extra", norm)

    def test_normalise_result_failure_keeps_error(self):
        norm = rtc._normalise_result(
            {
                "name": "AtenB",
                "exporter": "export-tracing",
                "dynamic": False,
                "success": 0,
                "error_step": "export",
                "error": "Boom!\nWith stack",
            }
        )
        self.assertEqual(norm["success"], 0)
        self.assertEqual(norm["dynamic"], 0)
        self.assertEqual(norm["error_step"], "export")
        self.assertEqual(norm["error"], "Boom!")
        self.assertIsNone(norm["abs"])
        self.assertIsNone(norm["rel"])

    def test_build_payload_totals(self):
        results = [
            {
                "name": "A",
                "exporter": "yobx-new-tracing",
                "dynamic": 0,
                "success": 1,
                "error_step": "",
                "error": "",
                "abs": 0.0,
                "rel": 0.0,
            },
            {
                "name": "A",
                "exporter": "export-tracing",
                "dynamic": 0,
                "success": 0,
                "error_step": "export",
                "error": "Boom",
                "abs": None,
                "rel": None,
            },
            {
                "name": "A",
                "exporter": "yobx-new-tracing",
                "dynamic": 1,
                "success": 1,
                "error_step": "",
                "error": "",
                "abs": 0.0,
                "rel": 0.0,
            },
        ]
        payload = rtc.build_payload(
            results,
            case_names=["A"],
            exporters=("yobx-new-tracing", "export-tracing"),
            dynamic=(False, True),
            commit="abc123",
        )
        self.assertEqual(payload["commit"], "abc123")
        self.assertEqual(payload["cases"], ["A"])
        self.assertEqual(payload["exporters"], ["yobx-new-tracing", "export-tracing"])
        self.assertEqual(payload["dynamic"], [False, True])
        self.assertEqual(
            payload["totals"]["yobx-new-tracing"],
            {"success": 2, "failure": 0, "total": 2},
        )
        self.assertEqual(
            payload["totals"]["export-tracing"],
            {"success": 0, "failure": 1, "total": 1},
        )
        self.assertEqual(payload["results"], results)
        # Date must be a parseable ISO timestamp.
        self.assertRegex(payload["date"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_parse_args_defaults(self):
        args = rtc.parse_args([])
        self.assertEqual(args.cache_dir, os.path.join("cache_data"))
        self.assertEqual(args.repo, "yet-another-onnx-builder")
        self.assertIsNone(args.exporters)
        self.assertIsNone(args.limit)

    def test_parse_args_custom_exporters(self):
        args = rtc.parse_args(["--exporter", "a", "--exporter", "b", "--limit", "3"])
        self.assertEqual(args.exporters, ["a", "b"])
        self.assertEqual(args.limit, 3)

    def test_default_exporters_include_yobx_and_dynamo(self):
        # The coverage dashboard exposes the default ``yobx`` exporter and
        # ``torch.onnx.export`` (referred to here as ``dynamo``) alongside the
        # other torch-to-ONNX exporters.
        self.assertIn("yobx", rtc.DEFAULT_EXPORTERS)
        self.assertIn("dynamo", rtc.DEFAULT_EXPORTERS)

    def test_existing_snapshot_is_valid_json(self):
        repo_root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(
            repo_root,
            "cache_data",
            "yet-another-onnx-builder",
            "torch_coverage.json",
        )
        if not os.path.exists(path):
            self.skipTest("torch_coverage.json snapshot not present in repo")
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        for key in ("date", "exporters", "cases", "results", "totals"):
            self.assertIn(key, payload)
        self.assertIsInstance(payload["results"], list)
        if payload["results"]:
            sample = payload["results"][0]
            for key in ("name", "exporter", "dynamic", "success"):
                self.assertIn(key, sample)


if __name__ == "__main__":
    unittest.main()
