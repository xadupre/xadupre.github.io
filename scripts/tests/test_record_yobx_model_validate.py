"""Tests for ``scripts.record_yobx_model_validate``."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_yobx_model_validate as rymv  # noqa: E402

_STANDARD_EXPORT_MODULES = ("torch", "yobx", "onnx")


def _require_modules(testcase, *modules):
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        testcase.skipTest(f"optional export stack is not installed ({', '.join(missing)})")


def _run_real_export_test(testcase, entry, cfg, dump_folder):
    """Exercise the real exporter path when the optional stack is installed."""
    required = {
        "yobx": _STANDARD_EXPORT_MODULES,
        "dynamo": _STANDARD_EXPORT_MODULES,
        "onnx-dynamo": _STANDARD_EXPORT_MODULES,
        "yobx-to_onnx": _STANDARD_EXPORT_MODULES,
        "olive-modelbuilder": (
            *_STANDARD_EXPORT_MODULES,
            "olive",
            "onnxruntime_genai",
        ),
    }
    _require_modules(testcase, *required[cfg["exporter"]])
    observed = rymv.run_validate_one(
        entry, cfg, dump_folder=dump_folder, quiet=False, verbose=1
    )
    hub_error = observed.get("error_config") or observed.get("error_export")
    if rymv._is_hf_hub_access_error(hub_error):
        testcase.skipTest(hub_error)
    testcase.assertEqual("OK", observed["export"])
    if cfg["exporter"] == "olive-modelbuilder":
        testcase.assertIn(observed["discrepancies"], {"OK", "SKIPPED"})
    else:
        testcase.assertEqual("OK", observed["discrepancies"])


class TestRecordYobxModelValidate(unittest.TestCase):
    def test_defaults(self):
        model_ids = {e["model"] for e in rymv.DEFAULT_MODELS}
        self.assertIn("arnir0/Tiny-LLM", model_ids)
        self.assertIn("mistralai/Mistral-7B-v0.3", model_ids)
        # Each model entry must declare the per-model fields used by the
        # recorder so the snapshot can carry them.
        for entry in rymv.DEFAULT_MODELS:
            self.assertIn("model", entry)
            self.assertIn("dtype", entry)
            self.assertIn("device", entry)
            self.assertIn("atol", entry)
            self.assertIn("task", entry)
        labels = {e["label"] for e in rymv.DEFAULT_EXPORTERS}
        self.assertIn("yobx", labels)
        self.assertIn("yobx-ort", labels)
        self.assertIn("dynamo-ir", labels)
        self.assertIn("onnx-dynamo-os_ort", labels)
        self.assertIn("yobx-to_onnx", labels)
        # Each exporter config must declare the three required fields.
        for cfg in rymv.DEFAULT_EXPORTERS:
            self.assertIn("label", cfg)
            self.assertIn("exporter", cfg)
            self.assertIn("optimization", cfg)
        self.assertEqual(rymv.DEFAULT_DTYPE, "float16")
        self.assertEqual(rymv.DEFAULT_DEVICE, "cpu")

    def test_coerce_model_entry_from_string(self):
        entry = rymv._coerce_model_entry("a/b")
        self.assertEqual(entry["model"], "a/b")
        self.assertEqual(entry["dtype"], rymv.DEFAULT_DTYPE)
        self.assertEqual(entry["device"], rymv.DEFAULT_DEVICE)
        self.assertEqual(entry["atol"], rymv.DEFAULT_ATOL)
        self.assertEqual(entry["task"], rymv.DEFAULT_TASK)

    def test_coerce_model_entry_from_dict_preserves_overrides(self):
        src = dict(model="a/b", dtype="float32", atol=0.5, device="cuda", task="t")
        entry = rymv._coerce_model_entry(src)
        self.assertEqual(entry["model"], "a/b")
        self.assertEqual(entry["dtype"], "float32")
        self.assertEqual(entry["device"], "cuda")
        self.assertEqual(entry["atol"], 0.5)
        self.assertEqual(entry["task"], "t")
        # Returns a copy so the input is not mutated.
        self.assertIsNot(entry, src)

    def test_stringify_error_collapses_whitespace_and_truncates(self):
        self.assertEqual(rymv._stringify_error(None), "")
        self.assertEqual(rymv._stringify_error("boom"), "boom")
        # Multi-line errors are collapsed into a single line so that the
        # actionable trailing lines (such as HuggingFace's "You need to
        # have sentencepiece installed ...") are preserved in the snapshot.
        self.assertEqual(rymv._stringify_error("boom\nrest"), "boom rest")
        self.assertEqual(
            rymv._stringify_error(
                "Couldn't instantiate the backend tokenizer from one of: \n"
                "(1) a `tokenizers` library serialization file, \n"
                "(2) a slow tokenizer instance to convert or \n"
                "(3) an equivalent slow tokenizer class to instantiate and convert. \n"
                "You need to have sentencepiece installed to convert a slow "
                "tokenizer to a fast one."
            ),
            "Couldn't instantiate the backend tokenizer from one of: "
            "(1) a `tokenizers` library serialization file, "
            "(2) a slow tokenizer instance to convert or "
            "(3) an equivalent slow tokenizer class to instantiate and convert. "
            "You need to have sentencepiece installed to convert a slow "
            "tokenizer to a fast one.",
        )
        long = "x" * 500
        out = rymv._stringify_error(long)
        self.assertTrue(out.endswith("..."))
        self.assertEqual(len(out), 400)

    def test_to_float_handles_non_finite(self):
        self.assertIsNone(rymv._to_float(None))
        self.assertIsNone(rymv._to_float("nope"))
        self.assertIsNone(rymv._to_float(float("inf")))
        self.assertIsNone(rymv._to_float(float("nan")))
        self.assertEqual(rymv._to_float(1.5), 1.5)
        self.assertEqual(rymv._to_float("2"), 2.0)

    def test_is_cell_working(self):
        self.assertTrue(rymv.is_cell_working({"export": "OK", "discrepancies": "OK"}))
        self.assertFalse(
            rymv.is_cell_working({"export": "FAILED", "discrepancies": "OK"})
        )
        self.assertFalse(
            rymv.is_cell_working({"export": "OK", "discrepancies": "FAILED"})
        )
        # Export OK but no discrepancy check -> not "working" for our purposes.
        self.assertFalse(rymv.is_cell_working({"export": "OK"}))
        self.assertFalse(rymv.is_cell_working({}))
        self.assertFalse(rymv.is_cell_working(None))

    def test_is_cell_working_honours_model_atol(self):
        # Even when ``validate_model`` reports a failed discrepancy check
        # (because it used a stricter default tolerance), a cell where the
        # observed max abs error is within the model's declared atol should
        # be considered working.
        summary = {
            "export": "OK",
            "discrepancies": "FAILED",
            "discrepancies_max_abs": 0.01,
            "discrepancies_atol": 1e-5,
        }
        self.assertTrue(rymv.is_cell_working(summary, model_atol=0.02))
        # Still failing when the per-model atol is stricter than the error.
        self.assertFalse(rymv.is_cell_working(summary, model_atol=0.001))
        # Export failure dominates even when within atol.
        self.assertFalse(
            rymv.is_cell_working({**summary, "export": "FAILED"}, model_atol=0.02)
        )
        # Without a max_abs we cannot conclude success.
        self.assertFalse(
            rymv.is_cell_working(
                {"export": "OK", "discrepancies": "FAILED"}, model_atol=0.02
            )
        )

    def test_first_error_picks_earliest_failing_step(self):
        step, msg = rymv._first_error(
            {
                "error_config": "bad config\nstack",
                "error_export": "export boom",
            }
        )
        self.assertEqual(step, "config")
        self.assertEqual(msg, "bad config stack")

        step, msg = rymv._first_error({"error_export": "export boom"})
        self.assertEqual(step, "export")
        self.assertEqual(msg, "export boom")

        step, msg = rymv._first_error({})
        self.assertEqual(step, "")
        self.assertEqual(msg, "")

    def test_normalise_result_success(self):
        cfg = {"label": "yobx", "exporter": "yobx", "optimization": "default"}
        row = rymv._normalise_result(
            "arnir0/Tiny-LLM",
            cfg,
            {
                "export": "OK",
                "discrepancies": "OK",
                "discrepancies_ok": 3,
                "discrepancies_total": 3,
                "discrepancies_max_abs": 1e-5,
                "discrepancies_atol": 1e-3,
                "n_nodes": 42,
                "top_op_types": "MatMul:5",
            },
            duration_s=1.25,
        )
        self.assertEqual(row["model_id"], "arnir0/Tiny-LLM")
        self.assertEqual(row["label"], "yobx")
        self.assertEqual(row["exporter"], "yobx")
        self.assertEqual(row["optimization"], "default")
        self.assertEqual(row["success"], 1)
        self.assertEqual(row["export"], "OK")
        self.assertEqual(row["discrepancies"], "OK")
        self.assertEqual(row["discrepancies_ok"], 3)
        self.assertEqual(row["discrepancies_total"], 3)
        self.assertAlmostEqual(row["discrepancies_max_abs"], 1e-5)
        self.assertAlmostEqual(row["discrepancies_atol"], 1e-3)
        self.assertEqual(row["n_nodes"], 42)
        self.assertEqual(row["top_op_types"], "MatMul:5")
        self.assertAlmostEqual(row["duration_s"], 1.25)
        self.assertEqual(row["error_step"], "")
        self.assertEqual(row["error"], "")

    def test_normalise_result_failure(self):
        cfg = {"label": "dynamo-ir", "exporter": "dynamo", "optimization": "ir"}
        row = rymv._normalise_result(
            "microsoft/Phi-4-reasoning",
            cfg,
            {"export": "FAILED", "error_export": "boom!\nstack"},
        )
        self.assertEqual(row["success"], 0)
        self.assertEqual(row["error_step"], "export")
        self.assertEqual(row["error"], "boom! stack")
        self.assertIsNone(row["discrepancies_max_abs"])
        self.assertEqual(row["discrepancies"], "")

    def test_normalise_result_success_within_model_atol(self):
        # ``validate_model`` flagged discrepancies as failed (because of its
        # stricter default tolerance), but the observed max abs error is
        # below the per-model atol, so the cell should be marked as working.
        cfg = {"label": "yobx", "exporter": "yobx", "optimization": "default"}
        row = rymv._normalise_result(
            "arnir0/Tiny-LLM",
            cfg,
            {
                "export": "OK",
                "discrepancies": "FAILED",
                "discrepancies_max_abs": 0.01,
                "discrepancies_atol": 1e-5,
            },
            model_atol=0.02,
        )
        self.assertEqual(row["success"], 1)
        self.assertEqual(row["error_step"], "")
        # ``discrepancies`` and ``discrepancies_atol`` should be rewritten so
        # the cached JSON reflects the per-model tolerance that made the cell
        # successful, instead of the stricter default used by ``validate_model``.
        self.assertEqual(row["discrepancies"], "OK")
        self.assertAlmostEqual(row["discrepancies_atol"], 0.02)
        self.assertEqual(row["error"], "")

    def test_merge_last_working_records_now_on_success(self):
        row = {"success": 1}
        out = rymv.merge_last_working(row, None, "2024-05-01T00:00:00Z", "abc")
        self.assertEqual(out["last_working_date"], "2024-05-01T00:00:00Z")
        self.assertEqual(out["last_working_commit"], "abc")

    def test_merge_last_working_preserves_previous_on_failure(self):
        row = {"success": 0}
        previous = {
            "last_working_date": "2024-01-01T00:00:00Z",
            "last_working_commit": "deadbeef",
        }
        out = rymv.merge_last_working(row, previous, "2024-05-01T00:00:00Z", "abc")
        self.assertEqual(out["last_working_date"], "2024-01-01T00:00:00Z")
        self.assertEqual(out["last_working_commit"], "deadbeef")

    def test_merge_last_working_no_previous_failure_yields_empty(self):
        row = {"success": 0}
        out = rymv.merge_last_working(row, None, "2024-05-01T00:00:00Z", "abc")
        self.assertEqual(out["last_working_date"], "")
        self.assertEqual(out["last_working_commit"], "")

    def test_build_payload_totals_and_dates(self):
        cfg_yobx = {"label": "yobx", "exporter": "yobx", "optimization": "default"}
        cfg_dyn = {"label": "dynamo-ir", "exporter": "dynamo", "optimization": "ir"}
        raw = [
            ("arnir0/Tiny-LLM", cfg_yobx, {"export": "OK", "discrepancies": "OK"}, 1.5),
            (
                "arnir0/Tiny-LLM",
                cfg_dyn,
                {"export": "FAILED", "error_export": "boom"},
                0.7,
            ),
            (
                "microsoft/Phi-4-reasoning",
                cfg_yobx,
                {"export": "FAILED", "error_export": "boom"},
                2.0,
            ),
            (
                "microsoft/Phi-4-reasoning",
                cfg_dyn,
                {"export": "OK", "discrepancies": "OK"},
                3.25,
            ),
        ]
        previous = {
            "results": [
                {
                    "model_id": "microsoft/Phi-4-reasoning",
                    "label": "yobx",
                    "last_working_date": "2024-01-01T00:00:00Z",
                    "last_working_commit": "old",
                }
            ],
            "tasks": {"microsoft/Phi-4-reasoning": "fill-mask"},
        }
        payload = rymv.build_payload(
            raw_results=raw,
            models=(
                dict(
                    model="arnir0/Tiny-LLM",
                    dtype="float16",
                    atol=0.02,
                    device="cpu",
                    task="text-generation",
                ),
                dict(
                    model="microsoft/Phi-4-reasoning",
                    dtype="float16",
                    atol=0.02,
                    device="cpu",
                    task="text-generation",
                ),
            ),
            exporters=(cfg_yobx, cfg_dyn),
            dtype="float16",
            device="cpu",
            commit="newcommit",
            previous_payload=previous,
            now="2024-05-01T00:00:00Z",
            tasks={"arnir0/Tiny-LLM": "text-generation"},
        )
        self.assertEqual(payload["date"], "2024-05-01T00:00:00Z")
        self.assertEqual(payload["dtype"], "float16")
        self.assertEqual(payload["device"], "cpu")
        self.assertEqual(
            payload["totals"]["yobx"], {"success": 1, "failure": 1, "total": 2}
        )
        self.assertEqual(
            payload["totals"]["dynamo-ir"], {"success": 1, "failure": 1, "total": 2}
        )
        rows_by_key = {(r["model_id"], r["label"]): r for r in payload["results"]}
        # New success refreshes last_working.
        self.assertEqual(
            rows_by_key[("arnir0/Tiny-LLM", "yobx")]["last_working_date"],
            "2024-05-01T00:00:00Z",
        )
        # Previous-known good is preserved on a fresh failure.
        self.assertEqual(
            rows_by_key[("microsoft/Phi-4-reasoning", "yobx")]["last_working_date"],
            "2024-01-01T00:00:00Z",
        )
        # No previous and now failing -> empty.
        self.assertEqual(
            rows_by_key[("arnir0/Tiny-LLM", "dynamo-ir")]["last_working_date"], ""
        )
        # Per-cell export duration is preserved in the row.
        self.assertAlmostEqual(
            rows_by_key[("arnir0/Tiny-LLM", "yobx")]["duration_s"], 1.5
        )
        self.assertAlmostEqual(
            rows_by_key[("microsoft/Phi-4-reasoning", "dynamo-ir")]["duration_s"], 3.25
        )
        # HuggingFace task is captured per row, taking the explicit value first
        # and falling back to the previous snapshot.
        self.assertEqual(
            rows_by_key[("arnir0/Tiny-LLM", "yobx")]["task"], "text-generation"
        )
        self.assertEqual(
            rows_by_key[("microsoft/Phi-4-reasoning", "yobx")]["task"],
            "text-generation",
        )
        self.assertEqual(
            payload["tasks"],
            {
                "arnir0/Tiny-LLM": "text-generation",
                "microsoft/Phi-4-reasoning": "text-generation",
            },
        )
        # Per-model dtype/device/atol are recorded in each row.
        self.assertEqual(rows_by_key[("arnir0/Tiny-LLM", "yobx")]["dtype"], "float16")
        self.assertEqual(rows_by_key[("arnir0/Tiny-LLM", "yobx")]["device"], "cpu")
        self.assertAlmostEqual(rows_by_key[("arnir0/Tiny-LLM", "yobx")]["atol"], 0.02)
        # Payload must be JSON-serialisable with allow_nan=False.
        json.dumps(payload, allow_nan=False)

    def test_load_existing_cache_missing_or_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(rymv._load_existing_cache(os.path.join(tmp, "x.json")), {})
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("not json")
            self.assertEqual(rymv._load_existing_cache(bad), {})
            arr = os.path.join(tmp, "arr.json")
            with open(arr, "w", encoding="utf-8") as fh:
                fh.write("[]")
            self.assertEqual(rymv._load_existing_cache(arr), {})

    def test_parse_args_defaults(self):
        args = rymv.parse_args([])
        self.assertEqual(args.cache_dir, os.path.join("cache_data"))
        self.assertEqual(args.repo, "yet-another-onnx-builder")
        self.assertIsNone(args.models)
        self.assertEqual(args.dtype, "float16")
        self.assertEqual(args.device, "cpu")
        self.assertIsNone(args.limit)
        self.assertIsNone(args.dump_folder)

    def test_parse_args_dump_folder(self):
        args = rymv.parse_args(["--dump-folder", "/tmp/dump"])
        self.assertEqual(args.dump_folder, "/tmp/dump")

    def test_parse_args_quiet_default(self):
        args = rymv.parse_args([])
        self.assertTrue(args.quiet)

    def test_parse_args_no_quiet(self):
        args = rymv.parse_args(["--no-quiet"])
        self.assertFalse(args.quiet)
        args = rymv.parse_args(["--no-quiet", "--quiet"])
        self.assertTrue(args.quiet)

    def test_parse_args_custom_models(self):
        args = rymv.parse_args(["--model", "a/b", "--model", "c/d", "--limit", "1"])
        self.assertEqual(args.models, ["a/b", "c/d"])
        self.assertEqual(args.limit, 1)

    def _write_extra_xlsx(self, path: str, value: float) -> None:
        """Write a minimal workbook with the ``extra`` sheet used by yobx."""
        from openpyxl import Workbook

        wb = Workbook()
        # Replace the default sheet with one named ``extra`` and populate it
        # with a small key/value table mimicking the yobx export report.
        default = wb.active
        wb.remove(default)
        ws = wb.create_sheet("extra")
        ws.append(["name", "value"])
        ws.append(["builder", "torch"])
        ws.append(["stat_time_export_and_post_processing", value])
        ws.append(["stat_time_post_process_exported_program", 0.001])
        wb.save(path)

    def test_read_yobx_export_duration_returns_metric(self):
        try:
            import openpyxl  # noqa: F401
        except Exception:
            self.skipTest("openpyxl is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            xlsx = os.path.join(tmp, "model.xlsx")
            self._write_extra_xlsx(xlsx, 1.5)
            self.assertAlmostEqual(rymv._read_yobx_export_duration(xlsx), 1.5)

    def test_read_yobx_export_duration_missing_file(self):
        self.assertIsNone(rymv._read_yobx_export_duration("/no/such/file.xlsx"))

    def test_read_yobx_export_duration_missing_sheet(self):
        try:
            from openpyxl import Workbook
        except Exception:
            self.skipTest("openpyxl is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            xlsx = os.path.join(tmp, "other.xlsx")
            wb = Workbook()
            wb.active.append(["a", "b"])
            wb.save(xlsx)
            self.assertIsNone(rymv._read_yobx_export_duration(xlsx))

    def test_list_xlsx_recurses(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(rymv._list_xlsx(tmp), set())
            sub = os.path.join(tmp, "sub")
            os.makedirs(sub)
            a = os.path.join(tmp, "a.xlsx")
            b = os.path.join(sub, "b.xlsx")
            with open(a, "w", encoding="utf-8") as fh:
                fh.write("")
            with open(b, "w", encoding="utf-8") as fh:
                fh.write("")
            # A non-xlsx file is ignored.
            with open(os.path.join(tmp, "skip.txt"), "w", encoding="utf-8") as fh:
                fh.write("")
            self.assertEqual(rymv._list_xlsx(tmp), {a, b})
            self.assertEqual(rymv._list_xlsx(None), set())
            self.assertEqual(rymv._list_xlsx("/no/such/folder"), set())

    def test_run_all_uses_xlsx_duration_for_yobx(self):
        try:
            import openpyxl  # noqa: F401
        except Exception:
            self.skipTest("openpyxl is not installed")
        cfg_yobx = {"label": "yobx", "exporter": "yobx", "optimization": "default"}
        cfg_dyn = {"label": "dynamo-ir", "exporter": "dynamo", "optimization": "ir"}

        with tempfile.TemporaryDirectory() as tmp:
            test = self

            def fake_run_validate_one(
                entry, exporter_cfg, verbose=0, dump_folder=None, quiet=True
            ):
                if exporter_cfg["exporter"] == "yobx":
                    # Mimic the yobx exporter writing a companion workbook.
                    xlsx = os.path.join(
                        dump_folder, f"{entry['model'].replace('/', '_')}.xlsx"
                    )
                    test._write_extra_xlsx(xlsx, 2.5)
                return {"export": "OK", "discrepancies": "OK"}

            original = rymv.run_validate_one
            rymv.run_validate_one = fake_run_validate_one
            try:
                results = rymv.run_all(
                    models=({"model": "a/b", "dtype": "float16", "device": "cpu"},),
                    exporters=(cfg_yobx, cfg_dyn),
                    dump_folder=tmp,
                )
            finally:
                rymv.run_validate_one = original

        durations = {item[1]["label"]: item[3] for item in results}
        # yobx duration is replaced by the metric from the extra sheet.
        self.assertAlmostEqual(durations["yobx"], 2.5)
        # The other exporter keeps its wall-clock measurement.
        self.assertGreaterEqual(durations["dynamo-ir"], 0.0)
        self.assertNotAlmostEqual(durations["dynamo-ir"], 2.5)

    def test_existing_snapshot_is_valid_json(self):
        repo_root = os.path.dirname(os.path.dirname(HERE))
        path = os.path.join(
            repo_root,
            "cache_data",
            "yet-another-onnx-builder",
            "model_validate.json",
        )
        if not os.path.exists(path):
            self.skipTest("model_validate.json snapshot not present in repo")
        with open(path, encoding="utf-8") as fh:

            def _reject(token: str) -> None:
                raise ValueError(f"non-JSON token in snapshot: {token}")

            payload = json.load(fh, parse_constant=_reject)
        for key in (
            "date",
            "exporters",
            "models",
            "results",
            "totals",
            "dtype",
            "device",
        ):
            self.assertIn(key, payload)
        self.assertIsInstance(payload["results"], list)

    def test_is_rate_limit_error_detects_429(self):
        self.assertTrue(
            rymv._is_rate_limit_error(
                Exception(
                    "HTTP Error 429 thrown while requesting HEAD "
                    "https://huggingface.co/mistralai/Mistral-7B-v0.3/"
                    "resolve/main/config.json"
                )
            )
        )
        self.assertTrue(
            rymv._is_rate_limit_error(Exception("429 Client Error: Too Many Requests"))
        )

        class _Resp:
            status_code = 429

        class _Err(Exception):
            response = _Resp()

        self.assertTrue(rymv._is_rate_limit_error(_Err("boom")))

    def test_is_rate_limit_error_ignores_other_errors(self):
        self.assertFalse(rymv._is_rate_limit_error(Exception("HTTP Error 404")))
        self.assertFalse(rymv._is_rate_limit_error(ValueError("bad model")))

    def test_is_hf_hub_access_error_detects_offline_message(self):
        msg = (
            "We couldn't connect to 'https://huggingface.co' to load the "
            "files, and couldn't find them in the cached files. Check "
            "your internet connection or see how to run the library in "
            "offline mode at "
            "'https://huggingface.co/docs/transformers/installation"
            "#offline-mode'."
        )
        self.assertTrue(rymv._is_hf_hub_access_error(msg))
        self.assertTrue(rymv._is_hf_hub_access_error(Exception(msg)))

    def test_is_hf_hub_access_error_ignores_unrelated_errors(self):
        self.assertFalse(rymv._is_hf_hub_access_error(None))
        self.assertFalse(rymv._is_hf_hub_access_error(""))
        self.assertFalse(rymv._is_hf_hub_access_error("ValueError: bad model"))
        # A rate-limit error mentions huggingface.co but is not a Hub
        # access error in the sense detected here.
        self.assertFalse(
            rymv._is_hf_hub_access_error(
                "HTTP Error 429 thrown while requesting HEAD "
                "https://huggingface.co/x/y/resolve/main/config.json"
            )
        )


class TestPerModelPerExporterDispatch(unittest.TestCase):
    def _check_cell(self, entry, cfg):
        with tempfile.TemporaryDirectory() as tmp:
            _run_real_export_test(self, entry, cfg, tmp)


def _slugify(value: str) -> str:
    """Return a valid Python identifier fragment for a test method name."""
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _make_cell_test(entry, cfg):
    def test(self):
        self._check_cell(entry, cfg)

    # test.__doc__ = (
    #     f"model-id coverage cell ({entry['model']!r}, {cfg['label']!r}) "
    #     "is routed to the right backend."
    # )
    return test


# Generate one test function per (model, exporter) cell of the model-id
# coverage page so that a failure pinpoints the exact cell that regressed
# (and so each model/exporter pairing is its own test rather than a single
# parametrised loop). This covers the Olive runtime (``olive-modelbuilder``)
# and the ``torch.onnx.export`` / Olive-backed columns that are expected to
# work on the tiny smoke-test model.
for _entry in rymv.DEFAULT_MODELS:
    if "arnir0" not in _entry["model"]:
        continue
    for _cfg in rymv.DEFAULT_EXPORTERS:
        if _cfg["label"] not in {
            "yobx",
            "dynamo-ir",
            "yobx-to_onnx",
            "olive-modelbuilder",
        }:
            continue
        _name = f"test_cell_{_slugify(_entry['model'])}__{_slugify(_cfg['label'])}"
        assert "atol" in _entry, f"incomplete {_entry!r}"
        setattr(
            TestPerModelPerExporterDispatch,
            _name,
            _make_cell_test(_entry, _cfg),
        )
del _entry, _cfg, _name


if __name__ == "__main__":
    unittest.main(verbosity=2)
