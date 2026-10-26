const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

const page = fs.readFileSync(
  path.join(__dirname, "../../dashboard/onnx/lost-compute.html"),
  "utf8"
);
const source = page.match(/<script>([\s\S]*?)<\/script>/)[1];

test("ONNX lost-compute aggregates daily merged PR activity", () => {
  assert.ok(source.includes('const PR_CSV_URL = "../../cache_data/onnx/pr_activity.csv";'));
  assert.ok(page.includes("../../cache_data/onnx/pr_activity.csv"));
  assert.ok(!page.includes("pr_stats.csv"));

  const start = source.indexOf("function parseCSVRaw");
  const end = source.indexOf("const PALETTE", start);
  const context = vm.createContext({});
  vm.runInContext(source.slice(start, end), context);
  const csv = [
    "date,merged_prs_1d",
    "2026-10-19T09:00:00Z,2",
    "2026-10-20T09:00:00Z,3",
    "2026-10-21T09:00:00Z,",
    "2026-10-26T09:00:00Z,1",
    "2026-10-27T09:00:00Z,-1",
  ].join("\n");
  const weekly = context.aggregatePRsWeekly(context.parsePRCSV(csv));
  const counts = JSON.parse(JSON.stringify(
    weekly.map(bucket => [bucket.week.toISOString().slice(0, 10), bucket.count])
  ));

  assert.deepEqual(counts, [["2026-10-19", 5], ["2026-10-26", 1]]);
});
