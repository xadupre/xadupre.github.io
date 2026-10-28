const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");
const stats = require("../../assets/pr-review-stats.js");

const header = [
  "pr_number", "title", "author", "merged_at", "commits", "comments",
  "copilot_comments", "copilot_commits", "review_agent_comments", "review_agent_high",
  "review_agent_medium", "review_agent_low", "review_agent_unclassified", "review_agent_security",
  "review_agent_resolved_threads", "review_agent_unresolved_threads", "additions", "deletions",
  "changed_files",
];
const columns = [
  "42", "<img src=x onerror=alert(1)>", "alice", "2026-10-24T10:00:00Z", "3", "9",
  "4", "0", "4", "1", "1", "0", "2", "1", "2", "1", "120", "30", "4",
];

test("metric parsing distinguishes zero, missing and invalid data", () => {
  const row = stats.parseValues(header, columns);
  assert.equal(row.review_agent_comments, 4);
  assert.equal(row.review_agent_low, 0);
  assert.equal(row.additions, 120);
  const old = stats.parseValues(header.slice(0, 8), columns.slice(0, 8));
  assert.equal(old.review_agent_comments, null);
  assert.equal(stats.tableValues(old)[0], "\u2014");
  assert.equal(stats.summary([row, old], ["review_agent_comments"]), "4 (1/2 PRs)");
  assert.equal(stats.summary([row, old], ["additions", "deletions"], true), "120.0 / 30.0 (1/2 PRs)");
  assert.equal(stats.summary([old], ["review_agent_comments"]), "\u2014 (0/1 PRs)");
  for (const bad of ["", " ", "-1", "NaN", "Infinity", "oops"]) {
    assert.equal(stats.parseValues(["additions"], [bad]).additions, null);
  }
});

class Element {
  constructor() {
    this.children = [];
    this.style = {};
    this.classList = {add() {}};
    this.textContent = "";
    this.parentElement = {style: {}};
  }
  appendChild(child) { this.children.push(child); }
  getContext() { return this; }
}

async function renderPage(repo, legacy) {
  const page = fs.readFileSync(path.join(__dirname, "../../dashboard", repo, "pr-stats.html"), "utf8");
  assert.ok(page.includes('src="../../assets/pr-review-stats.js"'));
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const context = vm.createContext({
    document: {
      getElementById: get,
      querySelector: get,
      createElement: () => new Element(),
    },
    loadChartJs: async () => {},
    fetch: async () => ({
      ok: true,
      text: async () => (legacy ? header.slice(0, 8) : header).join(",") + "\n" +
        (legacy ? columns.slice(0, 8) : columns).join(",") + "\n",
    }),
    Chart: class {
      constructor(canvas) { this.canvas = canvas; }
      resetZoom() {}
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../../assets/pr-review-stats.js"), "utf8"), context);
  const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(get("message").textContent, "");
  return get;
}

for (const repo of [
  "arranged-agents",
  "onnx-light", "onnx-light-cpu", "onnx-light-kernel-images", "onnx-light-ort",
  "onnx-light-viewer", "yet-another-onnx-builder", "yet-another-onnxruntime-extensions",
]) {
  test(repo + " renders new metrics without interpreting titles as HTML", async () => {
    const get = await renderPage(repo, false);
    const cells = get("lastObservationsBody").children[0].children;
    assert.equal(cells.length, 14);
    assert.equal(cells[3].textContent, columns[1]);
    assert.equal(cells[8].textContent, "4");
    assert.equal(cells[9].textContent, "1 / 1 / 0 / 2");
    assert.equal(cells[11].textContent, "2 / 1");
    assert.equal(cells[12].textContent, "120 / 30");
    assert.equal(get("#lastObservations thead tr").children.length, 6);
    assert.equal(get("#summary .grid").children[0].children[1].textContent, "4 (1/1 PRs)");
  });
}

test("legacy CSV still renders existing metrics with unknown review metrics", async () => {
  const get = await renderPage("onnx-light", true);
  const cells = get("lastObservationsBody").children[0].children;
  assert.equal(cells[4].textContent, 3);
  assert.equal(cells[8].textContent, "\u2014");
  assert.equal(cells[12].textContent, "\u2014 / \u2014");
});
