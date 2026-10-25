const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "../../assets/pr-activity.js"), "utf8");
const header = "date,open_prs,opened_prs_7d,closed_prs_7d,merged_prs_7d,opened_prs_1d,closed_prs_1d,merged_prs_1d,avg_open_age_days,avg_merged_age_days_365d";
const csv = header + "\n2026-10-24T00:00:00Z,0,14,7,7,3,2,1,0,2.5\n" +
  "2026-10-23T00:00:00Z,3,7,5,4,2,1,1,20.5,\n";

class Element {
  constructor() {
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.listeners = {};
    this.textContent = "";
  }
  appendChild(child) { this.children.push(child); return child; }
  get firstChild() { return this.children[0]; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
}

async function render({text = csv, tables = {}, missingCSV = false, missingPulls = false} = {}) {
  const elements = new Map();
  const root = new Element();
  root.dataset.prActivity = "xadupre/onnx-light";
  Object.defineProperty(root, "innerHTML", {
    set(html) {
      for (const match of html.matchAll(/data-id="([^"]+)"/g)) elements.set(match[1], new Element());
    },
  });
  root.querySelector = selector => elements.get(selector.match(/data-id="([^"]+)"/)[1]);
  const requests = [];
  const charts = new Map();
  class Chart {
    constructor(canvas, config) { this.config = config; charts.set(canvas, this); }
    static getChart(canvas) { return charts.get(canvas); }
    resetZoom() { this.reset = true; }
  }
  vm.runInNewContext(source, {
    document: {
      querySelector: () => root,
      createElement: () => new Element(),
    },
    Chart,
    loadChartJs: () => Promise.resolve(),
    fetch: async url => {
      requests.push(url);
      const isCSV = url.endsWith(".csv");
      return {
        ok: !(isCSV ? missingCSV : missingPulls), status: 404,
        text: async () => text, json: async () => tables,
      };
    },
  });
  await new Promise(resolve => setImmediate(resolve));
  return {elements, requests, chart: id => charts.get(elements.get(id)).config};
}

test("activity matches ONNX Runtime metrics, daily charts and separate age axes", async () => {
  const {elements, requests, chart} = await render();
  assert.deepEqual(requests.sort(), [
    "../../cache_data/onnx-light/open_pulls.json", "../../cache_data/onnx-light/pr_activity.csv",
  ]);
  assert.equal(elements.get("open").textContent, "0");
  assert.equal(elements.get("mergedAge").textContent, "2.5 days");
  assert.equal(elements.get("summary").style.display, "");
  assert.equal(chart("chartOpen").options.scales.x.time.unit, "day");
  assert.equal(chart("chartOpen").data.datasets[0].data[0].y, 3);
  const activity = chart("chartActivity").data.datasets;
  assert.equal(activity.length, 6);
  assert.equal(activity[0].data[1].y, 3);
  assert.equal(activity[1].data[1].y, 2);
  assert.equal(chart("chartAge").data.datasets[1].yAxisID, "yMerged");
  assert.equal(chart("chartAge").data.datasets[1].data[0].y, null);
  assert.equal(chart("chartAge").options.scales.yMerged.position, "right");
  assert.equal(elements.get("observationsBody").children[1].children[6].textContent, "N/A");
  assert.equal(elements.get("updated").dataset.source,
    "../../cache_data/onnx-light/pr_activity.csv,../../cache_data/onnx-light/open_pulls.json");
  const canvas = elements.get("chartAge");
  canvas.listeners.dblclick({currentTarget: canvas});
});

test("open PRs use safe text, repository links and seven-day age buckets", async () => {
  const title = "<img src=x onerror=alert(1)>";
  const {elements, chart} = await render({tables: {
    latest: [{number: 42, title, user: "author", created_at: "2026-10-24T00:00:00Z"}],
    oldest: [], ages_days: [0, 6, 7, 20, 20, null, "bad"],
  }});
  const cells = elements.get("latestPullsBody").firstChild.children;
  assert.equal(cells[0].firstChild.textContent, "#42 " + title);
  assert.equal(cells[0].firstChild.href, "https://github.com/xadupre/onnx-light/pull/42");
  assert.equal(cells[1].textContent, "author");
  assert.equal(cells[2].textContent, "2026-10-24");
  assert.equal(elements.get("oldestPullsBody").children.length, 0);
  assert.deepEqual(Array.from(chart("chartAgeDistribution").data.labels), ["0-6", "7-13", "14-20"]);
  assert.deepEqual(Array.from(chart("chartAgeDistribution").data.datasets[0].data), [2, 1, 2]);
});

test("legacy CSV leaves unknown daily counts and merged ages null", async () => {
  const {elements, chart} = await render({
    text: "date,open_prs,opened_prs_7d,closed_prs_7d,merged_prs_7d,avg_open_age_days\n" +
      "2026-10-24T00:00:00Z,0,0,0,0,0",
  });
  assert.equal(elements.get("mergedAge").textContent, "N/A");
  assert.equal(chart("chartActivity").data.datasets[0].data[0].y, null);
  assert.equal(chart("chartAge").data.datasets[1].data[0].y, null);
  assert.equal(chart("chartAgeDistribution").data.labels.length, 0);
});

test("unavailable or empty activity data do not suppress open PR tables", async () => {
  for (const options of [{missingCSV: true}, {text: header}]) {
    const {elements} = await render(options);
    assert.match(elements.get("message").textContent, /Unable to load PR activity data/);
    assert.equal(elements.get("openPulls").style.display, "");
    assert.equal(elements.get("observationsBody").children.length, 0);
  }
});

test("missing open PR data do not suppress activity charts", async () => {
  const {elements, chart} = await render({missingPulls: true});
  assert.match(elements.get("message").textContent, /Unable to load open pull requests/);
  assert.equal(chart("chartOpen").data.datasets.length, 1);
  assert.equal(elements.get("summary").style.display, "");
});
