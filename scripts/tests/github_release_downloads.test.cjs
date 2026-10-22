const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

const source = fs.readFileSync(
  path.join(__dirname, "../../assets/github-release-downloads.js"), "utf8");
const context = vm.createContext({});
vm.runInContext(source, context);

test("net changes require the exact date and repository baseline", () => {
  const history = [
    {date: "2026-09-21", totals: {repo: 100}},
    {date: "2026-10-20", totals: {repo: 130}},
    {date: "2026-10-21", totals: {repo: 125, newRepo: 4}},
  ];
  assert.equal(context.netChange(history, "repo", "2026-10-21", 1), -5);
  assert.equal(context.netChange(history, "repo", "2026-10-21", 30), 25);
  assert.equal(context.netChange(history, "repo", "2026-10-21", 2), null);
  assert.equal(context.netChange(history, "newRepo", "2026-10-21", 1), null);
  assert.equal(context.netChange([{date: "2026-10-20"}, ...history],
    "repo", "2026-10-21", 1), null);
  assert.equal(context.formatCount(0), "0");
  assert.equal(context.formatCount(null), "\u2014");
});

test("release links encode special tag characters", () => {
  assert.equal(context.releaseLink("onnx/onnx", "v1/preview?#"),
    "https://github.com/onnx/onnx/releases/tag/v1%2Fpreview%3F%23");
});

test("rolling periods cross UTC month and year boundaries without clamping changes", () => {
  const history = [
    {date: "2025-12-02", totals: {repo: 5}},
    {date: "2025-12-25", totals: {repo: 10}},
    {date: "2025-12-31", totals: {repo: 12}},
    {date: "2026-01-01", totals: {repo: 10}},
    {date: "2026-02-28", totals: {repo: 20}},
    {date: "2026-03-01", totals: {repo: 22}},
  ];
  assert.equal(context.netChange(history, "repo", "2026-01-01", 1), -2);
  assert.equal(context.netChange(history, "repo", "2026-01-01", 7), 0);
  assert.equal(context.netChange(history, "repo", "2026-01-01", 30), 5);
  assert.equal(context.netChange(history, "repo", "2026-03-01", 1), 2);
});

test("per-file history preserves zero, missing observations and legacy totals", () => {
  const data = {
    updated_at: "2026-10-21T06:00:00Z",
    history: [
      {date: "2026-10-18", totals: {repo: 100}},
      {date: "2026-10-19", assets: {repo: {"1": 0}}},
      {date: "2026-10-20", assets: {repo: {"1": 4}}},
      {date: "2026-10-21", assets: {repo: {"1": 6}}},
    ],
  };
  assert.deepEqual(Array.from(context.assetHistory(data, "repo",
    {id: 1, download_count: 6})), [null, 0, 4, 6]);
  assert.deepEqual(Array.from(context.assetHistory(data, "repo",
    {id: 2, download_count: 12})), [null, null, null, 12]);
});

test("owner scope excludes other ONNX projects and includes both requested families", () => {
  for (const repo of [
    "xadupre/onnx-light", "xadupre/onnx-light-cpu", "xadupre/yet-another-tool",
    "onnx/onnx", "microsoft/onnxruntime-genai", "microsoft/onnxruntime-extensions",
    "ggml-org/llama.cpp", "vllm-project/vllm",
  ]) assert.equal(context.isTrackedProject(repo), true, repo);
  for (const repo of [
    "xadupre/onnx-asv-benchmark", "xadupre/onnxruntime", "xadupre/onnx-diagnostic",
    "xadupre/onnx-lightish",
  ]) assert.equal(context.isTrackedProject(repo), false, repo);
});

test("projects omit aggregate totals and preserve period and selectable file curves", () => {
  class Element {
    constructor() {
      this.children = [];
      this.value = "";
      this.textContent = "";
      this.hidden = true;
      this.listeners = {};
    }
    appendChild(child) {
      this.children.push(child);
      if (!this.value && child.value) this.value = child.value;
      return child;
    }
    replaceChildren() { this.children = []; }
    addEventListener(name, fn) { this.listeners[name] = fn; }
    setAttribute(name, value) { this[name] = value; }
  }
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement() { return new Element(); },
  };
  const charts = [];
  const periodCharts = [];
  context.document = document;
  context.Chart = class {
    constructor(canvas, config) {
      assert.equal(document.getElementById("content").hidden, false);
      this.data = config.data;
      this.options = config.options;
      this.updates = 0;
      if (config.data.datasets.length === 0) charts.push(this);
      else periodCharts.push(this);
    }
    resize() {}
    update() { this.updates++; }
  };
  const data = {
    updated_at: "2026-10-21T06:00:00Z",
    repositories: [
      {repository: "xadupre/onnx-asv-benchmark", total: 999, releases: []},
      {repository: "xadupre/onnx-light-empty", total: 0, releases: []},
      {repository: "onnx/onnx", total: 17, releases: [{
        tag: "v0", total: 5, prerelease: false,
        assets: [{id: 1, name: "<script>alert(1)</script>.whl", download_count: 5}],
      }, {
        tag: "v1", total: 12, prerelease: true,
        assets: [{id: 2, name: "<script>alert(1)</script>.whl", download_count: 12}],
      }]},
    ],
    history: [
      {date: "2026-09-21", totals: {"onnx/onnx": 5}},
      {date: "2026-10-14", totals: {"onnx/onnx": 8}},
      {date: "2026-10-20", totals: {"onnx/onnx": 10}},
      {date: "2026-10-21", totals: {"onnx/onnx": 17, "xadupre/onnx-light-empty": 0}},
    ],
  };
  context.renderReleaseDownloads(data);
  assert.equal(document.getElementById("content").hidden, false);
  assert.equal(document.getElementById("totals").children.length, 2);
  const totalCells = document.getElementById("totals").children[0].children;
  assert.deepEqual(totalCells.slice(1).map(cell => cell.textContent), ["7", "9", "12"]);
  assert.deepEqual(document.getElementById("totals").children[1].children.slice(1)
    .map(cell => cell.textContent), ["\u2014", "\u2014", "\u2014"]);
  assert.equal(charts.length, 2);
  assert.equal(periodCharts.length, 1);
  assert.deepEqual(Array.from(periodCharts[0].data.datasets, series => series.label),
    ["Daily (1 day)", "Weekly (7 days)", "Monthly (30 days)"]);
  for (const [index, change] of [7, 9, 12].entries()) {
    assert.deepEqual(Array.from(periodCharts[0].data.datasets[index].data),
      [null, null, null, change]);
  }
  assert.equal(periodCharts[0].data.datasets[0].spanGaps, false);
  assert.equal(periodCharts[0].data.datasets[0].yAxisID, "y1");
  assert.equal(periodCharts[0].data.datasets[1].yAxisID, "y");
  assert.equal(periodCharts[0].data.datasets[2].yAxisID, "y");
  assert.equal(periodCharts[0].options.scales.y1.position, "right");
  assert.equal(charts[0].data.datasets.length, 0);
  assert.equal(charts[1].data.datasets.length, 0);
  const projects = document.getElementById("projects").children;
  assert.equal(projects.length, 2);
  assert.equal(projects[0].children[0].textContent, "onnx/onnx");
  assert.equal(projects[0].children[1].hidden, true);
  const releases = projects[0].children[3];
  assert.equal(releases.children.length, 2);
  releases.open = true;
  releases.listeners.toggle();
  const details = releases.children[2];
  assert.equal(details.children[0].textContent, "v1 (prerelease)");
  assert.equal(releases.children[3].children[0].textContent, "v0");
  const assetRows = details.children[3].children[0].children[1];
  assert.equal(assetRows.children.length, 0);
  details.open = true;
  details.listeners.toggle();
  const assetCell = assetRows.children[0].children[0];
  assert.equal(assetCell.textContent, "<script>alert(1)</script>.whl");
  const fileList = projects[0].children[2].children[1];
  assert.equal(fileList.children.length, 0);
  projects[0].children[2].open = true;
  projects[0].children[2].listeners.toggle();
  const first = fileList.children[0].children[0];
  const second = fileList.children[1].children[0];
  assert.equal(Boolean(first.checked), false);
  first.checked = true;
  first.listeners.change();
  assert.equal(projects[0].children[1].hidden, false);
  assert.equal(charts[0].data.datasets.length, 1);
  assert.equal(charts[0].data.datasets[0].label, "<script>alert(1)</script>.whl (v1)");
  assert.deepEqual(Array.from(charts[0].data.datasets[0].data), [null, null, null, 12]);
  assert.equal(charts[1].data.datasets.length, 0);
  second.checked = true;
  second.listeners.change();
  assert.equal(charts[0].data.datasets.length, 2);
  first.checked = false;
  first.listeners.change();
  assert.equal(charts[0].data.datasets.length, 1);
  assert.equal(charts[0].data.datasets[0].assetId, 1);
  assert.equal(charts[0].updates, 3);
  second.checked = false;
  second.listeners.change();
  assert.equal(charts[0].data.datasets.length, 0);
  assert.equal(projects[0].children[1].hidden, true);
  assert.equal(projects[1].children[1].hidden, true);
  assert.equal(projects[1].children[2].children[1].textContent, "No uploaded files.");
  assert.equal(projects[1].children[4].children[0].hidden, true);
  assert.match(projects[1].children[4].children[1].textContent, /Not enough daily history/);

  data.history = [data.history.at(-1)];
  document.getElementById("totals").replaceChildren();
  document.getElementById("projects").replaceChildren();
  periodCharts.length = 0;
  context.renderReleaseDownloads(data);
  assert.equal(periodCharts.length, 0);
  for (const project of document.getElementById("projects").children) {
    assert.match(project.children[4].children[1].textContent, /Not enough daily history/);
  }
  assert.deepEqual(document.getElementById("totals").children[0].children.slice(1)
    .map(cell => cell.textContent), ["\u2014", "\u2014", "\u2014"]);
});

test("large lists render only on expansion and in bounded batches without duplicates", () => {
  const details = context.document.createElement("details");
  const rendered = [];
  context.renderBatches(details, function* () {
    for (let i = 0; i < 250; i++) yield i;
  }, item => rendered.push(item));
  const more = details.children[0];
  assert.equal(rendered.length, 0);
  assert.equal(more.hidden, true);
  details.listeners.toggle();
  assert.equal(rendered.length, 0);
  details.open = true;
  details.listeners.toggle();
  assert.equal(rendered.length, 100);
  assert.equal(more.hidden, false);
  details.open = false;
  details.listeners.toggle();
  details.open = true;
  details.listeners.toggle();
  assert.equal(rendered.length, 100);
  more.listeners.click();
  assert.equal(rendered.length, 200);
  assert.equal(more.hidden, false);
  more.listeners.click();
  assert.deepEqual(rendered, Array.from({length: 250}, (_, i) => i));
  assert.equal(more.hidden, true);
});

test("empty and exactly full batches hide the continuation button", () => {
  for (const length of [0, 100]) {
    const details = context.document.createElement("details");
    const rendered = [];
    context.renderBatches(details, function* () {
      for (let i = 0; i < length; i++) yield i;
    }, item => rendered.push(item));
    details.open = true;
    details.listeners.toggle();
    assert.equal(rendered.length, length);
    assert.equal(details.children[0].hidden, true);
  }
});

test("summary headers show net changes without total downloads", () => {
  const page = fs.readFileSync(
    path.join(__dirname, "../../dashboard/github-release-downloads.html"), "utf8");
  const headers = Array.from(page.matchAll(/<th>(.*?)<\/th>/g), match => match[1]);
  assert.deepEqual(headers,
    ["Repository", "1-day net change", "7-day net change", "30-day net change"]);
});

test("missing data and network failures stay visible", async () => {
  let message;
  context.document = {getElementById() { return message; }};
  for (const failure of [
    async () => ({ok: false, status: 404}),
    async () => { throw new Error("network offline"); },
  ]) {
    message = {};
    context.fetch = failure;
    await context.loadReleaseDownloads();
    assert.equal(message.className, "error");
    assert.match(message.textContent, /Unable to display release downloads/);
  }
});
