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
  assert.equal(context.formatCount(0), "0");
  assert.equal(context.formatCount(null), "\u2014");
});

test("release links encode special tag characters", () => {
  assert.equal(context.releaseLink("onnx/onnx", "v1/preview?#"),
    "https://github.com/onnx/onnx/releases/tag/v1%2Fpreview%3F%23");
});

test("dashboard renders totals, asset names, selection and chart without HTML injection", () => {
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
  }
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement() { return new Element(); },
  };
  let chartConfig;
  let destroyed = 0;
  context.document = document;
  context.Chart = class {
    constructor(canvas, config) { chartConfig = config; }
    destroy() { destroyed++; }
  };
  const data = {
    updated_at: "2026-10-21T06:00:00Z",
    repositories: [
      {repository: "xadupre/onnx-empty", total: 0, releases: []},
      {repository: "onnx/onnx", total: 12, releases: [{
        tag: "v1", total: 12, prerelease: true,
        assets: [{name: "<script>alert(1)</script>.whl", download_count: 12}],
      }]},
    ],
    history: [
      {date: "2026-10-20", totals: {"onnx/onnx": 10}},
      {date: "2026-10-21", totals: {"onnx/onnx": 12, "xadupre/onnx-empty": 0}},
    ],
  };
  context.renderReleaseDownloads(data);
  assert.equal(document.getElementById("content").hidden, false);
  assert.equal(document.getElementById("totals").children.length, 2);
  assert.equal(document.getElementById("repository").value, "onnx/onnx");
  assert.equal(chartConfig.data.datasets[0].data[0], 10);
  const details = document.getElementById("releases").children[0];
  const assetCell = details.children[2].children[0].children[1].children[0].children[0];
  assert.equal(assetCell.textContent, "<script>alert(1)</script>.whl");
  document.getElementById("repository").value = "xadupre/onnx-empty";
  document.getElementById("repository").listeners.change();
  assert.equal(destroyed, 1);
  assert.equal(document.getElementById("releases").textContent, "No published releases.");
  assert.equal(chartConfig.data.datasets[0].data[0], null);
  assert.equal(chartConfig.data.datasets[0].data[1], 0);
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
