const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

const source = fs.readFileSync(
  path.join(__dirname, "../../assets/github-stars.js"), "utf8");

function browser(response) {
  function element() {
    return {
      children: [], textContent: "", hidden: true,
      appendChild(child) { this.children.push(child); },
      setAttribute() {},
    };
  }
  const elements = Object.fromEntries(
    ["message", "updated", "projects", "content"].map(id => [id, element()]));
  const charts = [];
  const context = vm.createContext({
    document: {
      getElementById: id => elements[id],
      createElement: element,
    },
    fetch: async () => response,
    Chart: function (canvas, config) { charts.push(config); },
  });
  vm.runInContext(source, context);
  context.window = {loadChartJs: async () => {}};
  return {context, elements, charts};
}

test("star series keeps zero, decreases and missing observations", () => {
  const {context} = browser({});
  const series = context.starSeries([
    {date: "2026-11-09", stars: {repo: 2}},
    {date: "2026-11-10", stars: {repo: 0}},
    {date: "2026-11-11", stars: {}},
    {date: "2026-11-12", stars: {repo: "3"}},
  ], "repo");
  assert.deepEqual(Array.from(series, point => point.y), [2, 0, null, null]);
  assert.equal(series[0].x, "2026-11-09T00:00:00.000Z");
});

test("missing collection days leave gaps across UTC month and year boundaries", () => {
  const {context} = browser({});
  const series = context.starSeries([
    {date: "2025-12-31", stars: {repo: 2}},
    {date: "2026-01-02", stars: {repo: 3}},
  ], "repo");
  assert.deepEqual(Array.from(series, point => point.y), [2, null, 3]);
  assert.equal(series[1].x, "2026-01-01T00:00:00.000Z");
  assert.equal(context.starSeries([], "repo").length, 0);
});

test("dashboard renders repository links, counts and separate charts", async () => {
  const {context, elements, charts} = browser({
    ok: true, json: async () => ({
      updated_at: "2026-11-12T06:00:00Z",
      history: [
        {date: "2026-11-11", stars: {"owner/old": 4}},
        {date: "2026-11-12", stars: {"onnx/onnx": 0}},
      ],
    }),
  });
  await context.loadStars();
  assert.equal(elements.content.hidden, false);
  assert.equal(elements.message.textContent, "");
  assert.equal(elements.projects.children.length, 2);
  const [current, old] = elements.projects.children;
  assert.equal(current.children[0].children[0].href, "https://github.com/onnx/onnx");
  assert.equal(current.children[1].textContent, "Stars: 0");
  assert.equal(old.children[1].textContent, "Stars: \u2014");
  assert.equal(charts.length, 2);
  assert.equal(charts[0].data.datasets[0].spanGaps, false);
  assert.deepEqual(Array.from(charts[0].data.datasets[0].data, p => p.y), [null, 0]);
});

test("unavailable, failed and empty data show an error without charts", async () => {
  for (const response of [
    {status: 404},
    {status: 503, ok: false},
    {ok: true, json: async () => ({history: []})},
  ]) {
    const {context, elements, charts} = browser(response);
    await context.loadStars();
    assert.equal(elements.message.className, "error");
    assert.ok(elements.message.textContent.length);
    if (response.status === 404) {
      assert.match(elements.message.textContent, /first collection/);
    }
    assert.equal(elements.content.hidden, true);
    assert.equal(charts.length, 0);
  }
});
