/* Shared review metrics for the merged-PR statistics dashboards. */
(function (root) {
  "use strict";

  const fields = [
    "review_agent_comments", "review_agent_high", "review_agent_medium",
    "review_agent_low", "review_agent_unclassified", "review_agent_security",
    "review_agent_resolved_threads", "review_agent_unresolved_threads",
    "additions", "deletions", "changed_files",
  ];

  function parseValues(header, columns) {
    return Object.fromEntries(fields.map(key => {
      const raw = columns[header.indexOf(key)];
      const number = raw == null || raw.trim() === "" ? NaN : Number(raw);
      return [key, Number.isFinite(number) && number >= 0 ? number : null];
    }));
  }

  function value(number) {
    return number == null ? "\u2014" : String(number);
  }

  function tableValues(row) {
    return [
      value(row.review_agent_comments),
      ["high", "medium", "low", "unclassified"].map(
        key => value(row["review_agent_" + key])
      ).join(" / "),
      value(row.review_agent_security),
      value(row.review_agent_resolved_threads) + " / " + value(row.review_agent_unresolved_threads),
      value(row.additions) + " / " + value(row.deletions),
      value(row.changed_files),
    ];
  }

  function summary(rows, keys, average) {
    const known = rows.filter(row => keys.every(key => row[key] != null));
    if (!known.length) return "\u2014 (0/" + rows.length + " PRs)";
    const totals = keys.map(key => {
      const total = known.reduce((sum, row) => sum + row[key], 0);
      return average ? (total / known.length).toFixed(1) : String(total);
    });
    return totals.join(" / ") + " (" + known.length + "/" + rows.length + " PRs)";
  }

  function render(rows) {
    const grid = document.querySelector("#summary .grid");
    const metrics = [
      ["inline review-agent comments", ["review_agent_comments"]],
      ["high / medium / low / unclassified", [
        "review_agent_high", "review_agent_medium", "review_agent_low", "review_agent_unclassified",
      ]],
      ["agent comments mentioning security", ["review_agent_security"]],
      ["agent threads: resolved / unresolved", [
        "review_agent_resolved_threads", "review_agent_unresolved_threads",
      ]],
      ["avg added / deleted lines per PR", ["additions", "deletions"], true],
      ["avg commits per PR", ["commits"], true],
    ];
    for (const [label, keys, average] of metrics) {
      const cell = document.createElement("div");
      cell.className = "cell";
      for (const [className, text] of [["k", label], ["v", summary(rows, keys, average)]]) {
        const element = document.createElement("div");
        element.className = className;
        element.textContent = text;
        cell.appendChild(element);
      }
      grid.appendChild(cell);
    }
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = "Review agents: Copilot and recognized review bots (CodeRabbit, Gemini, Codex, Cursor, Greptile, Sourcery, Claude, Qodo). " +
      "Counts cover inline review comments, not conversation comments or review summaries. " +
      "Severity is inferred from explicit priority/severity labels (P0/P1 = high, P2 = medium, P3 = low); unlabeled comments remain unclassified. " +
      "Security counts are keyword matches and may overlap severity. " +
      "Resolved/unresolved counts cover threads started by agents at collection time, not proof of addressed/ignored feedback. " +
      "\u2014 means unavailable, not zero; parentheses show metric coverage. Historical rows can be refreshed using the collector's --since option.";
    document.getElementById("summary").appendChild(note);

    const header = document.querySelector("#lastObservations thead tr");
    for (const label of [
      "agent comments", "high / medium / low / unclassified", "security mentions",
      "resolved / unresolved threads", "added / deleted lines", "changed files",
    ]) {
      const th = document.createElement("th");
      th.textContent = label;
      header.appendChild(th);
    }
  }

  const api = { parseValues, tableValues, summary, render };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PrReviewStats = api;
})(typeof window !== "undefined" ? window : globalThis);
