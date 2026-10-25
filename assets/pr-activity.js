(function () {
  "use strict";

  const root = document.querySelector("[data-pr-activity]");
  if (!root) return;
  const repository = root.dataset.prActivity;
  const project = repository.split("/")[1];
  const cache = "../../cache_data/" + project + "/";
  const pullURL = "https://github.com/" + repository + "/pull/";
  const element = id => root.querySelector('[data-id="' + id + '"]');

  root.innerHTML = `
    <p class="description">Daily pull request snapshots. Opened, closed, and merged
      counts cover the preceding seven days; open PR age is measured from creation
      to the snapshot. Merged PR age is measured from creation to merge over a
      rolling year (365 days). N/A means no valid yearly merged ages were recorded.</p>
    <div data-id="message" class="message"></div>
    <div data-id="summary" class="activity-summary" style="display:none;">
      <div class="metric"><div class="label">open PRs</div><div data-id="open" class="value"></div></div>
      <div class="metric"><div class="label">opened in preceding 7 days</div><div data-id="opened" class="value"></div></div>
      <div class="metric"><div class="label">closed in preceding 7 days</div><div data-id="closed" class="value"></div></div>
      <div class="metric"><div class="label">merged in preceding 7 days</div><div data-id="merged" class="value"></div></div>
      <div class="metric"><div class="label">average age of open PRs</div><div data-id="age" class="value"></div></div>
      <div class="metric"><div class="label">average age of PRs merged in preceding 365 days</div><div data-id="mergedAge" class="value"></div></div>
    </div>
    <div class="chart-wrap">
      <h2 class="chart-title">open pull requests</h2>
      <div class="chart-container"><canvas data-id="chartOpen"></canvas></div>
      <div class="hint">Drag to pan, mouse wheel or pinch to zoom. Double-click to reset.</div>
    </div>
    <div class="chart-wrap">
      <h2 class="chart-title">daily pull request activity</h2>
      <div class="chart-container"><canvas data-id="chartActivity"></canvas></div>
      <div class="hint">Daily opened, closed, and merged pull requests with their 7-day moving averages.</div>
    </div>
    <div class="chart-wrap">
      <h2 class="chart-title">average age of open and merged pull requests</h2>
      <div class="chart-container"><canvas data-id="chartAge"></canvas></div>
      <div class="hint">Age in days; merged PRs cover the preceding 365 days.</div>
    </div>
    <div class="chart-wrap">
      <h2 class="chart-title">distribution of open pull request ages</h2>
      <div class="chart-container"><canvas data-id="chartAgeDistribution"></canvas></div>
      <div class="hint">Number of currently open pull requests by age in days.</div>
    </div>
    <div data-id="observations" class="observations" style="display:none;">
      <h2>Latest snapshots</h2>
      <table>
        <thead><tr><th>date</th><th>open PRs</th><th>opened (7d)</th><th>closed (7d)</th><th>merged (7d)</th><th>average open age (days)</th><th>average merged age (365d, days)</th></tr></thead>
        <tbody data-id="observationsBody"></tbody>
      </table>
    </div>
    <div data-id="openPulls" class="observations" style="display:none;">
      <h2>10 latest open pull requests</h2>
      <table><thead><tr><th>pull request</th><th>author</th><th>opened</th></tr></thead><tbody data-id="latestPullsBody"></tbody></table>
      <h2>10 oldest open pull requests</h2>
      <table><thead><tr><th>pull request</th><th>author</th><th>opened</th></tr></thead><tbody data-id="oldestPullsBody"></tbody></table>
    </div>
    <footer data-id="updated" class="data-updated" style="display:none;"></footer>`;
  element("updated").dataset.source = cache + "pr_activity.csv," + cache + "open_pulls.json";

  function showError(text) {
    element("message").textContent = text;
    element("message").className = "message visible error";
  }

  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    if (lines.length < 2) return [];
    const header = lines.shift().split(",");
    const fields = {
      open: "open_prs", opened: "opened_prs_7d", closed: "closed_prs_7d",
      merged: "merged_prs_7d", openedDay: "opened_prs_1d",
      closedDay: "closed_prs_1d", mergedDay: "merged_prs_1d",
      age: "avg_open_age_days", mergedAge: "avg_merged_age_days_365d",
    };
    return lines.map(line => {
      const columns = line.split(",");
      const row = { date: columns[header.indexOf("date")] };
      for (const [key, name] of Object.entries(fields)) {
        const value = columns[header.indexOf(name)];
        row[key] = value === undefined || value === "" ? null : Number(value);
      }
      return row;
    }).filter(row => row.date && ["open", "opened", "closed", "merged", "age"]
      .every(key => Number.isFinite(row[key])))
      .sort((a, b) => new Date(a.date) - new Date(b.date));
  }

  const commonOptions = {
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: "nearest", intersect: false },
    plugins: {
      legend: { labels: { color: "#c9d1d9" } },
      zoom: {
        pan: { enabled: true, mode: "xy" },
        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "xy" },
      },
    },
    scales: {
      x: {
        type: "time", time: { unit: "day", tooltipFormat: "yyyy-MM-dd" },
        ticks: { color: "#c9d1d9" }, grid: { color: "rgba(139,148,158,0.25)" },
      },
      y: {
        beginAtZero: true, ticks: { color: "#c9d1d9" },
        grid: { color: "rgba(139,148,158,0.25)" },
      },
    },
  };

  function appendCells(body, values) {
    const tr = document.createElement("tr");
    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    body.appendChild(tr);
    return tr;
  }

  function render(rows) {
    const latest = rows[rows.length - 1];
    for (const key of ["open", "opened", "closed", "merged"]) {
      element(key).textContent = latest[key].toLocaleString();
    }
    for (const key of ["age", "mergedAge"]) {
      element(key).textContent = latest[key] === null ? "N/A" : latest[key].toFixed(1) + " days";
    }
    element("summary").style.display = "";
    const points = key => rows.map(row => ({ x: row.date, y: row[key] }));
    new Chart(element("chartOpen"), {
      type: "line",
      data: { datasets: [{ label: "open PRs", data: points("open"), borderColor: "#58a6ff", tension: 0.15 }] },
      options: commonOptions,
    });
    const datasets = [];
    for (const [key, color, background] of [
      ["opened", "#d2a8ff", "rgba(210,168,255,0.35)"],
      ["closed", "#f85149", "rgba(248,81,73,0.35)"],
      ["merged", "#3fb950", "rgba(63,185,80,0.35)"],
    ]) {
      datasets.push({ type: "bar", label: key + " per day", data: points(key + "Day"), backgroundColor: background });
      datasets.push({
        label: key + " 7-day moving average",
        data: rows.map(row => ({ x: row.date, y: row[key] / 7 })),
        borderColor: color, backgroundColor: color, tension: 0.15,
      });
    }
    new Chart(element("chartActivity"), { type: "line", data: { datasets }, options: commonOptions });
    new Chart(element("chartAge"), {
      type: "line",
      data: { datasets: [
        { label: "average open age (days)", data: points("age"), yAxisID: "y", borderColor: "#d2a8ff", tension: 0.15 },
        { label: "average merged age (365d, days)", data: points("mergedAge"), yAxisID: "yMerged", borderColor: "#3fb950", tension: 0.15 },
      ] },
      options: {
        ...commonOptions,
        scales: {
          ...commonOptions.scales,
          y: {
            ...commonOptions.scales.y, position: "left",
            title: { display: true, text: "average open age (days)", color: "#d2a8ff" },
          },
          yMerged: {
            ...commonOptions.scales.y, position: "right",
            title: { display: true, text: "average merged age (365d, days)", color: "#3fb950" },
            grid: { drawOnChartArea: false },
          },
        },
      },
    });
    for (const row of rows.slice(-12).reverse()) {
      appendCells(element("observationsBody"), [
        row.date.slice(0, 10), ...["open", "opened", "closed", "merged"].map(key => row[key].toLocaleString()),
        row.age.toFixed(2), row.mergedAge === null ? "N/A" : row.mergedAge.toFixed(2),
      ]);
    }
    element("observations").style.display = "";
  }

  function renderOpenPulls(tables) {
    for (const name of ["latest", "oldest"]) {
      for (const pull of tables[name] || []) {
        const tr = appendCells(element(name + "PullsBody"), ["", pull.user, (pull.created_at || "").slice(0, 10)]);
        const link = document.createElement("a");
        link.href = pullURL + encodeURIComponent(pull.number);
        link.textContent = "#" + pull.number + " " + pull.title;
        tr.firstChild.appendChild(link);
      }
    }
    const buckets = new Map();
    for (const age of (tables.ages_days || []).filter(Number.isFinite)) {
      const start = Math.floor(age / 7) * 7;
      buckets.set(start, (buckets.get(start) || 0) + 1);
    }
    const starts = [...buckets.keys()].sort((a, b) => a - b);
    new Chart(element("chartAgeDistribution"), {
      type: "bar",
      data: {
        labels: starts.map(start => `${start}-${start + 6}`),
        datasets: [{ label: "open PRs", data: starts.map(start => buckets.get(start)), backgroundColor: "#d2a8ff" }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#c9d1d9" } } },
        scales: {
          x: { ticks: { color: "#c9d1d9" }, title: { display: true, text: "age (days)", color: "#c9d1d9" } },
          y: { beginAtZero: true, ticks: { color: "#c9d1d9", precision: 0 }, title: { display: true, text: "open PRs", color: "#c9d1d9" } },
        },
      },
    });
    element("openPulls").style.display = "";
  }

  for (const id of ["chartOpen", "chartActivity", "chartAge"]) {
    element(id).addEventListener("dblclick", event => {
      const chart = Chart.getChart(event.currentTarget);
      if (chart) chart.resetZoom();
    });
  }
  loadChartJs()
    .then(() => fetch(cache + "pr_activity.csv", { cache: "no-store" }))
    .then(response => {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.text();
    })
    .then(text => {
      const rows = parseCSV(text);
      if (!rows.length) throw new Error("No PR activity snapshots are available.");
      render(rows);
    })
    .catch(error => showError("Unable to load PR activity data: " + error.message));

  fetch(cache + "open_pulls.json", { cache: "no-store" })
    .then(response => {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    })
    .then(tables => loadChartJs().then(() => renderOpenPulls(tables)))
    .catch(error => showError("Unable to load open pull requests: " + error.message));
})();
