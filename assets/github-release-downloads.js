"use strict";

function netChange(history, repository, day, days) {
  const date = new Date(day + "T00:00:00Z");
  date.setUTCDate(date.getUTCDate() - days);
  const baseline = history.find(row => row.date === date.toISOString().slice(0, 10));
  const current = history.find(row => row.date === day);
  if (!baseline || !current ||
      !Object.hasOwn(baseline.totals, repository) ||
      !Object.hasOwn(current.totals, repository)) return null;
  return current.totals[repository] - baseline.totals[repository];
}

function releaseLink(repository, tag) {
  return "https://github.com/" + repository.split("/").map(encodeURIComponent).join("/") +
    "/releases/tag/" + encodeURIComponent(tag);
}

function cell(row, text) {
  const element = document.createElement("td");
  element.textContent = text;
  row.appendChild(element);
  return element;
}

function formatCount(value) {
  return value === null ? "\u2014" : value.toLocaleString("en-US");
}

function renderReleaseDownloads(data) {
  if (!Array.isArray(data.repositories) || !data.repositories.length ||
      !Array.isArray(data.history) || !data.history.length) {
    throw new Error("Release download data are empty or invalid.");
  }
  const day = data.updated_at.slice(0, 10);
  document.getElementById("updated").textContent = "Collected: " + data.updated_at;
  const totals = document.getElementById("totals");
  const select = document.getElementById("repository");
  const repositories = [...data.repositories].sort((a, b) => b.total - a.total);
  for (const item of repositories) {
    const row = document.createElement("tr");
    const link = document.createElement("a");
    link.href = "https://github.com/" +
      item.repository.split("/").map(encodeURIComponent).join("/");
    link.textContent = item.repository;
    cell(row, "").appendChild(link);
    cell(row, formatCount(item.total));
    cell(row, formatCount(netChange(data.history, item.repository, day, 1)));
    cell(row, formatCount(netChange(data.history, item.repository, day, 30)));
    totals.appendChild(row);
    const option = document.createElement("option");
    option.value = item.repository;
    option.textContent = item.repository;
    select.appendChild(option);
  }
  let chart;
  function renderSelected() {
    const item = repositories.find(row => row.repository === select.value);
    const container = document.getElementById("releases");
    container.replaceChildren();
    if (!item.releases.length) {
      container.textContent = "No published releases.";
    }
    for (const release of [...item.releases].reverse()) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = release.tag + (release.prerelease ? " (prerelease)" : "") +
        ": " + formatCount(release.total) + " downloads";
      details.appendChild(summary);
      const link = document.createElement("a");
      link.href = releaseLink(item.repository, release.tag);
      link.textContent = "View release on GitHub";
      details.appendChild(link);
      if (!release.assets.length) {
        const empty = document.createElement("p");
        empty.textContent = "No uploaded files.";
        details.appendChild(empty);
      } else {
        const wrap = document.createElement("div");
        wrap.className = "table-wrap";
        const table = document.createElement("table");
        const head = document.createElement("thead");
        const header = document.createElement("tr");
        for (const title of ["File", "Downloads"]) {
          const th = document.createElement("th");
          th.textContent = title;
          header.appendChild(th);
        }
        head.appendChild(header);
        table.appendChild(head);
        const body = document.createElement("tbody");
        for (const asset of release.assets) {
          const row = document.createElement("tr");
          cell(row, asset.name);
          cell(row, formatCount(asset.download_count));
          body.appendChild(row);
        }
        table.appendChild(body);
        wrap.appendChild(table);
        details.appendChild(wrap);
      }
      container.appendChild(details);
    }
    if (chart) chart.destroy();
    chart = new Chart(document.getElementById("history"), {
      type: "line",
      data: {
        labels: data.history.map(row => row.date),
        datasets: [{
          label: item.repository,
          data: data.history.map(row => row.totals[item.repository] ?? null),
          borderColor: "#58a6ff",
          backgroundColor: "#58a6ff",
          spanGaps: false,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, title: { display: true, text: "Downloads" } } },
      },
    });
  }
  select.addEventListener("change", renderSelected);
  document.getElementById("content").hidden = false;
  renderSelected();
  document.getElementById("message").textContent = "";
}

async function loadReleaseDownloads() {
  try {
    const response = await fetch("../cache_data/github-release-downloads.json");
    if (!response.ok) {
      throw new Error("HTTP " + response.status +
        ". The daily DATA GitHub release downloads workflow must publish its first collection.");
    }
    const data = await response.json();
    await loadChartJs();
    renderReleaseDownloads(data);
  } catch (error) {
    const message = document.getElementById("message");
    message.textContent = "Unable to display release downloads: " + error.message;
    message.className = "error";
  }
}

if (typeof document !== "undefined") loadReleaseDownloads();
