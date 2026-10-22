"use strict";

function netChange(history, repository, day, days) {
  const date = new Date(day + "T00:00:00Z");
  date.setUTCDate(date.getUTCDate() - days);
  const baselineDay = date.toISOString().slice(0, 10);
  const baseline = history instanceof Map ? history.get(baselineDay) :
    history.find(row => row.date === baselineDay);
  const current = history instanceof Map ? history.get(day) :
    history.find(row => row.date === day);
  if (!baseline || !current ||
      !Object.hasOwn(baseline.totals ?? {}, repository) ||
      !Object.hasOwn(current.totals ?? {}, repository)) return null;
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

function isTrackedProject(repository) {
  if (!repository.startsWith("xadupre/")) return true;
  const name = repository.slice("xadupre/".length);
  return name === "onnx-light" || name.startsWith("onnx-light-") ||
    name.startsWith("yet-another-");
}

function assetHistory(data, repository, asset) {
  return data.history.map(row => {
    const counts = row.assets?.[repository];
    if (counts && Object.hasOwn(counts, String(asset.id))) return counts[asset.id];
    // The latest snapshot also supplies a real observation for legacy caches.
    if (row.date === data.updated_at.slice(0, 10)) return asset.download_count;
    return null;
  });
}

function renderReleaseDownloads(data) {
  if (!Array.isArray(data.repositories) || !data.repositories.length ||
      !Array.isArray(data.history) || !data.history.length) {
    throw new Error("Release download data are empty or invalid.");
  }
  const day = data.updated_at.slice(0, 10);
  document.getElementById("updated").textContent = "Collected: " + data.updated_at;
  const totals = document.getElementById("totals");
  const projects = document.getElementById("projects");
  const repositories = data.repositories.filter(item => isTrackedProject(item.repository))
    .sort((a, b) => b.total - a.total);
  const historyByDate = new Map(data.history.map(row => [row.date, row]));
  document.getElementById("content").hidden = false;
  for (const item of repositories) {
    const row = document.createElement("tr");
    const link = document.createElement("a");
    link.href = "https://github.com/" +
      item.repository.split("/").map(encodeURIComponent).join("/");
    link.textContent = item.repository;
    cell(row, "").appendChild(link);
    cell(row, formatCount(netChange(data.history, item.repository, day, 1)));
    cell(row, formatCount(netChange(data.history, item.repository, day, 7)));
    cell(row, formatCount(netChange(data.history, item.repository, day, 30)));
    totals.appendChild(row);
    const project = document.createElement("section");
    project.className = "project";
    const title = document.createElement("h2");
    title.textContent = item.repository;
    project.appendChild(title);
    const chartWrap = document.createElement("div");
    chartWrap.className = "chart-wrap";
    chartWrap.hidden = true;
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", item.repository + " per-file cumulative release downloads");
    chartWrap.appendChild(canvas);
    project.appendChild(chartWrap);
    projects.appendChild(project);
    const chart = new Chart(canvas, {
      type: "line",
      data: {
        labels: data.history.map(row => row.date),
        datasets: [],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, title: { display: true, text: "Downloads" } } },
      },
    });
    const files = document.createElement("details");
    const fileSummary = document.createElement("summary");
    fileSummary.textContent = "Files: check to add individual curves";
    files.appendChild(fileSummary);
    const fileList = document.createElement("div");
    fileList.className = "files";
    files.appendChild(fileList);
    project.appendChild(files);
    let fileIndex = 0;
    for (const release of [...item.releases].reverse()) {
      for (const asset of release.assets) {
        const label = document.createElement("label");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        label.appendChild(checkbox);
        const name = document.createElement("span");
        const caption = asset.name + " (" + release.tag + ")";
        name.textContent = " " + caption + ": " + formatCount(asset.download_count);
        label.appendChild(name);
        fileList.appendChild(label);
        const color = "hsl(" + (fileIndex++ * 137.5 % 360) + ", 75%, 65%)";
        checkbox.addEventListener("change", () => {
          if (checkbox.checked) {
            chart.data.datasets.push({
              assetId: asset.id,
              label: caption,
              data: assetHistory(data, item.repository, asset),
              borderColor: color,
              backgroundColor: color,
              spanGaps: false,
            });
          } else {
            chart.data.datasets = chart.data.datasets.filter(series => series.assetId !== asset.id);
          }
          chartWrap.hidden = chart.data.datasets.length === 0;
          if (!chartWrap.hidden) chart.resize();
          chart.update();
        });
      }
    }
    if (!fileList.children.length) fileList.textContent = "No uploaded files.";
    const container = document.createElement("details");
    const releaseSummary = document.createElement("summary");
    releaseSummary.textContent = "Releases and download counts";
    container.appendChild(releaseSummary);
    project.appendChild(container);
    if (!item.releases.length) {
      const empty = document.createElement("p");
      empty.textContent = "No published releases.";
      container.appendChild(empty);
    }
    for (const release of [...item.releases].reverse()) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = release.tag + (release.prerelease ? " (prerelease)" : "");
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
    const periodWrap = document.createElement("div");
    periodWrap.className = "chart-wrap";
    const periodCanvas = document.createElement("canvas");
    periodCanvas.setAttribute("aria-label", item.repository + " daily, weekly and monthly release downloads");
    periodWrap.appendChild(periodCanvas);
    project.appendChild(periodWrap);
    new Chart(periodCanvas, {
      type: "line",
      data: {
        labels: data.history.map(row => row.date),
        datasets: [
          {days: 1, label: "Daily (1 day)", color: "#58a6ff", yAxisID: "y1"},
          {days: 7, label: "Weekly (7 days)", color: "#f78166", yAxisID: "y"},
          {days: 30, label: "Monthly (30 days)", color: "#3fb950", yAxisID: "y"},
        ].map(metric => ({
          label: metric.label,
          data: data.history.map(row =>
            netChange(historyByDate, item.repository, row.date, metric.days)),
          borderColor: metric.color,
          backgroundColor: metric.color,
          yAxisID: metric.yAxisID,
          spanGaps: false,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, title: { display: true, text: "Week / month net downloads" } },
          y1: {
            position: "right",
            beginAtZero: true,
            title: { display: true, text: "Daily net downloads" },
            grid: { drawOnChartArea: false },
          },
        },
      },
    });
  }
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
