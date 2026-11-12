function starSeries(history, repository) {
  if (!history.length) return [];
  var observations = new Map(history.map(function (row) {
    return [row.date, row.stars || {}];
  }));
  var series = [];
  var end = Date.parse(history[history.length - 1].date + "T00:00:00Z");
  for (var day = Date.parse(history[0].date + "T00:00:00Z"); day <= end; day += 86400000) {
    var date = new Date(day).toISOString();
    var count = (observations.get(date.slice(0, 10)) || {})[repository];
    series.push({
      x: date,
      y: Number.isInteger(count) && count >= 0 ? count : null,
    });
  }
  return series;
}

async function loadStars() {
  var message = document.getElementById("message");
  try {
    var response = await fetch("../cache_data/github-stars.json");
    if (response.status === 404) {
      throw new Error("Star data are not yet available. Run the DATA GitHub stars workflow for the first collection.");
    }
    if (!response.ok) throw new Error("Unable to load star counts: HTTP " + response.status);
    var data = await response.json();
    var history = data.history || [];
    if (!history.length) throw new Error("No star observations available.");
    await window.loadChartJs();
    document.getElementById("updated").textContent = "Last collected: " + data.updated_at;
    var projects = document.getElementById("projects");
    var repositories = Array.from(new Set(history.flatMap(function (row) {
      return Object.keys(row.stars || {});
    }))).sort();
    repositories.forEach(function (repository) {
      var section = document.createElement("section");
      section.className = "project";
      var heading = document.createElement("h2");
      var link = document.createElement("a");
      link.href = "https://github.com/" + repository.split("/").map(encodeURIComponent).join("/");
      link.textContent = repository;
      heading.appendChild(link);
      section.appendChild(heading);
      var count = document.createElement("p");
      var latest = starSeries(history.slice(-1), repository)[0].y;
      count.textContent = "Stars: " + (latest === null ? "\u2014" : latest.toLocaleString());
      section.appendChild(count);
      var wrap = document.createElement("div");
      wrap.className = "chart-wrap";
      var canvas = document.createElement("canvas");
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", repository + " daily star history");
      wrap.appendChild(canvas);
      section.appendChild(wrap);
      projects.appendChild(section);
      new Chart(canvas, {
        type: "line",
        data: {datasets: [{
          label: "Stars", data: starSeries(history, repository),
          borderColor: "#58a6ff", backgroundColor: "#58a6ff",
          pointRadius: 2, spanGaps: false,
        }]},
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: {type: "time", time: {unit: "day"}},
            y: {beginAtZero: true, ticks: {precision: 0}},
          },
          plugins: {legend: {display: false}},
        },
      });
    });
    message.textContent = "";
    document.getElementById("content").hidden = false;
  } catch (error) {
    message.textContent = error.message;
    message.className = "error";
  }
}

if (typeof window !== "undefined") loadStars();
