(function () {
  "use strict";

  var owner = "xadupre";
  var repo = "xadupre.github.io";
  var apiBase = "https://api.github.com/repos/" + owner + "/" + repo;
  var runCacheKey = "workflow-status-runs-v1";
  var runCacheLifetime = 15 * 60 * 1000;

  function parseField(value, minimum, maximum, normalizeWeekday) {
    var result = new Set();
    value.split(",").forEach(function (part) {
      var stepParts = part.split("/");
      var range = stepParts[0];
      var step = stepParts.length > 1 ? Number(stepParts[1]) : 1;
      var bounds = range === "*" ? [minimum, maximum] : range.split("-").map(Number);
      var start = bounds[0];
      var end = bounds.length > 1 ? bounds[1] : bounds[0];
      if (!Number.isInteger(step) || step < 1 || !Number.isInteger(start) ||
          !Number.isInteger(end) || start < minimum || end > maximum || start > end) {
        throw new Error("Unsupported cron field: " + value);
      }
      for (var current = start; current <= end; current += step) {
        result.add(normalizeWeekday && current === 7 ? 0 : current);
      }
    });
    return result;
  }

  function parseCron(expression) {
    var fields = expression.trim().split(/\s+/);
    if (fields.length !== 5) throw new Error("Expected five cron fields: " + expression);
    return {
      minute: parseField(fields[0], 0, 59, false),
      hour: parseField(fields[1], 0, 23, false),
      day: parseField(fields[2], 1, 31, false),
      month: parseField(fields[3], 1, 12, false),
      weekday: parseField(fields[4], 0, 7, true),
      anyDay: fields[2] === "*",
      anyWeekday: fields[4] === "*"
    };
  }

  function matches(schedule, date) {
    var dayMatches = schedule.day.has(date.getUTCDate());
    var weekdayMatches = schedule.weekday.has(date.getUTCDay());
    var calendarMatches;
    if (schedule.anyDay && schedule.anyWeekday) {
      calendarMatches = true;
    } else if (schedule.anyDay) {
      calendarMatches = weekdayMatches;
    } else if (schedule.anyWeekday) {
      calendarMatches = dayMatches;
    } else {
      calendarMatches = dayMatches || weekdayMatches;
    }
    return schedule.minute.has(date.getUTCMinutes()) &&
      schedule.hour.has(date.getUTCHours()) &&
      schedule.month.has(date.getUTCMonth() + 1) &&
      calendarMatches;
  }

  function nextCron(expression, now) {
    var schedule = parseCron(expression);
    var candidate = new Date(now.getTime());
    candidate.setUTCSeconds(0, 0);
    candidate = new Date(candidate.getTime() + 60000);
    var limit = 400 * 24 * 60;
    for (var index = 0; index < limit; ++index) {
      if (matches(schedule, candidate)) return candidate;
      candidate = new Date(candidate.getTime() + 60000);
    }
    return null;
  }

  function nextWorkflowRun(crons, now) {
    var dates = crons.map(function (cron) {
      try {
        return nextCron(cron, now);
      } catch (_error) {
        return null;
      }
    }).filter(Boolean);
    if (!dates.length) return null;
    return new Date(Math.min.apply(null, dates.map(function (date) { return date.getTime(); })));
  }

  function formatDate(value) {
    if (!value) return "Never";
    var date = value instanceof Date ? value : new Date(value);
    if (isNaN(date.getTime())) return "Unknown";
    return new Intl.DateTimeFormat(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "UTC",
      timeZoneName: "short"
    }).format(date);
  }

  function fetchJson(url) {
    return fetch(url, {
      headers: { "Accept": "application/vnd.github+json" },
      cache: "no-store"
    }).then(function (response) {
      if (!response.ok) throw new Error("GitHub API returned " + response.status);
      return response.json();
    });
  }

  function loadRunCache(maximumAge) {
    try {
      var cached = JSON.parse(localStorage.getItem(runCacheKey));
      if (!cached || !Array.isArray(cached.runs) ||
          Date.now() - Number(cached.savedAt) > maximumAge) {
        return null;
      }
      return new Map(cached.runs.map(function (run) { return [run.path, run]; }));
    } catch (_error) {
      return null;
    }
  }

  function saveRunCache(latest) {
    try {
      localStorage.setItem(runCacheKey, JSON.stringify({
        savedAt: Date.now(),
        runs: Array.from(latest.values()).map(function (run) {
          return {
            path: run.path,
            html_url: run.html_url,
            run_started_at: run.run_started_at,
            created_at: run.created_at,
            status: run.status,
            conclusion: run.conclusion
          };
        })
      }));
    } catch (_error) {
      // The live result remains usable when browser storage is unavailable.
    }
  }

  function fetchLatestRuns(paths) {
    var cached = loadRunCache(runCacheLifetime);
    if (cached) return Promise.resolve(cached);

    var latest = new Map();
    var wanted = new Set(paths);

    function fetchPage(page) {
      var url = apiBase + "/actions/runs?per_page=100&exclude_pull_requests=true&page=" + page;
      return fetchJson(url).then(function (payload) {
        var runs = payload.workflow_runs || [];
        runs.forEach(function (run) {
          if (wanted.has(run.path) && !latest.has(run.path)) latest.set(run.path, run);
        });
        if (latest.size === wanted.size || runs.length < 100 || page >= 10) return latest;
        return fetchPage(page + 1);
      });
    }

    return fetchPage(1).then(function (runs) {
      saveRunCache(runs);
      return runs;
    }).catch(function (error) {
      var stale = loadRunCache(Infinity);
      if (!stale) throw error;
      stale.isStale = true;
      return stale;
    });
  }

  function conclusionLabel(run) {
    if (!run) return "";
    if (run.status !== "completed") return run.status || "";
    return run.conclusion || "completed";
  }

  function render(manifest, latestRuns) {
    var tbody = document.getElementById("workflow-status-body");
    var message = document.getElementById("workflow-status-message");
    var now = new Date();
    var rows = manifest.map(function (workflow) {
      return {
        workflow: workflow,
        run: latestRuns.get(workflow.path),
        next: nextWorkflowRun(workflow.crons || [], now)
      };
    });
    rows.sort(function (left, right) {
      if (left.next && right.next) return left.next - right.next;
      if (left.next) return -1;
      if (right.next) return 1;
      return left.workflow.name.localeCompare(right.workflow.name);
    });

    tbody.textContent = "";
    rows.forEach(function (item) {
      var tr = document.createElement("tr");
      var action = document.createElement("td");
      var actionLink = document.createElement("a");
      actionLink.href = "https://github.com/" + owner + "/" + repo + "/actions/workflows/" +
        item.workflow.path.split("/").pop();
      actionLink.textContent = item.workflow.name;
      action.appendChild(actionLink);

      var last = document.createElement("td");
      if (item.run) {
        var runLink = document.createElement("a");
        runLink.href = item.run.html_url;
        runLink.textContent = formatDate(item.run.run_started_at || item.run.created_at);
        last.appendChild(runLink);
        var conclusion = document.createElement("span");
        conclusion.className = "workflow-conclusion workflow-" + conclusionLabel(item.run);
        conclusion.textContent = conclusionLabel(item.run);
        last.appendChild(conclusion);
      } else {
        last.textContent = "No run found";
      }

      var next = document.createElement("td");
      next.textContent = item.next ? formatDate(item.next) : "Manual / event-driven";
      if ((item.workflow.crons || []).length) {
        next.title = item.workflow.crons.join(", ") + " (UTC)";
      }
      tr.appendChild(action);
      tr.appendChild(last);
      tr.appendChild(next);
      tbody.appendChild(tr);
    });
    message.textContent = latestRuns.isStale ?
      "Last runs are cached because the GitHub API is unavailable. Times are shown in UTC." :
      "Times are shown in UTC. Scheduled times may be delayed by GitHub.";
  }

  function showError(error) {
    var message = document.getElementById("workflow-status-message");
    message.textContent = "Unable to load workflow runs: " + error.message;
    message.className = "workflow-status-message workflow-error";
  }

  function init() {
    fetchJson("assets/workflow-manifest.json").then(function (manifest) {
      return fetchLatestRuns(manifest.map(function (workflow) {
        return workflow.path;
      })).then(function (latestRuns) {
        render(manifest, latestRuns);
      });
    }).catch(showError);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
