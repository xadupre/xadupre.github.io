/*
 * Populates elements with class "data-updated" with a "Data last updated: ..."
 * message based on the HTTP Last-Modified header of one or more data sources
 * listed in the element's data-source attribute (comma-separated). The latest
 * date across all sources is shown.
 */
(function () {
  function format(date) {
    if (!date || isNaN(date.getTime())) return null;
    return date.toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC");
  }

  function lastModified(url) {
    return fetch(url, { method: "HEAD", cache: "no-store" })
      .then(function (r) {
        if (!r.ok) return null;
        var lm = r.headers.get("Last-Modified");
        return lm ? new Date(lm) : null;
      })
      .catch(function () { return null; });
  }

  function update(el) {
    var raw = el.getAttribute("data-source") || "";
    var sources = raw.split(",").map(function (s) { return s.trim(); }).filter(Boolean);
    if (!sources.length) return;
    Promise.all(sources.map(lastModified)).then(function (dates) {
      var valid = dates.filter(function (d) { return d instanceof Date && !isNaN(d.getTime()); });
      if (!valid.length) return;
      var max = new Date(Math.max.apply(null, valid.map(function (d) { return d.getTime(); })));
      var text = format(max);
      if (!text) return;
      el.textContent = "Data last updated: " + text;
      el.style.display = "";
    });
  }

  function init() {
    var els = document.querySelectorAll(".data-updated");
    for (var i = 0; i < els.length; i++) update(els[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
