/*
 * Loads Chart.js and its date-fns adapter, hammerjs and the zoom plugin at
 * runtime, trying several CDNs in turn. Pages should call window.loadChartJs()
 * and only create charts once the returned promise resolves; this avoids the
 * "Chart is not defined" error that appears when a single CDN is unreachable.
 */
(function () {
  // Libraries are loaded in this order so that dependencies (Chart, then the
  // adapter/hammer, then the zoom plugin) are available when needed.
  var LIBS = [
    "chart.js@4.4.4/dist/chart.umd.min.js",
    "chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js",
    "hammerjs@2.0.8/hammer.min.js",
    "chartjs-plugin-zoom@2.2.0/dist/chartjs-plugin-zoom.min.js",
  ];

  var CDNS = [
    "https://cdn.jsdelivr.net/npm/",
    "https://unpkg.com/",
    "https://cdn.skypack.dev/",
  ];

  function loadOne(path) {
    return new Promise(function (resolve, reject) {
      var index = 0;
      function attempt() {
        var script = document.createElement("script");
        script.src = CDNS[index] + path;
        script.onload = function () { resolve(); };
        script.onerror = function () {
          index += 1;
          if (index < CDNS.length) {
            attempt();
          } else {
            reject(new Error("Failed to load " + path + " from any CDN"));
          }
        };
        document.head.appendChild(script);
      }
      attempt();
    });
  }

  function loadChartJs() {
    if (window.__chartJsReady) return window.__chartJsReady;
    window.__chartJsReady = LIBS.reduce(function (chain, path) {
      return chain.then(function () { return loadOne(path); });
    }, Promise.resolve());
    return window.__chartJsReady;
  }

  window.loadChartJs = loadChartJs;
})();
