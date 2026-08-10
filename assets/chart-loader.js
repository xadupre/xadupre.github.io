/*
 * Loads Chart.js and its date-fns adapter, hammerjs and the zoom plugin at
 * runtime from local vendored files. Pages should call window.loadChartJs()
 * and only create charts once the returned promise resolves.
 */
(function () {
  var loaderUrl = document.currentScript.src;

  // Libraries are loaded in this order so that dependencies (Chart, then the
  // adapter/hammer, then the zoom plugin) are available when needed.
  var LIBS = [
    "chart.umd.min.js",
    "chartjs-adapter-date-fns.bundle.min.js",
    "hammer.min.js",
    "chartjs-plugin-zoom.min.js",
  ];

  function loadOne(path) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = new URL("vendor/" + path, loaderUrl).href;
      script.onload = function () { resolve(); };
      script.onerror = function () {
        reject(new Error("Failed to load local chart dependency " + path));
      };
      document.head.appendChild(script);
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
