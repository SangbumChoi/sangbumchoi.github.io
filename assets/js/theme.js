// Site-wide light / dark switch. The initial theme is set inline in <head>
// (jarvis-theme-init.html) so the page never flashes the wrong colours.
(function () {
  var STORAGE_KEY = "daniel-os:theme";
  var COLORS = { light: "#ffffff", dark: "#0b0b0b" };
  var root = document.documentElement;
  var system = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function current() {
    return root.getAttribute("data-theme") === "dark" ? "dark" : "light";
  }

  function stored() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch (error) {
      return null;
    }
  }

  function apply(theme, remember) {
    root.setAttribute("data-theme", theme);
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", COLORS[theme]);
    if (remember) {
      try {
        window.localStorage.setItem(STORAGE_KEY, theme);
      } catch (error) {
        // Storage can be unavailable in private windows; the choice still applies to this page.
      }
    }
    document.dispatchEvent(new CustomEvent("daniel:themechange", { detail: { theme: theme } }));
  }

  window.danielTheme = { current: current, set: function (theme) { apply(theme, true); } };

  function init() {
    document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
      button.addEventListener("click", function () {
        apply(current() === "dark" ? "light" : "dark", true);
      });
    });
    // Follow the system setting until the visitor picks a theme themselves.
    if (system && system.addEventListener) {
      system.addEventListener("change", function (event) {
        if (stored() !== "light" && stored() !== "dark") apply(event.matches ? "dark" : "light", false);
      });
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
