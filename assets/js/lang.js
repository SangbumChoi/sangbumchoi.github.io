// Site-wide English / Korean switch. The initial language is set inline in <head>
// (jarvis-lang-init.html) so the page never flashes the wrong language.
(function () {
  var STORAGE_KEY = "daniel-os:lang";
  var root = document.documentElement;

  function current() {
    return root.getAttribute("data-lang") === "ko" ? "ko" : "en";
  }

  function syncControls(lang) {
    document.querySelectorAll("[data-set-lang]").forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.getAttribute("data-set-lang") === lang));
    });
    document.querySelectorAll("[data-placeholder-" + lang + "]").forEach(function (field) {
      field.setAttribute("placeholder", field.getAttribute("data-placeholder-" + lang));
    });
  }

  function apply(lang) {
    root.setAttribute("data-lang", lang);
    root.setAttribute("lang", lang);
    try {
      window.localStorage.setItem(STORAGE_KEY, lang);
    } catch (error) {
      // Storage can be unavailable in private windows; the choice still applies to this page.
    }
    syncControls(lang);
    document.dispatchEvent(new CustomEvent("daniel:langchange", { detail: { lang: lang } }));
  }

  window.danielLang = { current: current, set: apply };

  function init() {
    syncControls(current());
    document.querySelectorAll("[data-set-lang]").forEach(function (button) {
      button.addEventListener("click", function () {
        var next = button.getAttribute("data-set-lang");
        if (next !== current()) apply(next);
      });
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
