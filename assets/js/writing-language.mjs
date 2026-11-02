export function resolveWritingLanguage(search, preference) {
  const explicit = new URLSearchParams(search).get("lang");
  if (explicit === "en" || explicit === "ko") return explicit;
  return preference === "ko" ? "ko" : "en";
}

if (typeof document !== "undefined") {
  const controls = document.querySelector("[data-language-controls]");
  if (controls) {
    const key = "daniel:reading-language";
    let preference;
    try { preference = localStorage.getItem(key); } catch (_) { /* The page still works without storage. */ }
    const apply = (language) => {
      document.documentElement.dataset.lang = language;
      document.querySelectorAll("[data-writing-language]").forEach(element => {
        element.hidden = element.dataset.writingLanguage !== language;
      });
      document.querySelectorAll("[data-language-choice]").forEach(button => {
        button.setAttribute("aria-pressed", String(button.dataset.languageChoice === language));
      });
      const article = document.querySelector("[data-bilingual-article]");
      document.querySelectorAll(".note-entry a").forEach(link => {
        const url = new URL(link.href, location.href);
        url.searchParams.set("lang", language);
        link.href = url.href;
      });
      const heading = document.querySelector("h1");
      if (heading && article) {
        const title = heading.querySelector(`[data-writing-language="${language}"]`);
        document.title = `${title?.textContent.trim() || heading.textContent.trim()} - Sangbum Daniel Choi`;
      }
    };
    controls.hidden = false;
    apply(resolveWritingLanguage(location.search, preference));
    controls.addEventListener("click", event => {
      const button = event.target.closest("[data-language-choice]");
      if (!button) return;
      const language = button.dataset.languageChoice;
      try { localStorage.setItem(key, language); } catch (_) { /* Keep the selection for this page. */ }
      const url = new URL(location.href);
      url.searchParams.set("lang", language);
      history.replaceState(null, "", url);
      apply(language);
    });
    window.addEventListener("popstate", () => {
      let saved;
      try { saved = localStorage.getItem(key); } catch (_) { /* Use English when unavailable. */ }
      apply(resolveWritingLanguage(location.search, saved));
    });
  }
}
