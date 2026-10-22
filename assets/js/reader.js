// Reader tools for long pages: numbered contents, a "where am I" bar with a progress line in the header,
// links to each section, back to top, and numeric table cells that never wrap.
// Reading time is worked out at build time (_includes/reading-time.html).
(function () {
  var html = document.documentElement;
  var T = {
    ko: { contents: "목차", sections: function (n) { return "장 " + n + "개"; }, top: "맨 위로", link: "이 부분 링크", open: "목차 열기" },
    en: { contents: "Contents", sections: function (n) { return n + " sections"; }, top: "Back to top", link: "Link to this section", open: "Open contents" }
  };
  var NUM = /^[−\-+±~]?\s?[\d.,]+\s?(%|%p|ms|s|초|분|스텝|steps?|k|M|×)?$/;

  function lang() { return html.getAttribute("data-lang") === "ko" ? "ko" : "en"; }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  // Both languages in one element; the stylesheet shows the one that matches the page.
  function pair(cls, en, ko) {
    var e = el("span", cls), a = el("span", "i18n", en), b = el("span", "i18n", ko);
    a.lang = "en"; b.lang = "ko";
    e.appendChild(a); e.appendChild(b);
    return e;
  }
  function pad(n) { return (n < 10 ? "0" : "") + n; }
  function title(h) { return h.textContent.replace(/#$/, "").trim(); }
  function entry(h, i) {
    var li = el("li"), a = el("a");
    a.href = "#" + h.id;
    a.appendChild(el("span", "reader-toc__num", pad(i + 1)));
    a.appendChild(document.createTextNode(title(h)));
    li.appendChild(a);
    return li;
  }

  function init() {
    // A cell that holds only a number is marked, so it lines up and never wraps.
    document.querySelectorAll(".content-body td").forEach(function (td) {
      if (NUM.test(td.textContent.trim())) td.classList.add("is-num");
    });

    // Bilingual notes have one block per language; other pages (the CV) use the body itself.
    var blocks = [].slice.call(document.querySelectorAll(".content-body .lang-block"));
    var sections = ":scope > h2[id]";
    if (!blocks.length) {
      var body = document.querySelector(".content-body");
      if (!body) return;
      blocks = [body];
      sections = ":scope > h1[id], :scope > h2[id]";
    }
    function heads(block) { return [].slice.call(block.querySelectorAll(sections)); }

    var most = 0;
    blocks.forEach(function (block) {
      var L = block.lang === "ko" || (!block.lang && lang() === "ko") ? "ko" : "en";
      block.querySelectorAll(sections + ", :scope > h3[id]").forEach(function (h) {
        var a = el("a", "heading-anchor", "#");
        a.href = "#" + h.id;
        a.setAttribute("aria-label", T[L].link);
        h.appendChild(a);
      });
      var hs = heads(block);
      most = Math.max(most, hs.length);
      if (hs.length < 3) return;
      // Contents go after the opening paragraph, so the first thing you read is still the point of the page.
      var nav = el("nav", "reader-toc");
      nav.setAttribute("aria-label", T[L].contents);
      var det = el("details");
      det.open = window.innerWidth >= 900;
      var sum = el("summary");
      sum.appendChild(pair("reader-toc__label", T.en.contents, T.ko.contents));
      sum.appendChild(pair("reader-toc__count", T.en.sections(hs.length), T.ko.sections(hs.length)));
      det.appendChild(sum);
      var ol = el("ol");
      hs.forEach(function (h, i) { ol.appendChild(entry(h, i)); });
      det.appendChild(ol);
      nav.appendChild(det);
      var first = block.firstElementChild;
      block.insertBefore(nav, first && first.tagName === "P" ? first.nextSibling : block.firstChild);
    });

    var header = document.querySelector(".jarvis-header");
    var hero = document.querySelector(".content-hero");
    if (!header || most < 3) return;

    var now = el("div", "reader-now");
    var nowBtn = el("button", "reader-now__btn");
    nowBtn.type = "button";
    nowBtn.setAttribute("aria-expanded", "false");
    var nowNum = el("span", "reader-now__num"), nowTitle = el("span", "reader-now__title");
    var chev = el("span", "reader-now__chev", "▾");
    chev.setAttribute("aria-hidden", "true");
    nowBtn.appendChild(nowNum); nowBtn.appendChild(nowTitle); nowBtn.appendChild(chev);
    var panel = el("div", "reader-panel");
    panel.id = "reader-panel";
    panel.hidden = true;
    nowBtn.setAttribute("aria-controls", panel.id);
    now.appendChild(nowBtn); now.appendChild(panel);
    var progress = el("div", "reader-progress");
    progress.setAttribute("aria-hidden", "true");
    progress.appendChild(el("span"));
    header.appendChild(now); header.appendChild(progress);
    var top = el("a", "reader-top");
    top.href = "#content";
    top.appendChild(el("span", null, "↑")).setAttribute("aria-hidden", "true");
    document.body.appendChild(top);

    function visibleBlock() { return blocks.filter(function (b) { return b.getClientRects().length; })[0] || blocks[0]; }
    function setOpen(open) {
      panel.hidden = !open;
      nowBtn.setAttribute("aria-expanded", String(open));
    }
    function buildPanel() {
      var L = lang(), ol = el("ol");
      heads(visibleBlock()).forEach(function (h, i) {
        var li = entry(h, i);
        li.firstChild.addEventListener("click", function () { setOpen(false); });
        ol.appendChild(li);
      });
      panel.innerHTML = "";
      panel.appendChild(ol);
      nowBtn.setAttribute("aria-label", T[L].open);
      top.setAttribute("aria-label", T[L].top);
      top.title = T[L].top;
    }
    nowBtn.addEventListener("click", function () { setOpen(panel.hidden); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && !panel.hidden) { setOpen(false); nowBtn.focus(); }
    });
    document.addEventListener("click", function (e) {
      if (!panel.hidden && !now.contains(e.target)) setOpen(false);
    });

    function update() {
      var block = visibleBlock(), hs = heads(block), idx = -1, line = Math.max(140, window.innerHeight * 0.3);
      hs.forEach(function (h, i) { if (h.getBoundingClientRect().top < line) idx = i; });
      html.classList.toggle("reader-scrolled", hero ? hero.getBoundingClientRect().bottom < 60 : true);
      if (idx >= 0) {
        nowNum.textContent = pad(idx + 1) + "/" + pad(hs.length);
        nowTitle.textContent = title(hs[idx]);
      } else {
        nowNum.textContent = "00/" + pad(hs.length);
        var h1 = document.querySelector(".content-hero h1 .i18n[lang='" + lang() + "']") || document.querySelector(".content-hero h1");
        nowTitle.textContent = h1 ? h1.textContent : "";
      }
      panel.querySelectorAll("li").forEach(function (li, i) { li.classList.toggle("is-current", i === idx); });
      var r = block.getBoundingClientRect(), lead = window.innerHeight * 0.35;
      var p = Math.min(1, Math.max(0, (lead - r.top) / Math.max(1, r.height - lead)));
      progress.firstChild.style.transform = "scaleX(" + p.toFixed(4) + ")";
      top.classList.toggle("is-visible", window.scrollY > window.innerHeight * 1.5);
    }
    var ticking = false;
    window.addEventListener("scroll", function () {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () { ticking = false; update(); });
    }, { passive: true });
    window.addEventListener("resize", update);
    document.addEventListener("daniel:langchange", function () { buildPanel(); update(); });
    buildPanel();
    update();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
