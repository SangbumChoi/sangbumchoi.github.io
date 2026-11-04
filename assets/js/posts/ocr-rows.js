// Figures for the OCR row-repetition note: where the model looks while it writes each row,
// and P(EOS) right after each row. Numbers come from assets/data/posts/ocr-row-masks.json.
(function () {
  var NS = "http://www.w3.org/2000/svg";
  // Round half up so 88.55 shows as 88.6, matching the tables in the text.
  function r1(v) { return (v + 1e-9).toFixed(1); }
  var T = {
    en: {
      uniq: "Every row different", same: "Every row identical",
      cell: "While writing cells", nl: "At the row end <nl>",
      corner: "Writing ↓ · looking →",
      aheadCell: function (n) { return "The most-watched image row is ahead of the row being written in " + n + " of 8 rows."; },
      aheadNl: function (n) { return "The most-watched image row is ahead of the next row in " + n + " of 8 rows."; },
      legend: "Bold outline = the row it looks at most. Dotted outline = the row it should look at. Values are % of image attention.",
      base: "Identical rows, no mask",
      masks: { uniq_base: "Unique rows, no mask", all_only_cur: "Only the current row (needs row positions)", nl_block_hist_nl: "<nl> can't see earlier <nl>", nl_block_hist_all: "<nl> can't see any earlier output", nl_only_next: "<nl> sees only the next image row" },
      after: "after row", end: "real end", y: "P(EOS) right after each row, %",
      readout: function (b, m, label) { return "After row 7: no mask " + r1(b[6]) + "%, " + label + " " + r1(m[6]) + "%. After row 8, the real end: " + r1(b[7]) + "% and " + r1(m[7]) + "%."; },
      fail: "Couldn't load the figure data. Refresh the page and try again."
    },
    ko: {
      uniq: "행이 모두 다름", same: "행이 모두 같음",
      cell: "칸을 쓰는 동안", nl: "행 끝 <nl> 에서",
      corner: "쓰는 행 ↓ · 보는 행 →",
      aheadCell: function (n) { return "가장 많이 보는 이미지 행이 지금 쓰는 행보다 앞선 경우: 8행 중 " + n + "행"; },
      aheadNl: function (n) { return "가장 많이 보는 이미지 행이 다음 행보다 앞선 경우: 8행 중 " + n + "행"; },
      legend: "굵은 테두리 = 가장 많이 보는 행, 점선 = 원래 봐야 할 행. 값은 이미지 attention 중 % 예요.",
      base: "같은 행 표, 마스크 없음",
      masks: { uniq_base: "다른 행 표, 마스크 없음", all_only_cur: "지금 행만 보게 함 (행 위치 필요)", nl_block_hist_nl: "<nl> 이 앞 <nl> 을 못 봄", nl_block_hist_all: "<nl> 이 앞 출력을 모두 못 봄", nl_only_next: "<nl> 에서 다음 행만 봄" },
      after: "번째 행 뒤", end: "진짜 끝", y: "행을 쓴 직후 P(EOS), %",
      readout: function (b, m, label) { return "7번째 행 뒤: 마스크 없음 " + r1(b[6]) + "%, " + label + " " + r1(m[6]) + "%. 진짜 끝인 8번째 행 뒤: " + r1(b[7]) + "%, " + r1(m[7]) + "%."; },
      fail: "그림 데이터를 불러오지 못했어요. 새로고침한 뒤 다시 열어 주세요."
    }
  };

  function langOf(node) {
    var block = node.closest("[lang]");
    return block && block.getAttribute("lang") === "ko" ? "ko" : "en";
  }
  function html(tag, cls, parent, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function svg(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) {
      // CSS variables only resolve as style properties, not as SVG presentation attributes.
      if (typeof attrs[k] === "string" && attrs[k].indexOf("var(") === 0) e.style.setProperty(k, attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  }
  function tabs(parent, items, current, onPick) {
    var bar = html("div", "ocr-tabs", parent);
    items.forEach(function (it) {
      var b = html("button", null, bar, it[1]);
      b.type = "button";
      b.setAttribute("aria-pressed", it[0] === current ? "true" : "false");
      b.addEventListener("click", function () {
        bar.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        onPick(it[0]);
      });
    });
    return bar;
  }
  function argmax(a) { var j = 0; for (var i = 1; i < a.length; i++) if (a[i] > a[j]) j = i; return j; }

  function heatFigure(box, L, data) {
    var t = T[L], st = { table: "same", moment: "cell" };
    var head = html("div", "ocr-controls", box);
    tabs(head, [["uniq", t.uniq], ["same", t.same]], st.table, function (v) { st.table = v; draw(); });
    tabs(head, [["cell", t.cell], ["nl", t.nl]], st.moment, function (v) { st.moment = v; draw(); });
    var wrap = html("div", "ocr-heat-wrap", box), summary = html("p", "ocr-readout", box);
    html("p", "ocr-legend", box, t.legend);
    function draw() {
      var m = data.attn[st.table][st.moment], ahead = 0;
      wrap.textContent = "";
      html("div", "ocr-axis-label", wrap, t.corner);
      var table = html("table", "ocr-heat", wrap), tr = html("tr", null, html("thead", null, table));
      html("th", null, tr, "");
      for (var j = 0; j < 8; j++) html("th", null, tr, String(j + 1));
      var body = html("tbody", null, table);
      m.forEach(function (row, k) {
        var sum = row.reduce(function (a, b) { return a + b; }, 0) || 1, peak = argmax(row);
        var target = st.moment === "nl" ? Math.min(k + 1, 7) : k;
        if (peak > target) ahead++;
        var r = html("tr", null, body);
        html("th", null, r, String(k + 1));
        row.forEach(function (v, j) {
          var pct = 100 * v / sum, td = html("td", null, r, pct.toFixed(0));
          td.style.background = "color-mix(in srgb, var(--chart-1) " + Math.min(pct / 50, 1) * 60 + "%, transparent)";
          if (j === peak) td.classList.add("is-peak");
          if (j === target) td.classList.add("is-target");
        });
      });
      summary.textContent = st.moment === "nl" ? t.aheadNl(ahead) : t.aheadCell(ahead);
    }
    draw();
  }

  function eosFigure(box, L, data) {
    var t = T[L], st = { mask: "nl_block_hist_nl" };
    var keys = ["uniq_base", "all_only_cur", "nl_block_hist_nl", "nl_block_hist_all", "nl_only_next"];
    tabs(box, keys.map(function (k) { return [k, t.masks[k]]; }), st.mask, function (v) { st.mask = v; draw(); });
    html("div", "ocr-axis-label", box, t.y);
    var plot = html("div", "ocr-plot", box), readout = html("p", "ocr-readout", box);
    var legend = html("p", "ocr-legend", box);
    function draw() {
      var w = Math.max(280, plot.clientWidth), h = 240, l = 40, r = 18, top = 12, b = 40;
      var px = function (i) { return l + (w - l - r) * i / 7; }, py = function (v) { return top + (h - top - b) * (1 - v / 100); };
      plot.textContent = "";
      var s = svg("svg", { "class": "ocr-svg", viewBox: "0 0 " + w + " " + h, width: w, height: h, role: "img", "aria-label": t.y }, plot);
      svg("rect", { x: px(7) - 14, y: top, width: 28, height: h - top - b, fill: "var(--line)", opacity: "0.5" }, s);
      [0, 25, 50, 75, 100].forEach(function (v) {
        svg("line", { x1: l, x2: w - r, y1: py(v), y2: py(v), "class": "ocr-grid" }, s);
        svg("text", { x: l - 8, y: py(v) + 4, "text-anchor": "end", "class": "ocr-tick" }, s).textContent = v;
      });
      for (var i = 0; i < 8; i++) svg("text", { x: px(i), y: h - b + 20, "text-anchor": "middle", "class": "ocr-tick" }, s).textContent = i + 1;
      svg("text", { x: px(7) + 14, y: h - 4, "text-anchor": "end", "class": "ocr-tick is-strong" }, s).textContent = t.end;
      var base = data.eos.same_base, cur = data.eos[st.mask];
      [[base, "var(--chart-muted)"], [cur, "var(--chart-1)"]].forEach(function (sr) {
        svg("polyline", { points: sr[0].map(function (v, i) { return px(i) + "," + py(v); }).join(" "), fill: "none", stroke: sr[1], "stroke-width": "2.5" }, s);
        sr[0].forEach(function (v, i) { svg("circle", { cx: px(i), cy: py(v), r: 4, fill: sr[1] }, s); });
      });
      readout.textContent = t.readout(base, cur, t.masks[st.mask]);
      legend.textContent = "";
      [[t.base, "var(--chart-muted)"], [t.masks[st.mask], "var(--chart-1)"]].forEach(function (it) {
        var item = html("span", null, legend), key = html("i", "ocr-key", item);
        key.style.background = it[1];
        item.appendChild(document.createTextNode(it[0]));
      });
    }
    draw();
    if (window.ResizeObserver) {
      var last = plot.clientWidth;
      new ResizeObserver(function () { if (plot.clientWidth && Math.abs(plot.clientWidth - last) > 4) { last = plot.clientWidth; draw(); } }).observe(plot);
    }
  }

  var cache = {};
  function load(src) {
    if (!cache[src]) cache[src] = fetch(src).then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); });
    return cache[src];
  }
  // Layout depends on the figure's width, so a figure in the hidden language block waits until it is shown.
  function whenVisible(box, fn) {
    if (box.clientWidth > 0 || !window.ResizeObserver) return fn();
    var ro = new ResizeObserver(function () { if (box.clientWidth > 0) { ro.disconnect(); fn(); } });
    ro.observe(box);
  }
  function init() {
    document.querySelectorAll("[data-ocr]").forEach(function (box) {
      whenVisible(box, function () {
        var L = langOf(box);
        load(box.getAttribute("data-src")).then(function (data) {
          if (box.getAttribute("data-ocr") === "heat") heatFigure(box, L, data);
          else eosFigure(box, L, data);
        }).catch(function () { html("p", "ocr-readout", box, T[L].fail); });
      });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
