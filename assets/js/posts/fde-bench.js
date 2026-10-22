// Figures for the FDE-Bench field note: interactive hull/crossing demos, the case 002 map, and charts.
// Geometry mirrors case 002's evaluator: monotone-chain hull, on-edge counts as inside, strict segment crossing.
(function () {
  var NS = "http://www.w3.org/2000/svg";
  // Theme colours (assets/css/jarvis.css and fde-bench.css). The captions call A blue, B green and violations red.
  var C = { a: "var(--chart-1)", b: "var(--chart-3)", bad: "var(--accent)", s1: "var(--chart-1)", s2: "var(--chart-2)", s3: "var(--chart-3)", s4: "var(--chart-4)",
    ink: "var(--ink)", text: "var(--text)", text2: "var(--text-2)", text3: "var(--text-3)",
    route: "var(--fde-route)", hull: "var(--fde-hull)", store: "var(--fde-store)", dim: "var(--fde-dim)", range: "var(--fde-range)" };

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) {
      // CSS variables only resolve as style properties, not as SVG presentation attributes.
      if (typeof attrs[k] === "string" && attrs[k].indexOf("var(") === 0) e.style.setProperty(k, attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  }
  function html(tag, cls, parent, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function langOf(node) {
    var block = node.closest("[lang]");
    return block && block.getAttribute("lang") === "ko" ? "ko" : "en";
  }
  function fmt(n) { return Math.round(n).toLocaleString("en-US"); }
  function fmtW(w) { return w < 10 ? String(Math.round(w * 10) / 10) : fmt(w); }
  // Korean copula after a noun: 이에요 after a final consonant, 예요 otherwise.
  function copula(word) {
    var c = word.charCodeAt(word.length - 1);
    return c >= 0xac00 && c <= 0xd7a3 && (c - 0xac00) % 28 ? "이에요" : "예요";
  }

  // ---- geometry, ported line for line from tests/evaluator.py ----
  function cross(o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); }
  function hull(pts) {
    if (pts.length < 3) return pts.slice();
    pts = pts.slice().sort(function (p, q) { return p[0] - q[0] || p[1] - q[1]; });
    var h = [];
    pts.forEach(function (p) {
      while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop();
      h.push(p);
    });
    var lower = h.length;
    for (var i = pts.length - 2; i >= 0; i--) {
      var p = pts[i];
      while (h.length > lower && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop();
      h.push(p);
    }
    h.pop();
    return h;
  }
  function inHull(p, h) {
    if (h.length < 3) return false;
    for (var i = 0; i < h.length; i++) if (cross(h[i], h[(i + 1) % h.length], p) < 0) return false;
    return true;
  }
  function segX(p1, p2, p3, p4) {
    var d1 = cross(p3, p4, p1), d2 = cross(p3, p4, p2), d3 = cross(p1, p2, p3), d4 = cross(p1, p2, p4);
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
  }
  function meet(p1, p2, p3, p4) {
    var d = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]);
    var t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / d;
    return [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])];
  }
  function pts(list) { return list.map(function (p) { return p[0] + "," + p[1]; }).join(" "); }

  function svgPoint(svg, ev) {
    var pt = svg.createSVGPoint();
    pt.x = ev.clientX; pt.y = ev.clientY;
    var p = pt.matrixTransform(svg.getScreenCTM().inverse());
    return [p.x, p.y];
  }
  // A drag handle that lives outside the redrawn layers, so pointer capture survives each redraw.
  function makeHandle(svg, color, start, bounds, onMove) {
    var g = el("g", {}, svg);
    var halo = el("circle", { r: 16, fill: color, "fill-opacity": 0.18 }, g);
    var dot = el("circle", { r: 8, fill: color, stroke: C.ink, "stroke-width": 2 }, g);
    function place(p) { [halo, dot].forEach(function (c) { c.setAttribute("cx", p[0]); c.setAttribute("cy", p[1]); }); }
    place(start);
    draggable(svg, g, function (p) { place(p); onMove(p); }, bounds);
    return g;
  }
  function draggable(svg, handle, onMove, bounds) {
    var active = false;
    handle.style.cursor = "grab";
    handle.addEventListener("pointerdown", function (ev) { active = true; handle.setPointerCapture(ev.pointerId); handle.style.cursor = "grabbing"; ev.preventDefault(); });
    handle.addEventListener("pointermove", function (ev) {
      if (!active) return;
      var p = svgPoint(svg, ev);
      onMove([Math.max(bounds[0], Math.min(bounds[2], Math.round(p[0]))), Math.max(bounds[1], Math.min(bounds[3], Math.round(p[1])))]);
    });
    ["pointerup", "pointercancel"].forEach(function (t) { handle.addEventListener(t, function () { active = false; handle.style.cursor = "grab"; }); });
  }
  function tooltip(holder) {
    var tip = html("div", "fde-tip", holder);
    return {
      show: function (x, y, content) {
        tip.innerHTML = content;
        tip.style.display = "block";
        var w = holder.clientWidth, tw = tip.offsetWidth;
        tip.style.left = Math.max(0, Math.min(w - tw, x + 14)) + "px";
        tip.style.top = Math.max(0, y - tip.offsetHeight - 10) + "px";
      },
      hide: function () { tip.style.display = "none"; }
    };
  }

  var T = {
    en: {
      hullTitle: "Drag the green store", vehicle: "Vehicle", inside: "inside", intrusions: "Intrusions", counted: "counted",
      notCounted: "not counted", onEdge: "on the edge: counts", noHull: "2 stores: no hull of its own, but its stores still count",
      hullOf: function (v) { return "Hull of vehicle " + v; }, intrInto: function (o, n) { return n + (n === 1 ? " other store" : " other stores") + " inside " + o + "'s hull"; },
      crossTitle: "Drag the green stop", crossings: "Crossings", depot: "Depot", shared: "shared depot endpoint: not a crossing",
      self: "self-crossing: not counted", cross: "crossing: counted",
      score: "Score", longest: "Longest − shortest", total: "Total time", hours: "h", routes: "Routes", hulls: "Hulls", violations: "Violations",
      store: "Store", kg: "kg", insideOf: function (v) { return "inside vehicle " + v + "'s hull"; }, crossOf: function (a, b) { return "Vehicle " + a + " × vehicle " + b; },
      pair: function (a, b, i, c) { return "Vehicles " + a + " and " + b + ": " + i + (i === 1 ? " intrusion, " : " intrusions, ") + c + (c === 1 ? " crossing" : " crossings") + " between them"; },
      prev: "◀ Previous pair", next: "Next pair ▶", pairRank: function (k, n) { return "pair " + k + " of " + n; },
      axisHours: "Route duration (hours) · each dot is one truck", spread: "spread",
      allSame: function (n, h) { return "all " + n + " trucks at " + h.toFixed(2) + " h"; },
      dotTip: function (v, h, kg, n) { return "<b>Vehicle " + v + "</b><br>" + h.toFixed(2) + " h · " + fmt(kg) + " kg · " + n + " stops"; },
      costAxis: "Cost in the evaluator (lower is better)", parts: ["Total time", "Balance × 90", "Intrusions × 600", "Crossings × 1,800"],
      wPresets: ["This customer", "Driving time only", "Territories only"],
      wNames: ["Balance: per second of spread", "Intrusion: seconds each", "Crossing: seconds each"],
      wUnit: [function (w) { return "×" + fmtW(w); }, function (w) { return fmtW(w) + " s"; }, function (w) { return fmtW(w) + " s"; }],
      wAxis: "Cost under these weights (lower is better)", wParts: ["Driving time", "Balance", "Intrusions", "Crossings"],
      wRank: function (k) { return ["1st", "2nd", "3rd", "4th"][k - 1] || k + "th"; },
      wSum: function (win, isRef, rank, n) { return isRef ? "Winner: the FDE plan." : "Winner: " + win + ". The FDE plan ranks " + T.en.wRank(rank) + " of " + n + "."; },
      ksX: "Share of non-defaulters passed (riskiest first)", ksY: "Share of defaulters caught", ksAt: "KS", diag: "random ordering",
      ksTip: function (name, f, t) { return "<b>" + name + "</b><br>non-defaulters " + (f * 100).toFixed(1) + "% · defaulters " + (t * 100).toFixed(1) + "% · gap " + ((t - f) * 100).toFixed(1) + "%p"; }
    },
    ko: {
      hullTitle: "초록 매장을 끌어 보세요", vehicle: "차량", inside: "안", intrusions: "침범", counted: "셈",
      notCounted: "안 셈", onEdge: "경계 위: 셈", noHull: "매장 2곳: 자기 다각형은 없지만, 남의 다각형 안에 들어가면 셈",
      hullOf: function (v) { return v + "의 볼록 다각형"; }, intrInto: function (o, n) { return o + " 다각형 안의 남의 매장 " + n + "곳"; },
      crossTitle: "초록 정차 지점을 끌어 보세요", crossings: "교차", depot: "창고", shared: "창고를 함께 쓰는 끝점: 교차 아님",
      self: "자기 경로끼리 교차: 안 셈", cross: "다른 차끼리 교차: 셈",
      score: "점수", longest: "최장 − 최단", total: "총 시간", hours: "시간", routes: "경로", hulls: "다각형", violations: "위반",
      store: "매장", kg: "kg", insideOf: function (v) { return v + "번 차 다각형 안"; }, crossOf: function (a, b) { return a + "번 차 × " + b + "번 차"; },
      pair: function (a, b, i, c) { return a + "번과 " + b + "번 차 사이: 침범 " + i + "회, 교차 " + c + "회"; },
      prev: "◀ 이전 쌍", next: "다음 쌍 ▶", pairRank: function (k, n) { return n + "쌍 중 " + k + "번째"; },
      axisHours: "경로 시간 (시간) · 점 하나가 트럭 한 대", spread: "차이",
      allSame: function (n, h) { return n + "대 모두 " + h.toFixed(2) + "시간"; },
      dotTip: function (v, h, kg, n) { return "<b>" + v + "번 차</b><br>" + h.toFixed(2) + "시간 · " + fmt(kg) + " kg · 정차 " + n + "곳"; },
      costAxis: "채점기의 비용 (낮을수록 좋아요)", parts: ["총 시간", "균등 × 90", "침범 × 600", "교차 × 1,800"],
      wPresets: ["이 고객", "주행 시간만", "구역만"],
      wNames: ["균등: 시간 차이 1초에", "침범: 1회에 몇 초", "교차: 1회에 몇 초"],
      wUnit: [function (w) { return "×" + fmtW(w); }, function (w) { return fmtW(w) + "초"; }, function (w) { return fmtW(w) + "초"; }],
      wAxis: "이 가중치에서의 비용 (낮을수록 좋아요)", wParts: ["주행 시간", "균등", "침범", "교차"],
      wRank: function (k) { return k + "위"; },
      wSum: function (win, isRef, rank, n) { return isRef ? "1위는 FDE 정답이에요." : "1위는 " + win + copula(win) + ". FDE 정답은 " + n + "개 중 " + rank + "위예요."; },
      ksX: "통과시킨 정상 고객 비율 (위험한 사람부터)", ksY: "잡아낸 부도 고객 비율", ksAt: "KS", diag: "무작위 순서",
      ksTip: function (name, f, t) { return "<b>" + name + "</b><br>정상 " + (f * 100).toFixed(1) + "% · 부도 " + (t * 100).toFixed(1) + "% · 차이 " + ((t - f) * 100).toFixed(1) + "%p"; }
    }
  };

  // ---- 1. hull intrusion demo ----
  function hullDemo(box, L) {
    var t = T[L];
    var A = [[90, 70], [270, 55], [340, 170], [255, 300], [110, 270], [190, 170]];
    var B = [[430, 80], [590, 110], [575, 270], [420, 290], [230, 205]]; // last one is draggable
    var Cv = [[100, 170], [40, 330]];
    var names = { A: t.vehicle + " A", B: t.vehicle + " B", C: t.vehicle + " C" };
    var head = html("div", "fde-demo__head", box);
    var svg = el("svg", { viewBox: "0 0 640 350", "class": "fde-svg", role: "img", "aria-label": "Convex hull intrusion demo" }, box);
    var gHull = el("g", {}, svg), gMark = el("g", {}, svg), gPts = el("g", {}, svg);
    var legend = html("div", "fde-demo__legend", box);
    legend.innerHTML = '<span><i class="fde-key" style="background:' + C.a + '"></i>' + names.A + '</span><span><i class="fde-key" style="background:' + C.b + '"></i>' + names.B +
      '</span><span><i class="fde-key is-hollow"></i>' + names.C + " · " + t.noHull + '</span><span><i class="fde-ring"></i>' + t.intrusions + " (" + t.counted + ")</span>";
    function draw() {
      [gHull, gMark, gPts].forEach(function (g) { while (g.firstChild) g.removeChild(g.firstChild); });
      var hA = hull(A), hB = hull(B);
      el("polygon", { points: pts(hA), fill: C.a, "fill-opacity": 0.12, stroke: C.a, "stroke-width": 2 }, gHull);
      el("polygon", { points: pts(hB), fill: C.b, "fill-opacity": 0.12, stroke: C.b, "stroke-width": 2 }, gHull);
      var sets = { A: A, B: B, C: Cv }, hulls = { A: hA, B: hB }, count = { A: 0, B: 0 };
      Object.keys(hulls).forEach(function (owner) {
        Object.keys(sets).forEach(function (other) {
          if (other === owner) return;
          sets[other].forEach(function (p) {
            if (inHull(p, hulls[owner])) {
              count[owner] += 1;
              el("circle", { cx: p[0], cy: p[1], r: 11, fill: "none", stroke: C.bad, "stroke-width": 2.5 }, gMark);
            }
          });
        });
      });
      A.forEach(function (p) { el("circle", { cx: p[0], cy: p[1], r: 6, fill: C.a }, gPts); });
      B.forEach(function (p, i) { if (i < B.length - 1) el("circle", { cx: p[0], cy: p[1], r: 6, fill: C.b }, gPts); });
      Cv.forEach(function (p) { el("circle", { cx: p[0], cy: p[1], r: 6, fill: C.ink, stroke: C.text2, "stroke-width": 2 }, gPts); });
      el("line", { x1: Cv[0][0], y1: Cv[0][1], x2: Cv[1][0], y2: Cv[1][1], stroke: C.text3, "stroke-dasharray": "4 4" }, gPts);
      el("text", { x: 108, y: 186, "class": "fde-note" }, gPts).textContent = t.onEdge;
      el("text", { x: 56, y: 334, "class": "fde-note is-muted" }, gPts).textContent = names.C;
      head.innerHTML = '<span class="fde-demo__hint">' + t.hullTitle + '</span><span class="fde-demo__count"><b>' + t.intrusions + " " + (count.A + count.B) +
        "</b> = " + t.intrInto(names.A, count.A) + " + " + t.intrInto(names.B, count.B) + "</span>";
    }
    draw();
    makeHandle(svg, C.b, B[B.length - 1], [10, 10, 630, 340], function (p) { B[B.length - 1] = p; draw(); });
  }

  // ---- 2. crossing demo ----
  function crossDemo(box, L) {
    var t = T[L];
    var depot = [320, 325];
    var A = [[120, 270], [250, 50], [60, 60], [220, 180]];
    var B = [[430, 260], [580, 140], [470, 50], [45, 140]]; // last stop is draggable
    var head = html("div", "fde-demo__head", box);
    var svg = el("svg", { viewBox: "0 0 640 350", "class": "fde-svg", role: "img", "aria-label": "Route crossing demo" }, box);
    var g = el("g", {}, svg);
    var legend = html("div", "fde-demo__legend", box);
    legend.innerHTML = '<span><i class="fde-key" style="background:' + C.a + '"></i>' + t.vehicle + ' A</span><span><i class="fde-key" style="background:' + C.b + '"></i>' + t.vehicle +
      ' B</span><span><b class="fde-x">×</b>' + t.cross + '</span><span><i class="fde-ring is-gray"></i>' + t.self + '</span><span><i class="fde-key is-square"></i>' + t.shared + "</span>";
    function path(r) { return [depot].concat(r, [depot]); }
    function draw() {
      while (g.firstChild) g.removeChild(g.firstChild);
      var pa = path(A), pb = path(B), n = 0;
      el("polyline", { points: pts(pa), fill: "none", stroke: C.a, "stroke-width": 2.5 }, g);
      el("polyline", { points: pts(pb), fill: "none", stroke: C.b, "stroke-width": 2.5 }, g);
      for (var i = 0; i < pa.length - 1; i++) {
        for (var j = i + 2; j < pa.length - 1; j++) {
          if (segX(pa[i], pa[i + 1], pa[j], pa[j + 1])) {
            var m = meet(pa[i], pa[i + 1], pa[j], pa[j + 1]);
            el("circle", { cx: m[0], cy: m[1], r: 10, fill: "none", stroke: C.text3, "stroke-width": 2, "stroke-dasharray": "3 3" }, g);
          }
        }
        for (var k = 0; k < pb.length - 1; k++) {
          if (segX(pa[i], pa[i + 1], pb[k], pb[k + 1])) {
            n += 1;
            var q = meet(pa[i], pa[i + 1], pb[k], pb[k + 1]);
            el("text", { x: q[0], y: q[1] + 7, "text-anchor": "middle", "class": "fde-xmark", fill: C.bad }, g).textContent = "×";
          }
        }
      }
      A.forEach(function (p) { el("circle", { cx: p[0], cy: p[1], r: 6, fill: C.a }, g); });
      B.slice(0, -1).forEach(function (p) { el("circle", { cx: p[0], cy: p[1], r: 6, fill: C.b }, g); });
      el("rect", { x: depot[0] - 9, y: depot[1] - 9, width: 18, height: 18, fill: C.text }, g);
      el("text", { x: depot[0] + 16, y: depot[1] + 5, "class": "fde-note" }, g).textContent = t.depot;
      head.innerHTML = '<span class="fde-demo__hint">' + t.crossTitle + '</span><span class="fde-demo__count"><b>' + t.crossings + " " + n + "</b></span>";
    }
    draw();
    makeHandle(svg, C.b, B[B.length - 1], [10, 10, 630, 300], function (p) { B[B.length - 1] = p; draw(); });
  }

  // ---- 3. case 002 map ----
  function mapFigure(box, L, data) {
    var t = T[L], st = { v: 0, routes: true, hulls: true, viol: true };
    var bar = html("div", "fde-tabs", box);
    data.variants.forEach(function (v, i) {
      var b = html("button", "", bar, v[L]);
      b.type = "button";
      b.setAttribute("aria-pressed", i === 0 ? "true" : "false");
      b.addEventListener("click", function () { st.v = i; bar.querySelectorAll("button").forEach(function (x, k) { x.setAttribute("aria-pressed", k === i ? "true" : "false"); }); draw(); });
    });
    var stats = html("div", "fde-stats", box);
    var layers = html("div", "fde-layers", box);
    [["routes", t.routes], ["hulls", t.hulls], ["viol", t.violations]].forEach(function (l) {
      var lab = html("label", "", layers);
      var cb = html("input", "", lab);
      cb.type = "checkbox"; cb.checked = true;
      cb.addEventListener("change", function () { st[l[0]] = cb.checked; draw(); });
      lab.appendChild(document.createTextNode(l[1]));
    });
    var holder = html("div", "fde-plot", box);
    var svg = el("svg", { viewBox: "0 0 " + data.box[0] + " " + data.box[1], "class": "fde-svg is-map", role: "img", "aria-label": "Case 002 routes, hulls and violations" }, holder);
    var tip = tooltip(holder);
    var owner = [];
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var v = data.variants[st.v], S = data.stores;
      owner = new Array(S.length);
      v.routes.forEach(function (r, ri) { r.forEach(function (i) { owner[i] = ri; }); });
      if (st.hulls) v.hulls.forEach(function (h) { if (h.length >= 3) el("polygon", { points: pts(h.map(function (i) { return S[i]; })), fill: C.text, "fill-opacity": 0.04, stroke: C.hull, "stroke-width": 1.2 }, svg); });
      if (st.routes) v.routes.forEach(function (r) {
        if (!r.length) return;
        el("polyline", { points: pts([data.depot].concat(r.map(function (i) { return S[i]; }), [data.depot])), fill: "none", stroke: C.route, "stroke-width": 0.9 }, svg);
      });
      S.forEach(function (p) { el("circle", { cx: p[0], cy: p[1], r: 2.1, fill: C.store }, svg); });
      if (st.viol) {
        v.intrusions.forEach(function (x) { var p = S[x[0]]; el("circle", { cx: p[0], cy: p[1], r: 4.6, fill: "none", stroke: C.bad, "stroke-width": 1.4 }, svg); });
        v.crossings.forEach(function (c) { el("text", { x: c[0], y: c[1] + 5, "text-anchor": "middle", "class": "fde-xmark is-small", fill: C.bad }, svg).textContent = "×"; });
      }
      v.hulls.forEach(function (h, ri) {
        if (!h.length) return;
        var cx = 0, cy = 0;
        v.routes[ri].forEach(function (i) { cx += S[i][0]; cy += S[i][1]; });
        cx /= v.routes[ri].length; cy /= v.routes[ri].length;
        el("text", { x: cx, y: cy + 5, "text-anchor": "middle", "class": "fde-maplabel" }, svg).textContent = ri + 1;
      });
      el("rect", { x: data.depot[0] - 7, y: data.depot[1] - 7, width: 14, height: 14, fill: C.text }, svg);
      el("text", { x: data.depot[0] + 12, y: data.depot[1] + 5, "class": "fde-maplabel is-depot" }, svg).textContent = t.depot;
      var spread = Math.max.apply(null, v.hours) - Math.min.apply(null, v.hours);
      var total = v.hours.reduce(function (a, b) { return a + b; }, 0);
      stats.innerHTML = cell(t.score, v.score.toFixed(3)) + cell(t.intrusions, fmt(v.counts.intrusions)) + cell(t.crossings, fmt(v.counts.crossings)) +
        cell(t.longest, spread.toFixed(2) + " " + t.hours) + cell(t.total, total.toFixed(1) + " " + t.hours);
    }
    function cell(k, val) { return '<div><span>' + k + "</span><b>" + val + "</b></div>"; }
    svg.addEventListener("pointermove", function (ev) {
      var p = svgPoint(svg, ev), v = data.variants[st.v], best = -1, bd = 81;
      data.stores.forEach(function (s, i) { var d = (s[0] - p[0]) * (s[0] - p[0]) + (s[1] - p[1]) * (s[1] - p[1]); if (d < bd) { bd = d; best = i; } });
      var r = holder.getBoundingClientRect();
      if (st.viol) {
        for (var k = 0; k < v.crossings.length; k++) {
          var c = v.crossings[k];
          if ((c[0] - p[0]) * (c[0] - p[0]) + (c[1] - p[1]) * (c[1] - p[1]) < 36) { tip.show(ev.clientX - r.left, ev.clientY - r.top, "<b>" + t.crossings + "</b><br>" + t.crossOf(c[2] + 1, c[3] + 1)); return; }
        }
      }
      if (best < 0) { tip.hide(); return; }
      var s = data.stores[best], ins = v.intrusions.filter(function (x) { return x[0] === best; }).map(function (x) { return x[1] + 1; });
      tip.show(ev.clientX - r.left, ev.clientY - r.top, "<b>" + t.store + " · " + t.vehicle + " " + (owner[best] + 1) + "</b><br>" + s[2] + " " + t.kg +
        (ins.length ? '<br><span class="fde-bad">' + ins.map(t.insideOf).join(", ") + "</span>" : ""));
    });
    svg.addEventListener("pointerleave", tip.hide);
    draw();
  }

  // ---- 4. worst pairs zoom ----
  function pairFigure(box, L, data, key) {
    var t = T[L], v = data.variants.filter(function (x) { return x.key === key; })[0], S = data.stores;
    var tally = {};
    function add(a, b, kind) { var k = Math.min(a, b) + "-" + Math.max(a, b); tally[k] = tally[k] || { a: Math.min(a, b), b: Math.max(a, b), i: 0, c: 0 }; tally[k][kind] += 1; }
    v.intrusions.forEach(function (x) { add(x[1], x[2], "i"); });
    v.crossings.forEach(function (x) { add(x[2], x[3], "c"); });
    var pairs = Object.keys(tally).map(function (k) { return tally[k]; }).sort(function (p, q) { return (q.i + q.c) - (p.i + p.c); }).slice(0, 5);
    var idx = 0;
    var nav = html("div", "fde-tabs", box);
    var prev = html("button", "", nav, t.prev), rank = html("span", "fde-rank", nav), next = html("button", "", nav, t.next);
    prev.type = next.type = "button";
    prev.addEventListener("click", function () { idx = (idx + pairs.length - 1) % pairs.length; draw(); });
    next.addEventListener("click", function () { idx = (idx + 1) % pairs.length; draw(); });
    var cap = html("div", "fde-demo__head", box);
    var svg = el("svg", { "class": "fde-svg is-map", role: "img" }, box);
    var legend = html("div", "fde-demo__legend", box);
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var pr = pairs[idx], ra = v.routes[pr.a], rb = v.routes[pr.b];
      var all = ra.concat(rb).map(function (i) { return S[i]; });
      var xs = all.map(function (p) { return p[0]; }).concat([data.depot[0]]), ys = all.map(function (p) { return p[1]; }).concat([data.depot[1]]);
      var x0 = Math.min.apply(null, xs) - 20, y0 = Math.min.apply(null, ys) - 20, w = Math.max.apply(null, xs) + 20 - x0, h = Math.max.apply(null, ys) + 20 - y0;
      svg.setAttribute("viewBox", [x0, y0, w, h].join(" "));
      svg.setAttribute("aria-label", t.pair(pr.a + 1, pr.b + 1, pr.i, pr.c));
      var k = w / 600;
      S.forEach(function (p, i) { if (ra.indexOf(i) < 0 && rb.indexOf(i) < 0) el("circle", { cx: p[0], cy: p[1], r: 1.6 * k, fill: C.dim }, svg); });
      [[pr.a, C.a], [pr.b, C.b]].forEach(function (z) {
        var hull = v.hulls[z[0]].map(function (i) { return S[i]; });
        if (hull.length >= 3) el("polygon", { points: pts(hull), fill: z[1], "fill-opacity": 0.1, stroke: z[1], "stroke-width": 1.4 * k }, svg);
        el("polyline", { points: pts([data.depot].concat(v.routes[z[0]].map(function (i) { return S[i]; }), [data.depot])), fill: "none", stroke: z[1], "stroke-width": 1.3 * k, "stroke-opacity": 0.85 }, svg);
        v.routes[z[0]].forEach(function (i) { el("circle", { cx: S[i][0], cy: S[i][1], r: 2.6 * k, fill: z[1] }, svg); });
      });
      v.intrusions.forEach(function (x) {
        if (!((x[1] === pr.a && x[2] === pr.b) || (x[1] === pr.b && x[2] === pr.a))) return;
        el("circle", { cx: S[x[0]][0], cy: S[x[0]][1], r: 5.5 * k, fill: "none", stroke: C.bad, "stroke-width": 1.6 * k }, svg);
      });
      v.crossings.forEach(function (c) {
        if (Math.min(c[2], c[3]) !== pr.a || Math.max(c[2], c[3]) !== pr.b) return;
        var tx = el("text", { x: c[0], y: c[1] + 6 * k, "text-anchor": "middle", "class": "fde-xmark", fill: C.bad }, svg);
        tx.style.fontSize = (18 * k) + "px";
        tx.textContent = "×";
      });
      el("rect", { x: data.depot[0] - 7 * k, y: data.depot[1] - 7 * k, width: 14 * k, height: 14 * k, fill: C.text }, svg);
      rank.textContent = t.pairRank(idx + 1, pairs.length);
      cap.innerHTML = '<span class="fde-demo__count"><b>' + t.pair(pr.a + 1, pr.b + 1, pr.i, pr.c) + "</b></span>";
      legend.innerHTML = '<span><i class="fde-key" style="background:' + C.a + '"></i>' + t.vehicle + " " + (pr.a + 1) + '</span><span><i class="fde-key" style="background:' + C.b + '"></i>' +
        t.vehicle + " " + (pr.b + 1) + '</span><span><i class="fde-ring"></i>' + t.intrusions + '</span><span><b class="fde-x">×</b>' + t.crossings + "</span>";
    }
    draw();
  }

  // ---- 5. per-truck hours dot plot ----
  function hoursFigure(box, L, data) {
    var t = T[L];
    html("div", "fde-axis-label", box, t.axisHours);
    var holder = html("div", "fde-plot", box), tip = tooltip(holder);
    var narrow = holder.clientWidth < 560, rows = data.variants;
    var rowH = narrow ? 64 : 52, left = narrow ? 8 : 210, right = narrow ? 16 : 120, W = narrow ? holder.clientWidth : 800;
    var top = narrow ? 40 : 26, H = rows.length * rowH + top + 8;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, "class": "fde-svg", role: "img", "aria-label": "Route duration per truck for each submission" }, holder);
    var xMax = 12, pw = W - left - right;
    function sx(h) { return left + h / xMax * pw; }
    for (var g = 0; g <= xMax; g += 2) {
      el("line", { x1: sx(g), x2: sx(g), y1: 6, y2: H - 26, "class": "fde-grid" }, svg);
      el("text", { x: sx(g), y: H - 8, "text-anchor": "middle", "class": "fde-tick" }, svg).textContent = g;
    }
    rows.forEach(function (v, r) {
      var y = top + r * rowH, lo = Math.min.apply(null, v.hours), hi = Math.max.apply(null, v.hours);
      var label = narrow ? t.spread + " " + (hi - lo).toFixed(2) + " " + t.hours : null;
      el("text", narrow ? { x: left, y: y - 16, "class": "fde-rowlabel" } : { x: left - 14, y: y + 4, "text-anchor": "end", "class": "fde-rowlabel" }, svg).textContent = narrow ? v[L] + " · " + label : v[L];
      el("line", { x1: sx(lo), x2: sx(hi), y1: y, y2: y, stroke: C.range, "stroke-width": 6 }, svg);
      v.hours.forEach(function (h, i) {
        var dot = el("circle", { cx: sx(h), cy: y, r: 5, fill: v.key === "reference" ? C.s3 : C.s1, stroke: C.ink, "stroke-width": 2 }, svg);
        dot.addEventListener("pointerenter", function (ev) {
          var rr = holder.getBoundingClientRect();
          tip.show(ev.clientX - rr.left, ev.clientY - rr.top, t.dotTip(i + 1, h, v.loads[i], v.routes[i].length));
        });
        dot.addEventListener("pointerleave", tip.hide);
      });
      if (!narrow) el("text", { x: sx(hi) + 12, y: y + 4, "class": "fde-endlabel" }, svg).textContent = hi - lo < 0.01 ? t.allSame(v.hours.length, hi) : t.spread + " " + (hi - lo).toFixed(2) + " " + t.hours;
    });
  }

  // ---- 6. cost breakdown ----
  function costFigure(box, L, data) {
    var t = T[L], keys = ["time", "balance", "intrusion", "crossing"], colors = [C.s1, C.s2, C.s3, C.s4];
    html("div", "fde-axis-label", box, t.costAxis);
    var legend = html("div", "fde-demo__legend", box);
    legend.innerHTML = t.parts.map(function (p, i) { return '<span><i class="fde-key is-bar" style="background:' + colors[i] + '"></i>' + p + "</span>"; }).join("");
    var holder = html("div", "fde-plot", box), tip = tooltip(holder);
    var narrow = holder.clientWidth < 560, rows = data.variants;
    var rowH = narrow ? 62 : 46, left = narrow ? 8 : 210, right = narrow ? 16 : 120, W = narrow ? holder.clientWidth : 760;
    var top = narrow ? 26 : 12, H = rows.length * rowH + top + 18;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, "class": "fde-svg", role: "img", "aria-label": "Evaluator cost broken into its four parts" }, holder);
    var xMax = 4500000, pw = W - left - right;
    function sx(c) { return left + c / xMax * pw; }
    for (var g = 0; g <= xMax; g += 1000000) {
      el("line", { x1: sx(g), x2: sx(g), y1: 4, y2: H - 24, "class": "fde-grid" }, svg);
      el("text", { x: sx(g), y: H - 6, "text-anchor": "middle", "class": "fde-tick" }, svg).textContent = g === 0 ? "0" : (g / 1e6) + "M";
    }
    rows.forEach(function (v, r) {
      var y = top + r * rowH, x = 0;
      el("text", narrow ? { x: left, y: y - 8, "class": "fde-rowlabel" } : { x: left - 14, y: y + 15, "text-anchor": "end", "class": "fde-rowlabel" }, svg).textContent =
        narrow ? v[L] + " · " + fmt(v.cost.total) + " → " + v.score.toFixed(3) : v[L];
      keys.forEach(function (k, i) {
        var val = v.cost[k];
        if (!val) return;
        var w = Math.max(1, sx(x + val) - sx(x) - 2);
        var rect = el("rect", { x: sx(x) + (x ? 2 : 0), y: y, width: w, height: 22, fill: colors[i] }, svg);
        rect.addEventListener("pointerenter", function (ev) {
          var rr = holder.getBoundingClientRect();
          tip.show(ev.clientX - rr.left, ev.clientY - rr.top, "<b>" + v[L] + "</b><br>" + t.parts[i] + ": " + fmt(val) + " (" + (val / v.cost.total * 100).toFixed(0) + "%)");
        });
        rect.addEventListener("pointerleave", tip.hide);
        x += val;
      });
      if (!narrow) el("text", { x: sx(v.cost.total) + 10, y: y + 15, "class": "fde-endlabel" }, svg).textContent = fmt(v.cost.total) + " → " + v.score.toFixed(3);
    });
  }

  // ---- 7. the same plans priced by another company ----
  // The evaluator's cost is a price list: driving seconds × 1 + spread seconds × balance + intrusions × a + crossings × b.
  // Re-pricing the four plans shows that the winner depends on the list, not on the plans alone.
  function weightsFigure(box, L, data) {
    var t = T[L], colors = [C.s1, C.s2, C.s3, C.s4];
    var PRESETS = [[90, 600, 1800], [0, 0, 0], [0, 600, 1800]], MAX = [200, 3000, 6000];
    var w = PRESETS[0].slice();
    // Driving time absorbs the rounding left in the stored terms, so the customer's prices reproduce figure 6 exactly.
    var plans = data.variants.map(function (v) {
      return { v: v, q: [v.cost.total - v.cost.balance - v.cost.intrusion - v.cost.crossing, v.cost.balance / 90, v.counts.intrusions, v.counts.crossings] };
    });
    var bar = html("div", "fde-tabs", box);
    var presetButtons = PRESETS.map(function (p, i) {
      var b = html("button", "", bar, t.wPresets[i]);
      b.type = "button";
      b.addEventListener("click", function () { w = p.slice(); sync(); draw(); });
      return b;
    });
    var form = html("div", "fde-weights", box);
    // Sliders are squared so that small weights (the interesting ones near 1) are easy to reach.
    function toWeight(i, pos) { var x = pos / 100, v = x * x * MAX[i]; return v < 10 ? Math.round(v * 10) / 10 : Math.round(v); }
    function toPos(i, v) { return Math.round(Math.sqrt(v / MAX[i]) * 100); }
    var sliders = t.wNames.map(function (name, i) {
      var row = html("label", "", form);
      html("span", "fde-weights__name", row, name);
      var input = html("input", "", row);
      input.type = "range"; input.min = 0; input.max = 100; input.step = 1;
      var out = html("b", "fde-weights__val", row);
      input.addEventListener("input", function () { w[i] = toWeight(i, +input.value); sync(true); schedule(); });
      return { input: input, out: out };
    });
    var sum = html("p", "fde-weights__sum", box);
    sum.setAttribute("aria-live", "polite");
    html("div", "fde-axis-label", box, t.wAxis);
    var legend = html("div", "fde-demo__legend", box);
    legend.innerHTML = t.wParts.map(function (p, i) { return '<span><i class="fde-key is-bar" style="background:' + colors[i] + '"></i>' + p + "</span>"; }).join("");
    var holder = html("div", "fde-plot", box), tip = tooltip(holder);
    var narrow = holder.clientWidth < 560;
    var rowH = narrow ? 62 : 46, left = narrow ? 8 : 210, right = narrow ? 16 : 150, W = narrow ? holder.clientWidth : 760;
    var top = narrow ? 26 : 12, H = plans.length * rowH + top + 18;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, "class": "fde-svg" }, holder);
    svg.setAttribute("role", "img");

    function sync(fromSlider) {
      sliders.forEach(function (s, i) {
        if (!fromSlider) s.input.value = toPos(i, w[i]);
        s.out.textContent = t.wUnit[i](w[i]);
      });
      presetButtons.forEach(function (b, i) {
        b.setAttribute("aria-pressed", String(PRESETS[i].every(function (v, k) { return v === w[k]; })));
      });
    }
    var pending = false;
    function schedule() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; draw(); });
    }
    function niceStep(max) {
      var raw = max / 5, mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), n = raw / mag;
      return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
    }
    function tick(g, step) { return g === 0 ? "0" : step >= 1e6 ? (g / 1e6) + "M" : (g / 1e3) + "k"; }
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var rows = plans.map(function (p) {
        var parts = [p.q[0], w[0] * p.q[1], w[1] * p.q[2], w[2] * p.q[3]];
        return { p: p, parts: parts, total: parts.reduce(function (a, b) { return a + b; }, 0) };
      });
      var ref = rows.filter(function (r) { return r.p.v.key === "reference"; })[0];
      var order = rows.slice().sort(function (a, b) { return a.total - b.total; });
      rows.forEach(function (r) { r.rank = order.indexOf(r) + 1; r.score = ref.total / r.total; });
      var step = niceStep(order[order.length - 1].total), xMax = Math.ceil(order[order.length - 1].total / step) * step, pw = W - left - right;
      function sx(c) { return left + c / xMax * pw; }
      for (var g = 0; g <= xMax + 1; g += step) {
        el("line", { x1: sx(g), x2: sx(g), y1: 4, y2: H - 24, "class": "fde-grid" }, svg);
        el("text", { x: sx(g), y: H - 6, "text-anchor": "middle", "class": "fde-tick" }, svg).textContent = tick(g, step);
      }
      rows.forEach(function (r, k) {
        var v = r.p.v, y = top + k * rowH, x = 0, end = fmt(r.total) + " → " + r.score.toFixed(3) + " · " + t.wRank(r.rank);
        el("text", narrow ? { x: left, y: y - 8, "class": "fde-rowlabel" + (r.rank === 1 ? " is-winner" : "") } : { x: left - 14, y: y + 15, "text-anchor": "end", "class": "fde-rowlabel" + (r.rank === 1 ? " is-winner" : "") }, svg).textContent =
          narrow ? v[L] + " · " + end : v[L];
        r.parts.forEach(function (val, i) {
          if (!val) return;
          var bw = Math.max(1, sx(x + val) - sx(x) - 2);
          var rect = el("rect", { x: sx(x) + (x ? 2 : 0), y: y, width: bw, height: 22, fill: colors[i] }, svg);
          rect.addEventListener("pointerenter", function (ev) {
            var rr = holder.getBoundingClientRect();
            tip.show(ev.clientX - rr.left, ev.clientY - rr.top, "<b>" + v[L] + "</b><br>" + t.wParts[i] + ": " + fmt(val) + " (" + (val / r.total * 100).toFixed(0) + "%)");
          });
          rect.addEventListener("pointerleave", tip.hide);
          x += val;
        });
        if (!narrow) el("text", { x: sx(r.total) + 10, y: y + 15, "class": "fde-endlabel" + (r.rank === 1 ? " is-winner" : "") }, svg).textContent = end;
      });
      var win = order[0];
      sum.textContent = t.wSum(win.p.v[L], win === ref, ref.rank, rows.length);
      svg.setAttribute("aria-label", sum.textContent);
    }
    sync();
    draw();
  }

  // ---- 8. case 031 KS curves ----
  function ksFigure(box, L, d) {
    var t = T[L];
    var series = [
      { key: "reference", name: L === "ko" ? "FDE 정답" : "FDE reference", color: C.s1 },
      { key: "logistic", name: L === "ko" ? "A · 로지스틱 (120개 점수)" : "A · logistic (120 distinct scores)", color: C.s2 },
      { key: "banded", name: L === "ko" ? "E · 0.1 단위 반올림 (11개 등급)" : "E · rounded to 0.1 (11 bands)", color: C.s3 }
    ];
    var legend = html("div", "fde-demo__legend", box);
    legend.innerHTML = series.map(function (s) { return '<span><i class="fde-key" style="background:' + s.color + '"></i>' + s.name + "</span>"; }).join("") + '<span><i class="fde-dash"></i>' + t.diag + "</span>";
    var holder = html("div", "fde-plot is-square", box), tip = tooltip(holder);
    var W = Math.min(520, holder.clientWidth || 520), H = W, m = { l: 56, r: 16, t: 14, b: 50 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, "class": "fde-svg", role: "img", "aria-label": "KS curves for case 031" }, holder);
    function sx(x) { return m.l + x * pw; }
    function sy(y) { return m.t + (1 - y) * ph; }
    for (var g = 0; g <= 1.0001; g += 0.2) {
      el("line", { x1: sx(g), x2: sx(g), y1: sy(0), y2: sy(1), "class": "fde-grid" }, svg);
      el("line", { x1: sx(0), x2: sx(1), y1: sy(g), y2: sy(g), "class": "fde-grid" }, svg);
      el("text", { x: sx(g), y: sy(0) + 18, "text-anchor": "middle", "class": "fde-tick" }, svg).textContent = Math.round(g * 100) + "%";
      el("text", { x: sx(0) - 8, y: sy(g) + 4, "text-anchor": "end", "class": "fde-tick" }, svg).textContent = Math.round(g * 100) + "%";
    }
    el("text", { x: sx(0.5), y: H - 8, "text-anchor": "middle", "class": "fde-tick" }, svg).textContent = t.ksX;
    var yl = el("text", { x: 14, y: sy(0.5), "text-anchor": "middle", "class": "fde-tick", transform: "rotate(-90 14 " + sy(0.5) + ")" }, svg);
    yl.textContent = t.ksY;
    el("line", { x1: sx(0), y1: sy(0), x2: sx(1), y2: sy(1), stroke: C.text3, "stroke-dasharray": "4 4" }, svg);
    series.forEach(function (s, i) {
      var c = d.curves[s.key];
      el("polyline", { points: c.points.map(function (p) { return sx(p[0]) + "," + sy(p[1]); }).join(" "), fill: "none", stroke: s.color, "stroke-width": 2 }, svg);
      c.points.forEach(function (p, k) { if (s.key === "banded" && k > 0) el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 4, fill: s.color, stroke: C.ink, "stroke-width": 2 }, svg); });
      var a = c.ksAt;
      el("line", { x1: sx(a[0]), x2: sx(a[0]), y1: sy(a[0]), y2: sy(a[1]), stroke: s.color, "stroke-width": 3, "stroke-opacity": 0.55 }, svg);
      el("text", { x: sx(a[0]) + 6, y: sy(a[0]) + 14, "class": "fde-endlabel" }, svg).textContent = t.ksAt + " " + c.ks.toFixed(2);
    });
    var dot = el("circle", { r: 5, fill: "none", stroke: C.text, "stroke-width": 2, visibility: "hidden" }, svg);
    svg.addEventListener("pointermove", function (ev) {
      var p = svgPoint(svg, ev), best = null, bd = 400;
      series.forEach(function (s) {
        d.curves[s.key].points.forEach(function (q) {
          var dd = (sx(q[0]) - p[0]) * (sx(q[0]) - p[0]) + (sy(q[1]) - p[1]) * (sy(q[1]) - p[1]);
          if (dd < bd) { bd = dd; best = { s: s, q: q }; }
        });
      });
      if (!best) { tip.hide(); dot.setAttribute("visibility", "hidden"); return; }
      dot.setAttribute("cx", sx(best.q[0])); dot.setAttribute("cy", sy(best.q[1])); dot.setAttribute("visibility", "visible");
      var rr = holder.getBoundingClientRect();
      tip.show(ev.clientX - rr.left, ev.clientY - rr.top, t.ksTip(best.s.name, best.q[0], best.q[1]));
    });
    svg.addEventListener("pointerleave", function () { tip.hide(); dot.setAttribute("visibility", "hidden"); });
  }

  var cache = {};
  function load(url) {
    if (!cache[url]) cache[url] = fetch(url).then(function (r) { return r.json(); });
    return cache[url];
  }
  // Layout depends on the figure's width, so a figure in the hidden language block waits until it is shown.
  function whenVisible(box, fn) {
    if (box.clientWidth > 0 || !window.ResizeObserver) return fn();
    var ro = new ResizeObserver(function () { if (box.clientWidth > 0) { ro.disconnect(); fn(); } });
    ro.observe(box);
  }
  function init() {
    document.querySelectorAll("[data-fde]").forEach(function (box) { whenVisible(box, function () { render(box); }); });
  }
  function render(box) {
    {
      var kind = box.getAttribute("data-fde"), L = langOf(box), src = box.getAttribute("data-src");
      if (kind === "hull") return hullDemo(box, L);
      if (kind === "cross") return crossDemo(box, L);
      load(src).then(function (data) {
        if (kind === "map") mapFigure(box, L, data);
        else if (kind === "pairs") pairFigure(box, L, data, box.getAttribute("data-variant") || "greedy");
        else if (kind === "hours") hoursFigure(box, L, data);
        else if (kind === "cost") costFigure(box, L, data);
        else if (kind === "weights") weightsFigure(box, L, data);
        else if (kind === "ks") ksFigure(box, L, data);
      });
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
