// Interactive figures: the fusion overview (Figure 3) and the per-model
// breakdown (Figures 4-5). Data comes from assets/data.js (window.SER), which
// `python -m cv.site_data` writes from the cross-validation results.
(() => {
  const D = window.SER;
  const L = D.labels;
  const SVG = "http://www.w3.org/2000/svg";
  const $ = (sel) => document.querySelector(sel);

  // Round half up on the decimal value, so 0.7125 shows as 71.3% and a 0.175
  // gap as 17.5, whatever the float error.
  const r1 = (v) => (Math.round(v * 1000 + 1e-9) / 10).toFixed(1);
  const pct1 = (v) => r1(v) + "%";
  const pct0 = (v) => Math.round(v * 100) + "%";
  const pts = (v) => (v >= 0 ? "+" : "−") + r1(Math.abs(v)) + " pts";

  function h(tag, attrs = {}, parent, text) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  function s(tag, attrs = {}, parent, text) {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  // Per-class precision, recall and F1 from a confusion matrix (rows = actual).
  function scores(cm) {
    return cm.map((row, i) => {
      const tp = row[i];
      const actual = row.reduce((a, b) => a + b, 0);
      const predicted = cm.reduce((a, r) => a + r[i], 0);
      const p = predicted ? tp / predicted : 0;
      const r = actual ? tp / actual : 0;
      return [p, r, p + r ? (2 * p * r) / (p + r) : 0];
    });
  }

  // Horizontal bar with a 4px rounded data end and a square baseline end.
  function barPath(x0, x1, y, hgt) {
    const r = Math.min(4, hgt / 2, Math.max(0, x1 - x0));
    return `M${x0},${y}H${x1 - r}A${r},${r} 0 0 1 ${x1},${y + r}V${y + hgt - r}A${r},${r} 0 0 1 ${x1 - r},${y + hgt}H${x0}Z`;
  }

  // ── Tooltip: one element, filled with textContent, shown on hover and focus.
  const tip = $("#tip");
  function fillTip(title, rows) {
    tip.textContent = "";
    if (title) h("div", { class: "tip-title" }, tip, title);
    for (const [color, value, name] of rows) {
      const row = h("div", { class: "tip-row" }, tip);
      if (color) h("span", { class: "swatch line", style: `background:var(${color})` }, row);
      h("b", {}, row, value);
      h("span", { class: "name" }, row, name);
    }
  }
  function place(x, y) {
    const pad = 14;
    const { width, height } = tip.getBoundingClientRect();
    let left = x + pad;
    let top = y + pad;
    if (left + width > window.innerWidth - 8) left = x - width - pad;
    if (top + height > window.innerHeight - 8) top = y - height - pad;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }
  function bindTip(node, content) {
    const show = () => { fillTip(...content()); tip.classList.add("on"); };
    node.addEventListener("pointerenter", (e) => { show(); place(e.clientX, e.clientY); });
    node.addEventListener("pointermove", (e) => place(e.clientX, e.clientY));
    node.addEventListener("pointerleave", () => tip.classList.remove("on"));
    node.addEventListener("focus", () => {
      show();
      const r = node.getBoundingClientRect();
      place(r.left + r.width / 2, r.top + r.height / 2);
    });
    node.addEventListener("blur", () => tip.classList.remove("on"));
  }

  // ── Figure 2: speech alone → speech + face, with face alone as a reference.
  function drawFusion() {
    const host = $("#fusion-chart");
    host.textContent = "";
    const W = host.clientWidth;
    const narrow = W < 520;
    const left = narrow ? 4 : 220, right = 70, top = 30, bottom = 26;
    const rowH = narrow ? 68 : 50;
    const H = top + rowH * D.speech.length + bottom;
    const accs = D.speech.flatMap((m) => [m.alone.acc, m.fused.acc]).concat(D.face.acc);
    const lo = Math.floor((Math.min(...accs) - 0.02) * 20) / 20;
    const hi = Math.ceil((Math.max(...accs) + 0.03) * 20) / 20;
    const X = (v) => left + ((v - lo) / (hi - lo)) * (W - left - right);
    const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Accuracy of each speech model alone and fused with the face model" }, host);

    const ticks = [];
    for (let t = lo; t <= hi + 1e-9; t += 0.05) ticks.push(Math.round(t * 100) / 100);
    for (const t of ticks) {
      s("line", { x1: X(t), x2: X(t), y1: top - 8, y2: H - bottom, stroke: "var(--grid)", "stroke-width": 1 }, svg);
      s("text", { class: "tick", x: X(t), y: H - 8, "text-anchor": "middle" }, svg, Math.round(t * 100) + "%");
    }
    s("line", { x1: X(D.face.acc), x2: X(D.face.acc), y1: top - 8, y2: H - bottom, stroke: "var(--face)", "stroke-width": 2 }, svg);
    s("text", { class: "halo", x: X(D.face.acc) + 6, y: top - 14 }, svg, "Face only " + pct1(D.face.acc));

    D.speech.forEach((m, i) => {
      const y0 = top + rowH * i;
      const cy = y0 + (narrow ? 46 : rowH / 2 + 4);
      const hit = s("rect", { class: "hit", x: 0, y: y0, width: W, height: rowH, tabindex: 0,
        "aria-label": `${m.name}: ${pct1(m.alone.acc)} alone, ${pct1(m.fused.acc)} with face` }, svg);
      bindTip(hit, () => [m.name, [
        ["--speech", pct1(m.alone.acc), `speech only (95% CI ${r1(m.alone.lo)}–${r1(m.alone.hi)})`],
        ["--fused", pct1(m.fused.acc), `speech + face (95% CI ${r1(m.fused.lo)}–${r1(m.fused.hi)})`],
        [null, pts(m.fused.acc - m.alone.acc), "from adding face"],
        [null, String(m.alone.seeds), m.alone.seeds > 1 ? "seeds, averaged" : "seed"],
      ]]);
      const label = narrow
        ? { x: left, y: y0 + 16 }
        : { x: left - 16, y: cy + 4, "text-anchor": "end" };
      const name = s("text", { class: "row-label halo", "pointer-events": "none", ...label }, svg, m.name);
      if (narrow) {
        // Above the row, the label crosses the face-only line: mask it out.
        const bb = name.getBBox();
        svg.insertBefore(s("rect", { x: bb.x - 3, y: bb.y, width: bb.width + 6, height: bb.height, fill: "var(--bg)", "pointer-events": "none" }), name);
      }
      s("line", { x1: X(m.alone.acc), x2: X(m.fused.acc), y1: cy, y2: cy, stroke: "var(--rule)", "stroke-width": 2, "pointer-events": "none" }, svg);
      for (const [v, color] of [[m.alone.acc, "--speech"], [m.fused.acc, "--fused"]]) {
        s("circle", { cx: X(v), cy, r: 6, fill: `var(${color})`, stroke: "var(--bg)", "stroke-width": 2, "pointer-events": "none" }, svg);
        s("text", { class: "halo", x: X(v), y: cy - 12, "text-anchor": "middle", "pointer-events": "none" }, svg, pct1(v));
      }
      s("text", { class: "strong", x: X(m.fused.acc) + 12, y: cy + 4, "pointer-events": "none" }, svg, pts(m.fused.acc - m.alone.acc));
    });
  }

  // ── Figures 3-4 and the per-class table, for one speech model at a time.
  let current = D.speech.find((m) => m.id === "lora");

  function drawRecall(m) {
    const host = $("#recall-chart");
    host.textContent = "";
    const W = host.clientWidth;
    const alone = scores(m.alone.cm), fused = scores(m.fused.cm);
    const left = 76, right = 44, top = 6, bottom = 26;
    const barH = 7, gap = 2, groupGap = 14;
    const groupH = 3 * barH + 2 * gap;
    const rowH = groupH + groupGap;
    const H = top + L.length * rowH - groupGap + bottom + 6;
    const X = (v) => left + v * (W - left - right);
    const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `Recall for each emotion: ${m.name} alone, the face model, and both fused` }, host);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      s("line", { x1: X(t), x2: X(t), y1: top - 4, y2: H - bottom, stroke: t ? "var(--grid)" : "var(--rule)", "stroke-width": 1 }, svg);
      s("text", { class: "tick", x: X(t), y: H - 8, "text-anchor": "middle" }, svg, Math.round(t * 100) + "%");
    }
    L.forEach((emo, i) => {
      const y0 = top + i * rowH;
      const series = [
        ["--speech", alone[i][1], "speech only"],
        ["--face", D.face.prf[i][1], "face only"],
        ["--fused", fused[i][1], "speech + face"],
      ];
      const hit = s("rect", { class: "hit", x: 0, y: y0 - groupGap / 2 + 2, width: W, height: rowH - 4, tabindex: 0,
        "aria-label": `${emo}: ` + series.map(([, v, n]) => `${n} ${pct0(v)}`).join(", ") }, svg);
      bindTip(hit, () => [`${emo}, recall`, series.map(([c, v, n]) => [c, pct0(v), n])]);
      s("text", { class: "row-label", x: left - 10, y: y0 + groupH / 2 + 4, "text-anchor": "end", "pointer-events": "none" }, svg, emo);
      series.forEach(([color, v], j) => {
        s("path", { d: barPath(X(0), X(v), y0 + j * (barH + gap), barH), fill: `var(${color})`, "pointer-events": "none" }, svg);
      });
      s("text", { x: X(fused[i][1]) + 6, y: y0 + 2 * (barH + gap) + barH, "pointer-events": "none" }, svg, pct0(fused[i][1]));
    });

    // Under the figure: the biggest recall gains, and where the face model is weak.
    const gains = L.map((emo, i) => [emo, fused[i][1] - alone[i][1]]).filter(([, g]) => g > 0.005).sort((a, b) => b[1] - a[1]).slice(0, 3);
    let text = gains.length
      ? `Fusion lifts recall most on ${gains.map(([e, g], k) => `${e} (+${Math.round(g * 100)}${k ? "" : " pts"})`).join(", ").replace(/, ([^,]*)$/, " and $1")}. `
      : "";
    // The two emotions where speech is furthest ahead of the face model.
    const [a1, a2] = L.map((emo, i) => [i, alone[i][1] - D.face.prf[i][1]]).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([i]) => i);
    text += `Alone, the face model catches ${pct0(D.face.prf[a1][1])} of ${L[a1]} and ${pct0(D.face.prf[a2][1])} of ${L[a2]} clips, where this speech model gets ${pct0(alone[a1][1])} and ${pct0(alone[a2][1])}.`;
    $("#insight").textContent = text;
  }

  function bin(share) {
    if (share <= 0) return "z";
    if (share < 0.08) return "b1";
    if (share < 0.2) return "b2";
    if (share < 0.4) return "b3";
    if (share < 0.6) return "b4";
    if (share < 0.8) return "b5";
    return "b6";
  }
  const ABBR = { angry: "ang", disgust: "dis", fearful: "fea", happy: "hap", neutral: "neu", sad: "sad", surprised: "sur" };

  function drawMatrix(host, title, cm, accuracy) {
    host.textContent = "";
    h("h4", {}, host, title);
    h("div", { class: "note" }, host, `accuracy ${pct1(accuracy)}`);
    const table = h("table", { class: "cm" }, host);
    const head = h("tr", {}, h("thead", {}, table));
    h("th", { scope: "col" }, head);
    for (const emo of L) h("th", { scope: "col", title: emo }, head, ABBR[emo]);
    const body = h("tbody", {}, table);
    cm.forEach((row, i) => {
      const tr = h("tr", {}, body);
      h("th", { scope: "row" }, tr, L[i]);
      const total = Math.round(row.reduce((a, b) => a + b, 0));
      row.forEach((n, j) => {
        const k = Math.round(n);
        const td = h("td", { class: bin(n / total) }, tr, k ? String(k) : "");
        bindTip(td, () => [`${L[i]} → predicted ${L[j]}`, [[null, String(k), `of ${total} clips (${pct0(n / total)})`]]]);
      });
    });
  }

  function drawTable(m) {
    const host = $("#score-table");
    host.textContent = "";
    const table = h("table", { class: "booktabs scores" }, host);
    const thead = h("thead", {}, table);
    const top = h("tr", {}, thead);
    h("th", {}, top);
    for (const name of [m.name, "Face only", `${m.name} + face`]) h("th", { colspan: 3, scope: "colgroup", style: "text-align:center" }, top, name);
    const sub = h("tr", {}, thead);
    h("th", { scope: "col" }, sub, "Emotion");
    for (let k = 0; k < 3; k++) for (const c of ["P", "R", "F1"]) h("th", { scope: "col" }, sub, c);
    const tbody = h("tbody", {}, table);
    const a = scores(m.alone.cm), f = scores(m.fused.cm);
    L.forEach((emo, i) => {
      const tr = h("tr", {}, tbody);
      h("td", {}, tr, emo);
      for (const v of [...a[i], ...D.face.prf[i], ...f[i]]) h("td", {}, tr, v.toFixed(2));
    });
  }

  function drawModel() {
    const m = current;
    $("#model-summary").textContent =
      `${m.detail}. Accuracy over all 24 actors ${pct1(m.alone.acc)} alone, ${pct1(m.fused.acc)} with the face model (${pts(m.fused.acc - m.alone.acc)}); macro-F1 ${m.alone.f1.toFixed(3)} → ${m.fused.f1.toFixed(3)}; ${m.alone.seeds > 1 ? `mean of ${m.alone.seeds} seeds` : "one seed"}.`;
    drawRecall(m);
    drawMatrix($("#cm-alone"), `${m.name} alone`, m.alone.cm, m.alone.acc);
    drawMatrix($("#cm-fused"), `${m.name} + face`, m.fused.cm, m.fused.acc);
    drawTable(m);
  }

  const group = $("#model-select");
  for (const m of D.speech) {
    const label = h("label", {}, group);
    const input = h("input", { type: "radio", name: "speech-model", value: m.id }, label);
    input.checked = m === current;
    h("span", {}, label, m.name);
    input.addEventListener("change", () => { current = m; drawModel(); });
  }

  drawFusion();
  drawModel();
  let lastWidth = 0;
  new ResizeObserver(() => {
    const w = $("#fusion-chart").clientWidth;
    if (w === lastWidth) return;
    lastWidth = w;
    drawFusion();
    drawRecall(current);
  }).observe(document.body);
})();
