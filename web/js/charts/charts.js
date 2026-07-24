// Viz 1 (success-rate bars) + Viz 2 (density sweep) for the results chapter.
//
// Split into two layers so the module stays Node-importable (smoke-tested):
//   - linearScale is pure (no DOM) and unit-tested.
//   - chartFrame draws into a ctx passed by the caller; its scaling math is
//     linearScale-based and DOM-free. It only touches the ctx argument.
//   - renderSuccessBars / renderDensitySweep are the DOM controllers; all
//     canvas/window access lives inside them, never at module top level.

import { fitCanvas, clearCanvas } from "../util/canvas.js";

// ---- Pure helpers (no DOM) ----

// Map a numeric domain [d0,d1] to a numeric range [r0,r1]. Returns a function.
// Handles inverted ranges (e.g. screen-y grows downward) and degenerate domains.
export function linearScale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  if (span === 0) {
    const mid = (r0 + r1) / 2;
    return () => mid;
  }
  return (x) => r0 + ((x - d0) / span) * (r1 - r0);
}

// ---- Shared palette ----

const COLORS = {
  bg: "#0b0d10",
  axis: "#2a3138",
  grid: "#161b20",
  label: "#9aa4b2",
  text: "#cdd6e0",
  polyKAN: "#39ff14",
  relu: "#ffb454",
  prelu: "#22d3ee",
  silu: "#e879f9",
  sigmoid: "#f472b6",
  muted: "#6b7280",
  tooltipBg: "rgba(11,13,16,0.92)",
};

// Viz 1 color rule: green = PolyKAN, amber = ReLU, gray = everything else.
// Exact-name match so "LeakyReLU" is NOT mistaken for "ReLU".
function barColor(name) {
  const n = String(name).toLowerCase();
  if (n === "polykan") return COLORS.polyKAN;
  if (n === "relu") return COLORS.relu;
  return COLORS.muted;
}

// Viz 2 series (fixed order + colors per spec).
const DENSITY_SERIES = [
  { key: "polyKAN", label: "PolyKAN", color: COLORS.polyKAN },
  { key: "prelu",   label: "PReLU",   color: COLORS.prelu },
  { key: "silu",    label: "SiLU",    color: COLORS.silu },
  { key: "relu",    label: "ReLU",    color: COLORS.relu },
];

// Viz 3 ablation: 2 model families × 3 training conditions. Green family for
// PolyKAN, cyan family for PReLU (per spec).
const ABLATION_CONDITIONS = [
  { key: "full",       label: "full" },
  { key: "actOnly",    label: "activations only" },
  { key: "weightOnly", label: "weights only" },
];
const ABLATION_SERIES = [
  { key: "polyKAN", label: "PolyKAN", color: COLORS.polyKAN },
  { key: "prelu",   label: "PReLU",   color: COLORS.prelu },
];

// Viz 4 PCA: selectable activations. Button order is fixed for the selector.
const PCA_ACTIVATIONS = [
  { key: "polyKAN", label: "PolyKAN", color: COLORS.polyKAN },
  { key: "relu",    label: "ReLU",    color: COLORS.relu },
  { key: "prelu",   label: "PReLU",   color: COLORS.prelu },
  { key: "sigmoid", label: "Sigmoid", color: COLORS.sigmoid },
];

// Map a loss in [0,1] to a color along a light-orange -> dark-purple ramp,
// echoing the paper's Fig 1 parameter-space coloring (low loss = light).
function lossColor(loss) {
  const t = Math.max(0, Math.min(1, loss));
  const r = Math.round(255 + (59 - 255) * t);
  const g = Math.round(209 + (7 - 209) * t);
  const b = Math.round(160 + (100 - 160) * t);
  return `rgb(${r},${g},${b})`;
}

// Round tick values inside [lo,hi] on a 0.2 grid.
function niceXTicks(lo, hi) {
  const out = [];
  for (let v = 0; v <= 1.0001; v += 0.2) {
    const r = Math.round(v * 100) / 100;
    if (r >= lo && r <= hi) out.push(r);
  }
  return out;
}

// ---- Shared chart frame: background, padding, axes, ticks, labels. ----
// Returns { plotW, plotH, pad, xScale, yScale }. xScale/yScale are null when the
// matching *Domain option is omitted (callers may lay out a categorical axis
// themselves, as Viz 1 does for its activation names).
export function chartFrame(ctx, w, h, opts = {}) {
  const pad = opts.pad || { l: 48, r: 16, t: 16, b: 34 };
  const colors = opts.colors || COLORS;
  const plotW = Math.max(1, w - pad.l - pad.r);
  const plotH = Math.max(1, h - pad.t - pad.b);

  if (opts.bg !== false) {
    clearCanvas(ctx, w, h, colors.bg);
  }

  let xScale = null;
  let yScale = null;

  if (opts.xDomain) {
    xScale = linearScale(opts.xDomain, [pad.l, pad.l + plotW]);
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of (opts.xTicks || [])) {
      const v = typeof t === "number" ? t : t.value;
      const label = typeof t === "number" ? String(v) : (t.label != null ? t.label : String(v));
      const x = xScale(v);
      ctx.strokeStyle = colors.grid;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + plotH); ctx.stroke();
      ctx.fillStyle = colors.label;
      ctx.fillText(label, x, pad.t + plotH + 6);
    }
    ctx.strokeStyle = colors.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t + plotH); ctx.lineTo(pad.l + plotW, pad.t + plotH); ctx.stroke();
  }

  if (opts.yDomain) {
    yScale = linearScale(opts.yDomain, [pad.t + plotH, pad.t]); // inverted: y down on screen
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of (opts.yTicks || [])) {
      const v = typeof t === "number" ? t : t.value;
      const label = typeof t === "number" ? String(v) : (t.label != null ? t.label : String(v));
      const y = yScale(v);
      ctx.strokeStyle = colors.grid;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + plotW, y); ctx.stroke();
      ctx.fillStyle = colors.label;
      ctx.fillText(label, pad.l - 8, y);
    }
    ctx.strokeStyle = colors.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + plotH); ctx.stroke();
  }

  if (opts.border) {
    ctx.strokeStyle = colors.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l, pad.t, plotW, plotH);
  }

  ctx.fillStyle = colors.label;
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  if (opts.xLabel) {
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(opts.xLabel, pad.l + plotW / 2, h - 4);
  }
  if (opts.yLabel) {
    ctx.save();
    ctx.translate(12, pad.t + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(opts.yLabel, 0, 0);
    ctx.restore();
  }

  return { plotW, plotH, pad, xScale, yScale, colors };
}

// Track canvases that already have interaction listeners bound (single-bind).
const BOUND = new WeakSet();

// ---- Viz 1: success-rate horizontal bars. ----
// DOM controller. All canvas/window access is inside this function. Returns
// { redraw }; a no-op early-return when there is no canvas/DOM (e.g. Node).
export function renderSuccessBars(canvas, data) {
  if (!canvas || typeof canvas.getContext !== "function") return null;
  const doc = canvas.ownerDocument || (typeof document !== "undefined" ? document : null);
  if (!doc) return null;
  const win = doc.defaultView || (typeof window !== "undefined" ? window : null);

  const sorted = Array.isArray(data)
    ? data.slice().sort((a, b) => b.rate - a.rate)
    : [];

  let hoverIndex = -1;
  let bars = []; // populated by draw(); each { x, y, w, h, data, color }

  function draw() {
    const view = fitCanvas(canvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;

    const pad = { l: 96, r: 44, t: 16, b: 40 };
    const { plotW, plotH, xScale } = chartFrame(ctx, cssW, cssH, {
      pad,
      xDomain: [0, 1],
      xTicks: [0, 0.25, 0.5, 0.75, 1],
      xLabel: "success rate",
    });

    const n = sorted.length;
    const rowH = n > 0 ? plotH / n : 0;
    const barH = Math.max(2, rowH * 0.62);
    const x0 = xScale(0);

    bars = [];
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    for (let i = 0; i < n; i++) {
      const d = sorted[i];
      const yCenter = pad.t + rowH * (i + 0.5);
      const y = yCenter - barH / 2;
      const rate = Math.max(0, Math.min(1, d.rate));
      const w = Math.max(0, xScale(rate) - x0);
      const color = barColor(d.name);
      bars.push({ x: x0, y, w, h: barH, data: d, color });

      // bar
      ctx.fillStyle = color;
      ctx.globalAlpha = i === hoverIndex ? 1 : 0.88;
      ctx.fillRect(x0, y, w, barH);
      ctx.globalAlpha = 1;

      // name (left of axis)
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(d.name, x0 - 8, yCenter);

      // rate (right of bar end)
      ctx.fillStyle = color;
      ctx.textAlign = "left";
      ctx.fillText(d.rate.toFixed(2), x0 + w + 6, yCenter);
    }

    // tooltip overlay for the hovered row
    if (hoverIndex >= 0 && hoverIndex < bars.length) {
      const bar = bars[hoverIndex];
      const d = bar.data;
      const lines = [
        d.name,
        `rate: ${d.rate.toFixed(2)}`,
        `params: ${d.params}`,
        `monotonic: ${d.monotonic ? "yes" : "no"}`,
        `differentiable: ${d.differentiable ? "yes" : "no"}`,
      ];
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
      let tw = 0;
      for (const ln of lines) tw = Math.max(tw, ctx.measureText(ln).width);
      const bw = tw + 16;
      const bh = lines.length * 16 + 12;
      let bx = bar.x + bar.w + 10;
      if (bx + bw > cssW - 4) bx = Math.max(4, bar.x - bw - 10);
      let by = bar.y - 4;
      if (by + bh > cssH - 4) by = cssH - bh - 4;
      if (by < 4) by = 4;

      ctx.fillStyle = COLORS.tooltipBg;
      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeStyle = bar.color;
      ctx.lineWidth = 1;
      ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      for (let i = 0; i < lines.length; i++) {
        ctx.fillStyle = i === 0 ? bar.color : COLORS.text;
        ctx.fillText(lines[i], bx + 8, by + 7 + i * 16);
      }
    }
  }

  function onMove(e) {
    const rect = canvas.getBoundingClientRect();
    const my = e.clientY - rect.top;
    let idx = -1;
    for (let i = 0; i < bars.length; i++) {
      const b = bars[i];
      if (my >= b.y - 3 && my <= b.y + b.h + 3) { idx = i; break; }
    }
    canvas.style.cursor = idx >= 0 ? "pointer" : "default";
    if (idx !== hoverIndex) { hoverIndex = idx; draw(); }
  }
  function onLeave() {
    canvas.style.cursor = "default";
    if (hoverIndex !== -1) { hoverIndex = -1; draw(); }
  }

  if (!BOUND.has(canvas)) {
    BOUND.add(canvas);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);
    if (win) win.addEventListener("resize", draw);
  }
  draw();
  return { redraw: draw };
}

// ---- Viz 2: density sweep multi-line chart with toggleable legend. ----
// DOM controller. All canvas/window access is inside this function.
export function renderDensitySweep(canvas, data) {
  if (!canvas || typeof canvas.getContext !== "function") return null;
  const doc = canvas.ownerDocument || (typeof document !== "undefined" ? document : null);
  if (!doc) return null;
  const win = doc.defaultView || (typeof window !== "undefined" ? window : null);

  const points = data && Array.isArray(data.points) ? data.points : [];

  let dMin = Infinity;
  let dMax = -Infinity;
  for (const p of points) {
    if (typeof p.density !== "number") continue;
    if (p.density < dMin) dMin = p.density;
    if (p.density > dMax) dMax = p.density;
  }
  if (!isFinite(dMin) || !isFinite(dMax)) { dMin = 0; dMax = 1; }

  const enabled = Object.create(null);
  for (const s of DENSITY_SERIES) enabled[s.key] = true;

  let legendBoxes = []; // populated by draw()

  function draw() {
    const view = fitCanvas(canvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;

    const pad = { l: 48, r: 16, t: 38, b: 40 };
    const { plotW, plotH, xScale, yScale } = chartFrame(ctx, cssW, cssH, {
      pad,
      xDomain: [dMin, dMax],
      xTicks: niceXTicks(dMin, dMax),
      yDomain: [0, 1],
      yTicks: [0, 0.25, 0.5, 0.75, 1],
      xLabel: "initial density",
      yLabel: "success rate",
    });

    // lines, clipped to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, pad.t, plotW, plotH);
    ctx.clip();
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    for (const s of DENSITY_SERIES) {
      if (!enabled[s.key]) continue;
      ctx.strokeStyle = s.color;
      ctx.beginPath();
      let started = false;
      for (const p of points) {
        const v = p[s.key];
        if (typeof v !== "number") continue;
        const X = xScale(p.density);
        const Y = yScale(v);
        if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
      }
      ctx.stroke();
    }
    ctx.restore();

    // toggleable legend (top-right)
    legendBoxes = [];
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "middle";
    let x = cssW - pad.r;
    for (let i = DENSITY_SERIES.length - 1; i >= 0; i--) {
      const s = DENSITY_SERIES[i];
      const lw = ctx.measureText(s.label).width;
      const itemW = 16 + 6 + lw + 14;
      const left = x - itemW;
      const top = pad.t - 26;
      ctx.globalAlpha = enabled[s.key] ? 1 : 0.35;
      ctx.fillStyle = s.color;
      ctx.fillRect(left, top + 6, 16, 10);
      ctx.globalAlpha = 1;
      ctx.fillStyle = enabled[s.key] ? COLORS.text : COLORS.label;
      ctx.textAlign = "left";
      ctx.fillText(s.label, left + 16 + 6, top + 11);
      legendBoxes.push({ key: s.key, x: left, y: top, w: itemW, h: 22 });
      x = left;
    }
  }

  function pickLegend(mx, my) {
    for (const b of legendBoxes) {
      if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) return b.key;
    }
    return null;
  }

  function onMove(e) {
    const rect = canvas.getBoundingClientRect();
    const key = pickLegend(e.clientX - rect.left, e.clientY - rect.top);
    canvas.style.cursor = key ? "pointer" : "default";
  }
  function onClick(e) {
    const rect = canvas.getBoundingClientRect();
    const key = pickLegend(e.clientX - rect.left, e.clientY - rect.top);
    if (key) { enabled[key] = !enabled[key]; draw(); }
  }

  if (!BOUND.has(canvas)) {
    BOUND.add(canvas);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("click", onClick);
    if (win) win.addEventListener("resize", draw);
  }
  draw();
  return { redraw: draw };
}

// ---- Viz 3: ablation grouped bars (PolyKAN vs PReLU across 3 conditions). ----
// DOM controller. All canvas/window access lives inside this function. Renders
// two bars per condition, annotates each bar's trainable-param count, draws a
// fixed headline callout, and supports hover tooltips. Returns { redraw }.
export function renderAblation(canvas, data) {
  if (!canvas || typeof canvas.getContext !== "function") return null;
  const doc = canvas.ownerDocument || (typeof document !== "undefined" ? document : null);
  if (!doc) return null;
  const win = doc.defaultView || (typeof window !== "undefined" ? window : null);

  const ablation = data && typeof data === "object" ? data : {};
  let hover = -1;  // index into bars; -1 when nothing hovered
  let bars = [];    // populated by draw(): { x, y, w, h, color, rate, params, condLabel, seriesLabel }

  function draw() {
    const view = fitCanvas(canvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;

    const pad = { l: 48, r: 20, t: 56, b: 64 };
    const { plotW, plotH, pad: P, yScale } = chartFrame(ctx, cssW, cssH, {
      pad,
      yDomain: [0, 1],
      yTicks: [0, 0.25, 0.5, 0.75, 1],
      yLabel: "success rate",
    });

    // Headline callout (top center).
    ctx.font = "bold 13px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = COLORS.polyKAN;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("PolyKAN: 128/128 with or without weight training", P.l + plotW / 2, 14);

    // Legend (top-left, below the callout).
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "middle";
    let lx = P.l + 2;
    const ly = P.t - 20;
    for (const s of ABLATION_SERIES) {
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, ly + 4, 14, 10);
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = "left";
      ctx.fillText(s.label, lx + 18, ly + 9);
      lx += 18 + ctx.measureText(s.label).width + 20;
    }

    const nG = ABLATION_CONDITIONS.length;
    const groupW = plotW / nG;
    const innerPad = 0.16;
    const barAreaW = groupW * (1 - 2 * innerPad);
    const slotW = barAreaW / ABLATION_SERIES.length;
    const barW = slotW * 0.82;
    const baselineY = yScale(0);

    bars = [];
    for (let gi = 0; gi < nG; gi++) {
      const cond = ABLATION_CONDITIONS[gi];
      const groupCx = P.l + groupW * (gi + 0.5);

      for (let si = 0; si < ABLATION_SERIES.length; si++) {
        const s = ABLATION_SERIES[si];
        const cell = ablation[s.key] && ablation[s.key][cond.key];
        if (!cell) continue;
        const rate = Math.max(0, Math.min(1, cell.rate));
        const bx = groupCx - barAreaW / 2 + slotW * si + (slotW - barW) / 2;
        const bh = Math.max(0, baselineY - yScale(rate));
        const by = yScale(rate);
        const barIndex = gi * ABLATION_SERIES.length + si; // matches bars.push order
        const isHover = hover === barIndex;

        ctx.fillStyle = s.color;
        ctx.globalAlpha = isHover ? 1 : 0.9;
        ctx.fillRect(bx, by, barW, bh);
        ctx.globalAlpha = 1;

        // Rate above the bar.
        ctx.fillStyle = COLORS.text;
        ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(rate.toFixed(2), bx + barW / 2, by - 3);

        // Param count under the bar (within the group row).
        ctx.fillStyle = COLORS.label;
        ctx.textBaseline = "top";
        ctx.fillText(`${cell.params}p`, bx + barW / 2, P.t + plotH + 22);

        bars.push({
          x: bx, y: by, w: barW, h: bh, color: s.color,
          rate, params: cell.params, condLabel: cond.label, seriesLabel: s.label,
        });
      }

      // Condition label (group title).
      ctx.fillStyle = COLORS.text;
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(cond.label, groupCx, P.t + plotH + 42);
    }

    // Hover tooltip.
    if (hover >= 0 && hover < bars.length) {
      const b = bars[hover];
      const lines = [
        `${b.seriesLabel} · ${b.condLabel}`,
        `rate: ${b.rate.toFixed(2)}`,
        `params: ${b.params}`,
      ];
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
      let tw = 0;
      for (const ln of lines) tw = Math.max(tw, ctx.measureText(ln).width);
      const bw = tw + 16;
      const bh = lines.length * 16 + 12;
      let bx = b.x + b.w + 8;
      if (bx + bw > cssW - 4) bx = Math.max(4, b.x - bw - 8);
      let by = b.y - 4;
      if (by + bh > cssH - 4) by = cssH - bh - 4;
      if (by < 4) by = 4;
      ctx.fillStyle = COLORS.tooltipBg;
      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeStyle = b.color;
      ctx.lineWidth = 1;
      ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      for (let i = 0; i < lines.length; i++) {
        ctx.fillStyle = i === 0 ? b.color : COLORS.text;
        ctx.fillText(lines[i], bx + 8, by + 7 + i * 16);
      }
    }
  }

  function onMove(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let next = -1;
    for (let i = 0; i < bars.length; i++) {
      const b = bars[i];
      if (mx >= b.x - 2 && mx <= b.x + b.w + 2 && my >= b.y - 2 && my <= b.y + b.h + 2) {
        next = i; break;
      }
    }
    canvas.style.cursor = next >= 0 ? "pointer" : "default";
    if (next !== hover) { hover = next; draw(); }
  }
  function onLeave() {
    canvas.style.cursor = "default";
    if (hover !== -1) { hover = -1; draw(); }
  }

  if (!BOUND.has(canvas)) {
    BOUND.add(canvas);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);
    if (win) win.addEventListener("resize", draw);
  }
  draw();
  return { redraw: draw };
}

// ---- Viz 4: illustrative PCA scatter with an activation selector. ----
// DOM controller. All canvas/window access lives inside this function. The
// selector is a button row drawn on the canvas itself (single-bind); points are
// colored by loss along a light-orange -> dark-purple ramp, with circle markers
// for successful runs and x markers for failures. Returns { redraw, setActivation }.
export function renderPCA(canvas, data) {
  if (!canvas || typeof canvas.getContext !== "function") return null;
  const doc = canvas.ownerDocument || (typeof document !== "undefined" ? document : null);
  if (!doc) return null;
  const win = doc.defaultView || (typeof window !== "undefined" ? window : null);

  const pca = data && typeof data === "object" ? data : {};
  let active = "polyKAN";
  let buttons = []; // populated by draw(): { x, y, w, h, key }

  function pointsFor(key) {
    return Array.isArray(pca[key]) ? pca[key] : [];
  }

  function draw() {
    const view = fitCanvas(canvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;

    // Symmetric domain with headroom so the ~1.08 polyKAN cloud fits while the
    // grid stays on clean integer/half-integer ticks (honest coordinate mapping).
    const pts = pointsFor(active);
    let maxAbs = 1;
    for (const p of pts) {
      maxAbs = Math.max(maxAbs, Math.abs(p.pc1), Math.abs(p.pc2));
    }
    const M = Math.max(1.2, maxAbs * 1.1);
    const ticks = [-1, -0.5, 0, 0.5, 1]; // all within [-M, M] since M >= 1.2

    const pad = { l: 44, r: 16, t: 44, b: 40 };
    const { plotW, plotH, pad: P, xScale, yScale } = chartFrame(ctx, cssW, cssH, {
      pad,
      xDomain: [-M, M],
      xTicks: ticks,
      yDomain: [-M, M],
      yTicks: ticks,
      xLabel: "PC1",
      yLabel: "PC2",
    });

    // Faint origin cross for PCA convention.
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xScale(0), P.t); ctx.lineTo(xScale(0), P.t + plotH);
    ctx.moveTo(P.l, yScale(0)); ctx.lineTo(P.l + plotW, yScale(0));
    ctx.stroke();

    // Scatter (clipped to plot area).
    ctx.save();
    ctx.beginPath();
    ctx.rect(P.l, P.t, plotW, plotH);
    ctx.clip();
    for (const p of pts) {
      const X = xScale(p.pc1);
      const Y = yScale(p.pc2);
      const fill = lossColor(p.loss);
      if (p.success) {
        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.arc(X, Y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,0.45)";
        ctx.lineWidth = 1;
        ctx.stroke();
      } else {
        ctx.strokeStyle = fill;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(X - 4, Y - 4); ctx.lineTo(X + 4, Y + 4);
        ctx.moveTo(X + 4, Y - 4); ctx.lineTo(X - 4, Y + 4);
        ctx.stroke();
      }
    }
    ctx.restore();

    // Loss color-scale legend (top-left of plot).
    const lgW = 90, lgH = 8;
    const lgX = P.l + 6, lgY = P.t - 22;
    for (let i = 0; i < lgW; i++) {
      ctx.fillStyle = lossColor(i / (lgW - 1));
      ctx.fillRect(lgX + i, lgY, 1, lgH);
    }
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(lgX + 0.5, lgY + 0.5, lgW - 1, lgH - 1);
    ctx.fillStyle = COLORS.label;
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText("low loss", lgX, lgY - 2);
    ctx.textAlign = "right";
    ctx.fillText("high loss", lgX + lgW, lgY - 2);

    // Activation selector (button row, top-right of plot).
    buttons = [];
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "middle";
    let bx = P.l + plotW + 4;
    for (let i = PCA_ACTIVATIONS.length - 1; i >= 0; i--) {
      const a = PCA_ACTIVATIONS[i];
      const lw = ctx.measureText(a.label).width;
      const w = lw + 18;
      const x = bx - w;
      const y = P.t - 24;
      const on = a.key === active;
      if (on) {
        ctx.fillStyle = a.color;
        ctx.fillRect(x, y, w, 18);
        ctx.fillStyle = COLORS.bg;
      } else {
        ctx.strokeStyle = a.color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, w - 1, 17);
        ctx.fillStyle = a.color;
      }
      ctx.textAlign = "center";
      ctx.fillText(a.label, x + w / 2, y + 9);
      buttons.push({ x, y, w, h: 18, key: a.key });
      bx = x - 6;
    }
  }

  function pickButton(mx, my) {
    for (const b of buttons) {
      if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) return b.key;
    }
    return null;
  }

  function onMove(e) {
    const rect = canvas.getBoundingClientRect();
    const key = pickButton(e.clientX - rect.left, e.clientY - rect.top);
    canvas.style.cursor = key ? "pointer" : "default";
  }
  function onClick(e) {
    const rect = canvas.getBoundingClientRect();
    const key = pickButton(e.clientX - rect.left, e.clientY - rect.top);
    if (key && key !== active) { active = key; draw(); }
  }

  if (!BOUND.has(canvas)) {
    BOUND.add(canvas);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("click", onClick);
    if (win) win.addEventListener("resize", draw);
  }
  draw();
  return { redraw: draw, setActivation(k) { if (pca[k]) { active = k; draw(); } } };
}
