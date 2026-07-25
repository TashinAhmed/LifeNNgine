// Widget C — activation-function zoo + PolyKAN sculptor.
//
// Split into two layers so the module stays Node-importable:
//   - polyValue / isMonotonicOver are pure (no DOM) and unit-tested.
//   - createActivationPlot is the DOM controller; all DOM/canvas/window access
//     lives inside it, never at module top level.

import { fitCanvas, clearCanvas } from "../util/canvas.js";
import { RULE_POINTS } from "./rule-function.js";

// ---- Pure helpers (no DOM) ----

// Evaluate polynomial Σ coeffs[d]·xᵈ via iterative multiplication.
export function polyValue(coeffs, x) {
  let sum = 0;
  let xp = 1; // x^0
  for (let d = 0; d < coeffs.length; d++) {
    sum += coeffs[d] * xp;
    xp *= x;
  }
  return sum;
}

// True if the polynomial is non-decreasing OR non-increasing over [xMin, xMax].
// Samples `steps`+1 points and checks that successive differences never mix sign
// (a near-zero tolerance absorbs floating-point noise on flat regions).
export function isMonotonicOver(coeffs, xMin, xMax, steps = 200) {
  const n = Math.max(2, steps | 0);
  const ys = new Array(n + 1);
  let maxAbs = 0;
  for (let i = 0; i <= n; i++) {
    const x = xMin + ((xMax - xMin) * i) / n;
    const y = polyValue(coeffs, x);
    ys[i] = y;
    const a = Math.abs(y);
    if (a > maxAbs) maxAbs = a;
  }
  const eps = (maxAbs || 1) * 1e-9;
  let hasPos = false, hasNeg = false;
  for (let i = 1; i <= n; i++) {
    const d = ys[i] - ys[i - 1];
    if (d > eps) hasPos = true;
    else if (d < -eps) hasNeg = true;
    if (hasPos && hasNeg) return false;
  }
  return true;
}

// Standard activation functions (pure). PReLU uses leak slope a=0.25.
export const ACTIVATIONS = [
  { key: "relu",    label: "ReLU",    color: "#d97706", fn: (x) => Math.max(0, x) },
  { key: "prelu",   label: "PReLU",   color: "#db2777", fn: (x) => (x > 0 ? x : 0.25 * x) },
  { key: "silu",    label: "SiLU",    color: "#0891b2", fn: (x) => x / (1 + Math.exp(-x)) },
  { key: "sigmoid", label: "Sigmoid", color: "#7c3aed", fn: (x) => 1 / (1 + Math.exp(-x)) },
  { key: "tanh",    label: "Tanh",    color: "#2563eb", fn: (x) => Math.tanh(x) },
  { key: "square",  label: "Square",  color: "#ca8a04", fn: (x) => x * x },
];

// DOM controller for Widget C. All DOM/canvas access is inside this function.
// Returns { setPaused } (no-op, for interface parity), or a stub if both mounts
// are missing.
export function createActivationPlot(zooCanvas, polyCanvas, controlsEl) {
  const noop = { setPaused() {} };
  if (!zooCanvas && !polyCanvas) return noop;

  const doc = (zooCanvas && zooCanvas.ownerDocument)
    || (polyCanvas && polyCanvas.ownerDocument)
    || (controlsEl && controlsEl.ownerDocument)
    || (typeof document !== "undefined" ? document : null);
  if (!doc) return noop;
  const win = doc.defaultView || (typeof window !== "undefined" ? window : null);

  const PAD = { l: 40, r: 16, t: 18, b: 34 };
  const COLORS = {
    bg: "#e9ecf1",
    axis: "#8a929d",
    grid: "#d4d9e0",
    label: "#3c4043",
    rule: "#16a34a",
    poly: "#d97706",
  };

  // --- zoo state: x∈[-3,3], y∈[-1,2]; all activations on by default ---
  const enabled = Object.create(null);
  for (const a of ACTIVATIONS) enabled[a.key] = true;
  const ZOO_X = { min: -3, max: 3 };
  const ZOO_Y = { min: -1, max: 2 };

  // --- poly state: sculpt f(x)=w0+w1x+w2x² over neighbor count N∈[0,8] ---
  let w = [0.1, 0.1, 0.0]; // gentle ramp → monotonic by default
  const POLY_X = { min: 0, max: 8 };
  const POLY_Y = { min: -0.5, max: 1.5 };
  let readoutEl = null;

  function renderZoo() {
    if (!zooCanvas) return;
    const view = fitCanvas(zooCanvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;
    const plotW = Math.max(1, cssW - PAD.l - PAD.r);
    const plotH = Math.max(1, cssH - PAD.t - PAD.b);
    const xOf = (x) => PAD.l + ((x - ZOO_X.min) / (ZOO_X.max - ZOO_X.min)) * plotW;
    const yOf = (y) => PAD.t + plotH - ((y - ZOO_Y.min) / (ZOO_Y.max - ZOO_Y.min)) * plotH;

    clearCanvas(ctx, zooCanvas.width, zooCanvas.height, COLORS.bg);

    // gridlines at each integer x / y
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLORS.grid;
    for (let x = Math.ceil(ZOO_X.min); x <= ZOO_X.max; x++) {
      ctx.beginPath(); ctx.moveTo(xOf(x), PAD.t); ctx.lineTo(xOf(x), PAD.t + plotH); ctx.stroke();
    }
    for (let y = Math.ceil(ZOO_Y.min); y <= ZOO_Y.max; y++) {
      ctx.beginPath(); ctx.moveTo(PAD.l, yOf(y)); ctx.lineTo(PAD.l + plotW, yOf(y)); ctx.stroke();
    }

    // axes (emphasise y=0 and x=0)
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(PAD.l, yOf(0)); ctx.lineTo(PAD.l + plotW, yOf(0));
    ctx.moveTo(xOf(0), PAD.t); ctx.lineTo(xOf(0), PAD.t + plotH);
    ctx.stroke();

    // activation curves, clipped to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD.l, PAD.t, plotW, plotH);
    ctx.clip();
    ctx.lineWidth = 2;
    const N = 240;
    for (const a of ACTIVATIONS) {
      if (!enabled[a.key]) continue;
      ctx.strokeStyle = a.color;
      ctx.beginPath();
      for (let i = 0; i <= N; i++) {
        const x = ZOO_X.min + ((ZOO_X.max - ZOO_X.min) * i) / N;
        const X = xOf(x), Y = yOf(a.fn(x));
        if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
      ctx.stroke();
    }
    ctx.restore();

    // tick labels
    ctx.fillStyle = COLORS.label;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let x = Math.ceil(ZOO_X.min); x <= ZOO_X.max; x++) {
      ctx.fillText(String(x), xOf(x), PAD.t + plotH + 6);
    }
    ctx.fillText("x", PAD.l + plotW / 2, PAD.t + plotH + 20);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let y = Math.ceil(ZOO_Y.min); y <= ZOO_Y.max; y++) {
      ctx.fillText(String(y), PAD.l - 8, yOf(y));
    }
  }

  function renderPoly() {
    if (!polyCanvas) return;
    const view = fitCanvas(polyCanvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;
    const plotW = Math.max(1, cssW - PAD.l - PAD.r);
    const plotH = Math.max(1, cssH - PAD.t - PAD.b);
    const xOf = (x) => PAD.l + ((x - POLY_X.min) / (POLY_X.max - POLY_X.min)) * plotW;
    const yOf = (y) => PAD.t + plotH - ((y - POLY_Y.min) / (POLY_Y.max - POLY_Y.min)) * plotH;

    clearCanvas(ctx, polyCanvas.width, polyCanvas.height, COLORS.bg);

    // gridlines at integer N and y
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLORS.grid;
    for (let n = POLY_X.min; n <= POLY_X.max; n++) {
      ctx.beginPath(); ctx.moveTo(xOf(n), PAD.t); ctx.lineTo(xOf(n), PAD.t + plotH); ctx.stroke();
    }
    for (let y = Math.ceil(POLY_Y.min); y <= POLY_Y.max; y++) {
      ctx.beginPath(); ctx.moveTo(PAD.l, yOf(y)); ctx.lineTo(PAD.l + plotW, yOf(y)); ctx.stroke();
    }

    // axes
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(PAD.l, yOf(0)); ctx.lineTo(PAD.l + plotW, yOf(0));
    ctx.lineTo(PAD.l + plotW, PAD.t + plotH);
    ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, PAD.t + plotH);
    ctx.stroke();

    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD.l, PAD.t, plotW, plotH);
    ctx.clip();

    // ghost: B3/S23 rule bump (green). Each integer N owns bin [N-0.5, N+0.5)
    // so verticals form the bump automatically; target derived from the shared
    // RULE_POINTS so this widget can't drift from Widget B.
    const target = (n) => RULE_POINTS[n]?.target ?? 0;
    ctx.strokeStyle = COLORS.rule;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 3;
    ctx.lineJoin = "miter";
    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(target(0)));
    for (let n = 0; n <= 8; n++) {
      const v = target(n);
      ctx.lineTo(xOf(Math.max(0, n - 0.5)), yOf(v));
      ctx.lineTo(xOf(Math.min(8, n + 0.5)), yOf(v));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // the sculpted polynomial f(x)=w0+w1x+w2x² (amber)
    ctx.strokeStyle = COLORS.poly;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    const N = 240;
    for (let i = 0; i <= N; i++) {
      const x = POLY_X.min + ((POLY_X.max - POLY_X.min) * i) / N;
      const X = xOf(x), Y = yOf(polyValue(w, x));
      if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
    }
    ctx.stroke();
    ctx.restore();

    // monotonic readout (on canvas) using isMonotonicOver
    const mono = isMonotonicOver(w, POLY_X.min, POLY_X.max);
    const text = `monotonic: ${mono ? "yes" : "no"}`;
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    const tw = ctx.measureText(text).width;
    const bx = PAD.l + 8, by = PAD.t + 6;
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.fillRect(bx - 6, by - 2, tw + 12, 20);
    ctx.strokeStyle = mono ? COLORS.rule : COLORS.poly;
    ctx.lineWidth = 1;
    ctx.strokeRect(bx - 6, by - 2, tw + 12, 20);
    ctx.fillStyle = mono ? COLORS.rule : COLORS.poly;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(text, bx, by + 2);

    // tick labels
    ctx.fillStyle = COLORS.label;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let n = POLY_X.min; n <= POLY_X.max; n++) {
      ctx.fillText(String(n), xOf(n), PAD.t + plotH + 6);
    }
    ctx.fillText("live neighbors  N", PAD.l + plotW / 2, PAD.t + plotH + 20);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let y = Math.ceil(POLY_Y.min); y <= POLY_Y.max; y++) {
      ctx.fillText(String(y), PAD.l - 8, yOf(y));
    }

    // mirror readout to the DOM control
    if (readoutEl) {
      readoutEl.textContent = `monotonic: ${mono ? "yes" : "no"}`;
      readoutEl.style.color = mono ? COLORS.rule : COLORS.poly;
    }
  }

  // ---------------- controls ----------------
  if (controlsEl) {
    // Group 1: activation toggles (zoo legend)
    const zooGroup = doc.createElement("fieldset");
    zooGroup.style.cssText = "border:1px solid #cdd2da;padding:8px 12px;margin:0 0 10px 0;border-radius:3px;";
    const zooLegend = doc.createElement("legend");
    zooLegend.textContent = "Activations";
    zooLegend.style.color = COLORS.label;
    zooGroup.appendChild(zooLegend);
    for (const a of ACTIVATIONS) {
      const label = doc.createElement("label");
      label.style.cssText = "display:inline-flex;align-items:center;gap:4px;margin-right:12px;color:#3c4043;cursor:pointer;";
      const cb = doc.createElement("input");
      cb.type = "checkbox";
      cb.checked = true;
      cb.addEventListener("change", () => {
        enabled[a.key] = cb.checked;
        renderZoo();
      });
      const swatch = doc.createElement("span");
      swatch.style.cssText = `display:inline-block;width:12px;height:12px;background:${a.color};border:1px solid #0006;border-radius:2px;`;
      const text = doc.createElement("span");
      text.textContent = a.label;
      label.append(cb, swatch, text);
      zooGroup.appendChild(label);
    }

    // Group 2: poly sliders
    const polyGroup = doc.createElement("fieldset");
    polyGroup.style.cssText = "border:1px solid #cdd2da;padding:8px 12px;margin:0 0 10px 0;border-radius:3px;";
    const polyLegend = doc.createElement("legend");
    polyLegend.textContent = "Sculpt f(x) = w\u2080 + w\u2081x + w\u2082x\u00b2";
    polyLegend.style.color = COLORS.label;
    polyGroup.appendChild(polyLegend);

    const ranges = [
      { i: 0, name: "w\u2080", min: -1, max: 1, step: 0.05 },
      { i: 1, name: "w\u2081", min: -1, max: 1, step: 0.05 },
      { i: 2, name: "w\u2082", min: -0.3, max: 0.3, step: 0.02 },
    ];
    const fmt = (v) => (v >= 0 ? " " : "") + v.toFixed(2);
    for (const r of ranges) {
      const row = doc.createElement("div");
      row.style.cssText = "display:flex;align-items:center;gap:8px;margin:4px 0;";
      const label = doc.createElement("label");
      label.htmlFor = `poly-w-${r.i}`;
      label.textContent = r.name;
      label.style.cssText = "width:24px;color:#3c4043;";
      const input = doc.createElement("input");
      input.type = "range";
      input.id = `poly-w-${r.i}`;
      input.min = String(r.min);
      input.max = String(r.max);
      input.step = String(r.step);
      input.value = String(w[r.i]);
      input.style.flex = "1";
      const val = doc.createElement("span");
      val.style.cssText = "width:52px;text-align:right;color:#5b6168;font-family:ui-monospace,monospace;font-size:11px;";
      val.textContent = fmt(w[r.i]);
      input.addEventListener("input", () => {
        w[r.i] = Number(input.value) || 0;
        val.textContent = fmt(w[r.i]);
        renderPoly();
      });
      row.append(label, input, val);
      polyGroup.appendChild(row);
    }

    readoutEl = doc.createElement("p");
    readoutEl.style.cssText = "margin:8px 0 0 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#16a34a;";

    controlsEl.append(zooGroup, polyGroup, readoutEl);
  }

  if (win) win.addEventListener("resize", () => { renderZoo(); renderPoly(); });

  renderZoo();
  renderPoly();
  return { setPaused() {} };
}
