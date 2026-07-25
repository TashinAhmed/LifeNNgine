// Widget B — the B3/S23 rule as a function of neighbor count.
//
// Split into two layers so the module stays Node-importable:
//   - RULE_POINTS is pure (no DOM) and unit-tested.
//   - createRuleFunction is the DOM controller; all DOM/canvas/window access
//     lives inside it, never at module top level.

import { fitCanvas, clearCanvas } from "../util/canvas.js";

// Combined "next state can be alive" envelope of B3/S23 over neighbor counts
// N = 0..8. target = 1 iff a cell with N live neighbors is alive next step
// (a live cell survives at N∈{2,3}; a dead cell is born at N=3). This is the
// non-monotonic bump the network must learn.
export const RULE_POINTS = Array.from({ length: 9 }, (_, n) => ({
  n,
  target: n === 2 || n === 3 ? 1 : 0,
}));

// DOM controller for Widget B. All DOM/canvas access is inside this function.
// Returns { setPaused } (setPaused is a no-op, present for interface parity),
// or a no-op stub if the mount is missing.
export function createRuleFunction(canvas) {
  const noop = { setPaused() {} };
  if (!canvas) return noop;
  const doc = canvas.ownerDocument || (typeof document !== "undefined" ? document : null);
  if (!doc) return noop;
  const win = doc.defaultView || (typeof window !== "undefined" ? window : null);

  const PAD = { l: 40, r: 16, t: 18, b: 34 };
  const COLORS = {
    bg: "#e9ecf1",
    axis: "#8a929d",
    grid: "#d4d9e0",
    label: "#3c4043",
    rule: "#16a34a",
    relu: "#d97706",
    parabola: "#0891b2",
    probe: "#3c4043",
  };

  let probeN = 3; // integer neighbor count the probe currently inspects

  function render() {
    const view = fitCanvas(canvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;

    const plotW = Math.max(1, cssW - PAD.l - PAD.r);
    const plotH = Math.max(1, cssH - PAD.t - PAD.b);
    const xOf = (n) => PAD.l + (n / 8) * plotW;   // n in [0,8]
    const yOf = (v) => PAD.t + plotH - v * plotH; // v in [0,1]

    clearCanvas(ctx, canvas.width, canvas.height, COLORS.bg);

    // --- gridlines: vertical at each integer N, horizontal at y=0,1 ---
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLORS.grid;
    for (let n = 0; n <= 8; n++) {
      ctx.beginPath();
      ctx.moveTo(xOf(n), PAD.t);
      ctx.lineTo(xOf(n), PAD.t + plotH);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(PAD.l, yOf(1)); ctx.lineTo(PAD.l + plotW, yOf(1));
    ctx.moveTo(PAD.l, yOf(0)); ctx.lineTo(PAD.l + plotW, yOf(0));
    ctx.stroke();

    // --- axes ---
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t);
    ctx.lineTo(PAD.l, PAD.t + plotH);
    ctx.lineTo(PAD.l + plotW, PAD.t + plotH);
    ctx.stroke();

    // --- ghost: parabola x² scaled+shifted (cyan) — a non-monotonic
    // polynomial that already hugs the bump, hinting why polynomial
    // activations fit naturally. 1 - ((n-2.5)/2.5)^2, clamped to [0,1]. ---
    ctx.strokeStyle = COLORS.parabola;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 80; i++) {
      const n = (i / 80) * 8;
      const v = Math.min(1, Math.max(0, 1 - Math.pow((n - 2.5) / 2.5, 2)));
      const X = xOf(n), Y = yOf(v);
      if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
    }
    ctx.stroke();

    // --- ghost: ReLU-ish ray (amber) — monotonic ramp that can't fold back.
    // max(0, (n-2)/6): turns on at n=2 (like the rule) but only climbs. ---
    ctx.strokeStyle = COLORS.relu;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 80; i++) {
      const n = (i / 80) * 8;
      const v = Math.min(1, Math.max(0, (n - 2) / 6));
      const X = xOf(n), Y = yOf(v);
      if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // --- the rule: thick step segments (green). Each integer N owns the bin
    // [N-0.5, N+0.5); verticals appear automatically where neighbors differ,
    // yielding the flat-0 → block-1 over {2,3} → flat-0 bump. ---
    ctx.strokeStyle = COLORS.rule;
    ctx.lineWidth = 4;
    ctx.lineJoin = "miter";
    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(RULE_POINTS[0].target));
    for (let n = 0; n <= 8; n++) {
      const v = RULE_POINTS[n].target;
      ctx.lineTo(xOf(Math.max(0, n - 0.5)), yOf(v));
      ctx.lineTo(xOf(Math.min(8, n + 0.5)), yOf(v));
    }
    ctx.stroke();

    // data points at each integer N
    ctx.fillStyle = COLORS.rule;
    for (const { n, target } of RULE_POINTS) {
      ctx.beginPath();
      ctx.arc(xOf(n), yOf(target), 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // --- tick labels ---
    ctx.fillStyle = COLORS.label;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let n = 0; n <= 8; n++) ctx.fillText(String(n), xOf(n), PAD.t + plotH + 6);
    ctx.fillText("live neighbors  N", PAD.l + plotW / 2, PAD.t + plotH + 20);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("1", PAD.l - 8, yOf(1));
    ctx.fillText("0", PAD.l - 8, yOf(0));

    // --- draggable probe at integer N ---
    const pn = Math.max(0, Math.min(8, probeN | 0));
    const target = RULE_POINTS[pn].target;
    const px = xOf(pn);
    ctx.strokeStyle = COLORS.probe;
    ctx.globalAlpha = 0.8;
    ctx.lineWidth = 1.25;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(px, PAD.t);
    ctx.lineTo(px, PAD.t + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    ctx.fillStyle = COLORS.probe;
    ctx.strokeStyle = COLORS.rule;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, yOf(target), 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // probe label: "if a cell has N live neighbors → …"
    const live = target === 1;
    const detail = pn === 2 ? "live cell survives"
      : pn === 3 ? "birth (or survival)"
      : "dies / stays dead";
    const text = `N=${pn} → ${live ? "ALIVE" : "dead"}  ·  ${detail}`;
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    const tw = ctx.measureText(text).width;
    const bx = Math.min(Math.max(PAD.l + 4, px + 8), PAD.l + plotW - tw - 8);
    const by = PAD.t + 4;
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.fillRect(bx - 6, by - 2, tw + 12, 20);
    ctx.strokeStyle = COLORS.probe;
    ctx.lineWidth = 1;
    ctx.strokeRect(bx - 6, by - 2, tw + 12, 20);
    ctx.fillStyle = live ? COLORS.rule : COLORS.label;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(text, bx, by + 2);
  }

  // --- pointer: drag (or click) to scrub the probe across N ---
  function nFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const plotW = Math.max(1, rect.width - PAD.l - PAD.r);
    const n = Math.round(((e.clientX - rect.left - PAD.l) / plotW) * 8);
    return Math.max(0, Math.min(8, n));
  }
  let dragging = false;
  canvas.style.touchAction = "none";
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    try { canvas.setPointerCapture(e.pointerId); } catch { /* capture optional */ }
    probeN = nFromEvent(e);
    render();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    probeN = nFromEvent(e);
    render();
  });
  const stop = () => { dragging = false; };
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);
  canvas.addEventListener("pointerleave", stop);

  if (win) win.addEventListener("resize", render);

  render();
  return { setPaused() {} };
}
