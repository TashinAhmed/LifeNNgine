# LifeNNgine Interactive Page — Foundation Plan (Plan 2a of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full scrollytelling webpage (narrative + dark theme + interactive supporting widgets + results charts) for the LifeNNgine paper, leaving a clearly-marked slot for the live training arena (Plan 2b). The page consumes the verified `web/js/engine/` built in Plan 1.

**Architecture:** Vanilla HTML/CSS/JS, no dependencies, no build step. Each interactive module is a small controller that owns one `<canvas>` (or inline SVG), reads the engine barrel, and exposes a pure-logic surface that is unit-tested in Node. All DOM/canvas access lives inside `init()`/render functions (never at module top level) so every module stays Node-importable for smoke tests. Numeric results are hardcoded in `web/js/data/results.js` (transcribed from the paper) and tested against the paper's tables.

**Tech Stack:** Vanilla JS (ES modules), Node ≥ 18 `node:test` for logic tests, `<canvas>` + inline SVG for rendering. No npm dependencies, no KaTeX (math via Unicode/MathML), no chart library.

## Global Constraints

- All code under `web/`. Continue on branch `feat/web-demo` (engine already there).
- **No external dependencies.** No npm install. Node built-ins only.
- ES modules (`web/package.json` already has `"type":"module"`). `import`/`export` only.
- **No DOM/canvas access at module top level.** All `document`/`window`/`canvas`/`requestAnimationFrame` use must be inside functions. This keeps every module Node-importable for smoke tests. (This is stricter than Plan 1 because these modules are browser-facing but must still pass a Node smoke-import test.)
- The page must load and run by opening `web/index.html` directly (`file://`) — use ES module `<script type="module">` and relative paths. No fetch of local files (module imports only).
- Visual identity (spec §10): near-black bg `#0b0d10`, off-white text `#e8eaed`, subtle gray grid lines `#1c2025`; accents — live/success green `#39ff14`, contrast amber `#ffb454`, loss cyan `#22d3ee` / magenta `#e879f9`. Color-blind-aware (never rely on red/green alone).
- Typography: system sans stack; heavier weight for chapter headers. Math via Unicode (Σ, ᵢ, ²) or inline MathML — no KaTeX.
- Motif: the Life grid recurs (section number tiles, dividers, hero background).
- Honors `prefers-reduced-motion` (slow/stop auto-running grids). Pauses off-screen animation via IntersectionObserver (added in assembly task).
- Responsive: canvas widgets scale to container; on narrow screens stack vertically.
- Faithful data: every chart cites its source (Table/Figure number). The PCA chart is an **illustrative reproduction** (exact points cannot be extracted from Fig 1) and is labeled as such in its caption.

## Verification approach (read carefully)

- **Pure logic** (data values, monotonicity, presets, scaling math) is unit-tested in Node (`node --test`).
- **Smoke-import test**: a Node test imports each browser module to confirm no syntax errors and no top-level DOM access (importing must not throw). Run with `node --test`.
- **Rendering** cannot be verified in Node (no browser). It is verified by (a) code review and (b) the user opening `web/index.html`. Each visual task ends with a "manual verification checklist" the implementer ticks by self-review (the controller/user confirms visually later).
- Run all Node tests with `npm test` from `web/`.

---

## File Structure (additions under `web/`)

```
web/
├── index.html                      # full scrollytelling page (7 chapters)
├── css/
│   └── style.css                   # dark theme, typography, layout, responsive
├── js/
│   ├── util/
│   │   └── canvas.js               # DPR-aware resize, drawGrid, colors, ease
│   ├── data/
│   │   └── results.js              # hardcoded paper data (Table 2, Fig 4/6, illustrative PCA)
│   ├── widgets/
│   │   ├── life-grid.js            # Widget A: interactive Life grid + hero bg mode
│   │   ├── rule-function.js        # Widget B: rule interval plot + probe
│   │   └── activation-plot.js      # Widget C: activation zoo + PolyKAN sliders
│   ├── charts/
│   │   └── charts.js               # Viz 1–4 renderers (canvas/SVG) consuming results.js
│   └── main.js                     # bootstrap: init all widgets/charts, IO pause, reduced-motion
└── test/
    ├── smoke.test.js               # import every browser module (no top-level DOM)
    ├── results.test.js             # data integrity vs paper
    ├── life-grid.test.js           # preset/pattern + step logic
    └── activation-plot.test.js     # monotonicity readout logic
```

---

### Task 1: Page scaffold + dark theme + canvas utilities + hero

**Files:**
- Create: `web/index.html`, `web/css/style.css`, `web/js/util/canvas.js`, `web/js/main.js` (minimal bootstrap), `web/test/smoke.test.js` (initial: import canvas.js)

**Interfaces:**
- `canvas.js` produces: `fitCanvas(canvas) => {ctx, w, h, cssW, cssH}` (DPR-aware sizing), `drawGrid(ctx, grid, H, W, cellPx, {on, off, flash})`, `clearCanvas(ctx, w, h, bg)`.
- `main.js` exports `initHero(canvas, opts)` for the self-running dimmed Life background (used now); full bootstrap grows in the assembly task.
- `index.html` defines all 7 chapter `<section>`s with the canvases/containers each later task will populate, plus a placeholder `<div id="arena-mount">` for Plan 2b.

- [ ] **Step 1: Write `web/js/util/canvas.js`**

```js
// DPR-aware canvas helpers. DOM access only inside functions (Node-importable).

export function fitCanvas(canvas) {
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  const rect = canvas.getBoundingClientRect();
  const cssW = Math.max(1, Math.floor(rect.width));
  const cssH = Math.max(1, Math.floor(rect.height));
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: canvas.width, h: canvas.height, cssW, cssH, dpr };
}

export function clearCanvas(ctx, w, h, bg = "#0b0d10") {
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
}

// Draw a flat HxW binary grid. flash: optional Uint8Array same length; cells that
// just changed are drawn with the flash color border.
export function drawGrid(ctx, grid, H, W, cellPx, { on = "#39ff14", off = "#11151a", gridline = "#1c2025", flash = null, flashColor = "#ffb454" } = {}) {
  for (let h = 0; h < H; h++) {
    for (let w = 0; w < W; w++) {
      const i = h * W + w;
      ctx.fillStyle = grid[i] ? on : off;
      ctx.fillRect(w * cellPx, h * cellPx, cellPx, cellPx);
      if (flash && flash[i]) {
        ctx.strokeStyle = flashColor;
        ctx.lineWidth = Math.max(1, cellPx * 0.12);
        ctx.strokeRect(w * cellPx + 0.5, h * cellPx + 0.5, cellPx - 1, cellPx - 1);
      }
    }
  }
  ctx.strokeStyle = gridline;
  ctx.lineWidth = 1;
  for (let h = 0; h <= H; h++) {
    ctx.beginPath(); ctx.moveTo(0, h * cellPx); ctx.lineTo(W * cellPx, h * cellPx); ctx.stroke();
  }
  for (let w = 0; w <= W; w++) {
    ctx.beginPath(); ctx.moveTo(w * cellPx, 0); ctx.lineTo(w * cellPx, H * cellPx); ctx.stroke();
  }
}

// cubic ease in/out for smooth animation transitions
export function easeInOut(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
```

- [ ] **Step 2: Write `web/css/style.css`**

A complete dark theme. Provide: CSS reset basics, `:root` variables for the palette above, body bg/text/font, a `.page` max-width container, `.chapter` section spacing with a `.chapter__title` display weight, `.chapter__lead` lede styling, a `.grid-motif` divider, a `.canvas-card` framed container for canvases (bg `#0d1014`, 1px border `#1c2025`, radius 8px, padding), `.controls` row layout, `button`/`input[type=range]` styling in the accent palette, `.caption` small muted text, `.data-source` tiny caption, responsive `@media (max-width: 720px)` that stacks grids. Keep it ~150–220 lines, cohesive, no external fonts.

- [ ] **Step 3: Write `web/index.html`**

Full document. `<head>` with `<meta charset>`, viewport, `<title>LifeNNgine — …</title>`, `<link rel="stylesheet" href="css/style.css">`. Body:
- **Chapter 0 (hero)**: `<section class="chapter hero">` with title, subtitle, authors (Tashin Ahmed, Q. Tyrell Davis), ALIFE 2026, links (paper https://arxiv.org/abs/2606.23587, code https://github.com/TashinAhmed/LifeNNgine), and `<canvas id="hero-bg"></canvas>` behind.
- **Chapter 1**: heading "The Pocket Universe", lede prose, `<canvas id="life-grid">` + `<div class="controls" id="life-controls">`.
- **Chapter 2**: heading "The rule, as a function", `<canvas id="rule-function">`.
- **Chapter 3**: heading "Why it's hard (ReLU & lottery tickets)", narrative prose only (figure slot optional).
- **Chapter 4**: heading "Watch it learn, live", `<div id="arena-mount"><p class="placeholder">The live training arena arrives in the next update.</p></div>`.
- **Chapter 5**: heading "The activation-function zoo", `<canvas id="activation-zoo">` + PolyKAN sliders `<div id="poly-controls">`, then `<canvas id="viz-success">`.
- **Chapter 6**: heading "Robustness & results", `<canvas id="viz-density">`, `<canvas id="viz-ablation">`, `<canvas id="viz-pca">`.
- **Footer**: citation block (plain + a `<pre>` BibTeX) and CC BY 4.0 note.
- At end of body: `<script type="module" src="js/main.js"></script>`.

- [ ] **Step 4: Write minimal `web/js/main.js`**

For now, only the hero background. (Assembly task expands it.) It must guard DOM access inside functions.

```js
import { initHero } from "./widgets/life-grid.js"; // hero lives in life-grid.js (Task 3); for now create a local stub
```

To avoid a forward dependency, create the hero in this task as a small local function in `main.js`:

```js
import { lifeStep } from "./engine/life.js";
import { mulberry32 } from "./engine/rng.js";
import { fitCanvas, clearCanvas, drawGrid } from "./util/canvas.js";

const prefersReducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

export function initHero(canvas) {
  const H = 48, W = 96;
  let grid = new Uint8Array(H * W);
  const rng = mulberry32(2026);
  for (let i = 0; i < grid.length; i++) grid[i] = rng() < 0.35 ? 1 : 0;
  let raf = 0, last = 0;
  function frame(t) {
    if (t - last > 220) { grid = Uint8Array.from(lifeStep(grid, H, W)); last = t; }
    const { ctx, cssW, cssH } = fitCanvas(canvas);
    clearCanvas(ctx, canvas.width, canvas.height, "rgba(11,13,16,0)");
    drawGrid(ctx, grid, H, W, Math.min(cssW / W, cssH / H), { on: "rgba(57,255,20,0.10)", off: "transparent", gridline: "transparent" });
    raf = requestAnimationFrame(frame);
  }
  if (!prefersReducedMotion()) raf = requestAnimationFrame(frame);
  return () => cancelAnimationFrame(raf);
}

document.addEventListener("DOMContentLoaded", () => {
  const hero = document.getElementById("hero-bg");
  if (hero) window.__stopHero = initHero(hero);
});
```

- [ ] **Step 5: Write `web/test/smoke.test.js` (initial)**

```js
import { test } from "node:test";
import assert from "node:assert/strict";

test("canvas util imports without DOM access", async () => {
  const m = await import("../js/util/canvas.js");
  assert.equal(typeof m.fitCanvas, "function");
  assert.equal(typeof m.drawGrid, "function");
  assert.equal(typeof m.easeInOut, "function");
});
```

- [ ] **Step 6: Run tests**

Run: `cd web && node --test test/smoke.test.js`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add web/index.html web/css/style.css web/js/util/canvas.js web/js/main.js web/test/smoke.test.js
git commit -m "feat(web): page scaffold, dark theme, canvas utils, hero background"
```

**Manual verification checklist (self-review):** `index.html` references `css/style.css` and `js/main.js` with correct relative paths; all 7 chapters present with the element IDs listed; CSS variables match the palette; `main.js` has no top-level DOM access except the `DOMContentLoaded` listener.

---

### Task 2: Results data module + integrity tests

**Files:**
- Create: `web/js/data/results.js`, `web/test/results.test.js`
- Modify: `web/test/smoke.test.js` (add import of results.js)

**Interfaces:**
- `results.js` exports:
  - `SUCCESS_RATES`: array of `{ name, rate, params, monotonic, differentiable }` from Table 2.
  - `ABLATION`: `{ polyKAN: {full, actOnly, weightOnly}, prelu: {full, actOnly, weightOnly} }` each `{rate, params}` from Table 2 / Fig 6.
  - `DENSITY_SWEEP`: array of `{ density, polyKAN, prelu, silu, relu }` across 0.05–0.95 (approximated from Fig 4 documented anchors; field `approx: true`).
  - `PCA_ILLUSTRATIVE`: `{ polyKAN: [{pc1,pc2,loss,success}], relu: [...], prelu: [...], sigmoid: [...] }` (synthetic, `approx: true`).

- [ ] **Step 1: Write the data module**

Transcribe Table 2 exactly (the real numbers). For density sweep, build a curve from the paper's documented anchor points (PolyKAN ≈1.0 except dips to 0.75/0.9375/0.8125/0.9375 at d=0.90/0.35/0.30/0.20 and 0 at 0.95; PReLU min 0.0625 at 0.90; SiLU peaks ~1.0 near 0.50; ReLU low throughout, peak ~0.31 for m=2 at 0.40/0.60) and linearly interpolate at 0.05 steps; mark `approx: true`. For PCA, generate illustrative trajectories matching the described qualitative shapes (PolyKAN smooth fan-out to a diverse frontier; ReLU split by a ridge with many × failures; PReLU two bifurcated paths; Sigmoid a cross). Provide ~24 points per activation with `loss` (0..1, 1=worst) and `success` (bool).

- [ ] **Step 2: Write `web/test/results.test.js`**

Assert exact Table 2 values:
```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { SUCCESS_RATES, ABLATION, DENSITY_SWEEP, PCA_ILLUSTRATIVE } from "../js/data/results.js";

test("success rates match Table 2", () => {
  const byName = Object.fromEntries(SUCCESS_RATES.map((r) => [r.name, r]));
  assert.equal(byName.PolyKAN.rate, 1.0);
  assert.equal(byName.PolyKAN.params, 34);
  assert.equal(byName.Square.rate, 0.94);
  assert.equal(byName.SiLU.rate, 0.94);
  assert.equal(byName.RootSquare.rate, 0.50);
  assert.equal(byName.LeakyReLU.rate, 0.25);
  assert.equal(byName.CELU.rate, 0.06);
  assert.equal(byName.Sigmoid.rate, 0.0);
  assert.equal(byName.Tanh.rate, 0.0);
  assert.equal(byName.ReLU.rate, 0.0);
  assert.equal(byName.ReLU.params, 25);
});

test("ablation matches Table 2 / Fig 6", () => {
  assert.equal(ABLATION.polyKAN.full.rate, 1.0);
  assert.equal(ABLATION.polyKAN.actOnly.rate, 1.0);
  assert.equal(ABLATION.polyKAN.weightOnly.rate, 0.78);
  assert.equal(ABLATION.polyKAN.actOnly.params, 29);
  assert.equal(ABLATION.prelu.weightOnly.rate, 0.59);
});

test("density sweep is marked approximate and spans the range", () => {
  assert.equal(DENSITY_SWEEP.approx, true);
  const ds = DENSITY_SWEEP.points;
  assert.ok(ds[0].density <= 0.06);
  assert.ok(ds[ds.length - 1].density >= 0.94);
  assert.ok(ds.every((d) => ["polyKAN", "prelu", "silu", "relu"].every((k) => typeof d[k] === "number")));
});

test("PCA is illustrative and has the four activations", () => {
  assert.equal(PCA_ILLUSTRATIVE.approx, true);
  for (const k of ["polyKAN", "relu", "prelu", "sigmoid"]) {
    assert.ok(PCA_ILLUSTRATIVE[k].length >= 12, `${k} needs >=12 points`);
    assert.ok(PCA_ILLUSTRATIVE[k].every((p) => "pc1" in p && "pc2" in p && "loss" in p && "success" in p));
  }
});
```

(Adjust the test's accessor shapes — e.g. `DENSITY_SWEEP.points` — to match the exact structure you export.)

- [ ] **Step 3: Add smoke import** in `web/test/smoke.test.js` (a second test importing `../js/data/results.js` and asserting the four exports exist).

- [ ] **Step 4: Run tests**

Run: `cd web && npm test`
Expected: all PASS (data + smoke).

- [ ] **Step 5: Commit**

```bash
git add web/js/data/results.js web/test/results.test.js web/test/smoke.test.js
git commit -m "feat(web): add paper results data module + integrity tests"
```

---

### Task 3: Widget A — Interactive Life grid

**Files:**
- Create: `web/js/widgets/life-grid.js`, `web/test/life-grid.test.js`
- Modify: `web/js/main.js` (wire `#life-grid`)

**Interfaces:**
- `life-grid.js` exports:
  - `PRESETS`: `{ glider: [[0,1],[1,2],[2,0],[2,1],[2,2]], blinker: [[0,0],[0,1],[0,2]], block: [[0,0],[0,1],[1,0],[1,1]], pulsar: [...] }` (as [r,c] offsets).
  - `stampPreset(grid, H, W, preset, originR, originC) => Uint8Array` — pure: stamp a preset at origin (toroidal wrap). Tested in Node.
  - `createLifeGrid(canvas, controlsEl, { H=32, W=32 })` — controller: manages state, paint-on-drag, Step/Play/Randomize(+density slider)/Clear, preset buttons; renders via `drawGrid` with birth/death flash. Reuses `lifeStep` from engine.

- [ ] **Step 1: Write the failing logic test**

`web/test/life-grid.test.js`:
```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { PRESETS, stampPreset } from "../js/widgets/life-grid.js";
import { lifeStep } from "../js/engine/life.js";

function blank(H, W) { return new Uint8Array(H * W); }

test("glider stamped in empty torus moves after stepping", () => {
  const H = 10, W = 10;
  let g = stampPreset(blank(H, W), H, W, PRESETS.glider, 4, 4);
  const beforeAlive = g.reduce((s, v) => s + v, 0);
  const next = lifeStep(g, H, W);
  const afterAlive = next.reduce((s, v) => s + v, 0);
  assert.equal(beforeAlive, 5); // glider has 5 cells
  assert.equal(afterAlive, 5);  // glider preserves cell count after a step
});

test("stampPreset wraps toroidally near the edge", () => {
  const H = 5, W = 5;
  const g = stampPreset(blank(H, W), H, W, PRESETS.block, 4, 4); // bottom-right, wraps
  assert.equal(g[4 * 5 + 4] + g[0 * 5 + 0], 2); // (4,4) and wrapped (0,0) both set
});
```

- [ ] **Step 2: Run to confirm it fails** (`cd web && node --test test/life-grid.test.js`).

- [ ] **Step 3: Implement `life-grid.js`**

Export `PRESETS`, `stampPreset` (pure). Then `createLifeGrid(canvas, controlsEl, opts)`:
- state: `grid` (Uint8Array), `running`, `density=0.4`, `flash` (Uint8Array tracking changed cells for one frame).
- build controls DOM (Step, Play/Pause, Randomize, density range 0.05–0.95, Clear, and preset buttons) into `controlsEl`; style via existing CSS classes.
- pointer paint: on pointerdown/move with button pressed, toggle the cell under the cursor (paint mode = set to 1; support erase with shift).
- `render()` computes cellPx from canvas size, calls `fitCanvas`/`drawGrid` with flash; clears flash after draw.
- `step()` computes `prev`, `next = lifeStep(grid)`, sets flash where `prev!==next`, assigns grid.
- Play loop via `requestAnimationFrame` throttled to ~6 fps; respects a `stopped` flag (set by IntersectionObserver later — expose `setPaused(bool)`).
- Return `{ setPaused, reset, step }`.

- [ ] **Step 4: Run logic test → PASS.** Then run `npm test` → all pass.

- [ ] **Step 5: Wire in `main.js`**: in the DOMContentLoaded handler, if `#life-grid` exists, call `createLifeGrid(document.getElementById("life-grid"), document.getElementById("life-controls"))`.

- [ ] **Step 6: Commit**

```bash
git add web/js/widgets/life-grid.js web/test/life-grid.test.js web/js/main.js
git commit -m "feat(web): interactive Game of Life grid (Widget A)"
```

**Manual verification checklist:** presets stamp and evolve; paint toggles cells; Randomize respects density; Play animates; no top-level DOM access (smoke import added).

---

### Task 4: Widget B — Rule as a function

**Files:**
- Create: `web/js/widgets/rule-function.js`
- Modify: `web/test/smoke.test.js`, `web/js/main.js`

**Interfaces:**
- `rule-function.js` exports `RULE_POINTS` (pure): array over N=0..8 of `{ n, target }` where target = 1 if N∈{2,3} else 0 (the B3/S23 survival+birth combined shape the network must fit — note target=1 for N=2 or 3). Also `createRuleFunction(canvas)`.

- [ ] **Step 1: Write the module**

`RULE_POINTS`: for N=0..8, target = (N===2||N===3)?1:0. (This is the combined "next state could be alive" envelope; caption clarifies survival vs birth.)
`createRuleFunction(canvas)`:
- draw axes: x = neighbor count 0..8, y = 0..1.
- draw the rule shape as thick step segments (flat 0 → 1 at {2,3} → flat 0), in green.
- draw faint ghost overlays: a ReLU-ish ray (amber) and a parabola x² scaled (cyan) to motivate why monotonic/linear fits poorly and a non-monotonic polynomial fits naturally.
- a draggable vertical probe line at integer N showing the target value and a label "if a cell has N live neighbors → …".
- expose `setPaused()` no-op (consistent interface).

- [ ] **Step 2: Add smoke import** for `rule-function.js` and assert `RULE_POINTS` length 9 and target values (1 at 2,3; 0 elsewhere).

- [ ] **Step 3: Wire `main.js`** to call `createRuleFunction(document.getElementById("rule-function"))`.

- [ ] **Step 4: `npm test` → all pass.**

- [ ] **Step 5: Commit**

```bash
git add web/js/widgets/rule-function.js web/test/smoke.test.js web/js/main.js
git commit -m "feat(web): rule-as-a-function plot (Widget B)"
```

---

### Task 5: Widget C — Activation explorer

**Files:**
- Create: `web/js/widgets/activation-plot.js`, `web/test/activation-plot.test.js`
- Modify: `web/js/main.js`

**Interfaces:**
- `activation-plot.js` exports:
  - `polyValue(coeffs, x)` — pure: Σ coeffs[d]·xᵈ. Tested.
  - `isMonotonicOver(coeffs, xMin, xMax, steps=200)` — pure: returns true if the polynomial is non-decreasing (or non-increasing) over the sampled range. Tested.
  - `createActivationPlot(zooCanvas, polyCanvas, controlsEl)`.

- [ ] **Step 1: Write failing logic test**

`web/test/activation-plot.test.js`:
```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { polyValue, isMonotonicOver } from "../js/widgets/activation-plot.js";

test("polyValue evaluates a polynomial", () => {
  assert.equal(polyValue([1, 2, 3], 2), 1 + 2 * 2 + 3 * 4); // 17
});

test("w1=1 linear is monotonic; a downward parabola is not", () => {
  assert.equal(isMonotonicOver([0, 1, 0], -2, 2), true);   // f(x)=x
  assert.equal(isMonotonicOver([0, 0, 1], -2, 2), false);  // f(x)=x^2 (decreases then increases)
});
```

- [ ] **Step 2: Run to confirm fail.**

- [ ] **Step 3: Implement.** `polyValue`, `isMonotonicOver` (sample, check sign of successive differences doesn't flip). `createActivationPlot`:
- `zooCanvas`: axes x∈[-3,3], y∈[-1,2]; toggleable series for ReLU, PReLU(a=0.25), SiLU, Sigmoid, Tanh, Square (each a pure function; draw from a shared list with colors). A legend with checkboxes (checkboxes created in `controlsEl`).
- `polyCanvas`: plot f(x)=w0+w1x+w2x² with three range sliders (w0,w1,w2) in `controlsEl`, live re-plot on input. Ghost the rule shape (green) behind. A readout "monotonic: yes/no" using `isMonotonicOver`.

- [ ] **Step 4: Logic tests pass; `npm test` all pass.**

- [ ] **Step 5: Wire `main.js`** to `createActivationPlot(#activation-zoo, #activation-poly, #poly-controls)` — add a second canvas `#activation-poly` to Chapter 5 in `index.html` if not present.

- [ ] **Step 6: Commit**

```bash
git add web/js/widgets/activation-plot.js web/test/activation-plot.test.js web/js/main.js web/index.html
git commit -m "feat(web): activation explorer (Widget C)"
```

---

### Task 6: Charts — success-rate bars + density sweep

**Files:**
- Create: `web/js/charts/charts.js`
- Modify: `web/test/smoke.test.js`, `web/js/main.js`

**Interfaces:**
- `charts.js` exports `renderSuccessBars(canvas, data)`, `renderDensitySweep(canvas, data)`, plus a shared `chartFrame(ctx, w, h, opts)` helper (axes, padding, ticks). Pure scaling helpers `linearScale(domain, range)` exported and smoke-tested.

- [ ] **Step 1: Implement.**
- `renderSuccessBars`: horizontal bars from `SUCCESS_RATES` sorted descending; bar length ∝ rate; color green for PolyKAN, amber for ReLU, gray others; label name + rate; hover tooltip showing params/monotonic/differentiable (title attribute on a transparent overlay rect or a drawn tooltip on mousemove). Caption cites "Table 2".
- `renderDensitySweep`: multi-line chart from `DENSITY_SWEEP.points`; x=density 0.05–0.95, y=success rate 0–1; four series (polyKAN green, prelu cyan, silu magenta, relu amber); toggleable legend. Caption cites "Fig 4 (approx.)".

- [ ] **Step 2: Smoke import** of `charts.js` + assert `linearScale` maps domain endpoints to range endpoints.

- [ ] **Step 3: Wire `main.js`** to render `#viz-success` and `#viz-density` on DOMContentLoaded.

- [ ] **Step 4: `npm test` all pass.**

- [ ] **Step 5: Commit**

```bash
git add web/js/charts/charts.js web/test/smoke.test.js web/js/main.js
git commit -m "feat(web): success-rate bars (Viz 1) + density sweep (Viz 2)"
```

---

### Task 7: Charts — ablation + PCA

**Files:**
- Modify: `web/js/charts/charts.js` (add `renderAblation`, `renderPCA`)
- Modify: `web/test/smoke.test.js`, `web/js/main.js`

- [ ] **Step 1: Implement.**
- `renderAblation(canvas, data)`: grouped bars for PolyKAN and PReLU across full/actOnly/weightOnly; annotate param counts (34/29/25, 28/23/25); headline callout "PolyKAN: 128/128 with or without weight training". Caption cites "Fig 6 / Table 2".
- `renderPCA(canvas, data)`: scatter of `PCA_ILLUSTRATIVE`; points colored by loss (light→dark), marker circle=success / ×=failure; a selector (buttons in a small controls row) to switch activation (polyKAN/relu/prelu/sigmoid); pan/zoom optional (keep simple: just the selector + redraw). Caption MUST read "Illustrative reproduction — exact points not extractable from Fig 1."

- [ ] **Step 2: Smoke import** already covers charts.js; add an assertion that `renderAblation` and `renderPCA` are functions.

- [ ] **Step 3: Wire `main.js`** to render `#viz-ablation` and `#viz-pca`.

- [ ] **Step 4: `npm test` all pass.**

- [ ] **Step 5: Commit**

```bash
git add web/js/charts/charts.js web/test/smoke.test.js web/js/main.js
git commit -m "feat(web): ablation (Viz 3) + illustrative PCA (Viz 4)"
```

---

### Task 8: Assembly — wiring, IntersectionObserver, reduced-motion, polish

**Files:**
- Modify: `web/js/main.js`, `web/css/style.css`, `web/index.html`

- [ ] **Step 1: IntersectionObserver pause-when-offscreen.** In `main.js`, observe each `.canvas-card` (or the canvases); when off-screen, call the widget's `setPaused(true)`; resume on enter. This keeps animation/training off-screen cheap. Each widget returned a handle with `setPaused`; collect them.

- [ ] **Step 2: `prefers-reduced-motion`.** Hero already checks it; ensure all auto-running widgets (life grid play, hero) respect a single `REDUCED_MOTION` flag exported from a tiny util or read in `main.js` and passed to controllers.

- [ ] **Step 3: Responsive + polish.** Ensure canvases have CSS width 100% and a max-height; on `@media (max-width:720px)` stack Chapter 5's two canvases and reduce hero grid density if needed. Add a small sticky chapter nav (anchor links) and a "back to top".

- [ ] **Step 4: Final smoke + full suite.** `cd web && npm test` — all Node tests pass. Then open `web/index.html` in a browser and walk the manual verification checklist (all chapters render, widgets respond, charts legible, no console errors). Note any console errors as findings.

- [ ] **Step 5: Commit**

```bash
git add web/js/main.js web/css/style.css web/index.html
git commit -m "feat(web): assembly — IO pause, reduced-motion, responsive, nav"
```

---

## Self-Review (completed during authoring)

**Spec coverage (Plan 2a = spec §5 structure, §7 widgets A–C, §8 charts Viz 1–4, §10 visual system):**
- §5 page structure (7 chapters + footer): Task 1 (scaffold), fleshed across tasks. ✓
- §7 Widget A (life grid): Task 3. Widget B (rule function): Task 4. Widget C (activation explorer): Task 5. ✓
- §8 Viz 1 success bars: Task 6. Viz 2 density: Task 6. Viz 3 ablation: Task 7. Viz 4 PCA: Task 7 (illustrative, flagged). ✓
- §10 visual system (dark theme, palette, motif, responsive, reduced-motion): Task 1 + Task 8. ✓
- §6 arena deliberately deferred to Plan 2b (placeholder mount in Task 1). ✓
- Correctness/data integrity gate: Task 2 tests. ✓

**Placeholder scan:** Each task has complete code or a precise implementable spec + key code. No "TBD". (Rendering specs are intentionally spec-driven because canvas code cannot be unit-tested; this is stated in Verification Approach.)

**Plan 2b (next)** will: implement `web/js/arena.js` mounting into `#arena-mount`, the dual-column live PolyKAN-vs-ReLU training using the engine, real-time loss/accuracy sparklines, grid-convergence view, and controls (activation/density/lr/m/reset/speed); optionally a live PCA of the user's own runs as an honest replacement/overlay for the illustrative Viz 4.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-23-page-foundation.md`. Continuing Subagent-Driven Development (same as Plan 1).
