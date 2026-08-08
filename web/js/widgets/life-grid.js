// Widget A — interactive Game of Life grid.
//
// Split into two layers so the module stays Node-importable:
//   - PRESETS / stampPreset are pure (no DOM) and unit-tested.
//   - createLifeGrid is the DOM controller; all DOM/canvas/window access
//     lives inside it, never at module top level.

import { lifeStep } from "../engine/life.js";
import { fitCanvas, clearCanvas, drawGrid } from "../util/canvas.js";

// Preset patterns as [row, col] offsets from a stamp origin.
export const PRESETS = {
  glider: [[0, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
  blinker: [[0, 0], [0, 1], [0, 2]],
  block: [[0, 0], [0, 1], [1, 0], [1, 1]],
  // Pulsar (period-3 oscillator, 48 cells) in its canonical 13x13 box.
  pulsar: [
    [0, 2], [0, 3], [0, 4], [0, 8], [0, 9], [0, 10],
    [2, 0], [2, 5], [2, 7], [2, 12],
    [3, 0], [3, 5], [3, 7], [3, 12],
    [4, 0], [4, 5], [4, 7], [4, 12],
    [5, 2], [5, 3], [5, 4], [5, 8], [5, 9], [5, 10],
    [7, 2], [7, 3], [7, 4], [7, 8], [7, 9], [7, 10],
    [8, 0], [8, 5], [8, 7], [8, 12],
    [9, 0], [9, 5], [9, 7], [9, 12],
    [10, 0], [10, 5], [10, 7], [10, 12],
    [12, 2], [12, 3], [12, 4], [12, 8], [12, 9], [12, 10],
  ],
};

// Stamp a preset at (originR, originC) with toroidal wrap. Pure: returns a new
// Uint8Array, never mutates the input grid.
export function stampPreset(grid, H, W, preset, originR, originC) {
  const out = new Uint8Array(grid);
  for (const [dr, dc] of preset) {
    const r = (((originR + dr) % H) + H) % H;
    const c = (((originC + dc) % W) + W) % W;
    out[r * W + c] = 1;
  }
  return out;
}

// DOM controller for Widget A. All DOM/canvas access is inside this function.
// Returns { setPaused, reset, step }, or a no-op stub if the mount is missing.
export function createLifeGrid(canvas, controlsEl, opts = {}) {
  const noop = { setPaused() {}, reset() {}, step() {} };
  if (!canvas || !controlsEl) return noop;

  const { H = 32, W = 32 } = opts;
  const STEP_MS = 1000 / 6; // ~6 fps stepping while playing

  let grid = new Uint8Array(H * W);
  let flash = new Uint8Array(H * W); // cells that changed this step (one frame)
  let running = false;   // Play/Pause toggle
  let paused = false;    // visibility pause (setPaused), set later by IntersectionObserver
  let cellPx = 0;        // recomputed each render; used by pointer paint
  let originX = 0;       // pixel offset of the grid inside the canvas (for paint hit-test)
  let originY = 0;
  let raf = 0;
  let lastStep = 0;

  const doc = controlsEl.ownerDocument;

  function render() {
    const view = fitCanvas(canvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;
    cellPx = Math.min(cssW / W, cssH / H);
    // Center the square grid inside the (possibly wider) canvas so it never
    // hugs the left edge. With aspect-ratio:1/1 in CSS this is usually a no-op,
    // but it keeps the widget correct on any canvas shape.
    originX = (cssW - W * cellPx) / 2;
    originY = (cssH - H * cellPx) / 2;
    clearCanvas(ctx, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(originX, originY);
    drawGrid(ctx, grid, H, W, cellPx, { flash });
    ctx.restore();
    // Flash is a one-frame highlight of births/deaths; clear after drawing.
    flash.fill(0);
  }

  function doStep() {
    const prev = grid;
    const next = lifeStep(grid, H, W);
    for (let i = 0; i < next.length; i++) {
      if (prev[i] !== next[i]) flash[i] = 1;
    }
    grid = new Uint8Array(next); // lifeStep returns Float32Array; normalize to Uint8Array
  }

  function loop(t) {
    if (running && !paused && t - lastStep >= STEP_MS) {
      doStep();
      lastStep = t;
    }
    render();
    raf = requestAnimationFrame(loop);
  }

  // RAF lifecycle mirrors the hero pattern: cancel on pause (so the loop stops
  // reallocating the canvas bitmap every frame), restart on resume.
  function stop() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
  }
  function start() {
    if (raf || paused) return;
    raf = requestAnimationFrame(loop);
  }

  function randomize() {
    for (let i = 0; i < grid.length; i++) grid[i] = Math.random() < density ? 1 : 0;
    flash.fill(0);
  }

  function clearGrid() {
    grid.fill(0);
    flash.fill(0);
  }

  // ----- Pointer paint (drag to draw; Shift-drag to erase) -----
  let painting = false;
  function paintAt(e) {
    if (!cellPx) return;
    const rect = canvas.getBoundingClientRect();
    // Account for the centered origin so painting maps to the visible cell.
    const c = Math.floor((e.clientX - rect.left - originX) / cellPx);
    const r = Math.floor((e.clientY - rect.top - originY) / cellPx);
    if (r < 0 || r >= H || c < 0 || c >= W) return;
    grid[r * W + c] = e.shiftKey ? 0 : 1;
  }
  canvas.addEventListener("pointerdown", (e) => {
    painting = true;
    try { canvas.setPointerCapture(e.pointerId); } catch { /* capture optional */ }
    paintAt(e);
  });
  canvas.addEventListener("pointermove", (e) => { if (painting) paintAt(e); });
  const stopPaint = () => { painting = false; };
  canvas.addEventListener("pointerup", stopPaint);
  canvas.addEventListener("pointercancel", stopPaint);
  canvas.addEventListener("pointerleave", stopPaint);

  // ----- Controls -----
  function makeButton(label, onClick) {
    const b = doc.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.addEventListener("click", onClick);
    return b;
  }

  let density = 0.4;
  const densityLabel = doc.createElement("label");
  const densityInput = doc.createElement("input");
  densityInput.type = "range";
  densityInput.min = "0.05";
  densityInput.max = "0.95";
  densityInput.step = "0.05";
  densityInput.value = String(density);
  const updateDensityLabel = () => { densityLabel.textContent = `density ${density.toFixed(2)}`; };
  updateDensityLabel();
  densityInput.addEventListener("input", () => {
    density = Number(densityInput.value) || 0;
    updateDensityLabel();
  });

  const playBtn = makeButton("Play", () => {
    running = !running;
    playBtn.textContent = running ? "Pause" : "Play";
  });

  const stepBtn = makeButton("Step", () => { doStep(); });
  const randBtn = makeButton("Randomize", () => { randomize(); });
  const clearBtn = makeButton("Clear", () => { clearGrid(); });

  // Preset buttons stamp the pattern at the grid center.
  const presetBtns = Object.keys(PRESETS).map((name) =>
    makeButton(name, () => {
      grid = stampPreset(grid, H, W, PRESETS[name], (H / 2) | 0, (W / 2) | 0);
      flash.fill(0);
    })
  );

  controlsEl.append(stepBtn, playBtn, randBtn, densityLabel, densityInput, clearBtn, ...presetBtns);

  start();

  return {
    // Visibility pause hook for a later IntersectionObserver. Cancels the RAF
    // (then renders once so the static frame stays correct) and restarts it on
    // resume; the running/Play toggle is untouched, so Play keeps working.
    setPaused(p) {
      paused = !!p;
      if (paused) { stop(); render(); }
      else start();
    },
    // Clear the grid and stop playback.
    reset() { running = false; playBtn.textContent = "Play"; clearGrid(); },
    // Advance one generation (exposed for external triggering / testing).
    step() { doStep(); },
  };
}
