import { lifeStep } from "./engine/life.js";
import { LifeModel } from "./engine/model.js";
import { mulberry32 } from "./engine/rng.js";
import { fitCanvas, clearCanvas, drawGrid } from "./util/canvas.js";

// --- pure training helpers (DOM-free, Node-tested) ---

export function makeBatch(H, W, density, rng) {
  const input = new Float32Array(H * W);
  for (let i = 0; i < input.length; i++) input[i] = rng() < density ? 1 : 0;
  return { input, target: lifeStep(input, H, W) };
}

export function gridAccuracy(pred, target) {
  let correct = 0;
  for (let i = 0; i < pred.length; i++) {
    if ((pred[i] > 0.5 ? 1 : 0) === (target[i] > 0.5 ? 1 : 0)) correct++;
  }
  return correct / pred.length;
}

export function trainStep(model, batch, t, lr = 1e-3) {
  model.zeroGrad();
  const pred = model.forward(batch.input);
  model.backward(batch.target);
  model.step(lr, t);
  return { loss: model.computeLoss(pred, batch.target), pred };
}

// --- pure control helpers (DOM-free) ---
//
// Left-column activations offered in the arena selector; the values MUST match
// the keys accepted by engine/activations.js. The right column stays ReLU as a
// fixed baseline, so it is not in the selector.
export const ARENA_ACTIVATIONS = [
  { value: "polyKAN", label: "PolyKAN" },
  { value: "prelu", label: "PReLU" },
  { value: "silu", label: "SiLU" },
  { value: "square", label: "Square" },
  { value: "relu", label: "ReLU" },
];

export function activationLabel(kind) {
  const k = String(kind).toLowerCase();
  const found = ARENA_ACTIVATIONS.find((a) => a.value === k || a.value.toLowerCase() === k);
  return found ? found.label : String(kind);
}

// Normalize any-case input (opts or the <select>) to the canonical casing the
// engine expects (activations.js switch is case-sensitive: "polyKAN", etc.).
export function canonicalActivation(kind) {
  const k = String(kind).toLowerCase();
  const found = ARENA_ACTIVATIONS.find((a) => a.value.toLowerCase() === k);
  return found ? found.value : String(kind);
}

// Learning-rate slider maps an integer position [0..LR_STEPS] onto a log scale
// spanning [LR_MIN..LR_MAX]; slider steps are linear in position but the
// underlying lr is exponential, so the whole useful range is reachable.
export const LR_MIN = 1e-4;
export const LR_MAX = 1e-2;
export const LR_STEPS = 1000;

export function lrFromSlider(v) {
  const t = Math.max(0, Math.min(1, Number(v) / LR_STEPS));
  return LR_MIN * Math.pow(LR_MAX / LR_MIN, t);
}
export function lrToSlider(lr) {
  const clamped = Math.max(LR_MIN, Math.min(LR_MAX, lr));
  return Math.round(
    (Math.log(clamped) - Math.log(LR_MIN)) / (Math.log(LR_MAX) - Math.log(LR_MIN)) * LR_STEPS,
  );
}

export function clampDensity(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return 0.4;
  return Math.max(0.05, Math.min(0.95, v));
}
export function clampLr(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return 1e-3;
  return Math.max(LR_MIN, Math.min(LR_MAX, v));
}
export function clampSpeed(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return 300;
  return Math.max(1, Math.min(500, Math.round(v)));
}

// --- live training arena DOM controller ---
//
// Two LifeModels (left = PolyKAN default, right = ReLU) train live via rAF on
// FRESH random batches; a FIXED validation input shows each model's predicted
// next state converging toward the true next state; sparklines plot loss
// (cyan) + accuracy (magenta); a shared step counter. A controls bar lets the
// reader steer activation / density / lr / width / seed / speed live.
// density/lr/speed take effect on the next frame; activation/width/seed rebuild
// the models via reset(). All DOM/canvas/window access is inside this function
// so the module stays Node-importable (no module-top-level DOM).
export function createArena(mountEl, opts = {}) {
  const noop = { setPaused() {}, reset() {} };
  if (!mountEl) return noop;
  const doc = mountEl.ownerDocument || (typeof document !== "undefined" ? document : null);
  if (!doc) return noop;
  const win = doc.defaultView || (typeof window !== "undefined" ? window : null);

  const {
    H = 32,
    W = 32,
    valSeed = 4242,
    ringCap = 400,
    // Default speed ~300 updates/frame: PolyKAN reaches ~75% quickly, plateaus
    // at ~75% from ~3k-30k batch-1 updates, then climbs to 100% around ~30k-60k
    // updates (the ~60k cliff lands in ~3-4s at 300/frame). ReLU stays near the
    // ~58% majority-class baseline throughout. The engine is cheap; even at the
    // 500/frame clamp ceiling each rAF still yields to the browser and keeps the
    // UI responsive.
    speed = 300,
  } = opts;

  // Live-mutable hyperparameters. density/lr/speed read by the loop directly;
  // activation/width/seed are consumed by build() and only change via reset().
  // curActivation is kept in the engine's canonical casing (polyKAN, relu, ...).
  let curActivation = canonicalActivation(opts.activation ?? "polyKAN");
  let curWidth = Math.max(1, Math.round(opts.width ?? 1));
  let curSeed = (opts.seed | 0) || 1;
  let curDensity = clampDensity(opts.density ?? 0.4);
  let curLr = clampLr(opts.lr ?? 1e-3);
  let updatesPerFrame = clampSpeed(speed);

  // ----- DOM (built into mountEl) -----
  mountEl.innerHTML = "";
  const arenaEl = doc.createElement("div");
  arenaEl.className = "arena";

  // Controls bar: activation / density / lr / width / seed / speed. Wraps on
  // narrow screens (flex-wrap) — see .arena__controls in style.css.
  const controls = doc.createElement("div");
  controls.className = "arena__controls controls";

  function field(labelText, ...kids) {
    const lab = doc.createElement("label");
    lab.className = "arena__field";
    const txt = doc.createElement("span");
    txt.className = "arena__field-label";
    txt.textContent = labelText;
    lab.appendChild(txt);
    for (const k of kids) if (k) lab.appendChild(k);
    return lab;
  }

  // Activation <select> — drives the LEFT column only (right stays ReLU).
  const actSel = doc.createElement("select");
  actSel.className = "arena__select";
  actSel.setAttribute("aria-label", "Left column activation");
  for (const a of ARENA_ACTIVATIONS) {
    const o = doc.createElement("option");
    o.value = a.value;
    o.textContent = a.label;
    if (a.value === curActivation) o.selected = true;
    actSel.appendChild(o);
  }

  // Density slider — live (next batch uses the new density).
  const densRange = doc.createElement("input");
  densRange.type = "range";
  densRange.min = "0.05";
  densRange.max = "0.95";
  densRange.step = "0.01";
  densRange.value = curDensity.toFixed(2);
  densRange.setAttribute("aria-label", "Density");
  const densVal = doc.createElement("span");
  densVal.className = "arena__val";
  densVal.textContent = curDensity.toFixed(2);

  // Learning-rate slider — log scale (position 0..LR_STEPS), live.
  const lrRange = doc.createElement("input");
  lrRange.type = "range";
  lrRange.min = "0";
  lrRange.max = String(LR_STEPS);
  lrRange.step = "1";
  lrRange.value = String(lrToSlider(curLr));
  lrRange.setAttribute("aria-label", "Learning rate");
  const lrVal = doc.createElement("span");
  lrVal.className = "arena__val";
  lrVal.textContent = curLr.toExponential(1);

  // Width-m button group — reset() with the new width.
  const mGroup = doc.createElement("span");
  mGroup.className = "arena__mgroup";
  mGroup.setAttribute("role", "group");
  mGroup.setAttribute("aria-label", "Width m");
  const mBtns = {};
  for (const m of [1, 2, 4]) {
    const b = doc.createElement("button");
    b.type = "button";
    b.textContent = String(m);
    b.dataset.m = String(m);
    if (m === curWidth) b.classList.add("is-active");
    mBtns[m] = b;
    mGroup.appendChild(b);
  }

  // Seed input + reshuffle — both reset() with the (new) seed.
  const seedInput = doc.createElement("input");
  seedInput.type = "number";
  seedInput.min = "0";
  seedInput.step = "1";
  seedInput.value = String(curSeed);
  seedInput.className = "arena__seed";
  seedInput.setAttribute("aria-label", "Random seed");
  const reshuffleBtn = doc.createElement("button");
  reshuffleBtn.type = "button";
  reshuffleBtn.textContent = "reshuffle";

  // Speed slider — live (updates per frame).
  const speedRange = doc.createElement("input");
  speedRange.type = "range";
  speedRange.min = "1";
  speedRange.max = "500";
  speedRange.step = "1";
  speedRange.value = String(updatesPerFrame);
  speedRange.setAttribute("aria-label", "Updates per frame");
  const speedVal = doc.createElement("span");
  speedVal.className = "arena__val";
  speedVal.textContent = String(updatesPerFrame);

  controls.append(
    field("activation", actSel),
    field("density", densRange, densVal),
    field("lr", lrRange, lrVal),
    field("m", mGroup),
    field("seed", seedInput, reshuffleBtn),
    field("speed", speedRange, speedVal),
  );
  arenaEl.appendChild(controls);

  // Header bar: tag + shared step counter + pause/reset. (The chapter h2
  // already reads "Watch it learn, live", so the bar carries an informational
  // tag instead of repeating that title.)
  const bar = doc.createElement("div");
  bar.className = "arena__bar";
  const tag = doc.createElement("span");
  tag.className = "arena__tag";
  tag.textContent = "live · gradient descent";
  const stepsWrap = doc.createElement("span");
  stepsWrap.className = "arena__steps";
  stepsWrap.textContent = "steps ";
  const stepsEl = doc.createElement("strong");
  stepsEl.id = "arena-steps";
  stepsEl.textContent = "0";
  stepsWrap.appendChild(stepsEl);
  const pauseBtn = doc.createElement("button");
  pauseBtn.type = "button";
  pauseBtn.textContent = "Play";
  const resetBtn = doc.createElement("button");
  resetBtn.type = "button";
  resetBtn.textContent = "Reset";
  bar.append(tag, stepsWrap, pauseBtn, resetBtn);
  arenaEl.appendChild(bar);

  // Reference row: the FIXED validation input and its true next state.
  function makeRef(labelText, id) {
    const item = doc.createElement("div");
    item.className = "arena__ref-item";
    const lab = doc.createElement("div");
    lab.className = "arena__ref-label";
    lab.textContent = labelText;
    const c = doc.createElement("canvas");
    c.id = id;
    c.width = 128;
    c.height = 128;
    item.append(lab, c);
    return { item, canvas: c };
  }
  const ref = doc.createElement("div");
  ref.className = "arena__ref";
  const refIn = makeRef("input (t)", "arena-input");
  const refTrue = makeRef("true next (t+1)", "arena-true");
  const refCap = doc.createElement("div");
  refCap.className = "arena__ref-cap";
  refCap.textContent = "the rule it must learn";
  ref.append(refIn.item, refTrue.item, refCap);
  arenaEl.appendChild(ref);
  const inputCanvas = refIn.canvas;
  const trueCanvas = refTrue.canvas;

  // Two model columns. `lab` is returned so setActivation can relabel the left
  // column when the reader picks a new activation.
  const cols = doc.createElement("div");
  cols.className = "arena__cols";
  function makeCol(labelText) {
    const col = doc.createElement("div");
    col.className = "arena__col";
    const lab = doc.createElement("div");
    lab.className = "arena__col-label";
    lab.textContent = labelText;
    const pred = doc.createElement("canvas");
    pred.className = "arena-pred";
    pred.width = 288;
    pred.height = 288;
    const readout = doc.createElement("div");
    readout.className = "arena__readout";
    readout.textContent = "accuracy: 0.0%";
    const spark = doc.createElement("canvas");
    spark.className = "arena-spark";
    spark.width = 288;
    spark.height = 56;
    col.append(lab, pred, readout, spark);
    return { col, lab, pred, readout, spark };
  }
  const leftCol = makeCol(activationLabel(curActivation));
  const rightCol = makeCol("ReLU");
  cols.append(leftCol.col, rightCol.col);
  arenaEl.appendChild(cols);

  mountEl.appendChild(arenaEl);

  // ----- state -----
  let leftModel, rightModel;
  let valInput, valTrue;
  let trainRng;
  let t = 0;
  let stepCount = 0;
  let refGen = 0;                // how many Life steps the reference grid has advanced
  const REF_ADVANCE_EVERY = 100; // training steps between reference advances
  let userPaused = true; // start paused - reader hits Play when ready
  let visPaused = false;
  let raf = 0;

  function makeRing(cap) {
    return { arr: new Float32Array(cap), head: 0, len: 0, cap };
  }
  function ringPush(r, v) {
    r.arr[r.head] = v;
    r.head = (r.head + 1) % r.cap;
    if (r.len < r.cap) r.len++;
  }
  function ringMax(r) {
    let m = 0;
    for (let i = 0; i < r.len; i++) if (r.arr[i] > m) m = r.arr[i];
    return m;
  }
  const leftLoss = makeRing(ringCap), leftAcc = makeRing(ringCap);
  const rightLoss = makeRing(ringCap), rightAcc = makeRing(ringCap);

  function build() {
    // Fixed validation input (seeded, stable across frames; regenerated on
    // reset so a new seed/width/density reshapes the example too).
    const valRng = mulberry32(valSeed);
    valInput = new Float32Array(H * W);
    for (let i = 0; i < valInput.length; i++) valInput[i] = valRng() < curDensity ? 1 : 0;
    valTrue = lifeStep(valInput, H, W);

    leftModel = new LifeModel({ width: curWidth, depth: 1, activation: curActivation, seed: curSeed });
    leftModel.resize(H, W);
    rightModel = new LifeModel({ width: curWidth, depth: 1, activation: "relu", seed: curSeed });
    rightModel.resize(H, W);

    // Distinct training stream (XOR'd seed) so it diverges from model init.
    trainRng = mulberry32((curSeed ^ 0x9e3779b9) >>> 0);
    t = 0;
    stepCount = 0;
    refGen = 0;
    for (const r of [leftLoss, leftAcc, rightLoss, rightAcc]) { r.head = 0; r.len = 0; }

    stepsEl.textContent = "0";
    drawBinary(inputCanvas, valInput, "#16a34a");
    drawBinary(trueCanvas, valTrue, "#16a34a");
  }

  // ----- rendering -----
  function drawBinary(canvas, grid, onColor) {
    const view = fitCanvas(canvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;
    const cellPx = Math.min(cssW / W, cssH / H);
    clearCanvas(ctx, canvas.width, canvas.height);
    drawGrid(ctx, grid, H, W, cellPx, { on: onColor, off: "#f6f8fa", gridline: "#cdd2da" });
  }

  // Predicted grid: shade each cell by confidence (|p-0.5|*2), on-color where
  // p>0.5, so you can watch certainty crystallize as training converges.
  function drawPred(canvas, pred, onColor) {
    const view = fitCanvas(canvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;
    const cellPx = Math.min(cssW / W, cssH / H);
    clearCanvas(ctx, canvas.width, canvas.height);
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        const p = pred[h * W + w];
        const on = p > 0.5;
        const conf = on ? (p - 0.5) * 2 : (0.5 - p) * 2;
        ctx.globalAlpha = 0.12 + 0.88 * conf;
        ctx.fillStyle = on ? onColor : "#f6f8fa";
        ctx.fillRect(w * cellPx, h * cellPx, cellPx, cellPx);
      }
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#cdd2da";
    ctx.lineWidth = 1;
    for (let h = 0; h <= H; h++) { ctx.beginPath(); ctx.moveTo(0, h * cellPx); ctx.lineTo(W * cellPx, h * cellPx); ctx.stroke(); }
    for (let w = 0; w <= W; w++) { ctx.beginPath(); ctx.moveTo(w * cellPx, 0); ctx.lineTo(w * cellPx, H * cellPx); ctx.stroke(); }
  }

  function drawSpark(canvas, lossRing, accRing) {
    const view = fitCanvas(canvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;
    clearCanvas(ctx, canvas.width, canvas.height, "#e9ecf1");
    const padL = 6, padR = 6, padT = 5, padB = 5;
    const plotW = Math.max(1, cssW - padL - padR);
    const plotH = Math.max(1, cssH - padT - padB);
    ctx.strokeStyle = "#cdd2da";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH); ctx.stroke();

    const n = Math.max(lossRing.len, accRing.len);
    if (n < 2) return;
    const maxLoss = Math.max(1e-6, ringMax(lossRing));
    const xOf = (i) => padL + (plotW * i) / (n - 1);

    // Iterate oldest -> newest. Before the ring fills, data is at [0,len); once
    // full, `head` points at the oldest sample (next to be overwritten).
    function line(ring, valToY, color) {
      const start = ring.len < ring.cap ? 0 : ring.head;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let k = 0; k < ring.len; k++) {
        const v = ring.arr[(start + k) % ring.cap];
        const x = xOf(k), y = valToY(v);
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    line(lossRing, (v) => padT + plotH * (1 - Math.min(1, v / maxLoss)), "#0891b2"); // cyan = loss
    line(accRing, (v) => padT + plotH * (1 - v), "#c026d3"); // magenta = accuracy
  }

  // ----- training loop -----
  // One update: a SINGLE fresh batch trains BOTH models with the SAME t — a
  // fair head-to-head. trainStep does zeroGrad -> forward -> backward -> step.
  // Reads live state (curDensity/curLr) so the sliders take effect instantly.
  function trainOnce() {
    const batch = makeBatch(H, W, curDensity, trainRng);
    t++;
    const lRes = trainStep(leftModel, batch, t, curLr);
    const rRes = trainStep(rightModel, batch, t, curLr);
    ringPush(leftLoss, lRes.loss);
    ringPush(rightLoss, rRes.loss);
    ringPush(leftAcc, gridAccuracy(lRes.pred, batch.target));
    ringPush(rightAcc, gridAccuracy(rRes.pred, batch.target));
    stepCount++;
    if (stepCount % REF_ADVANCE_EVERY === 0) {
      // Evolve the displayed reference by one Life generation so gliders
      // and ash emerge over a long run. Training batches are unaffected.
      valInput = lifeStep(valInput, H, W);
      valTrue = lifeStep(valInput, H, W);
      refGen++;
      drawBinary(inputCanvas, valInput, "#16a34a");
      drawBinary(trueCanvas, valTrue, "#16a34a");
    }
  }

  function render() {
    // Re-run forward on the FIXED validation input to show convergence on the
    // example (independent of the fresh training batches).
    const lp = leftModel.forward(valInput);
    const rp = rightModel.forward(valInput);
    drawPred(leftCol.pred, lp, "#16a34a");  // green
    drawPred(rightCol.pred, rp, "#d97706"); // amber
    const lAcc = gridAccuracy(lp, valTrue);
    const rAcc = gridAccuracy(rp, valTrue);
    leftCol.readout.textContent = `accuracy: ${(lAcc * 100).toFixed(1)}%`;
    rightCol.readout.textContent = `accuracy: ${(rAcc * 100).toFixed(1)}%`;
    drawSpark(leftCol.spark, leftLoss, leftAcc);
    drawSpark(rightCol.spark, rightLoss, rightAcc);
    stepsEl.textContent = String(stepCount);
  }

  function frame() {
    for (let i = 0; i < updatesPerFrame; i++) trainOnce();
    render();
    raf = requestAnimationFrame(frame);
  }

  // RAF lifecycle mirrors the hero/life-grid pattern: cancel on pause (so the
  // loop stops re-allocating the canvas bitmap each frame), restart on resume.
  function stop() { if (raf) cancelAnimationFrame(raf); raf = 0; }
  function start() { if (raf || userPaused || visPaused) return; raf = requestAnimationFrame(frame); }

  // ----- setters (used by the controls bar) -----
  // activation/width/seed rebuild models + clear buffers via reset(); the
  // caller is responsible for any UI affordance (e.g. button active-state).
  function setActivation(kind) {
    curActivation = canonicalActivation(kind);
    leftCol.lab.textContent = activationLabel(curActivation);
    reset();
  }
  function setWidth(m) {
    curWidth = Math.max(1, Math.round(m));
    for (const k of [1, 2, 4]) mBtns[k].classList.toggle("is-active", k === curWidth);
    reset();
  }
  function setSeed(s) {
    const v = (s | 0) || 1;
    curSeed = v;
    seedInput.value = String(v);
    reset();
  }

  // ----- control wiring -----
  actSel.addEventListener("change", () => setActivation(actSel.value));

  densRange.addEventListener("input", () => {
    curDensity = clampDensity(parseFloat(densRange.value));
    densVal.textContent = curDensity.toFixed(2);
  });

  lrRange.addEventListener("input", () => {
    curLr = clampLr(lrFromSlider(parseInt(lrRange.value, 10)));
    lrVal.textContent = curLr.toExponential(1);
  });

  mGroup.addEventListener("click", (e) => {
    const target = e.target.closest("button[data-m]");
    if (!target) return;
    setWidth(parseInt(target.dataset.m, 10));
  });

  reshuffleBtn.addEventListener("click", () => {
    // Pick a fresh random seed (the reshuffle action), then reset with it.
    const v = (Math.random() * 1e9) | 0;
    setSeed(v);
  });

  seedInput.addEventListener("change", () => {
    const v = parseInt(seedInput.value, 10);
    if (Number.isFinite(v)) setSeed(v);
  });

  speedRange.addEventListener("input", () => {
    updatesPerFrame = clampSpeed(parseInt(speedRange.value, 10));
    speedVal.textContent = String(updatesPerFrame);
  });

  pauseBtn.addEventListener("click", () => {
    userPaused = !userPaused;
    pauseBtn.textContent = userPaused ? "Play" : "Pause";
    if (userPaused) stop(); else { render(); start(); }
  });
  function reset() {
    build();
    render();
    start(); // no-op if already running or user/vis paused
  }
  resetBtn.addEventListener("click", reset);

  if (win) win.addEventListener("resize", () => {
    drawBinary(inputCanvas, valInput, "#16a34a");
    drawBinary(trueCanvas, valTrue, "#16a34a");
    render();
  });

  // Reduced-motion (spec §10): do NOT auto-start; surface a Play button so the
  // reader begins training explicitly. Read from matchMedia inside this
  // function (not at module top level) so the module stays Node-importable.
  function prefersReducedMotion() {
    try {
      return !!win && typeof win.matchMedia === "function"
        && win.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch {
      return false;
    }
  }

  build();
  render();
  if (prefersReducedMotion()) {
    userPaused = true;
    pauseBtn.textContent = "Play";
  } else {
    start();
  }

  return {
    // Visibility pause (IntersectionObserver): cancel/restart the RAF.
    setPaused(p) {
      visPaused = !!p;
      if (visPaused) stop(); else start();
    },
    reset,
    // Exposed for smoke/static-verify tests: live hyperparameter accessors.
    getDensity: () => curDensity,
    getLr: () => curLr,
    getSpeed: () => updatesPerFrame,
    getActivation: () => curActivation,
    getWidth: () => curWidth,
    getSeed: () => curSeed,
  };
}
