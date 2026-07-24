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

// --- live training arena DOM controller ---
//
// Two LifeModels (left = PolyKAN default, right = ReLU) train live via rAF on
// FRESH random batches; a FIXED validation input shows each model's predicted
// next state converging toward the true next state; sparklines plot loss
// (cyan) + accuracy (magenta); a shared step counter. All DOM/canvas/window
// access is inside this function so the module stays Node-importable.
export function createArena(mountEl, opts = {}) {
  const noop = { setPaused() {}, reset() {} };
  if (!mountEl) return noop;
  const doc = mountEl.ownerDocument || (typeof document !== "undefined" ? document : null);
  if (!doc) return noop;

  const {
    activation = "polyKAN",
    width = 1,
    seed = 17,
    H = 32,
    W = 32,
    density = 0.4,
    speed = 30,
    lr = 1e-3,
    valSeed = 4242,
    ringCap = 400,
  } = opts;

  const leftLabel = String(activation).toLowerCase() === "polykan" ? "PolyKAN" : String(activation);
  const updatesPerFrame = Math.max(1, Math.min(200, Math.round(speed)));

  // ----- DOM (built into mountEl) -----
  mountEl.innerHTML = "";
  const arenaEl = doc.createElement("div");
  arenaEl.className = "arena";

  // Header bar: tag + shared step counter + pause/reset. (The chapter h2 already
  // reads "Watch it learn, live", so the bar carries an informational tag instead
  // of repeating that title.)
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
  pauseBtn.textContent = "Pause";
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

  // Two model columns.
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
    return { col, pred, readout, spark };
  }
  const leftCol = makeCol(leftLabel);
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
  let userPaused = false;
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
    // Fixed validation input (seeded, stable across frames; regenerated on reset).
    const valRng = mulberry32(valSeed);
    valInput = new Float32Array(H * W);
    for (let i = 0; i < valInput.length; i++) valInput[i] = valRng() < density ? 1 : 0;
    valTrue = lifeStep(valInput, H, W);

    leftModel = new LifeModel({ width, depth: 1, activation, seed });
    leftModel.resize(H, W);
    rightModel = new LifeModel({ width, depth: 1, activation: "relu", seed });
    rightModel.resize(H, W);

    // Distinct training stream (XOR'd seed) so it diverges from model init.
    trainRng = mulberry32((seed ^ 0x9e3779b9) >>> 0);
    t = 0;
    stepCount = 0;
    for (const r of [leftLoss, leftAcc, rightLoss, rightAcc]) { r.head = 0; r.len = 0; }

    stepsEl.textContent = "0";
    drawBinary(inputCanvas, valInput, "#39ff14");
    drawBinary(trueCanvas, valTrue, "#39ff14");
  }

  // ----- rendering -----
  function drawBinary(canvas, grid, onColor) {
    const view = fitCanvas(canvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;
    const cellPx = Math.min(cssW / W, cssH / H);
    clearCanvas(ctx, canvas.width, canvas.height);
    drawGrid(ctx, grid, H, W, cellPx, { on: onColor, off: "#11151a", gridline: "#1c2025" });
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
        ctx.fillStyle = on ? onColor : "#11151a";
        ctx.fillRect(w * cellPx, h * cellPx, cellPx, cellPx);
      }
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#1c2025";
    ctx.lineWidth = 1;
    for (let h = 0; h <= H; h++) { ctx.beginPath(); ctx.moveTo(0, h * cellPx); ctx.lineTo(W * cellPx, h * cellPx); ctx.stroke(); }
    for (let w = 0; w <= W; w++) { ctx.beginPath(); ctx.moveTo(w * cellPx, 0); ctx.lineTo(w * cellPx, H * cellPx); ctx.stroke(); }
  }

  function drawSpark(canvas, lossRing, accRing) {
    const view = fitCanvas(canvas);
    if (!view || !view.ctx) return;
    const { ctx, cssW, cssH } = view;
    clearCanvas(ctx, canvas.width, canvas.height, "#0b0d10");
    const padL = 6, padR = 6, padT = 5, padB = 5;
    const plotW = Math.max(1, cssW - padL - padR);
    const plotH = Math.max(1, cssH - padT - padB);
    ctx.strokeStyle = "#1c2025";
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
    line(lossRing, (v) => padT + plotH * (1 - Math.min(1, v / maxLoss)), "#22d3ee"); // cyan = loss
    line(accRing, (v) => padT + plotH * (1 - v), "#e879f9"); // magenta = accuracy
  }

  // ----- training loop -----
  // One update: a SINGLE fresh batch trains BOTH models with the SAME t — a fair
  // head-to-head. trainStep does zeroGrad -> forward -> backward -> step.
  function trainOnce() {
    const batch = makeBatch(H, W, density, trainRng);
    t++;
    const lRes = trainStep(leftModel, batch, t, lr);
    const rRes = trainStep(rightModel, batch, t, lr);
    ringPush(leftLoss, lRes.loss);
    ringPush(rightLoss, rRes.loss);
    ringPush(leftAcc, gridAccuracy(lRes.pred, batch.target));
    ringPush(rightAcc, gridAccuracy(rRes.pred, batch.target));
    stepCount++;
  }

  function render() {
    // Re-run forward on the FIXED validation input to show convergence on the
    // example (independent of the fresh training batches).
    const lp = leftModel.forward(valInput);
    const rp = rightModel.forward(valInput);
    drawPred(leftCol.pred, lp, "#39ff14");  // green
    drawPred(rightCol.pred, rp, "#ffb454"); // amber
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

  pauseBtn.addEventListener("click", () => {
    userPaused = !userPaused;
    pauseBtn.textContent = userPaused ? "Resume" : "Pause";
    if (userPaused) stop(); else { render(); start(); }
  });
  function reset() {
    build();
    render();
    start(); // no-op if already running or user/vis paused
  }
  resetBtn.addEventListener("click", reset);

  const win = doc.defaultView || (typeof window !== "undefined" ? window : null);
  if (win) win.addEventListener("resize", () => {
    drawBinary(inputCanvas, valInput, "#39ff14");
    drawBinary(trueCanvas, valTrue, "#39ff14");
    render();
  });

  build();
  render();
  start();

  return {
    // Visibility pause (IntersectionObserver): cancel/restart the RAF.
    setPaused(p) {
      visPaused = !!p;
      if (visPaused) stop(); else start();
    },
    reset,
  };
}
