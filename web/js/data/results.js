import { mulberry32 } from "../engine/rng.js";

export const SUCCESS_RATES = [
  { name: "PolyKAN",   rate: 1.0,  params: 34, monotonic: false, differentiable: true },
  { name: "PReLU",     rate: 0.97, params: 28, monotonic: true,  differentiable: false },
  { name: "Square",    rate: 0.94, params: 25, monotonic: false, differentiable: true },
  { name: "SiLU",      rate: 0.94, params: 25, monotonic: false, differentiable: true },
  { name: "RootSquare",rate: 0.50, params: 25, monotonic: true,  differentiable: false },
  { name: "LeakyReLU", rate: 0.25, params: 25, monotonic: true,  differentiable: false },
  { name: "CELU",      rate: 0.06, params: 25, monotonic: true,  differentiable: true },
  { name: "Sigmoid",   rate: 0.0,  params: 25, monotonic: true,  differentiable: true },
  { name: "Tanh",      rate: 0.0,  params: 25, monotonic: true,  differentiable: true },
  { name: "ReLU",      rate: 0.0,  params: 25, monotonic: true,  differentiable: false },
];

export const ABLATION = {
  polyKAN: {
    full:       { rate: 1.0,  params: 34 },
    actOnly:    { rate: 1.0,  params: 29 },
    weightOnly: { rate: 0.78, params: 25 },
  },
  prelu: {
    full:       { rate: 0.97, params: 28 },
    actOnly:    { rate: 1.0,  params: 23 },
    weightOnly: { rate: 0.59, params: 25 },
  },
};

const SWEEP_LO = 0.05;
const SWEEP_HI = 0.95;
const SWEEP_STEP = 0.05;

function clamp01(x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
function r2(x) {
  return Math.round(x * 100) / 100;
}
function r3(x) {
  return Math.round(x * 1000) / 1000;
}
function lerpAt(d, anchors) {
  if (d <= anchors[0][0]) return anchors[0][1];
  const last = anchors[anchors.length - 1];
  if (d >= last[0]) return last[1];
  for (let i = 0; i < anchors.length - 1; i++) {
    const [d0, v0] = anchors[i];
    const [d1, v1] = anchors[i + 1];
    if (d >= d0 && d <= d1) {
      const t = (d - d0) / (d1 - d0);
      return v0 + t * (v1 - v0);
    }
  }
  return last[1];
}

const POLYKAN_ANCHORS = [
  [0.05, 1.0], [0.2, 0.9375], [0.3, 0.8125], [0.35, 0.9375],
  [0.4, 1.0], [0.85, 1.0], [0.9, 0.75], [0.95, 0.0],
];
const PRELU_ANCHORS = [
  [0.05, 0.875], [0.5, 1.0], [0.8, 0.9375], [0.9, 0.0625], [0.95, 0.25],
];
const SILU_ANCHORS = [
  [0.05, 0.5], [0.5, 1.0], [0.95, 0.3],
];
const RELU_ANCHORS = [
  [0.05, 0.0], [0.4, 0.15], [0.6, 0.15], [0.95, 0.0],
];

const sweepPoints = [];
for (let d = SWEEP_LO; d <= SWEEP_HI + 1e-9; d += SWEEP_STEP) {
  const dd = r2(d);
  sweepPoints.push({
    density: dd,
    polyKAN: r3(clamp01(lerpAt(dd, POLYKAN_ANCHORS))),
    prelu: r3(clamp01(lerpAt(dd, PRELU_ANCHORS))),
    silu: r3(clamp01(lerpAt(dd, SILU_ANCHORS))),
    relu: r3(clamp01(lerpAt(dd, RELU_ANCHORS))),
  });
}

export const DENSITY_SWEEP = { approx: true, points: sweepPoints };

function polykanPca() {
  const rng = mulberry32(101);
  const pts = [];
  for (let i = 0; i < 24; i++) {
    const t = i / 23;
    const angle = (rng() * 2 - 1) * Math.PI;
    const r = 0.1 + t * 0.9 + rng() * 0.08;
    const loss = r3(clamp01(rng() * 0.3));
    pts.push({
      pc1: r3(r * Math.cos(angle)),
      pc2: r3(r * Math.sin(angle)),
      loss,
      success: loss < 0.5,
    });
  }
  return pts;
}

function reluPca() {
  const rng = mulberry32(202);
  const pts = [];
  for (let i = 0; i < 24; i++) {
    const pc1 = r3(rng() * 2 - 1);
    const pc2 = r3(pc1 * 0.8 + (rng() * 0.4 - 0.2));
    const fail = pc1 < 0;
    const loss = fail ? r3(0.55 + rng() * 0.45) : r3(rng() * 0.45);
    pts.push({ pc1, pc2, loss, success: !fail });
  }
  return pts;
}

function preluPca() {
  const rng = mulberry32(303);
  const pts = [];
  for (let i = 0; i < 24; i++) {
    const branch = i % 2;
    const t = Math.floor(i / 2) / 11;
    const dir = branch === 0 ? 1 : -1;
    const pc1 = r3(t + (rng() * 0.1 - 0.05));
    const pc2 = r3(dir * t * 0.8 + (rng() * 0.1 - 0.05));
    const loss = r3(clamp01(rng() * 0.4));
    pts.push({ pc1, pc2, loss, success: loss < 0.5 });
  }
  return pts;
}

function sigmoidPca() {
  const rng = mulberry32(404);
  const pts = [];
  for (let i = 0; i < 24; i++) {
    const horizontal = i % 2 === 0;
    const pc1 = horizontal ? r3(rng() * 2 - 1) : r3(rng() * 0.2 - 0.1);
    const pc2 = horizontal ? r3(rng() * 0.2 - 0.1) : r3(rng() * 2 - 1);
    const loss = r3(clamp01(0.3 + rng() * 0.5));
    pts.push({ pc1, pc2, loss, success: loss < 0.6 });
  }
  return pts;
}

export const PCA_ILLUSTRATIVE = {
  approx: true,
  polyKAN: polykanPca(),
  relu: reluPca(),
  prelu: preluPca(),
  sigmoid: sigmoidPca(),
};
