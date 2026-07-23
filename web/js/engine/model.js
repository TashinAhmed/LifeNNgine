// L(m,n) ActNN/PolyKAN model, faithful port of src/cann/models/base.py forward().
// Per depth block: Conv2d(1->2m,3x3,circular) -> act(2m) -> Conv2d(2m->1,1x1) -> act(1)
// then Conv2d(1->1,1x1) -> sigmoid.
// Hand-written forward (this task) + backward (next task) + Adam.

import { Activation } from "./activations.js";
import { mulberry32 } from "./rng.js";

export class LifeModel {
  constructor({
    width = 1,
    depth = 1,
    activation = "polyKAN",
    degree = 2,
    seed = 17,
    trainableWeights = true,
    trainableActivations = true,
  } = {}) {
    this.width = width;
    this.depth = depth;
    this.activationKind = activation;
    this.degree = degree;
    this.trainableWeights = trainableWeights;
    this.trainableActivations = trainableActivations;
    this.seed = seed;
    this.H = 0;
    this.W = 0;
    this._build();
  }

  _build() {
    const rng = mulberry32(this.seed);
    this._rng = rng;
    const m = this.width;
    const n = this.depth;

    // Per-block layers. For n=1: block0 conv3x3 (1->2m), conv1x1 (2m->1).
    this.blocks = [];
    for (let i = 0; i < n; i++) {
      const inCh = 1; // single-step input channel
      const conv3 = makeConv(2 * m, inCh, 3, rng); // 3x3, 1 -> 2m (outCh, inCh)
      const conv1 = makeConv(1, 2 * m, 1, rng);     // 1x1 dynamics, 2m -> 1
      const act0 = new Activation(this.activationKind, 2 * m, { degree: this.degree, rng });
      const act1 = new Activation(this.activationKind, 1, { degree: this.degree, rng });
      this.blocks.push({ conv3, conv1, act0, act1 });
    }
    this.outputConv = makeConv(1, 1, 1, rng);

    // Apply freezing for ablations (matches base.py trainable_weights=false logic):
    // freeze dynamics conv1 + outputConv weights/biases.
    if (!this.trainableWeights) {
      for (const b of this.blocks) freezeConv(b.conv1);
      freezeConv(this.outputConv);
    }
    if (!this.trainableActivations) {
      for (const b of this.blocks) {
        for (const p of b.act0.params) p.fill(0); // frozen activations keep coeffs (will not update)
        for (const p of b.act1.params) p.fill(0);
        b.act0.trainable = false;
        b.act1.trainable = false;
      }
    }

    // Build the canonical trainable-parameter list (for counts + optimizer).
    this._collectTrainable();
  }

  _collectTrainable() {
    const list = [];
    for (const b of this.blocks) {
      addConvParams(list, b.conv3, true); // 3x3 always trainable (neighborhood learner)
      addConvParams(list, b.conv1, b.conv1._trainable);
      addActParams(list, b.act0);
      addActParams(list, b.act1);
    }
    addConvParams(list, this.outputConv, this.outputConv._trainable);
    this._trainable = list; // entries: { get, set, grad, m, v, array, idx }
  }

  resize(H, W) {
    this.H = H;
    this.W = W;
  }

  reset(seed) {
    this.seed = seed ?? this.seed;
    this._build();
  }

  trainableCount() {
    let n = 0;
    for (const b of this.blocks) {
      n += countConv(b.conv3, true);
      n += countConv(b.conv1, b.conv1._trainable);
      n += countAct(b.act0);
      n += countAct(b.act1);
    }
    n += countConv(this.outputConv, this.outputConv._trainable);
    return n;
  }

  forward(input) {
    const { H, W } = this;
    if (!H || !W) throw new Error("Call resize(H,W) before forward");
    const m = this.width;
    const cache = { input, blocks: [] };

    // current representation: channels as array of Float32Array(H*W)
    let cur = [Float32Array.from(input)]; // 1 channel

    for (const b of this.blocks) {
      // Snapshot the channels this block actually consumes (correct conv3 input
      // gradient source for any depth, not just depth=1).
      const blockInput = cur;
      // conv3x3: 1 -> 2m, circular
      const z0 = conv3x3Forward(b.conv3, cur, H, W); // [2m][H*W]
      const a0 = applyActivation(b.act0, z0, H, W);  // [2m][H*W]
      // conv1x1: 2m -> 1
      const z1 = conv1x1Forward(b.conv1, a0, H, W);  // [1][H*W]
      const a1 = applyActivation(b.act1, z1, H, W);  // [1][H*W]
      cache.blocks.push({ blockInput, z0, a0, z1, a1 });
      cur = a1;
    }

    const logit = conv1x1Forward(this.outputConv, cur, H, W); // [1][H*W]
    const out = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) out[i] = 1 / (1 + Math.exp(-logit[0][i]));
    cache.logit = logit;
    cache.finalIn = cur;
    this._cache = cache;
    return out;
  }
}

/* ---------- conv helpers ---------- */

// Conv representation: { outCh, inCh, k, W: Float32Array[outCh*inCh*k*k], b: Float32Array[outCh], _trainable }
function makeConv(outCh, inCh, k, rng) {
  const n = outCh * inCh * k * k;
  const W = new Float32Array(n);
  const scale = Math.sqrt(1 / (inCh * k * k));
  for (let i = 0; i < n; i++) W[i] = (rng() * 2 - 1) * scale;
  return { outCh, inCh, k, W, b: new Float32Array(outCh), gW: new Float32Array(n), gb: new Float32Array(outCh), _trainable: true };
}

function freezeConv(c) {
  c._trainable = false;
  // base.py freezes to bias=0, weight=1/numel for dynamics/output layers
  for (let i = 0; i < c.b.length; i++) c.b[i] = 0;
  const v = 1 / c.W.length;
  for (let i = 0; i < c.W.length; i++) c.W[i] = v;
}

function countConv(c, trainable) {
  if (!trainable) return 0;
  return c.W.length + c.b.length;
}
function countAct(a) {
  if (!a.trainable) return 0;
  let n = 0;
  for (const p of a.params) n += p.length;
  return n;
}
function addConvParams(list, c, trainable) {
  if (!trainable) return;
  list.push({ array: c.W, grad: c.gW });
  list.push({ array: c.b, grad: c.gb });
}
function addActParams(list, a) {
  if (!a.trainable) return;
  for (let i = 0; i < a.params.length; i++) list.push({ array: a.params[i], grad: a.grads[i] });
}

// 3x3 conv forward (correlation) with circular padding. inChs: array of channel arrays.
function conv3x3Forward(c, inChs, H, W) {
  const out = [];
  const k = 3;
  for (let oc = 0; oc < c.outCh; oc++) {
    const ch = new Float32Array(H * W);
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        let s = c.b[oc];
        let wi = oc * c.inCh * k * k;
        for (let ic = 0; ic < c.inCh; ic++) {
          const inCh = inChs[ic];
          for (let kh = 0; kh < k; kh++) {
            const hh = (h + kh - 1 + H) % H;
            for (let kw = 0; kw < k; kw++) {
              const ww = (w + kw - 1 + W) % W;
              s += c.W[wi] * inCh[hh * W + ww];
              wi++;
            }
          }
        }
        ch[h * W + w] = s;
      }
    }
    out.push(ch);
  }
  return out;
}

// 1x1 conv forward. inChs: array of channel arrays.
function conv1x1Forward(c, inChs, H, W) {
  const out = [];
  for (let oc = 0; oc < c.outCh; oc++) {
    const ch = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) {
      let s = c.b[oc];
      for (let ic = 0; ic < c.inCh; ic++) s += c.W[oc * c.inCh + ic] * inChs[ic][i];
      ch[i] = s;
    }
    out.push(ch);
  }
  return out;
}

function applyActivation(a, z, H, W) {
  const out = [];
  for (let c = 0; c < a.channels; c++) {
    const ch = new Float32Array(H * W);
    const zc = z[c];
    for (let i = 0; i < H * W; i++) ch[i] = a.value(zc[i], c);
    out.push(ch);
  }
  return out;
}
