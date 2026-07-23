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
      // base.py freezes activations via requires_grad=False but KEEPS their
      // initialized values (does NOT zero them). The optimizer excludes
      // non-trainable activations, and accumulateGrad early-returns when
      // !trainable, so simply flipping the flag is the faithful freeze.
      for (const b of this.blocks) {
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

  zeroGrad() {
    const clear = (c) => { c.gW.fill(0); c.gb.fill(0); };
    for (const b of this.blocks) { clear(b.conv3); clear(b.conv1); b.act0.zeroGrad(); b.act1.zeroGrad(); }
    clear(this.outputConv);
  }

  // BCE loss against a 0/1 target, returns mean loss; sets this.lossValue.
  computeLoss(pred, target) {
    let s = 0;
    const n = pred.length;
    for (let i = 0; i < n; i++) {
      const p = Math.min(Math.max(pred[i], 1e-7), 1 - 1e-7);
      s += -(target[i] * Math.log(p) + (1 - target[i]) * Math.log(1 - p));
    }
    this.lossValue = s / n;
    return this.lossValue;
  }

  // Backprop assuming forward(input) just ran. target: Float32Array (0/1).
  backward(target) {
    const { H, W, _cache: cache } = this;
    const pred = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) pred[i] = 1 / (1 + Math.exp(-cache.logit[0][i]));
    this.computeLoss(pred, target);

    // dL/dlogit = pred - target (sigmoid + BCE combined)
    const dLogit0 = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) dLogit0[i] = (pred[i] - target[i]) / (H * W);

    // output conv (1->1): grad + d_a1
    const oc = this.outputConv;
    const a1last = cache.finalIn[0];
    for (let i = 0; i < H * W; i++) {
      oc.gW[0] += dLogit0[i] * a1last[i];
      oc.gb[0] += dLogit0[i];
    }
    let dA = [new Float32Array(H * W)]; // d/d a1 (1 channel)
    for (let i = 0; i < H * W; i++) dA[0][i] = dLogit0[i] * oc.W[0];

    // walk blocks in reverse
    for (let bi = this.blocks.length - 1; bi >= 0; bi--) {
      const b = this.blocks[bi];
      const c = cache.blocks[bi];
      // a1 = act1(z1); d_z1 = dA * slope ; accumulate act1 param grads
      const dz1 = new Float32Array(H * W);
      for (let i = 0; i < H * W; i++) {
        dz1[i] = dA[0][i] * b.act1.slope(c.z1[0][i], 0);
        b.act1.accumulateGrad(dA[0][i], c.z1[0][i], 0);
      }
      // conv1 (2m->1): gradW, gradb, d_a0
      const conv1 = b.conv1;
      const dA0 = [];
      for (let ch = 0; ch < conv1.inCh; ch++) dA0.push(new Float32Array(H * W));
      for (let i = 0; i < H * W; i++) {
        conv1.gb[0] += dz1[i];
        for (let ic = 0; ic < conv1.inCh; ic++) {
          conv1.gW[ic] += dz1[i] * c.a0[ic][i];
          dA0[ic][i] = dz1[i] * conv1.W[ic];
        }
      }
      // a0 = act0(z0); d_z0 per channel
      const channels = b.act0.channels;
      const dz0 = [];
      for (let ch = 0; ch < channels; ch++) {
        const arr = new Float32Array(H * W);
        for (let i = 0; i < H * W; i++) {
          arr[i] = dA0[ch][i] * b.act0.slope(c.z0[ch][i], ch);
          b.act0.accumulateGrad(dA0[ch][i], c.z0[ch][i], ch);
        }
        dz0.push(arr);
      }
      // conv3 (1->2m, 3x3 circular): gradW, gradb.
      // Use the per-block conv3 input (c.blockInput[0]) so the backward is
      // correct for any depth, not just depth=1.
      const conv3 = b.conv3;
      const inCh = c.blockInput[0];
      for (let oc2 = 0; oc2 < conv3.outCh; oc2++) {
        for (let i = 0; i < H * W; i++) conv3.gb[oc2] += dz0[oc2][i];
        let wi = oc2 * conv3.inCh * 9;
        for (let ic = 0; ic < conv3.inCh; ic++) {
          for (let kh = 0; kh < 3; kh++) {
            for (let kw = 0; kw < 3; kw++) {
              let g = 0;
              for (let h = 0; h < H; h++) {
                const hh = (h + kh - 1 + H) % H;
                for (let w = 0; w < W; w++) {
                  const ww = (w + kw - 1 + W) % W;
                  g += dz0[oc2][h * W + w] * inCh[hh * W + ww];
                }
              }
              conv3.gW[wi] += g;
              wi++;
            }
          }
        }
      }
      // (no d_input needed: input is data)
      dA = null;
    }
  }

  // Flat list of {array, grad, name} over ALL params (trainable + frozen), for gradcheck.
  paramEntries() {
    const out = [];
    for (let bi = 0; bi < this.blocks.length; bi++) {
      const b = this.blocks[bi];
      out.push({ array: b.conv3.W, grad: b.conv3.gW, name: `b${bi}.conv3.W` });
      out.push({ array: b.conv3.b, grad: b.conv3.gb, name: `b${bi}.conv3.b` });
      for (let i = 0; i < b.act0.params.length; i++) out.push({ array: b.act0.params[i], grad: b.act0.grads[i], name: `b${bi}.act0.p${i}` });
      out.push({ array: b.conv1.W, grad: b.conv1.gW, name: `b${bi}.conv1.W` });
      out.push({ array: b.conv1.b, grad: b.conv1.gb, name: `b${bi}.conv1.b` });
      for (let i = 0; i < b.act1.params.length; i++) out.push({ array: b.act1.params[i], grad: b.act1.grads[i], name: `b${bi}.act1.p${i}` });
    }
    out.push({ array: this.outputConv.W, grad: this.outputConv.gW, name: "out.W" });
    out.push({ array: this.outputConv.b, grad: this.outputConv.gb, name: "out.b" });
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
