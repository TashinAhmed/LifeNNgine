// Per-channel activation functions for the L(m,n) model.
// Each instance holds its own parameters (params) and gradients (grads).
// value(x,c) and slope(x,c) are scalar; accumulateGrad adds param grads.

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class Activation {
  constructor(kind, channels, { degree = 2, rng = Math.random } = {}) {
    this.kind = kind;
    this.channels = channels;
    this.degree = degree;
    this.rng = rng;
    this.params = []; // array of Float32Array (one entry per "param group")
    this.grads = [];
    this.trainable = false;
    this._init();
  }

  _init() {
    switch (this.kind) {
      case "relu":
      case "silu":
      case "square":
        this.trainable = false;
        break;
      case "prelu": {
        this.trainable = true;
        const p = new Float32Array(this.channels);
        for (let c = 0; c < this.channels; c++) p[c] = 0.25 + 0.0; // matches repo default a=0.25
        this.params.push(p);
        break;
      }
      case "polyKAN": {
        this.trainable = true;
        if (this.degree !== 2) throw new RangeError("polyKAN activation currently supports only degree=2");
        // coeffs per channel: [w0, w1, ..., wD], shape (channels, degree+1)
        const p = new Float32Array(this.channels * (this.degree + 1));
        for (let c = 0; c < this.channels; c++) {
          for (let d = 0; d <= this.degree; d++) {
            const base = d === 1 ? 1.0 : 0.0; // init linear, others ~0
            p[c * (this.degree + 1) + d] = base + (this.rng() - 0.5) * 0.02;
          }
        }
        this.params.push(p);
        break;
      }
      default:
        throw new Error(`Unknown activation: ${this.kind}`);
    }
    this.zeroGrad();
  }

  zeroGrad() {
    this.grads = this.params.map((p) => new Float32Array(p.length));
  }

  // polyKAN coefficient accessors
  _poly(c) {
    const D = this.degree;
    const off = c * (D + 1);
    const p = this.params[0];
    return [p[off], p[off + 1], p[off + 2]]; // supports degree 2 (w0,w1,w2)
  }

  value(x, c) {
    switch (this.kind) {
      case "relu":
        return x > 0 ? x : 0;
      case "square":
        return x * x;
      case "silu":
        return x * sigmoid(x);
      case "prelu": {
        const a = this.params[0][c];
        return x > 0 ? x : a * x;
      }
      case "polyKAN": {
        const [w0, w1, w2] = this._poly(c);
        return w0 + w1 * x + w2 * x * x;
      }
    }
  }

  slope(x, c) {
    switch (this.kind) {
      case "relu":
        return x > 0 ? 1 : 0;
      case "square":
        return 2 * x;
      case "silu": {
        const s = sigmoid(x);
        return s * (1 + x * (1 - s));
      }
      case "prelu":
        return x > 0 ? 1 : this.params[0][c];
      case "polyKAN": {
        const [w0, w1, w2] = this._poly(c);
        return w1 + 2 * w2 * x;
      }
    }
  }

  // dOut = upstream grad w.r.t. the activation output (scalar); x = pre-activation.
  accumulateGrad(dOut, x, c) {
    if (!this.trainable) return;
    switch (this.kind) {
      case "prelu": {
        // value = x>0 ? x : a*x ; d/da = x>0 ? 0 : x
        if (x <= 0) this.grads[0][c] += dOut * x;
        break;
      }
      case "polyKAN": {
        const D = this.degree;
        const off = c * (D + 1);
        const g = this.grads[0];
        g[off] += dOut;            // w0
        g[off + 1] += dOut * x;    // w1
        if (D >= 2) g[off + 2] += dOut * x * x; // w2
        break;
      }
    }
  }
}
