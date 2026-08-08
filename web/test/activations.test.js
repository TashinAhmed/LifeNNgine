import { test } from "node:test";
import assert from "node:assert/strict";
import { Activation } from "../js/engine/activations.js";
import { mulberry32 } from "../js/engine/rng.js";

test("relu is parameterless and correct", () => {
  const rng = mulberry32(42);
  const a = new Activation("relu", 3, { rng });
  assert.equal(a.trainable, false);
  assert.equal(a.params.length, 0);
  assert.equal(a.value(-2, 0), 0);
  assert.equal(a.value(3, 0), 3);
  assert.equal(a.slope(-1, 0), 0);
  assert.equal(a.slope(2, 0), 1);
});

test("square is parameterless and correct", () => {
  const rng = mulberry32(42);
  const a = new Activation("square", 1, { rng });
  assert.equal(a.value(3, 0), 9);
  assert.equal(a.slope(3, 0), 6);
});

test("silu value/slope match finite differences", () => {
  const rng = mulberry32(42);
  const a = new Activation("silu", 1, { rng });
  const x = 0.7;
  const sig = 1 / (1 + Math.exp(-x));
  assert.ok(Math.abs(a.value(x, 0) - x * sig) < 1e-6);
  // numerical slope check
  const eps = 1e-5;
  const num = (a.value(x + eps, 0) - a.value(x - eps, 0)) / (2 * eps);
  assert.ok(Math.abs(a.slope(x, 0) - num) < 1e-4);
});

test("prelu has one param per channel, slope uses a for x<=0", () => {
  const rng = mulberry32(42);
  const a = new Activation("prelu", 2, { rng });
  assert.equal(a.trainable, true);
  assert.equal(a.params[0].length, 2); // 2 channels
  const a0 = a.params[0][0];
  assert.equal(a.value(2, 0), 2);
  assert.equal(a.value(-2, 0), -2 * a0);
  assert.equal(a.slope(-2, 0), a0);
  assert.equal(a.slope(2, 0), 1);
});

test("polyKAN value/slope match coefficients", () => {
  const rng = mulberry32(42);
  const a = new Activation("polyKAN", 1, { degree: 2, rng });
  // params[0] = [w0,w1,w2] per channel
  const [w0, w1, w2] = a.params[0];
  const x = 1.3;
  assert.ok(Math.abs(a.value(x, 0) - (w0 + w1 * x + w2 * x * x)) < 1e-6);
  assert.ok(Math.abs(a.slope(x, 0) - (w1 + 2 * w2 * x)) < 1e-6);
});

test("polyKAN accumulateGrad produces consistent parameter grads", () => {
  const rng = mulberry32(42);
  const a = new Activation("polyKAN", 1, { degree: 2, rng });
  a.zeroGrad();
  const x = 0.9;
  const dOut = 0.5;
  a.accumulateGrad(dOut, x, 0);
  const [gw0, gw1, gw2] = a.grads[0];
  assert.ok(Math.abs(gw0 - dOut) < 1e-6);
  assert.ok(Math.abs(gw1 - dOut * x) < 1e-6);
  assert.ok(Math.abs(gw2 - dOut * x * x) < 1e-6);
});

test("prelu accumulateGrad only accrues on the x<=0 branch", () => {
  const rng = mulberry32(42);
  const a = new Activation("prelu", 1, { rng });
  a.zeroGrad();
  a.accumulateGrad(1.0, 2.0, 0); // x>0 -> da/da_param = 0
  assert.equal(a.grads[0][0], 0);
  a.accumulateGrad(1.0, -2.0, 0); // x<=0 -> da/da_param = x
  assert.equal(a.grads[0][0], -2.0);
});
