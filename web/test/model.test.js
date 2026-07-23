import { test } from "node:test";
import assert from "node:assert/strict";
import { LifeModel } from "../js/engine/model.js";

function makeModel(opts) {
  const m = new LifeModel(opts);
  m.resize(8, 8);
  return m;
}

test("L(1,1) polyKAN has 34 trainable params (full)", () => {
  const m = makeModel({ width: 1, depth: 1, activation: "polyKAN", seed: 1 });
  assert.equal(m.trainableCount(), 34);
});

test("L(1,1) polyKAN activations-only has 29 trainable params", () => {
  const m = makeModel({ width: 1, depth: 1, activation: "polyKAN", seed: 1, trainableWeights: false });
  assert.equal(m.trainableCount(), 29);
});

test("L(1,1) polyKAN weights-only has 25 trainable params", () => {
  const m = makeModel({ width: 1, depth: 1, activation: "polyKAN", seed: 1, trainableActivations: false });
  assert.equal(m.trainableCount(), 25);
});

test("L(1,1) relu has 25 trainable params", () => {
  const m = makeModel({ width: 1, depth: 1, activation: "relu", seed: 1 });
  assert.equal(m.trainableCount(), 25);
});

test("forward returns probabilities in (0,1) of correct length", () => {
  const m = makeModel({ width: 1, depth: 1, activation: "polyKAN", seed: 2 });
  const input = new Float32Array(64);
  for (let i = 0; i < 64; i++) input[i] = Math.random() < 0.5 ? 1 : 0;
  const out = m.forward(input);
  assert.equal(out.length, 64);
  for (const v of out) assert.ok(v > 0 && v < 1, `output not in (0,1): ${v}`);
});

test("forward is deterministic given fixed seed", () => {
  const m1 = makeModel({ width: 1, depth: 1, activation: "polyKAN", seed: 9 });
  const m2 = makeModel({ width: 1, depth: 1, activation: "polyKAN", seed: 9 });
  const input = new Float32Array(64).fill(0);
  input[0] = 1;
  assert.deepEqual(Array.from(m1.forward(input)), Array.from(m2.forward(input)));
});
