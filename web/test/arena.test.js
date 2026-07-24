import { test } from "node:test";
import assert from "node:assert/strict";
import { makeBatch, gridAccuracy, trainStep } from "../js/arena.js";
import { LifeModel } from "../js/engine/model.js";
import { mulberry32 } from "../js/engine/rng.js";

test("makeBatch returns input + lifeStep target of correct length", () => {
  const rng = mulberry32(1);
  const b = makeBatch(8, 8, 0.4, rng);
  assert.equal(b.input.length, 64);
  assert.equal(b.target.length, 64);
  assert.ok(b.target.every((v) => v === 0 || v === 1));
});

test("gridAccuracy is 1 for identical and <1 for a flipped bit", () => {
  const a = new Float32Array([1, 0, 1, 0]);
  assert.equal(gridAccuracy(a, a), 1);
  const t = new Float32Array([1, 0, 1, 0]);
  const p = new Float32Array([1, 1, 1, 0]); // one wrong
  assert.ok(gridAccuracy(p, t) < 1 && gridAccuracy(p, t) > 0);
});

// Deviation from brief (documented per train.test.js): single-batch-per-step
// plateaus near the majority-class BCE (~0.63) even at 20k steps and never
// halves. polyKAN's convergence cliff on 16x16@0.4 is ~11k outer steps with
// batch=8 (96k updates) — the repo's own proven config in train.test.js.
test("trainStep reduces loss over many steps on fresh data (polyKAN)", () => {
  const rng = mulberry32(7);
  const m = new LifeModel({ width: 1, depth: 1, activation: "polyKAN", seed: 17 });
  m.resize(16, 16);
  const first = trainStep(m, makeBatch(16, 16, 0.4, rng), 1, 1e-3).loss;
  let t = 1;
  for (let s = 0; s < 12000; s++) {
    for (let b = 0; b < 8; b++) trainStep(m, makeBatch(16, 16, 0.4, rng), ++t, 1e-3);
  }
  const last = trainStep(m, makeBatch(16, 16, 0.4, rng), ++t, 1e-3).loss;
  assert.ok(last < first * 0.5, `loss did not drop: ${first} -> ${last}`);
});
