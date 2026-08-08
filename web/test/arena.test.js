import { test } from "node:test";
import assert from "node:assert/strict";
import {
  makeBatch,
  gridAccuracy,
  trainStep,
  ARENA_ACTIVATIONS,
  activationLabel,
  canonicalActivation,
  lrFromSlider,
  lrToSlider,
  LR_MIN,
  LR_MAX,
  LR_STEPS,
  clampDensity,
  clampLr,
  clampSpeed,
} from "../js/arena.js";
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
// causes the BCE *loss* to plateau near the majority-class baseline (~0.63)
// even at 20k steps, so it never halves here. Note: this is a loss-only
// plateau — *accuracy* still converges to ~100% given enough updates, as the
// live arena demonstrates (polyKAN on 32x32@0.4, batch=1). polyKAN's
// convergence cliff on 16x16@0.4 is ~11k outer steps with batch=8 (96k
// updates) — the repo's own proven config in train.test.js.
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

// --- control helpers (pure, DOM-free) ---

test("ARENA_ACTIVATIONS lists the five supported activations with PolyKAN default", () => {
  assert.equal(ARENA_ACTIVATIONS.length, 5);
  assert.equal(ARENA_ACTIVATIONS[0].value, "polyKAN");
  assert.equal(ARENA_ACTIVATIONS[0].label, "PolyKAN");
  const keys = new Set(ARENA_ACTIVATIONS.map((a) => a.value));
  // Must match the kinds accepted by engine/activations.js.
  for (const k of ["polyKAN", "prelu", "silu", "square", "relu"]) {
    assert.ok(keys.has(k), `arena selector missing activation ${k}`);
  }
});

test("activationLabel maps each kind to its display name", () => {
  assert.equal(activationLabel("polyKAN"), "PolyKAN");
  assert.equal(activationLabel("POLYKAN"), "PolyKAN"); // case-insensitive
  assert.equal(activationLabel("prelu"), "PReLU");
  assert.equal(activationLabel("silu"), "SiLU");
  assert.equal(activationLabel("square"), "Square");
  assert.equal(activationLabel("relu"), "ReLU");
  assert.equal(activationLabel("unknown"), "unknown"); // pass-through
});

test("canonicalActivation normalizes to engine casing (activations.js is case-sensitive)", () => {
  assert.equal(canonicalActivation("polykan"), "polyKAN");
  assert.equal(canonicalActivation("POLYKAN"), "polyKAN");
  assert.equal(canonicalActivation("PReLU"), "prelu");
  assert.equal(canonicalActivation("RELU"), "relu");
  // unknown kinds pass through unchanged
  assert.equal(canonicalActivation("bogus"), "bogus");
});

test("lrFromSlider/lrToSlider hit the log-scale endpoints and round-trip", () => {
  assert.equal(lrFromSlider(0), LR_MIN);
  assert.ok(Math.abs(lrFromSlider(LR_STEPS) - LR_MAX) < 1e-12);
  // midpoint is geometric mean
  const mid = lrFromSlider(LR_STEPS / 2);
  assert.ok(Math.abs(mid - Math.sqrt(LR_MIN * LR_MAX)) < 1e-12);
  for (const lr of [LR_MIN, 3e-4, 1e-3, 5e-3, LR_MAX]) {
    const back = lrFromSlider(lrToSlider(lr));
    assert.ok(Math.abs(back - lr) / lr < 0.01, `lr round-trip drifted: ${lr} -> ${back}`);
  }
});

test("clamp helpers saturate at the documented bounds", () => {
  assert.equal(clampDensity(0.4), 0.4);
  assert.equal(clampDensity(0), 0.05);
  assert.equal(clampDensity(1), 0.95);
  assert.equal(clampDensity(NaN), 0.4);
  assert.equal(clampLr(1e-3), 1e-3);
  assert.equal(clampLr(1e-10), LR_MIN);
  assert.equal(clampLr(1), LR_MAX);
  assert.equal(clampLr(NaN), 1e-3);
  assert.equal(clampSpeed(150), 150);
  assert.equal(clampSpeed(0), 1);
  assert.equal(clampSpeed(999), 500);
  assert.equal(clampSpeed(NaN), 300);
});
