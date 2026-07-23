import { test } from "node:test";
import assert from "node:assert/strict";
import { LifeModel } from "../js/engine/model.js";
import { gradcheckModel } from "../js/engine/gradcheck.js";
import { mulberry32 } from "../js/engine/rng.js";

function randGrid(n, rng) {
  const g = new Float32Array(n);
  for (let i = 0; i < n; i++) g[i] = rng() < 0.5 ? 1 : 0;
  return g;
}

test("analytic gradients match finite differences for polyKAN L(1,1)", () => {
  const rng = mulberry32(5);
  const H = 6, W = 6;
  const m = new LifeModel({ width: 1, depth: 1, activation: "polyKAN", seed: 5 });
  m.resize(H, W);
  const input = randGrid(H * W, rng);
  const target = randGrid(H * W, rng);
  const res = gradcheckModel(m, input, target, 1e-3);
  // allow tiny float noise; every param should match to within 1e-2 absolute.
  assert.ok(res.maxAbsErr < 1e-2, `maxAbsErr too high: ${res.maxAbsErr}`);
});

test("analytic gradients match finite differences for prelu L(1,1)", () => {
  const rng = mulberry32(6);
  const H = 6, W = 6;
  const m = new LifeModel({ width: 1, depth: 1, activation: "prelu", seed: 6 });
  m.resize(H, W);
  const input = randGrid(H * W, rng);
  const target = randGrid(H * W, rng);
  const res = gradcheckModel(m, input, target, 1e-3);
  assert.ok(res.maxAbsErr < 1e-2, `maxAbsErr too high: ${res.maxAbsErr}`);
});

test("analytic gradients match finite differences for relu L(1,1)", () => {
  const rng = mulberry32(8);
  const H = 6, W = 6;
  const m = new LifeModel({ width: 1, depth: 1, activation: "relu", seed: 8 });
  m.resize(H, W);
  const input = randGrid(H * W, rng);
  const target = randGrid(H * W, rng);
  const res = gradcheckModel(m, input, target, 1e-3);
  assert.ok(res.maxAbsErr < 5e-2, `maxAbsErr too high (relu kink): ${res.maxAbsErr}`);
});

test("analytic gradients match finite differences for silu L(1,1)", () => {
  const rng = mulberry32(11);
  const H = 6, W = 6;
  const m = new LifeModel({ width: 1, depth: 1, activation: "silu", seed: 11 });
  m.resize(H, W);
  const input = randGrid(H * W, rng);
  const target = randGrid(H * W, rng);
  const res = gradcheckModel(m, input, target, 1e-3);
  assert.ok(res.maxAbsErr < 1e-2, `maxAbsErr too high: ${res.maxAbsErr}`);
});
