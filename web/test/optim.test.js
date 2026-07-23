import { test } from "node:test";
import assert from "node:assert/strict";
import { LifeModel } from "../js/engine/model.js";
import { lifeStep } from "../js/engine/life.js";
import { mulberry32 } from "../js/engine/rng.js";

test("Adam reduces loss on a single fixed target over 1000 steps", () => {
  const H = 8, W = 8;
  const m = new LifeModel({ width: 1, depth: 1, activation: "polyKAN", seed: 3 });
  m.resize(H, W);
  const rng = mulberry32(99);
  const input = new Float32Array(H * W);
  for (let i = 0; i < input.length; i++) input[i] = rng() < 0.4 ? 1 : 0;
  const target = lifeStep(input, H, W);
  const loss0 = m.computeLoss(m.forward(input), target);
  for (let t = 1; t <= 1000; t++) {
    m.zeroGrad();
    m.forward(input);
    m.backward(target);
    m.step(1e-3, t);
  }
  const loss1 = m.computeLoss(m.forward(input), target);
  assert.ok(loss1 < loss0 * 0.8, `loss did not drop enough: ${loss0} -> ${loss1}`);
});
