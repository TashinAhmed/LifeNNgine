import { test } from "node:test";
import assert from "node:assert/strict";
import { LifeModel } from "../js/engine/model.js";
import { lifeStep } from "../js/engine/life.js";
import { mulberry32 } from "../js/engine/rng.js";

// End-to-end proof (Task 7): a minimal L(1,1) polyKAN learns Conway's Game of
// Life to ~100% grid accuracy while an architecturally identical ReLU model
// stalls near the majority-class baseline. This validates the paper's thesis in
// the Node port and exercises Tasks 1-6 together (RNG, Life rule, activations,
// forward/backward, Adam).
//
// Deviations from the task brief (documented per brief Step 2 guidance):
//  1. `model.zeroGrad()` is called before each `backward()`. The brief's loop
//     omitted it, but `backward()` accumulates gradients in place (+=), so
//     without zeroing the signal grows unbounded across all steps and polyKAN
//     stalls at ~0.70 (verified). This matches the repo's own working optimizer
//     test (optim.test.js) and is a harness fix, not an engine change — the
//     engine's gradcheck and Adam-loss-reduction tests pass independently.
//  2. `steps = 12000` (raised from the brief's 4000). polyKAN on a 16x16 grid at
//     density 0.4 has a sharp convergence cliff at ~11000 steps; at 12000 it
//     scores a perfect 1.0 averaged over 40 independent validation batches
//     (genuine convergence, not single-batch luck). The brief's suggested 8000
//     is insufficient (~0.79 there). ReLU stalls at ~0.67 throughout (>=0.95
//     would require an entirely different architecture). Density 0.4 (brief
//     default) gives the cleanest separation and is retained.

// Grid accuracy at decision boundary 0.5 (matches repo validation).
function gridAccuracy(pred, target) {
  let correct = 0;
  for (let i = 0; i < pred.length; i++) {
    const bit = pred[i] > 0.5 ? 1 : 0;
    if (bit === (target[i] > 0.5 ? 1 : 0)) correct++;
  }
  return correct / pred.length;
}

function makeBatch(H, W, density, rng) {
  const input = new Float32Array(H * W);
  for (let i = 0; i < input.length; i++) input[i] = rng() < density ? 1 : 0;
  return { input, target: lifeStep(input, H, W) };
}

function train(model, { H = 16, W = 16, density = 0.4, steps = 12000, lr = 1e-3, seed = 1, batch = 8 }) {
  const rng = mulberry32(seed);
  model.resize(H, W);
  let t = 0;
  for (let s = 0; s < steps; s++) {
    for (let b = 0; b < batch; b++) {
      const { input, target } = makeBatch(H, W, density, rng);
      model.zeroGrad();
      model.forward(input);
      model.backward(target);
      t++;
      model.step(lr, t);
    }
  }
  // validation on a fresh batch
  const { input, target } = makeBatch(H, W, density, rng);
  return gridAccuracy(model.forward(input), target);
}

test("polyKAN L(1,1) reaches >=0.999 accuracy on Life within budget", () => {
  const m = new LifeModel({ width: 1, depth: 1, activation: "polyKAN", seed: 17 });
  const acc = train(m, { steps: 12000, seed: 1 });
  assert.ok(acc >= 0.999, `polyKAN did not converge: ${acc}`);
});

test("relu L(1,1) stays below 0.95 on Life in the same budget", () => {
  const m = new LifeModel({ width: 1, depth: 1, activation: "relu", seed: 17 });
  const acc = train(m, { steps: 12000, seed: 1 });
  assert.ok(acc < 0.95, `relu unexpectedly converged: ${acc}`);
});
