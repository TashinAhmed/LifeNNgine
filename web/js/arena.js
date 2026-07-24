import { lifeStep } from "./engine/life.js";
import { LifeModel } from "./engine/model.js";
import { mulberry32 } from "./engine/rng.js";
import { fitCanvas, clearCanvas, drawGrid } from "./util/canvas.js";

// --- pure training helpers (DOM-free, Node-tested) ---

export function makeBatch(H, W, density, rng) {
  const input = new Float32Array(H * W);
  for (let i = 0; i < input.length; i++) input[i] = rng() < density ? 1 : 0;
  return { input, target: lifeStep(input, H, W) };
}

export function gridAccuracy(pred, target) {
  let correct = 0;
  for (let i = 0; i < pred.length; i++) {
    if ((pred[i] > 0.5 ? 1 : 0) === (target[i] > 0.5 ? 1 : 0)) correct++;
  }
  return correct / pred.length;
}

export function trainStep(model, batch, t, lr = 1e-3) {
  model.zeroGrad();
  const pred = model.forward(batch.input);
  model.backward(batch.target);
  model.step(lr, t);
  return { loss: model.computeLoss(pred, batch.target), pred };
}

// placeholder controller (replaced in Task 2)
export function createArena() {}
