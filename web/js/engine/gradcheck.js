// Finite-difference gradient check: compares analytic backward() grads to
// central differences of the BCE loss. Pure, DOM-free.
import { mulberry32 } from "./rng.js";

export function gradcheckModel(model, input, target, eps = 1e-3) {
  const { H, W } = model;
  // analytic grads
  model.zeroGrad();
  model.forward(input);
  model.backward(target);
  const entries = model.paramEntries();
  const analytic = entries.map((e) => Array.from(e.grad));

  let maxAbsErr = 0;
  const perParam = [];
  for (let pi = 0; pi < entries.length; pi++) {
    const { array } = entries[pi];
    for (let i = 0; i < array.length; i++) {
      const orig = array[i];
      array[i] = orig + eps;
      const lp = lossOf(model, input, target);
      array[i] = orig - eps;
      const lm = lossOf(model, input, target);
      array[i] = orig;
      const num = (lp - lm) / (2 * eps);
      const ana = analytic[pi][i];
      const err = Math.abs(num - ana);
      if (err > maxAbsErr) maxAbsErr = err;
      perParam.push(err);
    }
  }
  return { maxAbsErr, perParam };
}

function lossOf(model, input, target) {
  const pred = model.forward(input);
  return model.computeLoss(pred, target);
}
