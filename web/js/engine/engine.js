// Public engine API for the interactive page (Plan 2).
// Re-exports the pure, DOM-free engine modules.
export { mulberry32 } from "./rng.js";
export { parseRule, neighborCount, lifeStep } from "./life.js";
export { Activation } from "./activations.js";
export { LifeModel } from "./model.js";
export { gradcheckModel } from "./gradcheck.js";
