import { test } from "node:test";
import assert from "node:assert/strict";
import { PRESETS, stampPreset } from "../js/widgets/life-grid.js";
import { lifeStep } from "../js/engine/life.js";

function blank(H, W) { return new Uint8Array(H * W); }

test("glider stamped in empty torus moves after stepping", () => {
  const H = 10, W = 10;
  let g = stampPreset(blank(H, W), H, W, PRESETS.glider, 4, 4);
  const beforeAlive = g.reduce((s, v) => s + v, 0);
  const next = lifeStep(g, H, W);
  const afterAlive = next.reduce((s, v) => s + v, 0);
  assert.equal(beforeAlive, 5); // glider has 5 cells
  assert.equal(afterAlive, 5);  // glider preserves cell count after a step
});

test("stampPreset wraps toroidally near the edge", () => {
  const H = 5, W = 5;
  const g = stampPreset(blank(H, W), H, W, PRESETS.block, 4, 4); // bottom-right, wraps
  assert.equal(g[4 * 5 + 4] + g[0 * 5 + 0], 2); // (4,4) and wrapped (0,0) both set
});
