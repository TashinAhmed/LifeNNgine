import { test } from "node:test";
import assert from "node:assert/strict";

test("canvas util imports without DOM access", async () => {
  const m = await import("../js/util/canvas.js");
  assert.equal(typeof m.fitCanvas, "function");
  assert.equal(typeof m.drawGrid, "function");
  assert.equal(typeof m.easeInOut, "function");
});

test("life-grid module imports without DOM access", async () => {
  const m = await import("../js/widgets/life-grid.js");
  assert.ok(m.PRESETS && m.PRESETS.glider && m.PRESETS.pulsar);
  assert.equal(typeof m.stampPreset, "function");
  assert.equal(typeof m.createLifeGrid, "function");
});

test("results data module exports the four datasets", async () => {
  const m = await import("../js/data/results.js");
  assert.ok(Array.isArray(m.SUCCESS_RATES) && m.SUCCESS_RATES.length === 10);
  assert.ok(m.ABLATION && m.ABLATION.polyKAN && m.ABLATION.prelu);
  assert.equal(m.DENSITY_SWEEP.approx, true);
  assert.ok(Array.isArray(m.DENSITY_SWEEP.points));
  assert.equal(m.PCA_ILLUSTRATIVE.approx, true);
  assert.ok(m.PCA_ILLUSTRATIVE.polyKAN && m.PCA_ILLUSTRATIVE.relu);
  assert.ok(m.PCA_ILLUSTRATIVE.prelu && m.PCA_ILLUSTRATIVE.sigmoid);
});
