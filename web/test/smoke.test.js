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

test("rule-function module imports without DOM access", async () => {
  const m = await import("../js/widgets/rule-function.js");
  assert.ok(Array.isArray(m.RULE_POINTS));
  assert.equal(m.RULE_POINTS.length, 9);
  // target = 1 only at N=2,3; 0 elsewhere (B3/S23 combined envelope)
  for (const { n, target } of m.RULE_POINTS) {
    assert.equal(target, (n === 2 || n === 3) ? 1 : 0);
  }
  assert.equal(typeof m.createRuleFunction, "function");
});

test("activation-plot module imports without DOM access", async () => {
  const m = await import("../js/widgets/activation-plot.js");
  assert.equal(typeof m.polyValue, "function");
  assert.equal(typeof m.isMonotonicOver, "function");
  assert.equal(typeof m.createActivationPlot, "function");
  // six standard activations, each with a pure fn
  assert.ok(Array.isArray(m.ACTIVATIONS) && m.ACTIVATIONS.length === 6);
  const keys = new Set(m.ACTIVATIONS.map((a) => a.key));
  for (const k of ["relu", "prelu", "silu", "sigmoid", "tanh", "square"]) {
    assert.ok(keys.has(k), `missing activation ${k}`);
  }
  // pure helpers behave without any DOM
  assert.equal(m.polyValue([1, 2, 3], 2), 17);
  assert.equal(m.isMonotonicOver([0, 0, 1], -2, 2), false);
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

test("charts module imports without DOM access and linearScale maps endpoints", async () => {
  const m = await import("../js/charts/charts.js");
  assert.equal(typeof m.linearScale, "function");
  assert.equal(typeof m.chartFrame, "function");
  assert.equal(typeof m.renderSuccessBars, "function");
  assert.equal(typeof m.renderDensitySweep, "function");
  // domain endpoints map to range endpoints
  const sx = m.linearScale([0, 1], [0, 100]);
  assert.equal(sx(0), 0);
  assert.equal(sx(1), 100);
  assert.equal(sx(0.5), 50);
  // inverted range (screen-y) maps correctly
  const sy = m.linearScale([0, 1], [100, 0]);
  assert.equal(sy(0), 100);
  assert.equal(sy(1), 0);
  // degenerate domain collapses to the range midpoint (no NaN)
  const sc = m.linearScale([3, 3], [10, 20]);
  assert.equal(sc(3), 15);
  assert.equal(sc(99), 15);
});
