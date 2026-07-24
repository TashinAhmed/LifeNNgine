import { test } from "node:test";
import assert from "node:assert/strict";
import {
  SUCCESS_RATES,
  ABLATION,
  DENSITY_SWEEP,
  PCA_ILLUSTRATIVE,
} from "../js/data/results.js";

test("success rates match Table 2", () => {
  const byName = Object.fromEntries(SUCCESS_RATES.map((r) => [r.name, r]));
  assert.equal(byName.PolyKAN.rate, 1.0);
  assert.equal(byName.PolyKAN.params, 34);
  assert.equal(byName.PReLU.rate, 0.97);
  assert.equal(byName.PReLU.params, 28);
  assert.equal(byName.Square.rate, 0.94);
  assert.equal(byName.Square.params, 25);
  assert.equal(byName.SiLU.rate, 0.94);
  assert.equal(byName.RootSquare.rate, 0.50);
  assert.equal(byName.LeakyReLU.rate, 0.25);
  assert.equal(byName.CELU.rate, 0.06);
  assert.equal(byName.Sigmoid.rate, 0.0);
  assert.equal(byName.Tanh.rate, 0.0);
  assert.equal(byName.ReLU.rate, 0.0);
  assert.equal(byName.ReLU.params, 25);
});

test("every success-rate entry has well-formed fields", () => {
  assert.equal(SUCCESS_RATES.length, 10);
  for (const r of SUCCESS_RATES) {
    assert.ok(typeof r.name === "string" && r.name.length > 0);
    assert.ok(typeof r.rate === "number" && r.rate >= 0 && r.rate <= 1);
    assert.ok(Number.isInteger(r.params) && r.params > 0);
    assert.equal(typeof r.monotonic, "boolean");
    assert.equal(typeof r.differentiable, "boolean");
  }
});

test("ablation matches Table 2 / Fig 6", () => {
  assert.equal(ABLATION.polyKAN.full.rate, 1.0);
  assert.equal(ABLATION.polyKAN.full.params, 34);
  assert.equal(ABLATION.polyKAN.actOnly.rate, 1.0);
  assert.equal(ABLATION.polyKAN.actOnly.params, 29);
  assert.equal(ABLATION.polyKAN.weightOnly.rate, 0.78);
  assert.equal(ABLATION.polyKAN.weightOnly.params, 25);

  assert.equal(ABLATION.prelu.full.rate, 0.97);
  assert.equal(ABLATION.prelu.full.params, 28);
  assert.equal(ABLATION.prelu.actOnly.rate, 1.0);
  assert.equal(ABLATION.prelu.actOnly.params, 23);
  assert.equal(ABLATION.prelu.weightOnly.rate, 0.59);
  assert.equal(ABLATION.prelu.weightOnly.params, 25);
});

test("density sweep is marked approximate and spans the range", () => {
  assert.equal(DENSITY_SWEEP.approx, true);
  const ds = DENSITY_SWEEP.points;
  assert.ok(Array.isArray(ds));
  assert.ok(ds.length >= 19);
  assert.equal(ds[0].density, 0.05);
  assert.ok(ds[0].density <= 0.06);
  assert.ok(ds[ds.length - 1].density >= 0.94);
  assert.ok(
    ds.every((d) =>
      ["polyKAN", "prelu", "silu", "relu"].every((k) => typeof d[k] === "number")
    )
  );
});

test("density sweep values are clamped to [0,1] and PolyKAN anchors hold", () => {
  const byDensity = Object.fromEntries(
    DENSITY_SWEEP.points.map((p) => [p.density, p])
  );
  for (const p of DENSITY_SWEEP.points) {
    for (const k of ["polyKAN", "prelu", "silu", "relu"]) {
      assert.ok(p[k] >= 0 && p[k] <= 1, `${k}@${p.density} out of range`);
    }
  }
  assert.equal(byDensity[0.9].polyKAN, 0.75);
  assert.equal(byDensity[0.95].polyKAN, 0);
  assert.equal(byDensity[0.3].polyKAN, 0.813);
  assert.equal(byDensity[0.9].prelu, 0.063);
});

test("PCA is illustrative and has the four activations", () => {
  assert.equal(PCA_ILLUSTRATIVE.approx, true);
  for (const k of ["polyKAN", "relu", "prelu", "sigmoid"]) {
    assert.ok(PCA_ILLUSTRATIVE[k].length >= 12, `${k} needs >=12 points`);
    for (const p of PCA_ILLUSTRATIVE[k]) {
      assert.ok("pc1" in p && "pc2" in p && "loss" in p && "success" in p);
      assert.ok(typeof p.loss === "number" && p.loss >= 0 && p.loss <= 1);
      assert.equal(typeof p.success, "boolean");
    }
  }
});

test("PCA qualitative shapes: PolyKAN mostly success, ReLU has failures", () => {
  const pkOk = PCA_ILLUSTRATIVE.polyKAN.filter((p) => p.success).length;
  assert.ok(pkOk >= PCA_ILLUSTRATIVE.polyKAN.length * 0.8, "PolyKAN should fan out to success frontier");
  const reluFail = PCA_ILLUSTRATIVE.relu.filter((p) => !p.success).length;
  assert.ok(reluFail >= 4, "ReLU should have many failure markers");
});
