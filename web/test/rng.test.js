import { test } from "node:test";
import assert from "node:assert/strict";
import { mulberry32 } from "../js/engine/rng.js";

test("mulberry32 is deterministic for a given seed", () => {
  const a = mulberry32(12345);
  const b = mulberry32(12345);
  const seqA = [a(), a(), a()];
  const seqB = [b(), b(), b()];
  assert.deepEqual(seqA, seqB);
});

test("mulberry32 yields values in [0,1)", () => {
  const r = mulberry32(7);
  for (let i = 0; i < 1000; i++) {
    const v = r();
    assert.ok(v >= 0 && v < 1, `value out of range: ${v}`);
  }
});

test("different seeds give different sequences", () => {
  const a = mulberry32(1);
  const b = mulberry32(2);
  assert.notDeepEqual([a(), a(), a()], [b(), b(), b()]);
});
