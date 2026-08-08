import { test } from "node:test";
import assert from "node:assert/strict";
import { parseRule, neighborCount, lifeStep } from "../js/engine/life.js";

function grid(H, W, ones) {
  const g = new Uint8Array(H * W);
  for (const [h, w] of ones) g[h * W + w] = 1;
  return g;
}

test("parseRule parses B3/S23", () => {
  const r = parseRule("B3/S23");
  assert.deepEqual([...r.birth].sort(), [3]);
  assert.deepEqual([...r.survive].sort(), [2, 3]);
});

test("neighborCount wraps toroidally and excludes self", () => {
  // single live cell in 3x3 torus: every cell's Moore neighborhood (with wrap)
  // contains the lone live cell, except the cell itself (excluded).
  const g = grid(3, 3, [[1, 1]]);
  const n = neighborCount(g, 3, 3);
  assert.equal(n[1 * 3 + 1], 0); // center excludes itself -> 0 live neighbors
  assert.equal(n[0 * 3 + 0], 1); // corner sees the lone live cell -> 1
});

test("blinker oscillates (period 2)", () => {
  // vertical blinker -> horizontal after one step
  let g = grid(5, 5, [[1, 2], [2, 2], [3, 2]]);
  let next = lifeStep(g, 5, 5);
  // horizontal: (2,1),(2,2),(2,3)
  const live = [];
  for (let i = 0; i < 25; i++) if (next[i]) live.push([Math.floor(i / 5), i % 5]);
  assert.deepEqual(live.sort((a, b) => a[0] - b[0] || a[1] - b[1]), [[2, 1], [2, 2], [2, 3]]);
  // back to vertical after second step
  let next2 = lifeStep(next, 5, 5);
  assert.deepEqual(Array.from(next2), Array.from(g));
});

test("block is a still life", () => {
  const g = grid(5, 5, [[1, 1], [1, 2], [2, 1], [2, 2]]);
  const next = lifeStep(g, 5, 5);
  assert.deepEqual(Array.from(next), Array.from(g));
});
