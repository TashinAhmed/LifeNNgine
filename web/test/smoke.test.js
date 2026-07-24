import { test } from "node:test";
import assert from "node:assert/strict";

test("canvas util imports without DOM access", async () => {
  const m = await import("../js/util/canvas.js");
  assert.equal(typeof m.fitCanvas, "function");
  assert.equal(typeof m.drawGrid, "function");
  assert.equal(typeof m.easeInOut, "function");
});
