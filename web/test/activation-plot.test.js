import { test } from "node:test";
import assert from "node:assert/strict";
import { polyValue, isMonotonicOver } from "../js/widgets/activation-plot.js";

test("polyValue evaluates a polynomial", () => {
  assert.equal(polyValue([1, 2, 3], 2), 1 + 2 * 2 + 3 * 4); // 17
});

test("w1=1 linear is monotonic; a downward parabola is not", () => {
  assert.equal(isMonotonicOver([0, 1, 0], -2, 2), true);   // f(x)=x
  assert.equal(isMonotonicOver([0, 0, 1], -2, 2), false);  // f(x)=x^2 (decreases then increases)
});
