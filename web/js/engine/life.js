// Conway-style Life-like cellular automata, toroidal (circular) boundaries.
// Pure, DOM-free. Used both for training targets and the interactive grid widget.

export function parseRule(rule) {
  const parts = rule.split("/");
  let bStr = "", sStr = "";
  for (const p of parts) {
    if (p[0] === "B" || p[0] === "b") bStr = p.slice(1);
    else if (p[0] === "S" || p[0] === "s") sStr = p.slice(1);
  }
  const toSet = (s) => new Set(s.split("").map((c) => Number(c)));
  return { birth: toSet(bStr), survive: toSet(sStr) };
}

// Moore-neighborhood live-cell count with circular wrap. Excludes the center cell.
export function neighborCount(grid, H, W) {
  const out = new Uint8Array(H * W);
  for (let h = 0; h < H; h++) {
    for (let w = 0; w < W; w++) {
      let n = 0;
      for (let dh = -1; dh <= 1; dh++) {
        for (let dw = -1; dw <= 1; dw++) {
          if (dh === 0 && dw === 0) continue;
          const hh = (h + dh + H) % H;
          const ww = (w + dw + W) % W;
          n += grid[hh * W + ww] ? 1 : 0;
        }
      }
      out[h * W + w] = n;
    }
  }
  return out;
}

// Advance one generation under the given Life-like rule. Returns Float32Array of 0/1.
export function lifeStep(grid, H, W, rule = "B3/S23") {
  const { birth, survive } = parseRule(rule);
  const counts = neighborCount(grid, H, W);
  const out = new Float32Array(H * W);
  for (let i = 0; i < grid.length; i++) {
    const c = counts[i];
    out[i] = grid[i] ? (survive.has(c) ? 1 : 0) : (birth.has(c) ? 1 : 0);
  }
  return out;
}
