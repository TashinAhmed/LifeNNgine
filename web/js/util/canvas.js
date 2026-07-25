// DPR-aware canvas helpers. DOM access only inside functions (Node-importable).

export function fitCanvas(canvas) {
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  const rect = canvas.getBoundingClientRect();
  const cssW = Math.max(1, Math.floor(rect.width));
  const cssH = Math.max(1, Math.floor(rect.height));
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: canvas.width, h: canvas.height, cssW, cssH, dpr };
}

export function clearCanvas(ctx, w, h, bg = "#e9ecf1") {
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
}

// Draw a flat HxW binary grid. flash: optional Uint8Array same length; cells that
// just changed are drawn with the flash color border.
export function drawGrid(ctx, grid, H, W, cellPx, { on = "#16a34a", off = "#f6f8fa", gridline = "#cdd2da", flash = null, flashColor = "#d97706" } = {}) {
  for (let h = 0; h < H; h++) {
    for (let w = 0; w < W; w++) {
      const i = h * W + w;
      ctx.fillStyle = grid[i] ? on : off;
      ctx.fillRect(w * cellPx, h * cellPx, cellPx, cellPx);
      if (flash && flash[i]) {
        ctx.strokeStyle = flashColor;
        ctx.lineWidth = Math.max(1, cellPx * 0.12);
        ctx.strokeRect(w * cellPx + 0.5, h * cellPx + 0.5, cellPx - 1, cellPx - 1);
      }
    }
  }
  ctx.strokeStyle = gridline;
  ctx.lineWidth = 1;
  for (let h = 0; h <= H; h++) {
    ctx.beginPath(); ctx.moveTo(0, h * cellPx); ctx.lineTo(W * cellPx, h * cellPx); ctx.stroke();
  }
  for (let w = 0; w <= W; w++) {
    ctx.beginPath(); ctx.moveTo(w * cellPx, 0); ctx.lineTo(w * cellPx, H * cellPx); ctx.stroke();
  }
}

// cubic ease in/out for smooth animation transitions
export function easeInOut(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
