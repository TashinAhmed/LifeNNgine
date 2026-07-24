import { lifeStep } from "./engine/life.js";
import { mulberry32 } from "./engine/rng.js";
import { fitCanvas, clearCanvas, drawGrid } from "./util/canvas.js";
import { createLifeGrid } from "./widgets/life-grid.js";
import { createRuleFunction } from "./widgets/rule-function.js";
import { createActivationPlot } from "./widgets/activation-plot.js";
import { renderSuccessBars, renderDensitySweep } from "./charts/charts.js";
import { SUCCESS_RATES, DENSITY_SWEEP } from "./data/results.js";

const prefersReducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

export function initHero(canvas) {
  const H = 48, W = 96;
  let grid = new Uint8Array(H * W);
  const rng = mulberry32(2026);
  for (let i = 0; i < grid.length; i++) grid[i] = rng() < 0.35 ? 1 : 0;
  let raf = 0, last = 0;
  function frame(t) {
    if (t - last > 220) { grid = Uint8Array.from(lifeStep(grid, H, W)); last = t; }
    const { ctx, cssW, cssH } = fitCanvas(canvas);
    clearCanvas(ctx, canvas.width, canvas.height, "rgba(11,13,16,0)");
    drawGrid(ctx, grid, H, W, Math.min(cssW / W, cssH / H), { on: "rgba(57,255,20,0.10)", off: "transparent", gridline: "transparent" });
    raf = requestAnimationFrame(frame);
  }
  if (!prefersReducedMotion()) raf = requestAnimationFrame(frame);
  return () => cancelAnimationFrame(raf);
}

document.addEventListener("DOMContentLoaded", () => {
  const hero = document.getElementById("hero-bg");
  if (hero) window.__stopHero = initHero(hero);

  const lifeCanvas = document.getElementById("life-grid");
  if (lifeCanvas) {
    window.__lifeGrid = createLifeGrid(lifeCanvas, document.getElementById("life-controls"));
  }

  const ruleCanvas = document.getElementById("rule-function");
  if (ruleCanvas) {
    window.__ruleFunction = createRuleFunction(ruleCanvas);
  }

  const zooCanvas = document.getElementById("activation-zoo");
  const polyCanvas = document.getElementById("activation-poly");
  const polyControls = document.getElementById("poly-controls");
  if (zooCanvas || polyCanvas) {
    window.__activationPlot = createActivationPlot(zooCanvas, polyCanvas, polyControls);
  }

  const vizSuccess = document.getElementById("viz-success");
  if (vizSuccess) {
    window.__vizSuccess = renderSuccessBars(vizSuccess, SUCCESS_RATES);
  }

  const vizDensity = document.getElementById("viz-density");
  if (vizDensity) {
    window.__vizDensity = renderDensitySweep(vizDensity, DENSITY_SWEEP);
  }
});
