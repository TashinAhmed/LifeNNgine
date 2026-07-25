import { lifeStep } from "./engine/life.js";
import { mulberry32 } from "./engine/rng.js";
import { fitCanvas, clearCanvas, drawGrid } from "./util/canvas.js";
import { createLifeGrid } from "./widgets/life-grid.js";
import { createRuleFunction } from "./widgets/rule-function.js";
import { createActivationPlot } from "./widgets/activation-plot.js";
import { createArena } from "./arena.js";
import { renderSuccessBars, renderDensitySweep, renderAblation, renderPCA } from "./charts/charts.js";
import { SUCCESS_RATES, DENSITY_SWEEP, ABLATION, PCA_ILLUSTRATIVE } from "./data/results.js";

// Single reduced-motion policy for the whole page (spec §10). Read once and
// passed to every controller that cares, so there is one source of truth.
const REDUCED_MOTION = (() => {
  try {
    return typeof window !== "undefined" && typeof window.matchMedia === "function"
      && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
})();

// Hero background: a faint, slowly self-evolving Game of Life grid motif.
// Returns { setPaused(bool), stop() }. Respects REDUCED_MOTION (no auto-start),
// so a visibility pause/resume from the IntersectionObserver is always safe.
export function initHero(canvas) {
  const H = 48, W = 96;
  let grid = new Uint8Array(H * W);
  const rng = mulberry32(2026);
  for (let i = 0; i < grid.length; i++) grid[i] = rng() < 0.35 ? 1 : 0;
  let raf = 0, last = 0;
  let paused = false;

  function frame(t) {
    if (t - last > 220) { grid = Uint8Array.from(lifeStep(grid, H, W)); last = t; }
    const view = fitCanvas(canvas);
    if (view && view.ctx) {
      const { ctx, cssW, cssH } = view;
      clearCanvas(ctx, canvas.width, canvas.height, "rgba(11,13,16,0)");
      drawGrid(ctx, grid, H, W, Math.min(cssW / W, cssH / H), { on: "rgba(57,255,20,0.10)", off: "transparent", gridline: "transparent" });
    }
    raf = requestAnimationFrame(frame);
  }

  function stop() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
  }
  function start() {
    if (raf || paused || REDUCED_MOTION) return;
    raf = requestAnimationFrame(frame);
  }

  if (!REDUCED_MOTION) start();

  return {
    setPaused(p) {
      paused = !!p;
      if (paused) stop();
      else start();
    },
    stop,
  };
}

document.addEventListener("DOMContentLoaded", () => {
  // Back-to-top: reveal after roughly one viewport of scrolling. The anchor
  // itself (#ch-hero) does the jump; CSS scroll-behavior (auto under reduced
  // motion) governs smoothness, so no JS scroll logic is needed here.
  const backToTop = document.querySelector(".back-to-top");
  if (backToTop) {
    const reveal = () => {
      backToTop.classList.toggle("is-visible", window.scrollY > window.innerHeight * 0.8);
    };
    window.addEventListener("scroll", reveal, { passive: true });
    reveal();
  }

  // Collect every widget handle paired with its on-screen root (the .canvas-card
  // or .chapter section the IntersectionObserver watches). Charts return
  // { redraw } without setPaused; the observer callback guards for that, so
  // registering them is harmless and keeps the collection uniform.
  const widgets = [];
  function register(canvas, handle) {
    if (!canvas || !handle) return;
    const root = canvas.closest(".canvas-card") || canvas.closest(".chapter") || canvas;
    widgets.push({ root, handle });
  }

  const heroCanvas = document.getElementById("hero-bg");
  if (heroCanvas) register(heroCanvas, initHero(heroCanvas));

  const lifeCanvas = document.getElementById("life-grid");
  if (lifeCanvas) register(lifeCanvas, createLifeGrid(lifeCanvas, document.getElementById("life-controls")));

  const ruleCanvas = document.getElementById("rule-function");
  if (ruleCanvas) register(ruleCanvas, createRuleFunction(ruleCanvas));

  const zooCanvas = document.getElementById("activation-zoo");
  const polyCanvas = document.getElementById("activation-poly");
  if (zooCanvas || polyCanvas) {
    // The zoo + poly pair share one handle; observe the card that contains both.
    register(zooCanvas || polyCanvas, createActivationPlot(zooCanvas, polyCanvas, document.getElementById("poly-controls")));
  }

  // Chapter 4 — live training arena. The controller owns the whole #arena-mount
  // subtree (multiple canvases), so we register the mount element itself with
  // the arena handle: register() only uses the element to locate its observed
  // root via closest(), and the arena handle exposes setPaused for IO pause.
  const arenaMount = document.getElementById("arena-mount");
  if (arenaMount) {
    // Speed defaults to ~300 inside createArena (≈3-4s to the ~60k PolyKAN
    // convergence cliff; see arena.js). Activation/width/seed are repeated
    // here only to keep the wiring self-documenting; reduced-motion is read
    // inside the controller.
    const arena = createArena(arenaMount, {
      activation: "polyKAN",
      width: 1,
      seed: 17,
      density: 0.4,
      lr: 1e-3,
      speed: 300,
    });
    register(arenaMount, arena);
  }

  const vizSuccess = document.getElementById("viz-success");
  if (vizSuccess) register(vizSuccess, renderSuccessBars(vizSuccess, SUCCESS_RATES));

  const vizDensity = document.getElementById("viz-density");
  if (vizDensity) register(vizDensity, renderDensitySweep(vizDensity, DENSITY_SWEEP));

  const vizAblation = document.getElementById("viz-ablation");
  if (vizAblation) register(vizAblation, renderAblation(vizAblation, ABLATION));

  const vizPca = document.getElementById("viz-pca");
  if (vizPca) register(vizPca, renderPCA(vizPca, PCA_ILLUSTRATIVE));

  // Pause off-screen animation/training to save CPU (spec §10). Feature-detected:
  // browsers without IntersectionObserver simply keep everything running.
  if (typeof IntersectionObserver !== "undefined") {
    const io = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const w = widgets.find((x) => x.root === entry.target);
        if (!w || !w.handle) continue;
        if (typeof w.handle.setPaused === "function") {
          w.handle.setPaused(!entry.isIntersecting);
        }
      }
    }, { rootMargin: "100px" });
    for (const { root } of widgets) io.observe(root);
  }
});
