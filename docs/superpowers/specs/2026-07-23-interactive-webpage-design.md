# Interactive Webpage for LifeNNgine — Design Spec

**Date:** 2026-07-23
**Status:** Approved (design phase)
**Reference page:** https://pub.sakana.ai/picbreeder-vlm/
**Paper:** Ahmed & Davis, "It's Much Easier for Neural Networks to Learn Game of Life Dynamics with the Right Activation Function: Polynomial Kolmogorov-Arnold Networks", ALIFE 2026 (arXiv:2606.23587)
**Repo:** LifeNNgine (`src/cann/`)

---

## 1. Goal

Create a single-page, long-form **scrollytelling** interactive research-demo webpage for the LifeNNgine paper, in the spirit of the picbreeder-vlm page (narrative prose interleaved with widgets that *genuinely run*). The centerpiece is a **live in-browser training arena** where a real, tiny PolyKAN network learns Conway's Game of Life update rule via JavaScript gradient descent, shown side-by-side with a ReLU network that struggles — demonstrating the paper's thesis interactively.

## 2. Key decisions (approved)

- **Interactivity depth:** Live in-browser training (real model, real gradient descent in JS).
- **Content scope:** Full scrollytelling narrative covering the whole paper.
- **Visual identity:** Dark, modern, with the Game of Life grid as recurring motif.
- **Engine approach (Approach A):** Vanilla JS + a tiny hand-written reverse-mode autodiff engine. No dependencies.

## 3. Stack & constraints

- 100% vanilla HTML / CSS / JS. **No build step, no bundler, no runtime npm dependencies, no external libraries** (no D3, Chart.js, TF.js, KaTeX).
- Hostable as plain static files; works by opening `index.html` or via any static server.
- Rendering on `<canvas>` and inline SVG only.
- All math rendered with inline Unicode/MathML — no KaTeX.
- Honors `prefers-reduced-motion`. Training/animation paused when off-screen (IntersectionObserver) to save CPU.

## 4. File layout

New `web/` directory at repo root:

```
web/
├── index.html                 # the scrollytelling page, all 7 chapters
├── css/
│   └── style.css              # dark theme, typography, layout, responsive
├── js/
│   ├── engine/                # the autodiff core + model
│   │   ├── tensor.js          # Float32Array tensor + reverse-mode autodiff
│   │   ├── ops.js             # conv3x3Circular, conv1x1, poly, relu, ... + backward
│   │   ├── life.js            # true B3/S23 rule (training targets + grid sim)
│   │   ├── model.js           # L(m,n) ActNN/PolyKAN port + Adam
│   │   └── gradcheck.js       # finite-difference gradient self-test (dev flag on load)
│   ├── widgets/
│   │   ├── life-grid.js       # Widget A
│   │   ├── rule-function.js   # Widget B
│   │   └── activation-plot.js # Widget C
│   ├── charts/                # Viz 1–4 (hardcoded paper data)
│   └── arena.js               # the centerpiece live-training widget
├── data/
│   └── results.js             # hardcoded values from Table 2 / Fig 3/4/6/1
└── assets/                    # logo, favicon, og:image
```

## 5. Page structure (7 chapters)

**Chapter 0 — Hero.** Title, subtitle ("…with the Right Activation Function"), authors (Tashin Ahmed, Q. Tyrell Davis), ALIFE 2026, links to paper/code. Background: a faint, slowly self-running Game of Life grid motif.

**Chapter 1 — The Pocket Universe.** Conway's Game of Life as toy physics. *Widget A: interactive Life grid.*

**Chapter 2 — The rule, as a function.** Life's update is a non-monotonic interval function over neighbor count N (0 at {0,1}, 1 at {2,3}, 0 at {4–8}). *Widget B: the rule-function visualizer.*

**Chapter 3 — Why it's "hard" (ReLU & lottery tickets).** The Springer & Kenyon result: minimal ReLU nets rarely learn the rule; scale / "winning tickets" needed. Learning-as-search framing.

**Chapter 4 — The centerpiece: watch it learn, live.** *Live in-browser training arena* (Section 6).

**Chapter 5 — The activation-function zoo.** *Widget C: activation explorer* + *Viz 1: success-rate bar chart.*

**Chapter 6 — Robustness & results.** *Viz 2 density sweep*, *Viz 3 knockout/ablation*, *Viz 4 parameter-space PCA.*

**Footer.** Citation (BibTeX), repo link, CC BY 4.0 note.

## 6. Centerpiece: live training engine & arena

### 6.1 Model (ported faithfully from `base.py` / `polykan.py`)

The L(m,n) ActNN architecture:

```
per depth block:  Conv2d(1 -> 2m, 3x3, circular pad) -> activation(2m)
                  Conv2d(2m -> 1, 1x1)               -> activation(1)
then:             Conv2d(1 -> 1, 1x1) -> sigmoid
```

Default L(1,1) PolyKAN = **34 trainable params** (matches paper). L(1,1) ReLU = **25**. Circular padding = toroidal grid, matching repo.

### 6.2 Autodiff core

A small reverse-mode autodiff over `Float32Array`-backed tensors. Ops (each with `.forward()` and `.backward()`):
- `conv3x3Circular` (3x3 conv with toroidal wrap)
- `conv1x1`
- `polyActivate` (per-channel f(x) = Σ wᵢ xⁱ)
- standard: `relu`, `prelu`, `silu`, `square`, `sigmoid`
- BCE loss

Gradients verified by finite-difference check (`gradcheck.js`) so we trust correctness.

### 6.3 Training loop (requestAnimationFrame)

Per step:
1. Sample a fresh random binary 32x32 grid at the chosen density.
2. Apply the true Life rule (B3/S23) via `life.js` to get the target.
3. Forward both models (PolyKAN and ReLU) on a batch of 8.
4. BCE loss -> backprop -> Adam update (lr = 1e-3 default).
5. "Epoch" = one model update (matches repo; no fixed dataset, continuous fresh data).
6. Early-stop when grid accuracy (decision boundary 0.5) = 100% for 2 contiguous steps (matches repo).

### 6.4 What the user sees, live

Two training columns (PolyKAN | ReLU), each showing:
- the input grid
- the true next-state grid
- the model's predicted next-state grid (converging cell-by-cell)
- a live loss/accuracy sparkline

Shared step counter. PolyKAN typically reaches 100% within view; ReLU stalls — the paper's thesis, demonstrated live.

### 6.5 Controls

Activation selector (PolyKAN / ReLU / PReLU / SiLU / Square …), density slider, learning rate, width `m` (1/2/4), seed, play/pause, reset, speed.

### 6.6 Performance

At 32x32, one forward+backward ≈ ~1k scalar ops × 34 params — trivial. Batch of 8 still allows tens of steps/frame, so training visibly completes in a few seconds.

## 7. Supporting interactive widgets

**Widget A — Interactive Life grid (Ch 1).** 32x32 toroidal canvas. Click/drag to paint/erase; Step, Play/Pause, Randomize + density slider, Clear; preset patterns (glider, blinker, block, pulsar). Birth/death flashes briefly. A dimmed self-running instance serves as the hero background.

**Widget B — Rule as a function (Ch 2).** Plot neighbor-count N (0–8) vs target state: flat-0 -> 1 at {2,3} -> flat-0. Draggable probe ("if a cell has N live neighbors…"). Faint overlays of a ReLU and a parabola to motivate why monotonic/linear is a poor fit and a non-monotonic polynomial is natural.

**Widget C — Activation explorer (Ch 5).** Left: standard zoo (ReLU, PReLU, SiLU, Sigmoid, Tanh, Square) overlaid on one axis, toggleable. Right: the live PolyKAN polynomial f(x)=w₀+w₁x+w₂x² with sliders for w₀/w₁/w₂; scrub to morph the curve, with the Life rule-shape ghosted behind. Readout states whether the curve is monotonic or non-monotonic (ties to Table 2).

## 8. Results & data visualizations

Underlying numbers are **hardcoded from the paper's tables/figures** (rendered as styled canvas/SVG, no chart library). Each chart cites its source in a caption.

**Viz 1 — Success-rate by activation (Ch 5).** Horizontal bars from Table 2: PolyKAN 1.0, Square 0.94, SiLU 0.94, RootSquare 0.50, LeakyReLU 0.25, CELU 0.06, Sigmoid 0.0, Tanh 0.0, ReLU 0.0. Hover reveals trainable-param count + monotonicity/differentiability. Punchline: ReLU dead last.

**Viz 2 — Density sweep (Ch 6).** Multi-line chart from Fig 4: success rate vs initial on-density (0.05–0.95) for PolyKAN/PReLU/SiLU/ReLU. Shows PolyKAN's broad robustness and the d₀=0.95 cliff. Toggleable legend.

**Viz 3 — Knockout / ablation (Ch 6).** Grouped bars from Fig 6 / Table 2: PolyKAN & PReLU each with full / activations-only / weights-only, with param counts (34/29/25 etc.). Headline: PolyKAN hits 128/128 whether or not weights train.

**Viz 4 — Parameter-space PCA (Ch 6).** Interactive 2D scatter from Fig 1: 128 trajectories in PC1/PC2, points colored by loss (light->dark), circle=success / x=failure. Selector swaps PolyKAN/PReLU/ReLU/Sigmoid to contrast PolyKAN's smooth solution fan-out vs ReLU's ridge-split failures. Pan/zoom.

**Data integrity:** values transcribed directly from the paper; the page never misrepresents results.

## 9. Correctness gates (built-in)

- `gradcheck.js`: finite-difference check of every op's backward; console error on failure. Runs on load under a dev flag.
- Assertion that ported L(1,1) PolyKAN has **34** trainable params and L(1,1) ReLU has **25** (guards against port drift).
- The arena's target rule uses the *same* `life.js` as Widget A, so what it learns is provably the real rule.

## 10. Visual system

- **Theme:** near-black background (`#0b0d10`), off-white text, subtle gray grid lines.
- **Accents:** vivid green (~`#39ff14`) for live cells / "alive" / PolyKAN success; warm amber for ReLU/contrast; cyan/magenta for loss curves. Color-blind-aware (not red/green-only).
- **Typography:** clean system sans for body, heavier display weight for chapter headers. Inline Unicode/MathML for math.
- **Motif:** the Life grid recurs in dividers, section-number tiles, and the hero background (3x3 / 1x1 grid theme).
- **Responsiveness:** canvas widgets scale to container; on narrow screens the arena stacks vertically (PolyKAN above ReLU).

## 11. Out of scope (YAGNI)

- No real PyTorch / Pyodide, no server.
- No n-step prediction (n>1) in the arena.
- No live re-running of the 128-run experiments (hardcoded results instead).
- No i18n.
- No external libraries of any kind.

## 12. Open questions / risks

- **Risk: autodiff correctness.** Mitigated by the finite-difference gradcheck gate and the param-count assertion.
- **Risk: training speed / visual convergence.** Mitigated by tiny model + batched steps per frame; default hyperparameters chosen so PolyKAN converges within a few seconds (matches paper's 128/128 success).
- **Data transcription:** density-sweep and PCA point values are read from figures; we will transcribe faithfully and cite source figures, accepting minor reading imprecision where the paper only plots (not tables) those values.
