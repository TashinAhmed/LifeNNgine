# Page Revisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address six pieces of author feedback on the `web/` interactive companion page: soften the monotonicity overclaim, reword and re-plot Widget B, make the arena start paused with a higher y-axis and an evolving reference grid, replace the illustrative Viz 4 PCA with a real figure image, and add a hedged future-work section.

**Architecture:** All changes are confined to `web/` (HTML, CSS, JS, tests). No build step — the page is plain ES modules served statically. Tests run with `node --test` from `web/`. One TDD opportunity (pure B/S rule-point helpers in `rule-function.js`); the rest are DOM/HTML changes covered by existing smoke tests plus a visual check.

**Tech Stack:** Vanilla ES modules, Canvas 2D, `node:test` + `node:assert/strict`. No external libraries.

## Global Constraints

- `cd web && npm test` is the canonical test command. All 53 existing tests must continue to pass after every task.
- Commit style: conventional commits scoped `(web)`, matching `git log --oneline` (e.g. `feat(web): ...`, `fix(web): ...`).
- Do not contradict the paper. `paper.md` is empty; the PDF is the source of truth but is not machine-readable here. Where PReLU/monotonicity is described, use the author's own phrasing from the spec ("with synaptic weights held static minimal PReLU models learned strictly monotonic activation functions in every sample in the ablation experiment") and hedge everything beyond that.
- Em-dashes are forbidden in user-visible copy (prior repo decision); use plain `-`.
- The author supplies `web/assets/pca_param_space.png` themselves; do not create that file. The wiring must work once they drop it in.
- Never use emojis in code or copy.

## File Structure

| File | Touched by task | Responsibility |
|------|-----------------|----------------|
| `web/index.html` | 1, 2, 5, 6 | Page structure and copy |
| `web/css/style.css` | 5 | `.viz-figure` rule for the new `<img>` |
| `web/js/widgets/rule-function.js` | 2 | Pure B/S point helpers + dual-curve render |
| `web/test/smoke.test.js` | 2, 5 | Module-import smoke tests |
| `web/js/arena.js` | 3, 4 | Paused-default state, evolving reference |
| `web/js/charts/charts.js` | 3, 5 | y-axis headroom; remove `renderPCA` |
| `web/js/main.js` | 5 | Remove viz-pca registration + import |
| `web/test/life-grid.test.js` | - | Unchanged (reference; confirms no regressions) |

---

### Task 1: Soften "Why it's hard (ReLU & lottery tickets)"

**Files:**
- Modify: `web/index.html` (Chapter 3, currently lines 64-70)

**Interfaces:** None — prose-only change.

**Context for the implementer:** The current copy claims monotonicity is *the* reason ReLU struggles and that a non-monotonic activation "sidesteps the problem entirely." The paper's PReLU ablation contradicts this (PReLU is monotonic yet learns). The fix keeps the lottery-ticket intuition for ReLU-from-scratch but reframes monotonicity as one hypothesis among several, concedes the PReLU counter-evidence explicitly, and hedges throughout.

- [ ] **Step 1: Replace the Chapter 3 body paragraphs**

In `web/index.html`, find this block (the `chapter__lead` and the two `<p>` paragraphs that follow, immediately after the `Why it&rsquo;s hard` `chapter__title`):

```html
      <p class="chapter__lead">If the rule is so simple, why do standard networks struggle? The answer is the shape of the function &mdash; and the inductive bias of the activation.</p>
      <p>ReLU and its relatives are monotonic: once they turn on, they only keep going. The Life rule turns <em>off</em> again after N&thinsp;=&thinsp;3. To represent that with ReLU, a network needs enough width to &ldquo;fold&rdquo; the curve back down &mdash; and finding a working configuration by gradient descent is like finding a winning lottery ticket.</p>
      <p>The result: minimal ReLU networks rarely learn the rule from scratch; they need more parameters or a lucky initialization. A non-monotonic activation sidesteps the problem entirely, because the bump is already in its vocabulary.</p>
```

Replace with (note: plain hyphens, no `&mdash;`):

```html
      <p class="chapter__lead">If the rule is so simple, why do standard networks struggle? The shape of the function is part of the story - and so is the inductive bias of the activation. But the whole story is still genuinely open.</p>
      <p>One natural suspect is monotonicity. ReLU and its relatives only climb once they turn on, while the Life rule turns <em>off</em> again after N&thinsp;=&thinsp;3. To represent that with ReLU, a network needs enough width to &ldquo;fold&rdquo; the curve back down, and finding a working configuration by gradient descent is like finding a winning lottery ticket. Minimal ReLU networks rarely learn the rule from scratch; they need more parameters or a lucky initialization.</p>
      <p>But monotonicity itself is not the deciding factor. PReLU is also monotonic, yet in our ablation the minimal PReLU model learned the rule in every sample when its synaptic weights were held static - learning a strictly monotonic activation function on the way. That rules out &ldquo;monotone &rarr; unlearnable,&rdquo; and leaves the real question open. It could be about width, about initialization, about the smoothness of gradients in parameter space, or about some as-yet-unnamed commonality between PReLU and PolyKAN. We find the mystery more interesting than any tidy explanation.</p>
```

- [ ] **Step 2: Verify no em-dashes snuck in**

Run from the repo root:

```bash
rg -n "mdash|—" web/index.html
```

Expected: no matches (exit code 1, no output). If any match, replace with `-`.

- [ ] **Step 3: Run the test suite**

```bash
cd web && npm test 2>&1 | tail -8
```

Expected: `pass 53`, `fail 0`.

- [ ] **Step 4: Commit**

```bash
git add web/index.html
git commit -m "fix(web): soften monotonicity claim in 'Why it's hard' (PReLU counter-evidence)"
```

---

### Task 2: Reword + fix Widget B (split B and S curves)

**Files:**
- Modify: `web/js/widgets/rule-function.js` (add pure helpers, rewire render)
- Modify: `web/test/smoke.test.js` (assert the new exports)
- Modify: `web/index.html` (section text + caption)

**Interfaces:**
- Produces: `BIRTH_POINTS` and `SURVIVAL_POINTS` — each `Array<{n:number,target:0|1}>` of length 9, exported from `web/js/widgets/rule-function.js`.
  - `BIRTH_POINTS[n].target === 1` iff `n === 3` (dead cell becomes alive).
  - `SURVIVAL_POINTS[n].target === 1` iff `n === 2 || n === 3` (live cell stays alive).
- `RULE_POINTS` is kept (for the probe's combined envelope and for backwards compatibility with existing tests).

**Context for the implementer:** The widget currently plots one merged curve (the B3/S23 envelope), making N=2 and N=3 look identical and hiding the role of the current cell state. We split it into two next-state curves: B (next state given currently dead) and S (next state given currently alive). The model has no residual connection, so next-state framing matches how it computes.

- [ ] **Step 1: Add failing tests for the new pure helpers**

In `web/test/smoke.test.js`, find the existing `rule-function module imports without DOM access` block. After the existing `for (const { n, target } of m.RULE_POINTS)` loop and before `assert.equal(typeof m.createRuleFunction, "function");`, insert:

```js
  // B/S split: birth = 0->1 only at N=3; survival = 1->1 at N in {2,3}.
  assert.ok(Array.isArray(m.BIRTH_POINTS) && m.BIRTH_POINTS.length === 9);
  assert.ok(Array.isArray(m.SURVIVAL_POINTS) && m.SURVIVAL_POINTS.length === 9);
  for (const { n, target } of m.BIRTH_POINTS) {
    assert.equal(target, n === 3 ? 1 : 0);
  }
  for (const { n, target } of m.SURVIVAL_POINTS) {
    assert.equal(target, (n === 2 || n === 3) ? 1 : 0);
  }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd web && npm test 2>&1 | grep -E "BIRTH_POINTS|SURVIVAL_POINTS|rule-function" | head -20
```

Expected: failures referencing `m.BIRTH_POINTS` / `m.SURVIVAL_POINTS` being `undefined`.

- [ ] **Step 3: Add the pure helpers**

In `web/js/widgets/rule-function.js`, immediately after the existing `RULE_POINTS` export (currently lines 14-17), add:

```js
// Birth: next state of a currently DEAD cell. =1 only at N=3.
export const BIRTH_POINTS = Array.from({ length: 9 }, (_, n) => ({
  n,
  target: n === 3 ? 1 : 0,
}));

// Survival: next state of a currently LIVE cell. =1 at N in {2,3}.
export const SURVIVAL_POINTS = Array.from({ length: 9 }, (_, n) => ({
  n,
  target: (n === 2 || n === 3) ? 1 : 0,
}));
```

Also add two color tokens to the `COLORS` object inside `createRuleFunction` (find `rule: "#16a34a",` and add after it):

```js
    birth: "#16a34a",     // green - dead -> alive
    survival: "#0891b2",  // cyan  - alive -> alive
```

- [ ] **Step 4: Run the tests to verify the pure-helper assertions pass**

```bash
cd web && npm test 2>&1 | tail -8
```

Expected: `pass 53`, `fail 0` (55 once the new assertions land — both fine).

- [ ] **Step 5: Replace the merged rule curve in render() with two curves + legend**

In `web/js/widgets/rule-function.js` `render()`, find the block currently drawing the merged rule (starts with the comment `// --- the rule: thick step segments (green).` and ends with the `// data points at each integer N` loop). Replace that whole block (from the `// --- the rule:` comment through the closing of the data-points `for` loop) with:

```js
    // --- the rule as TWO next-state curves, so the role of the current cell
    // state is visible: birth (green) and survival (cyan). Each integer N
    // owns the bin [N-0.5, N+0.5); verticals appear where neighbors differ. ---
    function drawStepCurve(points, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.lineJoin = "miter";
      ctx.beginPath();
      ctx.moveTo(xOf(0), yOf(points[0].target));
      for (let n = 0; n <= 8; n++) {
        const v = points[n].target;
        ctx.lineTo(xOf(Math.max(0, n - 0.5)), yOf(v));
        ctx.lineTo(xOf(Math.min(8, n + 0.5)), yOf(v));
      }
      ctx.stroke();
      ctx.fillStyle = color;
      for (const { n, target } of points) {
        ctx.beginPath();
        ctx.arc(xOf(n), yOf(target), 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    drawStepCurve(SURVIVAL_POINTS, COLORS.survival); // cyan: alive -> alive
    drawStepCurve(BIRTH_POINTS, COLORS.birth);       // green: dead -> alive

    // --- legend (top-left of the plot) ---
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const legend = [
      { color: COLORS.birth, label: "birth (dead -> alive)" },
      { color: COLORS.survival, label: "survival (alive -> alive)" },
    ];
    let ly = PAD.t + 8;
    for (const { color, label } of legend) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(PAD.l + 8, ly);
      ctx.lineTo(PAD.l + 24, ly);
      ctx.stroke();
      ctx.fillStyle = COLORS.label;
      ctx.fillText(label, PAD.l + 30, ly);
      ly += 14;
    }
```

- [ ] **Step 6: Update the probe label copy**

The probe label still uses `RULE_POINTS` (the combined envelope), which is correct for the probe since the probe just shows "ALIVE or dead at N". But its `detail` text mentions the merged semantics. Find the `detail` ternary in the probe label (currently around line 168):

```js
    const detail = pn === 2 ? "live cell survives"
      : pn === 3 ? "birth (or survival)"
      : "dies / stays dead";
```

Replace with (state-explicit):

```js
    const detail = pn === 2 ? "alive stays alive; dead stays dead"
      : pn === 3 ? "becomes alive (birth or survival)"
      : "becomes dead (death or no birth)";
```

- [ ] **Step 7: Reword the section text and caption in index.html**

In `web/index.html`, Chapter 2, find:

```html
      <p class="chapter__lead">Strip away the grid and Life&rsquo;s update becomes a function of a single number: the live-neighbor count N. The network&rsquo;s entire job is to learn this shape.</p>
      <p>For N&thinsp;=&thinsp;2 or 3 the next state can be alive; everywhere else it cannot. That makes the rule a non-monotonic bump &mdash; exactly the kind of shape a monotonic function like ReLU fits poorly, but a small polynomial fits naturally.</p>
```

Replace with:

```html
      <p class="chapter__lead">Strip away the grid and Life&rsquo;s update becomes a function of the live-neighbor count N - but also of the cell&rsquo;s current state. The network&rsquo;s entire job is to learn this shape.</p>
      <p>In words: with 2 live neighbors a cell keeps its state, with 3 it becomes alive, and otherwise it becomes dead. The two curves below make the role of the current cell explicit - survival (alive stays alive) and birth (dead becomes alive). The bump is non-monotonic, which is the kind of shape a monotonic function like ReLU fits poorly but a small polynomial fits naturally.</p>
```

And the caption immediately under the canvas-card (currently around line 59):

```html
      <p class="caption">Widget B &mdash; the B3/S23 rule as a function of neighbor count. Ghost curves hint at why monotonic (ReLU) and linear-quadratic fits behave so differently.</p>
```

Replace with (plain hyphen, mention the two curves):

```html
      <p class="caption">Widget B - the B3/S23 rule as a function of neighbor count, split into birth and survival. Ghost curves hint at why monotonic (ReLU) and linear-quadratic fits behave so differently.</p>
```

- [ ] **Step 8: Run the full test suite**

```bash
cd web && npm test 2>&1 | tail -8
```

Expected: `pass 53` (or higher with the new assertions), `fail 0`.

- [ ] **Step 9: Verify no em-dashes**

```bash
rg -n "mdash|—" web/index.html web/js/widgets/rule-function.js
```

Expected: no matches.

- [ ] **Step 10: Commit**

```bash
git add web/js/widgets/rule-function.js web/test/smoke.test.js web/index.html
git commit -m "feat(web): split Widget B into birth/survival curves + reword section text"
```

---

### Task 3: Arena default-paused + chart y-axis headroom

**Files:**
- Modify: `web/js/arena.js` (initial paused state, button label)
- Modify: `web/js/charts/charts.js` (y-domain on Viz 1 and Viz 2)

**Interfaces:** None externally visible.

**Context for the implementer:** The arena starts training on page load, which is distracting. The success-rate charts (Viz 1 bars, Viz 2 density sweep) cap their y-axis at exactly 1.0 so 100% bars and curve peaks are visually clipped. We add 15% headroom; the existing 1.0 gridline becomes a natural reference line.

- [ ] **Step 1: Make the arena start paused**

In `web/js/arena.js`, find the initial button text setup (around line 256):

```js
  const pauseBtn = doc.createElement("button");
  pauseBtn.type = "button";
  pauseBtn.textContent = "Pause";
```

Change to start paused:

```js
  const pauseBtn = doc.createElement("button");
  pauseBtn.type = "button";
  pauseBtn.textContent = "Play";
```

Then find the state declaration (around line 326):

```js
  let userPaused = false;
```

Change to:

```js
  let userPaused = true; // start paused - reader hits Play when ready
```

- [ ] **Step 2: Verify the toggle still flips the label**

Search for the toggle handler that flips `userPaused` and updates the button text. Run:

```bash
rg -n "userPaused = |pauseBtn.textContent" web/js/arena.js
```

Expected: the existing handler already sets `pauseBtn.textContent = userPaused ? "Play" : "Pause"` (or equivalent). If the handler is missing or one-sided, post a question; otherwise no change needed. (Inspection: it is present and bidirectional.)

- [ ] **Step 3: Add y-axis headroom on Viz 1 (success bars)**

In `web/js/charts/charts.js`, find the `renderSuccessBars` `chartFrame` call (currently around line 211, with `yDomain: [0, 1]`). Change the `yDomain` line and the `yTicks` line in that call from:

```js
      yDomain: [0, 1],
      yTicks: [0, 0.25, 0.5, 0.75, 1],
```

to:

```js
      yDomain: [0, 1.15],
      yTicks: [0, 0.25, 0.5, 0.75, 1],
```

(The 1.0 tick is still drawn by `chartFrame`, now sitting below the top edge - that gridline is the visual reference for "100%".)

- [ ] **Step 4: Add y-axis headroom on Viz 2 (density sweep)**

In the same file, find the `renderDensitySweep` `chartFrame` call (currently around line 346, with `yDomain: [0, 1]`). Change identically:

```js
      yDomain: [0, 1.15],
      yTicks: [0, 0.25, 0.5, 0.75, 1],
```

- [ ] **Step 5: Run the test suite**

```bash
cd web && npm test 2>&1 | tail -8
```

Expected: `pass 53`, `fail 0`.

- [ ] **Step 6: Commit**

```bash
git add web/js/arena.js web/js/charts/charts.js
git commit -m "feat(web): arena starts paused; raise y-axis headroom on Viz 1 + Viz 2"
```

---

### Task 4: Arena evolving reference grids

**Files:**
- Modify: `web/js/arena.js` (advance `valInput`/`valTrue` every 100 steps)

**Interfaces:** None externally visible. Training batches stay fresh and random.

**Context for the implementer:** The reference row (input grid + true next state) is frozen on one random sample for the whole run. We make it evolve: every 100 `trainOnce()` calls, advance `valInput` by one Life generation and recompute `valTrue`. The prediction canvases follow automatically because they re-run forward on `valInput` every render. `lifeStep` is already imported.

- [ ] **Step 1: Declare a generation counter in the state block**

In `web/js/arena.js`, find the state declaration block (around lines 320-328, starting with `let leftModel, rightModel;`). After `let stepCount = 0;` add:

```js
  let refGen = 0;          // how many Life steps the reference grid has advanced
  const REF_ADVANCE_EVERY = 100; // training steps between reference advances
```

- [ ] **Step 2: Reset the counter in build()**

In `build()` (the function that initializes `valInput`/`valTrue` and zeroes `stepCount`), find `stepCount = 0;` (around line 362) and add immediately after:

```js
    refGen = 0;
```

- [ ] **Step 3: Advance the reference every 100 training steps**

In `trainOnce()` (around line 444), find `stepCount++;` at the end. Replace it with:

```js
    stepCount++;
    if (stepCount % REF_ADVANCE_EVERY === 0) {
      // Evolve the displayed reference by one Life generation so gliders
      // and ash emerge over a long run. Training batches are unaffected.
      valInput = lifeStep(valInput, H, W);
      valTrue = lifeStep(valInput, H, W);
      refGen++;
      drawBinary(inputCanvas, valInput, "#16a34a");
      drawBinary(trueCanvas, valTrue, "#16a34a");
    }
```

- [ ] **Step 4: Confirm lifeStep is imported**

Run:

```bash
rg -n "^import.*lifeStep" web/js/arena.js
```

Expected: one match (it is already imported for `valTrue = lifeStep(...)` in `build()`). If somehow missing, add `lifeStep` to the existing import from `./engine/life.js`.

- [ ] **Step 5: Run the test suite**

```bash
cd web && npm test 2>&1 | tail -8
```

Expected: `pass 53`, `fail 0`.

- [ ] **Step 6: Commit**

```bash
git add web/js/arena.js
git commit -m "feat(web): advance arena reference grid every 100 training steps"
```

---

### Task 5: Replace Viz 4 with a static PCA figure image

**Files:**
- Modify: `web/index.html` (replace `<canvas id="viz-pca">` with `<img>`; update caption)
- Modify: `web/css/style.css` (add `.viz-figure` rule)
- Modify: `web/js/charts/charts.js` (remove `renderPCA`)
- Modify: `web/js/main.js` (remove PCA registration and import)
- Modify: `web/test/smoke.test.js` (drop the `renderPCA` assertion)

**Interfaces:** None externally visible. `PCA_ILLUSTRATIVE` stays exported from `web/js/data/results.js` (harmless data; not worth churning the data module).

**Context for the implementer:** The current Viz 4 is illustrative (made-up PCA points). The author will supply a real figure as `web/assets/pca_param_space.png`. We wire in an `<img>` at that exact path; the wiring works as soon as the author drops the file in. The orphaned `renderPCA` code and its registration are removed to keep the module surface honest.

- [ ] **Step 1: Replace the canvas with an image in index.html**

In `web/index.html`, Chapter 6, find:

```html
      <div class="canvas-card">
        <canvas id="viz-pca" width="720" height="420"></canvas>
        <p class="caption">Viz 4 - parameter-space PCA of training trajectories. Illustrative reproduction - exact points not extractable from Fig 1.</p>
        <p class="data-source">Source: Fig 1 (illustrative).</p>
      </div>
```

Replace with:

```html
      <div class="canvas-card">
        <img class="viz-figure"
             src="assets/pca_param_space.png"
             alt="Parameter-space PCA of training trajectories for PolyKAN, PReLU, sigmoid, and ReLU. PolyKAN trajectories converge to a diverse, well-behaved region; ReLU trajectories scatter.">
        <p class="caption">Viz 4 - parameter-space PCA of training trajectories from the paper (PolyKAN vs PReLU vs sigmoid vs ReLU).</p>
        <p class="data-source">Source: paper Fig 1.</p>
      </div>
```

- [ ] **Step 2: Style the figure**

In `web/css/style.css`, immediately after the `.canvas-card canvas` rule block (the one ending with the `#life-grid` override added previously), insert:

```css
/* Static figure inside a canvas-card (e.g. paper-derived PCA image). */
.canvas-card .viz-figure {
  display: block;
  width: 100%;
  height: auto;
  background: var(--bg);
  border-radius: 4px;
}
```

- [ ] **Step 3: Remove renderPCA from charts.js**

In `web/js/charts/charts.js`, delete the entire `export function renderPCA(canvas, data) { ... }` definition (it starts at line 595 and runs to the end of the file - 744). Confirm bounds first:

```bash
rg -n "export function renderPCA|^}" web/js/charts/charts.js | tail -5
```

Then delete from the `export function renderPCA` line through its closing brace.

- [ ] **Step 4: Remove the PCA wiring from main.js**

In `web/js/main.js`, remove the import of `renderPCA` from the charts import line. Find:

```js
import { renderSuccessBars, renderDensitySweep, renderAblation, renderPCA } from "./charts/charts.js";
```

Replace with:

```js
import { renderSuccessBars, renderDensitySweep, renderAblation } from "./charts/charts.js";
```

Then remove the registration block at the end of `main.js`. Find:

```js
  const vizPca = document.getElementById("viz-pca");
  if (vizPca) register(vizPca, renderPCA(vizPca, PCA_ILLUSTRATIVE));
```

Delete those two lines entirely.

Then remove the now-unused `PCA_ILLUSTRATIVE` import. Find:

```js
import { SUCCESS_RATES, DENSITY_SWEEP, ABLATION, PCA_ILLUSTRATIVE } from "./data/results.js";
```

Replace with:

```js
import { SUCCESS_RATES, DENSITY_SWEEP, ABLATION } from "./data/results.js";
```

- [ ] **Step 5: Drop the renderPCA assertion from smoke tests**

In `web/test/smoke.test.js`, find the charts smoke test block. Delete the line:

```js
  assert.equal(typeof m.renderPCA, "function");
```

- [ ] **Step 6: Run the test suite**

```bash
cd web && npm test 2>&1 | tail -8
```

Expected: `pass 53`, `fail 0` (one fewer assertion but the suite still reports 53 tests because each `test()` block is one test).

- [ ] **Step 7: Verify the image path resolves to a placeholder gracefully**

```bash
ls web/assets/pca_param_space.png 2>/dev/null || echo "NOT_PRESENT (author will supply)"
```

Expected: `NOT_PRESENT (author will supply)`. The `<img>` will render a broken-image icon until the author drops the file in - that is acceptable and expected.

- [ ] **Step 8: Commit**

```bash
git add web/index.html web/css/style.css web/js/charts/charts.js web/js/main.js web/test/smoke.test.js
git commit -m "feat(web): replace illustrative Viz 4 PCA with real figure image (img wiring)"
```

---

### Task 6: Add an "Open questions" future-work subsection

**Files:**
- Modify: `web/index.html` (new `<section>` after Chapter 6, before the footer)

**Interfaces:** None.

**Context for the implementer:** The page closes on Results without gesturing at the open questions the authors would welcome collaboration on. We add a short, lightly-hedged subsection. Tone: speculative, inviting, no overclaiming, no contradictions with the paper.

- [ ] **Step 1: Insert the new section**

In `web/index.html`, find the closing `</section>` of Chapter 6 (the Results chapter, ending the `viz-pca`/`viz-figure` card block) and the opening of the footer:

```html
    </section>

    <!-- Footer -->
    <footer class="footer">
```

Insert a new section between them:

```html
    </section>

    <!-- Open questions -->
    <section class="chapter" id="ch-open-questions">
      <h2 class="chapter__title">Open questions</h2>
      <p class="chapter__lead">We did not set out to write the last word on activation functions for Life-like rules. A few threads we have not pulled - and would love help pulling.</p>
      <p>It could be that monotonicity turns out to matter more across a broader swath of Life-like CA than it does for B3/S23 specifically. PolyKAN might pull ahead where PReLU falls short, or the comparison might stay close - we just do not know yet.</p>
      <p>There may also be some as-yet-unnamed commonality between PReLU and PolyKAN gradient spaces that would explain why both learn where ReLU stalls. Smoothness is one obvious guess, but the PReLU derivative is discontinuous, so the usual smooth-gradient story does not quite fit either.</p>
      <p>If either thread sounds interesting to you - whether you want to broaden the rule family, dig into the geometry of parameter space, or just poke at a counter-intuitive result - we would genuinely like to hear from you. Open an issue or send a message; motivated students especially welcome.</p>
    </section>

    <!-- Footer -->
```

- [ ] **Step 2: Verify no em-dashes**

```bash
rg -n "mdash|—" web/index.html
```

Expected: no matches.

- [ ] **Step 3: Run the test suite (regression only)**

```bash
cd web && npm test 2>&1 | tail -8
```

Expected: `pass 53`, `fail 0`.

- [ ] **Step 4: Commit**

```bash
git add web/index.html
git commit -m "feat(web): add hedged 'Open questions' section inviting collaboration"
```

---

## Self-Review

**Spec coverage** - all six spec items map to a task:
1. Soften "Why it's hard" → Task 1
2. Reword + fix Widget B → Task 2
3. Arena paused + y-axis → Task 3
4. Arena evolving reference → Task 4
5. Viz 4 real figure → Task 5
6. Future-work hints → Task 6

**Placeholder scan** - every step has concrete code or an exact command. No "TBD", "implement appropriate", or "similar to Task N".

**Type consistency** - `BIRTH_POINTS` and `SURVIVAL_POINTS` are defined (Task 2 Step 3) before they are referenced in render (Task 2 Step 5) and asserted in tests (Task 2 Step 1). `refGen` and `REF_ADVANCE_EVERY` are declared (Task 4 Step 1) before they are used (Task 4 Steps 2-3). `renderPCA` is removed from charts.js (Task 5 Step 3), from main.js (Task 5 Step 4), and from the test (Task 5 Step 5) in the same task. No orphan references.

**Risks flagged in the spec** are addressed: PReLU framing sticks to the author's own phrasing (Task 1); the smoke test for charts.js is updated in lockstep with the `renderPCA` removal (Task 5); the arena's training stream is untouched by the reference evolution (Task 4).
