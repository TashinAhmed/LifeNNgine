# Page revisions design - addressing author feedback

Date: 2026-08-06
Status: Approved
Scope: `web/` interactive page only

## Context

The paper author reviewed the interactive companion page and gave six pieces of
feedback. All six are in scope. Two are corrections of claims that overstate
what the paper actually shows; four are UI/UX refinements to existing widgets.

Guiding principles from the author:

- The paper is the ground truth; the page must not contradict it.
- The page may be more speculative and informal than the paper, but uncertainty
  must be reflected in hedged language ("could be", "might", "seems likely").
- The page is "an extra" - polish, not a parallel artifact.

## Items

### 1. Soften "Why it's hard (ReLU & lottery tickets)"

**Problem.** The current copy (index.html L64-70) asserts that monotonicity is
the reason ReLU struggles and that a non-monotonic activation "sidesteps the
problem entirely." The PReLU ablation in the paper contradicts this: PReLU is
monotonic, yet minimal PReLU models learn the rule in every sample when their
weights are held static. So monotonicity is not the determining factor.

**Fix.** Reframe monotonicity as one hypothesis among several. Keep the
lottery-ticket intuition for ReLU-from-scratch (it is still a useful story),
but explicitly concede the PReLU counter-evidence. Hedge throughout. Do not
claim non-monotonicity "solves" anything.

Files: `web/index.html` only.

### 2. Reword + fix Widget B ("The rule, as a function")

**Problem.** The current text says the next state "can be alive" for N=2 or 3,
which hides that the update depends on the current cell state. The widget's
merged curve makes N=2 and N=3 visually identical, reinforcing the ambiguity.

**Fix.**

- Section text (index.html L53-55): change to the explicit form - "with 2
  live neighbors a cell keeps its state, with 3 it becomes 1, otherwise it
  becomes 0."
- Widget (js/widgets/rule-function.js): split the merged curve into two
  next-state curves on the same axes:
  - S (survival): next state given currently alive. =1 for N in {2,3}, else 0.
  - B (birth): next state given currently dead. =1 for N=3, else 0.
- Add a small legend so the two curves are distinguishable.
- Keep the ghost curves (ReLU ramp, polynomial bump) - they still illustrate
  why monotonic vs non-monotonic fits behave differently.

Why next-state framing (not the Lenia update/delta framing): the model has no
residual connection to t=0, so its output literally is the next state. The
update framing would add a clipping story the model does not perform.

Files: `web/index.html`, `web/js/widgets/rule-function.js`.

### 3. Arena: default paused + y-axis headroom

**Problem.** The arena starts training immediately on page load, which is
distracting and burns CPU before the reader has context. Separately, Viz 1
and Viz 2 charts cap their y-axis at exactly 1.0, so 100% bar labels and the
top of success curves get clipped.

**Fix.**

- js/arena.js: initialize `userPaused = true`. The toggle starts labeled
  "Play". IntersectionObserver visibility-pause logic is unchanged.
- js/charts/charts.js: change `yDomain: [0, 1]` to `yDomain: [0, 1.15]` at
  the two success-rate charts (Viz 1 around L346, Viz 2 around L448). Add a
  faint reference line at y=1.0 so 100% is still anchored visually.

Files: `web/js/arena.js`, `web/js/charts/charts.js`.

### 4. Arena: evolving reference grids

**Problem.** The reference row (input grid + true next state) is frozen on a
single random sample for the whole run. The author wants it to evolve so
recognizable Life structures (gliders, ash) appear during a long training
run.

**Fix.**

- In `build()`, keep the current seeded random `valInput` and its `valTrue =
  lifeStep(valInput)` initialization.
- Add a generation counter. Every 100 `trainOnce()` calls, advance:
  `valInput = lifeStep(valInput); valTrue = lifeStep(valInput);` then redraw
  the two reference canvases.
- Training batches stay fresh and random - the evolution applies only to the
  displayed reference, not the training stream. (Random batches are needed
  for stochastic gradient descent.)
- The prediction canvases follow automatically because they re-run forward on
  `valInput` every render frame.
- Reset (`resetBtn`) regenerates the initial random `valInput` and zeroes the
  generation counter, matching current reset semantics.

Files: `web/js/arena.js` only.

### 5. Viz 4: real PCA figure (replaces illustrative canvas)

**Problem.** The current Viz 4 is an illustrative PCA scatter generated from
made-up data ("exact points not extractable from Fig 1"). The author objects
to infographics that are not based on real data.

**Fix.**

- The author will regenerate the parameter-space PCA figure from
  `notebooks/parameter_space.ipynb` with a palette tuned to the page theme,
  and drop it at `web/assets/pca_param_space.png`.
- index.html: replace `<canvas id="viz-pca">` with `<img class="viz-figure"
  src="assets/pca_param_space.png" alt="...">`.
- The associated `renderPCA` chart code in js/charts/charts.js and its
  registration in js/main.js become unused; remove them.
- Update the caption: drop "illustrative"; describe the figure as the
  parameter-space PCA of training trajectories from the paper.

Files: `web/index.html`, `web/css/style.css` (new `.viz-figure` rule),
`web/js/charts/charts.js` (remove renderPCA), `web/js/main.js` (remove PCA
registration and unused import). The actual PNG is supplied by the author.

### 6. Future-work hints

**Problem.** The page currently closes on Results without gesturing at the
open questions the author would actually welcome collaboration on.

**Fix.** Add a short, lightly-hedged paragraph as a new mini-section
"Open questions" at the end of Chapter 6 (Robustness & results), before the
footer. Concrete content:

- Monotonicity may turn out to matter more across the broader Life-like CA
  family - PolyKAN could pull ahead where PReLU falls short. (Hedged.)
- There may be an as-yet-unnamed commonality between PReLU and PolyKAN
  gradient spaces that would explain why both learn where ReLU stalls.
- An explicit invitation: if a reader wants to pursue either thread, the
  authors would love to hear from them.

Tone: speculative, inviting, no overclaiming. No contradictions with the
paper.

Files: `web/index.html` only.

## Out of scope

- Regenerating the themed PCA PNG (author will supply).
- Adjusting colormaps in other notebooks.
- Expanding the page to cover Lenia or the broader Life-like family beyond
  the hedged future-work mention.
- README.md or any non-`web/` files.

## Risks

- The PReLU/monotonicity reframe must be double-checked against the paper
  (paper.md is empty in the repo; the 2606.23587v1.pdf is the source of
  truth). Mitigation: stick to the author's own description of the result
  ("with synaptic weights held static minimal PReLU models learned strictly
  monotonic activation functions in every sample") and hedge everything else.
- Removing renderPCA must not break the test that imports charts.js. Will
  re-run `npm test` after the change.
- The arena's evolving reference must not perturb the deterministic-seed
  training stream; only the displayed reference evolves.
