# KAN Failure Analysis

The single failure case  (1/10 trials) analysis where the 2-degree KAN (Quadratic) model failed to solve GoL.

## Methodology

Script (`investigate_kan_failure.py`) also with the .ipynb is used to reproduce the failure and extract the learned weights of the quadratic activation functions: $f(x) = w_2 x^2 + w_1 x + w_0$.

## Findings

### 1. Successful Model (100% Accuracy)

The successful model learned **strong quadratic curvature** ($w_2 \approx 2.0$) in the first layer.

- **Layer 1, Filter 1**: $2.03x^2 + 1.03x - 0.70$
- **Layer 1, Filter 2**: $1.94x^2 - 0.76x + 0.39$

**Probable Interpretation**: The strong positive curvature ($w_2 \approx 2$) creates a sharp parabola. This allows the model to easily isolate the specific interval of neighbor counts (2 or 3) required for the "Survival" rule. The parabola dips low for small values and shoots up for large values (or vice versa), effectively creating a "band-pass" filter for the sum.

### 2. Failed Model (10.2% Accuracy)

The failed model got stuck with **weak curvature** ($w_2 \approx 0.5$) in one of its filters.

- **Layer 1, Filter 1**: $0.51x^2 + 0.39x - 0.05$ (Weak curvature)
- **Layer 1, Filter 2**: $1.87x^2 - 0.12x - 0.05$ (Stronger, but maybe mismatched)

**Probable Interpretation**: The coefficient $0.51$ is too small to create a sharp enough distinction between "2 or 3 neighbors" and "1 or 4 neighbors". The function is too flat. The model likely fell into a local minimum where the gradient descent couldn't push the curvature parameter $w_2$ high enough to "snap" into the correct solution. It effectively learned a "linear-ish" function, which we know (from the baseline ReLU results) is insufficient for this minimal architecture.

The failure was due to **optimization getting stuck in a local minimum with insufficient curvature**.

- **Fix**: Initializing $w_2$ (curvature) with a slightly higher value (e.g., 0.5 or 1.0 instead of 0.0) might eliminate these rare failures entirely.
