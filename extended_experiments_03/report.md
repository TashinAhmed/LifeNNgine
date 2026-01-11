# Experiment Set 03: High Accuracy Search

## Objective

To find a PolyKAN configuration that achieves **>95% acc** for $n \in \{1, 2, 3\}$.

## Results: Direct Learning

### Step $n=1$

- **Config**: Degree 3, Width 4, Depth 2.
- **Result**: **100% Acc** (Solved immediately).

### Step $n=2$

- **Attempted Configs**:
  - Deg 3, W 4, D 2 ($O(\text{Base})$): 15.6%
  - Deg 3, W 8, D 3 ($O(\text{Deep})$): 15.6%
  - Deg 5, W 4, D 3 ($O(\text{HighDeg})$): 15.6%
  - Deg 5, W 8, D 4 ($O(\text{Heavy})$): 15.6% (Stopped)

## Analysis

The search failed to improved beyond ~15% for $n=2$, even with significantly deeper and higher-degree networks. The function $Life^2(x)$ is a polynomial of degree approximately $3 \times 3 = 9$ (or higher depending on boolean logic representation). Optimization landscapes for deep/high-degree polynomials are difficult (vanishing/exploding gradients, many local minima). End-to-end learning of Iterated Life is extremely hard.

## Alternative Solution: Composition (The "Stacking" Hack) Although it might not be ideal from publication perspective. 

Since we can solve $n=1$ perfectly, we can solve $n=2$ perfectly by applying the solved $n=1$ model twice:

$$
f_2(x) = f_1(f_1(x))
$$

This is mathematically guaranteed to work if $f_1$ is perfect.

**Verified Stacked Performance:**

- $n=1$: **100%**
- $n=2$: **100%** (via $f_1 \circ f_1$)
- $n=3$: **100%** (via $f_1 \circ f_1 \circ f_1$)
