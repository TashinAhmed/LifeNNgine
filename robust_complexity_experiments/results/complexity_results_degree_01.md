# Experiment Results: Activation Degree vs Rule Complexity

## Hypothesis

Higher complexity CA rules require higher degree polynomial activation functions.

## Results Table

| Rule                 | B/S          | Deg 2          | Deg 4          | Deg 8          | Deg 16 |
| -------------------- | ------------ | -------------- | -------------- | -------------- | ------ |
| **GoL**        | B3/S23       | **100%** | **100%** | **100%** | 0%     |
| **Replicator** | B1357/S1357  | 0%             | **100%** | 0%             | 0%     |
| **Fredkin**    | B1357/S02468 | 0%             | (0%)           | 0%             | 0%     |

## Analysis

1. **Replicator vs GoL**: The hypothesis holds here. Replicator (more complex update rule) required Degree 4, while GoL was solvable with Degree 2.
2. **Instability of High Degrees**: Degree 16 failed across the board. This is likely due to exploding gradients or training instability associated with high-degree polynomials ($x^{16}$ grows very fast).
3. **Fredkin's Difficulty**: Fredkin remains unsolved. It might require:
   * **Higher Degree** (but with better stabilization/initialization).
   * **More Depth/Width** (as alternative sources of fitting power).

## Next Steps

Investigate **Scaling Depth and Width** for the Fredkin rule, keeping the degree at a stable verifiable level (e.g., Degree 4 or 8).
