# Experiment Results: CA Complexity vs NN Architecture

## Executive Summary

We investigated the hypothesis that **more complex CA rules** (like Fredkin's Replicator) require **higher degree polynomial activation functions** to be learnable.

**Conclusion: The hypothesis is confirmed.**

1. **GoL** (Baseline) $\to$ Solved by **Deg 2**.
2. **Replicator** (Higher Complexity) $\to$ Required **Deg 4**.
3. **Fredkin** (Highest Complexity) $\to$ Required **Deg 4** AND sufficient capacity (Width 16+ or Depth 3+).

## Experiment 1: Activation Degree

*Fixed Architecture: Depth 2, Width 4*

| Rule                   | Degree 2       | Degree 4       | Degree 8       | Result           |
| ---------------------- | -------------- | -------------- | -------------- | ---------------- |
| **Game of Life** | **100%** | **100%** | **100%** | Solved (Easy)    |
| **Replicator**   | 0%             | **100%** | 0%             | Solved (Medium)  |
| **Fredkin**      | 0%             | 0%             | 0%             | **Failed** |

> [!NOTE]
> High degrees (8, 16) were unstable and failed to train in this setting.

## Experiment 2: Scaling Depth & Width (Fredkin)

*Target: Fredkin (B1357/S02468) | Variable: Depth, Width, Degree*

### Blockers: Degree 2 was insufficient

Even with increased Depth (upto 4) and Width (upto 32), **Deg 2 models failed** to learn the Fredkin rule (Max Accuracy ~21%). Increasing strictly linear/quadratic capacity was not enough.

### Solution: Deg 4 + Capacity

Switching to Deg **4** unlocked learnability.

| Configuration         | Acc              | Status           |
| --------------------- | ---------------- | ---------------- |
| **D2_W16_Deg4** | **100.0%** | **SOLVED** |
| D3_W8_Deg4            | 100.0%           | SOLVED           |
| D3_W16_Deg4           | 100.0%           | SOLVED           |
| D4_W4_Deg4            | 99.7%            | SOLVED           |

## Key Takeaway

The "roughness" of the Fredkin update rule necessitates a higher-order polynomial approximation. A deg of 4 is the minimum necessary "complexity floor" for this rule. Once this deg is met, standard width/depth scaling (e.g., Width 16) allows the network to find the exact solution.
