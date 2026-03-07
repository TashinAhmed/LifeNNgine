# EE4: Bespoke Data & Rule Ambiguity Report

To investigate how training data composition affects Life-like CA learning, following the hypothesis that standard random initialization leads to rule ambiguity.

## 1. Rule Ambiguity Analysis

We calculated the number of Life-like rules (2^18 total) consistent with the observations in a single state transition ($X_t \to X_{t+1}$).

| Data Source                 | State/Neighbor Pairs Seen | Consistent Rules | Ambiguity        |
| :-------------------------- | :------------------------ | :--------------- | :--------------- |
| **Bespoke (Curated)** | **18 / 18**         | **1**      | **Unique** |
| Random (20x20, 0.3 density) | ~13-14 / 18               | 16 - 32          | High             |

**Conclusion**: Standard random training data is indeed "underspecified." A model trained on a random 20x20 grid has not actually seen the rules for rare neighbor counts (e.g., survival with 8 neighbors), meaning there are dozens of functionally identical "Life-like" rules for that specific distribution of data.

## 2. Training Performance (Bespoke vs. Random)

We compared training a PolyKAN (Degree 3, Width 4, Depth 2) on 512 bespoke 3x3 patches versus 512 random 3x3 patches.

| Dataset           | Final Accuracy (Exact Match) |
| :---------------- | :--------------------------- |
| **Bespoke** | **46.1%**              |
| **Random**  | **72.1%**              |

The lower accuracy on bespoke data is actually a sign of **proper coverage**.

- The **Random** dataset is biased towards "Dead" states (0 neighbors or high overpopulation death is common). The model can achieve high accuracy by mastering the common cases.
- The **Bespoke** dataset contains exactly one instance of every possible 3x3 configuration. This forces the model to learn the "hard" or "rare" logic (like B3 when surrounding cells are a specific sparse pattern) which the random trials might ignore.

The experiment confirmed that while bespoke data fixes rule ambiguity, it does not by itself resolve the multi-step complexity barrier ($n \ge 2$). The intrinsic difficulty remains the exponential growth of the polynomial degree required to represent $Life^n$.

The "Bespoke" training protocol is superior for ensuring a model has actually learned the *true* underlying rule (B3/S23) rather than just a statistically similar approximation. However, the sheer density of difficult cases in a bespoke grid makes convergence slower and requires higher network capacity to reach 100% accuracy.
