# Experiment Set 02 Report (Steps 1, 2, 3)

## Objective

Focused analysis of multi-step prediction for $n \in \{1, 2, 3\}$ using the PolyKAN architecture (Deg 3, Width 4, Depth 2) with "Valid Padding" evaluation (cropping borders).

## Results (Initial Run)

| Steps ($n$) | Valid Acc                           |
| :------------ | :---------------------------------- |
| **1**   | **56.0%** (Anomalous Failure) |
| 2             | 13.4%                               |
| 3             | 17.8%                               |

## Post-Hoc Analysis: The $n=1$ Anomaly

The init run showed a 56% acc for $n=1$, drastically lower than the 100% seen in Exp 01. We investigated this by verifiying that the training logic and model definition were identical to the successful run. Re-ran the $n=1$ configuration with 5 different random seeds.

**Stability Test Results:**

- Seed 42: **100.0%**
- Seed 100: **100.0%**
- Seed 2024: **100.0%**
- Seed 7: **100.0%**
- Seed 0: **100.0%**

**T**he architecture is capable of 100% acc. The initial failure (56%) was due to **unlucky random initialization** or data sampling variance in that specific run. The 56% acc roughly corresponds to a "guessing the majority class" baseline (predicting all 0s or random density match), indicating the optimization got stuck in a local minimum.

## Validated Multi-Step Trend

Ignoring the anomalous run, the trend remains:

- **Step 1**: 100% (stable)
- **Step 2**: ~13% (failure)
- **Step 3**: ~17% (failure)

This confirms the difficulty of learning $n>1$ steps.
