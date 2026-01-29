# EE2: Focused n=[1,2,3] Report

## Objective

Evaluate multi-step prediction for n=[1,2,3] using **Periodic Boundary Conditions** with multiple trials to avoid init variance.

## Method

- **Data**: Generated using `CAEngine` with `padding_mode='circular'`.
- **Model**: PolyKAN (Degree 3, Width 4, Depth 2) with `padding_mode='circular'`.
- **Trials**: 3 trials per step count, reporting best result.

## Results

| Steps ($n$) | Best Accuracy (3 Trials) |
| :------------ | :----------------------- |
| **1**   | **100.0%**         |
| 2             | 16.8%                    |
| 3             | 16.4%                    |

## Comparison

| Steps | EE2 (Periodic)   | Exp 02 (Cropped) |
| :---- | :--------------- | :--------------- |
| 1     | **100.0%** | 56.0% (Bad Init) |
| 2     | 16.8%            | 13.4%            |
| 3     | 16.4%            | 17.8%            |

- Periodic boundaries confirmed stable 100% for n=1 across all trials.
- Performance for n=2,3 remains low (~16%), consistent with the complexity hypothesis.
