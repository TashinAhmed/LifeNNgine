# KAN Degree 2 (Quadratic) Experiment Results

 "PReLU is 1-degree KAN", so tested a **2-degree KAN** (Quadratic) activation function on the minimal architecture.

## Experiment Setup

- **Architecture**: Conv2D(2, 3x3) -> **Quadratic** -> Conv2D(1, 1x1) -> **Quadratic** -> Conv2D(1, 1x1) -> Sigmoid
- **Quadratic Activation**: $f(x) = w_2 x^2 + w_1 x + w_0$ (learnable coefficients per channel).
- **Trials**: 10

## Results

| Variation                      | Success Rate (100% Acc) | Average Accuracy | Best Accuracy |
| :----------------------------- | :---------------------- | :--------------- | :------------ |
| **Quadratic (Degree 2)** | **9/10 (90%)**    | 90.00%           | 100.00%       |

## Comparison

| Activation                     | Success Rate  |
| :----------------------------- | :------------ |
| Baseline (ReLU)                | 10%           |
| PReLU (Degree 1)               | 20%           |
| **Quadratic (Degree 2)** | **90%** |

## Analysis

1. Moving to a quadratic activation function improved the reliability of the minimal architecture from 10-20% to **90%**.
2. **Why?**: (tashinahmed assumptions)) The GoL rules involve counting neighbors ("+" summation) and checking if the sum is 2 or 3. This non-linear logic (specifically the "2 or 3" interval check) is much easier to approximate with a quadratic function (a parabola can easily isolate an interval) than with piecewise linear functions (ReLU/PReLU) which require multiple neurons to construct a "bump".
3. **Conclusion**: The "hardness" of the GoL for minimal networks is largely due to the inefficiency of ReLU in representing the specific non-linearities required. A quadratic activation (2-degree KAN) solves this efficiently.
