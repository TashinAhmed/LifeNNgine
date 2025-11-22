# KAN Experiment Results

Implemented a custom `BSplineActivation` (PolyKAN) layer to mimic the KAN architecture within a CNN.

## Experiment Setup

- **Architecture**: Conv2D(2, 3x3) -> **PolyKAN** -> Conv2D(1, 1x1) -> **PolyKAN** -> Conv2D(1, 1x1) -> Sigmoid
- **PolyKAN Layer**: Learnable polynomial basis expansion + SiLU residual.
- **Trials**: 10

## Results

| Variation         | Success Rate (100% Acc) | Average Accuracy | Best Accuracy |
| :---------------- | :---------------------- | :--------------- | :------------ |
| **PolyKAN** | **0/10 (0%)**     | 0.00%            | 0.00%         |

## Analysis

1. **Failure**: The KAN-based architecture failed to converge in all 10 trials.
2. **Possible Reasons**:
   - **Over-parameterization**: Even though KANs are "efficient", adding learnable coefficients for every channel might have made the optimization landscape too complex for this extremely constrained 2-filter bottleneck.
   - **Initialization**: KANs are known to be sensitive to initialization and grid range. The fixed grid/basis might not have matched the data distribution of the Game of Life states.
   - **PReLU was better**: The simpler PReLU (which is a very basic 1-degree KAN) performed better (20% success) than this more complex polynomial KAN.

## Conclusion

While KANs are powerful, simply swapping them into this specific, highly unstable minimal architecture did not magically solve the problem. The "PReLU" experiment (which is a subset of KAN) showed promise, but a full polynomial expansion seems to have introduced more instability or optimization difficulties in this specific constrained setting.
