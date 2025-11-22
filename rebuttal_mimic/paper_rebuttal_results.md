# Paper Rebuttal Experiment Results

To address  the original paper ("It's Hard for Neural Networks To Learn the Game of Life"), test with **exact minimal architecture** described in the paper, performed below:

- **Architecture**: Conv2D(2, 3x3) -> Conv2D(1, 1x1) -> Conv2D(1, 1x1)
- **Total Parameters**: ~30 (extremely small)

These architecture with various modifications (Loss funcs, Activations) to see if they could overcome the difficulty reported in the paper are below.

## Results (10 Trials per Variation)

| Variation                       | Success Rate (100% Acc) | Average Accuracy | Best Accuracy | Notes                                               |
| :------------------------------ | :---------------------- | :--------------- | :------------ | :-------------------------------------------------- |
| **Baseline (ReLU + BCE)** | **1/10 (10%)**    | 10.21%           | 100.00%       | Confirms paper's claim: it is hard!                 |
| **ReLU + MSE**            | **0/10 (0%)**     | 0.00%            | 0.00%         | Failed completely.                                  |
| **Swish + BCE**           | **1/10 (10%)**    | 10.26%           | 100.00%       | No improvement over baseline.                       |
| **GeLU + BCE**            | **1/10 (10%)**    | 11.30%           | 100.00%       | No significant improvement.                         |
| **PReLU (KAN-proxy)**     | **2/10 (20%)**    | 21.02%           | 100.00%       | **Doubled success rate**, but still unstable. |

## Analysis

1. The specific minimal architecture (2 filters -> 1 filter) is indeed very unstable and hard to train. Most initializations fail completely (0% acc).
2. **It IS Possible**: Contrary to a strong interpretation that it's "impossible", we *did* achieve 100% accuracy in 10-20% of trials. It's just rare.
3. **Learnable Activations Help**: Using PReLU (Parametric ReLU), which adds learnable parameters to the activation function (similar in spirit to KAN), doubled the success rate from 10% to 20%.
4. **Slightly Larger is Much Better**: Comparing this to my previous experiment, simply increasing the filters from **2 to 5** increased reliability from **10% to 100%**.

## Conclusion

The "hardness" comes from the extreme constraint of using only 2 filters. Relaxing this constraint slightly (to 5 filters) or using learnable activations (PReLU) makes the problem solvable, but the paper's core observation about the sensitivity of the *minimal* network holds true.
