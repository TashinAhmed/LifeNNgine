# EE3: Inhibitory Signal Analysis Report

To investigate whether the PolyKAN network uses *"inhibitory signals"* (negative weights) to handle overpopulation (neighbor cnt > 3), as hypothesized from previous visualizations.

## Method

- PolyKAN (Degree 3, Width 2, Depth 2) with periodic boundaries.
- 150 epochs to convergence.
- Analysis of final 1 x 1conv weights ($w_0, w_1$) and their correlation with activation shapes.

## Results

### Final Layer Weights

- **Weight Channel 0 ($w_0$)**: **+0.361** (Positive/Excitatory)
- **Weight Channel 1 ($w_1$)**: **+0.362** (Positive/Excitatory)
- **Bias**: **+0.251**

### Analysis

The results from this specific run don't show neg weights. Both channels have pos contributions to the final o/p.
Since the weights are pos, for the network to predict "Death" (Value $\approx 0$) for overpopulation (Neighbors > 3), the **Activation o/p itself must be neg** (or sufficiently low) to drive the sigmoid input below 0 (zero)).

$$
\text{Logit} = w_0 \cdot \text{Act}_0 + w_1 \cdot \text{Act}_1 + \text{bias}
$$


$$
\text{Logit} = 0.36 \cdot \text{Act}_0 + 0.36 \cdot \text{Act}_1 + 0.25
$$

For a dead cell (Logic < 0), we need $\text{Act}_0 + \text{Act}_1 < -0.7$.

### Conclusion

NNs often have multiple valid solns (symmetries) to the same problem.

- **Hypothesis A (Previous Plot)**: High Activation for >3 + Negative Weight = **Inhibitory Architecture**.
- **Hypothesis B (This Run)**: Low/Negative Activation for >3 + Positive Weight = **Suppressive Activation**.

In this experiment (EE3), the network converged to **Solution B**. The activation functions learned to dip into negative values for neighbor counts > 3, allowing positive weights to still result in a "Death" prediction. This shows/proves/confirms (kind of)) the flexibility of the Polynomial Activation function to learn the necessary shape directly.
