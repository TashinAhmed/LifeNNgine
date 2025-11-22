# Single Small Model Experiment Results

To verify if **single** small neural networks can solve GoL, a rigorous experiment testing various filter sizes with multiple trials to account for initialization variance have been performed.

## Experiment Setup

- **Data**: 10,000 training grids, 2,000 validation grids.
- **Model**: 2-layer CNN (3x3 filters) + 1x1 output.
- **Trials**: 5 independent training runs per filter size.
- **Metric**: Exact Match Accuracy (100% correct pixels per grid).

## Results

| Filter Size  | Success Rate (100% Acc) | Average Accuracy | Best Accuracy | Notes                            |
| :----------- | :---------------------- | :--------------- | :------------ | :------------------------------- |
| **16** | 5/5 (100%)              | 100.00%          | 100.00%       | Perfect reliability.             |
| **12** | 4/5 (80%)               | 84.27%           | 100.00%       | 1 failure (21.35%).              |
| **10** | 5/5 (100%)              | 100.00%          | 100.00%       | Perfect reliability.             |
| **8**  | 5/5 (100%)              | 100.00%          | 100.00%       | Perfect reliability.             |
| **6**  | 5/5 (100%)              | 100.00%          | 100.00%       | Perfect reliability.             |
| **5**  | 5/5 (100%)              | 100.00%          | 100.00%       | **Minimal reliable size.** |
| **4**  | 4/5 (80%)               | 80.96%           | 100.00%       | 1 failure (4.80%).               |

## Conclusion

- A network with as few as **5 filters** per layer is sufficient to reliably achieve **100% accuracy** in every trial.
- Even **4 filters** can achieve 100% accuracy, though it is sensitive to initialization* (failed in 1 out of 5 trials).
- This definitively disproves the notion that "large" or "complex" networks are required. The GOL rules can be perfectly encoded in a very compact CNN.
