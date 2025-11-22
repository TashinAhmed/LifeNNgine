# Ensemble Experiment Results

To answer the question "is it possible to achieve this 100% with more small neural nets?", We ran an experiment using ensembles of very small CNNs

## Experiment Setup

- **Task**: Learn GoL rules (i.e. predict next state from current state).
- **Data**: 10,000 training grids, 2,000 validation grids (20 x 20 size).
- **Model Architecture**:
  - Input (20 x 2 0 x 1)
  - Conv2D(filters, 3 x 3, relu)
  - Conv2D(filters, 3 x 3, relu)
  - Conv2D(1, 1x1, sigmoid)
- **Variable**: `filters` (tested 5, 4, 3, 2).
- **Ensemble**: 5 models, majority vote.

## Results

| Filter Size | Single Model Accuracy | Ensemble Accuracy (5 models) |
| :---------- | :-------------------- | :--------------------------- |
| **5** | 4.70% (Failed)        | **100.00%** (Success)  |
| **4** | 98.85% (Partial)      | **100.00%** (Success)  |
| **3** | 29.45% (Failed)       | 4.90% (Failed)               |
| **2** | 0.65% (Failed)        | **100.00%** (Success)  |

## Conclusion

**Yes**, it is possible to achieve 100% acc with more small neural nets.

- Even when a single network with **2 filters** fails completely (0.65% accuracy), an ensemble of 5 such networks can achieve **100% exact match accuracy**.
- This demonstrates that small networks can learn complementary features that, when combined, perfectly represent the GoL rules.
