# EE4_1: Cross-Density Validation Report

To demonstrate that validating CA models on datasets with the same initial density as the training data can yield artificially high "solved" metrics (false positives), hiding the model's failure to learn the true underlying rule.

## Experiment Design

Trained $m=1, n=1$ PolyKAN and ReLU models on specific initial grid densities ($d \in \{0.1, 0.5, 0.9\}$) and evaluated them on two separate validation sets:

1. **Local (In-Distribution):** A validation set possessing the exact same density as the training set.
2. **Diverse (Generalization):** A validation set where density is uniformly distributed between 0.1 and 0.9.

## Results Summary (Averaged over 5 trials)

### **Training Density: 0.1** (Highly Sparse)

- **PolyKAN:** Local Acc = 91%, Diverse Acc = 80% (Failed 1 trial, succeeded 4)
- **ReLU:** Local Acc = 46%, Diverse Acc = **20%**
- *Observation:* When ReLU "learns", its local accuracy at $d=0.1$ is roughly double its actual generalization accuracy. The model exploits the class imbalance (mostly dead cells).

### **Training Density: 0.5** (Balanced)

- **PolyKAN:** Local Acc = 40%, Diverse Acc = 53%
- **ReLU:** Local Acc = 80%, Diverse Acc = 84%
- *Observation:* This represents the hardest regime to learn because there is no dominant background class to exploit. Performance relies on actually learning the B3/S23 rule.

### **Training Density: 0.9** (Highly Dense)

- **PolyKAN:** Local Acc = 86%, Diverse Acc = **5%**
- **ReLU:** Local Acc = 86%, Diverse Acc = **5%**
- *Observation:* The most glaring example of false positives. Both models achieved ~86% accuracy on their in-distribution local validation sets. However, when tested on diverse geometries, their accuracy plummeted to 5%. They simply learned to predict "everything dies from overpopulation" mapping perfectly to $d=0.9$ dynamics but failing entirely on standard rules.

The results confirm the hypothesis: **Testing a CA model on the same biased density distribution used for training guarantees false positives.**
The high local accuracy at extreme densities ($d=0.1, 0.9$) is a statistical artifact of class imbalance (e.g., $X_{t+1}$ being overwhelmingly composed of empty cells). To rigorously claim a model has "learned the rule," it must be validated across a uniform distribution of densities, as failure to do so masks catastrophic failures in generalization.
