# GoL_NN

rebuttal of It's hard for NNs to learn Game of Life

## Rebuttal Walkthrough

## Overview

We have tried to perform a rebuttal to the paper "It's Hard for Neural Networks To Learn the Game of Life". We tried to demonstrate that a standard CNN can learn the Game of Life rules with **100% exact match accuracy** (predicting every single cell correctly in a grid).

## Methodology

1. **Data Generation** : We generated random grids with varying densities (0.1 to 0.9) to ensure diversity. We used zero-padding for the Game of Life simulation to match the CNN's padding behavior.
2. **Model Architecture** : We used a simple CNN with 2 hidden layers (64 filters each, 3x3 kernels) and `ReLU` activation. The output layer uses `Sigmoid` activation.
3. **Training** : We trained for 100 epochs using the Adam optimizer.

## Results

* **Pixel-wise Accuracy** : 100.00%
* **Exact Match Accuracy** : 100.00%

### Minimal Model Experiment

We also tested a "minimal" architecture with only **10 filters** per layer (compared to 64 in the robust model).

* **Minimal Model Exact Match Accuracy** : 100.00%

This proves that neural networks *can* easily learn the Game of Life given appropriate data generation and training parameters, even with constrained architectures.

## How to Run

1. Navigate to the `rebuttal` directory.
2. Run `bash setup_env.sh` to create the virtual environment and install dependencies.
3. Open the notebook: `jupyter notebook rebuttal_gol.ipynb`.
4. Select the `Python (Rebuttal)` kernel.
