# Comparative CA Experiments 01

This dir contains experiments for comparing the learnability of different CA rules (GoL, Replicator, Fredkin, Morley) using NNs with PolyKAN.

## Structure

* `ca_rules.py`: PyTorch implementation of the CA rules.
* `models.py`: PyTorch implementation of the PolyKAN model.
* `train_heatmap.py`: Main script for grid search (Rules x Degrees x Widths x Depths).
* `visualize_functions.py`: Script to visualize learned activation functions and kernels.
* `generate_report.py`: Script to generate `results.md` and heatmaps from the training logs.
* `heatmap_results.jsonl`: Log file containing results from `train_heatmap.py`.

## Installation

Install:

```bash

pip install torch torchvision torchaudio tqdm matplotlib pandas seaborn
```

## Running Experiments

### 1. Grid Search (Heatmaps)

To run the full grid search:

```bash
PYTHONPATH=. python comparative_experiments/train_heatmap.py
```

* This will write results line-by-line to `comparative_experiments/heatmap_results.jsonl`.
* Note: This can take a long time. Use `--fast` for a quick debugging run.

### 2. Visualization (Learned Functions)

To inspect a specific rule and degree:

```bash
PYTHONPATH=. python comparative_experiments/visualize_functions.py --rule "Game of Life" --degree 3
PYTHONPATH=. python comparative_experiments/visualize_functions.py --rule "Fredkin" --degree 3
```

* Supported rules: "Game of Life", "Replicator", "Fredkin", "Morley".
* Plots are saved to `comparative_experiments/plots/`.

### 3. Generate Report

To parse the JSONL logs and update `results.md`:

```bash
PYTHONPATH=. python comparative_experiments/generate_report.py
```

## Troubleshooting

* **Missing Dependencies**: If you see `ModuleNotFoundError`, make sure you installed the requirements.
* **Module Path**: Always run with `PYTHONPATH=.` from the `rebuttal/` root to ensure imports work correctly.
