import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN


# Simple CNN with ReLU matching PolyKAN's setup (2 layers)
class ReLUNet(nn.Module):
    def __init__(self, width=16, depth=2):
        super().__init__()
        self.layers = nn.ModuleList()
        in_c = 1
        for i in range(depth):
            if i == 0:
                self.layers.append(
                    nn.Conv2d(in_c, width, 3, padding=1, padding_mode="circular")
                )
            else:
                self.layers.append(nn.Conv2d(in_c, width, 1, padding=0))
            self.layers.append(nn.ReLU())
            in_c = width

        self.output_conv = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.output_conv(x))


def generate_fixed_density_data(engine, num_samples, size, rule, density):
    """Generate (X, y) with a specific fixed density."""
    X = torch.zeros((num_samples, 1, size, size))
    X = (torch.rand_like(X) < density).float()
    rule_b, rule_s = engine._get_rule_params(rule)
    with torch.no_grad():
        y = engine.step(X, rule_b, rule_s)
    return X, y


def generate_uniform_density_data(engine, num_samples, size, rule):
    """Generate (X, y) with densities uniform uniformly distributed [0.1, 0.9]."""
    X = torch.zeros((num_samples, 1, size, size))
    densities = torch.rand(num_samples, 1, 1, 1) * 0.8 + 0.1
    X = (torch.rand_like(X) < densities).float()
    rule_b, rule_s = engine._get_rule_params(rule)
    with torch.no_grad():
        y = engine.step(X, rule_b, rule_s)
    return X, y


def train_model(model_class, model_kwargs, X_train, y_train, epochs=200):
    model = model_class(**model_kwargs)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    return model


def exact_match_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = (model(X) > 0.5).float()
        acc = (preds == y).all(dim=(1, 2, 3)).float().mean().item()
    return acc


def run_density_experiment():
    engine = CAEngine(padding_mode="circular")
    rule = "Game of Life"
    grid_size = 20
    train_samples = 2000
    val_samples = 1000

    densities = [0.1, 0.5, 0.9]
    trials = 5

    # The true test of rule learning
    X_val_diverse, y_val_diverse = generate_uniform_density_data(
        engine, val_samples, grid_size, rule
    )

    results = {
        "PolyKAN": {"in_dist": [], "cross_dist": []},
        "ReLU": {"in_dist": [], "cross_dist": []},
    }

    for d in densities:
        print(f"\n--- Training on Fixed Density: {d} ---")

        poly_in_accs, poly_cross_accs = [], []
        relu_in_accs, relu_cross_accs = [], []

        for trial in range(trials):
            # Generate new training and local valid data each trial
            X_train, y_train = generate_fixed_density_data(
                engine, train_samples, grid_size, rule, d
            )
            X_val_local, y_val_local = generate_fixed_density_data(
                engine, val_samples, grid_size, rule, d
            )

            # Train PolyKAN
            poly = train_model(
                PolyKAN,
                {"degree": 3, "width": 4, "depth": 2, "padding_mode": "circular"},
                X_train,
                y_train,
                epochs=150,
            )
            p_in = exact_match_accuracy(poly, X_val_local, y_val_local)
            p_cross = exact_match_accuracy(poly, X_val_diverse, y_val_diverse)
            poly_in_accs.append(p_in)
            poly_cross_accs.append(p_cross)

            # Train ReLU (Width 16 to ensure enough capacity normally)
            relu = train_model(
                ReLUNet, {"width": 16, "depth": 2}, X_train, y_train, epochs=150
            )
            r_in = exact_match_accuracy(relu, X_val_local, y_val_local)
            r_cross = exact_match_accuracy(relu, X_val_diverse, y_val_diverse)
            relu_in_accs.append(r_in)
            relu_cross_accs.append(r_cross)

            print(
                f"  Trial {trial + 1} | PolyKAN: Local {p_in:.2f}, Diverse {p_cross:.2f} | ReLU: Local {r_in:.2f}, Diverse {r_cross:.2f}"
            )

        # Averages
        results["PolyKAN"]["in_dist"].append(np.mean(poly_in_accs))
        results["PolyKAN"]["cross_dist"].append(np.mean(poly_cross_accs))
        results["ReLU"]["in_dist"].append(np.mean(relu_in_accs))
        results["ReLU"]["cross_dist"].append(np.mean(relu_cross_accs))

    # Plotting
    x = np.arange(len(densities))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # In-Distribution
    ax1.bar(x - width / 2, results["PolyKAN"]["in_dist"], width, label="PolyKAN")
    ax1.bar(x + width / 2, results["ReLU"]["in_dist"], width, label="ReLU")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("In-Distribution Accuracy \n(Tested on same density)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(densities)
    ax1.set_xlabel("Training Density")
    ax1.legend()
    ax1.set_ylim([0, 1.05])

    # Cross-Distribution
    ax2.bar(x - width / 2, results["PolyKAN"]["cross_dist"], width, label="PolyKAN")
    ax2.bar(x + width / 2, results["ReLU"]["cross_dist"], width, label="ReLU")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(
        "Cross-Distribution Generalization \n(Tested on diverse densities 0.1-0.9)"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(densities)
    ax2.set_xlabel("Training Density")
    ax2.legend()
    ax2.set_ylim([0, 1.05])

    plt.savefig('rebuttal/EE4_1/density_gap.png')
    plt.close()
    print("\nExperiment complete. Plot saved to rebuttal/EE4_1/density_gap.png")


if __name__ == "__main__":
    run_density_experiment()
