import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN

def analyze_interpretability():
    device = torch.device('cpu')
    engine = CAEngine(device=device)
    print("Training PolyKAN for Interpretability Analysis...")
    X_train, y_train = engine.generate_data(2000, 20, "Game of Life")
    X_val, y_val = engine.generate_data(1000, 20, "Game of Life")
    
    # Future References:
    # Width 1 to force it to learn the exact rule in one channel if possible?
    # Or Width 2 to allow splitting (Center, Neighbors).
    # Previous success was Width 2. So sticking to Width 2 atm.
    width = 2
    model = PolyKAN(degree=3, width=width, depth=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    conv1 = model.layers[0]
    poly1 = model.activations[0]
    kernels = conv1.weight.detach().numpy() # (width, 1, 3, 3)
    fig, axes = plt.subplots(1, width, figsize=(width*4, 3))
    if width == 1: axes = [axes]
    for i, ax in enumerate(axes):
        k = kernels[i, 0]
        sns.heatmap(k, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
        ax.set_title(f"Learned Filter {i}")
    plt.savefig('extended_experiments/interpretability/filters.png')
    plt.close()
    moore_kernel = torch.tensor([[1,1,1],[1,0,1],[1,1,1]], dtype=torch.float32).view(1,1,3,3)
    with torch.no_grad():
        neighbors = torch.nn.functional.conv2d(X_val, moore_kernel, padding=1)
        neighbors_flat = neighbors.view(-1).numpy()
        center_flat = X_val.view(-1).numpy()
    with torch.no_grad():
        filter_out = conv1(X_val)
        act_out = poly1(filter_out)
    for i in range(width):
        f_out = filter_out[:, i, :, :].reshape(-1).numpy()
        a_out = act_out[:, i, :, :].reshape(-1).numpy()
        # Subsample for plot clarity (1000 points)
        idx = np.random.choice(len(f_out), 2000, replace=False)
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(f_out[idx], a_out[idx], c=neighbors_flat[idx], cmap='viridis', alpha=0.6)
        plt.colorbar(sc, label='True Neighbor Count')
        x_min, x_max = f_out.min(), f_out.max()
        x_range = torch.linspace(x_min, x_max, 100).view(100, 1, 1, 1).to(device)
        coeffs = poly1.coeffs[:, i].detach().numpy() # (deg+1)
        y_curve = np.zeros_like(x_range.view(-1).numpy())
        for d, c in enumerate(coeffs):
            y_curve += c * (x_range.view(-1).numpy() ** d)
        plt.plot(x_range.view(-1).numpy(), y_curve, 'r-', linewidth=2, label='Learned Polynomial')
        plt.xlabel(f'Filter {i} Output')
        plt.ylabel(f'Activation {i} Output')
        plt.title(f'Channel {i}: Interpretation Analysis\nColor = True Neighbor Count')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'extended_experiments/interpretability/channel_{i}_analysis.png')
        plt.close()
    print("Interpretability analysis complete.")

if __name__ == "__main__":
    analyze_interpretability()
