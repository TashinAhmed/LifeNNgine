import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F
from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN

def analyze_interpretability():
    device = torch.device('cpu')
    engine = CAEngine(device=device, padding_mode='circular')
    print("Training PolyKAN for Interpretability Analysis (Periodic)...")
    X_train, y_train = engine.generate_data(2000, 20, "Game of Life")
    X_val, y_val = engine.generate_data(1000, 20, "Game of Life")
    width = 2
    model = PolyKAN(degree=3, width=width, depth=2, padding_mode='circular').to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    # Extract Internals
    model.eval()
    conv1 = model.layers[0]
    poly1 = model.activations[0]
    # Analyze Filters
    kernels = conv1.weight.detach().numpy()
    fig, axes = plt.subplots(1, width, figsize=(width*4, 3))
    if width == 1: axes = [axes]
    for i, ax in enumerate(axes):
        k = kernels[i, 0]
        sns.heatmap(k, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
        ax.set_title(f"Learned Filter {i}")
    plt.savefig('EE2/interpretability/filters.png')
    plt.close()
    
    # Ground Truth Neighbor Count
    moore_kernel = torch.tensor([[1,1,1],[1,0,1],[1,1,1]], dtype=torch.float32).view(1,1,3,3)
    with torch.no_grad():
        X_val_padded = F.pad(X_val, (1,1,1,1), mode='circular')
        neighbors = F.conv2d(X_val_padded, moore_kernel, padding=0)
        neighbors_flat = neighbors.reshape(-1).numpy()
        center_flat = X_val.reshape(-1).numpy()
    
    # Get Model Filter Outputs
    with torch.no_grad():
        filter_out = conv1(X_val)
        act_out = poly1(filter_out)
        
    # Plot for each channel
    for i in range(width):
        f_out = filter_out[:, i, :, :].reshape(-1).numpy()
        a_out = act_out[:, i, :, :].reshape(-1).numpy()
        
        idx = np.random.choice(len(f_out), 2000, replace=False)
        
        plt.figure(figsize=(10, 6))
        
        sc = plt.scatter(f_out[idx], a_out[idx], c=neighbors_flat[idx], cmap='viridis', alpha=0.6)
        plt.colorbar(sc, label='True Neighbor Count')
        
        # Learned polynomial curve
        coeffs = poly1.coeffs[:, i].detach().numpy()
        x_min, x_max = f_out.min(), f_out.max()
        x_range = np.linspace(x_min, x_max, 100)
        y_curve = np.zeros_like(x_range)
        for d, c in enumerate(coeffs):
            y_curve += c * (x_range ** d)
            
        plt.plot(x_range, y_curve, 'r-', linewidth=2, label='Learned Polynomial')
        
        plt.xlabel(f'Filter {i} Output')
        plt.ylabel(f'Activation {i} Output')
        plt.title(f'Channel {i}: Interpretation Analysis (Periodic)\nColor = True Neighbor Count')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'EE2/interpretability/channel_{i}_analysis.png')
        plt.close()
        
    print("Interpretability analysis complete.")

if __name__ == "__main__":
    analyze_interpretability()
