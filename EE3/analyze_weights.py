import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F
from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN

def analyze_weights():
    device = torch.device('cpu')
    engine = CAEngine(device=device, padding_mode='circular')
    print("Training PolyKAN for Weight Analysis (Periodic)...")
    X_train, y_train = engine.generate_data(2000, 20, "Game of Life")
    X_val, y_val = engine.generate_data(1000, 20, "Game of Life")
    
    # We use W=2 to isolate two learnable features (e.g. Birth/Survival v/s Overpopulation)
    width = 2
    model = PolyKAN(degree=3, width=width, depth=2, padding_mode='circular').to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    # Train to convergence (100% usually reached within 100 epochs)
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    conv1 = model.layers[0]     # Conv2d(1, width, 3, 3)
    poly1 = model.activations[0]
    final_conv = model.output_conv # Conv2d(width, 1, 1, 1)
    
    # Extract Final 1 x 1 Weights
    # shape: (out_channels, in_channels, 1, 1) -> (1, 2, 1, 1)
    final_weights = final_conv.weight.detach().reshape(-1).numpy()
    final_bias = final_conv.bias.detach().item()
    print(f"Final Weights: {final_weights}")
    print(f"Final Bias: {final_bias}")
    # GT Neighbor Count
    moore_kernel = torch.tensor([[1,1,1],[1,0,1],[1,1,1]], dtype=torch.float32).view(1,1,3,3)
    with torch.no_grad():
        X_val_padded = F.pad(X_val, (1,1,1,1), mode='circular')
        neighbors = F.conv2d(X_val_padded, moore_kernel, padding=0)
        neighbors_flat = neighbors.reshape(-1).numpy()
    # Get Model Filter o/p
    with torch.no_grad():
        filter_out = conv1(X_val)
        act_out = poly1(filter_out)
    # Plot for each channel
    for i in range(width):
        f_out = filter_out[:, i, :, :].reshape(-1).numpy()
        a_out = act_out[:, i, :, :].reshape(-1).numpy()
        w = final_weights[i]
        idx = np.random.choice(len(f_out), 2000, replace=False)
        plt.figure(figsize=(10, 6))
        # Scatter: Filter Output vs Activation Output
        sc = plt.scatter(f_out[idx], a_out[idx], c=neighbors_flat[idx], cmap='viridis', alpha=0.6)
        cbar = plt.colorbar(sc, label='True Neighbor Count')
        cbar.set_ticks(np.arange(0, 9))
        # Learned polynomial curve
        coeffs = poly1.coeffs[:, i].detach().numpy()
        x_min, x_max = f_out.min(), f_out.max()
        x_range = np.linspace(x_min, x_max, 100)
        y_curve = np.zeros_like(x_range)
        for d, c in enumerate(coeffs):
            y_curve += c * (x_range ** d)
        plt.plot(x_range, y_curve, 'r-', linewidth=3, label='Learned Polynomial')
        # Annotate with the final weight
        title_color = 'red' if w < 0 else 'green'
        signal_type = "INHIBITORY (Death)" if w < 0 else "EXCITATORY (Life)"
        plt.xlabel(f'Filter {i} Output (Linear)')
        plt.ylabel(f'Activation {i} Output (Polynomial)')
        plt.title(f'Channel {i}: Weight w_{i} = {w:.3f} [{signal_type}]\nColor = True Neighbor Count', fontsize=14, color=title_color)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'EE3/channel_{i}_weight_analysis.png')
        plt.close()
    # Save text report
    with open('EE3/weights_report.txt', 'w') as f:
        f.write(f"Final Weights: {final_weights}\n")
        f.write(f"Final Bias: {final_bias}\n")
    print("Weight analysis complete.")

if __name__ == "__main__":
    analyze_weights()
