import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN
from comparative_experiments.train_heatmap import train_one_model

def visualize_model(rule_name, degree, width=2, depth=2, device='cpu'):
    print(f"Training model for {rule_name} (Degree {degree})...")
    
    engine = CAEngine(device=device)
    X_train, y_train = engine.generate_data(2000, 20, rule_name)
    X_val, y_val = engine.generate_data(500, 20, rule_name)
    
    model = PolyKAN(degree=degree, width=width, depth=depth).to(device)
    
    # Train
    acc = train_one_model(model, X_train, y_train, X_val, y_val, device, epochs=300)
    print(f"Final Accuracy: {acc*100:.1f}%")
    
    # Analyze First Layer
    # Layer 0 is Conv2d(1, width, 3x3)
    conv1 = model.layers[0]
    weights = conv1.weight.detach().cpu().numpy() # (width, 1, 3, 3)
    
    # Plot Kernels
    fig, axes = plt.subplots(1, width, figsize=(width*3, 3))
    if width == 1: axes = [axes]
    
    for i, ax in enumerate(axes):
        w = weights[i, 0]
        ax.imshow(w, cmap='coolwarm')
        ax.set_title(f"Filter {i}")
        # Annotate values
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{w[r,c]:.1f}", ha='center', va='center', color='black')
                
    plt.suptitle(f"{rule_name} - Layer 1 Kernels")
    plt.tight_layout()
    os.makedirs('comparative_experiments/plots', exist_ok=True)
    plt.savefig(f'comparative_experiments/plots/{rule_name}_kernels.png')
    plt.close()
    
    # Analyze Activation
    # Plot phi(x) for x in [-1, 10] (assuming neighbor sums are roughly in this range if weights are ~1)
    
    # We need to know the effective range of inputs to the activation.
    # We can pass some data and record it, or just scan a reasonable range.
    # scanning [-2, 10]
    x_range = torch.linspace(-2, 10, 100).to(device)
    poly_layer = model.activations[0]
    # poly_layer expects (B, C, H, W). We can just pass (N, C, 1, 1)
    # Input to poly: (100, width, 1, 1)
    x_input = x_range.view(100, 1, 1, 1).repeat(1, width, 1, 1)
    
    with torch.no_grad():
        y_output = poly_layer(x_input).cpu().numpy() # (100, width, 1, 1)
        
    x_vals = x_range.cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    for i in range(width):
        plt.plot(x_vals, y_output[:, i, 0, 0], label=f"Channel {i}")
        
    plt.title(f"{rule_name} - Learned Activation (Layer 1)")
    plt.xlabel("Input (Neighbor Sum)")
    plt.ylabel("Activation Output")
    plt.grid(True)
    plt.legend()
    plt.savefig(f'comparative_experiments/plots/{rule_name}_activation.png')
    plt.close()
    print(f"Plots saved to comparative_experiments/plots/{rule_name}_*.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rule', type=str, default='Game of Life')
    parser.add_argument('--degree', type=int, default=3)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    visualize_model(args.rule, args.degree, device=device)
