import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN

def get_multistep_data(engine, num_samples, grid_size, steps, rule_name="Game of Life"):
    rule_b, rule_s = engine._get_rule_params(rule_name)
    X = torch.zeros((num_samples, 1, grid_size, grid_size), device=engine.device)
    densities = torch.rand(num_samples, 1, 1, 1, device=engine.device) * 0.8 + 0.1
    X = (torch.rand_like(X) < densities).float()
    current_state = X
    with torch.no_grad():
        for _ in range(steps):
            current_state = engine.step(current_state, rule_b, rule_s)
    return X, current_state

def train_multistep(steps, device='cpu', epochs=100):
    print(f"Training for {steps}-step prediction...")
    engine = CAEngine(device=device)
    # Larger grid to allow crop
    # If we want 20 x 20 valid o/p, input must be bigger? 
    # Or just evaluate on inner part of 20 x 20.
    grid_size = 24
    X_train, y_train = get_multistep_data(engine, 2000, grid_size, steps)
    X_val, y_val = get_multistep_data(engine, 500, grid_size, steps)
    # We use depth=2, width=4 to be closer to successful config
    model = PolyKAN(degree=3, width=4, depth=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                preds = (val_outputs > 0.5).float()

                ############################################
                # CROP Evaluation
                # Valid region is [steps:-steps]
                # If steps=1, crop 1 pixel border.
                # If steps >= grid_size//2, no valid region!
                ############################################

                crop = steps
                if crop >= grid_size // 2:
                    print(f"Warning: steps {steps} too large for grid {grid_size}")
                    acc = 0.0
                else:
                    valid_preds = preds[:, :, crop:-crop, crop:-crop]
                    valid_y = y_val[:, :, crop:-crop, crop:-crop]    
                    diff = torch.abs(valid_preds - valid_y)
                    # Use sum over remaining dims (B, 1, H', W')
                    sample_diff = diff.sum(dim=(1, 2, 3))
                    acc = (sample_diff == 0).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
    print(f"  -> Steps: {steps}, Best Valid Accuracy: {best_acc*100:.1f}%")
    return best_acc

def run_experiment():
    device = torch.device('cpu') 
    step_list = [1, 2, 4, 8] # steps list
    results = {}
    for s in step_list:
        acc = train_multistep(s, device=device)
        results[s] = acc
    with open('extended_experiments/multi_step/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    steps = list(results.keys())
    accs = list(results.values())
    plt.figure(figsize=(8, 5))
    plt.plot(steps, accs, marker='o', linestyle='-')
    plt.title("Game of Life Multi-Step Prediction (Center Crop)")
    plt.xlabel("Prediction Steps (n)")
    plt.ylabel("Accuracy (Valid Region)")
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.savefig('extended_experiments/multi_step/accuracy_plot.png')
    print("Results saved.")

if __name__ == "__main__":
    run_experiment()
