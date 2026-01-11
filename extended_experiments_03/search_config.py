import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
import itertools
from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN

def get_multistep_data(engine, num_samples, grid_size, steps):
    rule_b, rule_s = engine._get_rule_params("Game of Life")
    X = torch.zeros((num_samples, 1, grid_size, grid_size), device=engine.device)
    densities = torch.rand(num_samples, 1, 1, 1, device=engine.device) * 0.8 + 0.1
    X = (torch.rand_like(X) < densities).float()
    current_state = X
    with torch.no_grad():
        for _ in range(steps):
            current_state = engine.step(current_state, rule_b, rule_s)
    return X, current_state

def train_and_eval(model, X_train, y_train, X_val, y_val, steps, epochs=150):
    optimizer = optim.Adam(model.parameters(), lr=0.005) # Slightly lower LR for stability
    criterion = nn.BCELoss()
    best_acc = 0.0
    crop = steps
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                preds = (val_outputs > 0.5).float()
                valid_preds = preds[:, :, crop:-crop, crop:-crop]
                valid_y = y_val[:, :, crop:-crop, crop:-crop]
                diff = torch.abs(valid_preds - valid_y)
                if diff.numel() == 0: return 0.0 # Safety
                sample_diff = diff.sum(dim=(1, 2, 3))
                acc = (sample_diff == 0).float().mean().item()
                if acc > best_acc: best_acc = acc
                if best_acc >= 0.999: return 1.0 # Solved
    return best_acc

def run_search():
    device = torch.device('cpu')
    engine = CAEngine(device=device)
    grid_size = 24
    
    ################################################
    # Configs to try (increasing complexity)
    # Degrees: Higher degree needed for composition
    # Depths: Deeper needed for composition
    # Widths: Wider needed for spatial propagation?
    ################################################
    # Iterate through configs until we solve each step n.
    search_space = [
        # degree, width, depth
        (3, 4, 2), # Baseline (likely fails for n>1)
        (3, 8, 3), # Deeper
        (5, 4, 3), # Higher degree
        (5, 8, 4), # Deeper & Degree
        (5, 16, 5),# Heavy
        (8, 8, 5), # High Degree
        (3, 32, 8) # Deep & Wide
    ]
    
    steps_to_solve = [1, 2, 3]
    results = {}
    print("Starting Architecture Search for >95% Accuracy...")
    for s in steps_to_solve:
        print(f"\nTargeting Steps n={s}...")
        results[s] = {"accuracy": 0.0, "config": None}
        # Generate data once per step count
        X_train, y_train = get_multistep_data(engine, 2000, grid_size, s)
        X_val, y_val = get_multistep_data(engine, 500, grid_size, s)
        for deg, width, depth in search_space:
            print(f"  Trying Deg={deg}, Width={width}, Depth={depth}...")
            # Running 2 trials to avoid bad seed
            best_trial_acc = 0.0
            for t in range(2):
                torch.manual_seed(42 + t*100) # Reproducible seeds
                model = PolyKAN(degree=deg, width=width, depth=depth).to(device)
                acc = train_and_eval(model, X_train, y_train, X_val, y_val, s, epochs=200)
                if acc > best_trial_acc: best_trial_acc = acc
            print(f"    -> Acc: {best_trial_acc*100:.1f}%")
            if best_trial_acc >= 0.95:
                print(f"  >>> SOLVED n={s} with config {deg}/{width}/{depth} (Acc: {best_trial_acc*100:.1f}%)")
                results[s] = {
                    "accuracy": best_trial_acc,
                    "config": {"degree": deg, "width": width, "depth": depth}
                }
                break # Move to next step count
            # Update best found so far
            if best_trial_acc > results[s]["accuracy"]:
                results[s] = {
                    "accuracy": best_trial_acc,
                    "config": {"degree": deg, "width": width, "depth": depth}
                }
                
    with open('extended_experiments_03/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSearch Complete. Results saved.")

if __name__ == "__main__":
    run_search()
