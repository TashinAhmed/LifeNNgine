import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN

def get_data(engine, num_samples, grid_size):
    rule_b, rule_s = engine._get_rule_params("Game of Life")
    X = torch.zeros((num_samples, 1, grid_size, grid_size), device=engine.device)
    densities = torch.rand(num_samples, 1, 1, 1, device=engine.device) * 0.8 + 0.1
    X = (torch.rand_like(X) < densities).float()
    with torch.no_grad():
        y = engine.step(X, rule_b, rule_s)
    return X, y

def run_stability_test():
    device = torch.device('cpu')
    engine = CAEngine(device=device)
    grid_size = 24
    print("Generating data...")
    X_train, y_train = get_data(engine, 2000, grid_size)
    X_val, y_val = get_data(engine, 500, grid_size)
    seeds = [42, 100, 2024, 7, 0]
    results = []
    print(f"Running 5 trials for n=1 (Degree 3, Width 4, Depth 2)...")
    for seed in seeds:
        torch.manual_seed(seed)
        model = PolyKAN(degree=3, width=4, depth=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        best_acc = 0.0
        epochs = 150
        for epoch in range(epochs): 
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    preds = (val_outputs > 0.5).float()
                    valid_preds = preds[:, :, 1:-1, 1:-1] # Crop 1 pixel
                    valid_y = y_val[:, :, 1:-1, 1:-1]                    
                    diff = torch.abs(valid_preds - valid_y)
                    sample_diff = diff.sum(dim=(1, 2, 3))
                    acc = (sample_diff == 0).float().mean().item()
                    if acc > best_acc: best_acc = acc
                    if best_acc == 1.0: break # Early stop
        print(f"Seed {seed}: Acc = {best_acc*100:.1f}%")
        results.append(best_acc)
    print(f"Mean Accuracy: {np.mean(results)*100:.1f}%")
    print(f"Success Rate: {sum(r > 0.99 for r in results)}/5")

if __name__ == "__main__":
    run_stability_test()
