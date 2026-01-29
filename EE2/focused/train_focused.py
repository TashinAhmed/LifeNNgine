import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN

def get_periodic_data(engine, num_samples, grid_size, steps):
    rule_b, rule_s = engine._get_rule_params("Game of Life")
    X = torch.zeros((num_samples, 1, grid_size, grid_size), device=engine.device)
    densities = torch.rand(num_samples, 1, 1, 1, device=engine.device) * 0.8 + 0.1
    X = (torch.rand_like(X) < densities).float()
    current_state = X
    with torch.no_grad():
        for _ in range(steps):
            current_state = engine.step(current_state, rule_b, rule_s)
    return X, current_state

def train_periodic(steps, device='cpu', epochs=150):
    print(f"Training for {steps}-step prediction (Periodic)...")
    engine = CAEngine(device=device, padding_mode='circular')
    grid_size = 24
    X_train, y_train = get_periodic_data(engine, 2000, grid_size, steps)
    X_val, y_val = get_periodic_data(engine, 500, grid_size, steps)
    model = PolyKAN(degree=3, width=4, depth=2, padding_mode='circular').to(device)
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
                diff = torch.abs(preds - y_val)
                sample_diff = diff.sum(dim=(1, 2, 3))
                acc = (sample_diff == 0).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
    print(f"  -> Steps: {steps}, Best Accuracy: {best_acc*100:.1f}%")
    return best_acc

def run_experiment():
    device = torch.device('cpu') 
    step_list = [1, 2, 3]
    results = {}
    for s in step_list:
        # Run 3 trials to avoid seed variance
        trial_accs = []
        for t in range(3):
            torch.manual_seed(42 + s * 100 + t)
            acc = train_periodic(s, device=device)
            trial_accs.append(acc)
        results[s] = max(trial_accs)
    with open('EE2/focused/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    # Plot
    steps = list(results.keys())
    accs = list(results.values())
    plt.figure(figsize=(8, 5))
    plt.plot(steps, accs, marker='o', linestyle='-')
    plt.title("Game of Life Multi-Step Prediction (Periodic, n=[1,2,3])")
    plt.xlabel("Prediction Steps (n)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.xticks(step_list)
    plt.savefig('EE2/focused/accuracy_plot.png')
    print("Results saved to EE2/focused/")

if __name__ == "__main__":
    run_experiment()
