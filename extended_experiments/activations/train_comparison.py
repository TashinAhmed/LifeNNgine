import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN
from extended_experiments.activations.models_comparison import StandardCNN, count_parameters

def train_and_eval(model, X_train, y_train, X_val, y_val, device='cpu', epochs=100):
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
                if acc > best_acc: best_acc = acc
                if best_acc == 1.0: return 1.0
                
    return best_acc

def run_experiment():
    device = torch.device('cpu')
    engine = CAEngine(device=device)
    
    X_train, y_train = engine.generate_data(2000, 20, "Game of Life")
    X_val, y_val = engine.generate_data(500, 20, "Game of Life")
    
    configs = [
        ("PolyKAN_D3_W2", PolyKAN(degree=3, width=2, depth=2)),
        ("PolyKAN_D3_W4", PolyKAN(degree=3, width=4, depth=2)),
        ("ReLU_W4",       StandardCNN(width=4, activation='relu')),
        ("ReLU_W16",      StandardCNN(width=16, activation='relu')),
        ("ReLU_W32",      StandardCNN(width=32, activation='relu')),
        ("ReLU_W64",      StandardCNN(width=64, activation='relu')), # Even wider
        ("Sine_W4",       StandardCNN(width=4, activation='sine')),
        ("Sine_W16",      StandardCNN(width=16, activation='sine')),
        ("Gauss_W4",      StandardCNN(width=4, activation='gaussian')),
    ]
    
    results = []
    
    for name, model in tqdm(configs, desc="Training Models"):
        model = model.to(device)
        params = count_parameters(model)
        acc = train_and_eval(model, X_train, y_train, X_val, y_val, device=device)
        
        results.append({
            "name": name,
            "params": params,
            "accuracy": acc
        })
        
    with open('extended_experiments/activations/results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Plot Params vs Accuracy
    plt.figure(figsize=(10, 6))
    for r in results:
        plt.scatter(r['params'], r['accuracy'], label=r['name'], s=100)
        
    plt.xscale('log')
    plt.xlabel('Number of Parameters (Log Scale)')
    plt.ylabel('Accuracy')
    plt.title('Parameter Efficiency: PolyKAN vs Baselines')
    plt.legend()
    plt.grid(True)
    plt.savefig('extended_experiments/activations/efficiency_plot.png')
    print("Results saved.")

if __name__ == "__main__":
    run_experiment()
