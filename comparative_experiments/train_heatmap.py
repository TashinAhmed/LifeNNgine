import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
import time
from tqdm import tqdm
from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN

def train_one_model(model, X_train, y_train, X_val, y_val, device, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    best_acc = 0.0
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                preds = (val_outputs > 0.5).float()
                diff = torch.abs(preds - y_val)
                sample_diff = diff.sum(dim=(1, 2, 3))
                acc = (sample_diff == 0).float().mean().item()
                
                if acc > best_acc:
                    best_acc = acc
                
                if best_acc == 1.0:
                    model.train()
                    return 1.0
            model.train()
                    
    return best_acc

def run_experiments(args):
    # Force CPU for potentially better small-scale performance
    device = torch.device('cpu') 
    print(f"Using device: {device}")
    
    engine = CAEngine(device=device)
    
    rules = ["Game of Life", "Replicator", "Fredkin", "Morley"]
    degrees = [2, 3, 4]
    widths = [2, 4, 8]
    depths = [2, 3]
    
    trials = 3 if not args.fast else 1
    epochs = 100 if not args.fast else 10
    
    total_steps = len(rules) * len(degrees) * len(widths) * len(depths)
    pbar = tqdm(total=total_steps, desc="Grid Search")
    
    # Clean previous results
    with open('comparative_experiments/heatmap_results.jsonl', 'w') as f:
        pass
        
    for rule in rules:
        X_train, y_train = engine.generate_data(1000, 20, rule)
        X_val, y_val = engine.generate_data(200, 20, rule)
            
        for deg in degrees:
            for width in widths:
                for depth in depths:
                    success_count = 0
                    
                    for t in range(trials):
                        model = PolyKAN(degree=deg, width=width, depth=depth).to(device)
                        acc = train_one_model(model, X_train, y_train, X_val, y_val, device, epochs=epochs)
                        
                        if acc >= 0.999:
                            success_count += 1
                            
                    success_rate = success_count / trials
                    
                    result_entry = {
                        "rule": rule,
                        "degree": deg,
                        "width": width,
                        "depth": depth,
                        "success_rate": success_rate
                    }
                    
                    # Incremental Save
                    with open('comparative_experiments/heatmap_results.jsonl', 'a') as f:
                        f.write(json.dumps(result_entry) + "\n")
                        
                    pbar.set_postfix({"R": rule, "D": deg, "SR": f"{success_rate:.1f}"})
                    pbar.update(1)
                    
    pbar.close()
    print("Experiments completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Run fast mode for debugging')
    args = parser.parse_args()
    
    run_experiments(args)
