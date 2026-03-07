import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from comparative_experiments.ca_rules import CAEngine
from comparative_experiments.models import PolyKAN
from data_utils import get_bespoke_dataset

def train_model(X_train, y_train, X_val, y_val, name, epochs=100):
    device = torch.device('cpu')
    model = PolyKAN(degree=3, width=4, depth=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    train_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # val
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_preds = (val_outputs > 0.5).float()
            # Exact match for the whole grid/patch
            # For bespoke, X_val is (512, 1, 3, 3)
            # Checking if the entire 3x3 is correct (or just center)
            acc = (val_preds == y_val).all(dim=(1, 2, 3)).float().mean().item()
            val_accs.append(acc)
            
    return train_losses, val_accs

def run_experiment():
    device = torch.device('cpu')
    engine = CAEngine(device=device, padding_mode='circular')
    rule = "Game of Life"
    
    print("Generating Bespoke Data...")
    X_bespoke, y_bespoke = get_bespoke_dataset(engine, rule)
    
    print("Generating Random Data (matching size)...")
    # 512 samples of 3x3
    X_random, y_random = engine.generate_data(512, 3, rule)
    
    # We'll also use a larger validation set to check generalization
    X_val_large, y_val_large = engine.generate_data(1000, 20, rule)
    
    # Since model expects 20x20 usually, but PolyKAN is FCN, 3x3 works.
    # padding might affect things. 
    
    print("Training on Bespoke Data...")
    bespoke_losses, bespoke_accs = train_model(X_bespoke, y_bespoke, X_bespoke, y_bespoke, "Bespoke")
    
    print("Training on Random Data...")
    random_losses, random_accs = train_model(X_random, y_random, X_random, y_random, "Random")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(bespoke_losses, label='Bespoke Train Loss')
    plt.plot(random_losses, label='Random Train Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(bespoke_accs, label='Bespoke Accuracy (on self)')
    plt.plot(random_accs, label='Random Accuracy (on self)')
    plt.title('Accuracy')
    plt.legend()
    
    os.makedirs('EE4', exist_ok=True)
    plt.savefig('EE4/training_comparison.png')
    plt.close()
    
    print(f"Final Bespoke Acc: {bespoke_accs[-1]:.4f}")
    print(f"Final Random Acc: {random_accs[-1]:.4f}")

if __name__ == "__main__":
    run_experiment()
