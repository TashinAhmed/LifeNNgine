import torch
import numpy as np

def generate_bespoke_grid(grid_size=24):
    """
    Generates a grid that contains all 512 possible 3 x3 binary neighborhoods.
    A 512-neighborhood set fits in a roughly 23x23 grid if packed efficiently,
    but just tiling them for simplicity.
    """
    # 512 configs
    configs = []
    for i in range(512):
        # Convert i to 9bit binary
        bits = [(i >> j) & 1 for j in range(9)]
        configs.append(np.array(bits).reshape(3, 3))
    
    # Pack into a larger grid
    # Each 3x3 needs a 1cell gap or overlap to be independent?
    # tiling them 3x3 blocks.
    # 512 blocks. sqrt(512) approx 22.6. 
    # Doing 23x23 blocks of 3x3.
    
    # 23 * 3 = 69
    # That's a bit large. tring to fit them differently.
    # generating many small grids.
    
    # Generating a single large grid that definitely contains them.
    # A more elegant way can de Bruijn sequence-like packing!!! 
    # For simplicity, creating a batch of 512 3x3 patches
    # and pad them to the desired grid size or stack them.
    
    patch_size = 3
    num_configs = 512
    
    # Create a batch of (512, 1, 3, 3)
    data = torch.tensor(np.array(configs)).float().unsqueeze(1)
    
    # For training, two options:
    # 1. Provide these 512 patches as the training set.
    # 2. Embed them into a large grid.
    
    return data

def get_bespoke_dataset(engine, rule_name):
    """
    Returns (X, y) where X contains all 512 neighborhood configurations.
    """
    X = generate_bespoke_grid() # (512, 1, 3, 3)
    
    # Need to pad X to at least 3x3 for the engine to work if it uses padding internally,
    # or just ensure the engine's step works on 3x3.
    # CAEngine.step with padding='circular' or 'zero' on 3x3 works.
    
    rule_b, rule_s = engine._get_rule_params(rule_name)
    
    # For a 3x3 input,  center cell's transition is the key.
    # But for training a ConvNet, train on the whole 3x3 -> 3x3 transition.
    
    with torch.no_grad():
        y = engine.step(X, rule_b, rule_s)
        
    return X, y

if __name__ == "__main__":
    # Test
    grid = generate_bespoke_grid()
    print(f"Generated {grid.shape[0]} configurations.")
