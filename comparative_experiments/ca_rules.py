import torch
import torch.nn.functional as F
import numpy as np

class CAEngine:
    """CA Engine using PyTorch."""
    def __init__(self, device='cpu'):
        self.device = device
        # Std. Moore neighborhood kernel
        self.kernel = torch.tensor([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

    def step(self, grid, rule_b, rule_s):
        """
        Executes one step of the CA.
        grid: (B, 1, H, W) tensor, values 0 or 1.
        rule_b: list of birth counts (e.g. [3])
        rule_s: list of survival counts (e.g. [2, 3])
        """
        # Count neighbors
        # Pad=1 ensures 'same' conv size
        neighbors = F.conv2d(grid, self.kernel, padding=1)
        
        # Create masks
        # The reason behind this block so that we can 
        # implement this efficiently by checking membership
        # For simplicity and batch efficiency:
        
        birth_mask = torch.zeros_like(neighbors, dtype=torch.bool)
        for b in rule_b:
            birth_mask |= (neighbors == b)  
        survive_mask = torch.zeros_like(neighbors, dtype=torch.bool)
        for s in rule_s:
            survive_mask |= (neighbors == s)
        survives = (grid == 1) & survive_mask
        births = (grid == 0) & birth_mask
        return (survives | births).float()

    def generate_data(self, num_samples, grid_size, rule_name):
        """Generates (X, y) pairs for a given rule."""
        rule_b, rule_s = self._get_rule_params(rule_name)
        X = torch.zeros((num_samples, 1, grid_size, grid_size), device=self.device)
        # Vectorized init
        # Rand densities for each sample in order to ensure robust training
        densities = torch.rand(num_samples, 1, 1, 1, device=self.device) * 0.8 + 0.1 # 0.1 ~ 0.9
        X = (torch.rand_like(X) < densities).float()
        with torch.no_grad():
            y = self.step(X, rule_b, rule_s)
        return X, y

    def _get_rule_params(self, rule_name):
        rules = {
            "Game of Life": ([3], [2, 3]),
            "Replicator":   ([1, 3, 5, 7], [1, 3, 5, 7]),       # HighLife / Replicator variant
            "Fredkin":      ([1, 3, 5, 7], [0, 2, 4, 6, 8]),    # Parity property
            "Morley":       ([3, 6, 8], [2, 4, 5])              # aka Move
        }
        if rule_name not in rules:
            raise ValueError(f"Unknown rule: {rule_name}")
        return rules[rule_name]
