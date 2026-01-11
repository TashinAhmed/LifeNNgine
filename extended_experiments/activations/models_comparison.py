import torch
import torch.nn as nn
import torch.nn.functional as F
from comparative_experiments.models import PolynomialActivation

class SineActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x)

class GaussianActivation(nn.Module):
    def __init__(self):
        super().__init__()
        # Making it standard fixed for now, or simple learnable parameters like Poly?
        # Trying simple fixed e^(-x^2) first, maybe learnable scale.
        self.sigma = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        return torch.exp(- (x ** 2) / (2 * self.sigma ** 2))

class StandardCNN(nn.Module):
    """
    Standard CNN with ReLU activations.
    Struct: Conv -> ReLU -> Conv -> ReLU -> ... -> Sigmoid
    """
    def __init__(self, width=16, depth=2, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = 1
        
        if activation == 'relu':
            self.act_builder = lambda: nn.ReLU()
        elif activation == 'sine':
            self.act_builder = lambda: SineActivation()
        elif activation == 'gaussian':
            self.act_builder = lambda: GaussianActivation()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        for i in range(depth):
            if i == 0:
                self.layers.append(nn.Conv2d(in_channels, width, kernel_size=3, padding=1))
            else:
                self.layers.append(nn.Conv2d(in_channels, width, kernel_size=1, padding=0))
            self.layers.append(self.act_builder())
            in_channels = width
        self.output_conv = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_conv(x)
        return torch.sigmoid(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
