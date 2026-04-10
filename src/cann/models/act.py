import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Square(nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.pow(x,2)

class RootSquare(nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.pow(x,2))

class Gaussian(nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-x**2/(2))

class AGaussian(nn.Module):
  def __init__(self, channels=1):
    super().__init__()

    #self.a = nn.Parameter(1.0 + 1e-3 * torch.randn(1, channels,1,1))
    self.b = nn.Parameter(0.0 + 1e-3 * torch.randn(1, channels,1,1))
    self.c = nn.Parameter(1.0 + 1e-3 * torch.randn(1, channels,1,1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    #return self.a * torch.exp(-(x-self.b)**2/(2*self.c**2))
    return torch.exp(-(x-self.b)**2/(2*self.c**2))

class AdaptiveSigmoid(nn.Module):

  def __init__(self, channels=1):
    super().__init__()

    # a_sigmoid(x) = a * (1-exp(-b*x))/(1+exp(-b*x))
    self.a = nn.Parameter(1.0 + 1e-3 * torch.randn(1, channels,1,1))
    self.b = nn.Parameter(1.0 + 1e-3 * torch.randn(1, channels,1,1))

  def forward(self, x):
    out = self.a * (1 - torch.exp(-self.b*x)) / (1 + torch.exp(-self.b*x))
        
    return out

class PolynomialActivation(nn.Module):
    """
    Learnable polynomial activation: f(x) = sum(w_i * x^i)
    """
    def __init__(self, degree=2, channels=1):
        super().__init__()
        self.degree = degree
        self.channels = channels
        
        # Coeffs: [deg+1, 1, channels, 1, 1] for broadcasting
        # We want to learn a separate polynomial for each channel.
        # Shape: (deg+1, channels)
        self.coeffs = nn.Parameter(torch.randn(degree + 1, channels))
        
        # Initialize to act somewhat like identity/relu at start?
        # Initialization: w1=1, others small random.
        with torch.no_grad():
            self.coeffs.fill_(0.0)
            self.coeffs[1, :] = 1.0 # Linear term
            self.coeffs += torch.randn_like(self.coeffs) * 0.01

    def forward(self, x):
        # x shape: (B, C, H, W)
        # coeffs shape: (D+1, C)

        
        # Reshape coeffs for broadcasting: (D+1, 1, C, 1, 1)
        coeffs = self.coeffs.view(self.degree + 1, 1, self.channels, 1, 1)
        
        # Compute powers of x
        # output = w0 + w1*x + ...
        
        out = coeffs[0] # bias term
        x_pow = x
        out = out + coeffs[1] * x_pow
        
        for i in range(2, self.degree + 1):
          x_pow = x_pow * x
          out = out + coeffs[i] * x_pow
            
        return out
