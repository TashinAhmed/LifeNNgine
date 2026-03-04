import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ActNN(nn.Module):
    """
    Generic NN with the same arhcitecture
    Structure: Conv -> act -> Conv -> act -> ... -> Output
    """
    def __init__(self, width=1, depth=1, padding_mode='circular', activation=lambda **kwargs: torch.nn.ReLU()):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.depth = depth
        self.padding_mode = padding_mode
        
        # Input is 1 channel (grid state)
        in_channels = 1
        
        # Hidden layers
        # width is 'm' in Kenyon and Springer
        # depth is 'n', number of steps in the CA data, 
        # depth/n is also number of repeats of the (conv3by3 -> conv1by1) block

        for i in range(depth):
            # Neighborhood_ layer uses 3x3 kernel
            self.layers.append(nn.Conv2d(in_channels, 2*width, kernel_size=3, padding=1, padding_mode=padding_mode))
            self.activations.append(activation(wd=2*width))

            # Dynamics layer uses 1x1 kernel
            self.layers.append(nn.Conv2d(2*width, in_channels, kernel_size=1, padding=0))
            self.activations.append(activation(wd=in_channels))
            
            
        # Output layer
        # Map back to 1 channel with 1x1 conv
        self.output_conv = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)

    def forward(self, x):

        for layer, act in zip(self.layers, self.activations):
          x = layer(x)
          x = act(x)

        x = self.output_conv(x)
        return torch.sigmoid(x)

    def count_parameters(self):

      number_parameters = 0

      for param in self.parameters():
          number_parameters += param.numel()

      return number_parameters

    def get_parameters(self):

        params = np.array([])

        for param in self.parameters():
            params = np.append(params, param.detach().numpy().ravel())

        return params
