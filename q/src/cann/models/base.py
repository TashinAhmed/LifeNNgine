from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ActNN(nn.Module):
    """
    Generic NN with the same arhcitecture
    Structure: Conv -> act -> Conv -> act -> ... -> Output
    """
    def __init__(self, width=1, depth=1, padding_mode='circular', \
        activation=lambda **kwargs: torch.nn.ReLU(), share_weights: bool=False, \
        trainable_weights: bool=True,\
        trainable_activations: bool=True, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.depth = depth
        self.width = width
        self.padding_mode = padding_mode
        self.weight_sharing = share_weights 
        self.trainable_weights = trainable_weights
        self.trainable_activations = trainable_activations
        
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

          if self.weight_sharing:
            break 
            
        # Output layer
        # Map back to 1 channel with 1x1 conv
        self.output_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

        if self.trainable_weights:
          pass
        else:
          with torch.no_grad():
            for layer in self.layers[1:]:
              for param in layer.named_parameters():
                if "bias" in param[0]:
                  param[1][:] = 0.0
                else:
                  param[1][:] = 1. / param[1].numel()

                param[1].requires_grad = False

              for param in self.output_conv.named_parameters():
                if "bias" in param[0]:
                  param[1][:] = 0.0
                else:
                  param[1][:] = 1. / param[1].numel()

                param[1].requires_grad = False

        if self.trainable_activations:
          pass
        else:
          with torch.no_grad():
            for act in self.activations:
              for param in act.named_parameters():
                param[1].requires_grad = False

    def forward(self, x):
        
        for layer_index in range(2*self.depth): 
          if self.weight_sharing:
            param_index = layer_index % 2
          else:
            param_index = layer_index
          x = self.layers[param_index](x)
          x = self.activations[param_index](x)

        x = self.output_conv(x)
        return torch.sigmoid(x)

    def count_parameters(self):

      number_parameters = 0

      for param in self.parameters():
          number_parameters += param.numel()

      return number_parameters

    def count_trainable(self):

      self.train()

      number_trainable = 0

      for param in self.parameters():
        if param.requires_grad:
          number_trainable += param.numel()

      return number_trainable

    def get_parameters(self):

        params = np.array([])

        for param in self.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def set_parameters(self, my_params):

        with torch.no_grad():
          param_start = 0
          for name, param in self.named_parameters():

              param_stop = param_start + reduce(lambda x,y: x*y, param.shape)
              param[:] = torch.nn.Parameter(torch.tensor(\
                      my_params[param_start:param_stop].reshape(param.shape)))
              param_start = param_stop
