import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from src.cann.models.base import ActNN
from src.cann.models.act import PolynomialActivation


class PolyKAN(ActNN):
    """
    Polynomial NN for Cellular Automata learning.
    Structure: Conv -> Poly -> Conv -> Poly -> ... -> Output
    """
    #def __init__(self, width=2, depth=2, padding_mode='circular', activation=torch.nn.ReLU()):
    def __init__(self, degree=2, width=1, depth=1, padding_mode='circular',\
        trainable_weights=True, trainable_activations: bool=True):

      my_kwargs = dict(locals())
      my_kwargs.pop("self")
      my_kwargs.pop("__class__")

      def arg_activation(wd):
        return PolynomialActivation(degree=degree, channels=wd)

      my_kwargs["activation"] = arg_activation
      super().__init__(**my_kwargs)

class MiniPolyKAN(PolyKAN):
    """
    Polynomial NN for Cellular Automata learning.
    Structure: Conv -> Poly -> Conv -> Poly -> ... -> Output
    """
    def __init__(self, degree=2, width=1, depth=1, padding_mode='circular', \
        trainable_weights: bool=True, trainable_activations: bool=True):

        my_kwargs = dict(locals())
        my_kwargs.pop("self")
        my_kwargs.pop("__class__")

        super().__init__(**my_kwargs)

        self.layers = nn.ModuleList()
        self.depth = depth
        self.width = width
        self.padding_mode = padding_mode
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
          self.layers.append(nn.Conv2d(in_channels, width, kernel_size=3, padding=1, padding_mode=padding_mode))

          # Dynamics layer uses 1x1 kernel
          self.layers.append(nn.Conv2d(1+width, width, kernel_size=1, padding=0))

          if self.weight_sharing:
            break

        # Output layer
        # Map back to 1 channel with 1x1 conv
        self.output_conv = nn.Conv2d(width, in_channels, kernel_size=1, padding=0)

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


    def forward(self, x):
        for layer_index in range(2*self.depth):
          if self.weight_sharing:
            param_index = layer_index % 2
          else:
            param_index = layer_index

          if layer_index % 2 == 0:
            xa = self.layers[param_index](x)
            x = torch.cat([x, xa], axis=1)
            x = self.activations[param_index](x)
          else:
            x = self.layers[param_index](x)
            x = self.activations[param_index](x)

        x = self.output_conv(x)
        return torch.sigmoid(x)
