import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from src.cann.models.base import ActNN

class Minimal(ActNN):

    def __init__(self, width=1, depth=1, padding_mode='circular', \
        activation=lambda **kwargs: torch.nn.ReLU(), share_weights: bool=False, **kwargs):
        super().__init__()
        """

        """

        self.w11 = torch.tensor([[[[1,1,1.],[1.,.1,1.],[1.,1.,1.]]]])
        self.b11 = torch.tensor([-3])
        self.w12 = torch.tensor([[[[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]]]])
        self.b12 = torch.tensor([-2])

        self.w1 = torch.cat([self.w11, self.w12], axis=0)
        self.b1 = torch.cat([self.b11, self.b12], axis=0)


        self.w21 = torch.tensor([[[[-10.]], [[1.]]]])

        s = 20  
        self.w3 = torch.tensor([[[[s*2]]]])
        self.b3 = torch.tensor([-s])


        with torch.no_grad():
          
          for ii, param in enumerate(self.layers[0].named_parameters()):
            if ii % 2:
              #bias param
              param[1][:] = self.b1
            else:
              param[1][:] = self.w1

          for ii, param in enumerate(self.layers[1].named_parameters()):
            if ii % 2:
              #bias param
              param[1][:] *= 0 
            else:
              param[1][:] = self.w21

          for ii, param in enumerate(self.output_conv.named_parameters()):
            if ii % 2:
              #bias param
              param[1][:] = self.b3 
            else:
              param[1][:] = self.w3

        for param in self.parameters():
            param.requres_grad = False

