import unittest

import torch

from src.cann.models.minimal import Minimal

class TestMinimal(unittest.TestCase):

  def setUp(self):
    
    self.model = Minimal() 

  def test_forward(self):

    x = 1.0 * (torch.rand(1,1,32,32) > 0.5)

    y = self.model(x)

    self.assertEqual(x.shape, y.shape)
    
if __name__ == "__main__": # pragma: no cover
  unittest.main()
