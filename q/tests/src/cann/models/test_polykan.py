import unittest

import torch

from src.cann.models.base import ActNN
from src.cann.models.polykan import PolyKAN, MiniPolyKAN

class TestPolyKAN(unittest.TestCase):

  def setUp(self):
    
    self.model = PolyKAN() 

  def test_forward(self):

    x = 1.0 * (torch.rand(1,1,32,32) > 0.5)

    y = self.model(x)

    self.assertEqual(x.shape, y.shape)

class TestMiniPolyKAN(unittest.TestCase):

  def setUp(self):
    
    self.model = MiniPolyKAN() 

  def test_forward(self):

    x = 1.0 * (torch.rand(1,1,32,32) > 0.5)

    y = self.model(x)

    self.assertEqual(x.shape, y.shape)

if __name__ == "__main__": # pragma: no cover
  unittest.main()
