import unittest

import os
import torch

from src.cann.train import train

class TestTrain(unittest.TestCase):


  def setUp(self):
    pass

  def test_train(self):
    ca_steps = [2]
    epochs = 2
    disp_every = 1

    for tw in [True, False]:
      trainable_weights = tw 
      trainable_activations = not(tw) 
      do_logging = not(tw)
      early_stopping = not(tw)
      loss_weighting = not(tw)


      my_kwargs = dict(ca_steps=ca_steps, \
          do_logging=do_logging, \
          epochs=epochs,\
          disp_every=disp_every,\
          experiment_name="delete_test",\
          loss_weighting=loss_weighting,\
          early_stopping=early_stopping,\
          trainable_weights=trainable_weights,\
          activation=["PolyKAN", "ReLU"],\
          densities=[0.1,0.3,0.1],\
          trainable_activations=trainable_activations
          )

      train(**my_kwargs)

      if do_logging:

        should_exist = os.path.join("results","delete_test")
        self.assertTrue(os.path.exists(should_exist))
        os.system(f"rm -rf {should_exist}")
    
    self.assertTrue(True)
    
if __name__ == "__main__": # pragma: no cover
  unittest.main()
