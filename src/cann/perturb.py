import sys
import argparse
import os

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

import numpy as np
import pandas as pd

from src.cann.models.base import ActNN
from src.cann.models.polykan import PolyKAN, MiniPolyKAN
from src.cann.train import train_model, seed_all, get_cli_args, get_git_hash, BIG_PRIME

from carle.env import CARLE

def k_sign_perturb(parameters: np.ndarray, k: int=1) -> np.ndarray:
  
  probs = np.random.rand(*parameters.shape)
  new_parameters = 1. * parameters
  
  for k_sign in range(k):
    perturb_index = np.argmax(probs)
    new_parameters[perturb_index] *= -1.
    probs[perturb_index] *= 0

  return new_parameters

def j_zero_perturb(parameters: np.ndarray, j: int=0, guided: str="magnitude") -> np.ndarray:
  
  if guided == "magnitude":
    guidance = np.abs(parameters) / np.abs(parameters).max()
    probs = np.random.rand(*parameters.shape) * guidance
  else:
    probs = np.random.rand_like(parameters)
  new_parameters = 1. * parameters
  
  for j_zero in range(j):
    perturb_index = np.argmax(probs)
    new_parameters[perturb_index] *= -1.
    probs[perturb_index] *= 0

  return new_parameters

def uniform_perturb(parameters: np.ndarray, magnitude: float) -> np.ndarray:

  perturbs = np.random.randn(*parameters.shape)

  return parameters + magnitude * perturbs

def perturb(**kwargs):

  filepath = kwargs.get("filepath","") 
  epoch_indices = kwargs.get("epoch_index", [-1])
  repeats = kwargs.get("repeats", 1)
  base_seed = kwargs.get("pseudorandom_seed", 1)
  j_counts = kwargs.get("j_zero_perturbations", [0])
  k_counts = kwargs.get("k_sign_perturbations", [0])
  u_magnitudes = kwargs.get("uniform_magnitude", [0.0])
  experiment_tag = kwargs.get("experiment_tag", "perturb")
  ca_rule = kwargs.get("rulestring", "B3/S23")
  entry_point = kwargs.get("entry_point", "")
  git_hash = kwargs.get("git_has", "")
  number_epochs = kwargs.get("epochs", 125000)
  ca_step = kwargs.get("ca_steps", 1)

  run_number = 0
  disp_every = 10
  
  exp_df = pd.read_csv(filepath)
  del exp_df["Unnamed: 0"]

  for epoch_index in epoch_indices:

    for k_count in k_counts:

      j_count = 0
      u_magnitude = 0.

      for run_index in range(len(exp_df)):

        # only perturb/retrain runs that were originally successful.
        if exp_df["final_grid_accuracy"][run_index] == 1.0:
          my_seed = base_seed + run_number * BIG_PRIME
          seed_all(my_seed)

          activation_name = exp_df["activation_name"][run_index]
          run_filename = exp_df["run_filename"][run_index]
          degree = exp_df["degree"][run_index]
          width = exp_df["model_width"][run_index]
          depth = exp_df["model_depth"][run_index]

          run_df = pd.read_csv(run_filename)
          del run_df["Unnamed: 0"]

          parameters = np.load(run_df["parameters_filename"][0])[epoch_index]
          parameters = k_sign_perturb(parameters, k=k_count)

          experiment_name = f"perturb_{experiment_tag}_j{j_count}_k{k_count}_u{u_magnitude}_{run_number}"
          train_model(my_seed=my_seed,\
              experiment_name=experiment_name,\
              run_number=run_number,\
              ca_rule=ca_rule,\
              activation_name=activation_name,\
              degree=degree,\
              width=width,\
              depth=depth,\
              do_logging=True,\
              early_stopping=True,\
              log_filename=experiment_name,\
              epochs=number_epochs,\
              ca_step=ca_step,\
              git_hash=git_hash,\
              entry_point=entry_point,\
              params = parameters
              )

          run_number += 1
    for j_count in j_counts:
      k_count = 0
      u_magnitude = 0.

      for run_index in range(len(exp_df)):

        # only perturb/retrain runs that were originally successful.
        if exp_df["final_grid_accuracy"][run_index] == 1.0:
          my_seed = base_seed + run_number * BIG_PRIME
          seed_all(my_seed)

          activation_name = exp_df["activation_name"][run_index]
          run_filename = exp_df["run_filename"][run_index]
          degree = exp_df["degree"][run_index]
          width = exp_df["model_width"][run_index]
          depth = exp_df["model_depth"][run_index]

          run_df = pd.read_csv(run_filename)
          del run_df["Unnamed: 0"]

          parameters = np.load(run_df["parameters_filename"][0])[epoch_index]
          parameters = j_zero_perturb(parameters, j=j_count)

          experiment_name = f"perturb_{experiment_tag}_j{j_count}_k{k_count}_u{u_magnitude}_{run_number}"
          train_model(my_seed=my_seed,\
              experiment_name=experiment_name,\
              run_number=run_number,\
              ca_rule=ca_rule,\
              activation_name=activation_name,\
              degree=degree,\
              width=width,\
              depth=depth,\
              do_logging=True,\
              early_stopping=True,\
              log_filename=experiment_name,\
              epochs=number_epochs,\
              ca_step=ca_step,\
              git_hash=git_hash,\
              entry_point=entry_point,\
              params = parameters
              )

          run_number += 1
    for u_magnitude in u_magnitudes:
      k_count = 0
      j_count = 0

      for run_index in range(len(exp_df)):

        # only perturb/retrain runs that were originally successful.
        if exp_df["final_grid_accuracy"][run_index] == 1.0:
          my_seed = base_seed + run_number * BIG_PRIME
          seed_all(my_seed)

          activation_name = exp_df["activation_name"][run_index]
          run_filename = exp_df["run_filename"][run_index]
          degree = exp_df["degree"][run_index]
          width = exp_df["model_width"][run_index]
          depth = exp_df["model_depth"][run_index]

          run_df = pd.read_csv(run_filename)
          del run_df["Unnamed: 0"]

          parameters = np.load(run_df["parameters_filename"][0])[epoch_index]
          parameters = uniform_perturb(parameters, magnitude=u_magnitude)

          experiment_name = f"perturb_{experiment_tag}_j{j_count}_k{k_count}_u{u_magnitude}_{run_number}"
          train_model(my_seed=my_seed,\
              experiment_name=experiment_name,\
              run_number=run_number,\
              ca_rule=ca_rule,\
              activation_name=activation_name,\
              degree=degree,\
              width=width,\
              depth=depth,\
              do_logging=True,\
              early_stopping=True,\
              log_filename=experiment_name,\
              epochs=number_epochs,\
              ca_step=ca_step,\
              git_hash=git_hash,\
              entry_point=entry_point,\
              params = parameters
              )

          run_number += 1

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()

  parser.add_argument("-e", "--epochs", type=int, default=100,\
      help="number of 'epochs' to train for.")
  parser.add_argument("-f", "--filepath", type=str, default="",\
      help="filepath to restore parameters from for perturbation experiment")
  parser.add_argument("-g", "--uniform_magnitude", nargs="+", type=float, default=[0.0],\
      help="magnitude of uniform noise to use in perturbing parameters")
  parser.add_argument("-i", "--epoch_index", nargs="+", type=int, default=[-1],\
      help="epoch index to grab parameters from, e.g. 0 (first epoch, before training)"\
      " or -1 (final epoch, default)")
  parser.add_argument("-j", "--j_zero_perturbations", nargs="+", type=int, default=[0],\
      help="j, the number of times to perturb parameters by zeroing out")
  parser.add_argument("-k", "--k_sign_perturbations", nargs="+", type=int, default=[0],\
      help="k, the number of times to perturb parameters with a sign change")
  parser.add_argument("-p", "--pseudorandom_seed", type=int,  default=17,\
      help="seed used as the base seed for pseudorandom number generators")
  parser.add_argument("-r", "--repeats", type=int, default=1,\
      help="run  perturbations experiment r number of times (default 1)")
  parser.add_argument("-u", "--rulestring", type=str, default="B3/S23",\
      help="rulestring defining which Life-like CA rule to use. default is Life 'B3/S23'")
  parser.add_argument("-x", "--experiment_tag", type=str, default="perturb",\
    help="tag (string) used to identify the experiment")

  args = parser.parse_args()

  my_kwargs = dict(args._get_kwargs()) 

  my_kwargs["git_hash"] = get_git_hash() #.decode("utf8").replace("\n","")

  entry_point = []
  entry_point.append(os.path.split(sys.argv[0])[1])
  args_list = sys.argv[1:]
  my_kwargs["entry_point"] = get_cli_args(entry_point, args_list)

  perturb(**my_kwargs)
