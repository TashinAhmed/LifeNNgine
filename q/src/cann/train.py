import sys
import subprocess
import argparse
import os

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

import numpy as np

from src.cann.models.base import ActNN
from src.cann.models.polykan import PolyKAN, MiniPolyKAN
from src.cann.models.act import Square, RootSquare

from carle.env import CARLE
import matplotlib.pyplot as plt

import pandas as pd
#import duckdb

BIG_PRIME = 7919 # a big prime number used for seeds
ACT_DICT = dict(relu=lambda **kwargs: nn.ReLU(), \
    celu=lambda **kwargs: nn.CELU(),\
    silu=lambda wd: nn.SiLU(),\
    prelu=lambda wd: nn.PReLU(num_parameters=wd),\
    tanh=lambda **kwargs: nn.Tanh(),\
    sigmoid=lambda **kwargs: nn.Sigmoid(),\
    square=lambda **kwargs: Square(),\
    rsquare=lambda **kwargs: RootSquare(),\
    rootsquare=lambda **kwargs: RootSquare(),\
    leakyrelu=lambda **kwargs: nn.LeakyReLU())

def get_cli_args() -> str:
  
  sorted_args = []
  for arg_index in range(0, len(args_list)):

    if "-" in args_list[arg_index]:
      sorted_args.append([args_list[arg_index]])
    else:
      sorted_args[-1].append(args_list[arg_index])

  sorted_args.sort()
  entry_point.extend(sorted_args)
  string_args = str(entry_point).replace("[","").replace("]","").replace(",","").replace("'","")
  return string_args

def get_git_hash() -> str:

  hash_command = ["git", "rev-parse", "--verify", "HEAD"]
  git_hash = subprocess.check_output(hash_command)

  return git_hash.decode("utf8").replace("\n","")

def seed_all(my_seed):
  np.random.seed(my_seed)
  torch.random.manual_seed(my_seed)

def make_df_from_kwargs(**kwargs) -> pd.DataFrame:
  df = pd.DataFrame.from_dict(dict(kwargs))
  return df

def validate(env, model, ca_steps: int=1,\
    loss_fn: callable=F.binary_cross_entropy,\
    val_size: int=128):
  """
    validate the model with
  """

  
  density = torch.rand(val_size,1, 1,1) * 0.8 + 0.1
  x = 1.0 * (torch.rand(val_size, 1, 32, 32) > density)

  x2 = 1. * x
  for ii in range(5):
    x2 = env(x2)

  x = torch.cat([x,x2], axis=0)

  with torch.no_grad():
    y_target = 1. * x
    for ii in range(ca_steps):
      y_target = env(y_target)

    y_pred = model(x)

    loss = loss_fn(y_pred, y_target)
    accuracy = 1.0 * (y_pred.round() == y_target)

    my_grid_accuracy = (1.0 * (1.000 == accuracy.mean(axis=(1,2,3)))).mean().item()
    my_cell_accuracy = accuracy.mean().item()

  return loss.item(), my_cell_accuracy, my_grid_accuracy

def print_log(epoch, time_elapsed, save_file: str="", **kwargs):
  msg = f"epoch: {epoch}, time: {time_elapsed:.2e}"
  for key in kwargs.keys():
    if type(kwargs[key]) == torch.Tensor:
      msg += f",\t{key}: {kwargs[key]:.3e}"
    else:
      msg += f",\t{key}: {kwargs[key]}"

  print(msg)
  msg += "\n"

  if save_file == "":
    pass
  else:
    with open(save_file, "a") as f:
      f.write(msg)

def mkdir_w_parents(exp_dir):

  while not(os.path.exists(exp_dir)):
    dir_name = os.path.dirname(exp_dir)
    if dir_name == "":
      os.mkdir(exp_dir)
    elif not(os.path.exists(dir_name)):
      print(dir_name)
      mkdir_w_parents(dir_name)
    else:
      os.mkdir(exp_dir)

def train_model(my_seed: int=1337,\
    experiment_name: str="default",\
    run_number: int=0,\
    ca_rule: str="B3/S23",\
    lr: float=0.001,\
    density: float=0.5,\
    disp_every: int=10,\
    loss_weighting: bool=False,\
    activation_name: str="relu",\
    degree: int=2,\
    depth: int=1,\
    width: int=1,\
    trainable_weights: bool=True,\
    trainable_activations: bool=True,\
    do_logging: bool=False,\
    log_filename: str="",\
    early_stopping: bool=False,\
    epoch_size: int=8,\
    minibatch_size: int=8,\
    epochs: int=10,\
    dim: int=32,\
    ca_step: int=1,\
    git_hash: str="",\
    entry_point: str="",\
    params=None
    ):

  if git_hash == "":
    # use subprocess to get the current git hash, store
    git_hash = get_git_hash()

  # set up logging, experiment directory
  if do_logging:
    t00 = time.time()
    exp_dir = os.path.join("results", experiment_name)
    log_filename = os.path.join(exp_dir, f"log.txt")
    exp_results_filename = os.path.join(exp_dir, f"exp.csv")
    if os.path.exists(exp_dir):
      pass
    else:
      # make parents
      mkdir_w_parents(exp_dir)
  else:
    log_filename = ""

  # initialize lists for experiment results
  run_filename = []
  seed = []
  exp_model_name = []
  activation_name_list = []
  degree_list = []
  parameter_count = []
  trainable_count = []
  trainable_neurons = []
  trainable_coefficients = []
  density_0 = []
  model_width = []
  model_depth = []
  exp_ca_steps = []
  rulestring = []
  final_cell_accuracy = []
  final_grid_accuracy = []
  final_loss = []
  epoch_batch_size = []
  mini_batch_size = []
  total_samples = []
  model_updates = []
  wall_time = []
  git_hashs = []
  entry_points = []

  seed_all(my_seed)
  env = CARLE()
  env.rules_from_string(ca_rule)

  if "poly" in activation_name.lower():
    if "mini" in activation_name.lower():
      model = MiniPolyKAN(degree=degree, width=width, depth=depth, \
          trainable_weights=trainable_weights, \
          trainable_activations=trainable_activations)
    else:
      model = PolyKAN(degree=degree, width=width, depth=depth, \
          trainable_weights=trainable_weights, \
          trainable_activations=trainable_activations)

  else:
    my_act = ACT_DICT[activation_name.lower()]
    activation = my_act #mbda wd: my_act(num_parameters=wd)
    model = ActNN(degree=degree, width=width, activation=activation, depth=depth, \
        trainable_weights=trainable_weights, \
        trainable_activations=trainable_activations)

  if params is None:
    pass
  else:
    model.set_parameters(params)

  if do_logging:
    run_results_filename = os.path.join(exp_dir, f"run{run_number}.csv")
    parameters_filename = os.path.join(exp_dir, f"parameters_{run_number}.npy")

  # logging arrays
  model_name = []
  model_parameters = None
  bce_loss = []
  cell_accuracy = []
  grid_accuracy = []

  optimizer = Adam(model.parameters(), lr=lr)
  loss_fn = F.binary_cross_entropy

  my_model_name = model._get_name()
  print_log(-1, 0.0, run_number=run_number, seed=my_seed,\
      git_hash=git_hash,\
      entry_point=entry_point,\
      model_name=my_model_name, \
      ca_rule=ca_rule,\
      number_ca_steps=ca_step,\
      density=density,\
      parameter_count=model.count_parameters(), \
      trainable_count=model.count_trainable(), \
      learning_rate=lr, save_file=log_filename)

  t0_run = time.time()
  for epoch in range(epochs):
    
    x = 1.0 * (torch.rand(epoch_size, 1, dim, dim) > density)

    with torch.no_grad():
      y = 1 * x
      
      # n-step ca dynamics targets
      for cas in range(ca_step):
        y = env(y)

      y_pred = model(x[:minibatch_size])
      y_target = y[:minibatch_size]
      loss = loss_fn(y_pred, y_target)

    t0_epoch = time.time()
    t1 = time.time()

    if (epoch % disp_every) == 0 or epoch == (epochs-1):

      loss, my_cell_accuracy, my_grid_accuracy = validate(\
          env, model, ca_steps=ca_step, loss_fn=loss_fn)

      elapsed_epoch = t1 - t0_epoch
      elapsed_total = t1 - t0_run

      model_name.append(my_model_name)
      if model_parameters is None:
        model_parameters = model.get_parameters()[None,:]
      else:
        model_parameters = np.append(model_parameters, model.get_parameters()[None,:], axis=0)

      bce_loss.append(1.*loss)
      cell_accuracy.append(1. * my_cell_accuracy)
      grid_accuracy.append(1. * my_grid_accuracy)

      print_log(epoch, elapsed_total, save_file=log_filename,\
          bce_loss=bce_loss[-1], cell_accuracy=cell_accuracy[-1],\
          grid_accuracy=grid_accuracy[-1])

      if early_stopping:
        if my_grid_accuracy == 1.0:
          if solved:
            # if ca is solved two updates in a row, consider it solved
            assert grid_accuracy[-1] == 1.0
            break
          solved = 1
        else:
          solved = 0

    for batch_idx in range(0, epoch_size, minibatch_size):

      optimizer.zero_grad()
      y_pred = model(x[batch_idx:batch_idx+minibatch_size])
      y_target = y[batch_idx:batch_idx+minibatch_size]

      if loss_weighting:

        y_freq = y.mean()
        loss_weight = 0 * y
        loss_weight[y==0] = y_freq
        loss_weight[y==1] = 1. - y_freq
        loss = loss_fn(y_pred, y_target, weight=loss_weight)

      else:
        loss = loss_fn(y_pred, y_target)

      loss.backward()
      optimizer.step()

    t1 = time.time()
          
  if do_logging:
    # run results
    run_df = make_df_from_kwargs(model_name=model_name, parameters_filename=parameters_filename,\
        bce_loss=bce_loss, cell_accuracy=cell_accuracy, grid_accuracy=grid_accuracy)

    run_df.to_csv(run_results_filename)
    np.save(parameters_filename, model_parameters)

    # experiment results (summmary)
    run_filename.append(run_results_filename)
    seed.append(my_seed)
    exp_model_name.append(my_model_name)
    activation_name_list.append(activation_name)
    degree_list.append(degree)
    trainable_count.append(model.count_trainable())
    trainable_neurons.append(trainable_weights)
    trainable_coefficients.append(trainable_activations)
    density_0.append(density)
    parameter_count.append(model.count_parameters())
    model_width.append(model.width)
    model_depth.append(model.depth)
    exp_ca_steps.append(ca_step)
    final_cell_accuracy.append(my_cell_accuracy)
    final_grid_accuracy.append(my_grid_accuracy)
    final_loss.append(loss)
    epoch_batch_size.append(epoch_size)
    mini_batch_size.append(minibatch_size)
    total_samples.append((epoch+1) * epoch_size * minibatch_size)
    model_updates.append((epoch+1) * (epoch_size // minibatch_size))
    wall_time.append(time.time() - t00)
    rulestring.append(ca_rule)
    git_hashs.append(git_hash)
    entry_points.append(entry_point)

    exp_df = make_df_from_kwargs(run_filename=run_filename,\
        seed=seed,\
        model_name=exp_model_name,\
        activation_name=activation_name_list,\
        degree=degree_list,\
        trainable_count=trainable_count,\
        density_0=density_0,\
        parameter_count=parameter_count,\
        model_width=model_width,\
        model_depth=model_depth,\
        exp_ca_steps=exp_ca_steps,\
        rulestring=rulestring,\
        final_cell_accuracy=final_cell_accuracy,\
        final_grid_accuracy=final_grid_accuracy,\
        epoch_size=epoch_batch_size,\
        batch_size=mini_batch_size,\
        total_samples=total_samples,\
        model_updates=model_updates,\
        entry_point=entry_points,\
        git_hash=git_hashs,\
        wall_time=wall_time)

    # concatenate if exp results file already exists
    if os.path.exists(exp_results_filename):
      exp_df_0 = pd.read_csv(exp_results_filename, index_col=0)
      exp_df = pd.concat([exp_df_0, exp_df])

    exp_df.to_csv(exp_results_filename)

def train(**kwargs):
  # training details
  base_seed = kwargs.get("pseudorandom_seed", 13)
  number_runs = kwargs.get("runs", 1)
  epochs = kwargs.get("epochs", 10)
  early_stopping = kwargs.get("early_stopping", False) 
  batches = kwargs.get("batches", [2048, 2048])
  entry_point = kwargs.get("entry_point")
  git_hash = kwargs.get("git_hash")
  if len(batches) == 2:
    epoch_size = batches[0] 
    minibatch_size = batches[1]
  # logging data and rule
  do_logging = kwargs.get("do_logging", False)
  experiment_name = kwargs.get("experiment_name", "")
  ca_rule = kwargs.get("rulestring", "B3/S23")
  disp_every = kwargs.get("disp_every", epochs // 8)
  # data details
  dim = 32 
  ca_steps = kwargs.get("ca_steps", [1])
  densities = kwargs.get("densities", [0.5])
  if len(densities) == 3:
    min_density = densities[0]
    max_density = densities[1]
    step_density = densities[2]

    if step_density > (max_density - min_density):
      step_density = (max_density - min_density)/2

    densities = np.arange(min_density, max_density, step_density)

  lr = kwargs.get("lr", 1e-1)
  # model details
  degree = kwargs.get("degree", 2)
  overcompleteness = kwargs.get("overcompleteness", [1])
  activation_names = kwargs.get("activation", ["ReLU"])
  trainable_weights = kwargs.get("trainable_weights", True)
  trainable_activations = kwargs.get("trainable_activations", False)
  loss_weighting = kwargs.get("loss_weighting", False)

  run_number = -1

  solved = 0
  for activation_name in activation_names:
    for sample_number in range(number_runs):
      #for depth in ca_steps:
        for width in overcompleteness:
          for density in densities:
            for ca_step in ca_steps:
              depth = ca_step
              run_number += 1

              my_seed = base_seed + run_number * BIG_PRIME 
              train_model(my_seed=my_seed,\
                  experiment_name=experiment_name,\
                  run_number=run_number,\
                  lr=lr,\
                  density=density,\
                  disp_every=disp_every,\
                  ca_rule=ca_rule,\
                  activation_name=activation_name,\
                  degree=degree,\
                  depth=depth,\
                  width=width,\
                  trainable_weights=trainable_weights,\
                  trainable_activations=trainable_activations,\
                  do_logging=do_logging,\
                  early_stopping=early_stopping,\
                  epoch_size=epoch_size,\
                  minibatch_size=minibatch_size,\
                  epochs=epochs,
                  dim=dim,\
                  ca_step=ca_step,\
                  git_hash=git_hash,\
                  entry_point=entry_point,\
                  params=None)


if __name__ == "__main__": # pragma: no cover
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--activation", type=str, nargs="+", default=["PolyKAN"],\
      help="designates model, options: PolyKAN, MiniPolyKAN, ReLU, LeakyReLU, CELU, Sigmoid, Tanh,")
  parser.add_argument("-b", "--batches", type=int, nargs="+", default=[10000,8],\
      help="batch and minibatch size. training data is generated on the fly. .")
  parser.add_argument("-c", "--densities", type=float, nargs="+", default=[0.5],\
      help="density range for experiment. Pass one value, or (min, max, step_size) for a range.")
  parser.add_argument("-d", "--do_logging", action="store_true",\
      help="whether to log result to file. default is False.")
  parser.set_defaults(do_logging=False)
  parser.add_argument("-e", "--epochs", type=int, default=100,\
      help="number of 'epochs' to train for.")
  parser.add_argument("-l", "--lr", type=float, default=0.001, \
      help="learning rate, default 0.001")
  parser.add_argument("-m", "--overcompleteness", type=int, nargs="+", default=[1],\
      help="overcompleteness _m_, 3x3 neighborhood layers have 2m and 1x1 dynamics layers have m filter channels")
  parser.add_argument("-n", "--ca_steps", type=int, nargs="+", default=[1],\
      help="trainin on the n-step ca prediction problem, also determines depth of network")
  parser.add_argument("-o", "--loss_weighting", type=bool, default=False,\
      help="whether to use loss weighting (weight loss assigned for off/on cells by state frequency)")
  parser.add_argument("-p", "--pseudorandom_seed", type=int,  default=17,\
      help="seed used as the base seed for pseudorandom number generators")
  parser.add_argument("-r", "--runs", type=int, default=1,\
      help="number of times to run each experimental condition")
  parser.add_argument("-s", "--early_stopping", action="store_true",\
      help="add `-s` flag to use early stopping, (defaults to False with no flag)")
  parser.set_defaults(early_stopping=False)

  parser.add_argument("-t", "--trainable_activations", action="store_false",\
      help="turn off training for activation parameters, default is True.")
  parser.set_defaults(trainable_activations=True)
  parser.add_argument("-u", "--rulestring", type=str, default="B3/S23",\
      help="rulestring defining which Life-like CA rule to use. default is Life 'B3/S23'")
  parser.add_argument("-v", "--disp_every", type=int, default=10,\
      help="how frequently to display progress (and log checkpoints). display ever `disp_every` epochs.")
  parser.add_argument("-w", "--trainable_weights", action="store_false",\
      help="turn off training for neural weights, default is True.")
  parser.set_defaults(trainable_weights=True)
  parser.add_argument("-x", "--experiment_name", type=str, default="default_exp",\
      help="experiment name, used for making the results subfolder. ")

  args = parser.parse_args()
  my_kwargs = dict(args._get_kwargs())

  my_kwargs["git_hash"] = get_git_hash() #.decode("utf8").replace("\n","")

  # store the entry point (cli call) as a string and pass to train
  entry_point = []
  entry_point.append(os.path.split(sys.argv[0])[1])
  args_list = sys.argv[1:]


  # parse the entry point as a string
  my_kwargs["entry_point"] = get_cli_args() 

  train(**my_kwargs)

  print("OK")
