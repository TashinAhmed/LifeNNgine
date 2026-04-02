import os
from pathlib import Path

import numpy as np
import pandas as pd

import sklearn

import matplotlib
import matplotlib.pyplot as plt

# useful for type hints
from sklearn.decomposition._pca import PCA
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

FILEPATH = Path(__file__)

ROOT_DIR = os.path.split(os.path.split(os.path.split(FILEPATH)[0])[0])[0]

def get_params_losses_accuracies(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
  params = None

  for filename in df["run_filename"]:
    df_run = pd.read_csv(os.path.join(ROOT_DIR, filename))

    param_filepath = os.path.join(ROOT_DIR, (df_run["parameters_filename"].unique()[0]))

    if params is None:
      params = np.load(param_filepath)
      losses = df_run["bce_loss"]
      accuracies = df_run["grid_accuracy"]
      ends = 0 * losses
      ends[0] = -1
      ends[ends.shape[-1]-1] = 1
    else:
      loaded_params = np.load(param_filepath)
      params = np.append(params, loaded_params, axis=0)
      losses = np.append(losses, df_run["bce_loss"], axis=0)
      accuracies = np.append(accuracies, df_run["grid_accuracy"], axis=0)
      temp_ends = 0 * df_run["bce_loss"]
      temp_ends[0] = -1
      temp_ends[temp_ends.shape[-1]-1] = 1
      ends = np.append(ends, temp_ends, axis=0)

  return params, losses, accuracies, ends

def compute_dlosses(losses: np.ndarray)->np.ndarray:

  dlosses = 0 * losses
  for kk in range(1,losses.shape[0]):
    # dloss calculated as forward difference
    dlosses[kk] = losses[kk]-losses[kk-1]

  return dlosses

def fit_pca(params:np.ndarray, number_components: int=10) -> PCA:

  pca = sklearn.decomposition.PCA(n_components=number_components)
  pca.fit(params)

  return pca

def fit_return_principal_components(params: np.ndarray,\
    number_components: int=10) -> np.ndarray:

  pca = fit_principalpca(params, number_components)
  pcs = pca.transform(params)
  
  return pcs  

def plot_big_picture():
  pass

def plot_trajectory_on_ax(ax_main: Axes, df: pd.DataFrame, **kwargs) -> Axes:

  lw = kwargs.get("lw", 3)
  my_cmap =  kwargs.get("cmap", plt.get_cmap("plasma"))
  model_width = kwargs.get("model_width", 1)
  model_depth = kwargs.get("model_depth", 1)
  pc_indices = kwargs.get("pc_indices", [0,1])
  time_cmap = kwargs.get("time_cmap", plt.get_cmap("viridis"))
  color_by_dloss = kwargs.get("color_by_dloss", False)
  main_title = kwargs.get("main_title", "P.C.A. Training Trajectories")
  success_df = df[df["final_grid_accuracy"]==1.0]
  fontsize = kwargs.get("fontsize", 16)

  params, losses, accuracies, ends = get_params_losses_accuracies(df)

  pca = fit_pca(params) 
  pcs = pca.transform(params)

  my_color = my_cmap(losses)
  time_color = time_cmap(128)

  ax_main.scatter(pcs[:,pc_indices[0]], pcs[:,pc_indices[1]], marker="o",\
      color=my_color, alpha=0.025, s=32)

  for ii in range(ends.shape[0]):
    start_marker = "s"
    if ends[ii] == 1:
      my_marker = "X" if accuracies[ii] < 1.0 else "o"
        
      ax_main.scatter(pcs[ii,pc_indices[0]], pcs[ii,pc_indices[1]], marker=my_marker,\
          color=my_color[ii], alpha=0.75, s=256)

      if (ii+1) < ends.shape[0]:
        jj = ii+1
        ax_main.scatter(pcs[jj,pc_indices[0]], pcs[jj,pc_indices[1]], marker=start_marker,\
            color=my_color[jj], alpha=0.75, s=64)
        
  ax_main.set_ylabel(f"Principal Component {pc_indices[1]}", fontsize=fontsize)
  ax_main.set_xticklabels("")

  return ax_main

def plot_traj_proj_on_axes(ax_main: Axes, ax_projv: Axes, ax_projh: Axes, \
    df: pd.DataFrame, **kwargs) -> tuple[Figure, GridSpec]:

  lw = kwargs.get("lw", 3)
  my_cmap =  kwargs.get("cmap", plt.get_cmap("plasma"))
  model_width = kwargs.get("model_width", 1)
  model_depth = kwargs.get("model_depth", 1)
  pc_indices = kwargs.get("pc_indices", [0,1])
  time_cmap = kwargs.get("time_cmap", plt.get_cmap("viridis"))
  color_by_dloss = kwargs.get("color_by_dloss", False)
  main_title = kwargs.get("main_title", "P.C.A. Training Trajectories")
  fontsize = kwargs.get("fontsize", 16)
  fig_side = kwargs.get("fig_side", 3)
  marker_size = kwargs.get("marker_size", 32)
  small_marker_size = kwargs.get("small_marker_size", 2)
  clear_labels = kwargs.get("clear_labels", None)

  success_df = df[df["final_grid_accuracy"]==1.0]

  params, losses, accuracies, ends = get_params_losses_accuracies(df)

  pca = fit_pca(params) 
  pcs = pca.transform(params)

  my_color = my_cmap(losses)

  if (color_by_dloss):

    time_color = 0 * my_color
    dlosses = compute_dlosses(losses)
    pos_dloss = dlosses[dlosses > 0]
    neg_dloss = dlosses[dlosses <= 0]
    pos_dloss = pos_dloss / np.max(np.abs(dlosses))
    pos_dloss = np.abs(pos_dloss / np.max(np.abs(dlosses)))

    time_color[dlosses > 0] = time_cmap(256*(1-pos_dloss))
    time_color[dlosses <= 0] = time_cmap(256*neg_dloss)
  else:
    time_color = time_cmap(128)
    
  ax_main.scatter(pcs[:,pc_indices[0]], pcs[:,pc_indices[1]], marker="o",\
      color=my_color, alpha=0.025, s=marker_size)
  loss_list = losses.tolist()

  ax_projv.scatter(loss_list, pcs[:,pc_indices[1]], s=small_marker_size, color=time_color,\
      alpha = 0.1)
  ax_projh.scatter(pcs[:,pc_indices[0]], losses, s=small_marker_size, color=time_color,\
      alpha = 0.1)


  for ii in range(ends.shape[0]):
    start_marker = "s"
    if ends[ii] == 1:
      my_marker = "x" if accuracies[ii] < 1.0 else "o"
      my_scale = 6 if my_marker == "x" else 2
        
      ax_main.scatter(pcs[ii,pc_indices[0]], pcs[ii,pc_indices[1]], marker=my_marker,\
          color=my_color[ii], alpha=0.75, s=marker_size*my_scale)
        
      ax_projv.scatter(loss_list[ii], pcs[ii,pc_indices[1]], marker=my_marker,\
          color=my_color[ii], alpha=0.75, s=small_marker_size*my_scale*2)
        
      ax_projh.scatter(pcs[ii,pc_indices[0]], losses[ii], marker=my_marker,\
          color=my_color[ii], alpha=0.75, s=small_marker_size*my_scale*2)

      if (ii+1) < ends.shape[0]:
        jj = ii+1
        ax_main.scatter(pcs[jj,pc_indices[0]], pcs[jj,pc_indices[1]], marker=start_marker,\
            color=my_color[jj], alpha=0.75, s=small_marker_size*my_scale)
      
        ax_projv.scatter(loss_list[jj], pcs[jj,pc_indices[1]], marker=start_marker,\
            color=my_color[jj], alpha=0.75, s=small_marker_size*my_scale)
          
        ax_projh.scatter(pcs[jj,pc_indices[0]], losses[jj], marker=start_marker,\
            color=my_color[jj], alpha=0.75, s=small_marker_size*2)
        

  ax_projv.invert_xaxis() #([1.5, 0.75, 0.0])
  ax_projh.set_xlabel(f"Principal Component {pc_indices[0]}", fontsize=fontsize)
  ax_main.set_ylabel(f"Principal Component {pc_indices[1]}", fontsize=fontsize)
  ax_projv.set_xlabel(f"Loss", fontsize=fontsize)
  ax_projh.set_ylabel(f"Loss", fontsize=fontsize)
  ax_main.set_xticklabels("")
  ax_projv.yaxis.tick_right()
  ax_main.set_title(f"{main_title}", fontsize=fontsize+1)

  if clear_labels is not None:
    for axx, clear_label in zip([ax_main, ax_projv, ax_projh], clear_labels):
      if clear_label[0]:
        axx.set_title("")
      if clear_label[1]:
        axx.set_ylabel("")
      if clear_label[2]:
        axx.set_xlabel("")

def plot_trajectory_projection(df: pd.DataFrame, **kwargs) -> tuple[Figure, GridSpec]:

  lw = kwargs.get("lw", 3)
  my_cmap =  kwargs.get("cmap", plt.get_cmap("plasma"))
  model_width = kwargs.get("model_width", 1)
  model_depth = kwargs.get("model_depth", 1)
  pc_indices = kwargs.get("pc_indices", [0,1])
  time_cmap = kwargs.get("time_cmap", plt.get_cmap("viridis"))
  color_by_dloss = kwargs.get("color_by_dloss", False)
  main_title = kwargs.get("main_title", "P.C.A. Training Trajectories")
  fontsize = kwargs.get("fontsize", 16)

  fig_side = kwargs.get("fig_side", 3)

  fig = plt.figure(layout="constrained", figsize=(fig_side, fig_side))
  gs = matplotlib.gridspec.GridSpec(2,2, figure=fig,\
          width_ratios=[0.8,0.2],\
          height_ratios=[0.8, 0.2])

  ax_main = fig.add_subplot(gs[0,0])
  ax_projv = fig.add_subplot(gs[0,1])
  ax_projh = fig.add_subplot(gs[1,0])    

  success_df = df[df["final_grid_accuracy"]==1.0]

  params, losses, accuracies, ends = get_params_losses_accuracies(df)

  pca = fit_pca(params) 
  pcs = pca.transform(params)

  my_color = my_cmap(losses)

  if (color_by_dloss):

    time_color = 0 * my_color
    dlosses = compute_dlosses(losses)
    pos_dloss = dlosses[dlosses > 0]
    neg_dloss = dlosses[dlosses <= 0]
    pos_dloss = pos_dloss / np.max(np.abs(dlosses))
    pos_dloss = np.abs(pos_dloss / np.max(np.abs(dlosses)))

    time_color[dlosses > 0] = time_cmap(256*(1-pos_dloss))
    time_color[dlosses <= 0] = time_cmap(256*neg_dloss)
  else:
    time_color = time_cmap(128)
    
  ax_main.scatter(pcs[:,pc_indices[0]], pcs[:,pc_indices[1]], marker="o",\
      color=my_color, alpha=0.025, s=32)
  loss_list = losses.tolist()

  ax_projv.scatter(loss_list, pcs[:,pc_indices[1]], s=2, color=time_color,\
      alpha = 0.1)
  ax_projh.scatter(pcs[:,pc_indices[0]], losses, s=2, color=time_color,\
      alpha = 0.1)


  for ii in range(ends.shape[0]):
    start_marker = "s"
    if ends[ii] == 1:
      my_marker = "X" if accuracies[ii] < 1.0 else "o"
        
      ax_main.scatter(pcs[ii,pc_indices[0]], pcs[ii,pc_indices[1]], marker=my_marker,\
          color=my_color[ii], alpha=0.75, s=256)
        
      ax_projv.scatter(loss_list[ii], pcs[ii,pc_indices[1]], marker=my_marker,\
          color=my_color[ii], alpha=0.75, s=32)
        
      ax_projh.scatter(pcs[ii,pc_indices[0]], losses[ii], marker=my_marker,\
          color=my_color[ii], alpha=0.75, s=32)

      if (ii+1) < ends.shape[0]:
        jj = ii+1
        ax_main.scatter(pcs[jj,pc_indices[0]], pcs[jj,pc_indices[1]], marker=start_marker,\
            color=my_color[jj], alpha=0.75, s=64)
      
        ax_projv.scatter(loss_list[jj], pcs[jj,pc_indices[1]], marker=start_marker,\
            color=my_color[jj], alpha=0.75, s=4)
          
        ax_projh.scatter(pcs[jj,pc_indices[0]], losses[jj], marker=start_marker,\
            color=my_color[jj], alpha=0.75, s=4)
        
  ax_projv.invert_xaxis() #([1.5, 0.75, 0.0])
  ax_projh.set_xlabel(f"Principal Component {pc_indices[0]}", fontsize=fontsize)
  ax_main.set_ylabel(f"Principal Component {pc_indices[1]}", fontsize=fontsize)
  ax_projv.set_xlabel(f"Loss", fontsize=fontsize)
  ax_projh.set_ylabel(f"Loss", fontsize=fontsize)
  ax_main.set_xticklabels("")
  ax_projv.yaxis.tick_right()
  fig.suptitle(f"{main_title}", fontsize=fontsize+4)

  return fig, gs
  
def plot_param_trajectory(df_filename: list[str],\
    ax_title: str="",\
    my_cmap: ListedColormap=plt.get_cmap("plasma"),\
    only_success: bool=False,\
    pc_indices: list=[0,1]) -> [Figure, Axes]:

  if type(df_filename) == str:
    df_filename = [df_filename]

  df = pd.DataFrame()
  for df_fn in df_filename:
    try:
      dfa = pd.read_csv(os.path.join(ROOT_DIR, df_fn))
    except:
      dfa = pd.read_csv(df_fn)
    df = pd.concat([df,dfa])

  if only_success:
    df = df[df["final_grid_accuracy"]==1.0]

  figs, axes = [], []
  for width, depth in zip(df["model_width"].unique(), df["model_depth"].unique()):

    params, losses, accuracies, ends = get_params_losses_accuracies(df[df["model_width"] == width][df["model_depth"] == depth])

    pcs = fit_return_principal_components(params) 

    fig, ax = plt.subplots(1,1, figsize=(8,8))

    my_color = my_cmap(losses)

    ax.scatter(pcs[:,pc_indices[0]], pcs[:,pc_indices[1]], marker="o",\
        color=my_color, alpha=0.025, s=32)

    #edge_colors = ["r" if acc < 1.0 else "k" for acc in accuracies]
    print(ends.shape, accuracies.shape, my_color.shape, pcs.shape)
    for ii in range(ends.shape[0]):
      if ends[ii] == 1:
        my_marker = "X" if accuracies[ii] < 1.0 else "o"
        ax.scatter(pcs[ii,pc_indices[0]], pcs[ii,pc_indices[1]], marker=my_marker,\
            color=my_color[ii], alpha=0.75, s=256)


    ax.set_title(ax_title)

    figs.append(fig)
    axes.append(ax)

  return figs, axes

if __name__ == "__main__":

  pass
  print(ROOT_DIR)

