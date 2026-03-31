import os
from pathlib import Path

import numpy as np
import pandas as pd

import sklearn

import matplotlib.pyplot as plt

# useful for type hints
from sklearn.decomposition._pca import PCA
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

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

def fit_pca(params:np.ndarray, number_components: int=10) -> PCA:

  pca = sklearn.decomposition.PCA(n_components=number_components)
  pca.fit(params)

  return pca

def fit_return_principal_components(params: np.ndarray,\
    number_components: int=10) -> np.ndarray:

  pca = fit_pca(params, number_components)
  pcs = pca.transform(params)
  
  return pcs  

  
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

