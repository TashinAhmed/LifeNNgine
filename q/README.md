# Getting started

using `virtualenv` and pip (for now)
the `requirements.txt` file is a bit of a monster at the moment. (TODO: prune `requirements.txt` and use `uv`?)

```
virtualenv my_env
source my_env/bin/activate
pip install -r requirements.txt
pip install -e .
```

currently using CARLE as the CA engine

```
git clone git@github.com:riveSunder/carle.git
cd carle 
pip install -e .
cd ../
```


# Entry point for experiments

The entry point for running experiments is `src/cann/train.py`.

There are quite a few options when invoking an experiment:

``` 
options:
  -h, --help            show this help message and exit
  -a ACTIVATION [ACTIVATION ...], --activation ACTIVATION [ACTIVATION ...]
                        designates model, options: PolyKAN, MiniPolyKAN, ReLU, LeakyReLU, CELU, Sigmoid, Tanh,
  -b BATCHES [BATCHES ...], --batches BATCHES [BATCHES ...]
                        batch and minibatch size. training data is generated on the fly. .
  -c DENSITIES [DENSITIES ...], --densities DENSITIES [DENSITIES ...]
                        density range for experiment. Pass one value, or (min, max, step_size) for a range.
  -d, --do_logging      whether to log result to file. default is False.
  -e EPOCHS, --epochs EPOCHS
                        number of 'epochs' to train for.
  -l LR, --lr LR        learning rate, default 0.001
  -m OVERCOMPLETENESS [OVERCOMPLETENESS ...], --overcompleteness OVERCOMPLETENESS [OVERCOMPLETENESS ...]
                        overcompleteness _m_, 3x3 neighborhood layers have 2m and 1x1 dynamics layers have m filter channels
  -n CA_STEPS [CA_STEPS ...], --ca_steps CA_STEPS [CA_STEPS ...]
                        trainin on the n-step ca prediction problem, also determines depth of network
  -o LOSS_WEIGHTING, --loss_weighting LOSS_WEIGHTING
                        whether to use loss weighting (weight loss assigned for off/on cells by state frequency)
  -p PSEUDORANDOM_SEED, --pseudorandom_seed PSEUDORANDOM_SEED
                        seed used as the base seed for pseudorandom number generators
  -r RUNS, --runs RUNS  number of times to run each experimental condition
  -s, --early_stopping  add `-s` flag to use early stopping, (defaults to False with no flag)
  -t, --trainable_activations
                        turn off training for activation parameters, default is True.
  -u RULESTRING, --rulestring RULESTRING
                        rulestring defining which Life-like CA rule to use. default is Life 'B3/S23'
  -v DISP_EVERY, --disp_every DISP_EVERY
                        how frequently to display progress (and log checkpoints). display ever `disp_every` epochs.
  -w, --trainable_weights
                        turn off training for neural weights, default is True.
  -x EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        experiment name, used for making the results subfolder.

```

# Density experiments

```
# L(1,1) density experiment from 0.05 to 0.95 initial 'on' cells
python -m src.cann.train -d -s -e 12500 -l 0.001 -m 1 -r 128 -b 8 8 -c 0.05 1.0 0.05 -a PolyKAN -x density_polykan_l1_1
```

```
# L(1,1) density experiment 'zoomed in' to 0.28 to 0.5 initial 'on' cells
python -m src.cann.train -d -s -e 12500 -l 0.001 -m 1 -r 128 -b 8 8 -c 0.2 .51 0.01 -a PolyKAN -x density_polykan_l1_1_zoom
```

```
# L(1,2) (x2 overcomplete) density experiment from 0.05 to 0.95 initial 'on' cells
python -m src.cann.train -d -s -e 12500 -l 0.001 -m 2 -r 128 -b 8 8 -c 0.05 1.0 0.05 -a PolyKAN -x density_polykan_l1_2 
```

```
# L(1,1) density experiment from 0.05 to 0.95 initial 'on' cells
python -m src.cann,train -d -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 -c 0.05 1.0 0.05 -a ReLU -x density_relu_l1_1
```

```
# L(1,1) density experiment 'zoomed in' to 0.28 to 0.5 initial 'on' cells
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 -c 0.2 .51 0.01 -a ReLU -x density_relu_l1_1_zoom 
```

```
# L(1,2) (x2 overcomplete) density experiment from 0.05 to 0.95 initial 'on' cells
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 2 -r 128 -b 8 8 -c 0.05 1.0 0.05 -a ReLU -x density_relu_l1_2 
```

# Learning knockout/ablation

L(1,1) PolyKAN experiment with no learning on activation function coefficients

```
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 -a PolyKAN -t -x poly_act_knockout
```

L(1,1) PolyKAN experiment with no learning on neural weights

```
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 -a PolyKAN -w -x poly_neuron_knockout
```

L(1,1) PReLU experiment with no learning on activation function coefficients

```
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 -a PReLU -t -x prelu_act_knockout
```

L(1,1) PReLU experiment with no learning on neural weights

```
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 -a PReLU -w -x prelu_neuron_knockout
```

# n-step prediction

L(n,1) with steps n=[1,2,3,4,5] PolyKAN 

```
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 -c 0.38 -a PolyKAN -n 1 2 3 4 5 -x n12345_polykan_ln_1
```

L(n,1) with steps n=[1,2,3,4,5] ReLU 

```
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 2 4 8 16 32 -r 128 -b 8 8 -c 0.38 -a ReLU -n 1 2 3 4 5 -x n12345_relu_ln_1
```

L(n,1) with steps n=[1,2,3,4,5] PReLU 

```
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 -c 0.38 -a ReLU -n 1 2 3 4 5 -x n12345_prelu_ln_1
```

L(n,m) with steps n=[1,2,3,4,5] PolyKAN m=[2,4,8,16,32] 

```
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 2 4 8 16 32 -r 128 -b 8 8 -c 0.38 -a PolyKAN -n 1 2 3 4 5 -x n12345_polykan_ln_m
```

L(n,m) with steps n=[1,2,3,4,5] ReLU m=[2,4,8,16,32] 

```
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 2 4 8 16 32  -r 128 -b 8 8 -c 0.38 -a ReLU -n 1 2 3 4 5 -x n12345_relu_ln_m
```

L(n,m) with steps n=[1,2,3,4,5] PReLU m=[2,4,8,16,32] 

```
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 2 4 8 16 32  -r 128 -b 8 8 -c 0.38 -a PReLU -n 1 2 3 4 5 -x n12345_prelu_ln_m
```

^^ same pattern for L(n,4), L(n,8), L(n,16), L(n,32)

