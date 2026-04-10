# additional activation functions

# gaussian, non-monotonic
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 16 -b 8 8 -a Gaussian -n 1 -x gaussian_l1_1

# LU type
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 16 -b 8 8 -a Mish -n 1 -x mish_l1_1
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 16 -b 8 8 -a GELU -n 1 -x gelu_l1_1
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 16 -b 8 8 -a SELU -n 1 -x selu_l1_1
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 16 -b 8 8 -a ELU -n 1 -x elu_l1_1
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 16 -b 8 8 -a Softplus -n 1 -x elu_l1_1

# Softsign. has sigmoid like curve 
python -m src.cann.train -d -s -e 125000 -l 0.001 -m 1 -r 16 -b 8 8 -a Softsign -n 1 -x elu_l1_1
