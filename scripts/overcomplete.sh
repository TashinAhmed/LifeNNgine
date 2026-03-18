python -m src.cann.train -d -s -e 25000 -l 0.001 -m 2 4 8 -r 16 -b 8 8 -a PolyKAN -n 1 -x poly_l1_m
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 2 4 8 -r 16 -b 8 8 -a PReLU -n 1 -x prelu_l1_m
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 2 4 8 -r 16 -b 8 8 -a ReLU -n 1 -x relu_l1_m
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 2 4 8 -r 16 -b 8 8 -a SiLU -n 1 -x silu_l1_m
