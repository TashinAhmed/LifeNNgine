python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 -r 16 -b 8 8 -c 0.05 1.0 0.05 -a PolyKAN -x density_polykan_l1_1
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 -r 16 -b 8 8 -c 0.2 .51 0.01 -a PolyKAN -x density_polykan_l1_1_zoom
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 2 -r 16 -b 8 8 -c 0.05 1.0 0.05 -a PolyKAN -x density_polykan_l1_2

python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 -r 16 -b 8 8 -c 0.05 1.0 0.05 -a PReLU -x density_prelu_l1_1
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 -r 16 -b 8 8 -c 0.2 .51 0.01 -a PReLU -x density_prelu_l1_1_zoom
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 2 -r 16 -b 8 8 -c 0.05 1.0 0.05 -a PReLU -x density_prelu_l1_2

python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 -r 16 -b 8 8 -c 0.05 1.0 0.05 -a ReLU -x density_relu_l1_1
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 -r 16 -b 8 8 -c 0.2 .51 0.01 -a ReLU -x density_relu_l1_1_zoom
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 2 -r 16 -b 8 8 -c 0.05 1.0 0.05 -a ReLU -x density_relu_l1_2

python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 -r 16 -b 8 8 -c 0.05 1.0 0.05 -a SiLU -x density_silu_l1_1
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 -r 16 -b 8 8 -c 0.2 .51 0.01 -a SiLU -x density_silu_l1_1_zoom
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 2 -r 16 -b 8 8 -c 0.05 1.0 0.05 -a SiLU -x density_silu_l1_2
