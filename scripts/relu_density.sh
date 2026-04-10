# relu success rates are so low, we need many more samples to get smoother histogram.
python -m src.cann.train -d -s -e 50000 -l 0.001 -m 1 -r 128 -b 8 8 -c 0.05 1.0 0.05 -a ReLU -x density_relu_128runs_l1_1
python -m src.cann.train -d -s -e 50000 -l 0.001 -m 1 -r 128 -b 8 8 -c 0.2 .51 0.01 -a ReLU -x density_relu_128runs_l1_1_zoom
python -m src.cann.train -d -s -e 50000 -l 0.001 -m 2 -r 128 -b 8 8 -c 0.05 1.0 0.05 -a ReLU -x density_relu_128runs_l1_2
