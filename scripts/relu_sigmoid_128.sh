# long runs of relu and sigmoid, for parameter trajectories
# -c 0.62 corresponds to 0.38 initial cell 'on' density (optimum density from Kenyon & Springer)
python -m src.cann.train --do_logging -s -e 35000 -l 0.001 -m 1 -r 128 -b 8 8 -c 0.62 --a ReLU -x relu_no_knockout_l1_1
python -m src.cann.train --do_logging -s -e 35000 -l 0.001 -m 1 -r 128 -b 8 8 --a Sigmoid -x sigmoid_no_knockout_l1_1
