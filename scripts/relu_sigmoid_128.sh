# long runs of relu and sigmoid, for parameter trajectories
# max epochs set slightly higher than the longest successful PReLU run. (sigmoid networks never seem to learn Life)
# relu networks set to width=16, (width=1 networks almost never converge) 
# sigmoids stay minimal at width=1 (haven't seen sigmoid success at any width)
# -c 0.62 corresponds to 0.38 initial cell 'on' density (optimum density from Kenyon & Springer)
python -m src.cann.train --do_logging -s -e 35000 -l 0.001 -m 16 -r 128 -b 8 8 -c 0.62 --a ReLU -x relu_no_knockout_l1_1
python -m src.cann.train --do_logging -s -e 35000 -l 0.001 -m 1 -r 128 -b 8 8 --a Sigmoid -x sigmoid_no_knockout_l1_1
