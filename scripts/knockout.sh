# everything (activation function and neural weight parameters)
python -m src.cann.train --do_logging -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 --a PReLU -x prelu_no_knockout_l1_1
python -m src.cann.train --do_logging -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 --a PolyKAN -x poly_no_knockout_l1_1
python -m src.cann.train --do_logging -s -e 125000 -l 0.001 -m 1 -r 8 -b 8 8 --a AdaptiveSigmoid -x asigmoid_no_knockout_l1_1

# activation function parameters not learnable
python -m src.cann.train --do_logging -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 --a PReLU --trainable_activations -x prelu_act_knockout_coef_l1_1
python -m src.cann.train --do_logging -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 --a PolyKAN --trainable_activations -x poly_act_knockout_coef_l1_1
python -m src.cann.train --do_logging -s -e 125000 -l 0.001 -m 1 -r 8 -b 8 8 --a AdaptiveSigmoid --trainable_activations -x asigmoid_act_knockout_coef_l1_1

# neural weight parameters not learnable
python -m src.cann.train --do_logging -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 --a PolyKAN --trainable_weights -x poly_weight_knockout_neur_l1_1
python -m src.cann.train --do_logging -s -e 125000 -l 0.001 -m 1 -r 128 -b 8 8 --a PReLU --trainable_weights -x prelu_weight_knockout_neur_l1_1
python -m src.cann.train --do_logging -s -e 125000 -l 0.001 -m 1 -r 8 -b 8 8 --a AdaptiveSigmoid --trainable_weights -x asigmoid_weight_knockout_neur_l1_1

