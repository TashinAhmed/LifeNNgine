python -m src.cann.perturb --filepath results/poly_l1_1/exp.csv --epoch_index -1 0 -e 67500 -r 10  --k_sign_perturbations 0 1 2 3 4 5 6 7 8 --uniform_magnitude 0.0 0.25 0.5 0.75 1.0 -x k_perturb_retrain_polykan_l1_1

python -m src.cann.pertur --filepath results/prelu_l1_1/exp.csv b --epoch_index -1 0 -e 67500 -r 10  --k_sign_perturbations 0 1 2 3 4 5 6 7 8 --uniform_magnitude 0.0 0.25 0.5 0.75 1.0 -x k_perturb_retrain_prelu_l1_1

python -m src.cann.perturb --filepath results/relu_l1_1/exp.csv  --epoch_index -1 0 -e 67500 -r 10  --k_sign_perturbations 0 1 2 3 4 5 6 7 8 --uniform_magnitude 0.0 0.25 0.5 0.75 1.0 -x k_perturb_retrain_relu_l1_2

python -m src.cann.perturb --filepath results/silu_l1_1/exp.csv  --epoch_index -1 0 -e 67500 -r 10  --k_sign_perturbations 0 1 2 3 4 5 6 7 8 --uniform_magnitude 0.0 0.25 0.5 0.75 1.0 -x k_perturb_retrain_silu_l1_2

