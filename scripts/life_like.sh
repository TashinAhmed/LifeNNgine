python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a ReLU -n 1 -u B3/S023 -x relu_l1_m_dotlife
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a PolyKAN -n 1 -u B3/S023  -x poly_l1_m_dotlife
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a PReLU -n 1 -u B3/S023  -x prelu_l1_m_dotlife
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a SiLU -n 1 -u B3/S023  -x silu_l1_m_dotlife

python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a ReLU -n 1 -u B368/S245 -x relu_l1_m_morley
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a PolyKAN -n 1 -u B368/S245  -x poly_l1_m_morley
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a PReLU -n 1 -u B368/S245  -x prelu_l1_m_morley
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a SiLU -n 1 -u B368/S245  -x silu_l1_m_morley

python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a ReLU -n 1 -u B1357/S02468 -x relu_l1_m_fredkin
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a PolyKAN -n 1 -u B1357/S02468  -x poly_l1_m_fredkin
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a PReLU -n 1 -u B1357/S02468  -x prelu_l1_m_fredkin
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a SiLU -n 1 -u B1357/S02468  -x silu_l1_m_fredkin

python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a ReLU -n 1 -u B0145/S01234 -x relu_l1_m_asal_d
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a PolyKAN -n 1 -u B0145/S01234  -x poly_l1_m_asal_d
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a PReLU -n 1 -u B0145/S01234  -x prelu_l1_m_asal_d
python -m src.cann.train -d -s -e 25000 -l 0.001 -m 1 2 4 8 -r 4 -b 8 8 -a SiLU -n 1 -u B0145/S01234  -x silu_l1_m_asal_d
