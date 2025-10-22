import os, numpy as np
from numdiff import f_exp, df_exp, f_sin, df_sin, run_case

os.makedirs("data", exist_ok=True)
os.makedirs("figs", exist_ok=True)

run_case(f_exp, df_exp, a=1.0, tag="exp_a1")
run_case(f_sin, df_sin, a=float(np.pi/4), tag="sin_api4")
