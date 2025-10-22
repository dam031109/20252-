import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f_exp(x):  return np.exp(x)
def df_exp(x): return np.exp(x)

def f_sin(x):  return np.sin(x)
def df_sin(x): return np.cos(x)

def approx_all(f, a, h):
    s1 = (f(a + h) - f(a)) / h
    s2 = (f(a) - f(a - h)) / h
    s3 = (f(a + h) - f(a - h)) / (2.0 * h)
    return s1, s2, s3

def run_case(f, df, a, tag, out_dir_data="data", out_dir_figs="figs"):
    import os
    os.makedirs(out_dir_data, exist_ok=True)
    os.makedirs(out_dir_figs, exist_ok=True)

    hs = np.logspace(-1, -10, 10)  # 1e-1 ~ 1e-10
    rows = []
    true_val = df(a)

    for h in hs:
        s1, s2, s3 = approx_all(f, a, h)
        err1 = abs(s1 - true_val)
        err2 = abs(s2 - true_val)
        err3 = abs(s3 - true_val)
        rows.append([h, s1, s2, s3, true_val, err1, err2, err3])

    cols = ["h","approx1","approx2","approx3","true","err1","err2","err3"]
    T = pd.DataFrame(rows, columns=cols)

    fmt = lambda x: float(f"{x:.10e}")
    for c in cols:
        T[c] = T[c].map(fmt)

    out_table = f"{out_dir_data}/{tag}_table.csv"
    T.to_csv(out_table, index=False)

    i1, i2, i3 = T["err1"].idxmin(), T["err2"].idxmin(), T["err3"].idxmin()
    mins = pd.DataFrame({
        "식": ["(f(a+h)-f(a))/h", "(f(a)-f(a-h))/h", "(f(a+h)-f(a-h))/(2h)"],
        "h": [T.loc[i1,"h"], T.loc[i2,"h"], T.loc[i3,"h"]],
        "최소오차": [T.loc[i1,"err1"], T.loc[i2,"err2"], T.loc[i3,"err3"]],
    })
    out_mins = f"{out_dir_data}/{tag}_mins.csv"
    mins.to_csv(out_mins, index=False)

    plt.figure()
    plt.xscale("log"); plt.yscale("log")
    plt.plot(T["h"], T["err1"], marker="o", label="(f(a+h)-f(a))/h")
    plt.plot(T["h"], T["err2"], marker="s", label="(f(a)-f(a-h))/h")
    plt.plot(T["h"], T["err3"], marker="^", label="(f(a+h)-f(a-h))/(2h)")
    plt.xlabel("h"); plt.ylabel("오차")
    plt.legend()
    plt.tight_layout()
    out_fig = f"{out_dir_figs}/{tag}_error_curve.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()

    return out_table, out_mins, out_fig
