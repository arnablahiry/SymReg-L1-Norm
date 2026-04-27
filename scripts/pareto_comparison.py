"""
pareto_comparison.py  —  Single-plot complexity/accuracy comparison
====================================================================
X-axis: expression complexity (node count)
Y-axis: mean per-cosmology R² on the held-out test set (identical to
        cosmo_symreg_direct.py / nn_emulator.py test split)

Overlays:
  * PySR Direct pareto front  — one point per hall-of-fame entry, up to and
    including the 'best' complexity reported in best_expression.txt.
  * Horizontal lines for the non-symbolic / non-pareto methods:
      - NN emulator            (mean per-cosmology R²)
      - PhySO PCA SR           (mean per-cosmology R²)
      - PhySO 2G−1G shape SR   (mean per-cosmology R², if available)
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"]  = ["Computer Modern Roman"]


# =============================================================================
# PATHS
# =============================================================================

ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYSR_DIR     = f"{ROOT}/methods/pysr/direct/outputs_50nodes_new"
PYSR_HOF     = f"{ROOT}/methods/pysr/direct/runs/20260323_020048_jUwwTB_50nodes_new_best/hall_of_fame.csv"
PYSR_BEST    = f"{PYSR_DIR}/best_expression.txt"
PYSR_SCALER  = f"{PYSR_DIR}/cosmo_scaler.pkl"

NN_DIR       = f"{ROOT}/methods/nn/outputs"
NN_TRUTH     = f"{NN_DIR}/nn_test_truth.npy"
NN_COSMO     = f"{NN_DIR}/nn_test_cosmo.npy"
NN_PREDS     = f"{NN_DIR}/nn_test_predictions.npy"

COMP_DIR     = f"{ROOT}/comparison"
PHYSO_PCA    = f"{COMP_DIR}/physo_pca_test_predictions.npy"
PHYSO_2G1G   = f"{COMP_DIR}/physo_shape_test_predictions.npy"

OUT_PATH     = f"{COMP_DIR}/pareto_comparison.png"

# Complexity of the expression saved in best_expression.txt.
# Determined by manual inspection (matches the hall_of_fame row whose
# 1.1346143 / 0.0864375 constants survive into the canonical best_expression).
BEST_COMPLEXITY = 34


# =============================================================================
# HELPERS
# =============================================================================

def _mean_r2(truth, pred):
    """Mean per-cosmology R² across the 32 test cosmologies."""
    return float(np.mean([r2_score(truth[i], pred[i]) for i in range(len(truth))]))


def _build_pysr_eval(scaler, mean_peak, xc_eps=0.0005):
    """Return a function that evaluates any hall_of_fame Julia-style expression
    on the test set and returns a (32, 160) prediction array."""
    x_arr = np.arange(160, dtype=np.float64)
    x_c = x_arr - mean_peak
    x_c_safe = np.sign(x_c) * (np.abs(x_c) + xc_eps)

    def _sanitise(expr):
        # PySR uses Julia-style `square(.)`; Python works fine once we map it.
        # `^` → `**` in case some rows use it. `square(x)` is already a name
        # we'll provide in the eval namespace.
        return expr.replace("^", "**")

    def evaluate(expr_str, cosmo_rows):
        code = compile(_sanitise(expr_str), "<hof>", "eval")
        preds = np.empty((len(cosmo_rows), 160), dtype=np.float64)
        for i, c in enumerate(cosmo_rows):
            std = scaler.transform([c])[0]
            env = dict(
                Omega_m_std=std[0], sigma_8_std=std[1], w_std=std[2],
                x_c=x_c_safe,
                exp=np.exp, log=np.log, sin=np.sin, cos=np.cos,
                sqrt=np.sqrt, square=np.square,
                Abs=np.abs, abs=np.abs,
            )
            preds[i] = eval(code, {"__builtins__": {}}, env)
        return preds

    return evaluate


# =============================================================================
# PYSR PARETO
# =============================================================================

def pysr_pareto(truth, cosmo_test):
    hof = pd.read_csv(PYSR_HOF)

    with open(PYSR_SCALER, "rb") as f:
        scaler = pickle.load(f)

    with open(PYSR_BEST) as f:
        txt = f.read()
    mean_peak = float(re.search(r"x\s*-\s*([\-\d\.]+)", txt).group(1))
    best_expr = re.search(r"Expression\s*:\s*(.+)", txt).group(1).strip()

    best_complexity = BEST_COMPLEXITY
    print(f"  best complexity (from best_expression.txt) : {best_complexity}")

    hof = hof[hof["Complexity"] <= best_complexity].copy()
    evaluator = _build_pysr_eval(scaler, mean_peak)

    complexities, mean_r2s = [], []
    for _, row in hof.iterrows():
        expr = row["Equation"]
        try:
            preds = evaluator(expr, cosmo_test)
            r2 = _mean_r2(truth, preds)
        except Exception as e:
            print(f"    skipped complexity={row['Complexity']} ({type(e).__name__}: {e})")
            continue
        complexities.append(int(row["Complexity"]))
        mean_r2s.append(r2)
        print(f"    complexity={row['Complexity']:>3}   mean R²={r2:.4f}")

    return np.array(complexities), np.array(mean_r2s), best_complexity


# =============================================================================
# PLOT
# =============================================================================

def main():
    truth      = np.load(NN_TRUTH)
    cosmo_test = np.load(NN_COSMO)
    nn_preds   = np.load(NN_PREDS)

    print("Evaluating PySR Direct pareto front on test set ...")
    comps, r2s, best_c = pysr_pareto(truth, cosmo_test)

    nn_mean_r2 = _mean_r2(truth, nn_preds)
    print(f"\nNN emulator            mean R² : {nn_mean_r2:.4f}")

    phy_pca_r2 = None
    if os.path.exists(PHYSO_PCA):
        phy_pca_r2 = _mean_r2(truth, np.load(PHYSO_PCA))
        print(f"PhySO PCA SR           mean R² : {phy_pca_r2:.4f}")
    phy_2g1g_r2 = None
    if os.path.exists(PHYSO_2G1G):
        phy_2g1g_r2 = _mean_r2(truth, np.load(PHYSO_2G1G))
        print(f"PhySO 2G−1G shape SR   mean R² : {phy_2g1g_r2:.4f}")
    else:
        print("PhySO 2G−1G predictions not found yet — horizontal line skipped.")

    # -----  Plot  -----
    fig, ax = plt.subplots(figsize=(6, 4))

    pysr_color = "xkcd:red"
    ax.plot(comps, r2s, "o-", color=pysr_color, lw=1.6, ms=6,
            label="PySR Direct (pareto)", zorder=4)
    best_idx = np.argmax(comps == best_c)
    best_r2 = r2s[best_idx]
    ax.plot([comps[best_idx]], [best_r2], marker="o",
            mfc="white", mec=pysr_color, ms=11, mew=2.0,
            color=pysr_color, ls=(0, (5, 2)), lw=1.4,
            label=rf"PySR Direct Best (complexity = {best_c}, $R^2 = {best_r2:.4f}$)",
            zorder=5)

    hlines = [
        ("PySR Direct Best",       best_r2,      pysr_color,   (0, (5, 2))),
        ("NN Emulator",            nn_mean_r2,   "tab:green",  "--"),
        (r"$\Phi-SO$ PCA SR",           phy_pca_r2,   "tab:blue",   ":"),
        (r"$\Phi-SO$ $2\mathcal{G}-1\mathcal{G}$ Shape SR",   phy_2g1g_r2,  "tab:orange", (0, (3, 1, 1, 1))),
    ]
    for name, val, color, ls in hlines:
        if val is None:
            continue
        if name == "PySR Direct Best":
            ax.axhline(val, color=color, ls=ls, lw=1.4, alpha=0.7, zorder=3)
            continue
        ax.axhline(val, color=color, ls=ls, lw=1.8,
                   label=rf"{name}   ($R^2 = {val:.4f}$)")

    ax.set_xlabel("Node Complexity")
    ax.set_ylabel(r"Test $\bar{R^2}$")
    #ax.set_yscale('log')
    ax.set_ylim(0.97,1.001)
    ax.set_xlim(17,35)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
