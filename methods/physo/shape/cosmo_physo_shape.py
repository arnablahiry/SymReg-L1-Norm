"""
cosmo_physo_shape.py  —  PhySO SR on "2 Gaussians − 1 Gaussian" Shape Decomposition
======================================================================================
Fits the parametric model

    L1(x; Om, s8, w) = G1(x; A1, mu1, sig1) + G2(x; A2, mu2, sig2) − G3(x; A3, mu3, sig3)

to every training curve, then for each of the 9 shape parameters runs PhySO to find

    theta_k = f_k(Omega_m_std, sigma_8_std, w_std)

Saved outputs (in OUTPUT_DIR):
  test_cosmologies.pkl        — held-out 20% test set (same indices as PySR direct)
  cosmo_scaler.pkl            — StandardScaler fit on train only
  shape_params_train.npy      — (N_train, 9) fitted shape params
  shape_fit_r2_train.npy      — (N_train,) parametric-fit R² on train
  {pname}/best_expression.txt, expression.pkl  — PhySO expression per shape param
  shape_reconstructions_test.png
  shape_r2_distribution_test.png
  shape_predictions_test.npy  — (N_test, 160) reconstructed curves on test

Test split identical to PySR direct + NN emulator: random_state=42, TEST_SIZE=0.20.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import torch
import os
import sys
import pickle
import warnings
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

import physo
import physo.learn.monitoring as monitoring

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


# =============================================================================
# CONFIG
# =============================================================================

_REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CSV_PATH    = os.path.join(_REPO_ROOT, "data/csv/l1norm_training_data_b160.csv")
COSMO_COLS  = ["Omega_m", "sigma_8", "w"]
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# Keep budget modest — 9 expressions × 10 epochs × 2e5 evals ≈ 30-45 min total.
OP_NAMES   = ["mul", "add", "sub", "div", "inv", "n2", "exp", "log", "sqrt"]
N_FREE_PARAMS     = 3
FIXED_CONSTS      = [1.]
MAX_N_EVALUATIONS = int(2e5)
N_EPOCHS          = int(10)
PARALLEL_MODE     = True
N_CPUS            = 8

SHAPE_PARAM_NAMES = ["A1", "mu1", "sig1", "A2", "mu2", "sig2", "A3", "mu3", "sig3"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# PARAMETRIC MODEL: 2 peaks minus 1 central dip
# =============================================================================

def _gaussian(x, A, mu, sig):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)

def peaks_minus_dip(x, A1, mu1, sig1, A2, mu2, sig2, A3, mu3, sig3):
    return _gaussian(x, A1, mu1, sig1) \
         + _gaussian(x, A2, mu2, sig2) \
         - _gaussian(x, A3, mu3, sig3)


def initial_guess(x, y):
    peaks, _ = find_peaks(y, prominence=0.01 * y.max())
    if len(peaks) < 2:
        peaks = np.array([40, 120])
    else:
        peaks = peaks[np.argsort(y[peaks])[-2:]]
        peaks = np.sort(peaks)
    p1, p2 = peaks[0], peaks[-1]
    mid = int((p1 + p2) / 2)
    return [
        y[p1], float(p1), 15.0,
        y[p2], float(p2), 15.0,
        max(0.0, (y[p1] + y[p2]) / 2 - y[mid]), float(mid), 10.0,
    ]


def fit_shape(x, y):
    p0 = initial_guess(x, y)
    try:
        popt, _ = curve_fit(peaks_minus_dip, x, y, p0=p0, maxfev=5000)
        y_fit = peaks_minus_dip(x, *popt)
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return popt, y_fit, r2
    except Exception:
        return np.full(9, np.nan), np.full_like(y, np.nan), -np.inf


# =============================================================================
# DATA
# =============================================================================

def load_data():
    df = pd.read_csv(CSV_PATH)
    bin_cols = sorted([c for c in df.columns if c.startswith("bin_")],
                      key=lambda c: int(c.split("_")[1]))
    cosmo = df[COSMO_COLS].values.astype(np.float64)
    dv    = df[bin_cols].values.astype(np.float64)
    x_arr = np.arange(160, dtype=np.float64)
    print(f"Loaded {cosmo.shape[0]} cosmologies, DV length {dv.shape[1]}")
    return cosmo, dv, x_arr


def split_data(cosmo, dv):
    n = cosmo.shape[0]
    idx_train, idx_test = train_test_split(
        np.arange(n), test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"Split: train={len(idx_train)}  test={len(idx_test)}")
    with open(os.path.join(OUTPUT_DIR, "test_cosmologies.pkl"), "wb") as f:
        pickle.dump({"cosmo": cosmo[idx_test], "dv": dv[idx_test],
                     "indices": idx_test}, f)
    return cosmo[idx_train], dv[idx_train], cosmo[idx_test], dv[idx_test], \
           idx_train, idx_test


# =============================================================================
# FIT SHAPE PARAMS ON TRAINING SET
# =============================================================================

def fit_shape_all(dv_train, x_arr):
    N = dv_train.shape[0]
    params = np.zeros((N, 9))
    r2s = np.zeros(N)
    print(f"\nFitting peaks_minus_dip on {N} training curves ...")
    for i in range(N):
        popt, _, r2 = fit_shape(x_arr, dv_train[i])
        params[i] = popt
        r2s[i] = r2
    n_bad = np.isnan(params).any(axis=1).sum()
    print(f"  median shape-fit R² = {np.median(r2s):.4f}   "
          f"failed fits = {n_bad}/{N}")
    return params, r2s


# =============================================================================
# PHYSO HELPERS
# =============================================================================

def _physo_sr(X_std_np, y_np, target_name, run_subdir):
    os.makedirs(run_subdir, exist_ok=True)
    orig = os.getcwd()
    os.chdir(run_subdir)
    free_consts = [f"c{i}" for i in range(N_FREE_PARAMS)]
    logger = lambda: monitoring.RunLogger(save_path="sr.log", do_save=True)
    vis    = lambda: monitoring.RunVisualiser(
        epoch_refresh_rate=1, save_path="sr_curves.png",
        do_show=False, do_prints=True, do_save=True,
    )
    print(f"\n  [PhySO] target={target_name}  samples={X_std_np.shape[1]}")
    expr, _ = physo.SR(
        X_std_np, y_np,
        X_names           = ["Omega_m_std", "sigma_8_std", "w_std"],
        y_name            = target_name,
        fixed_consts      = FIXED_CONSTS,
        free_consts_names = free_consts,
        op_names          = OP_NAMES,
        get_run_logger    = logger,
        get_run_visualiser= vis,
        run_config        = cfg.custom_config,
        max_n_evaluations = MAX_N_EVALUATIONS,
        epochs            = N_EPOCHS,
        parallel_mode     = PARALLEL_MODE,
        n_cpus            = N_CPUS,
    )
    with open("best_expression.txt", "w") as f:
        f.write(f"Target     : {target_name}\nExpression : {expr}\n")
    with open("expression.pkl", "wb") as f:
        pickle.dump(expr, f)
    print(f"  [PhySO] Best: {expr}")
    os.chdir(orig)
    return expr


def _predict_batch(expr, cosmo_std):
    """cosmo_std (N,3) -> (N,)"""
    X_t = torch.tensor(cosmo_std.T.astype(np.float64), dtype=torch.float64)
    res = expr.execute(X_t)
    arr = res.detach().numpy() if hasattr(res, "detach") else np.asarray(res)
    return np.asarray(arr, dtype=np.float64).flatten()


# =============================================================================
# MAIN
# =============================================================================

def main():
    cosmo, dv, x_arr = load_data()
    cosmo_tr, dv_tr, cosmo_te, dv_te, idx_tr, idx_te = split_data(cosmo, dv)

    # Scaler on train only
    scaler = StandardScaler().fit(cosmo_tr)
    with open(os.path.join(OUTPUT_DIR, "cosmo_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Fit parametric model on training curves
    params_tr, shape_r2 = fit_shape_all(dv_tr, x_arr)
    np.save(os.path.join(OUTPUT_DIR, "shape_params_train.npy"), params_tr)
    np.save(os.path.join(OUTPUT_DIR, "shape_fit_r2_train.npy"),  shape_r2)

    # Mask out failed fits
    mask = ~np.isnan(params_tr).any(axis=1)
    cosmo_tr_good = cosmo_tr[mask]
    params_tr_good = params_tr[mask]
    cosmo_tr_std  = scaler.transform(cosmo_tr_good)           # (N,3)
    X_std_np      = cosmo_tr_std.T                             # (3,N)

    expressions = {}
    for j, pname in enumerate(SHAPE_PARAM_NAMES):
        y_np = params_tr_good[:, j].astype(np.float64)
        sub  = os.path.join(OUTPUT_DIR, pname)
        expr = _physo_sr(X_std_np, y_np, pname, sub)
        expressions[pname] = expr

    # Reconstruct on test set
    cosmo_te_std = scaler.transform(cosmo_te)
    preds_params = np.zeros((len(cosmo_te), 9))
    for j, pname in enumerate(SHAPE_PARAM_NAMES):
        preds_params[:, j] = _predict_batch(expressions[pname], cosmo_te_std)
    test_curves = np.stack([
        peaks_minus_dip(x_arr, *preds_params[i]) for i in range(len(cosmo_te))
    ])

    np.save(os.path.join(OUTPUT_DIR, "shape_predictions_test.npy"), test_curves)
    np.save(os.path.join(OUTPUT_DIR, "shape_params_pred_test.npy"),  preds_params)

    # R² report
    from sklearn.metrics import r2_score
    r2s = [r2_score(dv_te[i], test_curves[i]) for i in range(len(dv_te))]
    print(f"\nShape-SR test R²   median={np.median(r2s):.4f}   "
          f"min={np.min(r2s):.4f}   max={np.max(r2s):.4f}")

    # Quick plots
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for ax, i in zip(axes.ravel(), np.random.default_rng(RANDOM_SEED).choice(len(dv_te), 6, replace=False)):
        ax.plot(x_arr, dv_te[i],     "k-",  lw=1.4, label="truth")
        ax.plot(x_arr, test_curves[i], "b--", lw=1.3,
                label=f"PhySO 2G-1G  R²={r2_score(dv_te[i], test_curves[i]):.4f}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shape_reconstructions_test.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r2s, bins=15, edgecolor="k"); ax.set_xlabel("R²"); ax.set_ylabel("Count")
    ax.axvline(np.median(r2s), color="r", ls="--",
               label=f"median={np.median(r2s):.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shape_r2_distribution_test.png"), dpi=150)
    plt.close()
    print("Done. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
