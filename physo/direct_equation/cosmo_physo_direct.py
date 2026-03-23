"""
cosmo_physo_direct.py  —  Direct PhySO Symbolic Regression on Wavelet L1-Norm
==============================================================================
Finds a single closed-form expression:

    L1(x; Omega_m_std, sigma_8_std, w_std, x_c)  =  f(...)

Exact PhySO equivalent of cosmo_symreg_direct.py (PySR version).

Pipeline:
  1. Split cosmologies 80/20 train/test  (at cosmology level)
  2. Standardise Omega_m, sigma_8, w on TRAINING set only  →  save scaler
  3. Centre x around mean peak position (train only)       →  save mean_peak
  4. Subsample x points per cosmology (PhySO has no batching — keep rows < 10k)
  5. PhySO finds f(Omega_m_std, sigma_8_std, w_std, x_c)
  6. Evaluate final R² on TEST set only

Key differences from PySR version:
  - PhySO expects X as (n_vars, n_samples) — transposed vs PySR's (n_samples, n_vars)
  - No batching parameter — use X_SUBSAMPLE to keep design matrix under ~5k rows
  - Prediction uses expression.execute(X) not model.predict(X)
  - PhySO writes logs/plots to CWD — chdir to output subdir during SR call

Saved outputs:
  test_cosmologies.pkl     — 20% held-out cosmologies + datavectors
  cosmo_scaler.pkl         — StandardScaler fitted on training set
  mean_peak.pkl            — x centring offset
  best_expression.txt      — symbolic expression + full usage instructions
  expression.pkl           — PhySO expression object for reuse
  reconstructions_test.png — fits on TEST set
  r2_distribution_test.png — R² histogram on TEST set

Requirements:
    pip install physo scikit-learn numpy pandas scipy matplotlib torch
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
import argparse
from scipy.signal import find_peaks as _find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

try:
    import physo
    import physo.learn.monitoring as monitoring
except ImportError:
    raise ImportError("Install PhySO:  pip install physo")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import config as cfg
except ImportError:
    raise ImportError("config.py not found — place it in the same directory.")

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH    = "/Users/arnablahiry/repos/SymReg-L1-Norm/data/csv/l1norm_training_data_b160.csv"
COSMO_COLS  = ["Omega_m", "sigma_8", "w"]
OUTPUT_DIR  = "physo_direct_outputs"
TEST_SIZE   = 0.20
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# PhySO has no batching — subsample x points per cosmology to keep the
# design matrix under ~5,000 rows.
# With 125 train cosmologies and X_SUBSAMPLE=8: 125 × 20 = 2,500 rows.
# With X_SUBSAMPLE=4:                           125 × 40 = 5,000 rows.
# With X_SUBSAMPLE=1 (all x):                  125 × 160 = 20,000 rows (slow).
X_SUBSAMPLE = 4

# ── PhySO operator vocabulary ─────────────────────────────────────────────────
# Mirrors PySR version: ^ (pow) for super-Gaussian peaks, no abs (Laplacian cusps).
# PhySO operator names differ from PySR — mapped below.
OP_NAMES = [
    "add",    # x + y      (PySR: "+")
    "sub",    # x - y      (PySR: "-")
    "mul",    # x * y      (PySR: "*")
    "div",    # x / y      (PySR: "/")
    #"pow",    # x ^ y      (PySR: "^")  ← non-integer exponents for peak shape
    "n2",     # x^2        (PySR: "square")  ← cheap Gaussian baseline
    "exp",    # e^x        (PySR: "exp")
    #"log",    # ln(x)      (PySR: "log")
    # abs EXCLUDED — exp(-|f(x)|) creates Laplacian cusps at peaks
]

N_FREE_PARAMS     = 3         # free constants c0, c1, c2 per expression
FIXED_CONSTS      = [1.]
MAX_N_EVALUATIONS = int(1e6)  # increase to 1e7+ for publication runs
N_EPOCHS          = int(500)
PARALLEL_MODE     = True
N_CPUS            = 8

# Variable names passed to PhySO
VAR_NAMES = ["Omega_m_std", "sigma_8_std", "w_std", "x_c"]

# =============================================================================
# UTILITIES
# =============================================================================

def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def _get_peaks(y):
    peaks, _ = _find_peaks(y, distance=10)
    if len(peaks) < 2:
        peaks = [np.argmax(y[:80]), np.argmax(y[80:]) + 80]
    return sorted(peaks, key=lambda i: y[i], reverse=True)[:2]

# =============================================================================
# 1.  LOAD DATA
# =============================================================================

def load_data(path):
    df      = pd.read_csv(path)
    dv_cols = [c for c in df.columns if c not in COSMO_COLS]
    try:
        dv_cols = sorted(dv_cols, key=lambda c: int(c))
    except ValueError:
        pass
    assert len(dv_cols) == 160, f"Expected 160 DV columns, found {len(dv_cols)}"
    cosmo = df[COSMO_COLS].values.astype(np.float64)
    dv    = df[dv_cols].values.astype(np.float64)
    x_arr = np.arange(160, dtype=np.float64)
    print(f"Loaded {cosmo.shape[0]} cosmologies, DV length {dv.shape[1]}")
    for i, name in enumerate(COSMO_COLS):
        print(f"  {name:8s}: [{cosmo[:,i].min():.3f}, {cosmo[:,i].max():.3f}]")
    return cosmo, dv, x_arr

# =============================================================================
# 2.  TRAIN / TEST SPLIT  (at cosmology level)
# =============================================================================

def split_data(cosmo, dv):
    """
    Split at the COSMOLOGY level — not the row level.

    Each cosmology is one independent simulation. Splitting at row level
    would leak information: the same (Omega_m, sigma_8, w) appearing at
    different x values in both train and test means the model only needs
    to memorise values, not generalise. Splitting by cosmology means the
    test set contains entirely unseen parameter combinations.
    """
    idx_train, idx_test = train_test_split(
        np.arange(cosmo.shape[0]),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    cosmo_train, dv_train = cosmo[idx_train], dv[idx_train]
    cosmo_test,  dv_test  = cosmo[idx_test],  dv[idx_test]

    print(f"\nTrain/test split  ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}):")
    print(f"  Train: {len(idx_train)} cosmologies")
    print(f"  Test : {len(idx_test)} cosmologies")

    test_pkl = os.path.join(OUTPUT_DIR, "test_cosmologies.pkl")
    with open(test_pkl, "wb") as f:
        pickle.dump({"cosmo": cosmo_test, "dv": dv_test, "indices": idx_test}, f)
    print(f"  Saved test set → {test_pkl}")

    return cosmo_train, dv_train, cosmo_test, dv_test

# =============================================================================
# 3.  STANDARDISE  (train only)
# =============================================================================

def fit_scaler(cosmo_train):
    """
    Fit StandardScaler on training cosmologies only.

    w is centred at -1.0 while Omega_m ~0.3 and sigma_8 ~0.75.
    Without standardisation, w appears less informative to PhySO's RNN
    because its offset from zero is large. After standardisation all three
    variables have mean=0, std=1 — the RNN treats them with equal importance.
    """
    scaler = StandardScaler()
    scaler.fit(cosmo_train)

    print(f"\nStandardisation (training set only):")
    for i, name in enumerate(COSMO_COLS):
        print(f"  {name:8s}: mean={scaler.mean_[i]:.4f}  std={scaler.scale_[i]:.4f}")

    scaler_pkl = os.path.join(OUTPUT_DIR, "cosmo_scaler.pkl")
    with open(scaler_pkl, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler → {scaler_pkl}")

    return scaler

# =============================================================================
# 4.  MEAN PEAK  (train only)
# =============================================================================

def estimate_mean_peak(dv_train):
    """
    Centre x around the mean peak position across training cosmologies.
    This keeps x_c offsets small (~±15) rather than large (~80),
    helping PhySO's L-BFGS constant fitting converge faster.
    """
    peak_positions = []
    for i in range(dv_train.shape[0]):
        p = _get_peaks(dv_train[i])
        peak_positions.extend([float(p[0]), float(p[1])])
    mean_peak = np.mean(peak_positions)
    print(f"\n  Mean peak position (train): {mean_peak:.1f}"
          f"  →  x_c = x - {mean_peak:.1f}")

    with open(os.path.join(OUTPUT_DIR, "mean_peak.pkl"), "wb") as f:
        pickle.dump(mean_peak, f)

    return mean_peak

# =============================================================================
# 5.  BUILD TRAINING DESIGN MATRIX
# =============================================================================

def build_design_matrix(cosmo_train, dv_train, x_arr, scaler, mean_peak,
                         subsample=X_SUBSAMPLE):
    """
    Build flat design matrix from training data only.

    PhySO has no batching parameter (unlike PySR). To keep the number of
    rows manageable, we subsample every `subsample`-th x point per cosmology.

    subsample=4:  125 cosmo × 40 x-points = 5,000 rows   ← default
    subsample=1:  125 cosmo × 160 x-points = 20,000 rows  (slow but thorough)

    The subsampled points still span the full x range including both peaks
    and the dip — PhySO sees the complete curve shape.

    PhySO format: X is (n_vars, n_samples) — transposed vs PySR.
    Returns:
      X_physo  : (4, N_rows)  — PhySO format
      y_flat   : (N_rows,)
      x_c_full : (160,)       — full x_c grid for evaluation/prediction
    """
    N_cos   = cosmo_train.shape[0]
    x_c_full = x_arr - mean_peak
    x_sub    = x_c_full[::subsample]
    N_x_sub  = len(x_sub)

    cosmo_std = scaler.transform(cosmo_train)             # (N_cos, 3)
    cos_rep   = np.repeat(cosmo_std, N_x_sub, axis=0)    # (N_cos*N_x_sub, 3)
    x_rep     = np.tile(x_sub, N_cos)                    # (N_cos*N_x_sub,)
    y_flat    = dv_train[:, ::subsample].reshape(-1)      # (N_cos*N_x_sub,)

    # Stack into (N_rows, 4) then transpose to (4, N_rows) for PhySO
    X_flat   = np.column_stack([cos_rep, x_rep])          # (N_rows, 4)
    X_physo  = X_flat.T.astype(np.float64)                # (4, N_rows) ← PhySO

    print(f"\n  Training design matrix: {X_physo.shape[1]:,} rows × {X_physo.shape[0]} vars")
    print(f"  (subsampled every {subsample} x-points: {N_x_sub} per cosmology)")
    print(f"  y range: [{y_flat.min():.4f}, {y_flat.max():.4f}]")

    return X_physo, y_flat.astype(np.float64), x_c_full

# =============================================================================
# 6.  PHYSO SR CALL
# =============================================================================

def _run_physo(X_physo, y_flat, run_subdir):
    """
    Single PhySO SR call.

    X_physo : (4, N_rows) — [Omega_m_std, sigma_8_std, w_std, x_c]
    y_flat  : (N_rows,)
    """
    os.makedirs(run_subdir, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(run_subdir)   # PhySO writes all output files to CWD

    free_consts = [f"c{i}" for i in range(N_FREE_PARAMS)]

    run_logger = lambda: monitoring.RunLogger(
        save_path="sr.log", do_save=True
    )
    run_vis = lambda: monitoring.RunVisualiser(
        epoch_refresh_rate=1,
        save_path="sr_curves.png",
        do_show=False,
        do_prints=True,
        do_save=True,
    )

    CONFIG = cfg.custom_config

    print(f"\n  [PhySO] {X_physo.shape[1]:,} rows  {X_physo.shape[0]} vars  "
          f"free consts: {free_consts}")

    expression, logs = physo.SR(
        X_physo,
        y_flat,
        X_names           = VAR_NAMES,
        y_name            = "L1_{norm}",
        fixed_consts      = FIXED_CONSTS,
        free_consts_names = free_consts,
        op_names          = OP_NAMES,
        get_run_logger    = run_logger,
        get_run_visualiser= run_vis,
        run_config        = CONFIG,
        max_n_evaluations = MAX_N_EVALUATIONS,
        epochs            = N_EPOCHS,
        parallel_mode     = PARALLEL_MODE,
        n_cpus            = N_CPUS,
    )

    with open("expression.pkl", "wb") as f:
        pickle.dump(expression, f)

    print(f"\n  [PhySO] Best expression: {expression}")
    os.chdir(orig_dir)
    return expression

# =============================================================================
# 7.  PREDICTION HELPERS
# =============================================================================

def _make_X_physo(cosmo_row, x_c_full, scaler):
    """
    Build (4, 160) matrix for one cosmology — PhySO format.
    cosmo_row : raw [Omega_m, sigma_8, w]
    """
    N_x       = len(x_c_full)
    cosmo_std = scaler.transform([cosmo_row])[0]
    return np.array([
        np.full(N_x, cosmo_std[0]),   # Omega_m_std
        np.full(N_x, cosmo_std[1]),   # sigma_8_std
        np.full(N_x, cosmo_std[2]),   # w_std
        x_c_full,                      # x_c
    ], dtype=np.float64)               # (4, 160)


def _evaluate(expression, X_physo):
    """Call expression.execute() and return flat numpy array."""
    result = expression.execute(X_physo)
    return np.asarray(result, dtype=np.float64).flatten()

# =============================================================================
# 8.  MAIN RUN
# =============================================================================

def run_sr(cosmo, dv, x_arr):
    print("\n" + "="*60)
    print("DIRECT PhySO SR  —  f(Omega_m, sigma_8, w, x) → L1-norm")
    print("="*60)

    # ── Split ─────────────────────────────────────────────────────────────────
    cosmo_train, dv_train, cosmo_test, dv_test = split_data(cosmo, dv)

    # ── Standardise + centre x (train only) ───────────────────────────────────
    scaler    = fit_scaler(cosmo_train)
    mean_peak = estimate_mean_peak(dv_train)

    # ── Build training design matrix ───────────────────────────────────────────
    X_physo, y_flat, x_c_full = build_design_matrix(
        cosmo_train, dv_train, x_arr, scaler, mean_peak
    )

    # ── Run PhySO ──────────────────────────────────────────────────────────────
    print(f"\n  Running PhySO SR...")
    subdir     = os.path.join(OUTPUT_DIR, "sr_run")
    expression = _run_physo(X_physo, y_flat, subdir)

    # ── Save expression metadata ────────────────────────────────────────────────
    out_txt = os.path.join(OUTPUT_DIR, "best_expression.txt")
    with open(out_txt, "w") as f:
        f.write(f"Expression   : {expression}\n")
        f.write(f"Variables    : Omega_m_std, sigma_8_std, w_std, x_c\n\n")
        f.write(f"x_c = x - {mean_peak:.6f}\n\n")
        f.write(f"Standardisation (apply to raw cosmo params before evaluating):\n")
        for i, name in enumerate(COSMO_COLS):
            f.write(f"  {name}_std = ({name} - {scaler.mean_[i]:.6f})"
                    f" / {scaler.scale_[i]:.6f}\n")
        f.write(f"\nPython usage:\n")
        f.write(f"  scaler    = pickle.load(open('cosmo_scaler.pkl', 'rb'))\n")
        f.write(f"  mean_peak = pickle.load(open('mean_peak.pkl', 'rb'))\n")
        f.write(f"  y = predict(Omega_m, sigma_8, w, expression, mean_peak, scaler)\n")
    print(f"\n  Saved expression → {out_txt}")

    # ── Save expression object ─────────────────────────────────────────────────
    expr_pkl = os.path.join(OUTPUT_DIR, "expression.pkl")
    with open(expr_pkl, "wb") as f:
        pickle.dump(expression, f)
    print(f"  Saved expression object → {expr_pkl}")

    # ── Final evaluation on TEST SET ONLY ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION ON TEST SET ({cosmo_test.shape[0]} unseen cosmologies)")
    print(f"{'='*60}")
    _plot_reconstructions(expression, cosmo_test, dv_test, x_arr,
                          x_c_full, scaler)
    _plot_r2_distribution(expression, cosmo_test, dv_test, x_c_full, scaler)

    return expression, x_c_full, mean_peak, scaler

# =============================================================================
# 9.  PLOTS
# =============================================================================

def _plot_reconstructions(expression, cosmo, dv, x_arr, x_c_full, scaler,
                           n_show=6):
    n_show   = min(n_show, cosmo.shape[0])
    idx_show = np.random.choice(cosmo.shape[0], n_show, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    for ax, i in zip(axes, idx_show):
        Om, s8, w = cosmo[i]
        X_pred = _make_X_physo(cosmo[i], x_c_full, scaler)
        y_pred = _evaluate(expression, X_pred)
        r2     = _r2(dv[i], y_pred)
        ax.plot(x_arr, dv[i],  "k-",  lw=1.2, label="data")
        ax.plot(x_arr, y_pred, "b--", lw=1.5, label=f"PhySO  R²={r2:.3f}")
        ax.set_title(f"cosmo {i}  Ωm={Om:.2f}  σ8={s8:.2f}  w={w:.2f}",
                     fontsize=7)
        ax.legend(fontsize=6)

    plt.suptitle(f"Direct PhySO SR — TEST SET\n{expression}", fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "reconstructions_test.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved reconstructions (test) → {out}")


def _plot_r2_distribution(expression, cosmo, dv, x_c_full, scaler):
    r2_all = np.array([
        _r2(dv[i], _evaluate(expression,
                              _make_X_physo(cosmo[i], x_c_full, scaler)))
        for i in range(cosmo.shape[0])
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(r2_all, bins=20, edgecolor="k", color="steelblue")
    axes[0].set_xlabel("R²"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"R² — TEST SET  (median={np.median(r2_all):.4f})")
    axes[1].plot(np.sort(r2_all), "k-", lw=1.2)
    axes[1].axhline(0.99, color="r", ls="--", lw=1, label="R²=0.99")
    axes[1].set_xlabel("Cosmology (sorted)"); axes[1].set_ylabel("R²")
    axes[1].set_title("Sorted R² — TEST SET")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "r2_distribution_test.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved R² distribution (test) → {out}")

    print(f"\nFinal R² on TEST set ({len(r2_all)} unseen cosmologies):")
    print(f"  Median : {np.median(r2_all):.4f}")
    print(f"  Min    : {np.min(r2_all):.4f}")
    print(f"  Max    : {np.max(r2_all):.4f}")
    print(f"  >0.99  : {(r2_all > 0.99).sum()} / {len(r2_all)}")

# =============================================================================
# 10.  PREDICT FOR NEW (UNSEEN) COSMOLOGY
# =============================================================================

def predict(Omega_m, sigma_8, w, expression, mean_peak, scaler, x_arr=None):
    """
    Predict the full L1-norm datavector for any new cosmology.

    Parameters
    ----------
    Omega_m, sigma_8, w : float  — raw (unstandardised) cosmological parameters
    expression          : PhySO expression object  (from run_sr or expression.pkl)
    mean_peak           : float  — x centring offset (from run_sr or mean_peak.pkl)
    scaler              : fitted StandardScaler     (from run_sr or cosmo_scaler.pkl)
    x_arr               : np.ndarray or None  — defaults to np.arange(160)

    Returns
    -------
    y_pred : np.ndarray, shape (160,)

    Example
    -------
    >>> expression, x_c, mean_peak, scaler = run_sr(cosmo, dv, x_arr)
    >>> y_new = predict(0.30, 0.80, -1.0, expression, mean_peak, scaler)

    Loading from disk (no retraining needed):
    >>> expression = pickle.load(open("physo_direct_outputs/expression.pkl", "rb"))
    >>> scaler     = pickle.load(open("physo_direct_outputs/cosmo_scaler.pkl", "rb"))
    >>> mean_peak  = pickle.load(open("physo_direct_outputs/mean_peak.pkl", "rb"))
    >>> y_new = predict(0.30, 0.80, -1.0, expression, mean_peak, scaler)
    """
    if x_arr is None:
        x_arr = np.arange(160, dtype=np.float64)
    x_c_full = x_arr - mean_peak
    X_physo  = _make_X_physo([Omega_m, sigma_8, w], x_c_full, scaler)
    return _evaluate(expression, X_physo)

# =============================================================================
# 11.  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhySO direct SR: cosmology + x → L1-norm value",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subsample",  default=X_SUBSAMPLE, type=int,
                        help="Use every Nth x point per cosmology (reduces rows).")
    parser.add_argument("--n_free",    default=N_FREE_PARAMS, type=int,
                        help="Free constants per expression.")
    parser.add_argument("--max_evals", default=MAX_N_EVALUATIONS, type=int,
                        help="Max evaluations before stopping.")
    parser.add_argument("--ncpus",     default=N_CPUS, type=int,
                        help="CPUs for parallel evaluation.")
    parser.add_argument("--seed",      default=RANDOM_SEED, type=int,
                        help="Random seed.")
    args = parser.parse_args()

    X_SUBSAMPLE       = args.subsample
    N_FREE_PARAMS     = args.n_free
    MAX_N_EVALUATIONS = args.max_evals
    N_CPUS            = args.ncpus
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cosmo, dv, x_arr = load_data(CSV_PATH)

    expression, x_c_full, mean_peak, scaler = run_sr(cosmo, dv, x_arr)

    print(f"\n✓ Done.  Outputs → {OUTPUT_DIR}/")
    print(f"\nSaved files:")
    print(f"  test_cosmologies.pkl     — 20% held-out test set")
    print(f"  cosmo_scaler.pkl         — standardisation parameters")
    print(f"  mean_peak.pkl            — x centring offset")
    print(f"  best_expression.txt      — symbolic expression + usage")
    print(f"  expression.pkl           — PhySO expression object")
    print(f"  sr_run/sr.log            — PhySO training log")
    print(f"  sr_run/sr_curves.png     — reward curves")
    print(f"\nTo predict a new cosmology:")
    print(f"  y = predict(Omega_m, sigma_8, w, expression, mean_peak, scaler)")