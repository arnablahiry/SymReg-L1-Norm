"""
cosmo_symreg_direct.py  —  Direct Symbolic Regression on Wavelet L1-Norm
=========================================================================
Finds a single closed-form expression:

    L1(x; Omega_m_std, sigma_8_std, w_std, x_c)  =  f(...)

where *_std denotes standardised inputs and x_c = x - mean_peak.

Pipeline:
  1. Split cosmologies 80/20 train/test  (split at cosmology level, not row level)
  2. Standardise Omega_m, sigma_8, w on TRAINING set only, save scaler
  3. Build flat design matrix from training cosmologies only
  4. PySR finds f(Omega_m_std, sigma_8_std, w_std, x_c)
  5. Evaluate final R² and plots on TEST set only

On standardisation validity:
  The found expression is in standardised variable space, e.g.:
      f(Om_std, s8_std, w_std, x_c)
  This is a valid closed-form equation — just with a change of variables.
  To recover physical-space predictions for any new cosmology:
      Om_std = (Omega_m - scaler.mean_[0]) / scaler.scale_[0]
  The saved scaler makes this fully reproducible.

Saved outputs:
  test_cosmologies.pkl     — 20% held-out cosmologies + datavectors
  cosmo_scaler.pkl         — StandardScaler fitted on training set
  best_expression.txt      — symbolic expression + metadata
  reconstructions_test.png — fits on TEST set
  r2_distribution_test.png — R² histogram on TEST set

Requirements:
    pip install pysr scikit-learn numpy pandas scipy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as _find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
import os

try:
    from pysr import PySRRegressor
except ImportError:
    raise ImportError("Install PySR:  pip install pysr")

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH    = "/Users/arnablahiry/repos/SymReg-L1-Norm/data/csv/l1norm_training_data_b160.csv"
COSMO_COLS  = ["Omega_m", "sigma_8", "w"]
OUTPUT_DIR  = "symreg_direct_outputs_50nodes_new"
TEST_SIZE   = 0.20
RANDOM_SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many iterations between each "has the best equation changed?" check.
# Smaller = more frequent saves but slightly more overhead.
CHECKPOINT_ITERS = 10

# Small epsilon added to x_c to keep denominator away from zero when / is used.
# x_c = 0 sits exactly at the dip — without eps, expressions like c/x_c blow up.
# eps=0.5 shifts the zero-crossing by half a bin — imperceptible on the curve
# but eliminates the pole. Sign is preserved so the tilt asymmetry still works.
XC_EPS = 0.0005  # x_c min is already 0.5 (mean_peak lands between bins)
                  # eps is just a safety buffer — 0.0005 << 0.5 so distortion is negligible

PYSR_KWARGS = dict(
    niterations      = CHECKPOINT_ITERS,  # overridden in run_sr loop
    binary_operators = ["+", "-", "*", "/"],
    unary_operators  = ["exp", "square"],

    populations      = 30,
    population_size  = 50,
    maxsize          = 50,
    parsimony        = 5e-5,       
    batching         = True,       # needed for >10k rows
    batch_size       = 64,
    verbosity        = 1,
    random_state     = RANDOM_SEED,
)

# =============================================================================
# UTILITIES
# =============================================================================

def _safe_xc(x_c, eps=XC_EPS):
    """
    Shift x_c away from zero by eps, preserving sign.
    Prevents poles in expressions containing 1/x_c or x_c^(-n).
    x_c=0 is at the dip — even eps=0.5 (half a bin) is invisible on the curve.
    """
    return np.where(x_c >= 0, x_c + eps, x_c - eps)


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def _get_peaks(y):
    peaks, _ = _find_peaks(y, distance=10)
    if len(peaks) < 2:
        peaks = [np.argmax(y[:80]), np.argmax(y[80:]) + 80]
    return sorted(peaks, key=lambda i: y[i], reverse=True)[:2]

def _make_X(cosmo_row, x_c, scaler):
    """Build (N_x, 4) prediction matrix for one cosmology.
    x_c is shifted by XC_EPS to match training data transformation.
    """
    N_x       = len(x_c)
    cosmo_std = scaler.transform([cosmo_row])[0]
    return np.column_stack([
        np.full(N_x, cosmo_std[0]),
        np.full(N_x, cosmo_std[1]),
        np.full(N_x, cosmo_std[2]),
        _safe_xc(x_c),             # same eps shift as training
    ])

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
    to memorise values, not generalise across parameter space.

    Splitting by cosmology means the test set contains entirely unseen
    (Omega_m, sigma_8, w) combinations — a proper out-of-distribution test.
    """
    n_cos = cosmo.shape[0]
    idx_train, idx_test = train_test_split(
        np.arange(n_cos), test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    cosmo_train, dv_train = cosmo[idx_train], dv[idx_train]
    cosmo_test,  dv_test  = cosmo[idx_test],  dv[idx_test]

    print(f"\nTrain/test split  ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}):")
    print(f"  Train: {len(idx_train)} cosmologies")
    print(f"  Test : {len(idx_test)} cosmologies")

    # Save test set for later independent evaluation
    test_pkl = os.path.join(OUTPUT_DIR, "test_cosmologies.pkl")
    with open(test_pkl, "wb") as f:
        pickle.dump({
            "cosmo":   cosmo_test,
            "dv":      dv_test,
            "indices": idx_test,
        }, f)
    print(f"  Saved test set → {test_pkl}")

    return cosmo_train, dv_train, cosmo_test, dv_test

# =============================================================================
# 3.  STANDARDISE COSMOLOGICAL INPUTS  (fit on train only)
# =============================================================================

def fit_scaler(cosmo_train):
    """
    Fit StandardScaler on TRAINING cosmologies only — never on test.

    Why standardise:
      w is centred at -1.0 while Omega_m and sigma_8 are near 0.3 and 0.75.
      PySR's constant fitting struggles with large offsets from zero, making
      w appear less informative than it is. After standardisation all three
      variables have mean=0, std=1 — PySR treats them equally.

    Why fit on train only:
      Fitting on the full dataset leaks test-set statistics into the
      standardisation (data leakage). The scaler must only see training data.
    """
    scaler = StandardScaler()
    scaler.fit(cosmo_train)

    print(f"\nStandardisation (fitted on training set only):")
    for i, name in enumerate(COSMO_COLS):
        print(f"  {name:8s}: mean={scaler.mean_[i]:.4f}  std={scaler.scale_[i]:.4f}")

    scaler_pkl = os.path.join(OUTPUT_DIR, "cosmo_scaler.pkl")
    with open(scaler_pkl, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler → {scaler_pkl}")

    return scaler

# =============================================================================
# 4.  ESTIMATE MEAN PEAK  (training set only)
# =============================================================================

def estimate_mean_peak(dv_train):
    peak_positions = []
    for i in range(dv_train.shape[0]):
        p = _get_peaks(dv_train[i])
        peak_positions.extend([float(p[0]), float(p[1])])
    mean_peak = np.mean(peak_positions)
    print(f"\n  Mean peak position (train): {mean_peak:.1f}"
          f"  →  x_c = x - {mean_peak:.1f}")
    return mean_peak

# =============================================================================
# 5.  BUILD TRAINING DESIGN MATRIX
# =============================================================================

def build_design_matrix(cosmo_train, dv_train, x_arr, scaler, mean_peak):
    N_cos = cosmo_train.shape[0]
    N_x   = len(x_arr)
    x_c      = x_arr - mean_peak
    x_c_safe = _safe_xc(x_c)          # shift away from zero — prevents poles

    cosmo_std = scaler.transform(cosmo_train)             # (N_cos, 3)
    cos_rep   = np.repeat(cosmo_std, N_x, axis=0)         # (N_cos*N_x, 3)
    x_rep     = np.tile(x_c_safe, N_cos).reshape(-1, 1)   # (N_cos*N_x, 1)
    X_flat    = np.hstack([cos_rep, x_rep])                # (N_cos*N_x, 4)
    y_flat    = dv_train.reshape(-1)                       # (N_cos*N_x,)

    print(f"\n  Training design matrix: {X_flat.shape[0]:,} rows × {X_flat.shape[1]} features")
    print(f"  y range: [{y_flat.min():.4f}, {y_flat.max():.4f}]")
    print(f"  x_c range: [{x_c.min():.1f}, {x_c.max():.1f}]  "
          f"x_c_safe range: [{x_c_safe.min():.1f}, {x_c_safe.max():.1f}]  "
          f"(eps={XC_EPS})")
    return X_flat, y_flat, x_c, x_c_safe

# =============================================================================
# 6.  RUN SR
# =============================================================================

def run_sr(cosmo, dv, x_arr):
    print("\n" + "="*60)
    print("DIRECT SR  —  f(Omega_m, sigma_8, w, x) → L1-norm value")
    print("="*60)

    # Split at cosmology level
    cosmo_train, dv_train, cosmo_test, dv_test = split_data(cosmo, dv)

    # Standardise on train only
    scaler    = fit_scaler(cosmo_train)
    mean_peak = estimate_mean_peak(dv_train)

    # Build training design matrix
    X_flat, y_flat, x_c, x_c_safe = build_design_matrix(
        cosmo_train, dv_train, x_arr, scaler, mean_peak
    )

    # ── PySR with warm_start loop ────────────────────────────────────────────
    # Runs CHECKPOINT_ITERS iterations at a time. After each chunk, checks
    # whether the best equation has changed. If yes, saves a reconstruction
    # plot immediately so you can track progress without waiting for the full run.
    total_iters  = 1000               # total iterations to run
    n_chunks     = total_iters // CHECKPOINT_ITERS
    best_expr_str = None
    chunk_count   = 0

    print(f"\n  Running PySR on {X_flat.shape[0]:,} training rows...")
    print(f"  ({n_chunks} chunks of {CHECKPOINT_ITERS} iterations each)")
    print(f"  Reconstruction saved whenever a new best equation is found.\n")

    model = PySRRegressor(**PYSR_KWARGS)

    for chunk in range(n_chunks):
        if chunk == 0:
            model.fit(X_flat, y_flat,
                      variable_names=["Omega_m_std", "sigma_8_std", "w_std", "x_c"])
        else:
            # warm_start=True continues from where the previous fit left off
            model.set_params(warm_start=True, niterations=CHECKPOINT_ITERS)
            model.fit(X_flat, y_flat,
                      variable_names=["Omega_m_std", "sigma_8_std", "w_std", "x_c"])

        current_expr_str = str(model.sympy())
        chunk_count += 1

        if current_expr_str != best_expr_str:
            best_expr_str = current_expr_str
            iters_done    = chunk_count * CHECKPOINT_ITERS
            print(f"  [iter {iters_done:>5}] New best: {best_expr_str}")

            # Save reconstruction plot with iteration stamp in filename
            _plot_reconstructions(
                model, cosmo_test, dv_test, x_arr, x_c, scaler,
                label=f"iter{iters_done:05d}",
            )

    print(f"\nBest expression (standardised variable space):")
    print(f"  {model.sympy()}")

    # Save full expression metadata
    out_txt = os.path.join(OUTPUT_DIR, "best_expression.txt")
    with open(out_txt, "w") as f:
        f.write(f"Expression   : {model.sympy()}\n")
        f.write(f"Variables    : Omega_m_std, sigma_8_std, w_std, x_c\n\n")
        f.write(f"x_c      = x - {mean_peak:.6f}\n")
        f.write(f"x_c_safe = sign(x_c) * (|x_c| + {XC_EPS})  <- what the model actually sees\n\n")
        f.write(f"Standardisation (apply before evaluating):\n")
        for i, name in enumerate(COSMO_COLS):
            f.write(f"  {name}_std = ({name} - {scaler.mean_[i]:.6f})"
                    f" / {scaler.scale_[i]:.6f}\n")
        f.write(f"\nPython usage:\n")
        f.write(f"  scaler    = pickle.load(open('cosmo_scaler.pkl', 'rb'))\n")
        f.write(f"  y = predict(Omega_m, sigma_8, w, model,"
                f" mean_peak={mean_peak:.4f}, scaler=scaler)\n")
    print(f"  Saved → {out_txt}")

    # Pareto front
    _plot_pareto(model)

    # ── Final evaluation on TEST SET ONLY ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION ON TEST SET ({cosmo_test.shape[0]} unseen cosmologies)")
    print(f"{'='*60}")
    _plot_reconstructions(model, cosmo_test, dv_test, x_arr, x_c, scaler,
                          label='final')
    plot_all_residuals(model, cosmo_test, dv_test, x_arr, x_c, scaler)

    return model, x_c, mean_peak, scaler

# =============================================================================
# 7.  PLOTS
# =============================================================================

def _plot_pareto(model):
    try:
        eqs = model.equations_
        if eqs is None or "complexity" not in eqs.columns:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(eqs["complexity"], eqs["loss"], s=30, zorder=3)
        top = eqs.nsmallest(6, "loss")
        for _, row in top.iterrows():
            ax.annotate(str(row["sympy_format"])[:55],
                        (row["complexity"], row["loss"]),
                        fontsize=6, xytext=(5, 5),
                        textcoords="offset points", zorder=4)
        ax.set_xlabel("Complexity (# nodes)")
        ax.set_ylabel("Loss (MSE)")
        ax.set_yscale("log")
        ax.set_title("PySR Pareto Front")
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, "pareto_front.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"Saved Pareto front → {out}")
    except Exception:
        pass

def _plot_reconstructions(model, cosmo, dv, x_arr, x_c, scaler,
                           n_show=6, label=None):
    """
    Save a reconstruction plot.
    If label is given (e.g. "iter00050"), saves as reconstructions_iter00050.png
    AND overwrites reconstructions_test.png (the "current best" file).
    This means reconstructions_test.png always shows the latest best equation,
    while the stamped files give a full history.
    """
    np.random.seed(42)   # fix random sample so plots are comparable across iters
    n_show   = min(n_show, cosmo.shape[0])
    idx_show = np.random.choice(cosmo.shape[0], n_show, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for ax, i in zip(axes, idx_show):
        Om, s8, w = cosmo[i]
        y_pred = model.predict(_make_X(cosmo[i], x_c, scaler))
        r2 = _r2(dv[i], y_pred)
        ax.plot(x_arr, dv[i],  "k-",  lw=1.2, label="data")
        ax.plot(x_arr, y_pred, "b--", lw=1.5, label=f"PySR  R²={r2:.3f}")
        ax.set_title(f"cosmo {i}  Om={Om:.2f}  s8={s8:.2f}  w={w:.2f}", fontsize=7)
        ax.legend(fontsize=6)
    tag = f" [{label}]" if label else ""
    plt.suptitle(f"Direct SR — TEST SET{tag}\n{model.sympy()}", fontsize=8)
    plt.tight_layout()

    # Always overwrite the "current best" file
    out_latest = os.path.join(OUTPUT_DIR, "reconstructions_test.png")
    plt.savefig(out_latest, dpi=150)

    # Also save a stamped copy for history
    if label:
        out_stamp = os.path.join(OUTPUT_DIR, f"reconstructions_{label}.png")
        plt.savefig(out_stamp, dpi=150)
        print(f"  Saved → {out_stamp}  (and updated reconstructions_test.png)")
    else:
        print(f"  Saved → {out_latest}")

    plt.close()

def plot_all_residuals(model, cosmo, dv, x_arr, x_c, scaler):
    r2_all = np.array([
        _r2(dv[i], model.predict(_make_X(cosmo[i], x_c, scaler)))
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
# 8.  PREDICT FOR NEW (UNSEEN) COSMOLOGY
# =============================================================================

def predict(Omega_m, sigma_8, w, model, mean_peak, scaler, x_arr=None):
    """
    Predict the full L1-norm datavector for any new cosmology.

    Parameters
    ----------
    Omega_m, sigma_8, w : float  — raw (unstandardised) cosmological parameters
    model               : fitted PySRRegressor
    mean_peak           : float  — x centring offset (from run_sr)
    scaler              : fitted StandardScaler (from run_sr or cosmo_scaler.pkl)
    x_arr               : np.ndarray or None  — defaults to np.arange(160)

    Returns
    -------
    y_pred : np.ndarray, shape (160,)

    Example
    -------
    >>> model, x_c, mean_peak, scaler = run_sr(cosmo, dv, x_arr)
    >>> y_new = predict(0.30, 0.80, -1.0, model, mean_peak, scaler)

    Loading from disk (no retraining needed):
    >>> scaler = pickle.load(open("symreg_direct_outputs/cosmo_scaler.pkl", "rb"))
    >>> y_new  = predict(0.30, 0.80, -1.0, model, mean_peak, scaler)
    """
    if x_arr is None:
        x_arr = np.arange(160, dtype=np.float64)
    x_c = x_arr - mean_peak
    # _make_X applies _safe_xc internally — same eps as training
    return model.predict(_make_X([Omega_m, sigma_8, w], x_c, scaler))

# =============================================================================
# 9.  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    cosmo, dv, x_arr = load_data(CSV_PATH)

    model, x_c, mean_peak, scaler = run_sr(cosmo, dv, x_arr)

    print(f"\n✓ Done.  Outputs → {OUTPUT_DIR}/")
    print(f"\nSaved files:")
    print(f"  test_cosmologies.pkl     — 20% held-out test set")
    print(f"  cosmo_scaler.pkl         — standardisation parameters")
    print(f"  best_expression.txt      — symbolic expression + usage")
    print(f"  reconstructions_test.png — fits on test set")
    print(f"  r2_distribution_test.png — R² on test set")
    print(f"\nTo predict a new cosmology:")
    print(f"  y = predict(Omega_m, sigma_8, w, model, mean_peak={mean_peak:.2f}, scaler=scaler)")