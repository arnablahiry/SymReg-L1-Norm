"""
cosmo_symreg_direct_S8.py  —  Direct Symbolic Regression on Wavelet L1-Norm
============================================================================
Finds a single closed-form expression:

    L1(x; Omega_m_std, S_8_std, w_std, x_c)  =  f(...)

where:
    S_8     = sigma_8 * sqrt(Omega_m / 0.3)     (derived from raw parameters)
    *_std   denotes standardised inputs
    x_c     = x - mean_peak

Pipeline:
  1. Derive S_8 from raw (Omega_m, sigma_8)
  2. Split cosmologies 80/20 train/test  (split at cosmology level, not row level)
  3. Standardise Omega_m, S_8, w on TRAINING set only, save scaler
  4. Build flat design matrix from training cosmologies only
  5. PySR finds f(Omega_m_std, S_8_std, w_std, x_c)
  6. Evaluate final R² and plots on TEST set only

On standardisation validity:
  The found expression is in standardised variable space, e.g.:
      f(Om_std, S8_std, w_std, x_c)
  This is a valid closed-form equation — just with a change of variables.
  To recover physical-space predictions for any new cosmology:
      S_8     = sigma_8 * sqrt(Omega_m / 0.3)
      Om_std  = (Omega_m - scaler.mean_[0]) / scaler.scale_[0]
      S8_std  = (S_8     - scaler.mean_[1]) / scaler.scale_[1]
      w_std   = (w       - scaler.mean_[2]) / scaler.scale_[2]
  The saved scaler makes this fully reproducible.

Saved outputs:
  test_cosmologies.pkl     — 20% held-out cosmologies + datavectors  (S_8 stored, not sigma_8)
  cosmo_scaler.pkl         — StandardScaler fitted on [Omega_m, S_8, w] training set
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

_REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CSV_PATH    = os.path.join(_REPO_ROOT, "data/csv/l1norm_training_data_b160.csv")
RAW_COLS    = ["Omega_m", "sigma_8", "w"]   # columns present in the CSV
COSMO_COLS  = ["Omega_m", "S_8", "w"]       # features fed to PySR (sigma_8 replaced by S_8)
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs_S8_50nodes")
TEST_SIZE   = 0.20
RANDOM_SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many iterations between each "has the best equation changed?" check.
CHECKPOINT_ITERS = 10

# Small epsilon added to x_c to keep denominator away from zero when / is used.
XC_EPS = 0.0005

PYSR_KWARGS = dict(
    niterations      = CHECKPOINT_ITERS,  # overridden in run_sr loop
    binary_operators = ["+", "-", "*", "/"],
    unary_operators  = ["exp", "square"],

    populations      = 30,
    population_size  = 50,
    maxsize          = 50,
    parsimony        = 5e-5,
    batching         = True,
    batch_size       = 64,
    verbosity        = 1,
    random_state     = RANDOM_SEED,
)

# =============================================================================
# UTILITIES
# =============================================================================

def _compute_S8(Omega_m, sigma_8):
    """
    S_8 = sigma_8 * sqrt(Omega_m / 0.3)

    S_8 is the standard weak-lensing amplitude parameter. It is more directly
    constrained by large-scale structure surveys than sigma_8 alone because it
    captures the degeneracy between the clustering amplitude and the matter
    density that appears in the weak-lensing power spectrum.

    Using S_8 instead of sigma_8 as a feature:
      • Reduces the effective number of free parameters in expressions involving
        both Omega_m and sigma_8 — PySR need not rediscover the sqrt(Omega_m)
        dependence from scratch.
      • Keeps the feature space physically motivated.
      • Omega_m is still retained as a separate feature because it also enters
        through the growth factor and BAO scale, not only via S_8.

    Parameters
    ----------
    Omega_m : array-like
    sigma_8 : array-like

    Returns
    -------
    S_8 : np.ndarray, same shape as inputs
    """
    return np.asarray(sigma_8) * np.sqrt(np.asarray(Omega_m) / 0.3)


def _raw_to_features(cosmo_raw):
    """
    Convert raw parameter array (N, 3) with columns [Omega_m, sigma_8, w]
    to feature array (N, 3) with columns [Omega_m, S_8, w].

    sigma_8 is dropped from the returned array; S_8 takes its place.
    """
    cosmo_raw = np.atleast_2d(cosmo_raw)
    Om  = cosmo_raw[:, 0]
    s8  = cosmo_raw[:, 1]
    w   = cosmo_raw[:, 2]
    S8  = _compute_S8(Om, s8)
    return np.column_stack([Om, S8, w])


def _safe_xc(x_c, eps=XC_EPS):
    """
    Shift x_c away from zero by eps, preserving sign.
    Prevents poles in expressions containing 1/x_c or x_c^(-n).
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


def _make_X(feat_row, x_c, scaler):
    """
    Build (N_x, 4) prediction matrix for one cosmology.

    Parameters
    ----------
    feat_row : array-like, shape (3,)
        Feature vector [Omega_m, S_8, w] for a single cosmology
        (already converted from raw — sigma_8 NOT expected here).
    x_c      : np.ndarray, shape (N_x,)
        Centred x coordinates.
    scaler   : fitted StandardScaler on [Omega_m, S_8, w]

    Returns
    -------
    X : np.ndarray, shape (N_x, 4)
        Columns: [Omega_m_std, S_8_std, w_std, x_c_safe]
    """
    N_x        = len(x_c)
    feat_std   = scaler.transform(np.atleast_2d(feat_row))[0]   # standardise
    return np.column_stack([
        np.full(N_x, feat_std[0]),   # Omega_m_std
        np.full(N_x, feat_std[1]),   # S_8_std
        np.full(N_x, feat_std[2]),   # w_std
        _safe_xc(x_c),               # x_c with eps shift
    ])

# =============================================================================
# 1.  LOAD DATA
# =============================================================================

def load_data(path):
    df      = pd.read_csv(path)
    dv_cols = [c for c in df.columns if c not in RAW_COLS]
    try:
        dv_cols = sorted(dv_cols, key=lambda c: int(c))
    except ValueError:
        pass
    assert len(dv_cols) == 160, f"Expected 160 DV columns, found {len(dv_cols)}"

    cosmo_raw = df[RAW_COLS].values.astype(np.float64)   # [Omega_m, sigma_8, w]
    dv        = df[dv_cols].values.astype(np.float64)
    x_arr     = np.arange(160, dtype=np.float64)

    # Derive S_8 immediately; drop sigma_8 from further use
    cosmo_feat = _raw_to_features(cosmo_raw)              # [Omega_m, S_8, w]

    print(f"Loaded {cosmo_raw.shape[0]} cosmologies, DV length {dv.shape[1]}")
    print(f"\nRaw parameters:")
    for i, name in enumerate(RAW_COLS):
        print(f"  {name:8s}: [{cosmo_raw[:,i].min():.3f}, {cosmo_raw[:,i].max():.3f}]")
    print(f"\nDerived features (fed to PySR):")
    for i, name in enumerate(COSMO_COLS):
        print(f"  {name:8s}: [{cosmo_feat[:,i].min():.3f}, {cosmo_feat[:,i].max():.3f}]")

    return cosmo_feat, dv, x_arr   # return FEATURE array, not raw

# =============================================================================
# 2.  TRAIN / TEST SPLIT  (at cosmology level)
# =============================================================================

def split_data(cosmo_feat, dv):
    """
    Split at the COSMOLOGY level — not the row level.

    cosmo_feat has columns [Omega_m, S_8, w].
    sigma_8 is not stored — S_8 is the retained feature.
    """
    n_cos = cosmo_feat.shape[0]
    idx_train, idx_test = train_test_split(
        np.arange(n_cos), test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    cosmo_train, dv_train = cosmo_feat[idx_train], dv[idx_train]
    cosmo_test,  dv_test  = cosmo_feat[idx_test],  dv[idx_test]

    print(f"\nTrain/test split  ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}):")
    print(f"  Train: {len(idx_train)} cosmologies")
    print(f"  Test : {len(idx_test)} cosmologies")

    test_pkl = os.path.join(OUTPUT_DIR, "test_cosmologies.pkl")
    with open(test_pkl, "wb") as f:
        pickle.dump({
            "cosmo":    cosmo_test,    # columns: [Omega_m, S_8, w]
            "dv":       dv_test,
            "indices":  idx_test,
            "features": COSMO_COLS,   # record column order for future loading
        }, f)
    print(f"  Saved test set → {test_pkl}")
    print(f"  (test pickle stores [Omega_m, S_8, w] — sigma_8 not saved)")

    return cosmo_train, dv_train, cosmo_test, dv_test

# =============================================================================
# 3.  STANDARDISE FEATURES  (fit on train only)
# =============================================================================

def fit_scaler(cosmo_train):
    """
    Fit StandardScaler on TRAINING features [Omega_m, S_8, w] only.

    Why S_8 and not sigma_8:
      S_8 = sigma_8 * sqrt(Omega_m/0.3) absorbs the dominant cross-dependence
      between sigma_8 and Omega_m. After standardisation, PySR's constant
      fitting is not fighting a large offset AND does not need to recover the
      sqrt(Omega_m) factor from the data — it is already baked into the feature.

    Why fit on train only:
      Fitting on the full dataset leaks test-set statistics (data leakage).
    """
    scaler = StandardScaler()
    scaler.fit(cosmo_train)   # cosmo_train columns: [Omega_m, S_8, w]

    print(f"\nStandardisation (fitted on training set only):")
    for i, name in enumerate(COSMO_COLS):
        print(f"  {name:8s}: mean={scaler.mean_[i]:.4f}  std={scaler.scale_[i]:.4f}")

    scaler_pkl = os.path.join(OUTPUT_DIR, "cosmo_scaler.pkl")
    with open(scaler_pkl, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler → {scaler_pkl}")
    print(f"  (scaler expects columns in order: {COSMO_COLS})")

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
    """
    Builds flat (N_cos * N_x, 4) matrix with columns:
        [Omega_m_std, S_8_std, w_std, x_c_safe]

    cosmo_train columns: [Omega_m, S_8, w]  (sigma_8 already replaced)
    """
    N_cos    = cosmo_train.shape[0]
    N_x      = len(x_arr)
    x_c      = x_arr - mean_peak
    x_c_safe = _safe_xc(x_c)

    cosmo_std = scaler.transform(cosmo_train)              # (N_cos, 3) standardised
    cos_rep   = np.repeat(cosmo_std, N_x, axis=0)          # (N_cos*N_x, 3)
    x_rep     = np.tile(x_c_safe, N_cos).reshape(-1, 1)    # (N_cos*N_x, 1)
    X_flat    = np.hstack([cos_rep, x_rep])                 # (N_cos*N_x, 4)
    y_flat    = dv_train.reshape(-1)                        # (N_cos*N_x,)

    print(f"\n  Training design matrix: {X_flat.shape[0]:,} rows × {X_flat.shape[1]} features")
    print(f"  Features: {COSMO_COLS + ['x_c']}")
    print(f"  y range : [{y_flat.min():.4f}, {y_flat.max():.4f}]")
    print(f"  x_c range     : [{x_c.min():.1f}, {x_c.max():.1f}]")
    print(f"  x_c_safe range: [{x_c_safe.min():.1f}, {x_c_safe.max():.1f}]  (eps={XC_EPS})")
    return X_flat, y_flat, x_c, x_c_safe

# =============================================================================
# 6.  RUN SR
# =============================================================================

def run_sr(cosmo_feat, dv, x_arr):
    print("\n" + "="*60)
    print("DIRECT SR  —  f(Omega_m_std, S_8_std, w_std, x_c) → L1-norm")
    print("="*60)

    cosmo_train, dv_train, cosmo_test, dv_test = split_data(cosmo_feat, dv)
    scaler    = fit_scaler(cosmo_train)
    mean_peak = estimate_mean_peak(dv_train)

    X_flat, y_flat, x_c, x_c_safe = build_design_matrix(
        cosmo_train, dv_train, x_arr, scaler, mean_peak
    )

    total_iters   = 1000
    n_chunks      = total_iters // CHECKPOINT_ITERS
    best_expr_str = None
    chunk_count   = 0

    print(f"\n  Running PySR on {X_flat.shape[0]:,} training rows...")
    print(f"  ({n_chunks} chunks of {CHECKPOINT_ITERS} iterations each)")
    print(f"  Reconstruction saved whenever a new best equation is found.\n")

    model = PySRRegressor(**PYSR_KWARGS)

    for chunk in range(n_chunks):
        if chunk == 0:
            model.fit(X_flat, y_flat,
                      variable_names=["Omega_m_std", "S_8_std", "w_std", "x_c"])
        else:
            model.set_params(warm_start=True, niterations=CHECKPOINT_ITERS)
            model.fit(X_flat, y_flat,
                      variable_names=["Omega_m_std", "S_8_std", "w_std", "x_c"])

        current_expr_str = str(model.sympy())
        chunk_count += 1

        if current_expr_str != best_expr_str:
            best_expr_str = current_expr_str
            iters_done    = chunk_count * CHECKPOINT_ITERS
            print(f"  [iter {iters_done:>5}] New best: {best_expr_str}")
            _plot_reconstructions(
                model, cosmo_test, dv_test, x_arr, x_c, scaler,
                label=f"iter{iters_done:05d}",
            )

    print(f"\nBest expression (standardised variable space):")
    print(f"  {model.sympy()}")

    # Save expression metadata
    out_txt = os.path.join(OUTPUT_DIR, "best_expression.txt")
    with open(out_txt, "w") as f:
        f.write(f"Expression   : {model.sympy()}\n")
        f.write(f"Variables    : Omega_m_std, S_8_std, w_std, x_c\n\n")
        f.write(f"Derived parameter:\n")
        f.write(f"  S_8     = sigma_8 * sqrt(Omega_m / 0.3)\n\n")
        f.write(f"x_c      = x - {mean_peak:.6f}\n")
        f.write(f"x_c_safe = sign(x_c) * (|x_c| + {XC_EPS})  <- what the model actually sees\n\n")
        f.write(f"Standardisation (apply AFTER computing S_8, before evaluating):\n")
        for i, name in enumerate(COSMO_COLS):
            f.write(f"  {name}_std = ({name} - {scaler.mean_[i]:.6f})"
                    f" / {scaler.scale_[i]:.6f}\n")
        f.write(f"\nPython usage:\n")
        f.write(f"  scaler = pickle.load(open('cosmo_scaler.pkl', 'rb'))\n")
        f.write(f"  y = predict(Omega_m, sigma_8, w, model,"
                f" mean_peak={mean_peak:.4f}, scaler=scaler)\n")
        f.write(f"\nNote: predict() accepts raw (Omega_m, sigma_8, w).\n")
        f.write(f"      S_8 is computed internally before standardisation.\n")
    print(f"  Saved → {out_txt}")

    _plot_pareto(model)

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
    Save a reconstruction plot. cosmo has columns [Omega_m, S_8, w].
    Title shows S_8, not sigma_8.
    """
    np.random.seed(42)
    n_show   = min(n_show, cosmo.shape[0])
    idx_show = np.random.choice(cosmo.shape[0], n_show, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for ax, i in zip(axes, idx_show):
        Om, S8, w = cosmo[i]     # S_8, not sigma_8
        feat_row  = [Om, S8, w]
        y_pred    = model.predict(_make_X(feat_row, x_c, scaler))
        r2        = _r2(dv[i], y_pred)
        ax.plot(x_arr, dv[i],  "k-",  lw=1.2, label="data")
        ax.plot(x_arr, y_pred, "b--", lw=1.5, label=f"PySR  R²={r2:.3f}")
        ax.set_title(f"cosmo {i}  Om={Om:.2f}  S8={S8:.2f}  w={w:.2f}", fontsize=7)
        ax.legend(fontsize=6)
    tag = f" [{label}]" if label else ""
    plt.suptitle(f"Direct SR — TEST SET{tag}\n{model.sympy()}", fontsize=8)
    plt.tight_layout()

    out_latest = os.path.join(OUTPUT_DIR, "reconstructions_test.png")
    plt.savefig(out_latest, dpi=150)

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

    Accepts raw (Omega_m, sigma_8, w) — S_8 is computed internally.

    Parameters
    ----------
    Omega_m, sigma_8, w : float  — raw (unstandardised) cosmological parameters
    model               : fitted PySRRegressor
    mean_peak           : float  — x centring offset (from run_sr)
    scaler              : fitted StandardScaler on [Omega_m, S_8, w]
    x_arr               : np.ndarray or None  — defaults to np.arange(160)

    Returns
    -------
    y_pred : np.ndarray, shape (160,)

    Notes
    -----
    Internally:
        S_8      = sigma_8 * sqrt(Omega_m / 0.3)
        feat_row = [Omega_m, S_8, w]   (sigma_8 is NOT passed to the model)
        feat_std = scaler.transform(feat_row)
        y_pred   = model.predict([feat_std..., x_c_safe...])

    Example
    -------
    >>> model, x_c, mean_peak, scaler = run_sr(cosmo_feat, dv, x_arr)
    >>> y_new = predict(0.30, 0.80, -1.0, model, mean_peak, scaler)

    Loading from disk (no retraining needed):
    >>> scaler = pickle.load(open("cosmo_scaler.pkl", "rb"))
    >>> y_new  = predict(0.30, 0.80, -1.0, model, mean_peak, scaler)
    """
    if x_arr is None:
        x_arr = np.arange(160, dtype=np.float64)
    x_c      = x_arr - mean_peak
    S_8      = _compute_S8(Omega_m, sigma_8)          # derive S_8 from raw params
    feat_row = [Omega_m, S_8, w]                       # [Omega_m, S_8, w]
    return model.predict(_make_X(feat_row, x_c, scaler))

# =============================================================================
# 9.  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # load_data returns [Omega_m, S_8, w] — sigma_8 is converted on load
    cosmo_feat, dv, x_arr = load_data(CSV_PATH)

    model, x_c, mean_peak, scaler = run_sr(cosmo_feat, dv, x_arr)

    print(f"\n✓ Done.  Outputs → {OUTPUT_DIR}/")
    print(f"\nSaved files:")
    print(f"  test_cosmologies.pkl     — 20% held-out test set  [Omega_m, S_8, w]")
    print(f"  cosmo_scaler.pkl         — standardisation for [Omega_m, S_8, w]")
    print(f"  best_expression.txt      — symbolic expression + usage")
    print(f"  reconstructions_test.png — fits on test set")
    print(f"  r2_distribution_test.png — R² on test set")
    print(f"\nTo predict a new cosmology (pass raw sigma_8 — S_8 computed internally):")
    print(f"  y = predict(Omega_m, sigma_8, w, model, mean_peak={mean_peak:.2f}, scaler=scaler)")