"""
Symbolic Regression: Cosmological Parameters → Wavelet L1 Norm Datavector
==========================================================================
Two complementary approaches:

  APPROACH A — Shape Decomposition
    1. Fit each 160-point datavector with a double-Gaussian model
    2. Extract 6 shape parameters per cosmology (2 peaks × amplitude, centre, width)
    3. Run PySR independently on each shape param vs (Omega_m, sigma_8, w)
    → Interpretable closed-form expressions for each shape feature

  APPROACH B — Direct Symbolic Regression
    1. Stack all cosmologies: rows = (Omega_m, sigma_8, w, x), target = value
    2. PySR finds f(Omega_m, sigma_8, w, x) directly
    → Single expression covering the full curve

Requirements:
    pip install pysr numpy pandas scipy matplotlib
    (PySR also needs Julia; on first run it auto-installs — takes ~5 min)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
import os

# ── PySR ──────────────────────────────────────────────────────────────────────
try:
    from pysr import PySRRegressor
except ImportError:
    raise ImportError("Install PySR first:  pip install pysr")

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================

CSV_PATH = "/Users/arnablahiry/repos/SymReg-L1-Norm/data/csv/l1norm_training_data_b160.csv"          # path to your 163-column CSV

# Column names — change if yours differ
COSMO_COLS  = ["Omega_m", "sigma_8", "w"]
# If your datavector columns are named dv_0 … dv_159, set PREFIX = "dv_"
# If they are just integers 0 … 159, set PREFIX = None (auto-detected)
DV_PREFIX   = None                   # e.g. "dv_" or None

OUTPUT_DIR  = "symreg_outputs"       # where results are written
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PySR settings — increase niterations for a longer (better) search
# Approach A: cosmo params (Omega_m, sigma_8, w) → shape params
# No trig — cosmological parameters have no physical reason to enter sinusoidally
PYSR_KWARGS_A = dict(
    niterations        = 100,         # increase to 500+ for production
    binary_operators   = ["+", "-", "*", "/", "^"],
    unary_operators    = ["exp", "log", "sqrt", "abs"],
    populations        = 20,
    population_size    = 50,
    maxsize            = 30,
    parsimony          = 1e-4,
    verbosity          = 1,
    random_state       = 42,
)

# Approach B: (Omega_m, sigma_8, w, gaussian_features...) → datavector value
# We pre-compute Gaussian basis features from x so PySR only needs to learn
# how cosmology modulates the amplitudes/widths — not rediscover exp from scratch.
# No trig — the Gaussian structure is already baked into the features.
PYSR_KWARGS_B = dict(
    niterations        = 200,
    binary_operators   = ["+", "-", "*", "/"],
    unary_operators    = ["exp", "sqrt", "abs"],
    populations        = 20,
    population_size    = 50,
    maxsize            = 25,          # keep expressions simple
    parsimony          = 5e-4,        # stronger penalty → simpler expressions
    verbosity          = 1,
    random_state       = 42,
)

# Legacy alias — Approach A code references PYSR_KWARGS
PYSR_KWARGS = PYSR_KWARGS_A

# =============================================================================
# 1.  LOAD DATA
# =============================================================================

def load_data(path: str):
    df = pd.read_csv(path)

    # Auto-detect datavector columns
    global DV_PREFIX
    if DV_PREFIX is not None:
        dv_cols = [c for c in df.columns if c.startswith(DV_PREFIX)]
        dv_cols = sorted(dv_cols, key=lambda c: int(c.replace(DV_PREFIX, "")))
    else:
        # assume remaining columns after COSMO_COLS are the datavector
        dv_cols = [c for c in df.columns if c not in COSMO_COLS]
        # sort numerically if they look like integers
        try:
            dv_cols = sorted(dv_cols, key=lambda c: int(c))
        except ValueError:
            pass

    assert len(dv_cols) == 160, (
        f"Expected 160 datavector columns, found {len(dv_cols)}. "
        "Check COSMO_COLS / DV_PREFIX."
    )

    cosmo = df[COSMO_COLS].values.astype(np.float64)        # (150, 3)
    dv    = df[dv_cols].values.astype(np.float64)           # (150, 160)
    x_arr = np.arange(160, dtype=np.float64)

    print(f"Loaded {cosmo.shape[0]} cosmologies, datavector length {dv.shape[1]}")
    print(f"  Omega_m : [{cosmo[:,0].min():.3f}, {cosmo[:,0].max():.3f}]")
    print(f"  sigma_8 : [{cosmo[:,1].min():.3f}, {cosmo[:,1].max():.3f}]")
    print(f"  w       : [{cosmo[:,2].min():.3f}, {cosmo[:,2].max():.3f}]")
    return cosmo, dv, x_arr

# =============================================================================
# 2.  APPROACH A — SHAPE DECOMPOSITION
# =============================================================================
#
# Three shape models tried on every datavector. Best R² wins.
#
#  Model 1  —  Sum of 2 Gaussians (baseline, 6 params)
#              G(mu1,sig1) + G(mu2,sig2)
#
#  Model 2  —  2 Peaks + negative dip  (9 params)
#              G(mu1,sig1) + G(mu2,sig2) − G(mu_c,sig_c)
#              Directly controls the depth of the trough.
#
#  Model 3  —  Generalised DoG  (5 params)
#              A_w·exp(−|x−μc|^p / σ_w^p) − A_n·exp(−|x−μc|^p / σ_n^p)
#              One exponent p>2 sharpens both peaks & trough simultaneously.
#              Most compact → cleanest PySR expressions.
#
# =============================================================================

from scipy.signal import find_peaks as _find_peaks

# ── Model 1: sum of two Gaussians ────────────────────────────────────────────

def model_sum2g(x, A1, mu1, sig1, A2, mu2, sig2):
    return (A1 * np.exp(-0.5 * ((x - mu1) / sig1) ** 2) +
            A2 * np.exp(-0.5 * ((x - mu2) / sig2) ** 2))

PARAM_NAMES_SUM2G = ["A1", "mu1", "sig1", "A2", "mu2", "sig2"]

def _fit_sum2g(x, y, peaks):
    p0 = [y[peaks[0]], float(peaks[0]), 15.0,
          y[peaks[1]], float(peaks[1]), 15.0]
    bounds = ([0, 0, 1, 0, 0, 1],
              [np.inf, 160, 80, np.inf, 160, 80])
    return curve_fit(model_sum2g, x, y, p0=p0, bounds=bounds, maxfev=15_000)


# ── Model 2: 2 peaks + negative central Gaussian ─────────────────────────────

def model_dip(x, A1, mu1, sig1, A2, mu2, sig2, A3, mu3, sig3):
    """Two positive lobes minus a negative dip Gaussian."""
    return (A1 * np.exp(-0.5 * ((x - mu1) / sig1) ** 2) +
            A2 * np.exp(-0.5 * ((x - mu2) / sig2) ** 2) -
            A3 * np.exp(-0.5 * ((x - mu3) / sig3) ** 2))

PARAM_NAMES_DIP = ["A1","mu1","sig1", "A2","mu2","sig2", "A3_dip","mu3_dip","sig3_dip"]

def _fit_dip(x, y, peaks):
    mu_c = 0.5 * (float(peaks[0]) + float(peaks[1]))
    dip_depth = max(y[peaks[0]], y[peaks[1]]) - y[int(mu_c)]
    p0 = [y[peaks[0]], float(peaks[0]), 12.0,
          y[peaks[1]], float(peaks[1]), 12.0,
          max(dip_depth, 0.01), mu_c, 8.0]
    bounds = ([0, 0, 1, 0, 0, 1, 0, 0, 1],
              [np.inf, 160, 80, np.inf, 160, 80, np.inf, 160, 60])
    return curve_fit(model_dip, x, y, p0=p0, bounds=bounds, maxfev=20_000)


# ── Model 3: generalised DoG ──────────────────────────────────────────────────

def model_gdog(x, A_w, A_n, mu_c, sig_w, sig_n, p):
    """
    Generalised Difference of Gaussians:
        A_w · exp(−|x−μc|^p / σ_w^p) − A_n · exp(−|x−μc|^p / σ_n^p)

    p > 2  → super-Gaussian (flat tops, sharp tails)
    p = 2  → classic DoG
    Naturally creates two symmetric lobes around μc with a deep dip.
    """
    safe_p = np.clip(p, 1.5, 8.0)
    z      = np.abs(x - mu_c)
    return (A_w * np.exp(-(z / sig_w) ** safe_p) -
            A_n * np.exp(-(z / sig_n) ** safe_p))

PARAM_NAMES_GDOG = ["A_w", "A_n", "mu_c", "sig_w", "sig_n", "p_exp"]

def _fit_gdog(x, y, peaks):
    mu_c = 0.5 * (float(peaks[0]) + float(peaks[1]))
    half_sep = 0.5 * abs(float(peaks[1]) - float(peaks[0]))
    A_peak   = max(y[peaks[0]], y[peaks[1]])
    p0 = [A_peak * 1.5, A_peak * 0.5, mu_c,
          half_sep * 1.4, half_sep * 0.6, 3.0]
    bounds = ([0, 0, 0, 5, 2, 1.5],
              [np.inf, np.inf, 160, 100, 60, 8.0])
    return curve_fit(model_gdog, x, y, p0=p0, bounds=bounds, maxfev=20_000)


# ── Shared utilities ──────────────────────────────────────────────────────────

def _get_peaks(y):
    peaks, _ = _find_peaks(y, distance=10)
    if len(peaks) < 2:
        peaks = [np.argmax(y[:80]), np.argmax(y[80:]) + 80]
    return sorted(peaks, key=lambda i: y[i], reverse=True)[:2]


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ── SHAPE_MODEL: configure which model Approach A uses ───────────────────────
#
#  Options: "sum2g"  — Model 1 (original, 6 params)
#           "dip"    — Model 2 (2 peaks + neg dip, 9 params)
#           "gdog"   — Model 3 (generalised DoG, 6 params)  ← recommended
#           "best"   — fit all three per curve, keep highest R²
#
SHAPE_MODEL = "best"    # ← change this to lock a specific model

_MODEL_REGISTRY = {
    "sum2g": (model_sum2g, _fit_sum2g, PARAM_NAMES_SUM2G),
    "dip":   (model_dip,   _fit_dip,   PARAM_NAMES_DIP),
    "gdog":  (model_gdog,  _fit_gdog,  PARAM_NAMES_GDOG),
}


def fit_shape(x, y, model_key="best"):
    """
    Fit requested model (or all three if 'best') to one datavector.
    Returns (model_key_used, popt, y_fit, r2).
    """
    peaks = _get_peaks(y)

    if model_key != "best":
        fn, fit_fn, _ = _MODEL_REGISTRY[model_key]
        try:
            popt, _ = fit_fn(x, y, peaks)
            y_fit   = fn(x, *popt)
            return model_key, popt, y_fit, _r2(y, y_fit)
        except RuntimeError:
            return model_key, None, None, np.nan

    # "best": try all three, always pick highest R².
    # No threshold — just best fit wins.
    candidates = {}  # key -> (popt, y_fit, r2)
    for key, (fn, fit_fn, param_names) in _MODEL_REGISTRY.items():
        try:
            popt, _ = fit_fn(x, y, peaks)
            y_fit   = fn(x, *popt)
            r2      = _r2(y, y_fit)
            candidates[key] = (popt, y_fit, r2)
        except (RuntimeError, ValueError):
            pass

    if not candidates:
        return None, None, None, np.nan

    best_key = max(candidates, key=lambda k: candidates[k][2])
    popt, y_fit, r2 = candidates[best_key]
    return best_key, popt, y_fit, r2


# ── Both set dynamically during decompose_all ────────────────────────────────
SHAPE_PARAM_NAMES: list[str] = []   # filled in by decompose_all
DOMINANT_MODEL:    str        = ""  # filled in by decompose_all


def decompose_all(cosmo, dv, x_arr, plot=True):
    """
    Fit the best shape model to every datavector.
    Returns shape_params (N, K), r2_scores (N,), model_keys_used (N,).
    """
    global SHAPE_PARAM_NAMES

    N          = dv.shape[0]
    all_popts  = [None] * N
    r2_scores  = np.full(N, np.nan)
    model_used = [""] * N
    failed     = []
    model_counts = {k: 0 for k in _MODEL_REGISTRY}

    for i in range(N):
        key, popt, y_fit, r2 = fit_shape(x_arr, dv[i], SHAPE_MODEL)
        model_used[i] = key or "failed"
        if popt is not None and np.isfinite(r2):
            all_popts[i] = popt
            r2_scores[i] = r2
            if key:
                model_counts[key] = model_counts.get(key, 0) + 1
        else:
            failed.append(i)

    # Determine dominant model to fix param count for PySR
    if SHAPE_MODEL == "best":
        dominant = max(model_counts, key=model_counts.get)
        print(f"\n  Model selection: {model_counts}")
        print(f"  Using dominant model '{dominant}' param names for PySR")
    else:
        dominant = SHAPE_MODEL

    global DOMINANT_MODEL, SHAPE_PARAM_NAMES
    DOMINANT_MODEL    = dominant
    SHAPE_PARAM_NAMES = _MODEL_REGISTRY[dominant][2]
    n_params = len(SHAPE_PARAM_NAMES)

    # Build array — pad/truncate to dominant param count
    shape_params = np.full((N, n_params), np.nan)
    for i, popt in enumerate(all_popts):
        if popt is not None:
            k = min(len(popt), n_params)
            shape_params[i, :k] = popt[:k]

    print(f"\n[Decomposition] Fitted {N - len(failed)}/{N} curves successfully")
    if failed:
        print(f"  Failed indices: {failed}")
    print(f"  Median R²: {np.nanmedian(r2_scores):.4f}")
    print(f"  Min    R²: {np.nanmin(r2_scores):.4f}")
    print(f"  Shape params: {SHAPE_PARAM_NAMES}")

    # ── Diagnostic plot ──────────────────────────────────────────────────────
    if plot:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        axes = axes.flatten()
        good = np.where(r2_scores > 0.95)[0]
        bad  = np.argsort(r2_scores)[:2]
        show = list(np.random.choice(good, min(6, len(good)), replace=False)) + list(bad)
        for ax, idx in zip(axes, show):
            ax.plot(x_arr, dv[idx], "k-", lw=1.2, label="data")
            if not np.isnan(shape_params[idx, 0]):
                # re-evaluate using the model that was actually chosen
                used = model_used[idx]
                fn   = _MODEL_REGISTRY.get(used, _MODEL_REGISTRY[dominant])[0]
                try:
                    y_fit = fn(x_arr, *shape_params[idx, :len(_MODEL_REGISTRY[used][2])])
                    ax.plot(x_arr, y_fit, "r--", lw=1.5,
                            label=f"{used} R²={r2_scores[idx]:.3f}")
                except Exception:
                    pass
            ax.set_title(f"cosmo {idx}", fontsize=8)
            ax.legend(fontsize=6)
        plt.suptitle("Shape Decomposition — Sample Fits (model: best)", fontsize=11)
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, "decomposition_sample_fits.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved fit diagnostics → {out}")

        # ── Per-model R² violin ──────────────────────────────────────────────
        _plot_model_comparison(dv, x_arr)

    return shape_params, r2_scores


def _plot_model_comparison(dv, x_arr):
    """R² distribution for each of the 3 models across all cosmologies."""
    r2_all = {k: [] for k in _MODEL_REGISTRY}
    for i in range(dv.shape[0]):
        peaks = _get_peaks(dv[i])
        for key, (fn, fit_fn, _) in _MODEL_REGISTRY.items():
            try:
                popt, _ = fit_fn(x_arr, dv[i], peaks)
                r2_all[key].append(_r2(dv[i], fn(x_arr, *popt)))
            except (RuntimeError, ValueError):
                r2_all[key].append(np.nan)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels  = list(r2_all.keys())
    data    = [np.array(r2_all[k]) for k in labels]
    full_labels = {
        "sum2g": "Sum of 2G\n(6 params)",
        "dip":   "2G + neg dip\n(9 params)",
        "gdog":  "Gen. DoG\n(6 params)",
    }
    vp = ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels([full_labels[l] for l in labels])
    ax.set_ylabel("R²"); ax.set_title("Shape Model Comparison — R² Distribution")
    ax.set_ylim(0.85, 1.01)
    for i, (key, vals) in enumerate(r2_all.items()):
        med = np.nanmedian(vals)
        ax.text(i+1, 0.862, f"med={med:.4f}", ha="center", fontsize=8, color="navy")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "model_comparison_r2.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved model comparison → {out}")


def run_approach_a(cosmo, dv, x_arr):
    """
    APPROACH A: symbolic regression on each shape parameter separately.
    Outputs one PySR model per shape param.
    """
    print("\n" + "="*60)
    print("APPROACH A — Shape Decomposition + Symbolic Regression")
    print("="*60)

    shape_params, r2_scores = decompose_all(cosmo, dv, x_arr)

    # Drop rows where fit failed
    valid = ~np.isnan(shape_params[:, 0])
    X = cosmo[valid]            # (N_valid, 3)  — Omega_m, sigma_8, w
    P = shape_params[valid]     # (N_valid, 6)  — shape params

    print(f"\nRunning PySR on {valid.sum()} valid cosmologies...")

    results_a = {}

    for j, pname in enumerate(SHAPE_PARAM_NAMES):
        y_j = P[:, j]
        print(f"\n  [{j+1}/6] Regressing {pname}  "
              f"(range [{y_j.min():.3f}, {y_j.max():.3f}])")

        model = PySRRegressor(**PYSR_KWARGS)
        model.fit(X, y_j,
                  variable_names=COSMO_COLS)

        best = model.sympy()
        print(f"  Best expression for {pname}: {best}")

        results_a[pname] = model
        _plot_shape_param_parity(y_j, model.predict(X), pname)

    _plot_shape_param_correlations(cosmo[valid], shape_params[valid])
    _summarise_approach_a(results_a)
    return results_a, shape_params


def _plot_shape_param_parity(y_true, y_pred, pname):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y_true, y_pred, s=15, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    r, _ = pearsonr(y_true, y_pred)
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    ax.set_title(f"{pname}  (r={r:.3f})")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"parity_{pname}.png")
    plt.savefig(out, dpi=120); plt.close()


def _plot_shape_param_correlations(cosmo, shape_params):
    """Heatmap of correlations between cosmo params and shape params."""
    combined = np.hstack([cosmo, shape_params])
    labels   = COSMO_COLS + SHAPE_PARAM_NAMES
    corr     = np.corrcoef(combined.T)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation: Cosmo Params ↔ Shape Params")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\n  Saved correlation heatmap → {out}")


def _summarise_approach_a(results_a):
    dominant = DOMINANT_MODEL
    fn = _MODEL_REGISTRY[dominant][0]
    print("\n── Approach A Summary ──────────────────────────────────")
    print(f"  Shape model : {SHAPE_MODEL}  (function: {fn.__name__})")
    for pname, model in results_a.items():
        print(f"  {pname:12s}  →  {model.sympy()}")
    print()
    print("Usage: reconstruct a new datavector with:")
    print("  params = [models[p].predict([[Om, s8, w]])[0] for p in SHAPE_PARAM_NAMES]")
    print("  y_pred = shape_fn(x_arr, *params)")


# =============================================================================
# 3.  APPROACH B — DIRECT SYMBOLIC REGRESSION
# =============================================================================

def run_approach_b(cosmo, dv, x_arr, shape_params=None, n_components=5):
    """
    APPROACH B — PCA + Symbolic Regression.

    Genuinely different from Approach A (Gaussian decomposition):
      - No assumption that the curve is a sum/difference of Gaussians
      - PCA finds the actual data-driven modes of variation across cosmologies
      - PySR learns f(Omega_m, sigma_8, w) -> PC score for each component
      - Reconstruction: mean_curve + sum_k [ score_k(Om,s8,w) * PC_k(x) ]

    Why this is distinct from Approach A:
      Approach A asks: "how do the Gaussian parameters change with cosmology?"
      Approach B asks: "what are the natural modes of variation in the data,
                        and how do cosmological parameters drive those modes?"

    If the curves vary in ways a Gaussian can't capture (asymmetry, tail
    kurtosis, baseline offsets), PCA will find those modes and PySR will
    learn to predict them. The basis is data-driven, not physics-assumed.

    n_components: number of PCs to retain (check explained_variance plot
                  to decide — typically 3-5 suffice for >99% variance).
    """
    from sklearn.decomposition import PCA

    print("\n" + "="*60)
    print("APPROACH B — PCA + Symbolic Regression")
    print("="*60)

    # ── 1. PCA on the datavector matrix ──────────────────────────────────────
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(dv)          # (N_cosmo, n_components)
    evr    = pca.explained_variance_ratio_

    print(f"  PCA: {n_components} components explain "
          f"{evr.sum()*100:.2f}% of total variance")
    for k, e in enumerate(evr):
        print(f"    PC{k+1}: {e*100:.2f}%")

    # ── 2. Plot PCA diagnostics ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scree plot
    axes[0].bar(range(1, n_components+1), evr*100)
    axes[0].set_xlabel("PC"); axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Scree plot"); axes[0].set_xticks(range(1, n_components+1))

    # PC shapes
    for k in range(n_components):
        axes[1].plot(x_arr, pca.components_[k],
                     label=f"PC{k+1} ({evr[k]*100:.1f}%)", lw=1.2)
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set_xlabel("x"); axes[1].set_title("Principal component shapes")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "approach_b_pca_diagnostics.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved PCA diagnostics → {out}")

    # ── 3. Symbolic regression: cosmology → PC scores ─────────────────────────
    print(f"\n  Running PySR on {n_components} PC scores vs cosmological params...")
    pc_models = {}

    for k in range(n_components):
        y_k = scores[:, k]
        print(f"\n  [PC{k+1}] score range [{y_k.min():.4f}, {y_k.max():.4f}]")

        model_k = PySRRegressor(**PYSR_KWARGS_B)
        model_k.fit(cosmo, y_k, variable_names=COSMO_COLS)

        expr = model_k.sympy()
        print(f"  PC{k+1} expression: {expr}")
        pc_models[k] = model_k

        # Parity plot
        _plot_shape_param_parity(y_k, model_k.predict(cosmo), f"PC{k+1}")

    # ── 4. Evaluate reconstructions ───────────────────────────────────────────
    print("\n  Evaluating reconstructions on sample cosmologies...")
    n_show   = min(6, cosmo.shape[0])
    idx_show = np.random.choice(cosmo.shape[0], n_show, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    for ax, i in zip(axes, idx_show):
        Om, s8, w = cosmo[i]

        # Predict PC scores from cosmology using symbolic expressions
        pred_scores = np.array([
            pc_models[k].predict([[Om, s8, w]])[0]
            for k in range(n_components)
        ])
        # Reconstruct: mean + scores @ components
        y_pred = pca.mean_ + pred_scores @ pca.components_
        r2 = _r2(dv[i], y_pred)

        ax.plot(x_arr, dv[i],  "k-",  lw=1.2, label="data")
        ax.plot(x_arr, y_pred, "r--", lw=1.5, label=f"PySR R²={r2:.3f}")
        ax.set_title(
            f"cosmo {i}  Ωm={Om:.2f} σ8={s8:.2f} w={w:.2f}", fontsize=7)
        ax.legend(fontsize=6)

    plt.suptitle("Approach B — PCA + Symbolic Regression", fontsize=10)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "approach_b_fits.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved fit plot → {out}")

    # Summary
    print("\n── Approach B Summary ──────────────────────────────────")
    print(f"  Basis: {n_components} PCA components ({evr.sum()*100:.2f}% variance)")
    for k, model_k in pc_models.items():
        print(f"  PC{k+1} → {model_k.sympy()}")

    return pc_models, pca





def _plot_pareto_front(model):
    """Plot complexity vs loss Pareto front from PySR."""
    try:
        eqs = model.equations_
        if eqs is None or "complexity" not in eqs.columns:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(eqs["complexity"], eqs["loss"], s=30, zorder=3)
        # annotate top 5 by loss
        top = eqs.nsmallest(5, "loss")
        for _, row in top.iterrows():
            ax.annotate(str(row["sympy_format"])[:40],
                        (row["complexity"], row["loss"]),
                        fontsize=6, xytext=(5, 5),
                        textcoords="offset points", zorder=4)
        ax.set_xlabel("Complexity (# nodes)")
        ax.set_ylabel("Loss (MSE)")
        ax.set_yscale("log")
        ax.set_title("PySR Pareto Front — Approach B")
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, "approach_b_pareto.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"  Saved Pareto front → {out}")
    except Exception:
        pass


# =============================================================================
# 4.  COMPARISON UTILITY
# =============================================================================

def compare_approaches(cosmo, dv, x_arr, results_a, shape_params, pc_models,
                        pca, n_show=6):
    """
    Side-by-side reconstruction from both approaches for random cosmologies.
    """
    print("\n── Comparing both approaches ─────────────────────────────")
    valid = ~np.isnan(shape_params[:, 0])
    idx_pool = np.where(valid)[0]
    idx_show = idx_pool[np.random.choice(len(idx_pool), n_show, replace=False)]

    fig = plt.figure(figsize=(18, 4 * n_show))
    gs  = gridspec.GridSpec(n_show, 3, hspace=0.5, wspace=0.3)

    for row, i in enumerate(idx_show):
        Om, s8, w = cosmo[i]

        # — Approach A reconstruction —
        try:
            p_a = np.array([results_a[pname].predict([[Om, s8, w]])[0]
                            for pname in SHAPE_PARAM_NAMES])
            # Clip each predicted param to [min*0.5, max*2] of training range
            # to prevent PySR expressions from blowing up outside training hull
            valid_mask = ~np.isnan(shape_params[:, 0])
            p_train    = shape_params[valid_mask]
            p_lo = p_train.min(axis=0) - 0.5 * p_train.std(axis=0)
            p_hi = p_train.max(axis=0) + 0.5 * p_train.std(axis=0)
            p_a  = np.clip(p_a, p_lo, p_hi)
            dom  = DOMINANT_MODEL
            fn_a = _MODEL_REGISTRY[dom][0]
            y_a  = fn_a(x_arr, *p_a)
            r2_a = _r2(dv[i], y_a)
        except Exception:
            y_a, r2_a = np.zeros_like(x_arr), float("nan")

        # — Approach B reconstruction (PCA + symbolic PC scores) —
        try:
            pred_scores = np.array([
                pc_models[k].predict([[Om, s8, w]])[0]
                for k in range(len(pc_models))
            ])
            y_b  = pca.mean_ + pred_scores @ pca.components_
            r2_b = _r2(dv[i], y_b)
        except Exception:
            y_b, r2_b = np.zeros_like(x_arr), float("nan")

        title = f"cosmo {i}  Ωm={Om:.2f}  σ8={s8:.2f}  w={w:.2f}"

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.plot(x_arr, dv[i], "k.", ms=2); ax0.set_title(f"Data  —  {title}", fontsize=7)

        ax1 = fig.add_subplot(gs[row, 1])
        ax1.plot(x_arr, dv[i], "k.", ms=2)
        ax1.plot(x_arr, y_a, "b-", lw=1.5, label=f"R²={r2_a:.3f}")
        ax1.set_title(f"Approach A (decomp)\n{title}", fontsize=7); ax1.legend(fontsize=6)

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.plot(x_arr, dv[i], "k.", ms=2)
        ax2.plot(x_arr, y_b, "r-", lw=1.5, label=f"R²={r2_b:.3f}")
        ax2.set_title(f"Approach B (PCA)\n{title}", fontsize=7); ax2.legend(fontsize=6)

    out = os.path.join(OUTPUT_DIR, "comparison_both_approaches.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved comparison → {out}")


# =============================================================================
# 5.  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # ── Load ─────────────────────────────────────────────────────────────────
    cosmo, dv, x_arr = load_data(CSV_PATH)

    # ── Approach A ───────────────────────────────────────────────────────────
    results_a, shape_params = run_approach_a(cosmo, dv, x_arr)

    # ── Approach B ───────────────────────────────────────────────────────────
    # subsample_x=1  → use all 160 x values (slower, richer signal)
    # subsample_x=8  → every 8th point (faster first pass)
    pc_models, pca = run_approach_b(cosmo, dv, x_arr)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    compare_approaches(cosmo, dv, x_arr, results_a, shape_params, pc_models, pca)

    print("\n✓ Done.  All outputs written to:", OUTPUT_DIR)