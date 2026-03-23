"""
cosmo_physo_pca.py  —  PCA + PhySO Symbolic Regression on Wavelet L1-Norm
==========================================================================
Finds K closed-form expressions:

    score_k(Omega_m, sigma_8, w) = f_k(Omega_m_std, sigma_8_std, w_std)

that reconstruct the full datavector as:

    L1(x; Omega_m, sigma_8, w) = mean_curve(x)
                                + f_1(Om,s8,w) * PC_1(x)
                                + f_2(Om,s8,w) * PC_2(x)  + ...

Pipeline:
  1. Split cosmologies 80/20 train/test  (at cosmology level)
  2. Fit PCA on training datavectors only
  3. Standardise Omega_m, sigma_8, w on training set only  →  save scaler
  4. Run PhySO on each PC score vs standardised cosmo params
  5. Evaluate final R² on TEST set only

Saved outputs:
  test_cosmologies.pkl         — 20% held-out set
  cosmo_scaler.pkl             — StandardScaler (train only)
  pca_basis.pkl                — sklearn PCA object (train only)
  pca_mean.npy / pca_components.npy  — numpy arrays for standalone use
  PC{k}/best_expression.txt    — symbolic expression per PC
  PC{k}/expression.pkl         — PhySO expression object
  pca_diagnostics.png          — scree plot + PC shapes
  reconstructions_test.png     — fits on TEST set
  r2_distribution_test.png     — R² histogram on TEST set

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
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
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
OUTPUT_DIR  = "physo_pca_outputs"
TEST_SIZE   = 0.20
RANDOM_SEED = 42
N_COMPONENTS = 5      # PCA components — check scree plot, usually 3-5 suffice

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── PhySO settings ────────────────────────────────────────────────────────────
# PC scores are smooth scalar functions of 3 standardised variables.
# No trig — no physical motivation for oscillatory cosmo dependence.
OP_NAMES = [
    "mul",    # x * y
    "add",    # x + y
    "sub",    # x - y
    "div",    # x / y
    "inv",    # 1 / x
    "n2",     # x^2  — power-law building block
    "exp",    # e^x
    "log",    # ln(x)  — enables Omega_m^p via exp(p*log(Omega_m))
    "sqrt",   # sqrt(x)
]

N_FREE_PARAMS     = 3         # free constants c0, c1, c2 per expression
FIXED_CONSTS      = [1.]
MAX_N_EVALUATIONS = int(1e6)  # increase to 1e7+ for publication runs
N_EPOCHS          = int(20)
PARALLEL_MODE     = True
N_CPUS            = 8

# =============================================================================
# UTILITIES
# =============================================================================

def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

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

    Each cosmology is one independent simulation. The test set contains
    entirely unseen (Omega_m, sigma_8, w) combinations — a proper
    out-of-distribution evaluation. Row-level splitting would leak
    information since the same cosmology would appear in both sets.
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
# 3.  FIT PCA  (training datavectors only)
# =============================================================================

def fit_pca(dv_train, x_arr, n_components):
    """
    Fit PCA on TRAINING datavectors only.

    Fitting on all data would leak test-set structure into the PC shapes.
    The mean curve and PC eigenvectors are then fixed — they never change.
    """
    pca    = PCA(n_components=n_components)
    scores = pca.fit_transform(dv_train)   # (N_train, K)
    evr    = pca.explained_variance_ratio_

    print(f"\nPCA (fitted on training set):")
    print(f"  {n_components} components explain {evr.sum()*100:.3f}% of variance")
    for k, e in enumerate(evr):
        print(f"    PC{k+1}: {e*100:.3f}%  (cumulative: {evr[:k+1].sum()*100:.3f}%)")

    # Diagnostics plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(range(1, n_components+1), evr*100, color="steelblue", edgecolor="k")
    axes[0].set_xlabel("PC"); axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Scree Plot — Training Set")
    axes[0].set_xticks(range(1, n_components+1))
    for k, e in enumerate(evr):
        axes[0].text(k+1, e*100+0.1, f"{e*100:.2f}%", ha="center", fontsize=7)
    for k in range(n_components):
        axes[1].plot(x_arr, pca.components_[k],
                     label=f"PC{k+1} ({evr[k]*100:.1f}%)", lw=1.3)
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set_xlabel("x (wavelet scale index)")
    axes[1].set_title("PC Shapes  (fixed basis vectors)")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_diagnostics.png"), dpi=150)
    plt.close()
    print(f"  Saved PCA diagnostics → {os.path.join(OUTPUT_DIR, 'pca_diagnostics.png')}")

    # Save PCA basis
    pca_pkl = os.path.join(OUTPUT_DIR, "pca_basis.pkl")
    with open(pca_pkl, "wb") as f:
        pickle.dump(pca, f)
    np.save(os.path.join(OUTPUT_DIR, "pca_mean.npy"),       pca.mean_)
    np.save(os.path.join(OUTPUT_DIR, "pca_components.npy"), pca.components_)
    print(f"  Saved PCA basis → {pca_pkl}")
    print(f"  Saved pca_mean.npy + pca_components.npy")

    return pca, scores

# =============================================================================
# 4.  STANDARDISE COSMOLOGICAL INPUTS  (fit on train only)
# =============================================================================

def fit_scaler(cosmo_train):
    """
    Fit StandardScaler on TRAINING cosmologies only.

    Why standardise:
      w is centred at -1.0 while Omega_m and sigma_8 are near 0.3 and 0.75.
      Without standardisation, w appears less informative to the RNN because
      its offset from zero is large. After standardisation all three variables
      have mean=0, std=1 — PhySO's RNN treats them with equal importance.

    Why train only:
      Fitting on all data leaks test statistics into the standardisation.
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
# 5.  PHYSO HELPER
# =============================================================================

def _physo_sr(X_std_np, y_np, target_name, run_subdir):
    """
    Run one PhySO SR call.

    X_std_np : (3, N_train) — standardised cosmo params, PhySO row-per-variable format
    y_np     : (N_train,)   — target PC score
    """
    os.makedirs(run_subdir, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(run_subdir)

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

    print(f"\n  [PhySO] target={target_name}  "
          f"samples={X_std_np.shape[1]}  vars={X_std_np.shape[0]}")

    expression, logs = physo.SR(
        X_std_np,
        y_np,
        X_names           = ["Omega_m_std", "sigma_8_std", "w_std"],
        y_name            = target_name,
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

    with open("best_expression.txt", "w") as f:
        f.write(f"Target     : {target_name}\n")
        f.write(f"Inputs     : Omega_m_std, sigma_8_std, w_std\n")
        f.write(f"Expression : {expression}\n")

    with open("expression.pkl", "wb") as f:
        pickle.dump(expression, f)

    print(f"  [PhySO] Best: {expression}")
    os.chdir(orig_dir)
    return expression


def _predict_one(expression, cosmo_row_std):
    """Evaluate expression for one standardised cosmology. Returns scalar."""
    X = np.array(cosmo_row_std, dtype=np.float64).reshape(-1, 1)

    X_t    = torch.tensor(X, dtype=torch.float64)
    result_tensor = expression.execute(X_t)
    result    = result_tensor.detach().numpy() if hasattr(result_tensor, 'detach') else np.asarray(result_tensor)

    return float(result[0]) if hasattr(result, '__len__') else float(result)


def _predict_batch(expression, cosmo_std):
    """cosmo_std: (N, 3) standardised → (N,) predictions."""

    X_t    = torch.tensor(cosmo_std.T.astype(np.float64), dtype=torch.float64)
    result_tensor = expression.execute(X_t)
    result    = result_tensor.detach().numpy() if hasattr(result_tensor, 'detach') else np.asarray(result_tensor)

    return np.asarray(result, dtype=np.float64).flatten()

# =============================================================================
# 6.  RUN SR ON PC SCORES
# =============================================================================

def run_pca_sr(cosmo, dv, x_arr, n_components=N_COMPONENTS):
    print("\n" + "="*60)
    print("PCA + PhySO SR  —  cosmology → L1-norm datavector")
    print("="*60)

    # ── Split ─────────────────────────────────────────────────────────────────
    cosmo_train, dv_train, cosmo_test, dv_test = split_data(cosmo, dv)

    # ── PCA on training datavectors only ──────────────────────────────────────
    pca, scores_train = fit_pca(dv_train, x_arr, n_components)

    # ── Standardise cosmo params on training set only ─────────────────────────
    scaler     = fit_scaler(cosmo_train)
    cosmo_std  = scaler.transform(cosmo_train)     # (N_train, 3)
    X_std      = cosmo_std.T                        # (3, N_train) — PhySO format

    # ── PhySO on each PC score ─────────────────────────────────────────────────
    pc_models = {}
    print(f"\nRunning PhySO on {n_components} PC scores...")

    for k in range(n_components):
        y_k = scores_train[:, k].astype(np.float64)
        print(f"\n  [PC{k+1}/{n_components}]  "
              f"score range [{y_k.min():.4f}, {y_k.max():.4f}]")

        subdir = os.path.join(OUTPUT_DIR, f"PC{k+1}")
        expr   = _physo_sr(X_std, y_k, f"PC{k+1}_score", subdir)
        pc_models[k] = expr

        # Parity plot on training data
        y_pred_train = _predict_batch(expr, cosmo_std)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(y_k, y_pred_train, s=15, alpha=0.7, color="steelblue")
        lims = [min(y_k.min(), y_pred_train.min()),
                max(y_k.max(), y_pred_train.max())]
        ax.plot(lims, lims, "r--", lw=1)
        try:
            r, _ = pearsonr(y_k, y_pred_train)
            ax.set_title(f"PC{k+1} score — train  (r={r:.4f})")
        except Exception:
            ax.set_title(f"PC{k+1} score — train")
        ax.set_xlabel("True score"); ax.set_ylabel("Predicted score")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"parity_PC{k+1}_train.png"), dpi=120)
        plt.close()

    # ── Save summary expression file ──────────────────────────────────────────
    evr = pca.explained_variance_ratio_
    summary_path = os.path.join(OUTPUT_DIR, "expressions_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PCA + PhySO SR — Expression Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"PCA basis variance: {evr.sum()*100:.3f}%  ({n_components} components)\n\n")
        f.write("Full closed-form emulator:\n")
        f.write("  L1(x; Om, s8, w) = mean_curve(x)\n")
        for k in range(n_components):
            f.write(f"                   + [{pc_models[k]}] * PC{k+1}(x)\n")
        f.write("\nStandardisation:\n")
        for i, name in enumerate(COSMO_COLS):
            f.write(f"  {name}_std = ({name} - {scaler.mean_[i]:.6f})"
                    f" / {scaler.scale_[i]:.6f}\n")
        f.write("\nFiles:\n")
        f.write("  pca_mean.npy        — mean L1-norm curve, shape (160,)\n")
        f.write("  pca_components.npy  — PC shapes, shape (K, 160)\n")
        f.write("  cosmo_scaler.pkl    — StandardScaler\n")
    print(f"\nSaved expression summary → {summary_path}")

    # ── Evaluate on TEST SET ONLY ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION ON TEST SET ({cosmo_test.shape[0]} unseen cosmologies)")
    print(f"{'='*60}")
    cosmo_test_std = scaler.transform(cosmo_test)
    _plot_reconstructions(pc_models, pca, cosmo_test, cosmo_test_std,
                          dv_test, x_arr)
    _plot_r2_distribution(pc_models, pca, cosmo_test, cosmo_test_std,
                          dv_test)

    # ── Print final summary ────────────────────────────────────────────────────
    print(f"\n── Full Emulator ────────────────────────────────────────")
    print(f"  L1(x; Om, s8, w) = mean_curve(x)")
    for k in range(n_components):
        print(f"                   + [{pc_models[k]}] * PC{k+1}(x)")

    return pc_models, pca, scaler

# =============================================================================
# 7.  PLOTS
# =============================================================================

def _reconstruct(pc_models, pca, cosmo_row_std):
    """Reconstruct one datavector from standardised cosmo params."""
    K = len(pc_models)
    scores = np.array([_predict_one(pc_models[k], cosmo_row_std)
                       for k in range(K)])
    return pca.mean_ + scores @ pca.components_


def _plot_reconstructions(pc_models, pca, cosmo_test, cosmo_test_std,
                           dv_test, x_arr, n_show=6):
    n_show   = min(n_show, cosmo_test.shape[0])
    idx_show = np.random.choice(cosmo_test.shape[0], n_show, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    for ax, i in zip(axes, idx_show):
        Om, s8, w = cosmo_test[i]
        y_pred = _reconstruct(pc_models, pca, cosmo_test_std[i])
        r2     = _r2(dv_test[i], y_pred)
        ax.plot(x_arr, dv_test[i], "k-",  lw=1.2, label="data")
        ax.plot(x_arr, y_pred,     "r--", lw=1.5, label=f"PhySO  R²={r2:.4f}")
        ax.set_title(f"cosmo {i}  Ωm={Om:.2f}  σ8={s8:.2f}  w={w:.2f}",
                     fontsize=7)
        ax.legend(fontsize=6)

    plt.suptitle("PCA + PhySO SR — TEST SET", fontsize=10)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "reconstructions_test.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved reconstructions (test) → {out}")


def _plot_r2_distribution(pc_models, pca, cosmo_test, cosmo_test_std, dv_test):
    r2_all = np.array([
        _r2(dv_test[i], _reconstruct(pc_models, pca, cosmo_test_std[i]))
        for i in range(cosmo_test.shape[0])
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
# 8.  PREDICT FOR UNSEEN COSMOLOGY
# =============================================================================

def predict(Omega_m, sigma_8, w, pc_models, pca, scaler, x_arr=None):
    """
    Predict the full L1-norm datavector for any new cosmology.

    Parameters
    ----------
    Omega_m, sigma_8, w : float  — raw (unstandardised) cosmological parameters
    pc_models           : dict {k: PhySO expression}
    pca                 : fitted sklearn PCA object  (or load pca_basis.pkl)
    scaler              : fitted StandardScaler      (or load cosmo_scaler.pkl)
    x_arr               : np.ndarray or None  — defaults to np.arange(160)

    Returns
    -------
    y_pred  : np.ndarray (160,)
    scores  : np.ndarray (K,)   — predicted PC scores (diagnostic)

    How it works
    ------------
    1. Standardise: cosmo_std = scaler.transform([[Om, s8, w]])[0]
    2. Evaluate each f_k(cosmo_std) → K scalar scores
    3. Reconstruct: pca.mean_ + scores @ pca.components_

    Pure-numpy version (no PhySO needed after training):
    >>> pca_mean  = np.load("physo_pca_outputs/pca_mean.npy")
    >>> pca_comps = np.load("physo_pca_outputs/pca_components.npy")
    >>> scaler    = pickle.load(open("physo_pca_outputs/cosmo_scaler.pkl","rb"))
    >>> cosmo_std = scaler.transform([[Om, s8, w]])[0]
    >>> scores    = np.array([f_k(cosmo_std) for k in range(K)])  # hand-coded
    >>> y_pred    = pca_mean + scores @ pca_comps
    """
    if x_arr is None:
        x_arr = np.arange(160, dtype=np.float64)
    K          = len(pc_models)
    cosmo_std  = scaler.transform([[Omega_m, sigma_8, w]])[0]
    scores     = np.array([_predict_one(pc_models[k], cosmo_std)
                           for k in range(K)])
    y_pred     = pca.mean_ + scores @ pca.components_
    return y_pred, scores

# =============================================================================
# 9.  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhySO PCA SR: cosmology → wavelet L1-norm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_pcs",     default=N_COMPONENTS, type=int,
                        help="Number of PCA components.")
    parser.add_argument("--n_free",    default=N_FREE_PARAMS, type=int,
                        help="Free constants per PhySO expression.")
    parser.add_argument("--max_evals", default=MAX_N_EVALUATIONS, type=int,
                        help="Max evaluations per PhySO run.")
    parser.add_argument("--ncpus",     default=N_CPUS, type=int,
                        help="CPUs for parallel evaluation.")
    parser.add_argument("--seed",      default=RANDOM_SEED, type=int,
                        help="Random seed.")
    args = parser.parse_args()

    N_FREE_PARAMS     = args.n_free
    MAX_N_EVALUATIONS = args.max_evals
    N_CPUS            = args.ncpus
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cosmo, dv, x_arr = load_data(CSV_PATH)

    pc_models, pca, scaler = run_pca_sr(cosmo, dv, x_arr, n_components=args.n_pcs)

    print(f"\n✓ Done.  Outputs → {OUTPUT_DIR}/")
    print(f"\nSaved files:")
    print(f"  test_cosmologies.pkl     — 20% held-out test set")
    print(f"  cosmo_scaler.pkl         — standardisation parameters")
    print(f"  pca_basis.pkl            — PCA object")
    print(f"  pca_mean.npy             — mean datavector")
    print(f"  pca_components.npy       — PC shapes")
    print(f"  expressions_summary.txt  — full emulator equation")
    print(f"\nTo predict a new cosmology:")
    print(f"  y, scores = predict(Omega_m, sigma_8, w, pc_models, pca, scaler)")