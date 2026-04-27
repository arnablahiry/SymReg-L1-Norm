"""
evaluate_physo_pca.py  —  Evaluate trained PhySO PCA model on test set
=======================================================================
Loads all saved artifacts from training and evaluates on the held-out
test set. No retraining — purely inference + reconstruction.

Inverse transform:
    L1(x) = pca.mean_(x) + sum_k [ f_k(Om, s8, w) * PC_k(x) ]

    where f_k are the PhySO expressions, evaluated at standardised inputs.

Required files (all saved by cosmo_physo_pca.py):
    physo_pca_outputs/test_cosmologies.pkl    — held-out test set
    physo_pca_outputs/cosmo_scaler.pkl        — StandardScaler
    physo_pca_outputs/pca_basis.pkl           — fitted PCA object
    physo_pca_outputs/PC{k}/expression.pkl   — PhySO expression per PC

Usage:
    python evaluate_physo_pca.py
    python evaluate_physo_pca.py --output_dir physo_pca_outputs --n_pcs 5
"""

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import torch
import pickle
import os
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")   # where training saved everything
N_COMPONENTS = 5                     # must match what was used in training
RANDOM_SEED  = 42

# =============================================================================
# UTILITIES
# =============================================================================

def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _execute(expression, X_np):
    """
    Run a PhySO expression on a numpy input array.
    PhySO requires torch tensors internally — convert in, convert out.

    X_np : (n_vars, n_samples)  numpy array
    returns : (n_samples,)      numpy array
    """
    X_t    = torch.tensor(X_np, dtype=torch.float64)
    result = expression.execute(X_t)
    arr    = result.detach().numpy() if hasattr(result, 'detach') else np.asarray(result)
    return arr.flatten().astype(np.float64)


def predict_one(cosmo_row, expressions, pca, scaler):
    """
    Reconstruct the full L1-norm datavector for one cosmology.

    Steps:
      1. Standardise raw cosmo params using saved scaler
      2. Evaluate each f_k(cosmo_std) → K PC scores
      3. Inverse PCA: mean_curve + scores @ PC_shapes

    cosmo_row  : [Omega_m, sigma_8, w]  — raw values
    expressions: list of K PhySO expression objects
    pca        : fitted sklearn PCA
    scaler     : fitted StandardScaler

    Returns y_pred : (160,)
    """
    # Step 1: standardise
    cosmo_std = scaler.transform([cosmo_row])[0]       # (3,)

    # Step 2: evaluate each PC expression
    # PhySO format: (n_vars, n_samples) — here 1 sample
    X = cosmo_std.reshape(-1, 1).astype(np.float64)    # (3, 1)
    scores = np.array([
        _execute(expr, X)[0]
        for expr in expressions
    ])                                                  # (K,)

    # Step 3: inverse PCA transform
    # pca.mean_       : (160,)   — mean datavector across training set
    # pca.components_ : (K, 160) — PC shapes (fixed, from training)
    y_pred = pca.mean_ + scores @ pca.components_      # (160,)

    return y_pred, scores

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

def load_artifacts(output_dir, n_components):
    """Load all saved training artifacts from disk."""

    print(f"Loading artifacts from: {output_dir}/")

    # Test set
    test_pkl = os.path.join(output_dir, "test_cosmologies.pkl")
    with open(test_pkl, "rb") as f:
        test_data = pickle.load(f)
    cosmo_test = test_data["cosmo"]
    dv_test    = test_data["dv"]
    print(f"  Test set       : {cosmo_test.shape[0]} cosmologies")

    # StandardScaler
    scaler_pkl = os.path.join(output_dir, "cosmo_scaler.pkl")
    with open(scaler_pkl, "rb") as f:
        scaler = pickle.load(f)
    print(f"  Scaler         : loaded  "
          f"(means: {scaler.mean_.round(4).tolist()})")

    # PCA basis
    pca_pkl = os.path.join(output_dir, "pca_basis.pkl")
    with open(pca_pkl, "rb") as f:
        pca = pickle.load(f)
    evr = pca.explained_variance_ratio_
    print(f"  PCA basis      : {n_components} components, "
          f"{evr.sum()*100:.3f}% variance explained")
    for k, e in enumerate(evr):
        print(f"    PC{k+1}: {e*100:.3f}%")

    # PhySO expressions — one per PC
    expressions = []
    for k in range(n_components):
        expr_pkl = os.path.join(output_dir, f"PC{k+1}", "expression.pkl")
        with open(expr_pkl, "rb") as f:
            expr = pickle.load(f)
        expressions.append(expr)
        print(f"  PC{k+1} expression : {expr}")

    return cosmo_test, dv_test, scaler, pca, expressions

# =============================================================================
# EVALUATE ON TEST SET
# =============================================================================

def evaluate_test_set(cosmo_test, dv_test, expressions, pca, scaler,
                      output_dir):
    """
    Run full evaluation on test set:
      - Reconstruct each test datavector
      - Compute R² per cosmology
      - Save plots
    """
    x_arr    = np.arange(160, dtype=np.float64)
    N_test   = cosmo_test.shape[0]
    n_comps  = len(expressions)

    print(f"\nReconstructing {N_test} test cosmologies...")

    y_preds  = np.zeros((N_test, 160))
    all_scores = np.zeros((N_test, n_comps))
    r2_all   = np.zeros(N_test)

    for i in range(N_test):
        y_pred, scores       = predict_one(cosmo_test[i], expressions, pca, scaler)
        y_preds[i]           = y_pred
        all_scores[i]        = scores
        r2_all[i]            = _r2(dv_test[i], y_pred)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nR² on TEST set ({N_test} unseen cosmologies):")
    print(f"  Median : {np.median(r2_all):.4f}")
    print(f"  Mean   : {np.mean(r2_all):.4f}")
    print(f"  Min    : {np.min(r2_all):.4f}")
    print(f"  Max    : {np.max(r2_all):.4f}")
    print(f"  Std    : {np.std(r2_all):.4f}")
    print(f"  >0.99  : {(r2_all > 0.99).sum()} / {N_test}")
    print(f"  >0.95  : {(r2_all > 0.95).sum()} / {N_test}")

    # ── Reconstruction plot (random sample) ───────────────────────────────────
    np.random.seed(RANDOM_SEED)
    n_show   = min(6, N_test)
    idx_show = np.random.choice(N_test, n_show, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for ax, i in zip(axes, idx_show):
        Om, s8, w = cosmo_test[i]
        ax.plot(x_arr, dv_test[i],  "k-",  lw=1.2, label="data")
        ax.plot(x_arr, y_preds[i],  "r--", lw=1.5,
                label=f"PhySO PCA  R²={r2_all[i]:.4f}")
        ax.set_title(f"cosmo {i}  Om={Om:.2f}  s8={s8:.2f}  w={w:.2f}",
                     fontsize=7)
        ax.legend(fontsize=6)
    plt.suptitle("PhySO PCA — TEST SET Reconstructions", fontsize=10)
    plt.rcParams.update({'text.usetex': False})
    plt.tight_layout()
    out = os.path.join(output_dir, "eval_reconstructions_test.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\nSaved reconstructions → {out}")

    # ── R² distribution ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(r2_all, bins=20, edgecolor="k", color="steelblue")
    axes[0].set_xlabel("R²"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"R² — TEST SET  (median={np.median(r2_all):.4f})")
    axes[0].axvline(np.median(r2_all), color="r", ls="--", lw=1,
                    label=f"median={np.median(r2_all):.4f}")
    axes[0].legend(fontsize=8)
    axes[1].plot(np.sort(r2_all), "k-", lw=1.2)
    axes[1].axhline(0.99, color="r",  ls="--", lw=1, label="R²=0.99")
    axes[1].axhline(0.95, color="orange", ls="--", lw=1, label="R²=0.95")
    axes[1].set_xlabel("Cosmology (sorted by R²)")
    axes[1].set_ylabel("R²")
    axes[1].set_title("Sorted R² — TEST SET")
    axes[1].legend(fontsize=8)
    plt.rcParams.update({'text.usetex': False})
    plt.tight_layout()
    out = os.path.join(output_dir, "eval_r2_distribution_test.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved R² distribution → {out}")

    # ── PC score distributions ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_comps, figsize=(3 * n_comps, 3))
    if n_comps == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        ax.hist(all_scores[:, k], bins=15, edgecolor="k", color="steelblue")
        ax.set_title(f"PC{k+1} scores", fontsize=8)
        ax.set_xlabel("Predicted score"); ax.set_ylabel("Count")
    plt.suptitle("PC Score Distributions — TEST SET", fontsize=9)
    plt.rcParams.update({'text.usetex': False})
    plt.tight_layout()
    out = os.path.join(output_dir, "eval_pc_scores_test.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved PC score distributions → {out}")

    # ── Worst fits — useful for diagnosing which cosmologies fail ─────────────
    n_worst  = min(6, N_test)
    idx_worst = np.argsort(r2_all)[:n_worst]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for ax, i in zip(axes, idx_worst):
        Om, s8, w = cosmo_test[i]
        ax.plot(x_arr, dv_test[i],  "k-",  lw=1.2, label="data")
        ax.plot(x_arr, y_preds[i],  "r--", lw=1.5,
                label=f"PhySO PCA  R²={r2_all[i]:.4f}")
        ax.set_title(f"cosmo {i}  Om={Om:.2f}  s8={s8:.2f}  w={w:.2f}",
                     fontsize=7)
        ax.legend(fontsize=6)
    plt.suptitle("PhySO PCA — WORST FITS on TEST SET", fontsize=10)
    plt.rcParams.update({'text.usetex': False})
    plt.tight_layout()
    out = os.path.join(output_dir, "eval_worst_fits_test.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved worst fits → {out}")

    # ── Save numerical results ────────────────────────────────────────────────
    import pandas as pd
    df_results = pd.DataFrame({
        "Omega_m": cosmo_test[:, 0],
        "sigma_8": cosmo_test[:, 1],
        "w":       cosmo_test[:, 2],
        "r2":      r2_all,
        **{f"PC{k+1}_score": all_scores[:, k] for k in range(n_comps)},
    })
    csv_out = os.path.join(output_dir, "eval_test_results.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"Saved numerical results → {csv_out}")

    return r2_all, y_preds, all_scores

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained PhySO PCA model on held-out test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir",  default=OUTPUT_DIR,
                        help="Directory containing training outputs.")
    parser.add_argument("--n_pcs",       default=N_COMPONENTS, type=int,
                        help="Number of PCA components (must match training).")
    parser.add_argument("--seed",        default=RANDOM_SEED, type=int,
                        help="Random seed for plot sampling.")
    args = parser.parse_args()

    OUTPUT_DIR   = args.output_dir
    N_COMPONENTS = args.n_pcs
    RANDOM_SEED  = args.seed

    # ── Load everything from disk ─────────────────────────────────────────────
    cosmo_test, dv_test, scaler, pca, expressions = load_artifacts(
        OUTPUT_DIR, N_COMPONENTS
    )

    # Force-disable TeX rendering — PhySO sets it True during import
    import matplotlib
    matplotlib.rcParams.update({'text.usetex': False})

    # ── Evaluate on test set ──────────────────────────────────────────────────
    r2_all, y_preds, scores = evaluate_test_set(
        cosmo_test, dv_test, expressions, pca, scaler, OUTPUT_DIR
    )

    print(f"\n✓ Done.  All evaluation outputs → {OUTPUT_DIR}/")
    print(f"\nOutputs:")
    print(f"  eval_reconstructions_test.png  — sample reconstructions")
    print(f"  eval_worst_fits_test.png       — worst R² cases")
    print(f"  eval_r2_distribution_test.png  — R² histogram + sorted curve")
    print(f"  eval_pc_scores_test.png        — predicted PC score distributions")
    print(f"  eval_test_results.csv          — R² + PC scores per cosmology")

    print(f"\nQuick prediction for a new cosmology:")
    print(f"  y, scores = predict_one([Om, s8, w], expressions, pca, scaler)")