"""
nn_emulator.py — Neural Network Emulator for Wavelet L1-Norm
=============================================================
Maps 3 cosmological parameters (Omega_m, sigma_8, w) → full 160-bin datavector.

Train/val/test split:
  * 80/20 cosmology-level split (random_state=42) — TEST SET IDENTICAL TO
    cosmo_symreg_direct.py for fair comparison.
  * The 80% "pool" is further split 85/15 into train / validation.
  * Early stopping / checkpoint selection is on VALIDATION only.
  * The test set is touched exactly ONCE at the end to report the final R².

Device: Apple Metal (MPS) GPU if available.

The final figure overlays:
  - Truth
  - NN emulator prediction (this file)
  - PySR Direct-SR prediction (from best_expression.txt)
  - PhySO PCA-SR prediction (mean + sum_k f_k * PC_k)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import re

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# =============================================================================
# CONFIG  — test split must match cosmo_symreg_direct.py exactly
# =============================================================================

REPO_ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH          = os.path.join(REPO_ROOT, "data/csv/l1norm_training_data_b160.csv")
COSMO_COLS        = ["Omega_m", "sigma_8", "w"]
TEST_SIZE         = 0.20
VAL_FRAC_OF_TRAIN = 0.15
RANDOM_SEED       = 42
OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PySR Direct SR artifacts
PYSR_DIR_BEST   = os.path.join(REPO_ROOT, "methods/pysr/direct/outputs_50nodes_new/best_expression.txt")
PYSR_DIR_SCALER = os.path.join(REPO_ROOT, "methods/pysr/direct/outputs_50nodes_new/cosmo_scaler.pkl")

# PhySO PCA SR artifacts
PHYSO_DIR = os.path.join(REPO_ROOT, "methods/physo/pca/outputs")
PHYSO_N_COMPONENTS = 3        # matches expressions_summary.txt

# Training hyperparameters
BATCH_SIZE  = 16
LR          = 3e-4
EPOCHS      = 2000
PATIENCE    = 300                       # early-stop patience on val R²
HIDDEN      = [128, 256, 256, 128]


# =============================================================================
# DEVICE
# =============================================================================

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple MPS (GPU)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


# =============================================================================
# DATA — train/val/test split with test identical to PySR direct
# =============================================================================

def load_and_split():
    df = pd.read_csv(CSV_PATH)
    bin_cols = [c for c in df.columns if c.startswith("bin_")]
    cosmo = df[COSMO_COLS].values.astype(np.float32)
    dv    = df[bin_cols].values.astype(np.float32)

    n_cos = cosmo.shape[0]
    idx_pool, idx_test = train_test_split(
        np.arange(n_cos), test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    # Further split the pool into train / val. Different seed so val
    # cosmologies are not correlated with the test split.
    idx_train, idx_val = train_test_split(
        idx_pool, test_size=VAL_FRAC_OF_TRAIN, random_state=RANDOM_SEED + 1
    )

    print(f"\nSplit (random_state={RANDOM_SEED}):")
    print(f"  Train: {len(idx_train)} cosmologies")
    print(f"  Val  : {len(idx_val)} cosmologies  (for early stopping only)")
    print(f"  Test : {len(idx_test)} cosmologies  (identical to PySR direct)")

    cosmo_train = cosmo[idx_train]; dv_train = dv[idx_train]
    cosmo_val   = cosmo[idx_val];   dv_val   = dv[idx_val]
    cosmo_test  = cosmo[idx_test];  dv_test  = dv[idx_test]

    # Scale inputs and outputs — fit on TRAIN only (no val, no test)
    x_scaler = StandardScaler().fit(cosmo_train)
    y_scaler = MinMaxScaler().fit(dv_train)

    Xtr = x_scaler.transform(cosmo_train); Ytr = y_scaler.transform(dv_train)
    Xva = x_scaler.transform(cosmo_val);   Yva = y_scaler.transform(dv_val)
    Xte = x_scaler.transform(cosmo_test);  Yte = y_scaler.transform(dv_test)

    with open(os.path.join(OUTPUT_DIR, "nn_x_scaler.pkl"), "wb") as f:
        pickle.dump(x_scaler, f)
    with open(os.path.join(OUTPUT_DIR, "nn_y_scaler.pkl"), "wb") as f:
        pickle.dump(y_scaler, f)
    with open(os.path.join(OUTPUT_DIR, "split_indices.pkl"), "wb") as f:
        pickle.dump({"train": idx_train, "val": idx_val, "test": idx_test}, f)

    raw = dict(
        cosmo_train=cosmo_train, dv_train=dv_train,
        cosmo_val=cosmo_val,     dv_val=dv_val,
        cosmo_test=cosmo_test,   dv_test=dv_test,
    )
    scaled = dict(Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte)
    return raw, scaled, x_scaler, y_scaler


# =============================================================================
# MODEL
# =============================================================================

class CosmoEmulator(nn.Module):
    def __init__(self, n_in, n_out, hidden):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optim.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_r2(model, X, Y, device):
    model.eval()
    pred = model(X.to(device)).cpu().numpy()
    return r2_score(Y.numpy(), pred), pred


# =============================================================================
# PYSR DIRECT PREDICTION
# =============================================================================

def _build_pysr_predictor():
    """Parse best_expression.txt and return a callable f(cosmo_raw[3,]) -> (160,)."""
    with open(PYSR_DIR_BEST) as f:
        txt = f.read()

    # Expression line
    expr = re.search(r"Expression\s*:\s*(.+)", txt).group(1).strip()
    mean_peak = float(re.search(r"x\s*-\s*([\-\d\.]+)", txt).group(1))

    with open(PYSR_DIR_SCALER, "rb") as f:
        scaler = pickle.load(f)

    x_arr = np.arange(160, dtype=np.float64)
    x_c = x_arr - mean_peak
    x_c_safe = np.sign(x_c) * (np.abs(x_c) + 0.0005)

    # Compile expression once
    code = compile(expr.replace("^", "**"), "<pysr>", "eval")

    def predict(cosmo_raw):
        std = scaler.transform([cosmo_raw])[0]
        local = {
            "Omega_m_std": std[0], "sigma_8_std": std[1], "w_std": std[2],
            "x_c": x_c_safe,
            "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos,
            "sqrt": np.sqrt, "square": np.square, "Abs": np.abs, "abs": np.abs,
        }
        return np.asarray(eval(code, {"__builtins__": {}}, local), dtype=np.float64)

    return predict


# =============================================================================
# PHYSO PCA PREDICTION
# =============================================================================

def _build_physo_predictor():
    """Return callable f(cosmo_raw[3,]) -> (160,) using PhySO PCA reconstruction."""
    with open(os.path.join(PHYSO_DIR, "cosmo_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(PHYSO_DIR, "pca_basis.pkl"), "rb") as f:
        pca = pickle.load(f)
    expressions = []
    for k in range(PHYSO_N_COMPONENTS):
        with open(os.path.join(PHYSO_DIR, f"PC{k+1}", "expression.pkl"), "rb") as f:
            expressions.append(pickle.load(f))

    def predict(cosmo_raw):
        cosmo_std = scaler.transform([cosmo_raw])[0].astype(np.float64)
        X = cosmo_std.reshape(-1, 1)
        X_t = torch.tensor(X, dtype=torch.float64)
        scores = np.array([
            expr.execute(X_t).detach().numpy().flatten()[0]
            for expr in expressions
        ])
        return pca.mean_ + scores @ pca.components_[:PHYSO_N_COMPONENTS]

    return predict


# =============================================================================
# COMPARISON PLOT (user-spec)
# =============================================================================

def plot_comparison_test(cosmo_test, dv_test,
                         nn_preds, pysr_preds,
                         physo_pca_preds=None, physo_2g1g_preds=None,
                         n_show=6, out_path=None):
    """
    2×3 grid, sharex + sharey, no suptitle, no external legend.
    Overlays: truth, NN, PySR-Direct, (optional) PhySO-PCA, (optional) PhySO-2G-1G.
    Per-method R² (with colored line marker) shown in top-right of each subplot.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(dv_test), size=min(n_show, len(dv_test)), replace=False)
    x_arr = np.arange(dv_test.shape[1])

    fig, axes = plt.subplots(
        2, 3, figsize=(13, 6.6),
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.08, "wspace": 0.06},
    )
    axes = axes.ravel()

    style = {
        "Truth":       dict(color="black",       lw=1.6, ls="-"),
        "NN":          dict(color="tab:green",   lw=1.3, ls="--"),
        "PySR Direct": dict(color="xkcd:red",    lw=1.3, ls="-."),
        "PhySO PCA":   dict(color="tab:blue",    lw=1.3, ls=":"),
        "PhySO 2G−1G": dict(color="tab:orange",  lw=1.3, ls=(0, (3, 1, 1, 1))),
    }

    for ax, i in zip(axes, idx):
        ax.plot(x_arr, dv_test[i],     label="Truth",       **style["Truth"])
        ax.plot(x_arr, nn_preds[i],    label="NN",          **style["NN"])
        ax.plot(x_arr, pysr_preds[i],  label="PySR Direct", **style["PySR Direct"])
        if physo_pca_preds is not None:
            ax.plot(x_arr, physo_pca_preds[i],  label="PhySO PCA",   **style["PhySO PCA"])
        if physo_2g1g_preds is not None:
            ax.plot(x_arr, physo_2g1g_preds[i], label="PhySO 2G−1G", **style["PhySO 2G−1G"])

        # Cosmological params in top-left
        om, s8, w = cosmo_test[i]
        ax.text(0.03, 0.97,
                rf"$\Omega_m = {om:.3f}$" "\n"
                rf"$\sigma_8 = {s8:.3f}$" "\n"
                rf"$w = {w:.3f}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="lightgray", alpha=0.85))

        # Color-coded method legend + R² in top-right (replaces external legend)
        lines = [
            ("NN",          nn_preds[i]),
            ("PySR Direct", pysr_preds[i]),
        ]
        if physo_pca_preds is not None:
            lines.append(("PhySO PCA", physo_pca_preds[i]))
        if physo_2g1g_preds is not None:
            lines.append(("PhySO 2G−1G", physo_2g1g_preds[i]))

        n_lines = len(lines)
        y0, dy = 0.97, 0.065
        for k, (name, pred) in enumerate(lines):
            r2 = r2_score(dv_test[i], pred)
            ax.text(0.97, y0 - k * dy,
                    f"━  {name}:  " + rf"$R^2 = {r2:.4f}$",
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=8, color=style[name]["color"],
                    family="monospace")

    # Shared axis labels
    for ax in axes[-3:]:
        ax.set_xlabel(r"$\Vert C_{i,f}\Vert$")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel(r"$\ell_1$ norm")

    fig.subplots_adjust(top=0.98, left=0.07, right=0.99, bottom=0.09)
    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, "comparison_test.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison plot → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = get_device()
    raw, sc, x_scaler, y_scaler = load_and_split()

    # Tensors
    Xtr = torch.tensor(sc["Xtr"], dtype=torch.float32)
    Ytr = torch.tensor(sc["Ytr"], dtype=torch.float32)
    Xva = torch.tensor(sc["Xva"], dtype=torch.float32)
    Yva = torch.tensor(sc["Yva"], dtype=torch.float32)
    Xte = torch.tensor(sc["Xte"], dtype=torch.float32)
    Yte = torch.tensor(sc["Yte"], dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True)

    model = CosmoEmulator(Xtr.shape[1], Ytr.shape[1], HIDDEN).to(device)
    print(f"\nModel: {Xtr.shape[1]} → {HIDDEN} → {Ytr.shape[1]}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optim_ = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    # Training — early stop on VALIDATION (never touch test here)
    best_val_r2, best_state, epochs_no_improve = -np.inf, None, 0
    history = {"epoch": [], "train_loss": [], "val_r2": []}
    print(f"\nTraining  (epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, patience={PATIENCE})")

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, loader, optim_, loss_fn, device)
        sched.step()
        val_r2, _ = eval_r2(model, Xva, Yva, device)

        if val_r2 > best_val_r2 + 1e-5:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            history["epoch"].append(epoch)
            history["train_loss"].append(tr_loss)
            history["val_r2"].append(val_r2)
            print(f"  Epoch {epoch:5d} | train MSE={tr_loss:.6f} | "
                  f"val R²={val_r2:.4f} | best val R²={best_val_r2:.4f}")

        if epochs_no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}  (no val improvement for {PATIENCE} epochs)")
            break

    # Load best checkpoint (selected on val) and do ONE test evaluation
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUTPUT_DIR, "nn_best_model.pt"))

    test_r2_scaled, nn_pred_norm = eval_r2(model, Xte, Yte, device)
    nn_pred = y_scaler.inverse_transform(nn_pred_norm)

    r2_per_cosmo = np.array([
        r2_score(raw["dv_test"][i], nn_pred[i]) for i in range(len(nn_pred))
    ])
    print("\n" + "=" * 55)
    print(f"  Val  R² (best, used for checkpoint) : {best_val_r2:.4f}")
    print(f"  Test R² (scaled, one-shot)          : {test_r2_scaled:.4f}")
    print(f"  Test R² per-cosmology  median       : {np.median(r2_per_cosmo):.4f}")
    print(f"  Test R² per-cosmology  min          : {r2_per_cosmo.min():.4f}")
    print(f"  Test R² per-cosmology  max          : {r2_per_cosmo.max():.4f}")
    print("=" * 55)

    # Save NN predictions + truth for downstream comparisons
    np.save(os.path.join(OUTPUT_DIR, "nn_test_predictions.npy"), nn_pred)
    np.save(os.path.join(OUTPUT_DIR, "nn_test_truth.npy"),       raw["dv_test"])
    np.save(os.path.join(OUTPUT_DIR, "nn_test_cosmo.npy"),       raw["cosmo_test"])

    # Build PySR-Direct predictor and evaluate on the same test cosmologies
    print("\nReconstructing test set with PySR-Direct ...")
    pysr_pred_fn = _build_pysr_predictor()
    pysr_preds   = np.stack([pysr_pred_fn(c) for c in raw["cosmo_test"]])
    pysr_r2 = np.array([r2_score(raw["dv_test"][i], pysr_preds[i])
                        for i in range(len(pysr_preds))])
    print(f"  PySR-Direct  test R²  (median): {np.median(pysr_r2):.4f}   min: {pysr_r2.min():.4f}")
    np.save(os.path.join(OUTPUT_DIR, "pysr_direct_test_predictions.npy"), pysr_preds)

    # PhySO-PCA is optional — skip if PhySO isn't importable in this env
    physo_preds = None
    try:
        print("\nReconstructing test set with PhySO-PCA ...")
        physo_pred_fn = _build_physo_predictor()
        physo_preds = np.stack([physo_pred_fn(c) for c in raw["cosmo_test"]])
        physo_r2 = np.array([r2_score(raw["dv_test"][i], physo_preds[i])
                             for i in range(len(physo_preds))])
        print(f"  PhySO-PCA    test R²  (median): {np.median(physo_r2):.4f}   min: {physo_r2.min():.4f}")
        np.save(os.path.join(OUTPUT_DIR, "physo_pca_test_predictions.npy"), physo_preds)
    except Exception as e:
        print(f"  (skipped PhySO overlay — {type(e).__name__}: {e})")

    # Final comparison figure
    plot_comparison_test(
        raw["cosmo_test"], raw["dv_test"],
        nn_preds=nn_pred, pysr_preds=pysr_preds, physo_preds=physo_preds,
    )

    # Training curve (diagnostic)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(history["epoch"], history["train_loss"]); a1.set_yscale("log")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Train MSE"); a1.set_title("Training loss")
    a2.plot(history["epoch"], history["val_r2"])
    a2.set_xlabel("Epoch"); a2.set_ylabel("Val R²"); a2.set_title("Validation R²")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "nn_training_curve.png"), dpi=150)
    plt.close()

    print("\nDone. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
