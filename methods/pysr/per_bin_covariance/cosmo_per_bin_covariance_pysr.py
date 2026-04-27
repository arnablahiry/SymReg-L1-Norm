"""
cosmo_per_bin_covariance_pysr.py
================================
Learn the diagonal covariance elements of the L1-norm datavector with PySR:

    C_ii(Omega_m, sigma_8, w, i)

By default the script derives a per-cosmology covariance matrix from rebinned
datavectors and trains only on the diagonal first. An external covariance file
can still be supplied later if needed.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CSV_PATH = os.path.join(REPO_ROOT, "data", "csv", "l1norm_training_data_b160.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
COSMO_COLS = ["Omega_m", "sigma_8", "w"]
TEST_SIZE = 0.20
RANDOM_SEED = 42
N_SHOW = 6
DEFAULT_N_REBIN = 20

DEFAULT_COVARIANCE_CANDIDATES = [
    os.path.join(REPO_ROOT, "data", "diag_covariance_b160.npy"),
    os.path.join(REPO_ROOT, "data", "covariance_diag_b160.npy"),
    os.path.join(REPO_ROOT, "data", "covariance_b160.npy"),
]

PYSR_KWARGS = dict(
    niterations=300,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "square", "sqrt", "abs"],
    populations=24,
    population_size=40,
    maxsize=35,
    parsimony=1e-4,
    batching=True,
    batch_size=256,
    verbosity=1,
    random_state=RANDOM_SEED,
)
PER_INDEX_PYSR_KWARGS = dict(
    niterations=300,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "square", "sqrt", "abs"],
    populations=18,
    population_size=30,
    maxsize=20,
    parsimony=1e-4,
    batching=False,
    verbosity=0,
    random_state=RANDOM_SEED,
)


def get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def make_pysr_regressor(**kwargs):
    try:
        from pysr import PySRRegressor
    except ImportError as exc:
        raise ImportError("Install PySR first: pip install pysr") from exc
    return PySRRegressor(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit PySR to diagonal covariance elements C_ii.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv-path", default=CSV_PATH, help="Datavector CSV path.")
    parser.add_argument(
        "--covariance-path",
        default=None,
        help="Path to diagonal covariance targets. Supports .npy/.npz/.pkl/.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for saved artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Held-out cosmology fraction.",
    )
    parser.add_argument(
        "--n-rebin",
        type=int,
        default=DEFAULT_N_REBIN,
        help="Rebin the 160-bin datavector to this smaller length before building covariance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for the cosmology split and PySR.",
    )
    parser.add_argument(
        "--niterations",
        type=int,
        default=PYSR_KWARGS["niterations"],
        help="PySR iteration budget.",
    )
    parser.add_argument(
        "--strategy",
        choices=["per-index", "global"],
        default="per-index",
        help="Fit one PySR expression per diagonal bin or one global expression in (cosmology, i).",
    )
    parser.add_argument(
        "--demo-proxy",
        action="store_true",
        help="Use a placeholder diagonal target diag(bin_i^2) instead of derived covariance.",
    )
    return parser.parse_args()


def load_datavector_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    dv_cols = [c for c in df.columns if c not in COSMO_COLS]
    dv_cols = sorted(dv_cols, key=lambda name: int(name.split("_")[-1]))

    cosmo = df[COSMO_COLS].values.astype(np.float64)
    dv = df[dv_cols].values.astype(np.float64)
    i_arr = np.arange(len(dv_cols), dtype=np.float64)

    print(f"Loaded CSV from {path}")
    print(f"  Cosmologies : {cosmo.shape[0]}")
    print(f"  DV bins     : {dv.shape[1]}")
    return cosmo, dv, i_arr, dv_cols


def _find_existing_covariance_path(user_path: Optional[str]) -> Optional[str]:
    if user_path is not None:
        return user_path
    for candidate in DEFAULT_COVARIANCE_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_serialized_object(path: str):
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=True)
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if len(data.files) != 1:
            raise ValueError(
                f"{path} contains multiple arrays. Please save a single target array."
            )
        return data[data.files[0]]
    if path.endswith(".pkl"):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    if path.endswith(".csv"):
        return pd.read_csv(path).values
    raise ValueError(f"Unsupported covariance file format: {path}")


def load_diagonal_covariance_targets(
    covariance_path: str,
    n_cosmo: int,
    n_bins: int,
) -> np.ndarray:
    payload = _load_serialized_object(covariance_path)

    if isinstance(payload, dict):
        for key in ("diag_cov", "cov_diag", "diag", "covariance", "cov"):
            if key in payload:
                payload = payload[key]
                break

    arr = np.asarray(payload, dtype=np.float64)

    if arr.shape == (n_cosmo, n_bins):
        diag_cov = arr
    elif arr.shape == (n_cosmo, n_bins, n_bins):
        diag_cov = np.diagonal(arr, axis1=1, axis2=2)
    else:
        raise ValueError(
            "Covariance target has incompatible shape. "
            f"Expected ({n_cosmo}, {n_bins}) or ({n_cosmo}, {n_bins}, {n_bins}), "
            f"got {arr.shape}."
        )

    if np.any(diag_cov < 0):
        print("Warning: negative diagonal entries found; clipping to 0 for log-space fitting.")
        diag_cov = np.clip(diag_cov, 0.0, None)

    print(f"Loaded diagonal covariance targets from {covariance_path}")
    return diag_cov


def build_demo_proxy_targets(dv: np.ndarray) -> np.ndarray:
    proxy = np.square(np.clip(dv, 0.0, None))
    print("Using demo proxy targets: diag_cov_proxy = max(datavector, 0)^2")
    return proxy


def derive_rebinned_covariance(
    dv: np.ndarray,
    n_rebin: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cosmo, n_raw = dv.shape
    if n_raw % n_rebin != 0:
        raise ValueError(
            f"n_rebin={n_rebin} must evenly divide the datavector length {n_raw}."
        )

    chunk_size = n_raw // n_rebin
    if chunk_size < 2:
        raise ValueError(
            "Need at least 2 raw values per rebinned bin to form a covariance. "
            f"Received n_rebin={n_rebin}, which gives chunk_size={chunk_size}."
        )

    rebinned_samples = dv.reshape(n_cosmo, n_rebin, chunk_size)
    rebinned_means = rebinned_samples.mean(axis=2)
    centered = rebinned_samples - rebinned_means[:, :, None]
    covariance = centered @ np.transpose(centered, (0, 2, 1))
    covariance /= (chunk_size - 1)
    diag_cov = np.diagonal(covariance, axis1=1, axis2=2)

    print("\nDerived covariance from rebinned datavectors:")
    print(f"  Original DV length : {n_raw}")
    print(f"  Rebinned length    : {n_rebin}")
    print(f"  Chunk size         : {chunk_size}")
    print(f"  Covariance shape   : {covariance.shape}")
    return rebinned_means, covariance, diag_cov


def split_data(
    cosmo: np.ndarray,
    covariance: np.ndarray,
    diag_cov: np.ndarray,
    dv: np.ndarray,
    rebinned_means: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    indices = np.arange(cosmo.shape[0])
    idx_train, idx_test = train_test_split(indices, test_size=test_size, random_state=seed)

    print("\nTrain/test split at cosmology level:")
    print(f"  Train cosmologies: {len(idx_train)}")
    print(f"  Test cosmologies : {len(idx_test)}")

    return (
        idx_train,
        idx_test,
        cosmo[idx_train],
        cosmo[idx_test],
        covariance[idx_train],
        covariance[idx_test],
        diag_cov[idx_train],
        diag_cov[idx_test],
        dv[idx_train],
        dv[idx_test],
        rebinned_means[idx_train],
        rebinned_means[idx_test],
    )


def fit_scalers(cosmo_train: np.ndarray, i_arr: np.ndarray) -> Tuple[StandardScaler, StandardScaler]:
    cosmo_scaler = StandardScaler().fit(cosmo_train)
    index_scaler = StandardScaler().fit(i_arr.reshape(-1, 1))

    print("\nCosmology standardization:")
    for idx, name in enumerate(COSMO_COLS):
        print(
            f"  {name:8s}: mean={cosmo_scaler.mean_[idx]:.6f}  "
            f"std={cosmo_scaler.scale_[idx]:.6f}"
        )
    print(
        "  i_index : "
        f"mean={index_scaler.mean_[0]:.6f}  std={index_scaler.scale_[0]:.6f}"
    )

    return cosmo_scaler, index_scaler


def compute_log_shift(diag_cov_train: np.ndarray) -> float:
    positive = diag_cov_train[diag_cov_train > 0]
    if positive.size == 0:
        return 1e-16
    return min(1e-12, 0.1 * positive.min())


def compute_per_index_shifts(diag_cov_train: np.ndarray) -> np.ndarray:
    shifts = np.zeros(diag_cov_train.shape[1], dtype=np.float64)
    for idx in range(diag_cov_train.shape[1]):
        shifts[idx] = compute_log_shift(diag_cov_train[:, idx])
    return shifts


def build_design_matrix(
    cosmo_subset: np.ndarray,
    diag_cov_subset: np.ndarray,
    i_arr: np.ndarray,
    cosmo_scaler: StandardScaler,
    index_scaler: StandardScaler,
    target_shift: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cosmo, n_bins = diag_cov_subset.shape

    cosmo_std = cosmo_scaler.transform(cosmo_subset)
    i_std = index_scaler.transform(i_arr.reshape(-1, 1)).reshape(-1)

    cos_rep = np.repeat(cosmo_std, n_bins, axis=0)
    i_rep = np.tile(i_std, n_cosmo).reshape(-1, 1)
    X_flat = np.hstack([cos_rep, i_rep])

    y_flat = np.log10(diag_cov_subset.reshape(-1) + target_shift)
    return X_flat, y_flat


def invert_target_transform(y_log: np.ndarray, target_shift: float) -> np.ndarray:
    return np.clip((10.0 ** y_log) - target_shift, 0.0, None)


def invert_target_transform_per_index(y_log: np.ndarray, target_shift: np.ndarray) -> np.ndarray:
    return np.clip((10.0 ** y_log) - target_shift, 0.0, None)


def predict_diagonal(
    Omega_m: float,
    sigma_8: float,
    w: float,
    model,
    cosmo_scaler: StandardScaler,
    index_scaler: StandardScaler,
    n_bins: int,
    target_shift: float,
) -> np.ndarray:
    i_arr = np.arange(n_bins, dtype=np.float64)
    cosmo_std = cosmo_scaler.transform([[Omega_m, sigma_8, w]])[0]
    i_std = index_scaler.transform(i_arr.reshape(-1, 1)).reshape(-1)
    X = np.column_stack(
        [
            np.full(n_bins, cosmo_std[0]),
            np.full(n_bins, cosmo_std[1]),
            np.full(n_bins, cosmo_std[2]),
            i_std,
        ]
    )
    pred_log = model.predict(X)
    return invert_target_transform(pred_log, target_shift)


def r2_score_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def train_per_index_models(
    cosmo_train: np.ndarray,
    diag_cov_train: np.ndarray,
    cosmo_scaler: StandardScaler,
    shifts: np.ndarray,
    niterations: int,
    seed: int,
) -> list[Dict[str, object]]:
    X_train = cosmo_scaler.transform(cosmo_train)
    models: list[Dict[str, object]] = []

    print("\nRunning PySR per diagonal bin...")
    for i_idx in range(diag_cov_train.shape[1]):
        y = diag_cov_train[:, i_idx]
        y_log = np.log10(y + shifts[i_idx])

        if np.std(y_log) < 1e-10:
            models.append(
                {
                    "kind": "constant",
                    "value_log": float(np.mean(y_log)),
                    "equation": f"{float(np.mean(y_log)):.12g}",
                }
            )
            print(f"  bin {i_idx:02d}: constant target")
            continue

        kwargs = dict(PER_INDEX_PYSR_KWARGS)
        kwargs["niterations"] = niterations
        kwargs["random_state"] = seed + i_idx
        model = make_pysr_regressor(**kwargs)
        model.fit(
            X_train,
            y_log,
            variable_names=["Omega_m_std", "sigma_8_std", "w_std"],
        )
        eq = str(model.sympy())
        models.append({"kind": "pysr", "model": model, "equation": eq})
        print(f"  bin {i_idx:02d}: {eq}")

    return models


def predict_per_index_models(
    cosmo_subset: np.ndarray,
    models: list[Dict[str, object]],
    cosmo_scaler: StandardScaler,
    shifts: np.ndarray,
) -> np.ndarray:
    X = cosmo_scaler.transform(cosmo_subset)
    preds_log = np.zeros((cosmo_subset.shape[0], len(models)), dtype=np.float64)

    for idx, info in enumerate(models):
        if info["kind"] == "constant":
            preds_log[:, idx] = float(info["value_log"])
        else:
            preds_log[:, idx] = info["model"].predict(X)

    return invert_target_transform_per_index(preds_log, shifts.reshape(1, -1))


def evaluate_per_index_models(
    models: list[Dict[str, object]],
    cosmo_test: np.ndarray,
    diag_cov_test: np.ndarray,
    cosmo_scaler: StandardScaler,
    shifts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    preds = predict_per_index_models(cosmo_test, models, cosmo_scaler, shifts)
    r2_all = np.array(
        [r2_score_1d(diag_cov_test[idx], preds[idx]) for idx in range(cosmo_test.shape[0])],
        dtype=np.float64,
    )
    return preds, r2_all


def evaluate_model(
    model,
    cosmo_test: np.ndarray,
    diag_cov_test: np.ndarray,
    cosmo_scaler: StandardScaler,
    index_scaler: StandardScaler,
    target_shift: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = diag_cov_test.shape[1]
    preds = np.zeros_like(diag_cov_test)
    r2_all = np.zeros(cosmo_test.shape[0], dtype=np.float64)

    for idx in range(cosmo_test.shape[0]):
        preds[idx] = predict_diagonal(
            Omega_m=cosmo_test[idx, 0],
            sigma_8=cosmo_test[idx, 1],
            w=cosmo_test[idx, 2],
            model=model,
            cosmo_scaler=cosmo_scaler,
            index_scaler=index_scaler,
            n_bins=n_bins,
            target_shift=target_shift,
        )
        r2_all[idx] = r2_score_1d(diag_cov_test[idx], preds[idx])
    return preds, r2_all


def plot_reconstructions(
    output_dir: str,
    i_arr: np.ndarray,
    cosmo_test: np.ndarray,
    diag_cov_test: np.ndarray,
    preds: np.ndarray,
    r2_all: np.ndarray,
) -> None:
    plt = get_pyplot()
    rng = np.random.default_rng(RANDOM_SEED)
    idx_show = rng.choice(cosmo_test.shape[0], size=min(N_SHOW, cosmo_test.shape[0]), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    for ax, idx in zip(axes, idx_show):
        Om, s8, w = cosmo_test[idx]
        ax.plot(i_arr, diag_cov_test[idx], "k-", lw=1.25, label="target")
        ax.plot(i_arr, preds[idx], "r--", lw=1.5, label=f"PySR  R²={r2_all[idx]:.3f}")
        ax.set_title(f"cosmo {idx}  Om={Om:.3f}  s8={s8:.3f}  w={w:.3f}", fontsize=8)
        ax.set_xlabel("i")
        ax.set_ylabel("C_ii")
        ax.legend(fontsize=7)

    plt.suptitle("Diagonal covariance on held-out cosmologies", fontsize=10)
    plt.tight_layout()
    out = os.path.join(output_dir, "diag_reconstructions_test.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved reconstruction plot -> {out}")


def plot_r2_distribution(output_dir: str, r2_all: np.ndarray) -> None:
    plt = get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(r2_all, bins=20, edgecolor="k", color="steelblue")
    axes[0].set_xlabel("R²")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"R² on held-out cosmologies (median={np.median(r2_all):.4f})")

    axes[1].plot(np.sort(r2_all), "k-", lw=1.25)
    axes[1].set_xlabel("Cosmology (sorted)")
    axes[1].set_ylabel("R²")
    axes[1].set_title("Sorted R²")

    plt.tight_layout()
    out = os.path.join(output_dir, "r2_distribution_test.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved R² distribution -> {out}")


def plot_pareto(model, output_dir: str) -> None:
    plt = get_pyplot()
    try:
        eqs = model.equations_
        if eqs is None or "complexity" not in eqs.columns:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(eqs["complexity"], eqs["loss"], s=30)
        ax.set_xlabel("Complexity")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.set_title("PySR Pareto front")
        plt.tight_layout()
        out = os.path.join(output_dir, "pareto_front.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved Pareto front -> {out}")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not save Pareto front: {exc}")


def save_per_index_expressions(
    output_dir: str,
    models: list[Dict[str, object]],
    shifts: np.ndarray,
    target_source: str,
    n_rebin: int,
) -> None:
    out = os.path.join(output_dir, "best_expressions_per_index.txt")
    with open(out, "w") as handle:
        handle.write(f"Target source : {target_source}\n")
        handle.write("Strategy      : per-index\n")
        handle.write(f"n_rebin       : {n_rebin}\n\n")
        for idx, info in enumerate(models):
            handle.write(f"bin {idx:02d}\n")
            handle.write(f"  shift      : {shifts[idx]:.6e}\n")
            handle.write(f"  equation   : {info['equation']}\n\n")
    print(f"Saved expressions summary -> {out}")


def save_artifacts(
    output_dir: str,
    idx_test: np.ndarray,
    cosmo_test: np.ndarray,
    covariance_test: np.ndarray,
    diag_cov_test: np.ndarray,
    dv_test: np.ndarray,
    rebinned_means_test: np.ndarray,
    cosmo_scaler: StandardScaler,
    index_scaler: StandardScaler,
    target_shift: float,
    model,
    target_source: str,
    n_rebin: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "test_cosmologies.pkl"), "wb") as handle:
        pickle.dump(
            {
                "indices": idx_test,
                "cosmo": cosmo_test,
                "covariance": covariance_test,
                "diag_cov": diag_cov_test,
                "datavector": dv_test,
                "rebinned_datavector": rebinned_means_test,
                "target_source": target_source,
            },
            handle,
        )

    np.save(os.path.join(output_dir, "rebinned_covariance_test.npy"), covariance_test)

    with open(os.path.join(output_dir, "cosmo_scaler.pkl"), "wb") as handle:
        pickle.dump(cosmo_scaler, handle)

    with open(os.path.join(output_dir, "index_scaler.pkl"), "wb") as handle:
        pickle.dump(index_scaler, handle)

    with open(os.path.join(output_dir, "target_transform.pkl"), "wb") as handle:
        pickle.dump({"log10_shift": target_shift}, handle)

    with open(os.path.join(output_dir, "best_expression.txt"), "w") as handle:
        handle.write(f"Target source : {target_source}\n")
        handle.write("Fitted target : log10(C_ii + shift)\n")
        handle.write(f"shift         : {target_shift:.6e}\n\n")
        handle.write(f"n_rebin       : {n_rebin}\n")
        handle.write("Variables     : Omega_m_std, sigma_8_std, w_std, i_std\n")
        handle.write(f"Expression    : {model.sympy()}\n\n")
        handle.write("Usage:\n")
        handle.write("  1. Standardize cosmology with cosmo_scaler.pkl\n")
        handle.write("  2. Standardize bin index i with index_scaler.pkl\n")
        handle.write("  3. Evaluate the PySR expression in log10-space\n")
        handle.write("  4. Convert back with C_ii = 10**pred - shift\n")


def save_per_index_artifacts(
    output_dir: str,
    idx_test: np.ndarray,
    cosmo_test: np.ndarray,
    covariance_test: np.ndarray,
    diag_cov_test: np.ndarray,
    dv_test: np.ndarray,
    rebinned_means_test: np.ndarray,
    cosmo_scaler: StandardScaler,
    shifts: np.ndarray,
    models: list[Dict[str, object]],
    target_source: str,
    n_rebin: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_cosmologies.pkl"), "wb") as handle:
        pickle.dump(
            {
                "indices": idx_test,
                "cosmo": cosmo_test,
                "covariance": covariance_test,
                "diag_cov": diag_cov_test,
                "datavector": dv_test,
                "rebinned_datavector": rebinned_means_test,
                "target_source": target_source,
                "strategy": "per-index",
            },
            handle,
        )

    np.save(os.path.join(output_dir, "rebinned_covariance_test.npy"), covariance_test)

    with open(os.path.join(output_dir, "cosmo_scaler.pkl"), "wb") as handle:
        pickle.dump(cosmo_scaler, handle)

    with open(os.path.join(output_dir, "target_transform.pkl"), "wb") as handle:
        pickle.dump({"per_index_log10_shift": shifts}, handle)

    save_per_index_expressions(
        output_dir=output_dir,
        models=models,
        shifts=shifts,
        target_source=target_source,
        n_rebin=n_rebin,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cosmo, dv, i_arr, _ = load_datavector_csv(args.csv_path)
    n_cosmo, n_bins_raw = dv.shape

    covariance_path = _find_existing_covariance_path(args.covariance_path)
    if covariance_path is not None:
        rebinned_means, covariance, derived_diag_cov = derive_rebinned_covariance(dv, args.n_rebin)
        diag_cov = load_diagonal_covariance_targets(
            covariance_path,
            n_cosmo,
            args.n_rebin,
        )
        if diag_cov.shape != derived_diag_cov.shape:
            raise ValueError(
                f"External diagonal targets have shape {diag_cov.shape}, "
                f"but n_rebin={args.n_rebin} implies diagonal shape {derived_diag_cov.shape}."
            )
        target_source = covariance_path
    elif args.demo_proxy:
        rebinned_means, covariance, _ = derive_rebinned_covariance(dv, args.n_rebin)
        i_arr = np.arange(args.n_rebin, dtype=np.float64)
        if dv.shape[1] != args.n_rebin:
            if dv.shape[1] % args.n_rebin != 0:
                raise ValueError(
                    f"n_rebin={args.n_rebin} must evenly divide datavector length {n_bins_raw}."
                )
            proxy_source = rebinned_means
        else:
            proxy_source = dv
        diag_cov = build_demo_proxy_targets(proxy_source)
        target_source = "demo_proxy_from_rebinned_datavector_squared"
    else:
        rebinned_means, covariance, diag_cov = derive_rebinned_covariance(dv, args.n_rebin)
        target_source = f"derived_from_rebinned_datavector_covariance_n{args.n_rebin}"

    i_arr = np.arange(args.n_rebin, dtype=np.float64)

    (
        idx_train,
        idx_test,
        cosmo_train,
        cosmo_test,
        covariance_train,
        covariance_test,
        diag_cov_train,
        diag_cov_test,
        dv_train,
        dv_test,
        rebinned_means_train,
        rebinned_means_test,
    ) = split_data(
        cosmo=cosmo,
        covariance=covariance,
        diag_cov=diag_cov,
        dv=dv,
        rebinned_means=rebinned_means,
        test_size=args.test_size,
        seed=args.seed,
    )

    cosmo_scaler, index_scaler = fit_scalers(cosmo_train, i_arr)
    if args.strategy == "global":
        target_shift = compute_log_shift(diag_cov_train)

        X_train, y_train = build_design_matrix(
            cosmo_subset=cosmo_train,
            diag_cov_subset=diag_cov_train,
            i_arr=i_arr,
            cosmo_scaler=cosmo_scaler,
            index_scaler=index_scaler,
            target_shift=target_shift,
        )

        print("\nTraining design matrix:")
        print(f"  X shape     : {X_train.shape}")
        print(f"  y_log range : [{y_train.min():.6f}, {y_train.max():.6f}]")

        pysr_kwargs = dict(PYSR_KWARGS)
        pysr_kwargs["niterations"] = args.niterations
        pysr_kwargs["random_state"] = args.seed

        print("\nRunning PySR for diagonal covariance...")
        model = make_pysr_regressor(**pysr_kwargs)
        model.fit(
            X_train,
            y_train,
            variable_names=["Omega_m_std", "sigma_8_std", "w_std", "i_std"],
        )

        print("\nBest expression:")
        print(f"  {model.sympy()}")

        preds, r2_all = evaluate_model(
            model=model,
            cosmo_test=cosmo_test,
            diag_cov_test=diag_cov_test,
            cosmo_scaler=cosmo_scaler,
            index_scaler=index_scaler,
            target_shift=target_shift,
        )
    else:
        target_shift = compute_per_index_shifts(diag_cov_train)
        print("\nPer-index target log shifts:")
        print("  " + ", ".join(f"{value:.2e}" for value in target_shift))

        models = train_per_index_models(
            cosmo_train=cosmo_train,
            diag_cov_train=diag_cov_train,
            cosmo_scaler=cosmo_scaler,
            shifts=target_shift,
            niterations=args.niterations,
            seed=args.seed,
        )
        preds, r2_all = evaluate_per_index_models(
            models=models,
            cosmo_test=cosmo_test,
            diag_cov_test=diag_cov_test,
            cosmo_scaler=cosmo_scaler,
            shifts=target_shift,
        )

    print("\nHeld-out test performance:")
    print(f"  Median R² : {np.median(r2_all):.4f}")
    print(f"  Min R²    : {np.min(r2_all):.4f}")
    print(f"  Max R²    : {np.max(r2_all):.4f}")

    if args.strategy == "global":
        save_artifacts(
            output_dir=args.output_dir,
            idx_test=idx_test,
            cosmo_test=cosmo_test,
            covariance_test=covariance_test,
            diag_cov_test=diag_cov_test,
            dv_test=dv_test,
            rebinned_means_test=rebinned_means_test,
            cosmo_scaler=cosmo_scaler,
            index_scaler=index_scaler,
            target_shift=target_shift,
            model=model,
            target_source=target_source,
            n_rebin=args.n_rebin,
        )
    else:
        save_per_index_artifacts(
            output_dir=args.output_dir,
            idx_test=idx_test,
            cosmo_test=cosmo_test,
            covariance_test=covariance_test,
            diag_cov_test=diag_cov_test,
            dv_test=dv_test,
            rebinned_means_test=rebinned_means_test,
            cosmo_scaler=cosmo_scaler,
            shifts=target_shift,
            models=models,
            target_source=target_source,
            n_rebin=args.n_rebin,
        )
    plot_reconstructions(args.output_dir, i_arr, cosmo_test, diag_cov_test, preds, r2_all)
    plot_r2_distribution(args.output_dir, r2_all)
    if args.strategy == "global":
        plot_pareto(model, args.output_dir)

    print(f"\nDone. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
