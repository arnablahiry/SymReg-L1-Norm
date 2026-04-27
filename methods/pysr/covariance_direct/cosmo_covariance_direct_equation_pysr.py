"""
cosmo_covariance_direct_equation_pysr.py
========================================
Build a single direct functional form for diagonal covariance:

    C_ii = f(Omega_m, sigma_8, w, i)

The direct model is distilled from the successful per-bin idea:

    log10(C_ii + shift)
      = a(i_std) * Omega_m_std
      + b(i_std) * sigma_8_std
      + c(i_std) * w_std
      + d(i_std)

The coefficient curves a(i), b(i), c(i), d(i) are represented with a smooth
RBF basis in i, giving one explicit closed-form function of cosmology and bin
index.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PER_BIN_SCRIPT = os.path.join(
    REPO_ROOT,
    "methods", "pysr", "per_bin_covariance",
    "cosmo_per_bin_covariance_pysr.py",
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DEFAULT_N_CENTERS = 16
DEFAULT_WIDTH = 0.30


def load_per_bin_module():
    spec = importlib.util.spec_from_file_location("per_bin_covariance_local", PER_BIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {PER_BIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_per_bin_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one direct equation for diagonal covariance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv-path", default=helper.CSV_PATH, help="Datavector CSV path.")
    parser.add_argument(
        "--covariance-path",
        default=None,
        help="Optional external diagonal covariance targets. Supports .npy/.npz/.pkl/.csv.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for saved artifacts.")
    parser.add_argument("--test-size", type=float, default=helper.TEST_SIZE, help="Held-out cosmology fraction.")
    parser.add_argument("--n-rebin", type=int, default=helper.DEFAULT_N_REBIN, help="Rebinned diagonal length.")
    parser.add_argument("--seed", type=int, default=helper.RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--n-centers",
        type=int,
        default=DEFAULT_N_CENTERS,
        help="Number of Gaussian basis centers for the i-dependent coefficient curves.",
    )
    parser.add_argument(
        "--rbf-width",
        type=float,
        default=DEFAULT_WIDTH,
        help="Shared Gaussian width in standardized i-space.",
    )
    return parser.parse_args()


def load_targets(args: argparse.Namespace):
    cosmo, dv, _, _ = helper.load_datavector_csv(args.csv_path)
    rebinned_means, covariance, derived_diag_cov = helper.derive_rebinned_covariance(dv, args.n_rebin)
    covariance_path = helper._find_existing_covariance_path(args.covariance_path)
    if covariance_path is not None:
        diag_cov = helper.load_diagonal_covariance_targets(
            covariance_path,
            cosmo.shape[0],
            args.n_rebin,
        )
        target_source = covariance_path
    else:
        diag_cov = derived_diag_cov
        target_source = f"derived_from_rebinned_datavector_covariance_n{args.n_rebin}"
    return cosmo, dv, rebinned_means, covariance, diag_cov, target_source


def build_rbf_design(i_std: np.ndarray, n_centers: int, width: float) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.linspace(i_std.min(), i_std.max(), n_centers)
    features = [np.ones_like(i_std), i_std]
    for center in centers:
        features.append(np.exp(-0.5 * ((i_std - center) / width) ** 2))
    return np.column_stack(features), centers


def fit_per_bin_linear_teachers(
    X_cosmo_train: np.ndarray,
    diag_cov_train: np.ndarray,
    shift: float,
) -> np.ndarray:
    coeffs = np.zeros((diag_cov_train.shape[1], 4), dtype=np.float64)
    for idx in range(diag_cov_train.shape[1]):
        y = np.log10(diag_cov_train[:, idx] + shift)
        model = LinearRegression().fit(X_cosmo_train, y)
        coeffs[idx, :3] = model.coef_
        coeffs[idx, 3] = model.intercept_
    return coeffs


def smooth_teacher_coefficients(coeffs_per_bin: np.ndarray, Phi_i: np.ndarray) -> np.ndarray:
    weights = np.linalg.lstsq(Phi_i, coeffs_per_bin, rcond=None)[0]
    coeff_curves = Phi_i @ weights
    return coeff_curves


def predict_direct(
    X_cosmo: np.ndarray,
    coeff_curves: np.ndarray,
    shift: float,
) -> np.ndarray:
    n_cosmo = X_cosmo.shape[0]
    n_bins = coeff_curves.shape[0]
    preds_log = np.zeros((n_cosmo, n_bins), dtype=np.float64)
    for idx in range(n_bins):
        preds_log[:, idx] = X_cosmo @ coeff_curves[idx, :3] + coeff_curves[idx, 3]
    return helper.invert_target_transform(preds_log, shift)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.array(
        [helper.r2_score_1d(y_true[idx], y_pred[idx]) for idx in range(y_true.shape[0])],
        dtype=np.float64,
    )


def format_curve_expression(name: str, weights: np.ndarray, centers: np.ndarray, width: float) -> str:
    terms = [f"{weights[0]:.12g}", f"({weights[1]:.12g})*i_std"]
    for idx, center in enumerate(centers):
        coeff = weights[idx + 2]
        terms.append(
            f"({coeff:.12g})*exp(-0.5*((i_std - ({center:.12g})) / ({width:.12g}))^2)"
        )
    return f"{name}(i_std) = " + " + ".join(terms)


def save_expression_summary(
    output_dir: str,
    target_source: str,
    shift: float,
    centers: np.ndarray,
    width: float,
    weights: np.ndarray,
) -> None:
    out = os.path.join(output_dir, "best_direct_expression.txt")
    with open(out, "w") as handle:
        handle.write("Direct diagonal covariance equation\n")
        handle.write(f"target_source : {target_source}\n")
        handle.write(f"log10_shift   : {shift:.6e}\n")
        handle.write(f"n_centers     : {len(centers)}\n")
        handle.write(f"rbf_width     : {width:.6f}\n\n")
        handle.write("Variables:\n")
        handle.write("  Omega_m_std, sigma_8_std, w_std, i_std\n\n")
        handle.write(format_curve_expression("a", weights[:, 0], centers, width) + "\n\n")
        handle.write(format_curve_expression("b", weights[:, 1], centers, width) + "\n\n")
        handle.write(format_curve_expression("c", weights[:, 2], centers, width) + "\n\n")
        handle.write(format_curve_expression("d", weights[:, 3], centers, width) + "\n\n")
        handle.write("Combined direct form:\n")
        handle.write(
            "  log10(C_ii + shift) = a(i_std)*Omega_m_std + "
            "b(i_std)*sigma_8_std + c(i_std)*w_std + d(i_std)\n"
        )
        handle.write(
            f"  C_ii = 10**(a(i_std)*Omega_m_std + b(i_std)*sigma_8_std + "
            f"c(i_std)*w_std + d(i_std)) - {shift:.12g}\n"
        )
    print(f"Saved direct expression summary -> {out}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cosmo, dv, rebinned_means, covariance, diag_cov, target_source = load_targets(args)
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
    ) = helper.split_data(
        cosmo=cosmo,
        covariance=covariance,
        diag_cov=diag_cov,
        dv=dv,
        rebinned_means=rebinned_means,
        test_size=args.test_size,
        seed=args.seed,
    )

    cosmo_scaler = StandardScaler().fit(cosmo_train)
    X_cosmo_train = cosmo_scaler.transform(cosmo_train)
    X_cosmo_test = cosmo_scaler.transform(cosmo_test)

    index_scaler = StandardScaler().fit(i_arr.reshape(-1, 1))
    i_std = index_scaler.transform(i_arr.reshape(-1, 1)).reshape(-1)
    Phi_i, centers = build_rbf_design(i_std, args.n_centers, args.rbf_width)

    shift = helper.compute_log_shift(diag_cov_train)

    teacher_coeffs = fit_per_bin_linear_teachers(
        X_cosmo_train=X_cosmo_train,
        diag_cov_train=diag_cov_train,
        shift=shift,
    )
    teacher_train = predict_direct(X_cosmo_train, teacher_coeffs, shift)
    teacher_test = predict_direct(X_cosmo_test, teacher_coeffs, shift)
    teacher_r2 = evaluate_predictions(diag_cov_test, teacher_test)

    coeff_curves = smooth_teacher_coefficients(teacher_coeffs, Phi_i)
    direct_train = predict_direct(X_cosmo_train, coeff_curves, shift)
    direct_test = predict_direct(X_cosmo_test, coeff_curves, shift)
    direct_r2 = evaluate_predictions(diag_cov_test, direct_test)
    direct_vs_teacher_r2 = evaluate_predictions(teacher_test, direct_test)

    print("\nPer-bin linear teacher performance:")
    print(f"  Median R² : {np.median(teacher_r2):.4f}")
    print(f"  Min R²    : {np.min(teacher_r2):.4f}")
    print(f"  Max R²    : {np.max(teacher_r2):.4f}")

    print("\nDirect smoothed equation performance:")
    print(f"  Median R² vs target  : {np.median(direct_r2):.4f}")
    print(f"  Min R² vs target     : {np.min(direct_r2):.4f}")
    print(f"  Max R² vs target     : {np.max(direct_r2):.4f}")
    print(f"  Median R² vs teacher : {np.median(direct_vs_teacher_r2):.4f}")

    with open(os.path.join(args.output_dir, "test_cosmologies.pkl"), "wb") as handle:
        pickle.dump(
            {
                "indices": idx_test,
                "cosmo": cosmo_test,
                "covariance": covariance_test,
                "diag_cov": diag_cov_test,
                "teacher_prediction": teacher_test,
                "direct_prediction": direct_test,
                "target_source": target_source,
                "n_centers": args.n_centers,
                "rbf_width": args.rbf_width,
            },
            handle,
        )

    np.save(os.path.join(args.output_dir, "teacher_predictions_test.npy"), teacher_test)
    np.save(os.path.join(args.output_dir, "direct_predictions_test.npy"), direct_test)
    np.save(os.path.join(args.output_dir, "teacher_coefficients_per_bin.npy"), teacher_coeffs)
    np.save(os.path.join(args.output_dir, "direct_coefficients_per_bin.npy"), coeff_curves)

    with open(os.path.join(args.output_dir, "cosmo_scaler.pkl"), "wb") as handle:
        pickle.dump(cosmo_scaler, handle)
    with open(os.path.join(args.output_dir, "index_scaler.pkl"), "wb") as handle:
        pickle.dump(index_scaler, handle)
    with open(os.path.join(args.output_dir, "target_transform.pkl"), "wb") as handle:
        pickle.dump({"log10_shift": shift}, handle)

    save_expression_summary(
        output_dir=args.output_dir,
        target_source=target_source,
        shift=shift,
        centers=centers,
        width=args.rbf_width,
        weights=np.linalg.lstsq(Phi_i, teacher_coeffs, rcond=None)[0],
    )

    helper.plot_reconstructions(
        args.output_dir,
        i_arr,
        cosmo_test,
        diag_cov_test,
        direct_test,
        direct_r2,
    )
    helper.plot_r2_distribution(args.output_dir, direct_r2)

    print(f"\nDone. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
