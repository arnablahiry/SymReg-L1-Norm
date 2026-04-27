"""
test_shapes.py  —  Test candidate parametric shapes for the double-lobe
========================================================================
Fits 4 candidate functions to every training simulation and compares
median R² to find the best parametric form for Stage 1.

Candidates:
  1. Difference of Gaussians (DoG)
  2. Mexican Hat wavelet       (physically motivated for wavelet L1 norms)
  3. Gaussian * quadratic      (generates a natural dip at centre)
  4. Sum of two Gaussians      (symmetric lobes, no enforced dip)

Usage:
  python test_shapes.py --n_bins 20 --seed 42
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument("-nb", "--n_bins", default=20, type=int)
parser.add_argument("-s",  "--seed",   default=42, type=int)
config = vars(parser.parse_args())
N_BINS = config["n_bins"]
SEED   = config["seed"]

# ==============================================================================
# CANDIDATE SHAPES
# ==============================================================================

def dog(x, c0, c1, c2, c3, c4):
    """Difference of Gaussians — 5 params"""
    outer = c0 * np.exp(-((x - c1) ** 2) / (2 * c2 ** 2))
    inner = c3 * np.exp(-((x - c1) ** 2) / (2 * c4 ** 2))
    return outer - inner

def mexican_hat(x, c0, c1, c2):
    """Mexican hat wavelet: c0*(1-t²)*exp(-t²/2), t=(x-c1)/c2 — 3 params"""
    t = (x - c1) / c2
    return c0 * (1 - t ** 2) * np.exp(-t ** 2 / 2)

def gauss_quadratic(x, c0, c1, c2, c3):
    """Gaussian envelope * (x-c1)²: natural zero at centre — 4 params"""
    return c0 * ((x - c1) ** 2) * np.exp(-((x - c1) ** 2) / (2 * c2 ** 2)) + c3

def two_gaussians(x, c0, c1, c2, c3, c4, c5):
    """Sum of two independent Gaussians — 6 params"""
    left  = c0 * np.exp(-((x - c1) ** 2) / (2 * c2 ** 2))
    right = c3 * np.exp(-((x - c4) ** 2) / (2 * c5 ** 2))
    return left + right

# Each entry: (name, function, p0_fn, bounds_fn)
# p0_fn and bounds_fn take N_BINS as argument so they scale with bin count
CANDIDATES = [
    (
        "DoG",
        dog,
        lambda n: [0.25, n/2, n/6, 0.2, n/20],
        lambda n: ([0, 0, n/10, 0, 0.1],
                   [np.inf, n, n/2, np.inf, n/4])
    ),
    (
        "Mexican Hat",
        mexican_hat,
        lambda n: [0.25, n/2, n/6],
        lambda n: ([0,      0,   0.5],
                   [np.inf, n,   n  ])
    ),
    (
        "Gauss*Quadratic",
        gauss_quadratic,
        lambda n: [0.001, n/2, n/6, 0.0],
        lambda n: ([0,      0,   0.5,  -np.inf],
                   [np.inf, n,   n,     np.inf ])
    ),
    (
        "Two Gaussians",
        two_gaussians,
        lambda n: [0.2, n/3, n/8, 0.2, 2*n/3, n/8],
        lambda n: ([0, 0,   0.5, 0, 0,     0.5],
                   [np.inf, n, n, np.inf, n, n  ])
    ),
]

# ==============================================================================
# LOAD DATA
# ==============================================================================

'''with open('df_train.pkl', 'rb') as f:
    df_train = pickle.load(f)'''

df_train = pd.read_csv('/Users/arnablahiry/repos/SymReg-L1-Norm/data/csv/l1norm_training_data_b160.csv')

n_sims   = len(df_train)
bin_axis = np.arange(N_BINS, dtype=float)
print(f"Testing {len(CANDIDATES)} shapes on {n_sims} simulations x {N_BINS} bins\n")

# ==============================================================================
# FIT ALL CANDIDATES ON ALL SIMULATIONS
# ==============================================================================

results = {name: {'r2': [], 'rmse': [], 'failed': 0} for name, *_ in CANDIDATES}

for i in range(n_sims):
    y_sim = np.array([df_train.iloc[i][f'bin_{j}'] for j in range(N_BINS)])
    ss_tot = np.sum((y_sim - y_sim.mean()) ** 2)

    for name, func, p0_fn, bounds_fn in CANDIDATES:
        try:
            popt, _ = curve_fit(
                func, bin_axis, y_sim,
                p0     = p0_fn(N_BINS),
                bounds = bounds_fn(N_BINS),
                maxfev = 20000,
            )
            y_pred = func(bin_axis, *popt)
            ss_res = np.sum((y_sim - y_pred) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse   = np.sqrt(np.mean((y_sim - y_pred) ** 2))
            results[name]['r2'].append(r2)
            results[name]['rmse'].append(rmse)
        except RuntimeError:
            results[name]['failed'] += 1

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================

print(f"{'Shape':<20} {'n_params':>8} {'median R²':>10} {'mean R²':>10} "
      f"{'min R²':>8} {'failed':>7}")
print("-" * 68)

summary = []
for name, func, *_ in CANDIDATES:
    r2_arr   = np.array(results[name]['r2'])
    n_params = func.__code__.co_argcount - 1   # subtract x
    row = {
        'name'    : name,
        'n_params': n_params,
        'median'  : np.nanmedian(r2_arr),
        'mean'    : np.nanmean(r2_arr),
        'min'     : np.nanmin(r2_arr),
        'failed'  : results[name]['failed'],
    }
    summary.append(row)
    print(f"{name:<20} {n_params:>8} {row['median']:>10.4f} {row['mean']:>10.4f} "
          f"{row['min']:>8.4f} {row['failed']:>7}")

best_name = max(summary, key=lambda r: r['median'])['name']
print(f"\nBest shape by median R²: {best_name}")

# ==============================================================================
# VISUALISATION: R² distributions per candidate
# ==============================================================================

fig, axes = plt.subplots(1, len(CANDIDATES), figsize=(14, 4), sharey=True)
for ax, (name, *_) in zip(axes, CANDIDATES):
    r2_arr = np.array(results[name]['r2'])
    ax.hist(r2_arr, bins=20, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axvline(np.nanmedian(r2_arr), color='red', lw=1.5, ls='--',
               label=f'median={np.nanmedian(r2_arr):.3f}')
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("R²", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, ls='--', alpha=0.4)
axes[0].set_ylabel("count", fontsize=9)
fig.suptitle("R² distribution per candidate shape", fontsize=12)
plt.tight_layout()
plt.savefig("shape_comparison_r2.png", dpi=200)
plt.close()
print("\nSaved: shape_comparison_r2.png")

# ==============================================================================
# VISUALISATION: overlay best shape on a few example simulations
# ==============================================================================

best_func   = next(f for n, f, *_ in CANDIDATES if n == best_name)
best_p0     = next(p for n, _, p, *_ in CANDIDATES if n == best_name)
best_bounds = next(b for n, _, _, b, *_ in CANDIDATES if n == best_name)

n_examples = min(6, n_sims)
fig2, axes2 = plt.subplots(2, 3, figsize=(13, 7))
for k, ax2 in enumerate(axes2.flatten()):
    y_sim = np.array([df_train.iloc[k][f'bin_{j}'] for j in range(N_BINS)])
    try:
        popt, _ = curve_fit(best_func, bin_axis, y_sim,
                            p0=best_p0(N_BINS), bounds=best_bounds(N_BINS),
                            maxfev=20000)
        y_pred  = best_func(bin_axis, *popt)
        ss_res  = np.sum((y_sim - y_pred) ** 2)
        ss_tot  = np.sum((y_sim - y_sim.mean()) ** 2)
        r2      = 1 - ss_res / ss_tot
        ax2.plot(bin_axis, y_sim,   color='steelblue', lw=1.5, label='data')
        ax2.plot(bin_axis, y_pred,  color='tomato',    lw=1.5, ls='--',
                 label=f'{best_name} R²={r2:.3f}')
        ax2.set_title(f"sim {k}  Ωm={df_train.iloc[k]['Omega_m']:.2f}  "
                      f"σ8={df_train.iloc[k]['sigma_8']:.2f}", fontsize=8)
    except RuntimeError:
        ax2.set_title(f"sim {k} — fit failed", fontsize=8)
    ax2.legend(fontsize=7)
    ax2.grid(True, lw=0.3, ls='--', alpha=0.4)

fig2.suptitle(f"Best shape ({best_name}) on 6 example simulations", fontsize=12)
plt.tight_layout()
plt.savefig("shape_best_examples.png", dpi=200)
plt.close()
print("Saved: shape_best_examples.png")