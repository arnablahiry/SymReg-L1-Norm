import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import argparse
import os
import pickle
from scipy.optimize import curve_fit

matplotlib.rcParams['text.usetex'] = False

# ==============================================================================
# ARGS
# ==============================================================================

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-nb", "--n_bins", default=20, type=int)
parser.add_argument("-s",  "--seed",   default=42, type=int)
config = vars(parser.parse_args())

N_BINS = config["n_bins"]
SEED   = config["seed"]

# ==============================================================================
# PARAMETRIC FORM
# ==============================================================================

def dog(idx, c0, c1, c2, c3, c4):
    """
    Difference of Gaussians — double-lobe with central dip.
      c0: outer amplitude
      c1: centre
      c2: outer width  (must be > c4 for lobes to appear)
      c3: inner amplitude
      c4: inner width
    """
    outer = c0 * np.exp(-((idx - c1) ** 2) / (2 * c2 ** 2))
    inner = c3 * np.exp(-((idx - c1) ** 2) / (2 * c4 ** 2))
    return outer - inner

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':

    np.random.seed(SEED)

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================

    df = pd.read_csv(
        f'/Users/arnablahiry/repos/SymReg-L1-Norm/data/csv/l1norm_training_data_b{N_BINS}.csv'
    )
    for j in range(N_BINS):
        assert f'bin_{j}' in df.columns, f"Column bin_{j} missing."

    df_test  = df.sample(frac=0.2, random_state=SEED)
    df_train = df.drop(df_test.index).reset_index(drop=True)
    with open('df_test.pkl',  'wb') as f: pickle.dump(df_test,  f)
    with open('df_train.pkl', 'wb') as f: pickle.dump(df_train, f)

    n_sims   = len(df_train)
    bin_axis = np.arange(N_BINS, dtype=float)
    cosmo_cols = ['Omega_m', 'sigma_8', 'w']

    print(f"Loaded {n_sims} training simulations, {N_BINS} bins each.")

    # ==========================================================================
    # CURVE FIT PER SIMULATION
    # ==========================================================================

    records      = []   # will become the stage1_constants.csv
    failed_sims  = []
    y_fitted_all = np.zeros((n_sims, N_BINS))

    # Initial guess: centre in the middle, moderate widths, equal amplitudes
    p0     = [1.0, N_BINS / 2, N_BINS / 5, 0.5, N_BINS / 10]
    bounds = (
        [0,   0,    0.5,  0,    0.1 ],   # lower bounds
        [np.inf, N_BINS, N_BINS, np.inf, N_BINS]   # upper bounds
    )

    for i in range(n_sims):
        y_sim = np.array([df_train.iloc[i][f'bin_{j}'] for j in range(N_BINS)])

        try:
            popt, _ = curve_fit(
                dog, bin_axis, y_sim,
                p0     = p0,
                bounds = bounds,
                maxfev = 10000,
            )
            c0, c1, c2, c3, c4 = popt
            y_fitted_all[i] = dog(bin_axis, *popt)

            records.append({
                'Omega_m': df_train.iloc[i]['Omega_m'],
                'sigma_8': df_train.iloc[i]['sigma_8'],
                'w':       df_train.iloc[i]['w'],
                'c0': c0, 'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4,
            })

        except RuntimeError:
            print(f"  Warning: curve_fit failed for simulation {i} — skipping.")
            failed_sims.append(i)

    df_consts = pd.DataFrame(records)
    df_consts.to_csv('stage1_constants.csv', index=False)

    n_ok = len(records)
    print(f"\nFitted {n_ok}/{n_sims} simulations successfully.")
    if failed_sims:
        print(f"Failed sims: {failed_sims}")
    print(f"\nConstant statistics:")
    print(df_consts[['c0','c1','c2','c3','c4']].describe().to_string())
    print(f"\nSaved: stage1_constants.csv")

    # ==========================================================================
    # VISUALISATION 1: overlay fits on datavectors
    # ==========================================================================

    fig, ax = plt.subplots(figsize=(11, 5))
    for i in range(n_sims):
        y_sim = np.array([df_train.iloc[i][f'bin_{j}'] for j in range(N_BINS)])
        ax.plot(bin_axis, y_sim,       color='steelblue', lw=0.6, alpha=0.25,
                label='data' if i == 0 else None)
        if i not in failed_sims:
            ax.plot(bin_axis, y_fitted_all[i], color='tomato',    lw=0.8, alpha=0.4,
                    label='DoG fit' if i == 0 else None)
    ax.set_xlabel("bin index", fontsize=12)
    ax.set_ylabel("L1 norm",   fontsize=12)
    ax.set_title(f"Stage 1: DoG fits vs all training datavectors (n={n_ok})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("stage1_fits_overlay.png", dpi=200)
    plt.close()
    print("Saved: stage1_fits_overlay.png")

    # ==========================================================================
    # VISUALISATION 2: constants vs cosmological parameters
    # ==========================================================================
    # A quick look at whether each constant varies smoothly with cosmology.
    # Clean trends → Stage 2 SR will find a good expression.
    # Noisy / flat → that constant doesn't carry cosmological information.

    const_names = ['c0', 'c1', 'c2', 'c3', 'c4']
    fig2, axes = plt.subplots(len(const_names), len(cosmo_cols),
                               figsize=(12, 14), sharex='col')
    for ci, cn in enumerate(const_names):
        for cj, cosmo in enumerate(cosmo_cols):
            ax2 = axes[ci, cj]
            ax2.scatter(df_consts[cosmo], df_consts[cn],
                        s=10, alpha=0.6, color='steelblue')
            ax2.set_xlabel(cosmo,  fontsize=9)
            ax2.set_ylabel(cn,     fontsize=9)
            ax2.tick_params(labelsize=7)
            ax2.grid(True, lw=0.3, ls='--', alpha=0.4)

    fig2.suptitle("Stage 1 constants vs cosmological parameters", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("stage1_constants_vs_cosmo.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: stage1_constants_vs_cosmo.png")