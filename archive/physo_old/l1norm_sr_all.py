"""
l1norm_sr_stage1.py  —  Stage 1: Find curve shape f(idx; c0, c1, c2)
=====================================================================
Stack all simulations into one dataset.
X = bin index (0..N_BINS-1)   — the only input variable
y = L1 norm value

SR finds ONE tree that fits all simulations with global constants.
The constant values are discarded — only the tree structure matters.
That tree is then used in Stage 2 (curve_fit per simulation).

Usage:
  python l1norm_sr_stage1.py --n_bins 20 --parameters 3 --seed 42
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import pandas as pd
import argparse
import os
import pickle

import physo
import physo.learn.monitoring as monitoring
import config as custom_config

matplotlib.rcParams['text.usetex'] = False

PARALLEL_MODE_DEFAULT = True
N_CPUS_DEFAULT        = 8

# ==============================================================================
# ARGS
# ==============================================================================

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-nb",    "--n_bins",        default=20,                       type=int)
parser.add_argument("-fp",    "--parameters",    default=3,                        type=int)
parser.add_argument("-ll",    "--length_loc",    default=custom_config.LENGTH_LOC, type=int)
parser.add_argument("-ls",    "--length_scale",  default=custom_config.LENGTH_SCALE, type=int)
parser.add_argument("-s",     "--seed",          default=42,                       type=int)
parser.add_argument("-p",     "--parallel_mode", default=PARALLEL_MODE_DEFAULT)
parser.add_argument("-ncpus", "--ncpus",         default=N_CPUS_DEFAULT,           type=int)
config = vars(parser.parse_args())

N_BINS           = config["n_bins"]
N_FREE_PARAMS    = config["parameters"]
LENGTH_LOC_ARG   = config["length_loc"]
LENGTH_SCALE_ARG = config["length_scale"]
SEED             = config["seed"]
PARALLEL_MODE    = bool(config["parallel_mode"])
N_CPUS           = config["ncpus"]

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================

    df = pd.read_csv(
        f'/Users/arnablahiry/repos/SymReg-L1-Norm/data/csv/l1norm_training_data_b{N_BINS}.csv'
    )

    df = df.mean(axis=0).to_frame().T

    for j in range(N_BINS):
        assert f'bin_{j}' in df.columns, f"Column bin_{j} missing."

    df_test  = df.sample(frac=0.2, random_state=SEED)
    df_train = df.drop(df_test.index)
    with open('df_test.pkl',  'wb') as f: pickle.dump(df_test,  f)
    with open('df_train.pkl', 'wb') as f: pickle.dump(df_train, f)

    n_sims = len(df_train)
    print(f"Loaded {n_sims} training simulations, {N_BINS} bins each.")

    # ==========================================================================
    # STACK: X = bin_idx, y = L1_norm
    # ==========================================================================
    # Each simulation contributes N_BINS rows.
    # Cosmology is NOT an input here — Stage 1 only cares about curve shape.

    stacked_idx = []
    stacked_y   = []

    for j in range(N_BINS):
        stacked_idx.append(np.full(n_sims, j, dtype=float))
        stacked_y.append(df_train[f'bin_{j}'].to_numpy())

    idx_flat = np.concatenate(stacked_idx)             # (n_sims * N_BINS,)
    y_flat   = np.concatenate(stacked_y)               # (n_sims * N_BINS,)

    # PhySO expects X shape: (n_variables, n_samples)
    X = idx_flat[np.newaxis, :]                        # (1, n_sims * N_BINS)
    y = y_flat                                         # (n_sims * N_BINS,)

    print(f"Stacked dataset: {X.shape[1]} rows ({n_sims} sims x {N_BINS} bins)")
    print(f"X (bin_idx): min={X.min():.0f}  max={X.max():.0f}")
    print(f"y (L1_norm): min={y.min():.4f}  max={y.max():.4f}  "
          f"mean={y.mean():.4f}  std={y.std():.4f}")
    print(f"Any NaN in y: {np.isnan(y).any()}  Any NaN in X: {np.isnan(X).any()}")

    # ==========================================================================
    # QUICK VISUALISATION: overlay all datavectors
    # ==========================================================================

    fig, ax = plt.subplots(figsize=(10, 4))
    bin_axis = np.arange(N_BINS)
    for i in range(n_sims):
        row = np.array([df_train.iloc[i][f'bin_{j}'] for j in range(N_BINS)])
        ax.plot(bin_axis, row, lw=0.6, alpha=0.4)
    ax.set_xlabel("bin index")
    ax.set_ylabel("L1 norm")
    ax.set_title(f"All {n_sims} training datavectors")
    plt.tight_layout()
    plt.savefig("stage1_datavectors.png", dpi=150)
    plt.close()

    # ==========================================================================
    # RUN NAMING
    # ==========================================================================

    RUN_NAME = (
        "L1NORM_STAGE1_bins%d_fp%d_lloc%d_lscale%d_s%d"
        % (N_BINS, N_FREE_PARAMS, LENGTH_LOC_ARG, LENGTH_SCALE_ARG, SEED)
    )
    if not os.path.exists(RUN_NAME):
        os.makedirs(RUN_NAME)
    os.chdir(os.path.join(os.path.dirname(__file__), RUN_NAME))

    # ==========================================================================
    # SR CONFIG
    # ==========================================================================

    CONFIG = custom_config.custom_config
    for prior in CONFIG['priors_config']:
        if prior[0] == 'SoftLengthPrior':
            prior[1]['length_loc'] = LENGTH_LOC_ARG
            prior[1]['scale']      = LENGTH_SCALE_ARG

    FREE_CONSTS_NAMES = [f"c{i}" for i in range(N_FREE_PARAMS)]
    FIXED_CONSTS      = [1.]

    # Ops relevant for a smooth double-lobe curve.
    # exp/sqrt capture lobe decay; n2 captures peak shape.
    OP_NAMES = [
        "mul", "add", "sub", "div", 'log',
        "inv", "neg", "n2", "sqrt", "exp",
    ]

    save_path_training_curves = 'sr_curves.png'
    save_path_log             = 'sr.log'

    run_logger = lambda: monitoring.RunLogger(
        save_path = save_path_log,
        do_save   = True
    )
    run_visualiser = lambda: monitoring.RunVisualiser(
        epoch_refresh_rate = 1,
        save_path          = save_path_training_curves,
        do_show            = False,
        do_prints          = True,
        do_save            = True,
    )

    # ==========================================================================
    # SR RUN
    # ==========================================================================

    print(f"\n{'='*60}")
    print(f"STAGE 1: Finding curve shape f(bin_idx; c0..c{N_FREE_PARAMS-1})")
    print(f"  Sims:    {n_sims}")
    print(f"  Bins:    {N_BINS}")
    print(f"  Samples: {X.shape[1]}")
    print(f"  Consts:  {FREE_CONSTS_NAMES}  (global — values discarded after)")
    print(f"  Ops:     {OP_NAMES}")
    print(f"{'='*60}\n")

    expression, logs = physo.SR(
        X,
        y,
        X_names    = ['bin'],
        X_units    = [[0, 0, 0]],   # dimensionless
        y_name     = 'L1_{norm}',
        y_units    = [0, 0, 0],     # dimensionless
        fixed_consts      = FIXED_CONSTS,
        free_consts_names = FREE_CONSTS_NAMES,
        op_names          = OP_NAMES,
        get_run_logger     = run_logger,
        get_run_visualiser = run_visualiser,
        run_config        = CONFIG,
        max_n_evaluations = int(1e99),
        epochs            = int(100),
        parallel_mode     = PARALLEL_MODE,
        n_cpus            = N_CPUS,
    )

    # ==========================================================================
    # RESULTS
    # ==========================================================================

    print(f"\nStage 1 finished.")
    print(f"Tree found:  {expression}")
    print(f"\nNOTE: constant values above are global (fitted to average cosmology).")
    print(f"They will be discarded. Only the tree structure is used in Stage 2.")

    with open("stage1_tree.txt", "w") as f:
        f.write(f"Stage 1 tree (structure only — constants are placeholders):\n")
        f.write(f"{expression}\n\n")
        f.write(f"Inputs:  ['bin_idx']\n")
        f.write(f"Consts:  {FREE_CONSTS_NAMES}\n")
        f.write(f"N_BINS:  {N_BINS}\n")
        f.write(f"N_SIMS:  {n_sims}\n")