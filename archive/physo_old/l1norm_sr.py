"""
l1norm_sr.py  —  Symbolic Regression on Wavelet L1-Norm Datavectors
====================================================================
Goal: find a compact symbolic formula relating cosmological parameters
      (Omega_m, sigma_8, w) to a single rebinned wavelet L1-norm bin.

Workflow:
  1. Load datavectors + cosmological params from CSV
  2. Rebin 400-length L1-norm vectors into N_BINS summary bins
  3. Compute feature importance (Pearson + Mutual Information)
  4. For each bin j: run PhySO SR with X = [Omega_m, sigma_8, w], y = bin_j
  5. Save best expression, training curves, and logs per bin

Usage:
  python l1norm_sr.py --bin 0 --parameters 3 --seed 42

Compared to tau_sr.py:
  - Target is a rebinned L1-norm bin, not tau
  - X is always [Omega_m, sigma_8, w] (3 inputs, not 8-11)
  - Outer loop over bins added
  - Residual SR option added to handle Omega_m dominance
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import argparse
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler   # NEW: normalize inputs for SR stability

# ── PhySO package ──────────────────────────────────────────────────────────────
import physo
import physo.learn.monitoring as monitoring

import pickle

# ── Local config (hyperparameters, priors, RNN architecture) ──────────────────
import config as custom_config


# ==============================================================================
# PARALLELIZATION DEFAULTS
# ==============================================================================
# PhySO can parallelize expression evaluation across CPUs.
# WARNING: parallel mode can cause race conditions with large sample counts.
# PyTorch will still use multiple cores internally via BLAS/OpenMP even in
# single-process mode, so effective parallelism is maintained.
PARALLEL_MODE_DEFAULT = True
N_CPUS_DEFAULT        = 8


# ==============================================================================
# COMMAND-LINE ARGUMENTS
# ==============================================================================

parser = argparse.ArgumentParser(
    description     = "Runs a Wavelet L1-Norm SR job per rebinned bin.",
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

# Which bin of the rebinned datavector to target (0-indexed).
# Run this script once per bin, e.g. --bin 0, --bin 1, ..., --bin 4
parser.add_argument("-b", "--bin", default=0,
                    help="Which rebinned datavector bin to use as target y.")

# Total number of rebinned bins to split the 400-length datavectors into.
# e.g. 5 → each bin averages 80 of the 400 original scales.
parser.add_argument("-nb", "--n_bins", default=5,
                    help="Number of bins to rebin the 400-length datavectors into.")

# Number of free (learnable) numerical constants in the candidate expressions.
# e.g. 3 → SR can use c0, c1, c2 as free parameters optimized per expression.
# More free constants = more flexible expressions but slower optimization.
parser.add_argument("-fp", "--parameters", default=3,
                    help="Number of free constants to use (c0, c1, ...).")

# Soft length prior: center of the Gaussian penalty on expression length.
# Expressions near this length are preferred; longer ones are penalized.
parser.add_argument("-ll", "--length_loc", default=custom_config.LENGTH_LOC,
                    help="Soft length prior location (preferred expression length).")

# Soft length prior: standard deviation of the Gaussian penalty.
# Larger = more tolerance for long/short expressions.
parser.add_argument("-ls", "--length_scale", default=custom_config.LENGTH_SCALE,
                    help="Soft length prior scale (tolerance around length_loc).")

# Random seed for reproducibility of RNN initialization and data shuffling.
parser.add_argument("-s", "--seed", default=0,
                    help="Random seed.")

# Whether to run SR on residuals after removing the Omega_m contribution.
# Useful when Omega_m dominates and SR ignores sigma_8 and w.
# True → first fit f(Omega_m), then SR on y - f(Omega_m) with [sigma_8, w] only.
parser.add_argument("-res", "--residual_mode", default=False,
                    help="If True, run residual SR to isolate sigma_8 and w contributions.")

# Parallelization settings (see top of file for caveats).
parser.add_argument("-p",     "--parallel_mode", default=PARALLEL_MODE_DEFAULT,
                    help="Whether to use parallel expression evaluation.")
parser.add_argument("-ncpus", "--ncpus",          default=N_CPUS_DEFAULT,
                    help="Number of CPUs for parallel mode.")

config = vars(parser.parse_args())

# ── Parse arguments into typed variables ──────────────────────────────────────

# Which bin index to target as y
BIN_IDX       = int(config["bin"])

# Total number of bins for rebinning
N_BINS        = int(config["n_bins"])

# Random seed
SEED          = int(config["seed"])

# Number of free constants (named c0, c1, ..., c_{N-1})
N_FREE_PARAMS = int(config["parameters"])

# Soft length prior parameters
LENGTH_LOC_ARG   = int(config["length_loc"])
LENGTH_SCALE_ARG = int(config["length_scale"])

# Residual SR mode (tackles Omega_m dominance problem)
RESIDUAL_MODE = bool(int(config["residual_mode"]))

# Parallel execution config
PARALLEL_MODE = bool(config["parallel_mode"])
N_CPUS        = int(config["ncpus"])


# ==============================================================================
# MATPLOTLIB SAFETY (avoid TeX rendering race conditions in multiprocessing)
# ==============================================================================
import matplotlib
matplotlib.rcParams['text.usetex'] = False


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':

    # ── Reproducibility ────────────────────────────────────────────────────────
    # Fix both numpy and torch seeds so results are reproducible across runs
    # with the same --seed argument.
    np.random.seed(SEED)
    torch.manual_seed(SEED)


    # ==========================================================================
    # LOAD DATASET
    # ==========================================================================
    # CSV was already rebinned by prepare_data.py into N_BINS columns:
    #   Omega_m, sigma_8, w, bin_0, bin_1, ..., bin_{N_BINS-1}
    # NO further rebinning needed here — just read the target column directly.

    df = pd.read_csv(
        f'/Users/arnablahiry/repos/SymReg-L1-Norm/data/csv/l1norm_training_data_b{N_BINS}.csv'
    )

    df_test = df.sample(frac=0.2)
    df_train = df.drop(df_test.index)

    # Save the test set
    with open('df_test.pkl', 'wb') as f:
        pickle.dump(df_test, f)

    with open('df_train.pkl', 'wb') as f:
        pickle.dump(df_train, f)

    # Verify the expected bin column actually exists in the CSV
    assert f'bin_{BIN_IDX}' in df_train.columns, (
        f"Column 'bin_{BIN_IDX}' not found in CSV. "
        f"Available bin columns: {[c for c in df_train.columns if c.startswith('bin_')]}\n"
        f"Did you run: python prepare_data.py -b {N_BINS} ?"
    )

    # ── Cosmological input variables (X) ──────────────────────────────────────
    X_names = ['Omega_m', 'sigma_8', 'w']

    # ── Select target bin directly from CSV ───────────────────────────────────
    # bin_{BIN_IDX} was already computed as the mean of (400 // N_BINS) raw scales
    # by prepare_data.py. No slicing or averaging needed here.
    y_name = f'L1_bin{BIN_IDX}'
    y      = df_train[f'bin_{BIN_IDX}'].to_numpy()             # shape: (n_samples,)


    #y = np.log10(y)

    # ── Build X matrix ────────────────────────────────────────────────────────
    # PhySO expects shape (n_variables, n_samples)
    X = df_train[X_names].to_numpy().T                         # shape: (3, n_samples)

    # ── Normalize inputs ──────────────────────────────────────────────────────
    # Standardizing ensures Omega_m, sigma_8, w compete equally during SR search.
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X.T).T               # shape: (3, n_samples)

    print(f"Loaded {X.shape[1]} samples, targeting bin_{BIN_IDX} of {N_BINS}")
    print(f"y (log10) stats:  min={y.min():.3f}  max={y.max():.3f}  "
          f"mean={y.mean():.3f}  std={y.std():.3f}")
    print(f"Any NaN in y: {np.isnan(y).any()}  —  Any NaN in X: {np.isnan(X).any()}")

    # ==========================================================================
    # RUN NAMING AND DIRECTORY SETUP
    # ==========================================================================
    # Each run gets a unique name encoding all hyperparameters.
    # All outputs (logs, plots, best expression) are saved in this directory.

    RUN_NAME = (
        "L1NORM_SR_bin%d_of%d_res%i_fp%d_lloc%d_lscale%d_s%d"
        % (BIN_IDX, N_BINS, int(RESIDUAL_MODE), N_FREE_PARAMS,
           LENGTH_LOC_ARG, LENGTH_SCALE_ARG, SEED)
    )

    # File paths for outputs (relative to RUN_NAME directory, set below)
    PATH_DATA       = f"{RUN_NAME}_data.csv"           # input data used for this run
    PATH_DATA_CORR  = f"{RUN_NAME}_features.csv"       # feature importance table
    PATH_DATA_PLOT  = f"{RUN_NAME}_data.png"           # scatter plots X[i] vs y

    # Create run directory and cd into it so all PhySO outputs land here
    if not os.path.exists(RUN_NAME):
        os.makedirs(RUN_NAME)
    os.chdir(os.path.join(os.path.dirname(__file__), RUN_NAME))


    # ==========================================================================
    # SAVE AND VISUALIZE DATASET
    # ==========================================================================

    # Save the dataset used for this specific bin run (useful for auditing)
    df_save = pd.DataFrame(
        data    = np.concatenate((y[np.newaxis, :], X), axis=0).T,
        columns = [y_name] + X_names
    )
    df_save.to_csv(PATH_DATA, sep=";")

    # ── Scatter plots: each X variable vs the target bin ─────────────────────
    # A quick visual check: you should see a clear trend for variables that matter.
    # If sigma_8 and w show flat/noisy scatter → they genuinely don't drive this bin.
    n_dim = X.shape[0]                                    # = 3 for [Omega_m, sigma_8, w]
    fig, ax = plt.subplots(n_dim, 1, figsize=(10, int(n_dim * 5)))
    for i in range(n_dim):
        curr_ax = ax if n_dim == 1 else ax[i]
        curr_ax.plot(X[i], y, 'k.', alpha=0.6)
        curr_ax.set_xlabel(X_names[i])
        curr_ax.set_ylabel(y_name)
        curr_ax.set_title(f"Scatter: {X_names[i]} vs {y_name}")
    plt.tight_layout()
    fig.savefig(PATH_DATA_PLOT)
    plt.close(fig)                                        # don't display when running on server


    # ==========================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ==========================================================================
    # Before running SR (which is expensive), quantify how much each of the
    # 3 cosmo params actually influences bin_j. This diagnoses the Omega_m
    # dominance problem and tells you whether sigma_8/w are even detectable.

    # Pearson correlation: measures linear dependence. Fast but misses nonlinear effects.
    pearsons = np.array([
        np.corrcoef(X.T[:, i], y)[0, 1]
        for i in range(X.T.shape[1])
    ])

    # Mutual Information: measures any statistical dependence (linear or nonlinear).
    # More expensive but more complete. High MI → SR should find a relationship.
    mi = mutual_info_regression(X.T, y, random_state=0)

    # Combine into a DataFrame and sort by MI (most important first)
    df_features = pd.DataFrame({
        'Variable'   : X_names,
        'Pearson_r'  : pearsons,
        'Abs_r'      : np.abs(pearsons),
        'Mutual_Info': mi
    }).sort_values('Mutual_Info', ascending=False)
    df_features.to_csv(PATH_DATA_CORR, sep=";")

    # Print summary to console so you can see importance ranking immediately
    print(f"\n{'='*60}")
    print(f"Feature importance for {y_name}:")
    print(df_features.to_string(index=False))
    print(f"{'='*60}\n")


    # ==========================================================================
    # OPTIONAL: RESIDUAL SR MODE
    # ==========================================================================
    # Problem: if Omega_m dominates (high MI, high Pearson_r), PhySO will find
    # f(Omega_m) and never bother adding sigma_8 or w (not worth the complexity cost).
    #
    # Solution: Stage 1 → fit f(Omega_m) alone. Stage 2 → SR on the residual
    # y - f(Omega_m), using only [sigma_8, w] as inputs.
    # This forces SR to discover the sigma_8/w dependence explicitly.

    if RESIDUAL_MODE:
        print(f"{RUN_NAME}: Running Stage 1 — fitting Omega_m contribution only")

        # Stage 1: SR with only Omega_m as input
        # Uses fewer free params and simpler ops to get a clean Omega_m baseline
        expr_Om, _ = physo.SR(
            X[[0]],                                       # only Omega_m row
            y,
            X_names           = [X_names[0]],            # ['Omega_m']
            y_name            = y_name,
            fixed_consts      = [1.],
            free_consts_names = ["c0", "c1"],             # just 2 constants for simple fit
            op_names          = ["mul", "pow", "log", "exp"],
            run_config        = custom_config.custom_config,
            max_n_evaluations = int(1e6),                 # shorter run for stage 1
            epochs            = int(1e9),
            parallel_mode     = PARALLEL_MODE,
            n_cpus            = N_CPUS,
        )

        # Compute residual: what Omega_m alone cannot explain
        y_Om_pred = expr_Om.execute(X[[0]])               # evaluate best Omega_m expression
        y_residual = y - y_Om_pred                        # residual signal
        print(f"Stage 1 expression: {expr_Om}")
        print(f"Residual std / original std: {y_residual.std() / y.std():.3f}")

        # Stage 2: SR on residual with sigma_8 and w only
        X_for_sr    = X[[1, 2]]                          # sigma_8 and w rows
        y_for_sr    = y_residual
        X_names_sr  = [X_names[1], X_names[2]]           # ['sigma_8', 'w']
        y_name_sr   = f"{y_name}_residual"
        print(f"{RUN_NAME}: Running Stage 2 — SR on residual with sigma_8 and w")

    else:
        # Standard mode: SR directly on all 3 params vs the bin target
        X_for_sr   = X_scaled                            # normalized [Omega_m, sigma_8, w]
        y_for_sr   = y
        X_names_sr = X_names                             # ['Omega_m', 'sigma_8', 'w']
        y_name_sr  = y_name


    # ==========================================================================
    # SR CONFIGURATION
    # ==========================================================================

    # Load the global config dict from config.py
    CONFIG = custom_config.custom_config

    # Override the soft length prior from command-line args.
    # This allows tuning per-run without editing config.py.
    for prior in CONFIG['priors_config']:
        if prior[0] == 'SoftLengthPrior':
            prior[1]['length_loc'] = LENGTH_LOC_ARG      # preferred expression length
            prior[1]['scale']      = LENGTH_SCALE_ARG    # tolerance around that length

    # Plot the soft length prior shape so you can visually inspect the penalty curve
    '''physo.config.utils.soft_length_plot(
        CONFIG,
        save_path="soft_length_prior.png",
        do_show=False
    )'''

    # ── Free constants ────────────────────────────────────────────────────────
    # Named c0, c1, ..., c_{N_FREE_PARAMS-1}. These are numerically optimized
    # per expression candidate. More = more flexible but slower per evaluation.
    FREE_CONSTS_NAMES = [f"c{i}" for i in range(N_FREE_PARAMS)]

    # Fixed constants: numerical values that appear literally in expressions
    # (not learned). [1.] allows SR to use the number 1 directly.
    FIXED_CONSTS = [1.]

    # ── Allowed mathematical operations ──────────────────────────────────────
    # These define the vocabulary of the expression tree search.
    # For cosmological power spectra, power-law relations dominate, so
    # mul/div/pow/exp/log are most relevant.
    # sin/tanh excluded: no physical motivation for oscillatory L1-norm formulas.
    OP_NAMES = [
        "mul",    # multiplication:  x * y
        "add",    # addition:        x + y
        "sub",    # subtraction:     x - y
        "div",    # division:        x / y
        "inv",    # reciprocal:      1 / x
        "neg",    # negation:        -x
        "n2",     # square:          x^2
        "sqrt",   # square root:     sqrt(x)
        "exp",    # exponential:     e^x
        #"log",    # natural log:     ln(x)
        #"pow",  # general power:   x^y  (add back if integer exponents not enough)
        # "sin",  # sine:            sin(x)  (not expected in L1-norm scaling)
        # "tanh", # hyperbolic tan:  tanh(x) (not expected)
    ]

    # Run indefinitely (stopped by max_n_evaluations or manually).
    # PhySO's internal epoch counter is unreliable for wall-time budgeting.
    MAX_N_EVALUATIONS = int(1e99)
    N_EPOCHS          = int(100)


    # ==========================================================================
    # LOGGING AND VISUALIZATION CONFIG
    # ==========================================================================

    save_path_training_curves = 'sr_curves.png'   # reward/loss curves saved here
    save_path_log             = 'sr.log'           # text log of best expressions per epoch

    # RunLogger: saves best expressions and their rewards to a CSV log file
    run_logger = lambda: monitoring.RunLogger(
        save_path = save_path_log,
        do_save   = True
    )

    # RunVisualiser: plots training reward curves and prints best expressions
    # do_show=False → suppress interactive display (for cluster/server runs)
    run_visualiser = lambda: monitoring.RunVisualiser(
        epoch_refresh_rate = 1,          # update plot every epoch
        save_path          = save_path_training_curves,
        do_show            = False,      # no GUI popup (set True for local debugging)
        do_prints          = True,       # print best expressions to console each epoch
        do_save            = True,       # save curve plot to file
    )


    # ==========================================================================
    # SR RUN
    # ==========================================================================

    print(f"\n{RUN_NAME} : Starting SR task")
    print(f"  Target:   {y_name_sr}")
    print(f"  Inputs:   {X_names_sr}")
    print(f"  Samples:  {X_for_sr.shape[1]}")
    print(f"  Free consts: {FREE_CONSTS_NAMES}")
    print(f"  Operators: {OP_NAMES}\n")

    expression, logs = physo.SR(
        X_for_sr,                             # input matrix: (n_vars, n_samples)
        y_for_sr,                             # target vector: (n_samples,)

        # Variable and target names (used in expression printout)
        X_names           = X_names_sr,
        y_name            = y_name_sr,

        # Fixed numerical constants available in expressions (e.g., 1.0)
        fixed_consts      = FIXED_CONSTS,

        # Learnable constant names (optimized per expression candidate)
        free_consts_names = FREE_CONSTS_NAMES,

        # Mathematical operations defining the expression vocabulary
        op_names          = OP_NAMES,

        # Logging callbacks
        get_run_logger     = run_logger,
        get_run_visualiser = run_visualiser,

        # Full hyperparameter config (from config.py, modified above)
        run_config        = CONFIG,

        # Safety cap: stop after this many expression evaluations regardless of epochs
        max_n_evaluations = MAX_N_EVALUATIONS,

        # Number of training epochs (set high; rely on max_n_evaluations to stop)
        epochs            = N_EPOCHS,

        # Parallelization
        parallel_mode     = PARALLEL_MODE,
        n_cpus            = N_CPUS,
    )

    # ==========================================================================
    # RESULTS
    # ==========================================================================

    print(f"\n{RUN_NAME} : SR task finished")
    print(f"Best expression found:  {expression}")

    # Save the best expression string to a dedicated file for easy retrieval
    with open("best_expression.txt", "w") as f:
        f.write(f"Target: {y_name_sr}\n")
        f.write(f"Inputs: {X_names_sr}\n")
        f.write(f"Expression: {expression}\n")

    # If in residual mode, print the combined full expression
    if RESIDUAL_MODE:
        print(f"\nStage 1 (Omega_m):   {expr_Om}")
        print(f"Stage 2 (residual):  {expression}")
        print(f"Full model:          {expr_Om}  +  {expression}")
        with open("best_expression.txt", "a") as f:
            f.write(f"\nFull model (Stage1 + Stage2): {expr_Om}  +  {expression}\n")
