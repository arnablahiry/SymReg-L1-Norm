"""
prepare_data.py  —  Convert numpy arrays to CSV for l1norm_sr.py
================================================================
Run this ONCE before running l1norm_sr.py.

Usage:
    python prepare_data.py --bins 10
    python prepare_data.py -b 5

Output:
    ../data/l1norm_training_data_b{BINS}.csv
    e.g. ../data/l1norm_training_data_b10.csv  for --bins 10
"""

import numpy as np
import pandas as pd
import argparse
import os


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

parser = argparse.ArgumentParser(
    description     = "Prepare L1-norm numpy data as CSV for l1norm_sr.py",
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

# Number of bins to rebin the raw datavectors into.
# e.g. --bins 10 → each bin averages (400 / 10) = 40 consecutive scales.
# This is the same N_BINS used later in l1norm_sr.py — keep them consistent.
parser.add_argument("-b", "--bins", type=int, default=10,
                    help="Number of bins to rebin the raw 160-length datavectors into.")

args         = parser.parse_args()
N_BINS_REBIN = args.bins    # how many summary bins to create from the raw datavector


# ==============================================================================
# LOAD YOUR NUMPY ARRAYS
# ==============================================================================

Om_sigma8_w     = np.load('/Users/arnablahiry/repos/SymReg-L1-Norm/data/Om_sigma8_w_array_for_arnab.npy')
theory_l1_kappa = np.load('/Users/arnablahiry/repos/SymReg-L1-Norm/data/theory_l1_kappa_all_cleaned_for_arnab.npy')[:,120:280]

print(f"Om_sigma8_w shape:     {Om_sigma8_w.shape}")      # expected: (N_sims, 3)
print(f"theory_l1_kappa shape: {theory_l1_kappa.shape}")  # expected: (N_sims, 160)


# ==============================================================================
# SANITY CHECKS
# ==============================================================================

# Simulation count must match between params and datavectors
assert Om_sigma8_w.shape[0] == theory_l1_kappa.shape[0], \
    "Mismatch: number of simulations differs between params and datavectors!"

N_sims = Om_sigma8_w.shape[0]
N_RAW  = theory_l1_kappa.shape[1]   # raw datavector length (e.g. 400)

print(f"\nNumber of simulations: {N_sims}")
print(f"Raw datavector length: {N_RAW}")
print(f"Rebinning into:        {N_BINS_REBIN} bins  ({N_RAW // N_BINS_REBIN} scales per bin)")

# N_BINS_REBIN must evenly divide the raw datavector length
assert N_RAW % N_BINS_REBIN == 0, (
    f"--bins {N_BINS_REBIN} does not evenly divide datavector length {N_RAW}. "
    f"Valid choices: {[d for d in range(1, N_RAW+1) if N_RAW % d == 0]}"
)


# ==============================================================================
# BUILD DATAFRAME — COSMOLOGICAL PARAMETERS
# ==============================================================================
# Column order assumed: 0 = Omega_m, 1 = sigma_8, 2 = w
# Adjust names here if your array has a different order.

df_params = pd.DataFrame(
    Om_sigma8_w,
    columns=['Omega_m', 'sigma_8', 'w']
)


# ==============================================================================
# REBIN DATAVECTORS
# ==============================================================================
# Average consecutive groups of (N_RAW // N_BINS_REBIN) scales into one bin.
# Shape goes from (N_sims, N_RAW) → (N_sims, N_BINS_REBIN)

bin_size    = N_RAW // N_BINS_REBIN
dv_rebinned = (
    theory_l1_kappa
    .reshape(N_sims, N_BINS_REBIN, bin_size)   # (N_sims, N_BINS_REBIN, bin_size)
    .mean(axis=2)                               # (N_sims, N_BINS_REBIN)
)

# Name each rebinned bin column as bin_0, bin_1, ..., bin_{N_BINS_REBIN-1}
# l1norm_sr.py will select y = bin_j for whichever --bin j you pass
df_dv = pd.DataFrame(
    dv_rebinned,
    columns=[f'bin_{j}' for j in range(N_BINS_REBIN)]
)


# ==============================================================================
# COMBINE INTO ONE DATAFRAME
# ==============================================================================

df = pd.concat([df_params, df_dv], axis=1)

print(f"\nFinal DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst row preview:")
print(df.iloc[0])


# ==============================================================================
# REMOVE ROWS WITH NON-POSITIVE BIN VALUES  (required for log10-space SR)
# ==============================================================================
# SR will run on log10(y), so any bin value <= 0 is invalid.
# Removes any simulation row where at least one rebinned bin is <= 0.

bin_cols   = [f'bin_{j}' for j in range(N_BINS_REBIN)]
# smallest non-negative value in each row
row_min_nonneg = df[bin_cols].where(df[bin_cols] >= 0).min(axis=1)

# replace values <= 0 with that row value
df[bin_cols] = df[bin_cols].mask(df[bin_cols] <= 0, row_min_nonneg, axis=0)


# ==============================================================================
# SAVE TO CSV
# ==============================================================================
# Output filename encodes the number of bins so different binnings
# don't overwrite each other. l1norm_sr.py reads this file by name.

out_dir  = '/Users/arnablahiry/repos/SymReg-L1-Norm/data/csv'
out_path = os.path.join(out_dir, f'l1norm_training_data_b{N_BINS_REBIN}.csv')

os.makedirs(out_dir, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"\n✓ Saved to: {out_path}")
print(f"  Shape:    {df.shape}")
print(f"\nReady to run l1norm_sr.py:")
for b in range(N_BINS_REBIN):
    print(f"  python l1norm_sr.py --bin {b} --n_bins {N_BINS_REBIN} --parameters 3 --seed 0")