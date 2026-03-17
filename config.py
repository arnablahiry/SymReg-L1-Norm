"""
config.py  —  PhySO hyperparameter configuration for Wavelet L1-Norm Symbolic Regression
==================================================================================
All SR performance is highly sensitive to these values.
Tune them to your specific problem (number of input variables, noise level,
expected expression complexity) for best results.

Key tuning targets for the L1-norm / cosmology task:
  - LENGTH_LOC / LENGTH_SCALE  : controls expression complexity tolerance
  - BATCH_SIZE                 : how many candidate expressions per epoch
  - risk_factor                : how aggressively to exploit top expressions
  - entropy_weight             : exploration vs exploitation balance
  - free_const_opti_args       : how well free constants (c0, c1, ...) are fitted
"""

import physo
import torch
import numpy as np

# ==============================================================================
# EXPRESSION COMPLEXITY CONTROLS
# ==============================================================================

# Hard ceiling on expression tree size (number of tokens/nodes).
# Expressions longer than this are ALWAYS rejected regardless of reward.
# 200 is generous — typical good expressions are 5–30 nodes.
# Lower this (e.g. 30) if you want to force simpler formulas.
MAX_LENGTH = 20

# Soft length prior: a Gaussian penalty centered at LENGTH_LOC.
# Expressions shorter than LENGTH_LOC are slightly penalized for being too simple;
# expressions much longer than LENGTH_LOC are heavily penalized.
# → LENGTH_LOC = 100 : very permissive, SR can explore complex expressions freely.
# → For cosmology L1-norm (expected power-law ~5-15 nodes), try LENGTH_LOC = 8–15.
LENGTH_LOC = 10

# Standard deviation of the Gaussian soft length penalty.
# Larger = broader tolerance around LENGTH_LOC (less aggressive penalty for length).
# Smaller = sharper enforcement (expressions must be close to LENGTH_LOC in size).
# → LENGTH_SCALE = 12 is moderate. Try 5–8 for stricter complexity control.
LENGTH_SCALE = 5


# ==============================================================================
# REWARD CONFIG
# ==============================================================================
# The reward function measures how well a candidate expression fits the data.
# It is the core fitness signal driving the reinforcement learning search.

reward_config = {
    # SquashedNRMSE: Normalized Root Mean Squared Error, squashed to [0, 1].
    # NRMSE = RMSE / std(y), so it is scale-independent (good for varying y ranges).
    # Squashing maps reward 0→1 so that terrible fits don't collapse learning.
    # Alternative: physo.physym.reward.SquashedNRMSELog for log-scale targets.
    "reward_function"     : physo.physym.reward.SquashedNRMSE,

    # If True, expressions that violate physical units are given reward=0.
    # Set True only if you provide unit information for X and y.
    # For L1-norm (dimensionless cosmological stat), units not enforced → False.
    "zero_out_unphysical" : False,

    # If True, duplicate expressions (same symbolic form, different constants)
    # receive reward=0 to avoid the RNN memorizing trivial variants.
    # False = allow duplicates (more exploration, less efficiency).
    "zero_out_duplicates" : False,

    # If True, among duplicates, only the simplest (lowest complexity) is kept.
    # Only matters if zero_out_duplicates=True.
    "keep_lowest_complexity_duplicate" : False,
}


# ==============================================================================
# LEARNING CONFIG
# ==============================================================================
# Controls the RNN reinforcement learning loop that generates candidate expressions.

# Number of candidate expressions sampled per epoch.
# Larger = more diverse search per epoch, but slower per epoch.
# 1000 is standard. If you have few CPUs, reduce to 500.
BATCH_SIZE = 1000

# Adam optimizer for the RNN policy network.
# lr=0.0025 is a safe default. Too high → unstable training. Too low → slow convergence.
GET_OPTIMIZER = lambda model: torch.optim.Adam(
    model.parameters(),
    lr=0.0025,
)

learning_config = {
    # ── Batch / sequence settings ──────────────────────────────────────────────

    # How many candidate expressions to sample each epoch (see BATCH_SIZE above).
    'batch_size'       : BATCH_SIZE,

    # Maximum number of tokens in an expression sequence during generation.
    # Must match MAX_LENGTH above. Longer = can express complex formulas.
    'max_time_step'    : MAX_LENGTH,

    # Number of training epochs. Set to a large number; you will stop manually
    # or via max_n_evaluations in the main script.
    'n_epochs'         : int(1e9),

    # ── Loss / gradient settings ────────────────────────────────────────────────

    # Discount factor for reward propagation back through the expression tree.
    # Lower = rewards decay faster toward root (earlier tokens matter less).
    # 0.7 is standard for tree-structured RL. Range: 0.5–0.99.
    'gamma_decay'      : 0.7,

    # Entropy regularization weight — encourages the RNN to explore diverse
    # expressions rather than collapsing to a single formula too early.
    # Higher = more exploration (useful early), lower = more exploitation (useful late).
    # Range: 0.001–0.05. Increase to 0.01–0.02 if SR gets stuck on Omega_m only.
    'entropy_weight'   : 0.005,

    # ── Reward exploitation settings ────────────────────────────────────────────

    # Only the top (risk_factor * 100)% of expressions by reward are used to
    # update the RNN (REINFORCE with baseline). 
    # 0.10 = top 10%. Lower = more selective (faster convergence, risk of premature
    # convergence). Higher = more noisy updates (slower but safer).
    'risk_factor'      : 0.10,

    # Pre-built reward computer using the reward_config above.
    'rewards_computer' : physo.physym.reward.make_RewardsComputer(**reward_config),

    # Optimizer factory defined above.
    'get_optimizer'    : GET_OPTIMIZER,

    # If True, the RNN observes physical units of tokens during generation.
    # Helps constrain search if units are provided. Safe to leave True.
    'observe_units'    : True,
}


# ==============================================================================
# FREE CONSTANT OPTIMIZATION CONFIG
# ==============================================================================
# After the RNN proposes an expression structure (e.g., c0 * Omega_m^c1 * sigma8^c2),
# the free constants (c0, c1, c2, ...) are numerically optimized to best fit the data.
# This is done via gradient-based optimization (L-BFGS by default).

free_const_opti_args = {
    # Loss used to optimize free constants.
    # "MSE" = Mean Squared Error. Good default.
    # "RMSE" or "NRMSE" can also be used for scale-invariant fitting.
    'loss'   : "MSE",

    # Optimization method. L-BFGS is fast and accurate for smooth functions.
    # Alternative: 'Adam' if constants are in a very complex landscape.
    'method' : 'LBFGS',

    'method_args': {
        # Total gradient descent steps for constant optimization per expression.
        # 20 is enough for well-behaved power-law expressions.
        # Increase to 50–100 if constants are slow to converge.
        'n_steps' : 20,

        # Convergence tolerance. Stop early if gradient norm < tol.
        # 1e-8 is tight (accurate). Loosen to 1e-6 if speed is a concern.
        'tol'     : 1e-8,

        'lbfgs_func_args' : {
            # L-BFGS inner iterations per step. 4 is efficient.
            # Increase to 8–10 for very flat/noisy loss landscapes.
            'max_iter'       : 4,

            # Line search algorithm. 'strong_wolfe' ensures stable step sizes
            # even for non-convex constant landscapes. Recommended to keep.
            'line_search_fn' : "strong_wolfe",
        },
    },
}


# ==============================================================================
# PRIORS CONFIG
# ==============================================================================
# Priors constrain the expression search space by penalizing or forbidding
# certain structural patterns. They encode domain knowledge about what
# physically reasonable expressions look like.

priors_config = [

    # ── Structural balance ─────────────────────────────────────────────────────

    # Ensures operators of all arities (unary, binary) are sampled uniformly.
    # Without this, binary operators (mul, add) dominate and unary (exp, log) are rare.
    ("UniformArityPrior", None),

    # ── Length / complexity priors ─────────────────────────────────────────────

    # Hard constraint: expression must have between min_length and max_length tokens.
    # min_length=4 prevents trivial 1-token expressions (e.g., just "c0").
    # max_length=MAX_LENGTH hard-caps complexity (same value as MAX_LENGTH above).
    ("HardLengthPrior", {"min_length": 4, "max_length": MAX_LENGTH}),

    # Soft Gaussian penalty on expression length (see LENGTH_LOC, LENGTH_SCALE above).
    # This is the primary knob for balancing simplicity vs accuracy in SR.
    # → CRITICAL for the Omega_m-only collapse problem: if LENGTH_LOC is too low,
    #   the prior kills any attempt to add sigma_8 or w to the expression.
    #   Try LENGTH_LOC=10, LENGTH_SCALE=5 to force exploration of multi-variable formulas.
    ("SoftLengthPrior", {"length_loc": LENGTH_LOC, "scale": LENGTH_SCALE}),

    # ── Redundancy elimination ─────────────────────────────────────────────────

    # Prevents useless inverse patterns like inv(inv(x)) → x, or div(x, inv(y)).
    # These waste expression budget without adding new information.
    ("NoUselessInversePrior", None),

    # ── Nesting depth constraints ──────────────────────────────────────────────
    # Prevent deeply nested functions that are numerically unstable or physically
    # unreasonable. exp(exp(x)) diverges; log(log(x)) is rarely physical.

    # exp can only appear once in a chain (no exp(exp(x))).
    ("NestedFunctions", {"functions": ["exp"], "max_nesting": 1}),

    # log can only appear once in a chain (no log(log(x))).
    ("NestedFunctions", {"functions": ["log"], "max_nesting": 1}),

    # inv (1/x) can be nested up to 3 levels — allows rational-like expressions.
    # e.g., 1/(1 + 1/x) is valid; deeper chains are penalized.
    ("NestedFunctions", {"functions": ["inv"], "max_nesting": 3}),

    # No nested trigonometry beyond depth 2 (sin(sin(x)) allowed, deeper not).
    # For cosmological statistics, trig is rarely needed — consider removing sin
    # from OP_NAMES entirely if you want to prevent trig expressions.
    ("NestedTrigonometryPrior", {"max_nesting": 2}),

    # Uncomment to enforce physical unit consistency. Requires unit annotations
    # on all X variables and y. Very powerful but requires setup.
    # ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps}),
]


# ==============================================================================
# RNN CELL CONFIG
# ==============================================================================
# Architecture of the RNN that generates expression tokens sequentially.

cell_config = {
    # Number of hidden units in the LSTM cell.
    # 128 is standard. Increase to 256 for very complex target expressions.
    # Decrease to 64 for faster training if expressions are expected to be simple.
    "hidden_size"     : 128,

    # Number of stacked LSTM layers.
    # 1 is usually sufficient. 2 can help for very complex expression spaces.
    "n_layers"        : 1,

    # If True, disables gradient flow through the RNN (for debugging only).
    # Always False for real training.
    "is_lobotomized"  : False,
}


# ==============================================================================
# FINAL RUN CONFIG  —  passed directly to physo.SR()
# ==============================================================================
# Bundles all sub-configs into one dict consumed by the main SR call.

custom_config = {
    "learning_config"      : learning_config,       # RL training loop params
    "reward_config"        : reward_config,          # Fitness function params
    "free_const_opti_args" : free_const_opti_args,  # Constant optimization params
    "priors_config"        : priors_config,          # Expression structure constraints
    "cell_config"          : cell_config,            # RNN architecture params
}
