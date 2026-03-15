"""
Belief model for adaptive fractionation: grids, belief updates, and Bellman operators.

The patient's PTV-OAR overlap distribution is modelled as a Gaussian with running
mean (mu) and standard deviation (sigma), updated after each fraction via Welford's
online algorithm.  All beliefs are discretised onto fixed (_MU_GRID, _SIGMA_GRID)
grids so that the DP value tables can be indexed directly.

Branch probabilities _P_BELIEF[mi, si, j] = P(overlap in bin j | belief (mu_grid[mi],
sigma_grid[si])) are precomputed once at module load and reused across all calls.
"""

import numpy as np
from scipy.stats import norm

from .helper_functions import nearest_idx


# ---------------------------------------------------------------------------
# Belief grids
# ---------------------------------------------------------------------------

# Non-uniform mu grid optimized for the observed 58-patient cohort prefix-mean distribution
# under a fixed 280-point budget.
_MU_GRID = np.unique(np.concatenate([
    np.linspace(0.0,  1.0,  70),  # 70 points, step ~0.0145 cc
    np.linspace(1.05, 4.0,  73),  # 73 points, step ~0.0410 cc
    np.linspace(4.1,  10.0, 79),  # 79 points, step ~0.0756 cc
    np.linspace(10.2, 16.0, 28),  # 28 points, step ~0.2148 cc
    np.linspace(16.5, 30.0, 30),  # 30 points, step ~0.4655 cc
]))  # 280 grid points total

# Non-uniform sigma grid: fine resolution in [0, 0.7] cc where 75% of patients fall,
# coarser in the tail.  Range extended to 3.5 cc to avoid clipping outlier patients
# (observed max σ ≈ 3.3 cc on the 58-patient cohort).
# Peak memory: (4, 70, 441, 280, 10) × 8 bytes ≈ 2.75 GB.
_SIGMA_GRID = np.unique(np.concatenate([
    np.linspace(0.05, 0.7, 5),  # step ~0.163 cc  (p0–p75 of clinical σ)
    np.linspace(0.8,  1.8, 3),  # step ~0.500 cc  (p75–p90)
    np.linspace(2.0,  3.5, 2),  # step ~1.500 cc  (tail)
]))  # 10 grid points total
_SIGMA_MIN = float(_SIGMA_GRID[0])

# Fixed overlap state space covering the full belief grid (0 to max_mu + 4·max_sigma), shared across all beliefs.
_VOLUME_SPACE = np.linspace(0.0, _MU_GRID[-1] + 4 * _SIGMA_GRID[-1], 441)  # 0.1cc steps (matching TPS volume discretisation)


# ---------------------------------------------------------------------------
# Branch probabilities (precomputed once at module load)
# ---------------------------------------------------------------------------

def _precompute_branch_probabilities():
    """Return p[mi, si, j] = P(overlap in bin j | belief (mu_grid[mi], sigma_grid[si])).

    Uses CDF differences over each bin. Left/right tails are folded into the first/last
    bin so that probabilities sum to 1 for every belief.
    """
    spacing = _VOLUME_SPACE[1] - _VOLUME_SPACE[0]  # bin width of the overlap grid (varies along the grid)
    upper_bounds = _VOLUME_SPACE + spacing / 2
    lower_bounds = _VOLUME_SPACE - spacing / 2
    # P(bin j | mu, sigma) = CDF(upper_bounds[j]) - CDF(lower_bounds[j]), broadcast over all (mu, sigma) pairs
    probabilities = (
        norm.cdf(upper_bounds[None, None, :], loc=_MU_GRID[:, None, None], scale=_SIGMA_GRID[None, :, None])
        - norm.cdf(lower_bounds[None, None, :], loc=_MU_GRID[:, None, None], scale=_SIGMA_GRID[None, :, None])
    )
    # Fold left and right tails into the boundary bins so each row sums to 1
    probabilities[:, :, 0] += norm.cdf(lower_bounds[0], loc=_MU_GRID[:, None], scale=_SIGMA_GRID[None, :])
    probabilities[:, :, -1] += 1.0 - norm.cdf(upper_bounds[-1], loc=_MU_GRID[:, None], scale=_SIGMA_GRID[None, :])
    return probabilities

_P_BELIEF = _precompute_branch_probabilities()  # shape: (N_mu, N_sigma, N_overlap), computed once at module load


# ---------------------------------------------------------------------------
# Belief update (Welford) and Bellman operators
# ---------------------------------------------------------------------------

def _hypothetical_belief_grid_indices(mu, sigma, volume_space, observation_count):
    """For each hypothetical next overlap in volume_space, compute where the updated
    belief (mu, sigma) would land on the (_MU_GRID, _SIGMA_GRID) grid.

    Returns next_mi, next_si: (N_overlap,) integer index arrays into _MU_GRID and
    _SIGMA_GRID, used to look up future values in the DP values array.
    Uses Welford's online algorithm to update the running mean and variance.
    """
    mu_prime = (observation_count * mu + volume_space) / (observation_count + 1)  # new mean after observing each hypothetical overlap
    sigma2_prime = (observation_count * sigma ** 2 + (volume_space - mu) * (volume_space - mu_prime)) / (observation_count + 1)  # new variance after observing each hypothetical overlap
    sigma_prime = np.sqrt(np.maximum(sigma2_prime, _SIGMA_MIN ** 2))  # new sigma, clamped to avoid zero with few observations
    mu_prime = np.clip(mu_prime, _MU_GRID[0], _MU_GRID[-1])          # clamp new mu to grid's range
    sigma_prime = np.clip(sigma_prime, _SIGMA_GRID[0], _SIGMA_GRID[-1])  # clamp new sigma to grid's range
    next_mu_index = nearest_idx(mu_prime, _MU_GRID)        # nearest mu grid index for each hypothetical next overlap
    next_sigma_index = nearest_idx(sigma_prime, _SIGMA_GRID)  # nearest sigma grid index for each hypothetical next overlap
    return next_mu_index, next_sigma_index  # indices rather than values so callers can directly index into the DP values array


def _bellman_expectation(values_prev, volume_space, p_branch, mu, sigma, observation_count):
    """Bellman expectation over all hypothetical next overlaps for a single belief (mu, sigma).

    Sums future values weighted by branch probabilities:
        sum_j p_branch[j] * values_prev[d, j, next_mi[j], next_si[j]]

    values_prev: (N_dose, N_overlap, N_mu, N_sigma)
    p_branch:    (N_overlap,) probability of each next overlap under current belief
    Returns:     (N_dose,) expected future value for each accumulated dose state
    """
    next_mi, next_si = _hypothetical_belief_grid_indices(mu, sigma, volume_space, observation_count) # If overlap was j, where would the belief (mu, sigma) land on the grid?
    branch_vals = values_prev[:, np.arange(len(volume_space)), next_mi, next_si]  # Value function of the next belief state
    return (branch_vals * p_branch[None, :]).sum(axis=1) # Probability-weighed sum of next states' value functions


def _bellman_expectation_full_grid(values_prev, observation_count):
    """Compute the expected future value for every (accumulated_dose, belief_mu, belief_sigma) triple.

    Vectorised full-grid counterpart of _bellman_expectation. Where _bellman_expectation
    operates on a single belief (mu, sigma), this function operates on all (N_mu × N_sigma)
    beliefs simultaneously, making it suitable for the backward-induction DP loop.

    Steps:
      1. Welford belief update: for each (mu_grid, sigma_grid, overlap_bin) triple, compute
         the updated belief (mu', sigma') after observing that overlap.
      2. Map (mu', sigma') to the nearest grid indices (next_mi, next_si).
      3. Accumulate the probability-weighted future values across all overlap bins.

    Args:
        values_prev: value function of the next DP state; shape (N_dose, N_overlap, N_mu, N_sigma).
        observation_count (int): number of overlap observations already made at this fraction
                                 (used as the sample count n in the Welford update formula).

    Returns:
        np.ndarray: expected future value for each (dose, mu, sigma); shape (N_dose, N_mu, N_sigma).
    """
    N_dose, N_overlap, N_mu, N_sigma = values_prev.shape[0], len(_VOLUME_SPACE), len(_MU_GRID), len(_SIGMA_GRID)

    # ----- Step 1: belief transitions for all (mi, si, j) at this observation_count -----

    # Expand grids into 3D arrays so they can broadcast together:
    mu_vals = _MU_GRID[:, None, None]        # (N_mu, 1, 1)
    sigma_vals = _SIGMA_GRID[None, :, None]  # (1, N_sigma, 1)
    o_vals = _VOLUME_SPACE[None, None, :]    # (1, 1, N_overlap)

    mu_prime = (observation_count * mu_vals + o_vals) / (observation_count + 1)  # new mean after observing each hypothetical overlap
    delta = o_vals - mu_vals         # deviation of new observation from old mean
    delta_prime = o_vals - mu_prime  # deviation of new observation from new mean (Welford variance term)
    sigma2_prime = (observation_count * sigma_vals ** 2 + delta * delta_prime) / (observation_count + 1)  # new variance (Welford online update)
    sigma_prime = np.sqrt(np.maximum(sigma2_prime, _SIGMA_MIN ** 2))  # new std, floored at _SIGMA_MIN
    mu_prime = np.clip(mu_prime, _MU_GRID[0], _MU_GRID[-1])
    sigma_prime = np.clip(sigma_prime, _SIGMA_GRID[0], _SIGMA_GRID[-1])
    # Nearest grid indices — shape (N_mu, N_sigma, N_overlap)
    # searchsorted is O(N*log(G)) vs argmin's O(N*G), ~35x faster for G=280.
    next_mi = nearest_idx(mu_prime, _MU_GRID)
    next_si = nearest_idx(sigma_prime, _SIGMA_GRID)

    # ----- Steps 2+3: accumulate probability-weighted future values in a j-loop -----
    # This avoids materialising the full (N_dose, N_mu, N_sigma, N_overlap) = ~150 MB
    # branch_vals array; instead each j-slice is ~0.75 MB and is processed on the fly.
    next_b = next_mi * N_sigma + next_si  # flat belief index, shape (N_mu, N_sigma, N_overlap)
    vals_flat = values_prev.reshape(N_dose, N_overlap, -1)  # (N_dose, N_overlap, N_belief)
    future_value_prob_full = np.zeros((N_dose, N_mu, N_sigma))
    for j in range(N_overlap):
        # vals_flat[:, j, next_b[:, :, j]]: (N_dose, N_mu, N_sigma) — one j-slice
        future_value_prob_full += vals_flat[:, j, :][:, next_b[:, :, j]] * _P_BELIEF[None, :, :, j]

    return future_value_prob_full
