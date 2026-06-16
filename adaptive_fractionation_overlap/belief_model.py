"""
Belief model for adaptive fractionation: grids, belief updates, and Bellman operators.

The patient's PTV-OAR overlap distribution is modelled as a Gaussian with running
mean (mu) and empirical variance (S^2).  The DP state stores (mu, S^2) on fixed
(_MU_GRID, _S2_GRID) grids. Predictive overlap probabilities are Gaussian with
sigma = sigma_MAP(n, S^2), i.e. sigma is derived from the current sample count,
empirical variance, and prior via the same MAP rule used by std_calc.

Branch probabilities therefore depend on observation_count as well as the prior
parameters. _P_BELIEF is retained only as the default table for
observation_count=1 and the default prior; the solver itself uses
observation-count-specific branch tables generated on demand.

Tail-folding note
-----------------
Branch-probability tables fold the probability mass that falls below
_VOLUME_SPACE[0] or above _VOLUME_SPACE[-1] into the boundary bins, so each row
sums to exactly 1.0. current_belief_probdist applies the same boundary
tail-folding for the actual observed patient belief at each fraction.
"""

from functools import lru_cache

import numpy as np
from scipy.stats import norm

from .constants import DEFAULT_ALPHA, DEFAULT_BETA
from .helper_functions import nearest_idx, sigma_map_from_variance, welford_update


# ---------------------------------------------------------------------------
# Belief grids
# ---------------------------------------------------------------------------

# Non-uniform mu grid optimized for the observed 58-patient cohort prefix-mean distribution
# under a 150-point budget. A sweep over N_mu in [50, 500] showed quality saturates at ~150:
# above N_mu=150 the benefit fluctuates within ±0.005 ccGy (quantisation noise), making
# N_mu=150 the principled choice — 99.95% of N=500 benefit at 2.3× lower compute than N_mu=280.
_MU_GRID = np.unique(np.concatenate([
    np.linspace(0.0, 1.0, 38),   # 38 points, step ~0.0270 cc
    np.linspace(1.05, 4.0, 39),  # 39 points, step ~0.0776 cc
    np.linspace(4.1, 10.0, 42),  # 42 points, step ~0.1439 cc
    np.linspace(10.2, 16.0, 15), # 15 points, step ~0.4143 cc
    np.linspace(16.5, 30.0, 16), # 16 points, step ~0.9000 cc
]))  # 150 grid points total

# The DP now stores empirical variance S^2 rather than sigma directly. We retain the previous
# sigma-grid coverage by squaring those reference sigma knots, and prepend an exact 0.0 state
# so repeated identical overlaps can remain on-grid without artificial inflation.
_S2_GRID = np.unique(np.concatenate([
    [0.0],
    np.linspace(0.05, 0.7, 5) ** 2,
    np.linspace(0.8, 1.8, 3) ** 2,
    np.linspace(2.0, 4.5, 3) ** 2,
]))  # 12 grid points total

# Fixed overlap state space: 0 to 44 cc in 0.1 cc steps, matching TPS output resolution.
# Hardcoded to 44 cc so bin width stays exactly 0.1 cc.
_VOLUME_SPACE = np.linspace(0.0, 44.0, 441)  # 0.1 cc steps, fixed upper bound

# Minimum sigma used when computing branch probabilities. Mirrors the floor from the
# pre-MAP code (sqrt(S^2) was clamped to _SIGMA_MIN before snapping to the sigma grid).
# Without this, sigma_MAP(n>=2, S^2=0.0) collapses to ~0.001 cc because the log-kernel
# diverges as sigma->0 when alpha-n < 0, turning branch probabilities into a near-point-mass.
_SIGMA_MIN = 0.05  # cc


# ---------------------------------------------------------------------------
# Branch probabilities (observation-count specific)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _sigma_map_for_s2_grid(observation_count: int, alpha: float, beta: float):
    """Return sigma_MAP(n, S^2) for every S^2 grid point."""
    return sigma_map_from_variance(_S2_GRID, observation_count, alpha, beta)


@lru_cache(maxsize=None)
def _branch_probabilities_for_grid(observation_count: int, alpha: float, beta: float):
    """Return p[mi, vi, j] = P(overlap in bin j | belief (mu_grid[mi], S2_grid[vi]))."""
    spacing = _VOLUME_SPACE[1] - _VOLUME_SPACE[0]
    upper_bounds = _VOLUME_SPACE + spacing / 2
    lower_bounds = _VOLUME_SPACE - spacing / 2
    sigma_grid = np.maximum(_sigma_map_for_s2_grid(observation_count, alpha, beta), _SIGMA_MIN)
    probabilities = (
        norm.cdf(upper_bounds[None, None, :], loc=_MU_GRID[:, None, None], scale=sigma_grid[None, :, None])
        - norm.cdf(lower_bounds[None, None, :], loc=_MU_GRID[:, None, None], scale=sigma_grid[None, :, None])
    )
    probabilities[:, :, 0] += norm.cdf(lower_bounds[0], loc=_MU_GRID[:, None], scale=sigma_grid[None, :])
    probabilities[:, :, -1] += 1.0 - norm.cdf(upper_bounds[-1], loc=_MU_GRID[:, None], scale=sigma_grid[None, :])
    return probabilities


# Retained for tests/debugging only. The solver uses _branch_probabilities_for_grid(...)
# with the actual observation_count and prior parameters in each DP call.
_P_BELIEF = _branch_probabilities_for_grid(1, DEFAULT_ALPHA, DEFAULT_BETA)


# ---------------------------------------------------------------------------
# Belief update (Welford on S^2) and Bellman operators
# ---------------------------------------------------------------------------

def _hypothetical_belief_grid_indices(mu, s2, volume_space, observation_count):
    """Map hypothetical next observations to the nearest (_MU_GRID, _S2_GRID) indices.

    The state update is Welford on empirical variance:

        mu'  = (n * mu + x) / (n + 1)
        S'^2 = (n * S^2 + (x - mu) * (x - mu')) / (n + 1)

    sigma itself is not updated directly here; it is derived later from (n, S^2)
    when branch probabilities are needed.
    """
    mu_prime, s2_prime = welford_update(mu, s2, observation_count, volume_space)
    mu_prime = np.clip(mu_prime, _MU_GRID[0], _MU_GRID[-1])
    s2_prime = np.clip(s2_prime, _S2_GRID[0], _S2_GRID[-1])
    next_mu_index = nearest_idx(mu_prime, _MU_GRID)
    next_s2_index = nearest_idx(s2_prime, _S2_GRID)
    return next_mu_index, next_s2_index


def _bellman_expectation(values_prev, volume_space, p_branch, mu, s2, observation_count):
    """Bellman expectation over all hypothetical next overlaps for a single belief (mu, S^2).

    Sums future values weighted by branch probabilities:
        sum_j p_branch[j] * values_prev[d, j, next_mi[j], next_vi[j]]

    values_prev: (N_dose, N_overlap, N_mu, N_s2)
    p_branch:    (N_overlap,) probability of each next overlap under current belief
    Returns:     (N_dose,) expected future value for each accumulated dose state
    """
    next_mi, next_vi = _hypothetical_belief_grid_indices(mu, s2, volume_space, observation_count)
    branch_vals = values_prev[:, np.arange(len(volume_space)), next_mi, next_vi]
    return (branch_vals * p_branch[None, :]).sum(axis=1)


def _bellman_expectation_full_grid(values_prev, observation_count, alpha, beta):
    """Compute the expected future value for every (accumulated_dose, belief_mu, belief_S^2) triple.

    For each current belief (mu, S^2):
      1. Convert S^2 to sigma_MAP(n, S^2) for the current observation_count to obtain
         branch probabilities over the next overlap.
      2. Apply the Welford update on (mu, S^2) for every hypothetical next overlap.
      3. Snap the updated (mu', S'^2) to the nearest belief-grid indices.
      4. Accumulate probability-weighted future values across all overlap bins.

    Args:
        values_prev: value function of the next DP state; shape (N_dose, N_overlap, N_mu, N_s2).
        observation_count (int): number of overlap observations already made at this fraction.
        alpha, beta (float): prior parameters used by sigma_MAP(n, S^2).

    Returns:
        np.ndarray: expected future value for each (dose, mu, S^2); shape (N_dose, N_mu, N_s2).
    """
    N_dose, N_overlap = values_prev.shape[0], len(_VOLUME_SPACE)
    N_mu, N_s2 = len(_MU_GRID), len(_S2_GRID)

    mu_vals = _MU_GRID[:, None, None]   # (N_mu, 1, 1)
    s2_vals = _S2_GRID[None, :, None]   # (1, N_s2, 1)
    o_vals = _VOLUME_SPACE[None, None, :]  # (1, 1, N_overlap)

    mu_prime, s2_prime = welford_update(mu_vals, s2_vals, observation_count, o_vals)
    mu_prime = np.clip(mu_prime, _MU_GRID[0], _MU_GRID[-1])
    s2_prime = np.clip(s2_prime, _S2_GRID[0], _S2_GRID[-1])

    next_mi = nearest_idx(mu_prime, _MU_GRID)
    next_vi = nearest_idx(s2_prime, _S2_GRID)

    p_branch = _branch_probabilities_for_grid(observation_count, alpha, beta).astype(values_prev.dtype, copy=False)

    next_b = next_mi * N_s2 + next_vi
    vals_flat = values_prev.reshape(N_dose, N_overlap, -1)
    future_value_prob_full = np.zeros((N_dose, N_mu, N_s2), dtype=values_prev.dtype)
    for j in range(N_overlap):
        future_value_prob_full += vals_flat[:, j, :][:, next_b[:, :, j]] * p_branch[None, :, :, j]

    return future_value_prob_full


# ---------------------------------------------------------------------------
# Current-belief branch probabilities (evaluated on-the-fly, not precomputed)
# ---------------------------------------------------------------------------

def current_belief_probdist(mu: float, sigma: float) -> np.ndarray:
    """Return P(overlap in bin j | current patient belief (mu, sigma)) over _VOLUME_SPACE.

    Evaluates Gaussian CDF differences over the fixed _VOLUME_SPACE grid and
    folds probability mass outside the supported overlap range into the boundary
    bins. The returned probabilities therefore sum to 1.0, matching the branch
    probability tables used by the DP solver.

    Args:
        mu (float): current belief mean (cc).
        sigma (float): current predictive standard deviation (cc).

    Returns:
        np.ndarray: probability of each overlap bin j; shape (N_overlap,).
    """
    sigma = max(sigma, _SIGMA_MIN)
    spacing = _VOLUME_SPACE[1] - _VOLUME_SPACE[0]
    upper_bounds = _VOLUME_SPACE + spacing / 2
    lower_bounds = _VOLUME_SPACE - spacing / 2
    probabilities = norm.cdf(upper_bounds, loc=mu, scale=sigma) - norm.cdf(lower_bounds, loc=mu, scale=sigma)
    probabilities[0] += norm.cdf(lower_bounds[0], loc=mu, scale=sigma)
    probabilities[-1] += 1.0 - norm.cdf(upper_bounds[-1], loc=mu, scale=sigma)
    return probabilities
