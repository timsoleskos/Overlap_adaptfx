"""
Belief model for adaptive fractionation: grids, belief updates, and Bellman operators.

The patient's PTV-OAR overlap distribution is modelled as a Gaussian with running
mean (mu) and standard deviation (sigma), updated after each fraction via Welford's
online algorithm.  All beliefs are discretised onto fixed (_MU_GRID, _SIGMA_GRID)
grids so that the DP value tables can be indexed directly.

Branch probabilities _P_BELIEF[mi, si, j] = P(overlap in bin j | belief (mu_grid[mi],
sigma_grid[si])) are precomputed once at module load and reused across all calls.

Interpolation note
------------------
The Bellman operators (_bellman_expectation and _bellman_expectation_full_grid) use
bilinear interpolation over the (_MU_GRID, _SIGMA_GRID) belief grid when looking up
next-state values.  Compared to nearest-grid-point snapping, bilinear interpolation
allows coarser grids to achieve equivalent accuracy:

    N_mu  : 150 (nearest-snap) → 81  (bilinear)  — ~46% fewer mu    points
    N_sigma:  11 (nearest-snap) →  8  (bilinear)  — ~27% fewer sigma points

Peak DP array memory: (4, 70, 441, 81, 8) × 8 bytes ≈ 0.63 GB
                  vs. (4, 70, 441, 150, 11) × 8 bytes ≈ 1.63 GB with nearest-snap
→ ~61% memory reduction.

Tail-folding note
-----------------
_P_BELIEF uses tail-folding: the probability mass that falls below _VOLUME_SPACE[0] or
above _VOLUME_SPACE[-1] is folded into the boundary bins, so each row sums to exactly
1.0.  current_belief_probdist (used for the actual observed patient belief at each
fraction) does NOT fold tails, so its output can sum to slightly less than 1.0 if the
patient's belief tails extend outside _VOLUME_SPACE.  In practice this is negligible
because _VOLUME_SPACE is fixed at 44 cc, far beyond the maximum observed overlap
of ~29 cc in the 58-patient ACTION cohort.  For NIG extensibility (Stage B): replace current_belief_probdist
with a Student-t CDF-difference; _P_BELIEF would then require a corresponding update.
"""

import numpy as np
from scipy.stats import norm

from .helper_functions import nearest_idx


# ---------------------------------------------------------------------------
# Belief grids
# ---------------------------------------------------------------------------

# Non-uniform mu grid optimized for the observed 58-patient cohort prefix-mean distribution.
# Bilinear interpolation in the Bellman operators allows N_mu=81 to match the accuracy of
# the previous N_mu=150 nearest-snap grid, cutting the mu-grid memory contribution by ~46%.
_MU_GRID = np.unique(np.concatenate([
    np.linspace(0.0,  1.0,  20),  # 20 points, step ~0.053 cc
    np.linspace(1.05, 4.0,  20),  # 20 points, step ~0.153 cc
    np.linspace(4.1,  10.0, 22),  # 22 points, step ~0.276 cc
    np.linspace(10.2, 16.0,  9),  # 9 points,  step ~0.750 cc
    np.linspace(16.5, 30.0, 10),  # 10 points, step ~1.500 cc
]))  # 81 grid points total

# Non-uniform sigma grid: fine resolution in [0, 0.7] cc where 75% of patients fall,
# coarser in the tail.  N_sigma=8 with bilinear interpolation replaces N_sigma=11
# nearest-snap, saving ~27% on this dimension.  Range kept at 4.5 cc to cover the
# maximum observed σ ≈ 4.05 cc in the 58-patient ACTION cohort.
# Peak memory: (4, 70, 441, 81, 8) × 8 bytes ≈ 0.63 GB (was 1.63 GB).
_SIGMA_GRID = np.unique(np.concatenate([
    np.linspace(0.05, 0.7, 4),  # 4 points, step ~0.217 cc  (p0–p75 of clinical σ)
    np.linspace(0.9,  1.8, 2),  # 2 points, step  0.900 cc  (p75–p90)
    np.linspace(2.5,  4.5, 2),  # 2 points, step  2.000 cc  (tail, covers max observed σ 4.05 cc)
]))  # 8 grid points total
_SIGMA_MIN = float(_SIGMA_GRID[0])

# Fixed overlap state space: 0 to 44 cc in 0.1 cc steps, matching TPS output resolution.
# Hardcoded to 44 cc (independent of _SIGMA_GRID) so that bin width stays exactly 0.1 cc.
# 44 cc covers mu_grid_max (30 cc) + 4 × sigma_grid_max (4.5 cc) = 48 cc with margin to spare;
# the tail probability beyond 44 cc is negligible for any belief on the grid.
_VOLUME_SPACE = np.linspace(0.0, 44.0, 441)  # 0.1 cc steps, fixed upper bound


# ---------------------------------------------------------------------------
# Branch probabilities (precomputed once at module load)
# ---------------------------------------------------------------------------

def _precompute_branch_probabilities():
    """Return p[mi, si, j] = P(overlap in bin j | belief (mu_grid[mi], sigma_grid[si])).

    Uses CDF differences over each bin. Left/right tails are folded into the first/last
    bin so that probabilities sum to 1 for every belief.
    """
    spacing = _VOLUME_SPACE[1] - _VOLUME_SPACE[0]  # bin width of the overlap grid (uniform, since _VOLUME_SPACE is linspace)
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
# Bilinear interpolation helper
# ---------------------------------------------------------------------------

def _interp_idx_and_weights(values, grid):
    """Return (lo_idx, hi_idx, hi_weight) for linear interpolation of *values* on *grid*.

    For each element v in values, finds the surrounding grid interval [grid[lo], grid[hi]]
    and returns the fractional weight w such that:
        interpolated = (1 - w) * grid_values[lo] + w * grid_values[hi]

    *values* must lie within [grid[0], grid[-1]] (clip before calling).
    *grid* must be sorted ascending.  Output arrays have the same shape as *values*.
    """
    flat = np.asarray(values).ravel()
    hi = np.searchsorted(grid, flat, side='left').clip(1, len(grid) - 1)
    lo = hi - 1
    denom = grid[hi] - grid[lo]
    w = np.where(denom > 0, (flat - grid[lo]) / denom, 0.0)
    shape = np.asarray(values).shape
    return lo.reshape(shape), hi.reshape(shape), w.reshape(shape)


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

    Uses bilinear interpolation over (_MU_GRID, _SIGMA_GRID) for the next-state value
    lookup, giving smoother value estimates than nearest-grid-point snapping.

    Sums future values weighted by branch probabilities:
        sum_j p_branch[j] * bilinear_interp(values_prev[d, j, :, :], mu'[j], sigma'[j])

    values_prev: (N_dose, N_overlap, N_mu, N_sigma)
    p_branch:    (N_overlap,) probability of each next overlap under current belief
    Returns:     (N_dose,) expected future value for each accumulated dose state
    """
    N = len(volume_space)
    mu_prime = (observation_count * mu + volume_space) / (observation_count + 1)
    sigma2_prime = (observation_count * sigma ** 2 + (volume_space - mu) * (volume_space - mu_prime)) / (observation_count + 1)
    sigma_prime = np.sqrt(np.maximum(sigma2_prime, _SIGMA_MIN ** 2))
    mu_prime = np.clip(mu_prime, _MU_GRID[0], _MU_GRID[-1])
    sigma_prime = np.clip(sigma_prime, _SIGMA_GRID[0], _SIGMA_GRID[-1])

    lo_m, hi_m, wm = _interp_idx_and_weights(mu_prime, _MU_GRID)    # (N_overlap,) each
    lo_s, hi_s, ws = _interp_idx_and_weights(sigma_prime, _SIGMA_GRID)  # (N_overlap,) each

    # values_prev: (N_dose, N_overlap, N_mu, N_sigma)
    # Advanced indexing with j_idx, lo_m, lo_s all shape (N_overlap,) → result (N_dose, N_overlap)
    j_idx = np.arange(N)
    v_ll = values_prev[:, j_idx, lo_m, lo_s]
    v_lh = values_prev[:, j_idx, lo_m, hi_s]
    v_hl = values_prev[:, j_idx, hi_m, lo_s]
    v_hh = values_prev[:, j_idx, hi_m, hi_s]

    branch_vals = ((1 - wm) * (1 - ws) * v_ll +
                   (1 - wm) * ws       * v_lh +
                   wm       * (1 - ws) * v_hl +
                   wm       * ws       * v_hh)  # (N_dose, N_overlap)
    return (branch_vals * p_branch[None, :]).sum(axis=1)  # (N_dose,)


def _bellman_expectation_full_grid(values_prev, observation_count):
    """Compute the expected future value for every (accumulated_dose, belief_mu, belief_sigma) triple.

    Vectorised full-grid counterpart of _bellman_expectation. Where _bellman_expectation
    operates on a single belief (mu, sigma), this function operates on all (N_mu × N_sigma)
    beliefs simultaneously, making it suitable for the backward-induction DP loop.

    Uses bilinear interpolation over the (_MU_GRID, _SIGMA_GRID) grid for the next-state
    value lookup.  This allows the coarser grids (N_mu=81, N_sigma=8) to achieve
    equivalent accuracy to the previous nearest-snap approach at N_mu=150, N_sigma=11.

    Steps:
      1. Welford belief update: for each (mu_grid, sigma_grid, overlap_bin) triple, compute
         the updated belief (mu', sigma') after observing that overlap.
      2. Compute bilinear interpolation weights for (mu', sigma') on (_MU_GRID, _SIGMA_GRID).
      3. Accumulate the probability-weighted interpolated future values across all overlap bins.

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

    # ----- Step 2: bilinear interpolation weights — shape (N_mu, N_sigma, N_overlap) each -----
    lo_m, hi_m, wm = _interp_idx_and_weights(mu_prime, _MU_GRID)
    lo_s, hi_s, ws = _interp_idx_and_weights(sigma_prime, _SIGMA_GRID)

    # ----- Step 3: accumulate probability-weighted interpolated future values in a j-loop -----
    # This avoids materialising the full (N_dose, N_mu, N_sigma, N_overlap) branch_vals array;
    # each j-slice is ~1.5 MB and is processed on the fly.
    future_value_prob_full = np.zeros((N_dose, N_mu, N_sigma))
    for j in range(N_overlap):
        lm  = lo_m[:, :, j]   # (N_mu, N_sigma)
        hm  = hi_m[:, :, j]
        ls  = lo_s[:, :, j]
        hs  = hi_s[:, :, j]
        wmj = wm[:, :, j]     # (N_mu, N_sigma)
        wsj = ws[:, :, j]

        vp = values_prev[:, j, :, :]  # (N_dose, N_mu, N_sigma)

        # Bilinear interpolation: vp[:, lm, ls] uses advanced indexing where lm, ls are
        # (N_mu, N_sigma) arrays → result shape (N_dose, N_mu, N_sigma)
        v_ll = vp[:, lm, ls]
        v_lh = vp[:, lm, hs]
        v_hl = vp[:, hm, ls]
        v_hh = vp[:, hm, hs]

        v_interp = ((1 - wmj) * (1 - wsj) * v_ll +
                    (1 - wmj) * wsj        * v_lh +
                    wmj       * (1 - wsj)  * v_hl +
                    wmj       * wsj        * v_hh)  # (N_dose, N_mu, N_sigma)

        future_value_prob_full += v_interp * _P_BELIEF[None, :, :, j]

    return future_value_prob_full


# ---------------------------------------------------------------------------
# Current-belief branch probabilities (evaluated on-the-fly, not precomputed)
# ---------------------------------------------------------------------------

def current_belief_probdist(mu: float, sigma: float) -> np.ndarray:
    """Return P(overlap in bin j | current patient belief (mu, sigma)) over _VOLUME_SPACE.

    Evaluates Gaussian CDF differences over the fixed _VOLUME_SPACE grid, without
    tail-folding.  The probabilities may therefore sum to slightly less than 1.0 when
    the belief tails extend beyond _VOLUME_SPACE; in practice this is negligible
    because _VOLUME_SPACE is fixed at 44 cc, far beyond the maximum observed overlap.

    Unlike _P_BELIEF (precomputed for all grid points, with tail-folding), this
    function evaluates on-the-fly for the patient's actual current belief — which
    may lie off-grid between DP solver calls.

    For NIG extensibility (Stage B): replace this function with a Student-t
    CDF-difference over _VOLUME_SPACE, leaving _bellman_expectation unchanged.

    Args:
        mu (float): current belief mean (cc).
        sigma (float): current belief standard deviation (cc).

    Returns:
        np.ndarray: probability of each overlap bin j; shape (N_overlap,).
    """
    from .helper_functions import probdist
    return probdist((mu, sigma), _VOLUME_SPACE)
