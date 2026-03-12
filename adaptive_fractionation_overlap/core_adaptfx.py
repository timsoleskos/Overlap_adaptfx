"""
Core solvers for adaptive fractionation dose optimisation.

Public API
----------
adaptive_fractionation_core : belief-state DP solver (Stage A); call this at each fraction.
adaptfx_full                : convenience wrapper that simulates a complete treatment plan.
precompute_plan             : pre-tabulates dose recommendations for every possible overlap
                              volume, so clinical staff can prepare before imaging.

Algorithm
---------
At each treatment fraction the solver runs backward-induction dynamic programming over a
5D state space: (accumulated_dose, overlap_volume, belief_mu, belief_sigma, fraction).
The belief (mu, sigma) tracks a running Gaussian estimate of the patient's overlap
distribution; it is updated after each observed overlap via Welford's online algorithm.
Because the belief grid is fixed at module load, all belief-branch probabilities are
precomputed once (_P_BELIEF) and reused across every call.
"""

__all__ = [  # limits what `from core_adaptfx import *` exposes
    "adaptive_fractionation_core",
    "adaptfx_full",
    "precompute_plan",
]

import numpy as np
import pandas as pd
from scipy.stats import norm

from .constants import (
    DEFAULT_MIN_DOSE, 
    DEFAULT_MAX_DOSE, 
    DEFAULT_MEAN_DOSE,
    DEFAULT_DOSE_STEPS, 
    DEFAULT_NUMBER_OF_FRACTIONS,
    DEFAULT_ALPHA,
    DEFAULT_BETA
)

from .helper_functions import (
    std_calc,
    get_state_space,
    probdist,
    max_action,
    penalty_calc_single,
    penalty_calc_matrix,
    min_dose_to_deliver,
    linear_interp,
    nearest_idx,
)

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

# Fixed overlap state space covering the full belief grid (0 to max_mu + 4·max_sigma),
# shared across all beliefs. A patient-specific range would be too narrow to represent
# future outcomes for beliefs far from the current observation, corrupting the DP.
_VOLUME_SPACE = np.linspace(0.0, _MU_GRID[-1] + 4 * _SIGMA_GRID[-1], 441)  # 0.1 cc steps

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


def _hypothetical_belief_grid_indices(mu, sigma, volume_space, n_t):
    """For each hypothetical next overlap in volume_space, compute where the updated
    belief (mu, sigma) would land on the (_MU_GRID, _SIGMA_GRID) grid.

    Returns next_mi, next_si: (N_overlap,) integer index arrays into _MU_GRID and
    _SIGMA_GRID, used to look up future values in the DP values array.
    Uses Welford's online algorithm to update the running mean and variance.
    """
    mu_prime = (n_t * mu + volume_space) / (n_t + 1)  # new mean after observing each hypothetical overlap
    sigma2_prime = (n_t * sigma ** 2 + (volume_space - mu) * (volume_space - mu_prime)) / (n_t + 1)  # new variance after observing each hypothetical overlap
    sigma_prime = np.sqrt(np.maximum(sigma2_prime, _SIGMA_MIN ** 2))  # new sigma, clamped to avoid zero with few observations
    mu_prime = np.clip(mu_prime, _MU_GRID[0], _MU_GRID[-1])          # clamp new mu to grid's range
    sigma_prime = np.clip(sigma_prime, _SIGMA_GRID[0], _SIGMA_GRID[-1])  # clamp new sigma to grid's range
    next_mi = nearest_idx(mu_prime, _MU_GRID)        # nearest mu grid index for each hypothetical next overlap
    next_si = nearest_idx(sigma_prime, _SIGMA_GRID)  # nearest sigma grid index for each hypothetical next overlap
    return next_mi, next_si  # indices rather than values so callers can directly index into the DP values array


def _bellman_expectation(values_prev, volume_space, p_branch, mu, sigma, n_t):
    """Bellman expectation over all hypothetical next overlaps for a single belief (mu, sigma).

    Sums future values weighted by branch probabilities:
        sum_j p_branch[j] * values_prev[d, j, next_mi[j], next_si[j]]

    values_prev: (N_dose, N_overlap, N_mu, N_sigma)
    p_branch:    (N_overlap,) probability of each next overlap under current belief
    Returns:     (N_dose,) expected future value for each accumulated dose state
    """
    next_mi, next_si = _hypothetical_belief_grid_indices(mu, sigma, volume_space, n_t) # If overlap was j, where would the belief (mu, sigma) land on the grid?
    branch_vals = values_prev[:, np.arange(len(volume_space)), next_mi, next_si]  # Value function of the next belief state
    return (branch_vals * p_branch[None, :]).sum(axis=1) # Probability-weighed sum of next states' value functions


def _set_infeasible_state(fixed_dose, values, N_overlap, remaining_fractions):
    """Set DP arrays to reflect an infeasible treatment plan.

    Called when the prescribed dose is unreachable or unavoidably exceeded.
    Marks all future states as invalid and returns a constant policy and sentinel value.
    """
    current_fraction_policy = np.ones(N_overlap) * fixed_dose
    if remaining_fractions > 1:  # skip on last fraction: no future fractions exist, values array is empty
        values[:] = -1e12  # mark all future states as invalid — DP is not run in infeasible cases
    actual_value = np.ones(1) * -1e12  # signal "infeasible case" to downstream code
    return current_fraction_policy, actual_value


def adaptive_fractionation_core(fraction: int, volumes: np.ndarray, accumulated_dose: float, number_of_fractions: int = DEFAULT_NUMBER_OF_FRACTIONS, min_dose: float = DEFAULT_MIN_DOSE, max_dose: float = DEFAULT_MAX_DOSE, mean_dose:float = DEFAULT_MEAN_DOSE, dose_steps: float = DEFAULT_DOSE_STEPS, alpha: float = DEFAULT_ALPHA, beta:float = DEFAULT_BETA):
    """Belief-state DP solver. Computes the recommended dose for the current fraction.

    Optimizes fractionation by minimizing expected PTV underdosage cost: lower dose
    when PTV-OAR overlap is large, higher dose when overlap is small.

    Args:
        fraction (int): current fraction number (1-indexed).
        volumes (np.ndarray): all overlap volumes observed so far, including the current fraction.
        accumulated_dose (float): total physical dose delivered to PTV before this fraction (Gy).
        number_of_fractions (int, optional): total number of fractions. Defaults to 5.
        min_dose (float, optional): minimum physical dose per fraction (Gy). Defaults to 7.5.
        max_dose (float, optional): maximum physical dose per fraction (Gy). Defaults to 9.5.
        mean_dose (float, optional): prescribed mean dose per fraction (Gy). Defaults to 8.
        dose_steps (float, optional): dose grid resolution (Gy). Defaults to 0.1.
        alpha (float, optional): shape parameter of the gamma prior on sigma. Defaults to 1.838.
        beta (float, optional): scale parameter of the gamma prior on sigma. Defaults to 0.265.

    Returns:
        list: [policies, current_fraction_policy, volume_space, physical_dose, penalty_added,
               values, dose_space, initial_probs, final_penalty]
    """
    prescribed_dose = number_of_fractions * mean_dose  # total prescribed dose (Gy)
    observed_overlap = volumes[-1]  # overlap observed at the current fraction
    min_accumulated_dose_after_fraction = accumulated_dose + min_dose  # minimum possible accumulated dose after this fraction

    # --- Current patient belief from observed volumes ---
    initial_belief_mu = volumes.mean()
    initial_belief_sigma = std_calc(volumes, alpha, beta)
     # Next, set precomputed branch probabilities: p_belief[mi, si, j] = P(overlap bin j | belief (mu_grid[mi], sigma_grid[si]))
    p_belief = _P_BELIEF  # shape: (N_mu, N_sigma, N_overlap)

    # The DP uses a fixed space, with p_belief summing to ~1 for every belief in the grid.
    # Future overlap outcomes are correctly represented regardless of the current belief.
    volume_space = _VOLUME_SPACE
    initial_probs = probdist((initial_belief_mu, initial_belief_sigma), volume_space)

    # --- State and action spaces ---
    dose_space = np.arange(min_accumulated_dose_after_fraction, prescribed_dose, dose_steps)  # reachable accumulated PTV doses from this fraction onwards
    overdose_sentinel = prescribed_dose + 0.05  # any accumulated PTV dose mapped here represents an overdose; value just needs to be outside the valid dose range
    dose_space = np.concatenate((dose_space, [prescribed_dose, overdose_sentinel]))  # prescribed_dose appended explicitly: arange excludes stop value and float steps can miss it
    action_space = np.arange(min_dose, max_dose + 0.01, dose_steps)  # deliverable doses per fraction; +0.01 ensures max_dose is included

    # --- Dimension shorthands ---
    N_mu = len(_MU_GRID)
    N_sigma = len(_SIGMA_GRID)
    N_overlap = len(volume_space)
    N_dose = len(dose_space)

    remaining_ptv_dose = prescribed_dose - accumulated_dose
    remaining_fractions = number_of_fractions + 1 - fraction

    # --- Value Function and Policy Tables (5D: one entry per future fraction) ---
    values = np.zeros((remaining_fractions - 1, N_dose, N_overlap, N_mu, N_sigma))    # V(s): DP Value Function over all future states; axes: (fraction_index, accumulated_dose, overlap, belief_mu, belief_sigma)
    policies = np.zeros((remaining_fractions - 1, N_dose, N_overlap, N_mu, N_sigma))  # π(s): optimal Dose Action for each future state; axes: (fraction_index, accumulated_dose, overlap, belief_mu, belief_sigma)
    current_fraction_policy = np.zeros(N_overlap)  # optimal dose for the current fraction, as a function of current overlap (will be populated below)

    if remaining_ptv_dose < remaining_fractions * min_dose:  # overshoot unavoidable (prescribed dose will be exceeded); deliver min_dose.
        actual_policy = min_dose
        current_fraction_policy, actual_value = _set_infeasible_state(actual_policy, values, N_overlap, remaining_fractions)
    elif remaining_ptv_dose > remaining_fractions * max_dose:  # underdose unavoidable (prescribed dose unreachable); deliver max_dose.
        actual_policy = max_dose
        current_fraction_policy, actual_value = _set_infeasible_state(actual_policy, values, N_overlap, remaining_fractions)
    else:
        min_float = np.finfo(np.float64).min
        for state, fraction_state in enumerate(np.arange(number_of_fractions, fraction - 1, -1)):
            n_t = int(fraction_state)

            if state == number_of_fractions - 1:
                # Actual first fraction (only reached when fraction == 1).
                overlap_penalty = penalty_calc_matrix(action_space, volume_space, min_dose)
                actual_penalty = penalty_calc_single(action_space, min_dose, observed_overlap)
                future_value_prob = _bellman_expectation(values[state - 1], volume_space, initial_probs, initial_belief_mu, initial_belief_sigma, n_t)
                future_values = linear_interp(dose_space, future_value_prob, action_space)
                values_actual_frac = -overlap_penalty + future_values
                current_fraction_policy = action_space[values_actual_frac.argmax(axis=1)]
                actual_value = -actual_penalty + future_values
                actual_policy = action_space[actual_value.argmax()]

            elif fraction_state == fraction and fraction != number_of_fractions:
                # Actual fraction (not first, not last).
                delivered_doses_clipped = action_space[0: max_action(accumulated_dose, action_space, prescribed_dose) + 1]
                overlap_penalty = penalty_calc_matrix(delivered_doses_clipped, volume_space, min_dose)
                actual_penalty = penalty_calc_single(delivered_doses_clipped, min_dose, observed_overlap)
                future_doses = accumulated_dose + delivered_doses_clipped
                future_doses[future_doses > prescribed_dose] = overdose_sentinel
                penalties = np.zeros(future_doses.shape)
                penalties[future_doses > prescribed_dose] = -1e12
                future_value_prob = _bellman_expectation(values[state - 1], volume_space, initial_probs, initial_belief_mu, initial_belief_sigma, n_t)
                future_values = linear_interp(dose_space, future_value_prob, future_doses)
                values_actual_frac = -overlap_penalty + future_values + penalties
                current_fraction_policy = delivered_doses_clipped[values_actual_frac.argmax(axis=1)]
                actual_value = -actual_penalty + future_values + penalties
                actual_policy = delivered_doses_clipped[actual_value.argmax()]

            elif fraction == number_of_fractions:
                # Actual fraction is the last fraction; action is fixed.
                best_action = prescribed_dose - accumulated_dose
                if accumulated_dose > prescribed_dose:
                    best_action = 0
                if best_action < min_dose:
                    best_action = min_dose
                if best_action > max_dose:
                    best_action = max_dose
                actual_policy = best_action
                actual_value = np.zeros(1)

            else:
                # Hypothetical future fractions: fill values[state] for all beliefs.
                if state != 0:
                    # --- Intermediate future fraction ---
                    # Step 1: belief transitions for all (mi, si, j) at this n_t
                    mu_vals = _MU_GRID[:, None, None]          # (N_mu, 1, 1)
                    sigma_vals = _SIGMA_GRID[None, :, None]    # (1, N_sigma, 1)
                    o_vals = volume_space[None, None, :]       # (1, 1, N_overlap)
                    mu_prime = (n_t * mu_vals + o_vals) / (n_t + 1)
                    delta = o_vals - mu_vals
                    delta_prime = o_vals - mu_prime
                    sigma2_prime = (n_t * sigma_vals ** 2 + delta * delta_prime) / (n_t + 1)
                    sigma_prime = np.sqrt(np.maximum(sigma2_prime, _SIGMA_MIN ** 2))
                    mu_prime = np.clip(mu_prime, _MU_GRID[0], _MU_GRID[-1])
                    sigma_prime = np.clip(sigma_prime, _SIGMA_GRID[0], _SIGMA_GRID[-1])
                    # Nearest grid indices — shape (N_mu, N_sigma, N_overlap)
                    # searchsorted is O(N*log(G)) vs argmin's O(N*G), ~35x faster for G=280.
                    next_mi = nearest_idx(mu_prime, _MU_GRID)
                    next_si = nearest_idx(sigma_prime, _SIGMA_GRID)

                    # Steps 2+3: accumulate weighted future values in a j-loop.
                    # This avoids materialising the full (N_dose, N_mu, N_sigma, N_overlap) = ~150 MB
                    # branch_vals array; instead each j-slice is ~0.75 MB and is processed on the fly.
                    next_b = next_mi * N_sigma + next_si  # flat belief index, shape (N_mu, N_sigma, N_overlap)
                    vals_flat = values[state - 1].reshape(N_dose, N_overlap, -1)  # (N_dose, N_overlap, N_belief)
                    future_value_prob_full = np.zeros((N_dose, N_mu, N_sigma))
                    for j in range(N_overlap):
                        # vals_flat[:, j, next_b[:, :, j]]: (N_dose, N_mu, N_sigma) — one j-slice
                        future_value_prob_full += vals_flat[:, j, :][:, next_b[:, :, j]] * p_belief[None, :, :, j]

                    # Step 4: vectorised interpolation over dose for all (mi, si) simultaneously
                    overlap_penalty = penalty_calc_matrix(action_space, volume_space, min_dose)
                    max_allowed_actions = np.minimum(action_space[-1], prescribed_dose - dose_space)
                    max_action_indices = np.abs(
                        action_space[None, :] - max_allowed_actions[:, None]
                    ).argmin(axis=1)
                    max_action_indices = np.where(max_action_indices == 0, 1, max_action_indices)
                    valid_actions = (
                        np.arange(action_space.size)[None, :] <= max_action_indices[:, None]
                    )  # (N_dose, N_action)

                    future_doses = dose_space[:, None] + action_space[None, :]  # (N_dose, N_action)
                    overdosed = future_doses > prescribed_dose
                    future_doses_clipped = np.where(overdosed, overdose_sentinel, future_doses)
                    overdose_pens = np.where(overdosed, -1e12, 0.0)

                    q = future_doses_clipped.ravel()
                    hi_idx = np.clip(np.searchsorted(dose_space, q, side='right'), 1, N_dose - 1)
                    lo_idx = hi_idx - 1
                    denom = dose_space[hi_idx] - dose_space[lo_idx]
                    w = np.clip(
                        np.where(denom > 0, (q - dose_space[lo_idx]) / denom, 0.0), 0.0, 1.0
                    )
                    # future_values_full: (N_dose*N_action, N_mu, N_sigma) → reshaped
                    fvp_lo = future_value_prob_full[lo_idx, :, :]
                    fvp_hi = future_value_prob_full[hi_idx, :, :]
                    future_values_full = (
                        fvp_lo * (1.0 - w[:, None, None]) + fvp_hi * w[:, None, None]
                    ).reshape(N_dose, len(action_space), N_mu, N_sigma)

                    # Step 5: value/policy update.
                    # vs_base_T: (N_dose, N_mu, N_sigma, N_action) — action axis last for cache-friendly argmax.
                    # Preallocated buffers avoid per-iteration heap allocations.
                    # Flat-base index replaces take_along_axis (avoids one temporary array per j).
                    vs_base = future_values_full + overdose_pens[:, :, None, None]
                    vs_base = np.where(valid_actions[:, :, None, None], vs_base, min_float)
                    vs_base_T = vs_base.transpose(0, 2, 3, 1).copy()  # (N_dose, N_mu, N_sigma, N_action) contiguous
                    N_action = len(action_space)
                    vs_j_buf = np.empty_like(vs_base_T)
                    ai_buf = np.empty((N_dose, N_mu, N_sigma), dtype=np.intp)
                    # Precompute flat index base so values can be gathered without take_along_axis
                    _d = np.arange(N_dose)[:, None, None]
                    _m = np.arange(N_mu)[None, :, None]
                    _s = np.arange(N_sigma)[None, None, :]
                    flat_base = (_d * N_mu * N_sigma + _m * N_sigma + _s) * N_action
                    vs_flat = vs_j_buf.ravel()  # flat view updated in-place each iteration
                    values_state = np.empty((N_dose, N_overlap, N_mu, N_sigma))
                    policies_state = np.empty((N_dose, N_overlap, N_mu, N_sigma))
                    for j in range(N_overlap):
                        np.subtract(vs_base_T, overlap_penalty[j], out=vs_j_buf)
                        np.argmax(vs_j_buf, axis=-1, out=ai_buf)
                        values_state[:, j, :, :] = vs_flat[flat_base + ai_buf]
                        policies_state[:, j, :, :] = action_space[ai_buf]
                    values[state] = values_state
                    policies[state] = policies_state

                else:
                    # --- Terminal fraction (state == 0) ---
                    best_actions = prescribed_dose - dose_space
                    best_actions[best_actions > max_dose] = max_dose
                    best_actions[best_actions < min_dose] = min_dose
                    future_accumulated_dose = dose_space + best_actions
                    last_penalty = penalty_calc_single(
                        best_actions[:, None], min_dose, volume_space[None, :]
                    )  # (N_dose, N_overlap)
                    underdose_penalty = np.zeros(future_accumulated_dose.shape)
                    overdose_penalty = np.zeros(future_accumulated_dose.shape)
                    underdose_penalty[np.round(future_accumulated_dose, 2) < prescribed_dose] = -1e12
                    overdose_penalty[np.round(future_accumulated_dose, 2) > prescribed_dose] = -1e12
                    terminal_val = (
                        -last_penalty
                        + underdose_penalty[:, None]
                        + overdose_penalty[:, None]
                    )  # (N_dose, N_overlap)
                    # Broadcast over belief dims: terminal value is belief-independent
                    values[state] = terminal_val[:, :, None, None]
                    policies[state] = best_actions[:, None, None, None] * np.ones(
                        (N_dose, N_overlap, N_mu, N_sigma)
                    )

    physical_dose = np.round(actual_policy, 2)
    penalty_added = penalty_calc_single(physical_dose, min_dose, observed_overlap)
    final_penalty = np.max(actual_value) - penalty_added
    return [policies, current_fraction_policy, volume_space, physical_dose, penalty_added, values, dose_space, initial_probs, final_penalty]
    
   
def adaptfx_full(volumes: list, number_of_fractions: int = DEFAULT_NUMBER_OF_FRACTIONS, min_dose: float = DEFAULT_MIN_DOSE, max_dose: float = DEFAULT_MAX_DOSE, mean_dose: float = DEFAULT_MEAN_DOSE, dose_steps: float = DEFAULT_DOSE_STEPS, alpha: float = DEFAULT_ALPHA, beta:float = DEFAULT_BETA):
    """Computes a full adaptive fractionation plan when all overlap volumes are given.

    Args:
        volumes (list): list of all volume overlaps observed
        number_of_fractions (float, optional): number of fractions delivered. Defaults to 5.
        min_dose (float, optional): minimum phyical dose delivered in each fraction. Defaults to 7.5.
        max_dose (float, optional): maximum dose delivered in each fraction. Defaults to 9.5.
        mean_dose (int, optional): mean dose to be delivered over all fractions. Defaults to 8.

    Returns:
        numpy arrays: physical dose (array with all optimal doses to be delivered),
        accumullated_doses (array with the accumulated dose in each fraction),
        total_penalty (final penalty after fractionation if all suggested doses are applied)
    """
    physical_doses = np.zeros(number_of_fractions)
    accumulated_doses = np.zeros(number_of_fractions)
    for index, frac in enumerate(range(1,number_of_fractions +1)):
        if frac != number_of_fractions:
            physical_dose = adaptive_fractionation_core(fraction = frac, volumes = np.array(volumes[:-number_of_fractions+frac]), accumulated_dose = accumulated_doses[index], number_of_fractions= number_of_fractions, min_dose = min_dose, max_dose = max_dose, mean_dose = mean_dose, dose_steps = dose_steps, alpha = alpha, beta = beta)[3]  # return index 3 is the recommended dose
            accumulated_doses[index+1] = accumulated_doses[index] + physical_dose
        else:
            physical_dose = adaptive_fractionation_core(fraction = frac, volumes = np.array(volumes),accumulated_dose = accumulated_doses[index], number_of_fractions= number_of_fractions, min_dose = min_dose, max_dose = max_dose, mean_dose = mean_dose, dose_steps = dose_steps, alpha = alpha, beta = beta)[3]  # final fraction uses full observed volume history
        physical_doses[index] = physical_dose
    total_penalty = 0
    for index, dose in enumerate(physical_doses):
        total_penalty -= penalty_calc_single(dose, min_dose, volumes[-number_of_fractions+index])
    return physical_doses, accumulated_doses, total_penalty


def precompute_plan(fraction: int, volumes: np.ndarray, accumulated_dose: float, number_of_fractions: int = DEFAULT_NUMBER_OF_FRACTIONS, min_dose: float = DEFAULT_MIN_DOSE, max_dose: float = DEFAULT_MAX_DOSE, mean_dose:float = DEFAULT_MEAN_DOSE, dose_steps = DEFAULT_DOSE_STEPS, alpha: float = DEFAULT_ALPHA, beta:float = DEFAULT_BETA):
    """Precomputes all possible delivered doses in the next fraction by looping through possible
    observed overlap volumes. Returning a df and two lists with the overlap volumes and
    the respective dose that would be delivered.

    Args:
        fraction (int): number of actual fraction
        volumes (np.ndarray): list of all volume overlaps observed so far
        accumulated_dose (float): accumulated physical dose in tumor
        number_of_fractions (int, optional): number of fractions given in total. Defaults to 5.
        min_dose (float, optional): minimum phyical dose delivered in each fraction. Defaults to 7.5.
        max_dose (float, optional): maximum dose delivered in each fraction. Defaults to 9.5.
        mean_dose (int, optional): mean dose to be delivered over all fractions. Defaults to 8.
        alpha (float, optional): alpha value of gamma distribution. Defaults to 1.8380125313579265.
        beta (float, optional): beta value of gamma distribution. Defaults to 0.2654168553532238.

    Returns:
        pd.Dataframe, lists: Returns a dataframe with volumes and respective doses, and volumes and doses separated in two lists.
    """
    volumes = np.asarray(volumes, dtype=float)
    std = std_calc(volumes, alpha, beta)
    distribution_params = (volumes.mean(), std)
    volume_space = get_state_space(distribution_params)
    distribution_max = 6.5 if volume_space.max() < 6.5 else volume_space.max()
    min_dose_deliverable = min_dose_to_deliver(accumulated_dose=accumulated_dose,fractions_left = number_of_fractions - fraction + 1, prescribed_dose = mean_dose*number_of_fractions, min_dose = min_dose, max_dose = max_dose)
    volumes_to_check = [0.0]  # start from 0 cc and grow in 0.1 cc increments until we meet stop criteria
    predicted_policies = []
    volumes_with_candidate = np.empty(volumes.size + 1, dtype=float)  # reuse one buffer to avoid reallocating arrays on every loop iteration
    volumes_with_candidate[:-1] = volumes  # keep all observed volumes fixed and update only the candidate last element

    while True:
        volume = volumes_to_check[-1]
        volumes_with_candidate[-1] = volume  # inject the candidate overlap volume for this iteration
        physical_dose = adaptive_fractionation_core(
            fraction = fraction,
            volumes = volumes_with_candidate,
            accumulated_dose = accumulated_dose,
            number_of_fractions = number_of_fractions,
            min_dose = min_dose,
            max_dose = max_dose,
            mean_dose = mean_dose,
            dose_steps = dose_steps,
            alpha = alpha,
            beta = beta
        )[3]
        predicted_policies.append(physical_dose)

        covered_distribution_range = volume >= distribution_max  # keep at least the clinically relevant volume range
        reached_minimum_policy = physical_dose <= min_dose_deliverable  # continue until policy reaches minimum deliverable dose
        if covered_distribution_range and reached_minimum_policy:
            break

        volumes_to_check.append(np.round(volume + 0.1, 10))  # avoid float drift from repeated +0.1 operations

    volumes_to_check = np.asarray(volumes_to_check)
    predicted_policies = np.asarray(predicted_policies)
        
    data = {'volume': volumes_to_check,
            'dose': predicted_policies}
    volume_x_dose = pd.DataFrame(data)
    return volume_x_dose, volumes_to_check, predicted_policies
