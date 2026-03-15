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
from numba import njit, prange

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
    next_mi = nearest_idx(mu_prime, _MU_GRID)        # nearest mu grid index for each hypothetical next overlap
    next_si = nearest_idx(sigma_prime, _SIGMA_GRID)  # nearest sigma grid index for each hypothetical next overlap
    return next_mi, next_si  # indices rather than values so callers can directly index into the DP values array


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


@njit(parallel=True, cache=True)
def _fill_values_policies(future_values_masked_action_last, overlap_penalty, flat_index_base, action_space, values_state, policies_state):
    """Fill values_state and policies_state over all overlap bins in parallel.

    For each overlap bin j, subtracts the per-action penalty from the future-value
    array, finds the best action via argmax, gathers the corresponding value, and
    records the best action.  Parallelised over j (N_overlap=441 independent iterations).

    future_values_masked_action_last: (N_dose, N_mu, N_sigma, N_action) — future values with invalid actions masked to min_float, action axis last for cache-friendly argmax
    overlap_penalty:                  (N_overlap, N_action)             — OAR penalty for each (overlap bin, action) pair
    flat_index_base:                  (N_dose, N_mu, N_sigma)           — flat index of the first action entry for each (dose, mu, sigma) triple
    action_space:                     (N_action,)                       — deliverable doses
    values_state:                     (N_dose, N_overlap, N_mu, N_sigma) — output: best achievable value for each (dose, overlap, belief) combination
    policies_state:                   (N_dose, N_overlap, N_mu, N_sigma) — output: optimal dose action for each (dose, overlap, belief) combination
    """
    N_dose   = future_values_masked_action_last.shape[0]
    N_mu     = future_values_masked_action_last.shape[1]
    N_sigma  = future_values_masked_action_last.shape[2]
    N_overlap = overlap_penalty.shape[0]
    for j in prange(N_overlap):
        vs_j = future_values_masked_action_last - overlap_penalty[j]   # subtract this overlap bin's penalty from all future values; shape: (N_dose, N_mu, N_sigma, N_action), thread-local
        ai = np.argmax(vs_j, axis=3)                                   # index of the best action for each (dose, mu, sigma) triple; shape: (N_dose, N_mu, N_sigma)
        vs_j_flat = vs_j.ravel()
        for d in range(N_dose):
            for m in range(N_mu):
                for s in range(N_sigma):
                    values_state[d, j, m, s]   = vs_j_flat[flat_index_base[d, m, s] + ai[d, m, s]]  # gather the best value using the precomputed flat index
                    policies_state[d, j, m, s] = action_space[ai[d, m, s]]                           # record the corresponding optimal dose action


# TODO: remove default parameter values — callers should always pass explicit values;
#       replace defaults with raised exceptions (TypeError/ValueError) when a value is missing.
def adaptive_fractionation_core(fraction_index_today: int, volumes: np.ndarray, accumulated_dose: float, number_of_fractions: int = DEFAULT_NUMBER_OF_FRACTIONS, min_dose: float = DEFAULT_MIN_DOSE, max_dose: float = DEFAULT_MAX_DOSE, mean_dose:float = DEFAULT_MEAN_DOSE, dose_steps: float = DEFAULT_DOSE_STEPS, alpha: float = DEFAULT_ALPHA, beta:float = DEFAULT_BETA):
    """Belief-state DP solver. Computes the recommended dose for the current fraction.

    Optimizes fractionation by minimizing expected PTV underdosage cost: lower dose
    when PTV-OAR overlap is large, higher dose when overlap is small.

    Args:
        fraction_index_today (int): today's fraction number (1-indexed).
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
               values, dose_space, current_overlap_probs, optimal_state_value]
              where optimal_state_value = np.max(actual_value) = the best total expected OAR cost
              from this fraction to the end of treatment (negative value; less negative = better).
              penalty_added = the immediate OAR cost incurred at this fraction for the recommended dose.
    """
    prescribed_dose = number_of_fractions * mean_dose  # total prescribed dose (Gy)
    observed_overlap = volumes[-1]  # overlap observed at the current fraction
    min_accumulated_dose_after_fraction = accumulated_dose + min_dose  # minimum possible accumulated dose after this fraction

    # --- Current patient belief from observed volumes ---
    initial_belief_mu = volumes.mean()
    initial_belief_sigma = std_calc(volumes, alpha, beta)
    volume_space = _VOLUME_SPACE
    current_overlap_probs = probdist((initial_belief_mu, initial_belief_sigma), volume_space)  # probability of each overlap bin j given current belief (mu, sigma)

    # --- State and action spaces ---
    dose_space = np.arange(min_accumulated_dose_after_fraction, prescribed_dose, dose_steps)  # reachable accumulated PTV doses from this fraction onwards
    overdose_sentinel = prescribed_dose + 0.05  # any accumulated PTV dose mapped here represents an overdose; value just needs to be outside the valid dose range
    dose_space = np.concatenate((dose_space, [prescribed_dose, overdose_sentinel]))  # prescribed_dose appended explicitly: arange excludes stop value and float steps can miss it
    action_space = np.arange(min_dose, max_dose + 0.01, dose_steps)  # deliverable doses per fraction; +0.01 ensures max_dose is included

    # Verify that every action_space value lies exactly on the dose_space grid (same step size and alignment).
    # This is required for the direct index lookup in the DP loop to be correct — if this fails, the lookup
    # would silently return values from the wrong grid point instead of raising an error.
    assert np.allclose(np.mod(action_space - dose_space[0], dose_steps), 0, atol=1e-9), (
        "action_space values must all lie on the dose_space grid; "
        "direct index lookup requires that action_space and dose_space share the same step size and alignment. "
        f"dose_steps={dose_steps}, dose_space[0]={dose_space[0]}, action_space[0]={action_space[0]}"
    )

    # --- Dimension shorthands ---
    N_mu = len(_MU_GRID)
    N_sigma = len(_SIGMA_GRID)
    N_overlap = len(volume_space)
    N_dose = len(dose_space)

    remaining_ptv_dose = prescribed_dose - accumulated_dose
    remaining_fractions = number_of_fractions - fraction_index_today + 1  # includes the current fraction (not yet delivered)

    # --- Value Function and Policy Tables (5D state space, one table per future DP state) ---
    values = np.zeros((remaining_fractions - 1, N_dose, N_overlap, N_mu, N_sigma))    # V(s): DP Value Function over all future states; axes: (fraction_index, accumulated_dose, overlap, belief_mu, belief_sigma)
    policies = np.zeros((remaining_fractions - 1, N_dose, N_overlap, N_mu, N_sigma))  # π(s): optimal Dose Action for each future state; axes: (fraction_index, accumulated_dose, overlap, belief_mu, belief_sigma)
    current_fraction_policy = np.zeros(N_overlap)  # optimal dose for the current fraction, as a function of current overlap (will be populated below)

    if remaining_ptv_dose < remaining_fractions * min_dose:  # if overshoot is unavoidable (prescribed dose will be exceeded), then deliver min_dose.
        recommended_dose = min_dose
        current_fraction_policy, actual_value = _set_infeasible_state(recommended_dose, values, N_overlap, remaining_fractions)  # skip DP: plan is infeasible regardless of overlap
    elif remaining_ptv_dose > remaining_fractions * max_dose:  # if underdose is unavoidable (prescribed dose unreachable), then deliver max_dose.
        recommended_dose = max_dose
        current_fraction_policy, actual_value = _set_infeasible_state(recommended_dose, values, N_overlap, remaining_fractions)  # skip DP: plan is infeasible regardless of overlap
    else:
        min_float = np.finfo(np.float64).min
        overlap_penalty = penalty_calc_matrix(action_space, volume_space, min_dose)  # OAR immediate cost for each (action, overlap) combination; constant across all DP iterations
        for i, fraction_number in enumerate(np.arange(number_of_fractions, fraction_index_today - 1, -1)):  # backward iteration (starting from last treatment fraction, ending at today's fraction)
            # i runs from 0 (last treatment fraction) to remaining_fractions-1 (today's fraction)

            observation_count = int(fraction_number)  # number of overlap observations accumulated at this fraction

            if fraction_number != fraction_index_today:  # Evaluating a hypothetical future fraction (not today's)
                if i != 0:  # not the terminal fraction - run full DP update

                    # ----- Bellman expectation over all (belief_mu, belief_sigma) grid points ----- 
                    # For each belief, computes the probability-weighted sum of next-state values across all possible overlap outcomes.
                    # For every (belief_mu, belief_sigma, overlap_bin) triple:
                    #   1. Apply Welford update to get the next belief,
                    #   2. Snap it to the nearest grid point,
                    #   3. Accumulate probability-weighted future values.
                    # Result: expected future value for each (accumulated_dose, belief) pair.
                    future_value_prob_full = _bellman_expectation_full_grid(values[i - 1], observation_count)  # shape: (N_dose, N_mu, N_sigma)

                    # ----- Compute valid_actions mask -----
                    # Includes clipping the action space to prevent overshooting prescribed dose.
                    min_dose_reserved_for_future = (number_of_fractions - fraction_number) * min_dose  # minimum dose budget that must be reserved for all remaining fractions after this one
                    max_allowed_actions = np.minimum(action_space[-1], prescribed_dose - dose_space - min_dose_reserved_for_future)  # for each accumulated dose state, the largest single-fraction dose that still leaves enough budget for future fractions
                    max_action_indices = np.abs(action_space[None, :] - max_allowed_actions[:, None]).argmin(axis=1)  # index into action_space of the largest allowed dose for each accumulated dose state
                    max_action_indices = np.where(max_action_indices == 0, 1, max_action_indices)  # clamp to at least index 1 so that every dose state has at least one valid action available
                    valid_actions = (np.arange(action_space.size)[None, :] <= max_action_indices[:, None])  # boolean mask: True where the action can be taken without exceeding the prescribed dose; shape: (N_dose, N_action)

                    # ----- Look up future_value_prob_full at each future accumulated dose -----
                    #   action_space and dose_space share the same dose_steps, so every future dose lands exactly on a dose_space grid point.
                    #   np.round removes any sub-epsilon floating-point noise accumulated by np.arange, before the searchsorted call.
                    future_doses = dose_space[:, None] + action_space[None, :]  # total accumulated dose that would result from taking each action at each dose state; shape: (N_dose, N_action)
                    query_doses = future_doses.ravel()  # flatten to 1D: one query dose per (dose_state, action) pair; shape: (N_dose * N_action,)
                    dose_indices = np.clip(np.searchsorted(dose_space, np.round(query_doses, decimals=10), side='left'), 0, N_dose - 1)  # index of the matching dose_space grid point for each query dose
                    future_values_full = future_value_prob_full[dose_indices].reshape(N_dose, len(action_space), N_mu, N_sigma)  # look up the value of the next state reached by each (dose_state, action) pair, across all belief grid points; shape: (N_dose, N_action, N_mu, N_sigma)

                    # ----- Value & Policy update -----
                    # Mask out invalid actions by setting the value of the state they lead to as min_float (a very large
                    # negative number), so the Numba kernel never selects an overdosing action as the optimal policy.
                    future_values_masked = np.where(valid_actions[:, :, None, None], future_values_full, min_float)  # shape: (N_dose, N_action, N_mu, N_sigma)

                    # Transpose to (N_dose, N_mu, N_sigma, N_action) so the action axis is last — this makes the
                    # argmax inside the Numba kernel scan a contiguous memory region, which is cache-friendly (~18% faster).
                    future_values_masked_action_last = future_values_masked.transpose(0, 2, 3, 1).copy()  # shape: (N_dose, N_mu, N_sigma, N_action), contiguous

                    # Precompute flat_index_base: for each (dose, mu, sigma) triple, the flat index in the ravelled
                    # future_values_masked_action_last where that triple's action entries begin.
                    # Layout is (N_dose, N_mu, N_sigma, N_action), so the flat position of element [d, m, s, a] is:
                    #   (d * N_mu * N_sigma  +  m * N_sigma  +  s) * N_action  +  a
                    # flat_index_base stores the part before '+ a', so the kernel only needs to add ai[d,m,s] at
                    # lookup time — avoiding a take_along_axis call and its temporary allocations.
                    dose_broadcast_idx  = np.arange(N_dose)[:, None, None]  # shape (N_dose, 1, 1): broadcasts over the (N_dose, N_mu, N_sigma) output
                    mu_broadcast_idx    = np.arange(N_mu)[None, :, None]    # shape (1, N_mu, 1):   broadcasts over the (N_dose, N_mu, N_sigma) output
                    sigma_broadcast_idx = np.arange(N_sigma)[None, None, :] # shape (1, 1, N_sigma): broadcasts over the (N_dose, N_mu, N_sigma) output
                    flat_index_base = (dose_broadcast_idx * N_mu * N_sigma + mu_broadcast_idx * N_sigma + sigma_broadcast_idx) * len(action_space)  # shape: (N_dose, N_mu, N_sigma)

                    values_state  = np.empty((N_dose, N_overlap, N_mu, N_sigma))  # Value Function for this DP step; will be filled by the Numba kernel
                    policies_state = np.empty((N_dose, N_overlap, N_mu, N_sigma)) # Optimal Policy for this DP step; will be filled by the Numba kernel

                    # For each (dose, overlap, belief) combination, find the action that maximises
                    # total state value (future value minus immediate OAR cost), and record that value and action in values_state and policies_state.
                    _fill_values_policies(future_values_masked_action_last, overlap_penalty, flat_index_base, action_space, values_state, policies_state)
                    values[i]   = values_state
                    policies[i] = policies_state

                else:  # Hypothetical future fraction is the terminal one (i == 0, last fraction of treatment)
                    # At the terminal fraction there is no choice: deliver exactly the remaining dose needed to reach
                    # prescribed_dose, clamped to the daily dose limits [min_dose, max_dose].
                    best_actions = np.clip(prescribed_dose - dose_space, min_dose, max_dose)  # shape: (N_dose,)
                    future_accumulated_dose = dose_space + best_actions  # total accumulated dose after the terminal fraction; shape: (N_dose,)

                    # OAR penalty incurred at the terminal fraction for each (dose_state, overlap_bin) combination.
                    terminal_oar_penalty = penalty_calc_single(best_actions[:, None], min_dose, volume_space[None, :])  # shape: (N_dose, N_overlap)

                    # Apply a large negative penalty (-1e12) to dose states where the terminal fraction still leaves
                    # the patient underdosed or overdosed. np.round guards against floating-point noise near prescribed_dose.
                    underdose_penalty = np.zeros(future_accumulated_dose.shape)
                    overdose_penalty  = np.zeros(future_accumulated_dose.shape)
                    underdose_penalty[np.round(future_accumulated_dose, 2) < prescribed_dose] = -1e12
                    overdose_penalty[np.round(future_accumulated_dose, 2) > prescribed_dose] = -1e12

                    # Terminal state value = immediate cost + feasibility penalties.
                    # The value is independent of belief (mu, sigma) because at the terminal fraction the
                    # dose decision is fully determined by the remaining dose, not the overlap belief.
                    terminal_state_value = (-terminal_oar_penalty + underdose_penalty[:, None] + overdose_penalty[:, None])  # shape: (N_dose, N_overlap)
                    values[i]   = terminal_state_value[:, :, None, None]                                           # broadcast over belief dimensions; shape: (N_dose, N_overlap, N_mu, N_sigma)
                    policies[i] = best_actions[:, None, None, None] * np.ones((N_dose, N_overlap, N_mu, N_sigma))  # same best action for every (overlap, belief) combination

            elif fraction_index_today == 1:  # Today is the first fraction of treatment.
                # Immediate OAR cost for each action at today's actually observed overlap.
                immediate_cost = penalty_calc_single(action_space, min_dose, observed_overlap)

                # Expected future state value for each possible next accumulated dose, integrated over all
                # possible overlap outcomes weighted by the current belief (mu, sigma).
                future_value_prob = _bellman_expectation(values[i - 1], volume_space, current_overlap_probs, initial_belief_mu, initial_belief_sigma, observation_count)

                # Future state value for each action — each action leads to a different next accumulated dose.
                # action_space and dose_space share the same step and starting point (both begin at min_dose for
                # fraction 1), so every action maps exactly to a dose_space grid point (use direct lookup).
                action_dose_indices = np.clip(np.searchsorted(dose_space, np.round(action_space, decimals=10), side='left'), 0, N_dose - 1)
                future_values = future_value_prob[action_dose_indices] # value of state reached by each possible action

                # Total state value for every (action, overlap) combination = future value minus immediate OAR cost.
                # Used to determine the best action for each possible overlap outcome (the full policy).
                state_values_full_grid = -overlap_penalty + future_values
                current_fraction_policy = action_space[state_values_full_grid.argmax(axis=1)]  # optimal action for each overlap bin

                # Total state value for each action at the actually observed overlap = future value minus immediate OAR cost.
                actual_value = -immediate_cost + future_values
                recommended_dose = action_space[actual_value.argmax()]  # optimal action given today's observed overlap

            elif fraction_index_today == number_of_fractions: # Today's fraction is the last (action is fixed: deliver exactly the remaining dose).
                recommended_dose = np.clip(remaining_ptv_dose, min_dose, max_dose)  # Clamp remaining dose to daily prescription limits
                actual_value = np.zeros(1)  # total state value = 0; no future fractions remain and the immediate cost is subtracted separately via penalty_added on line 434

            else: # Current fraction is in the middle (not first, not last).
                delivered_doses_clipped = action_space[0: max_action(accumulated_dose, action_space, prescribed_dose) + 1]
                overlap_penalty = penalty_calc_matrix(delivered_doses_clipped, volume_space, min_dose)
                actual_penalty = penalty_calc_single(delivered_doses_clipped, min_dose, observed_overlap)
                future_doses = accumulated_dose + delivered_doses_clipped
                future_doses[future_doses > prescribed_dose] = overdose_sentinel
                penalties = np.zeros(future_doses.shape)
                penalties[future_doses > prescribed_dose] = -1e12
                future_value_prob = _bellman_expectation(values[i - 1], volume_space, current_overlap_probs, initial_belief_mu, initial_belief_sigma, observation_count)
                future_values = linear_interp(dose_space, future_value_prob, future_doses)
                values_actual_frac = -overlap_penalty + future_values + penalties
                current_fraction_policy = delivered_doses_clipped[values_actual_frac.argmax(axis=1)]
                actual_value = -actual_penalty + future_values + penalties
                recommended_dose = delivered_doses_clipped[actual_value.argmax()]

    physical_dose = np.round(recommended_dose, 2)
    penalty_added = penalty_calc_single(physical_dose, min_dose, observed_overlap)
    optimal_state_value = np.max(actual_value)  # best total expected OAR cost from this fraction to end of treatment (negative; less negative = better)
    return [policies, current_fraction_policy, volume_space, physical_dose, penalty_added, values, dose_space, current_overlap_probs, optimal_state_value]
    
   
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
            physical_dose = adaptive_fractionation_core(fraction_index_today = frac, volumes = np.array(volumes[:-number_of_fractions+frac]), accumulated_dose = accumulated_doses[index], number_of_fractions= number_of_fractions, min_dose = min_dose, max_dose = max_dose, mean_dose = mean_dose, dose_steps = dose_steps, alpha = alpha, beta = beta)[3]  # return index 3 is the recommended dose
            accumulated_doses[index+1] = accumulated_doses[index] + physical_dose
        else:
            physical_dose = adaptive_fractionation_core(fraction_index_today = frac, volumes = np.array(volumes),accumulated_dose = accumulated_doses[index], number_of_fractions= number_of_fractions, min_dose = min_dose, max_dose = max_dose, mean_dose = mean_dose, dose_steps = dose_steps, alpha = alpha, beta = beta)[3]  # final fraction uses full observed volume history
        physical_doses[index] = physical_dose
    total_penalty = 0
    for index, dose in enumerate(physical_doses):
        total_penalty -= penalty_calc_single(dose, min_dose, volumes[-number_of_fractions+index])
    return physical_doses, accumulated_doses, total_penalty


def precompute_plan(fraction_index_today: int, volumes: np.ndarray, accumulated_dose: float, number_of_fractions: int = DEFAULT_NUMBER_OF_FRACTIONS, min_dose: float = DEFAULT_MIN_DOSE, max_dose: float = DEFAULT_MAX_DOSE, mean_dose:float = DEFAULT_MEAN_DOSE, dose_steps = DEFAULT_DOSE_STEPS, alpha: float = DEFAULT_ALPHA, beta:float = DEFAULT_BETA):
    """Precomputes all possible delivered doses in the next fraction by looping through possible
    observed overlap volumes. Returning a df and two lists with the overlap volumes and
    the respective dose that would be delivered.

    Args:
        fraction_index_today (int): today's fraction number (1-indexed)
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
    min_dose_deliverable = min_dose_to_deliver(accumulated_dose=accumulated_dose,fractions_left = number_of_fractions - fraction_index_today + 1, prescribed_dose = mean_dose*number_of_fractions, min_dose = min_dose, max_dose = max_dose)
    volumes_to_check = [0.0]  # start from 0 cc and grow in 0.1 cc increments until we meet stop criteria
    predicted_policies = []
    volumes_with_candidate = np.empty(volumes.size + 1, dtype=float)  # reuse one buffer to avoid reallocating arrays on every loop iteration
    volumes_with_candidate[:-1] = volumes  # keep all observed volumes fixed and update only the candidate last element

    while True:
        volume = volumes_to_check[-1]
        volumes_with_candidate[-1] = volume  # inject the candidate overlap volume for this iteration
        physical_dose = adaptive_fractionation_core(
            fraction_index_today = fraction_index_today,
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
