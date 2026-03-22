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
The belief grids, branch probabilities, and Bellman operators live in belief_model.py.
"""

__all__ = [  # limits what `from core_adaptfx import *` exposes
    "adaptive_fractionation_core",
    "adaptfx_full",
    "precompute_plan",
]

import numpy as np
import pandas as pd
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
    penalty_calc_single,
    penalty_calc_matrix,
    min_dose_to_deliver,
    linear_interp,
)

from .belief_model import (
    _MU_GRID,
    _SIGMA_GRID,
    _SIGMA_MIN,
    _VOLUME_SPACE,
    _MU_GRID_NIG,
    _SIGMA_GRID_NIG,
    _SIGMA_MIN_NIG,
    _LOG_OFFSET,
    _bellman_expectation,
    _bellman_expectation_full_grid,
    _bellman_expectation_nig,
    _bellman_expectation_full_grid_nig,
    _get_p_belief_nig,
    current_belief_probdist,
    current_belief_probdist_nig,
)


_INFEASIBILITY_SENTINEL = -1e12  # large negative value that marks infeasible DP states; all uses in this module must reference this constant
_NIG_SCAN_MAX_CC = 35.0  # Stage B precompute_plan scan upper bound; fixed clinical ceiling (covers ACTION cohort max ~29 cc + margin) rather than percentile heuristic, which blows up for heavy-tailed Student-t near Cauchy


def _set_infeasible_state(fixed_dose, values, N_overlap, remaining_fractions):
    """Set DP arrays to reflect an infeasible treatment plan.

    Called when the prescribed dose is unreachable or unavoidably exceeded.
    Marks all future states as invalid and returns a constant policy and sentinel value.
    """
    current_fraction_policy = np.ones(N_overlap) * fixed_dose
    if remaining_fractions > 1:  # skip on last fraction: no future fractions exist, values array is empty
        values[:] = _INFEASIBILITY_SENTINEL  # mark all future states as invalid — DP is not run in infeasible cases
    actual_value = np.ones(1) * _INFEASIBILITY_SENTINEL  # signal "infeasible case" to downstream code
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


def _build_dp_context(fraction_index_today, number_of_fractions, accumulated_dose, min_dose, max_dose, mean_dose, dose_steps, use_nig=False):
    """Build dose/action grids and run the DP backward sweep for all future fractions.

    Computes everything that is independent of the overlap actually observed at
    fraction_index_today: the dose/action grids, feasibility check, and value/policy
    tables for fractions fraction_index_today+1 through number_of_fractions.

    The returned context dict is reusable across multiple candidate overlap values,
    which is the key optimisation exploited by precompute_plan: run the expensive
    backward sweep once and then call _resolve_current_fraction once per candidate.

    Returns a dict with keys:
        values           (remaining_fractions-1, N_dose, N_overlap, N_mu, N_sigma)
        policies         same shape
        dose_space       (N_dose,)
        action_space     (N_action,)
        overlap_penalty  (N_overlap, N_action), or None when infeasible
        prescribed_dose, accumulated_dose, remaining_ptv_dose, remaining_fractions,
        overdose_sentinel, min_dose, max_dose, N_dose, N_overlap,
        is_infeasible (bool), fixed_dose (float or None)
    """
    prescribed_dose = number_of_fractions * mean_dose
    min_accumulated_dose_after_fraction = accumulated_dose + min_dose  # minimum possible accumulated dose after this fraction

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

    N_mu = len(_MU_GRID_NIG) if use_nig else len(_MU_GRID)
    N_sigma = len(_SIGMA_GRID_NIG) if use_nig else len(_SIGMA_GRID)
    N_overlap = len(_VOLUME_SPACE)
    N_dose = len(dose_space)

    remaining_ptv_dose = prescribed_dose - accumulated_dose
    remaining_fractions = number_of_fractions - fraction_index_today + 1  # includes the current fraction (not yet delivered)

    # For Stage B: validate and precompute NIG branch probability table BEFORE allocating the
    # large DP arrays so that a missing-constants RuntimeError is raised immediately.
    p_belief_nig = _get_p_belief_nig(number_of_fractions) if use_nig else None

    values  = np.zeros((remaining_fractions - 1, N_dose, N_overlap, N_mu, N_sigma))  # V(s): DP Value Function over all future states
    policies = np.zeros((remaining_fractions - 1, N_dose, N_overlap, N_mu, N_sigma))  # π(s): optimal Dose Action for each future state

    base = dict(
        dose_space=dose_space, action_space=action_space, prescribed_dose=prescribed_dose,
        accumulated_dose=accumulated_dose, remaining_ptv_dose=remaining_ptv_dose,
        remaining_fractions=remaining_fractions, overdose_sentinel=overdose_sentinel,
        min_dose=min_dose, max_dose=max_dose, N_dose=N_dose, N_overlap=N_overlap,
        use_nig=use_nig,
    )

    if remaining_ptv_dose < remaining_fractions * min_dose:  # overshoot unavoidable: prescribed dose will be exceeded
        if remaining_fractions > 1:
            values[:] = _INFEASIBILITY_SENTINEL
        return dict(**base, values=values, policies=policies, overlap_penalty=None, is_infeasible=True, fixed_dose=min_dose)
    if remaining_ptv_dose > remaining_fractions * max_dose:  # underdose unavoidable: prescribed dose unreachable
        if remaining_fractions > 1:
            values[:] = _INFEASIBILITY_SENTINEL
        return dict(**base, values=values, policies=policies, overlap_penalty=None, is_infeasible=True, fixed_dose=max_dose)

    min_float = np.finfo(np.float64).min
    overlap_penalty = penalty_calc_matrix(action_space, _VOLUME_SPACE, min_dose)  # OAR immediate cost for each (action, overlap) combination; constant across all DP iterations

    # Backward sweep over future fractions only (fraction_index_today+1 through number_of_fractions).
    # The current fraction (fraction_index_today) is handled separately in _resolve_current_fraction,
    # allowing its cheap per-overlap lookup to be called independently for each candidate overlap.
    for i, fraction_number in enumerate(np.arange(number_of_fractions, fraction_index_today, -1)):
        # i runs from 0 (last treatment fraction) to remaining_fractions-2 (fraction_index_today+1)
        observation_count = int(fraction_number)  # number of overlap observations accumulated at this fraction

        if i != 0:  # not the terminal fraction — run full DP update

            # ----- Bellman expectation over all (belief_mu, belief_sigma) grid points -----
            # For each belief, computes the probability-weighted sum of next-state values across all possible overlap outcomes.
            # For every (belief_mu, belief_sigma, overlap_bin) triple:
            #   1. Apply Welford update to get the next belief,
            #   2. Snap it to the nearest grid point,
            #   3. Accumulate probability-weighted future values.
            # Result: expected future value for each (accumulated_dose, belief) pair.
            # Stage B uses log-NIG grids and a log-space Welford update.
            # Stage A uses the precomputed Gaussian branch probabilities.
            if use_nig:
                future_value_prob_full = _bellman_expectation_full_grid_nig(
                    values[i - 1], observation_count, p_belief_nig[observation_count - 1])
            else:
                future_value_prob_full = _bellman_expectation_full_grid(values[i - 1], observation_count)

            # ----- Compute valid_actions mask -----
            # Includes clipping the action space to prevent overshooting prescribed dose.
            min_dose_reserved_for_future = (number_of_fractions - fraction_number) * min_dose  # minimum dose budget that must be reserved for all remaining fractions after this one
            max_allowed_actions = np.minimum(action_space[-1], prescribed_dose - dose_space - min_dose_reserved_for_future)  # for each accumulated dose state, the largest single-fraction dose that still leaves enough budget for future fractions
            max_action_indices = np.abs(action_space[None, :] - max_allowed_actions[:, None]).argmin(axis=1)  # index into action_space of the largest allowed dose for each accumulated dose state
            max_action_indices = np.where(max_action_indices == 0, 1, max_action_indices)  # clamp to at least index 1 so that every dose state has at least two valid actions available
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
            dose_broadcast_idx  = np.arange(N_dose)[:, None, None]  # shape (N_dose, 1, 1)
            mu_broadcast_idx    = np.arange(N_mu)[None, :, None]    # shape (1, N_mu, 1)
            sigma_broadcast_idx = np.arange(N_sigma)[None, None, :] # shape (1, 1, N_sigma)
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
            terminal_oar_penalty = penalty_calc_single(best_actions[:, None], min_dose, _VOLUME_SPACE[None, :])  # shape: (N_dose, N_overlap)

            # Apply _INFEASIBILITY_SENTINEL to dose states where the terminal fraction still leaves
            # the patient underdosed or overdosed. np.round guards against floating-point noise near prescribed_dose.
            underdose_penalty = np.zeros(future_accumulated_dose.shape)
            overdose_penalty  = np.zeros(future_accumulated_dose.shape)
            underdose_penalty[np.round(future_accumulated_dose, 2) < prescribed_dose] = _INFEASIBILITY_SENTINEL
            overdose_penalty[np.round(future_accumulated_dose, 2) > prescribed_dose] = _INFEASIBILITY_SENTINEL

            # Terminal state value = immediate cost + feasibility penalties.
            # The value is independent of belief (mu, sigma) because at the terminal fraction the
            # dose decision is fully determined by the remaining dose, not the overlap belief.
            terminal_state_value = (-terminal_oar_penalty + underdose_penalty[:, None] + overdose_penalty[:, None])  # shape: (N_dose, N_overlap)
            values[i]   = terminal_state_value[:, :, None, None]                                           # broadcast over belief dimensions; shape: (N_dose, N_overlap, N_mu, N_sigma)
            policies[i] = best_actions[:, None, None, None] * np.ones((N_dose, N_overlap, N_mu, N_sigma))  # broadcast N_mu, N_sigma are already correct for NIG or Stage A

    return dict(**base, values=values, policies=policies, overlap_penalty=overlap_penalty, is_infeasible=False, fixed_dose=None)


def _resolve_current_fraction(ctx, fraction_index_today, number_of_fractions, observed_overlap, prior_volumes, alpha, beta):
    """Given precomputed DP context, look up the recommended dose for a specific observed overlap.

    This is the cheap per-observation step that complements _build_dp_context. The backward
    sweep in _build_dp_context is independent of which overlap is actually observed today;
    this function consumes that cached result and applies the current-fraction Bellman step
    for one specific observed_overlap.

    Args:
        ctx: dict returned by _build_dp_context
        fraction_index_today (int): current fraction number (1-indexed)
        number_of_fractions (int): total number of fractions
        observed_overlap (float): overlap volume observed at the current fraction
        prior_volumes (np.ndarray): overlap volumes observed before this fraction (volumes[:-1])
        alpha, beta (float): belief prior parameters

    Returns:
        (recommended_dose, actual_value, current_fraction_policy, current_overlap_probs)
        where current_overlap_probs is None for infeasible plans.
    """
    values             = ctx['values']
    dose_space         = ctx['dose_space']
    action_space       = ctx['action_space']
    overlap_penalty    = ctx['overlap_penalty']
    prescribed_dose    = ctx['prescribed_dose']
    accumulated_dose   = ctx['accumulated_dose']
    remaining_ptv_dose = ctx['remaining_ptv_dose']
    remaining_fractions = ctx['remaining_fractions']
    overdose_sentinel  = ctx['overdose_sentinel']
    N_dose             = ctx['N_dose']
    N_overlap          = ctx['N_overlap']
    min_dose           = ctx['min_dose']
    max_dose           = ctx['max_dose']

    use_nig = ctx.get('use_nig', False)

    if ctx['is_infeasible']:
        fixed_dose = ctx['fixed_dose']
        current_fraction_policy = np.ones(N_overlap) * fixed_dose
        actual_value = np.ones(1) * _INFEASIBILITY_SENTINEL
        return fixed_dose, actual_value, current_fraction_policy, None

    # Current belief: estimate from all volumes including today's observation.
    # Stage A: Gaussian belief parameterised by Welford (s_bar, std_calc(alpha,beta)).
    # Stage B: Log-NIG grid coordinates are (s_bar_log, sigma_running) — the running log-mean
    # and log-std.  The NIG posterior parameters (mu_n, beta_n, etc.) are derived from these
    # sufficient statistics inside the pre-computed branch-probability table and Bellman
    # operators; they must NOT be used as grid coordinates here.
    all_volumes = np.append(prior_volumes, observed_overlap)
    if use_nig:
        log_vols = np.log(all_volumes + _LOG_OFFSET)  # log-transform observations
        n        = len(log_vols)
        s_bar_log = float(log_vols.mean())
        m2_log    = float(np.sum((log_vols - s_bar_log) ** 2))  # Welford M2 in log-space
        # Grid coordinates: running log-mean and running log-std (consistent with how
        # _precompute_nig_branch_probabilities and _bellman_expectation_full_grid_nig index the grid).
        initial_belief_mu    = s_bar_log
        initial_belief_sigma = max(float(np.sqrt(m2_log / n)), _SIGMA_MIN_NIG)
        current_overlap_probs = current_belief_probdist_nig(n, s_bar_log, m2_log)
    else:
        initial_belief_mu    = all_volumes.mean()
        initial_belief_sigma = std_calc(all_volumes, alpha, beta)
        current_overlap_probs = current_belief_probdist(initial_belief_mu, initial_belief_sigma)

    observation_count = fraction_index_today  # observations accumulated at this fraction (including today's)

    if fraction_index_today == number_of_fractions:  # last fraction: dose is fully determined by remaining budget
        recommended_dose = np.clip(remaining_ptv_dose, min_dose, max_dose)
        actual_value = np.zeros(1)  # no future fractions remain; immediate cost is accounted for separately via penalty_added
        current_fraction_policy = np.full(N_overlap, recommended_dose)
        return recommended_dose, actual_value, current_fraction_policy, current_overlap_probs

    # Expected future state value under the current belief — uses values[-1] (next fraction's value table).
    # Stage B uses log-space Welford updates; Stage A uses cc-space.
    if use_nig:
        future_value_prob = _bellman_expectation_nig(values[-1], current_overlap_probs, initial_belief_mu, initial_belief_sigma, observation_count)
    else:
        future_value_prob = _bellman_expectation(values[-1], _VOLUME_SPACE, current_overlap_probs, initial_belief_mu, initial_belief_sigma, observation_count)

    if fraction_index_today == 1:  # first fraction: dose_space starts at min_dose, so actions map directly onto grid
        # Immediate OAR cost for each action at today's actually observed overlap.
        immediate_cost = penalty_calc_single(action_space, min_dose, observed_overlap)

        # Future state value for each action — each action leads to a different next accumulated dose.
        # action_space and dose_space share the same step and starting point (both begin at min_dose for
        # fraction 1), so every action maps exactly to a dose_space grid point (use direct lookup).
        action_dose_indices = np.clip(np.searchsorted(dose_space, np.round(action_space, decimals=10), side='left'), 0, N_dose - 1)
        future_values = future_value_prob[action_dose_indices]  # value of state reached by each possible action

        # Total state value for every (action, overlap) combination = future value minus immediate OAR cost.
        # Used to determine the best action for each possible overlap outcome (the full policy).
        state_values_full_grid = -overlap_penalty + future_values
        current_fraction_policy = action_space[state_values_full_grid.argmax(axis=1)]  # optimal action for each overlap bin

        # Total state value for each action at the actually observed overlap = future value minus immediate OAR cost.
        actual_value = -immediate_cost + future_values
        recommended_dose = action_space[actual_value.argmax()]  # optimal action given today's observed overlap

    else:  # middle fraction: accumulated_dose may not align to dose_space grid, use interpolation
        # Determine feasible actions: daily doses, from min_dose, up to the largest dose that still leaves
        # enough budget for all remaining fractions to each deliver at least min_dose.
        min_dose_reserved_for_future = (number_of_fractions - fraction_index_today) * min_dose  # minimum dose budget that must be reserved for all remaining fractions after this one
        max_allowed_action = np.minimum(action_space[-1], prescribed_dose - accumulated_dose - min_dose_reserved_for_future)  # largest single-fraction dose that still leaves enough budget for future fractions
        max_action_index = np.abs(action_space - max_allowed_action).argmin()  # index into action_space of the largest allowed dose
        max_action_index = max(max_action_index, 1)  # clamp to at least index 1 so that at least one action is available
        feasible_actions = action_space[:max_action_index + 1]

        # Immediate OAR cost, for each feasible action, across all overlap bins.
        feasible_overlap_penalty = penalty_calc_matrix(feasible_actions, _VOLUME_SPACE, min_dose)  # named distinctly from the full-grid `overlap_penalty` used in the DP backward sweep above

        # Immediate OAR cost for each feasible action at today's actually observed overlap.
        immediate_cost = penalty_calc_single(feasible_actions, min_dose, observed_overlap)

        future_accumulated_dose = accumulated_dose + feasible_actions  # total accumulated dose after delivering each feasible action

        # Actions that would exceed the prescription are infeasible: replace their future dose
        # with the overdose sentinel (the last dose_space point) so linear_interp stays
        # in-bounds, then apply _INFEASIBILITY_SENTINEL to ensure they are never selected.
        future_accumulated_dose[future_accumulated_dose > prescribed_dose] = overdose_sentinel
        overdose_penalties = np.zeros(future_accumulated_dose.shape)
        overdose_penalties[future_accumulated_dose > prescribed_dose] = _INFEASIBILITY_SENTINEL

        # Future state value for each feasible action — each action leads to a different next accumulated dose.
        # Interpolation is required because accumulated_dose is an external caller input that may not lie
        # exactly on a dose_space grid point (unlike fraction 1, where future doses always land on-grid by construction).
        future_values = linear_interp(dose_space, future_value_prob, future_accumulated_dose)  # value of state reached by each possible action

        # Total state value for every (action, overlap) combination = future value minus immediate OAR cost.
        # Used to determine the best action for each possible overlap outcome (the full policy).
        state_values_full_grid = -feasible_overlap_penalty + future_values + overdose_penalties
        current_fraction_policy = feasible_actions[state_values_full_grid.argmax(axis=1)]  # optimal action for each overlap bin

        # Total state value for each action at the actually observed overlap = future value minus immediate OAR cost.
        actual_value = -immediate_cost + future_values + overdose_penalties
        recommended_dose = feasible_actions[actual_value.argmax()]  # optimal action given today's observed overlap

    return recommended_dose, actual_value, current_fraction_policy, current_overlap_probs


def adaptive_fractionation_core(fraction_index_today: int, volumes: np.ndarray, accumulated_dose: float, number_of_fractions: int, min_dose: float, max_dose: float, mean_dose: float, dose_steps: float, alpha: float, beta: float, use_nig: bool = False):
    """Belief-state DP solver. Computes the recommended dose for the current fraction.

    Optimizes fractionation by minimizing expected PTV underdosage cost: lower dose
    when PTV-OAR overlap is large, higher dose when overlap is small.

    Args:
        fraction_index_today (int): today's fraction number (1-indexed).
        volumes (np.ndarray): all overlap volumes observed so far, including the current fraction.
        accumulated_dose (float): total physical dose delivered to PTV before this fraction (Gy).
        number_of_fractions (int): total number of fractions.
        min_dose (float): minimum physical dose per fraction (Gy).
        max_dose (float): maximum physical dose per fraction (Gy).
        mean_dose (float): prescribed mean dose per fraction (Gy).
        dose_steps (float): dose grid resolution (Gy).
        alpha (float): shape parameter of the gamma prior on sigma.
        beta (float): scale parameter of the gamma prior on sigma.

    Returns:
        list: [policies, current_fraction_policy, volume_space, physical_dose, penalty_added,
               values, dose_space, current_overlap_probs, optimal_state_value]
              where optimal_state_value = np.max(actual_value) = the best total expected OAR cost
              from this fraction to the end of treatment (negative value; less negative = better).
              penalty_added = the immediate OAR cost incurred at this fraction for the recommended dose.
    """
    assert fraction_index_today >= 1, "fraction_index_today must be >= 1 (1-indexed)"
    assert fraction_index_today <= number_of_fractions, "fraction_index_today cannot exceed number_of_fractions"
    if fraction_index_today == 1:
        assert accumulated_dose == 0.0, (
            f"accumulated_dose must be 0.0 for the first fraction, got {accumulated_dose}. "
            "No dose can have been delivered before fraction 1."
        )

    volumes = np.asarray(volumes, dtype=float)
    observed_overlap = volumes[-1]

    ctx = _build_dp_context(fraction_index_today, number_of_fractions, accumulated_dose, min_dose, max_dose, mean_dose, dose_steps, use_nig=use_nig)
    recommended_dose, actual_value, current_fraction_policy, current_overlap_probs = _resolve_current_fraction(
        ctx, fraction_index_today, number_of_fractions, observed_overlap, volumes[:-1], alpha, beta
    )

    physical_dose = np.round(recommended_dose, 2)
    penalty_added = penalty_calc_single(physical_dose, min_dose, observed_overlap)
    optimal_state_value = np.max(actual_value)
    return [ctx['policies'], current_fraction_policy, _VOLUME_SPACE, physical_dose, penalty_added, ctx['values'], ctx['dose_space'], current_overlap_probs, optimal_state_value]


def adaptfx_full(volumes: list, number_of_fractions: int = DEFAULT_NUMBER_OF_FRACTIONS, min_dose: float = DEFAULT_MIN_DOSE, max_dose: float = DEFAULT_MAX_DOSE, mean_dose: float = DEFAULT_MEAN_DOSE, dose_steps: float = DEFAULT_DOSE_STEPS, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA, use_nig: bool = False):
    """Computes a full adaptive fractionation plan when all overlap volumes are given.

    Args:
        volumes (list): list of all volume overlaps observed
        number_of_fractions (int, optional): number of fractions delivered. Defaults to 5.
        min_dose (float, optional): minimum physical dose delivered in each fraction (Gy). Defaults to 6.0.
        max_dose (float, optional): maximum dose delivered in each fraction (Gy). Defaults to 10.0.
        mean_dose (float, optional): mean dose to be delivered over all fractions (Gy). Defaults to 8.0.
        dose_steps (float, optional): dose grid resolution (Gy). Defaults to 0.5.
        alpha (float, optional): shape parameter of the gamma prior on sigma. Defaults to 1.072846744379587.
        beta (float, optional): scale parameter of the gamma prior on sigma. Defaults to 0.7788684130749829.

    Returns:
        tuple:
            physical_doses (np.ndarray): optimal dose for each fraction (Gy); shape (number_of_fractions,).
            accumulated_doses (np.ndarray): accumulated PTV dose before each fraction (Gy); shape (number_of_fractions,).
                accumulated_doses[0] is always 0.0; accumulated_doses[k] = sum of physical_doses[:k].
            total_penalty (float): negated sum of OAR penalties across all fractions (≥ 0 for doses > min_dose).
                Positive means OAR dose was incurred. This is the negative of the sum of penalty_calc_single
                values, so callers can compare it directly with the uniform-fractionation OAR cost.
    """
    physical_doses = np.zeros(number_of_fractions)
    accumulated_doses = np.zeros(number_of_fractions)
    for index, frac in enumerate(range(1,number_of_fractions +1)):
        if frac != number_of_fractions:
            physical_dose = adaptive_fractionation_core(fraction_index_today = frac, volumes = np.array(volumes[:-number_of_fractions+frac]), accumulated_dose = accumulated_doses[index], number_of_fractions= number_of_fractions, min_dose = min_dose, max_dose = max_dose, mean_dose = mean_dose, dose_steps = dose_steps, alpha = alpha, beta = beta, use_nig=use_nig)[3]  # return index 3 is the recommended dose
            accumulated_doses[index+1] = accumulated_doses[index] + physical_dose
        else:
            physical_dose = adaptive_fractionation_core(fraction_index_today = frac, volumes = np.array(volumes),accumulated_dose = accumulated_doses[index], number_of_fractions= number_of_fractions, min_dose = min_dose, max_dose = max_dose, mean_dose = mean_dose, dose_steps = dose_steps, alpha = alpha, beta = beta, use_nig=use_nig)[3]  # final fraction uses full observed volume history
        physical_doses[index] = physical_dose
    total_penalty = 0
    for index, dose in enumerate(physical_doses):
        total_penalty -= penalty_calc_single(dose, min_dose, volumes[-number_of_fractions+index])
    return physical_doses, accumulated_doses, total_penalty


def precompute_plan(fraction_index_today: int, volumes: np.ndarray, accumulated_dose: float, number_of_fractions: int = DEFAULT_NUMBER_OF_FRACTIONS, min_dose: float = DEFAULT_MIN_DOSE, max_dose: float = DEFAULT_MAX_DOSE, mean_dose: float = DEFAULT_MEAN_DOSE, dose_steps: float = DEFAULT_DOSE_STEPS, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA, use_nig: bool = False):
    """Precomputes all possible delivered doses in the next fraction by looping through possible
    observed overlap volumes. Returning a df and two lists with the overlap volumes and
    the respective dose that would be delivered.

    Args:
        fraction_index_today (int): today's fraction number (1-indexed)
        volumes (np.ndarray): list of all volume overlaps observed so far
        accumulated_dose (float): accumulated physical dose in tumor
        number_of_fractions (int, optional): number of fractions given in total. Defaults to 5.
        min_dose (float, optional): minimum physical dose delivered in each fraction (Gy). Defaults to 6.0.
        max_dose (float, optional): maximum dose delivered in each fraction (Gy). Defaults to 10.0.
        mean_dose (float, optional): mean dose to be delivered over all fractions (Gy). Defaults to 8.0.
        dose_steps (float, optional): dose grid resolution (Gy). Defaults to 0.5.
        alpha (float, optional): shape parameter of the gamma prior on sigma. Defaults to 1.072846744379587.
        beta (float, optional): scale parameter of the gamma prior on sigma. Defaults to 0.7788684130749829.

    Returns:
        pd.Dataframe, lists: Returns a dataframe with volumes and respective doses, and volumes and doses separated in two lists.
    """
    volumes = np.asarray(volumes, dtype=float)
    if use_nig:
        # Stage B: use a fixed clinical upper bound as scan ceiling.
        # The Student-t predictive has heavier tails than the Gaussian, so a percentile-based
        # heuristic can blow up near-Cauchy. _NIG_SCAN_MAX_CC=35 cc covers the ACTION cohort
        # maximum overlap (~29 cc) with margin and is well within _VOLUME_SPACE (44 cc).
        distribution_max = _NIG_SCAN_MAX_CC
    else:
        std = std_calc(volumes, alpha, beta)
        distribution_params = (volumes.mean(), std)
        volume_space = get_state_space(distribution_params)  # 0.1th–99.9th percentile range of the current belief; used only to determine the scan stop criterion below
        # Minimum clinically relevant scan range: 6.5 cc covers the 99th percentile of the observed
        # 58-patient cohort overlap distribution, ensuring the table always spans a useful range even
        # when the patient's current belief is very narrow (e.g. only 1 observation so far).
        distribution_max = 6.5 if volume_space.max() < 6.5 else volume_space.max()
    min_dose_deliverable = min_dose_to_deliver(accumulated_dose=accumulated_dose, fractions_left=number_of_fractions - fraction_index_today + 1, prescribed_dose=mean_dose * number_of_fractions, min_dose=min_dose, max_dose=max_dose)

    # Run the DP backward sweep once — it is independent of which overlap will be observed
    # at fraction_index_today, so the result is shared across all candidate overlap values.
    ctx = _build_dp_context(fraction_index_today, number_of_fractions, accumulated_dose, min_dose, max_dose, mean_dose, dose_steps, use_nig=use_nig)

    # Scan always starts at 0.0 cc, regardless of the patient's observed overlap history.
    # This ensures the table is complete for any overlap that could be observed at the next fraction,
    # including very small values that fall far below the patient's current belief mean.
    volumes_to_check = [0.0]  # start from 0 cc and grow in 0.1 cc increments until we meet stop criteria
    predicted_policies = []

    while True:
        volume = volumes_to_check[-1]
        # For each candidate overlap, only the cheap current-fraction lookup is needed —
        # the expensive backward sweep was already done once above.
        recommended_dose, _, _, _ = _resolve_current_fraction(
            ctx, fraction_index_today, number_of_fractions, volume, volumes, alpha, beta
        )
        physical_dose = np.round(recommended_dose, 2)
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
