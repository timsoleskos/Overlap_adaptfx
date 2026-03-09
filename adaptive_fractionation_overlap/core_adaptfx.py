"""
This is the core of adaptive fractionation that computes the optimal dose for each fraction
"""
import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

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
    _linear_interp,
    _nearest_idx,
)


# Belief-state grid constants (Stage A reduced-state)
# Non-uniform mu grid optimized for the observed 58-patient cohort prefix-mean distribution
# under a fixed 280-point budget (~2x prior dense-grid compute target).
# Segment 1 [0.00, 1.00]  with 70 points  (step ~0.0145 cc)
# Segment 2 [1.05, 4.00]  with 73 points  (step ~0.0410 cc)
# Segment 3 [4.10, 10.00] with 79 points  (step ~0.0756 cc)
# Segment 4 [10.20, 16.00] with 28 points (step ~0.2148 cc)
# Segment 5 [16.50, 30.00] with 30 points (step ~0.4655 cc)
_MU_GRID = np.unique(np.concatenate([
    np.linspace(0.0, 1.0, 70),
    np.linspace(1.05, 4.0, 73),
    np.linspace(4.1, 10.0, 79),
    np.linspace(10.2, 16.0, 28),
    np.linspace(16.5, 30.0, 30),
]))  # 280 grid points total
# Non-uniform sigma grid: fine resolution in [0, 0.7] cc where 75% of patients fall,
# coarser in the tail.  Range extended to 3.5 cc to avoid clipping outlier patients
# (observed max σ ≈ 3.3 cc on the 58-patient cohort).
# Segment 1 [0.05, 0.70]  15 pts  step ~0.046 cc  (p0–p75 of clinical σ)
# Segment 2 [0.80, 1.80]   8 pts  step ~0.143 cc  (p75–p90)
# Segment 3 [2.00, 3.50]   7 pts  step ~0.250 cc  (tail)
_SIGMA_GRID = np.unique(np.concatenate([
    np.linspace(0.05, 0.7, 15),
    np.linspace(0.8,  1.8,  8),
    np.linspace(2.0,  3.5,  7),
]))  # 30 grid points total
_SIGMA_MIN = float(_SIGMA_GRID[0])

# Fixed wide overlap state space for the DP: must cover the full belief grid range so
# that p_belief[mi, si, :].sum() ≈ 1 for all (mi, si).  Patient-specific volume_space
# would be valid for one fraction's initial belief but can have degenerate bin widths
# (e.g. spacing=0 when many negative linspace values are clipped to 0), and—crucially—
# the narrow range fails to represent future overlap outcomes for beliefs far from the
# patient's current distribution, corrupting the DP.
_VOLUME_SPACE = np.linspace(0.0, _MU_GRID[-1] + 4 * _SIGMA_GRID[-1], 200)

# Precompute branch probabilities once at module load (all inputs are module-level constants).
# p_belief[mi, si, j] = P(overlap bin j | belief (mu_grid[mi], sigma_grid[si]))
# Left/right tails assigned to first/last bin so probabilities sum to 1.
def _compute_p_belief():
    spacing = _VOLUME_SPACE[1] - _VOLUME_SPACE[0]
    ub = _VOLUME_SPACE + spacing / 2
    lb = _VOLUME_SPACE - spacing / 2
    p = (
        _norm.cdf(ub[None, None, :], loc=_MU_GRID[:, None, None], scale=_SIGMA_GRID[None, :, None])
        - _norm.cdf(lb[None, None, :], loc=_MU_GRID[:, None, None], scale=_SIGMA_GRID[None, :, None])
    )
    p[:, :, 0] += _norm.cdf(lb[0], loc=_MU_GRID[:, None], scale=_SIGMA_GRID[None, :])
    p[:, :, -1] += 1.0 - _norm.cdf(ub[-1], loc=_MU_GRID[:, None], scale=_SIGMA_GRID[None, :])
    return p

_P_BELIEF = _compute_p_belief()  # shape: (N_mu, N_sigma, N_overlap), computed once


def _belief_update_1d(mu, sigma, volume_space, n_t):
    """Welford running-moment update for scalar belief (mu, sigma) over all overlap branches.

    At fraction n_t we have seen n_t observations; o' is the (n_t+1)-th.
    Returns next_mi, next_si: (N_overlap,) int arrays of nearest grid indices.
    """
    mu_prime = (n_t * mu + volume_space) / (n_t + 1)
    sigma2_prime = (n_t * sigma ** 2 + (volume_space - mu) * (volume_space - mu_prime)) / (n_t + 1)
    sigma_prime = np.sqrt(np.maximum(sigma2_prime, _SIGMA_MIN ** 2))
    mu_prime = np.clip(mu_prime, _MU_GRID[0], _MU_GRID[-1])
    sigma_prime = np.clip(sigma_prime, _SIGMA_GRID[0], _SIGMA_GRID[-1])
    next_mi = _nearest_idx(mu_prime, _MU_GRID)
    next_si = _nearest_idx(sigma_prime, _SIGMA_GRID)
    return next_mi, next_si


def _future_value_1d(values_prev, volume_space, p_branch, mu, sigma, n_t):
    """Expected future value for a single scalar starting belief (mu, sigma).

    Computes: sum_j p_branch[j] * values_prev[d, j, next_mi[j], next_si[j]]

    values_prev: (N_dose, N_overlap, N_mu, N_sigma)
    p_branch:    (N_overlap,) branch probabilities from current belief
    Returns:     (N_dose,)
    """
    next_mi, next_si = _belief_update_1d(mu, sigma, volume_space, n_t)
    branch_vals = values_prev[:, np.arange(len(volume_space)), next_mi, next_si]  # (N_dose, N_overlap)
    return (branch_vals * p_branch[None, :]).sum(axis=1)


def policy_calc(fixed_mean_volume: float, fixed_std: float, number_of_fractions: int = DEFAULT_NUMBER_OF_FRACTIONS, min_dose: float = DEFAULT_MIN_DOSE, max_dose: float = DEFAULT_MAX_DOSE, mean_dose:float = DEFAULT_MEAN_DOSE, dose_steps: float = DEFAULT_DOSE_STEPS):
    """The core function computes the optimal dose for a single fraction.
    The function optimizes the fractionation based on an objective function
    which aims to maximize the tumor coverage, i.e. minimize the dose when
    PTV-OAR overlap is large and maximize the dose when the overlap is small.

    Args:
        fraction (int): number of actual fraction
        volumes (np.ndarray): list of all volume overlaps observed so far
        accumulated_dose (float): accumulated physical dose in tumor
        number_of_fractions (int, optional): number of fractions given in total. Defaults to 5.
        min_dose (float, optional): minimum phyical dose delivered in each fraction. Defaults to 7.5.
        max_dose (float, optional): maximum dose delivered in each fraction. Defaults to 9.5.
        mean_dose (int, optional): mean dose to be delivered over all fractions. Defaults to 8.
    Returns:
        numpy arrays and floats: returns 9 arrays: policies (all future policies), policies_overlap (policies of the actual overlaps),
        volume_space (all considered overlap volumes), physical_dose (physical dose to be delivered in the actual fraction),
        penalty_added (penalty added in the actual fraction if physical_dose is applied), values (values of all future fractions. index 0 is the last fraction),
        probabilits (probability of each overlap volume to occure), final_penalty (projected final penalty starting from the actual fraction)
    """
    goal = number_of_fractions * mean_dose #dose to be reached

    distribution_params = (fixed_mean_volume, fixed_std)  # Keep as (mean, std) to avoid creating a frozen scipy distribution object (expensive in tight loops).
    accumulated_dose = 0
    minimum_future = accumulated_dose + min_dose

    volume_space = get_state_space(distribution_params)  # Helper expects (mean, std) tuple parameters.
    probabilities = probdist(distribution_params,volume_space) #produce probabilities of the respective volumes
    volume_space = volume_space.clip(0) #clip the volume space to 0cc as negative volumes do not exist
    dose_space = np.arange(minimum_future,goal, dose_steps) #spans the dose space delivered to the tumor
    dose_space = np.concatenate((dose_space, [goal, goal + 0.05])) # add an additional state that overdoses and needs to be prevented
    bound = goal + 0.05
    delivered_doses = np.arange(min_dose,max_dose + 0.01,dose_steps) #spans the action space of all deliverable doses
    policies_overlap = np.zeros(len(volume_space))
    values = np.zeros(((number_of_fractions - 1), len(dose_space), len(volume_space))) # 2d values list with first index being the accumulated dose and second being the overlap volume
    policies = np.zeros(((number_of_fractions - 1), len(dose_space), len(volume_space)))
    if goal - accumulated_dose < (number_of_fractions + 1 - 1) * min_dose:
        actual_policy = min_dose
        policies = np.ones(200)*actual_policy
        policies_overlap = np.ones(200)*actual_policy
        values = np.ones(((number_of_fractions - 1), len(dose_space), len(volume_space))) * -1000000000000 
    elif goal - accumulated_dose > (number_of_fractions + 1 - 1) * max_dose:
        actual_policy = max_dose
        policies = np.ones(200)*actual_policy
        policies_overlap = np.ones(200)*actual_policy
        values = np.ones(((number_of_fractions - 1), len(dose_space), len(volume_space))) * -1000000000000 
    else:
        for state in range(number_of_fractions):
            if (state == number_of_fractions - 1):  # first fraction with no prior dose delivered so we dont loop through dose_space
                overlap_penalty = penalty_calc_matrix(delivered_doses, volume_space, min_dose) #This means only values over min_dose get a penalty. Values below min_dose do not get a reward
                future_value_prob = (values[state - 1] * probabilities).sum(axis=1)
                future_values = _linear_interp(dose_space, future_value_prob, delivered_doses)  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions
                values_actual_frac = -overlap_penalty + future_values
                policies_overlap = delivered_doses[values_actual_frac.argmax(axis = 1)]
            else: #any fraction that is not the actual one
                future_value_prob = (values[state - 1] * probabilities).sum(axis=1)
                if state != 0:
                    overlap_penalty = penalty_calc_matrix(delivered_doses, volume_space, min_dose) #This means only values over min_dose get a penalty.
                    max_allowed_actions = np.minimum(delivered_doses[-1], goal - dose_space)
                    max_action_indices = np.abs(delivered_doses.reshape(1, -1) - max_allowed_actions.reshape(-1, 1)).argmin(axis=1)
                    max_action_indices = np.where(max_action_indices == 0, 1, max_action_indices)
                    valid_actions = np.arange(delivered_doses.size).reshape(1, -1) <= max_action_indices.reshape(-1, 1)

                    future_doses = dose_space.reshape(-1, 1) + delivered_doses.reshape(1, -1)
                    overdosed = future_doses > goal
                    future_doses = np.where(overdosed, bound, future_doses) #all overdosing doses are set to the penalty state
                    future_values = _linear_interp(dose_space, future_value_prob, future_doses)  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
                    penalties = np.zeros(future_doses.shape)
                    penalties[overdosed] = -1000000000000
                    vs = -overlap_penalty.T.reshape(1, delivered_doses.size, len(volume_space)) + future_values.reshape(len(dose_space), delivered_doses.size, 1) + penalties.reshape(len(dose_space), delivered_doses.size, 1)
                    vs = np.where(valid_actions.reshape(len(dose_space), delivered_doses.size, 1), vs, np.finfo(np.float64).min)
                    policies[state] = delivered_doses[vs.argmax(axis=1)]
                    values[state] = vs.max(axis=1)

                else:  # last fraction when looping, only give the final penalty
                    best_actions = goal - dose_space
                    best_actions[best_actions > max_dose] = max_dose
                    best_actions[best_actions < min_dose] = min_dose
                    future_accumulated_dose = dose_space + best_actions
                    last_penalty = penalty_calc_single(best_actions.reshape(-1, 1), min_dose, volume_space.reshape(1, -1))
                    underdose_penalty = np.zeros(future_accumulated_dose.shape)
                    overdose_penalty = np.zeros(future_accumulated_dose.shape)
                    underdose_penalty[np.round(future_accumulated_dose,2) < goal] = -1000000000000 #in theory one can change this such that underdosing is penalted linearly
                    overdose_penalty[np.round(future_accumulated_dose,2) > goal] = -1000000000000
                    values[state] = (- last_penalty + underdose_penalty.reshape(-1, 1) + overdose_penalty.reshape(-1, 1))  # gives the value of each action for all sparing factors. elements 0-len(sparingfactors) are the Values for
                    policies[state] = best_actions.reshape(-1, 1)
                

    return [policies, policies_overlap, volume_space, values, dose_space, probabilities]
    


def adaptive_fractionation_core(fraction: int, volumes: np.ndarray, accumulated_dose: float, number_of_fractions: int = DEFAULT_NUMBER_OF_FRACTIONS, min_dose: float = DEFAULT_MIN_DOSE, max_dose: float = DEFAULT_MAX_DOSE, mean_dose:float = DEFAULT_MEAN_DOSE, dose_steps: float = DEFAULT_DOSE_STEPS, alpha: float = DEFAULT_ALPHA, beta:float = DEFAULT_BETA):
    """The core function computes the optimal dose for a single fraction.
    The function optimizes the fractionation based on an objective function
    which aims to maximize the tumor coverage, i.e. minimize the dose when
    PTV-OAR overlap is large and maximize the dose when the overlap is small.

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
        numpy arrays and floats: returns 9 arrays: policies (all future policies), policies_overlap (policies of the actual overlaps),
        volume_space (all considered overlap volumes), physical_dose (physical dose to be delivered in the actual fraction),
        penalty_added (penalty added in the actual fraction if physical_dose is applied), values (values of all future fractions. index 0 is the last fraction),
        probabilits (probability of each overlap volume to occure), final_penalty (projected final penalty starting from the actual fraction)
    """
    goal = number_of_fractions * mean_dose
    actual_volume = volumes[-1]
    if fraction == 1:
        accumulated_dose = 0
    minimum_future = accumulated_dose + min_dose

    # Belief state initialisation from observed volumes (Stage A reduced-state)
    mu_start = volumes.mean()
    sigma_start = std_calc(volumes, alpha, beta)
    # Use the fixed wide volume space for the DP so that p_belief sums to ~1 for every
    # belief in the grid and future overlap outcomes are correctly represented regardless
    # of the current fraction's initial belief.
    volume_space = _VOLUME_SPACE
    initial_probs = probdist((mu_start, sigma_start), volume_space)

    dose_space = np.arange(minimum_future, goal, dose_steps)
    dose_space = np.concatenate((dose_space, [goal, goal + 0.05]))
    bound = goal + 0.05
    delivered_doses = np.arange(min_dose, max_dose + 0.01, dose_steps)

    N_mu = len(_MU_GRID)
    N_sigma = len(_SIGMA_GRID)
    N_overlap = len(volume_space)
    N_dose = len(dose_space)

    # Branch probabilities: use the module-level precomputed constant (volume_space == _VOLUME_SPACE).
    p_belief = _P_BELIEF  # shape: (N_mu, N_sigma, N_overlap)

    n_states = number_of_fractions - fraction
    values = np.zeros((n_states, N_dose, N_overlap, N_mu, N_sigma))
    policies = np.zeros((n_states, N_dose, N_overlap, N_mu, N_sigma))
    policies_overlap = np.zeros(N_overlap)

    if goal - accumulated_dose < (number_of_fractions + 1 - fraction) * min_dose:
        actual_policy = min_dose
        policies_overlap = np.ones(N_overlap) * actual_policy
        if n_states > 0:
            values[:] = -1e12
        actual_value = np.ones(1) * -1e12
    elif goal - accumulated_dose > (number_of_fractions + 1 - fraction) * max_dose:
        actual_policy = max_dose
        policies_overlap = np.ones(N_overlap) * actual_policy
        if n_states > 0:
            values[:] = -1e12
        actual_value = np.ones(1) * -1e12
    else:
        min_float = np.finfo(np.float64).min
        for state, fraction_state in enumerate(np.arange(number_of_fractions, fraction - 1, -1)):
            n_t = int(fraction_state)

            if state == number_of_fractions - 1:
                # Actual first fraction (only reached when fraction == 1).
                overlap_penalty = penalty_calc_matrix(delivered_doses, volume_space, min_dose)
                actual_penalty = penalty_calc_single(delivered_doses, min_dose, actual_volume)
                future_value_prob = _future_value_1d(
                    values[state - 1], volume_space, initial_probs, mu_start, sigma_start, n_t
                )
                future_values = _linear_interp(dose_space, future_value_prob, delivered_doses)
                values_actual_frac = -overlap_penalty + future_values
                policies_overlap = delivered_doses[values_actual_frac.argmax(axis=1)]
                actual_value = -actual_penalty + future_values
                actual_policy = delivered_doses[actual_value.argmax()]

            elif fraction_state == fraction and fraction != number_of_fractions:
                # Actual fraction (not first, not last).
                delivered_doses_clipped = delivered_doses[0: max_action(accumulated_dose, delivered_doses, goal) + 1]
                overlap_penalty = penalty_calc_matrix(delivered_doses_clipped, volume_space, min_dose)
                actual_penalty = penalty_calc_single(delivered_doses_clipped, min_dose, actual_volume)
                future_doses = accumulated_dose + delivered_doses_clipped
                future_doses[future_doses > goal] = bound
                penalties = np.zeros(future_doses.shape)
                penalties[future_doses > goal] = -1e12
                future_value_prob = _future_value_1d(
                    values[state - 1], volume_space, initial_probs, mu_start, sigma_start, n_t
                )
                future_values = _linear_interp(dose_space, future_value_prob, future_doses)
                values_actual_frac = -overlap_penalty + future_values + penalties
                policies_overlap = delivered_doses_clipped[values_actual_frac.argmax(axis=1)]
                actual_value = -actual_penalty + future_values + penalties
                actual_policy = delivered_doses_clipped[actual_value.argmax()]

            elif fraction == number_of_fractions:
                # Actual fraction is the last fraction; action is fixed.
                best_action = goal - accumulated_dose
                if accumulated_dose > goal:
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
                    next_mi = _nearest_idx(mu_prime, _MU_GRID)
                    next_si = _nearest_idx(sigma_prime, _SIGMA_GRID)

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
                    overlap_penalty = penalty_calc_matrix(delivered_doses, volume_space, min_dose)
                    max_allowed_actions = np.minimum(delivered_doses[-1], goal - dose_space)
                    max_action_indices = np.abs(
                        delivered_doses[None, :] - max_allowed_actions[:, None]
                    ).argmin(axis=1)
                    max_action_indices = np.where(max_action_indices == 0, 1, max_action_indices)
                    valid_actions = (
                        np.arange(delivered_doses.size)[None, :] <= max_action_indices[:, None]
                    )  # (N_dose, N_action)

                    future_doses = dose_space[:, None] + delivered_doses[None, :]  # (N_dose, N_action)
                    overdosed = future_doses > goal
                    future_doses_clipped = np.where(overdosed, bound, future_doses)
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
                    ).reshape(N_dose, len(delivered_doses), N_mu, N_sigma)

                    # Step 5: value/policy update.
                    # vs_base_T: (N_dose, N_mu, N_sigma, N_action) — action axis last for cache-friendly argmax.
                    # Preallocated buffers avoid per-iteration heap allocations.
                    # Flat-base index replaces take_along_axis (avoids one temporary array per j).
                    vs_base = future_values_full + overdose_pens[:, :, None, None]
                    vs_base = np.where(valid_actions[:, :, None, None], vs_base, min_float)
                    vs_base_T = vs_base.transpose(0, 2, 3, 1).copy()  # (N_dose, N_mu, N_sigma, N_action) contiguous
                    N_action = len(delivered_doses)
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
                        policies_state[:, j, :, :] = delivered_doses[ai_buf]
                    values[state] = values_state
                    policies[state] = policies_state

                else:
                    # --- Terminal fraction (state == 0) ---
                    best_actions = goal - dose_space
                    best_actions[best_actions > max_dose] = max_dose
                    best_actions[best_actions < min_dose] = min_dose
                    future_accumulated_dose = dose_space + best_actions
                    last_penalty = penalty_calc_single(
                        best_actions[:, None], min_dose, volume_space[None, :]
                    )  # (N_dose, N_overlap)
                    underdose_penalty = np.zeros(future_accumulated_dose.shape)
                    overdose_penalty = np.zeros(future_accumulated_dose.shape)
                    underdose_penalty[np.round(future_accumulated_dose, 2) < goal] = -1e12
                    overdose_penalty[np.round(future_accumulated_dose, 2) > goal] = -1e12
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
    penalty_added = penalty_calc_single(physical_dose, min_dose, actual_volume)
    final_penalty = np.max(actual_value) - penalty_added
    return [policies, policies_overlap, volume_space, physical_dose, penalty_added, values, dose_space, initial_probs, final_penalty]
    
   
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
