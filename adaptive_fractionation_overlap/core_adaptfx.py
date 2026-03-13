"""
This is the core of adaptive fractionation that computes the optimal dose for each fraction
"""

__all__ = [
    "adaptive_fractionation_core",
    "adaptfx_full",
    "precompute_plan",
]
import numpy as np
import pandas as pd

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
)


def _linear_interp(x_values: np.ndarray, y_values: np.ndarray, query_points):
    """Fast linear interpolation for 1D/2D query arrays."""
    query = np.asarray(query_points)
    return np.interp(query.ravel(), x_values, y_values).reshape(query.shape)


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
    goal = number_of_fractions * mean_dose #dose to be reached
    actual_volume = volumes[-1]
    if fraction == 1:
        accumulated_dose = 0
    minimum_future = accumulated_dose + min_dose 
    std = std_calc(volumes, alpha, beta)
    distribution_params = (volumes.mean(), std)
    volume_space = get_state_space(distribution_params)
    probabilities = probdist(distribution_params,volume_space) #produce probabilities of the respective volumes
    volume_space = volume_space.clip(0) #clip the volume space to 0cc as negative volumes do not exist
    dose_space = np.arange(minimum_future,goal, dose_steps) #spans the dose space delivered to the tumor
    dose_space = np.concatenate((dose_space, [goal, goal + 0.05])) # add an additional state that overdoses and needs to be prevented
    bound = goal + 0.05
    delivered_doses = np.arange(min_dose,max_dose + 0.01,dose_steps) #spans the action space of all deliverable doses
    policies_overlap = np.zeros(len(volume_space))
    values = np.zeros(((number_of_fractions - fraction), len(dose_space), len(volume_space))) # 2d values list with first index being the accumulated dose and second being the overlap volume
    policies = np.zeros(((number_of_fractions - fraction), len(dose_space), len(volume_space)))
    if goal - accumulated_dose < (number_of_fractions + 1 - fraction) * min_dose:
        actual_policy = min_dose
        policies = np.ones(200)*actual_policy
        policies_overlap = np.ones(200)*actual_policy
        values = np.ones(((number_of_fractions - fraction), len(dose_space), len(volume_space))) * -1000000000000 
        actual_value = np.ones(1) * -1000000000000
    elif goal - accumulated_dose > (number_of_fractions + 1 - fraction) * max_dose:
        actual_policy = max_dose
        policies = np.ones(200)*actual_policy
        policies_overlap = np.ones(200)*actual_policy
        values = np.ones(((number_of_fractions - fraction), len(dose_space), len(volume_space))) * -1000000000000 
        actual_value = np.ones(1) * -1000000000000
    else:
        for state, fraction_state in enumerate(np.arange(number_of_fractions, fraction-1, -1)):
            if (state == number_of_fractions - 1):  # first fraction with no prior dose delivered so we dont loop through dose_space
                overlap_penalty = penalty_calc_matrix(delivered_doses, volume_space, min_dose) #This means only values over min_dose get a penalty. Values below min_dose do not get a reward
                actual_penalty = penalty_calc_single(delivered_doses, min_dose, actual_volume)
                future_value_prob = (values[state - 1] * probabilities).sum(axis=1)
                future_values = _linear_interp(dose_space, future_value_prob, delivered_doses)  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions
                values_actual_frac = -overlap_penalty + future_values
                policies_overlap = delivered_doses[values_actual_frac.argmax(axis = 1)]
                actual_value = -actual_penalty + future_values
                actual_policy = delivered_doses[actual_value.argmax()]

            elif (fraction_state == fraction and fraction != number_of_fractions):  # actual fraction but not first fraction
                delivered_doses_clipped = delivered_doses[0 : max_action(accumulated_dose, delivered_doses, goal)+1]
                overlap_penalty = penalty_calc_matrix(delivered_doses_clipped, volume_space, min_dose) #This means only values over min_dose get a penalty.
                actual_penalty = penalty_calc_single(delivered_doses_clipped, min_dose, actual_volume)
                future_doses = accumulated_dose + delivered_doses_clipped
                future_doses[future_doses > goal] = bound
                penalties = np.zeros(future_doses.shape)
                penalties[future_doses > goal] = -1000000000000
                future_value_prob = (values[state - 1] * probabilities).sum(axis=1)
                future_values = _linear_interp(dose_space, future_value_prob, future_doses)  # for each dose and volume overlap calculate the penalty of the action and add the future value. We will only have as many future values as we have doses (not volumes dependent)
                values_actual_frac = -overlap_penalty + future_values + penalties
                policies_overlap = delivered_doses_clipped[values_actual_frac.argmax(axis = 1)]
                actual_value =-actual_penalty + future_values + penalties
                actual_policy = delivered_doses_clipped[actual_value.argmax()]
        
            elif (fraction == number_of_fractions):  #actual fraction is also the final fraction we do not need to calculate any penalty as the last action is fixed. 
                best_action = goal - accumulated_dose
                if accumulated_dose > goal:
                    best_action = 0
                if best_action < min_dose:
                    best_action = min_dose
                if best_action > max_dose:
                    best_action = max_dose
                actual_policy = best_action
                actual_value = np.zeros(1)
        
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
                
    physical_dose = np.round(actual_policy,2)
    penalty_added = penalty_calc_single(physical_dose, min_dose, actual_volume)
    final_penalty = np.max(actual_value) - penalty_added
    return [policies, policies_overlap, volume_space, physical_dose, penalty_added, values, dose_space, probabilities, final_penalty]
    
   
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
