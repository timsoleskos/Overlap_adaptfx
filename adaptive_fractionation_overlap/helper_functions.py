# -*- coding: utf-8 -*-
"""
In this file are all helper functions that are needed for the adaptive fractionation calculation
"""

__all__ = [
    "std_calc",
    "get_state_space",
    "probdist",
    "penalty_calc_single",
    "penalty_calc_matrix",
    "actual_policy_plotter",
    "analytic_plotting",
    "min_dose_to_deliver",
    "build_dose_decision_lines",
]

import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from .constants import SLOPE, INTERCEPT



def std_calc(measured_data, alpha, beta):
    """
    calculates the most likely standard deviation for a list of k overlap volumes and a gamma prior
    measured_data: list/array with k overlap volumes

    Parameters
    ----------
    measured_data : list/array
        list/array with k overlap volumes
    alpha : float
        shape of gamma distribution
    beta : float
        scale of gamma distribution

    Returns
    -------
    std : float
        most likely std based on the measured data and gamma prior

    Note: the search grid covers [0.001, 10) cc in 0.001 cc steps.  If the true MAP
    sigma exceeds ~9.999 cc the returned value silently saturates at 9.999 cc.
    In clinical practice (58-patient cohort max σ ≈ 3.3 cc) this cap is never reached.
    """
    n = len(measured_data)
    std_values = np.arange(0.001, 10, 0.001)
    measured_variance = np.var(measured_data)
    # OLD (Gamma prior on σ):
    # likelihood_values = (
    #     std_values ** (alpha - 1)
    #     / std_values ** (n - 1)
    #     * np.exp(-1 / beta * std_values)
    #     * np.exp(-measured_variance / (2 * (std_values**2 / n)))
    # )
    likelihood_values = (
        std_values ** (-n - 2 * alpha)
        * np.exp(-(n * measured_variance / 2 + 1 / beta) / std_values ** 2)
    )
    std = std_values[np.argmax(likelihood_values)]
    return std



def get_state_space(distribution):
    """
    Returns a 200-point linspace spanning the 0.1st to 99.9th percentile of a normal distribution.

    NOTE: this is NOT the DP's state space (_VOLUME_SPACE in belief_model.py).
    It is used only in precompute_plan to determine the scan stop criterion (distribution_max).

    Parameters
    ----------
    distribution : tuple(float, float)
        (mean, std) parameters of a normal distribution

    Returns
    -------
    state_space : np.ndarray
        200 evenly-spaced points from the 0.1th to 99.9th percentile.
    """
    mean_volume, std_volume = distribution
    lower_bound = norm.ppf(0.001, loc=mean_volume, scale=std_volume)
    upper_bound = norm.ppf(0.999, loc=mean_volume, scale=std_volume)

    return np.linspace(lower_bound,upper_bound,200)

def probdist(X,state_space):
    """
    This function produces a probability distribution based on the normal distribution X

    Note: unlike _P_BELIEF (belief_model.py), this function does NOT fold the left/right
    tails into the boundary bins, so the returned probabilities may sum to less than 1.0
    when the distribution tails extend beyond state_space.  See current_belief_probdist
    in belief_model.py for the version used by the Stage A DP solver.

    Parameters
    ----------
    X : tuple(float, float)
        (mean, std) parameters of a normal distribution.
    state_space : np.ndarray
        uniform grid of bin centres (must be equally spaced).

    Returns
    -------
    prob : np.array
        probability of each bin; sums to ≤ 1.0 (< 1.0 when tails extend beyond state_space).

    """
    spacing = state_space[1]-state_space[0]
    upper_bounds = state_space + spacing/2
    lower_bounds = state_space - spacing/2
    mean_volume, std_volume = X
    prob = norm.cdf(upper_bounds, loc=mean_volume, scale=std_volume) - norm.cdf(lower_bounds, loc=mean_volume, scale=std_volume)
    return prob

def penalty_calc_single(physical_dose, min_dose, actual_volume, intercept=INTERCEPT, slope=SLOPE):
    """
    This function calculates the penalty for the given dose and volume by adding the triangle arising from the dose gradient
    if the dose delivered is larger than the uniform fractionated dose.
    
    Parameters
    ----------
    physical_dose : float or array
        The physical dose delivered
    min_dose : float
        The minimum dose threshold
    actual_volume : float or array
        The actual overlap volume
    intercept : float, optional
        Penalty function intercept (default from constants)
    slope : float, optional
        Penalty function slope (default from constants)
        
    Returns
    -------
    penalty_added : float or array
        The calculated penalty
    """
    # Handle both scalar and array inputs
    physical_dose = np.asarray(physical_dose)
    min_dose = np.asarray(min_dose)
    
    # Calculate penalty for all cases
    steepness = np.abs(intercept + slope * actual_volume)
    penalty_added = (physical_dose - min_dose) * (actual_volume) + (physical_dose - min_dose)**2*steepness/2
    
    # Set penalty to 0 where physical_dose < min_dose
    penalty_added = np.where(physical_dose < min_dose, 0, penalty_added)
    
    # Return scalar if input was scalar, otherwise return array
    if np.isscalar(penalty_added) or penalty_added.shape == ():
        return float(penalty_added)
    return penalty_added


def penalty_calc_matrix(delivered_doses, volume_space, min_dose, intercept=INTERCEPT, slope=SLOPE):
    """
    This function calculates the penalty for the given dose and volume by adding the triangle arising from the dose gradient
    if the dose delivered is larger than the uniform fractionated dose.

    Parameters
    ----------
    delivered_doses : array
        Array of delivered doses
    volume_space : array
        Array of overlap volumes
    min_dose : float
        The minimum dose threshold
    intercept : float, optional
        Penalty function intercept (default from constants)
    slope : float, optional
        Penalty function slope (default from constants)

    Returns
    -------
    overlap_penalty : array, shape (len(volume_space), len(delivered_doses))
        The calculated penalty matrix for all dose-volume combinations.

    Note: unlike penalty_calc_single, this function does NOT zero out entries where
    delivered_doses < min_dose; those entries will be negative.  The caller is
    responsible for masking or ignoring infeasible (below-min-dose) actions.
    """
    steepness = np.abs(intercept + slope * volume_space)
    overlap_penalty_linear = (np.outer(volume_space, (delivered_doses - min_dose)))
    overlap_penalty_quadratic = np.outer(steepness,(delivered_doses - min_dose)**2)/2
    overlap_penalty = overlap_penalty_linear + overlap_penalty_quadratic
    return overlap_penalty


def actual_policy_plotter(policies_overlap: np.ndarray,volume_space: np.ndarray, probabilities: np.ndarray = None):
    """plots the actual policy given the overlap in volume space and the policies in policies overlap

    Args:
        policies_overlap (np.ndarray): policy for each overlap
        volume_space (np.ndarray): considered overlaps
        probabilities (np.ndarray): probability distribution of overlaps

    Returns:
        matplotlib figure: a figure with the actual policy plotted
    """
    color = 'tab:red'
    fig, ax = plt.subplots()
    ax.plot(volume_space,policies_overlap, label = 'optimal dose', color = color)
    ax.set_xlabel('Volume overlap in cc') 
    ax.set_ylabel('optimal dose')
    ax.set_title('policy of actual fraction')
    
    if probabilities is not None:
        color = 'tab:blue'
        ax2 = ax.twinx()
        ax2.set_ylabel('probability')
        ax2.plot(volume_space,probabilities, label = 'probabilities', color = color)
    fig.legend()
    return fig

def analytic_plotting(fraction: int, number_of_fractions: int, values: np.ndarray, volume_space: np.ndarray, dose_space: np.ndarray):
    """plots all future values given the values calculated by adaptive_fractionation_core.
    Only available for fractions 1 - (number of fractions - 1)

    INCOMPATIBLE WITH STAGE A OUTPUT: adaptive_fractionation_core (Stage A, belief-state DP)
    returns a 5D values array with shape (remaining_fractions, N_dose, N_overlap, N_mu, N_sigma).
    This function expects a 3D array with shape (remaining_fractions, volume_space, dose_space)
    from the earlier Stage 0 solver.  Passing a 5D array raises ValueError.

    Args:
        fraction (int): number of actual fraction
        number_of_fractions (int): total number of fractions
        values (np.ndarray): remaining_fractions x volume_space x dose_space dimensional array with values for each volume/dose pair
        volume_space (np.ndarray): 1 dimensional array with all considered volume overlaps
        dose_space (np.ndarray): 1 dimensional array with all considered future accumulated doses

    Returns:
        matplotlib.fig: returns a figure with all values plotted as subfigures

    Raises:
        ValueError: if values is not a 3D array (e.g. 5D Stage A output is passed).
    """
    if values.ndim != 3:
        raise ValueError(
            f"analytic_plotting expects a 3D values array (remaining_fractions × volume_space × dose_space), "
            f"got shape {values.shape} ({values.ndim}D). "
            "The Stage A belief-state DP (adaptive_fractionation_core) returns a 5D values array "
            "and is not compatible with this plotting function."
        )
    values = values.copy()
    values[values < -10000000000] = 10000000000
    min_Value = np.min(values)
    values[values == 10000000000] = 1.1*min_Value
    colormap = matplotlib.colormaps['jet']
    number_of_plots = number_of_fractions - fraction
    fig, axs = plt.subplots(1,number_of_plots, figsize = (number_of_plots*10,10))
    if number_of_plots > 1:
        for index, ax in enumerate(axs): 
            img = ax.imshow(values[number_of_plots - index-1],extent = [volume_space.min(), volume_space.max(), dose_space.max(),dose_space.min()],cmap=colormap,aspect = 'auto')
            ax.set_title(f'value of fraction {fraction + index + 1}', fontsize = 24)
            ax.set_xlabel('overlap volume', fontsize = 24)
            ax.set_ylabel('accumulated dose', fontsize = 24)
            ax.tick_params(axis='both', which='both', labelsize=20)
        cbar = plt.colorbar(img, ax = ax)  
        cbar.set_label('state value', fontsize = 24)
    else:
        img = axs.imshow(values[0], extent = [volume_space.min(), volume_space.max(), dose_space.max(),dose_space.min()],cmap=colormap,aspect = 'auto')
        axs.set_title(f'value of fraction {fraction + 1}', fontsize = 24)
        axs.set_xlabel('overlap volume', fontsize = 24)
        axs.set_ylabel('accumulated dose', fontsize = 24)
        axs.tick_params(axis='both', which='both', labelsize=20)
        cbar = plt.colorbar(img, ax = axs)  
        cbar.set_label('state value', fontsize = 24) 

    return fig

def linear_interp(x_values: np.ndarray, y_values: np.ndarray, query_points):
    """Fast linear interpolation for 1D/2D query arrays."""
    query = np.asarray(query_points)
    return np.interp(query.ravel(), x_values, y_values).reshape(query.shape)


def nearest_idx(values, grid):
    """Return nearest-grid-point indices for every element in *values*.

    Uses searchsorted (O(n log G)) instead of argmin (O(n*G)), so it scales
    well when the grid is large.  *grid* must be sorted ascending.
    """
    flat = np.asarray(values).ravel()
    hi = np.searchsorted(grid, flat, side='left').clip(1, len(grid) - 1)
    lo = hi - 1
    idx = np.where(flat - grid[lo] <= grid[hi] - flat, lo, hi)
    return idx.reshape(np.asarray(values).shape)


def min_dose_to_deliver(accumulated_dose: float, fractions_left: int, prescribed_dose: float, min_dose: float, max_dose: float) -> float:
    """
    This function calculates the minimal dose that needs to be delivered in the current fraction to still reach the goal

    Parameters
    ----------
    accumulated_dose : float
        accumulated dose so far
    fractions_left : int
        number of fractions left including the current one
    min_dose : float
        minimal dose that can be delivered in one fraction
    max_dose : float
        maximal dose that can be delivered in one fraction

    Returns
    -------
    float
        minimal dose that needs to be delivered in the current fraction
    """
    min_dose_to_deliver_calculated = (prescribed_dose - accumulated_dose) - ((fractions_left - 1) *max_dose)
    return min_dose if min_dose_to_deliver_calculated < min_dose else min_dose_to_deliver_calculated


def _format_number(value: float) -> str:
    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def build_dose_decision_lines(volume_x_dose) -> list:
    """Build human-readable dose decision rules from a volume-dose table."""
    if volume_x_dose is None or volume_x_dose.empty:
        return []

    volumes = volume_x_dose["volume"].to_numpy()
    doses = volume_x_dose["dose"].to_numpy()
    if len(volumes) == 0:
        return []

    lines = []
    start_idx = 0
    for i in range(1, len(doses)):
        if doses[i] != doses[i - 1]:
            start_vol = volumes[start_idx]
            end_vol = volumes[i]
            dose = doses[i - 1]
            if start_idx == 0:
                lines.append(
                    f"- Volume < {_format_number(end_vol)} cc: deliver {_format_number(dose)} Gy"
                )
            else:
                lines.append(
                    f"- {_format_number(start_vol)} cc ≤ volume < {_format_number(end_vol)} cc: deliver {_format_number(dose)} Gy"
                )
            start_idx = i

    start_vol = volumes[start_idx]
    dose = doses[start_idx]
    lines.append(
        f"- Volume ≥ {_format_number(start_vol)} cc: deliver {_format_number(dose)} Gy"
    )
    return lines
