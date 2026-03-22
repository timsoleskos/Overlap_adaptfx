# -*- coding: utf-8 -*-
"""
In this file are all helper functions that are needed for the adaptive fractionation calculation
"""

import numpy as np
from scipy.stats import norm, gamma
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from .constants import SLOPE, INTERCEPT


# ── Mixture distribution ───────────────────────────────────────────────────────

class MixtureDist:
    """
    Mixture of two Normal distributions sharing the same location but with
    different scales:  w_low * N(mu, sigma_low) + w_high * N(mu, sigma_high).

    Provides .pdf(), .cdf(), and .ppf() so it is a drop-in replacement for
    scipy.stats.norm in get_state_space() and probdist().
    """
    def __init__(self, mu, sigma_low, sigma_high, w_high):
        self.mu         = mu
        self.sigma_low  = sigma_low
        self.sigma_high = sigma_high
        self.w_high     = float(w_high)
        self.w_low      = 1.0 - self.w_high
        self._d_low     = norm(loc=mu, scale=sigma_low)
        self._d_high    = norm(loc=mu, scale=sigma_high)

    def pdf(self, x):
        return self.w_low * self._d_low.pdf(x) + self.w_high * self._d_high.pdf(x)

    def cdf(self, x):
        return self.w_low * self._d_low.cdf(x) + self.w_high * self._d_high.cdf(x)

    def ppf(self, q):
        lo = min(self._d_low.ppf(1e-6), self._d_high.ppf(1e-6))
        hi = max(self._d_low.ppf(1 - 1e-6), self._d_high.ppf(1 - 1e-6))
        if np.isscalar(q):
            return brentq(lambda x: self.cdf(x) - q, lo, hi)
        return np.array([brentq(lambda x: self.cdf(x) - qi, lo, hi)
                         for qi in np.atleast_1d(q)])


def fit_mixture_params(overlap_array):
    """
    Fits a 2-component Gaussian mixture model to per-patient within-treatment
    standard deviations to identify 'stable' and 'volatile' patient sub-groups.

    Fitting is done in log-space for numerical stability. Falls back to a
    simple percentile split when sklearn is unavailable or the GMM components
    are too close to be meaningful (ratio < 1.5).

    Parameters
    ----------
    overlap_array : np.ndarray, shape (N, 6)
        [planning_vol, frac1, ..., frac5] per patient.

    Returns
    -------
    sigma_low : float    Typical within-patient std for stable patients.
    sigma_high : float   Typical within-patient std for volatile patients.
    pi_volatile : float  Prior probability that a patient is volatile.
    """
    treatment     = overlap_array[:, 1:]                           # (N, 5)
    patient_stds  = np.maximum(treatment.std(axis=1, ddof=1), 0.05)

    try:
        from sklearn.mixture import GaussianMixture
        log_stds = np.log(patient_stds).reshape(-1, 1)
        best, best_bic = None, np.inf
        for seed in range(30):
            gmm = GaussianMixture(n_components=2, random_state=seed, n_init=1)
            gmm.fit(log_stds)
            bic = gmm.bic(log_stds)
            if bic < best_bic:
                best_bic, best = bic, gmm
        idx_low, idx_high = np.argsort(best.means_.flatten())
        sigma_low   = np.exp(best.means_.flatten()[idx_low])
        sigma_high  = np.exp(best.means_.flatten()[idx_high])
        pi_volatile = float(best.weights_[idx_high])
        # Sanity: if components nearly identical, fall through to percentile method
        if sigma_high / sigma_low < 1.5:
            raise ValueError("Components too close; using percentile fallback.")
    except Exception:
        # Percentile fallback: bottom 75% → stable, top 25% → volatile
        threshold   = np.percentile(patient_stds, 75)
        sigma_low   = patient_stds[patient_stds <= threshold].mean()
        sigma_high  = patient_stds[patient_stds >  threshold].mean()
        pi_volatile = 0.25

    return sigma_low, sigma_high, pi_volatile


def mixture_posterior_weight(volumes, mu, sigma_low, sigma_high, pi_volatile):
    """
    Bayesian update of P(volatile | observations).

    Given k observed volumes and a mean estimate mu, compute the posterior
    probability that this patient belongs to the volatile component, using
    the Normal likelihood under each component.

    Parameters
    ----------
    volumes : array-like       Observed volumes (including planning scan).
    mu : float                 Current mean estimate (volumes.mean()).
    sigma_low, sigma_high : float   Component stds from fit_mixture_params.
    pi_volatile : float        Prior P(volatile) from fit_mixture_params.

    Returns
    -------
    p_volatile_post : float   Posterior P(volatile | volumes).
    """
    log_L_stable   = np.sum(norm.logpdf(volumes, loc=mu, scale=sigma_low))
    log_L_volatile = np.sum(norm.logpdf(volumes, loc=mu, scale=sigma_high))
    log_prior      = np.array([np.log(max(1 - pi_volatile, 1e-10)),
                                np.log(max(pi_volatile,     1e-10))])
    log_post       = np.array([log_L_stable + log_prior[0],
                                log_L_volatile + log_prior[1]])
    log_post      -= log_post.max()   # numerical stability
    post           = np.exp(log_post)
    return post[1] / post.sum()



def hierarchical_update(volumes, mu0, sigma2, tau2):
    """
    Two-level Normal-Normal hierarchical posterior predictive for the next
    overlap volume.

    Model
    -----
        x_ij | mu_i  ~  N(mu_i,  sigma2)   within-patient noise
        mu_i         ~  N(mu0,   tau2)      between-patient variation

    Given k = len(volumes) observations the posterior for mu_i is Normal, and
    the predictive for the next observation is:

        x_new | x ~ N(mu_post, sqrt(sigma2 + sigma2_mean_post))

    Parameters
    ----------
    volumes : array-like
        Observed overlap volumes for this patient so far.
    mu0 : float
        Population mean overlap volume (estimated from training cohort).
    sigma2 : float
        Within-patient variance (population estimate from training cohort).
    tau2 : float
        Between-patient variance of patient means (population estimate).

    Returns
    -------
    mu_post : float
        Posterior mean (shrunk toward mu0 when few observations available).
    sigma_pred : float
        Predictive std — within-patient noise plus remaining mean uncertainty.
    """
    k = len(volumes)
    x_bar = np.mean(volumes)
    lambda_prior = 1.0 / tau2
    lambda_data = k / sigma2
    lambda_total = lambda_prior + lambda_data
    mu_post = (lambda_prior * mu0 + lambda_data * x_bar) / lambda_total
    sigma2_mean_post = 1.0 / lambda_total
    sigma_pred = np.sqrt(sigma2 + sigma2_mean_post)
    return mu_post, sigma_pred


def fit_hierarchical_params(overlap_array):
    """
    Estimates population-level hyperparameters for the hierarchical model from
    a cohort of patients.  Only treatment fractions (columns 1-5) are used so
    that planning-scan bias does not contaminate the within-patient variance
    estimate.

    Parameters
    ----------
    overlap_array : np.ndarray, shape (N, 6)
        Each row: [planning_vol, frac1, frac2, frac3, frac4, frac5].

    Returns
    -------
    mu0 : float   Population mean of per-patient treatment means.
    sigma2 : float  Population mean of per-patient treatment variances.
    tau2 : float  Population variance of per-patient treatment means.
    """
    treatment = overlap_array[:, 1:]          # shape (N, 5)
    patient_means = treatment.mean(axis=1)
    patient_vars  = treatment.var(axis=1)
    mu0    = patient_means.mean()
    sigma2 = patient_vars.mean()
    tau2   = patient_means.var()
    return mu0, sigma2, tau2


def fit_planning_calibration(overlap_array):
    """
    Fits a linear calibration from planning scan volume to treatment mean volume
    using OLS regression: treatment_mean = a * planning_vol + b.

    Parameters
    ----------
    overlap_array : np.ndarray, shape (N, 6)
        Each row is one patient: [planning_vol, frac1, frac2, frac3, frac4, frac5]

    Returns
    -------
    a : float
        Slope of the linear calibration
    b : float
        Intercept of the linear calibration
    """
    planning = overlap_array[:, 0]
    treatment_means = overlap_array[:, 1:].mean(axis=1)
    a, b = np.polyfit(planning, treatment_means, 1)
    return a, b


def calibrate_planning_vol(planning_vol, a, b):
    """
    Applies linear calibration to a single planning scan volume.

    Parameters
    ----------
    planning_vol : float
    a : float
        Slope from fit_planning_calibration
    b : float
        Intercept from fit_planning_calibration

    Returns
    -------
    float
        Calibrated planning volume (clipped to >= 0)
    """
    return max(0.0, a * planning_vol + b)


def data_fit(data):
    """
    This function fits a normal distribution for the given data

    Parameters
    ----------
    data : array or list
        list with n elements for each observed overlap volume

    Returns
    -------
    frozen function
        normal distribution
    """
    mu, std = norm.fit(data)
    return norm(loc = mu, scale = std)

def hyperparam_fit(data):
    """
    This function fits the alpha and beta value for the prior

    Parameters
    ----------
    data : array
        a nxk matrix with n the amount of patints and k the amount of sparing factors per patient.

    Returns
    -------
    list
        alpha and beta hyperparameter.
    """
    vars = data.var(axis=1)
    alpha, loc, beta = gamma.fit(vars, floc=0)
    return [alpha, beta]

def std_calc(measured_data, alpha, beta):
    """
    calculates the most likely standard deviation for a list of k overlap volumes and a gamma prior
    measured_data: list/array with k sparing factors

    Parameters
    ----------
    measured_data : list/array
        list/array with k overlap volumes
    alpha : float
        shape of gamma distribution
    beta : float
        scale of gamma distrinbution

    Returns
    -------
    std : float
        most likely std based on the measured data and gamma prior

    """
    n = len(measured_data)
    std_values = np.arange(0.001, 10, 0.001)
    measured_variance = np.var(measured_data)
    likelihood_values = (
        std_values ** (alpha - 1)
        / std_values ** (n - 1)
        * np.exp(-1 / beta * std_values)
        * np.exp(-measured_variance / (2 * (std_values**2 / n)))
    )
    std = std_values[np.argmax(likelihood_values)]
    return std



def get_state_space(distribution):
    """
    This function spans the state space for different volumes based on a probability distribution

    Parameters
    ----------
    distribution : frozen function
        normal distribution

    Returns
    -------
    state_space: Array spanning from the 2% percentile to the 98% percentile with a normalized spacing to define 100 states
        np.array
    """
    lower_bound = distribution.ppf(0.001)
    upper_bound = distribution.ppf(0.999)

    return np.linspace(lower_bound,upper_bound,200)

def probdist(X,state_space):
    """
    This function produces a probability distribution based on the normal distribution X

    Parameters
    ----------
    X : scipy.stats._distn_infrastructure.rv_frozen
        distribution function.

    Returns
    -------
    prob : np.array
        array with probabilities for each sparing factor.

    """
    spacing = state_space[1]-state_space[0]
    upper_bounds = state_space + spacing/2
    lower_bounds = state_space - spacing/2
    prob = X.cdf(upper_bounds) - X.cdf(lower_bounds)
    return np.array(prob) #note: this will only add up to roughly 96% instead of 100%

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
    overlap_penalty : array
        The calculated penalty matrix for all dose-volume combinations
    """
    steepness = np.abs(intercept + slope * volume_space)
    overlap_penalty_linear = (np.outer(volume_space, (delivered_doses - min_dose)))
    overlap_penalty_quadratic = np.outer(steepness,(delivered_doses - min_dose)**2)/2
    overlap_penalty = overlap_penalty_linear + overlap_penalty_quadratic
    return overlap_penalty


def max_action(accumulated_dose, dose_space, goal):
    """
    Computes the maximal dose that can be delivered to the tumor in each fraction depending on the actual accumulated dose

    Parameters
    ----------
    accumulated_dose : float
        accumulated tumor dose so far.
    dose_space : list/array
        array with all discrete dose steps.
    goal : float
        prescribed tumor dose.
    Returns
    -------
    sizer : integer
        gives the size of the resized actionspace to reach the prescribed tumor dose.

    """
    max_action = min(max(dose_space), goal - accumulated_dose)
    sizer = np.argmin(np.abs(dose_space - max_action))
    sizer = 1 if sizer == 0 else sizer #Make sure that at least the minimum dose is delivered
    return sizer

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

    Args:
        fraction (int): number of actual fraction
        number_of_fractions (int): total number of fractions
        values (np.ndarray): remaining_fractions x volume_space x dose_space dimensional array with values for each volume/dose pair
        volume_space (np.ndarray): 1 dimensional array with all considered volume overlaps
        dose_space (np.ndarray): 1 dimensional array with all considered future accumulated doses

    Returns:
        matplotlib.fig: returns a figure with all values plotted as subfigures
    """
    values[values < -10000000000] = 10000000000
    min_Value = np.min(values)
    values[values == 10000000000] = 1.1*min_Value
    colormap = plt.cm.get_cmap('jet')
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

def min_dose_to_deliver(accumulated_dose: float, fractions_left: int, prescribed_dose: float, min_dose: float, max_dose: float = None) -> float:
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
    max_dose : float, optional
        maximal dose that can be delivered in one fraction, by default None

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