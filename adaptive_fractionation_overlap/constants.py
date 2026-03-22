# -*- coding: utf-8 -*-
"""
Constants used throughout the adaptive fractionation package.

This module contains global constants that are used across different modules
in the adaptive fractionation calculations.
"""

__all__ = [
    "SLOPE",
    "INTERCEPT",
    "DEFAULT_MIN_DOSE",
    "DEFAULT_MAX_DOSE",
    "DEFAULT_MEAN_DOSE",
    "DEFAULT_DOSE_STEPS",
    "DEFAULT_NUMBER_OF_FRACTIONS",
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "NIG_MU_0",
    "NIG_KAPPA_0",
    "NIG_ALPHA_0",
    "NIG_BETA_0",
    "NIG_LOG_OFFSET",
    "NIG_LOG_MU_0",
    "NIG_LOG_KAPPA_0",
    "NIG_LOG_ALPHA_0",
    "NIG_LOG_BETA_0",
]

# Default penalty function parameters
SLOPE = -0.65
INTERCEPT = 0.0

# Other common constants
DEFAULT_MIN_DOSE = 6.0
DEFAULT_MAX_DOSE = 10.0
DEFAULT_MEAN_DOSE = 8.0
DEFAULT_DOSE_STEPS = 0.5
DEFAULT_NUMBER_OF_FRACTIONS = 5

# Gamma distribution parameters
DEFAULT_ALPHA = 1.072846744379587
DEFAULT_BETA = 0.7788684130749829

# NIG (Normal-Inverse-Gamma) prior hyperparameters for Stage B — original-scale model (deprecated).
# Kept for reference; use_nig=True now uses the log-space model (NIG_LOG_*) instead.
# alpha0 ≈ 1.0 (constraint floor): the unconstrained optimum is below 1 (heavy tails),
# so alpha0=1 was enforced, giving Student-t dof=3 at fraction 1 (heaviest permissible).
# kappa0 ≈ 0.16: very weak prior on the mean — posterior mean converges to s_bar quickly.
NIG_MU_0    = 0.548993239693186   # prior mean (cc)
NIG_KAPPA_0 = 0.155866152519564   # prior pseudo-observations for the mean
NIG_ALPHA_0 = 1.000045399929762   # prior shape parameter (>= 1 enforced; dof_1 = 2*(alpha0+0.5) = 3)
NIG_BETA_0  = 0.156459260387199   # prior scale parameter

# Log-NIG (Normal-Inverse-Gamma in log-space) hyperparameters for Stage B.
# Fitted by empirical Bayes (sequential NIG log-marginal likelihood) on the 58-patient
# ACTION cohort, operating on log(overlap + NIG_LOG_OFFSET) observations.
# The log-transform converts the right-skewed overlap distribution (skewness=2.73) to
# near-symmetric (skewness=-0.15), making the NIG unimodal assumption valid.
# alpha0 = 2.74 is an interior solution (not constrained) giving Student-t dof=6.5 at
# fraction 1 — a well-behaved predictive with finite variance.
# Fitted via scripts/fit_nig_hyperparams.py with --log-space flag.
NIG_LOG_OFFSET  = 0.1                    # epsilon: log(v + NIG_LOG_OFFSET) transform (cc)
NIG_LOG_MU_0    = 0.666160089085257      # prior log-mean (nats)
NIG_LOG_KAPPA_0 = 0.001                  # prior pseudo-observations for the log-mean (near-zero: suppresses cross-term)
NIG_LOG_ALPHA_0 = 1.527375502971251      # prior shape; dof_1 = 2*(alpha0+0.5) = 4.05; refitted with kappa0 fixed
NIG_LOG_BETA_0  = 0.185147093672529      # prior log-scale (nats^2); refitted with kappa0 fixed