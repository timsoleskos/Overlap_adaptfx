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
# NOTE: alpha and beta now parameterise a Gamma prior on precision τ=1/σ² (units: cc⁻²).
# Values below are a rough translation from the original prior-on-σ (mean σ≈0.84 cc →
# mean τ≈1.42 cc⁻²); they need proper re-fitting to the 58-patient cohort.
DEFAULT_ALPHA = 1.072846744379587                  # OLD (prior on σ): 1.072846744379587
DEFAULT_BETA = 1.3228446444370503                  # OLD (prior on σ): 0.7788684130749829