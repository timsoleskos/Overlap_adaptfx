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
DEFAULT_ALPHA = 1.072846744379587
DEFAULT_BETA = 0.7788684130749829