# -*- coding: utf-8 -*-
"""
Constants used throughout the adaptive fractionation package.

This module contains global constants that are used across different modules
in the adaptive fractionation calculations.
"""

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

# Dynamic programming sentinel values
INFEASIBLE_VALUE = -1_000_000_000_000.0
OVERDOSE_STATE_OFFSET = 0.05
DOSE_GRID_EPSILON = 0.01

COHORT_MAX_OVERLAP_CC = 6.5  # 99th percentile overlap in the 58-patient cohort
PRECOMPUTE_SCAN_STEP_CC = 0.1
