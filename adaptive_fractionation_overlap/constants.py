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

# precompute_plan scan parameters
# Minimum upper bound (cc) for the precompute volume scan. 6.5 cc covers the 99th
# percentile of the 58-patient cohort overlap distribution, so the table always
# spans a clinically relevant range even when the current belief is very narrow.
COHORT_MAX_OVERLAP_CC = 6.5
# Step size (cc) between successive candidate overlap volumes in the precompute scan.
PRECOMPUTE_SCAN_STEP_CC = 0.1
