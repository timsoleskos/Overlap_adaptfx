"""
Test coverage improvements for the belief-state DP branch.

Covers gaps not addressed by the existing suite:

  1. std_calc            — prior-mode dominance at n=1; variance ordering.
  2. min_dose_to_deliver — both code branches (zero prior coverage).
  3. nearest_idx         — new Stage-A grid-snapping utility.
  4. actual_policy_plotter(probabilities=None) — uncovered branch.
  5. analytic_plotting single-subplot (number_of_plots==1) — uncovered branch.
  6. adaptive_fractionation_core infeasibility — forced min/max dose paths.
  7. _hypothetical_belief_grid_indices Welford update — directional mu and
     diminishing-returns sigma behaviour.
  8. _bellman_expectation vs _bellman_expectation_full_grid — single-belief
     consistency against the vectorised full-grid implementation.
  9. adaptfx_full total-penalty consistency — negated per-fraction sum check
     (n=2 fractions, terminal-state DP only, < 1 s).
"""

import numpy as np
import pytest

from adaptive_fractionation_overlap.constants import DEFAULT_ALPHA, DEFAULT_BETA
from adaptive_fractionation_overlap.helper_functions import (
    std_calc,
    min_dose_to_deliver,
    nearest_idx,
    actual_policy_plotter,
    analytic_plotting,
    penalty_calc_single,
)
from adaptive_fractionation_overlap.core_adaptfx import (
    adaptive_fractionation_core,
    adaptfx_full,
)
from adaptive_fractionation_overlap.belief_model import (
    _MU_GRID,
    _SIGMA_GRID,
    _VOLUME_SPACE,
    _P_BELIEF,
    _hypothetical_belief_grid_indices,
    _bellman_expectation,
    _bellman_expectation_full_grid,
)


# ---------------------------------------------------------------------------
# 1. std_calc — Bayesian prior-vs-likelihood balance
# ---------------------------------------------------------------------------

class TestStdCalcBeliefStateUpdate:

    def test_n1_returns_value_near_prior_mode(self):
        """With a single observation the sample variance is zero, so the gamma
        prior dominates and the MAP equals beta*(alpha-1)."""
        result = std_calc(np.array([3.0]), DEFAULT_ALPHA, DEFAULT_BETA)
        prior_mode = DEFAULT_BETA * (DEFAULT_ALPHA - 1)
        assert abs(result - prior_mode) < 0.005, (
            f"n=1 result {result:.4f} should be near prior mode {prior_mode:.4f}"
        )

    def test_higher_sample_variance_gives_larger_std_estimate(self):
        low_var  = np.array([3.0, 3.1, 2.9, 3.0, 3.1])   # var ≈ 0.004 cc²
        high_var = np.array([0.5, 5.5, 1.0, 6.0, 0.8])   # var ≈ 5.7  cc²
        assert std_calc(high_var, DEFAULT_ALPHA, DEFAULT_BETA) > std_calc(low_var, DEFAULT_ALPHA, DEFAULT_BETA)


# ---------------------------------------------------------------------------
# 2. min_dose_to_deliver — both formula branches
# ---------------------------------------------------------------------------

class TestMinDoseToDeliver:

    def test_clamps_to_min_dose_when_future_fractions_can_absorb_remainder(self):
        # (40-0) - 4*10 = 0 < 6 → clamp to min_dose
        assert min_dose_to_deliver(
            accumulated_dose=0.0, fractions_left=5,
            prescribed_dose=40.0, min_dose=6.0, max_dose=10.0,
        ) == pytest.approx(6.0)

    def test_returns_calculated_value_when_above_min_dose(self):
        # (40-20) - 1*10 = 10 ≥ 6 → return 10
        assert min_dose_to_deliver(
            accumulated_dose=20.0, fractions_left=2,
            prescribed_dose=40.0, min_dose=6.0, max_dose=10.0,
        ) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 3. nearest_idx — new Stage-A grid-snapping utility
# ---------------------------------------------------------------------------

class TestNearestIdx:

    def test_exact_grid_hit(self):
        grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nearest_idx(3.0, grid) == 2

    def test_rounds_to_lower_when_equidistant(self):
        # The implementation uses <=, so the lower neighbour wins at ties.
        grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nearest_idx(2.5, grid) == 1

    def test_rounds_to_nearest(self):
        grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nearest_idx(2.3, grid) == 1   # closer to 2
        assert nearest_idx(2.7, grid) == 2   # closer to 3

    def test_out_of_range_clamps_to_boundary(self):
        grid = np.array([1.0, 2.0, 3.0])
        assert nearest_idx(0.0, grid) == 0    # below → first
        assert nearest_idx(10.0, grid) == 2   # above → last


# ---------------------------------------------------------------------------
# 4. actual_policy_plotter without the optional probabilities argument
# ---------------------------------------------------------------------------

def test_actual_policy_plotter_without_probabilities_does_not_raise():
    """The probabilities=None branch (no twin axis) was never exercised."""
    fig = actual_policy_plotter(
        np.array([9.0, 8.5, 8.0, 7.5, 7.0]),
        np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
    )
    assert len(fig.axes) == 1, "No twin axis should be added when probabilities=None"


# ---------------------------------------------------------------------------
# 5. analytic_plotting single-subplot branch (number_of_plots == 1)
# ---------------------------------------------------------------------------

def test_analytic_plotting_single_subplot_does_not_raise():
    """fraction=4 with number_of_fractions=5 → number_of_plots=1 (else-branch)."""
    analytic_plotting(
        fraction=4, number_of_fractions=5,
        values=np.random.rand(1, 8, 12),
        volume_space=np.linspace(0, 4, 8),
        dose_space=np.linspace(24, 40, 12),
    )


# ---------------------------------------------------------------------------
# 6. adaptive_fractionation_core infeasibility paths
# ---------------------------------------------------------------------------

class TestAdaptiveFractionationCoreInfeasibility:

    _COMMON = dict(
        number_of_fractions=5, min_dose=6.0, max_dose=10.0,
        dose_steps=0.5, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
    )

    def test_under_prescribed_returns_min_dose(self):
        # prescribed=20 < 5*min_dose=30 → force min_dose
        result = adaptive_fractionation_core(
            fraction_index_today=1, volumes=np.array([2.0]),
            accumulated_dose=0.0, mean_dose=4.0, **self._COMMON,
        )
        assert result[3] == pytest.approx(6.0)

    def test_over_prescribed_returns_max_dose(self):
        # prescribed=60 > 5*max_dose=50 → force max_dose
        result = adaptive_fractionation_core(
            fraction_index_today=1, volumes=np.array([2.0]),
            accumulated_dose=0.0, mean_dose=12.0, **self._COMMON,
        )
        assert result[3] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 7. _hypothetical_belief_grid_indices — Welford update properties
# ---------------------------------------------------------------------------

class TestWelfordBeliefUpdateConvergence:

    def test_observing_above_mean_increases_updated_mu(self):
        mu, sigma = 3.0, 1.0
        next_mi, _ = _hypothetical_belief_grid_indices(mu, sigma, np.array([mu + 2.0]), 3)
        assert _MU_GRID[next_mi[0]] > mu

    def test_fewer_prior_observations_give_greater_sigma_reduction(self):
        """The Welford update weight is 1/(n+1), so a consistent new observation
        has a larger impact when n is small.
        For obs=mu: sigma' = sqrt(n/(n+1)) * sigma → smaller for n=1 than n=10.
        """
        mu, sigma = 3.0, 1.0
        obs = np.array([mu])
        _, si_n1  = _hypothetical_belief_grid_indices(mu, sigma, obs, 1)
        _, si_n10 = _hypothetical_belief_grid_indices(mu, sigma, obs, 10)
        assert _SIGMA_GRID[si_n1[0]] <= _SIGMA_GRID[si_n10[0]]


# ---------------------------------------------------------------------------
# 8. _bellman_expectation vs _bellman_expectation_full_grid consistency
# ---------------------------------------------------------------------------

def test_bellman_single_belief_matches_full_grid_slice():
    """_bellman_expectation for one (mu, sigma) point must match the
    corresponding slice of _bellman_expectation_full_grid."""
    rng = np.random.default_rng(42)
    values_prev = rng.random((3, len(_VOLUME_SPACE), len(_MU_GRID), len(_SIGMA_GRID)))

    full = _bellman_expectation_full_grid(values_prev, observation_count=2)

    mi, si = 50, 3
    single = _bellman_expectation(
        values_prev, _VOLUME_SPACE, _P_BELIEF[mi, si],
        _MU_GRID[mi], _SIGMA_GRID[si], observation_count=2,
    )

    np.testing.assert_allclose(full[:, mi, si], single, rtol=1e-5)


# ---------------------------------------------------------------------------
# 9. adaptfx_full total-penalty consistency
# ---------------------------------------------------------------------------

def test_adaptfx_full_total_penalty_equals_negated_per_fraction_sum():
    """total_penalty must equal -sum(penalty_calc_single) over treatment fractions.
    Uses n=2 so the DP runs only the terminal state — no Numba kernel, < 1 s.
    """
    volumes, n_frac, min_dose = [1.0, 0.5, 2.0], 2, 6.0

    physical_doses, _, total_penalty = adaptfx_full(
        volumes=volumes, number_of_fractions=n_frac, min_dose=min_dose,
        max_dose=10.0, mean_dose=8.0, dose_steps=0.5,
        alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
    )

    expected = sum(
        -float(penalty_calc_single(dose, min_dose, volumes[-n_frac + i]))
        for i, dose in enumerate(physical_doses)
    )
    assert total_penalty == pytest.approx(expected, abs=1e-12)
