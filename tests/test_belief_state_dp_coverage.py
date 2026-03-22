"""
Test coverage improvements for the belief-state DP branch.

Targets gaps not covered by the existing test suite:

  1. std_calc           — Bayesian prior-vs-likelihood balance (n=1 prior mode,
                          ordering by empirical variance, large-n convergence).
  2. min_dose_to_deliver — zero prior unit coverage; all formula branches.
  3. nearest_idx        — new Stage-A utility; zero prior coverage.
  4. linear_interp      — new Stage-A utility; zero prior coverage.
  5. actual_policy_plotter (probabilities=None) — previously uncovered branch.
  6. analytic_plotting single-subplot — number_of_plots==1 branch never executed.
  7. _build_dp_context infeasibility — both sentinel-fill branches, no DP run.
  8. adaptive_fractionation_core infeasibility — public-API surface for (7).
  9. _hypothetical_belief_grid_indices Welford convergence — mu and sigma update.
 10. _bellman_expectation vs _bellman_expectation_full_grid — single-belief
     consistency against the vectorised full-grid version.
 11. adaptfx_full total-penalty consistency — negated per-fraction sum must
     equal the scalar returned by adaptfx_full (n=2, terminal-state DP only).

All tests run in well under 2 minutes in CI.  The only potentially moderate-cost
test is test_single_belief_matches_full_grid (full N_mu × N_sigma grid, N_dose=3).
"""

import numpy as np
import pytest

from adaptive_fractionation_overlap.constants import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_MIN_DOSE,
    DEFAULT_MAX_DOSE,
)
from adaptive_fractionation_overlap.helper_functions import (
    std_calc,
    min_dose_to_deliver,
    nearest_idx,
    linear_interp,
    actual_policy_plotter,
    analytic_plotting,
    penalty_calc_single,
)
from adaptive_fractionation_overlap.core_adaptfx import (
    _build_dp_context,
    _INFEASIBILITY_SENTINEL,
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
    """Verify the MAP std estimator respects prior dominance and data ordering."""

    def test_n1_returns_value_near_prior_mode(self):
        """With a single observation the likelihood data term vanishes (zero sample
        variance), so the result should equal the gamma-prior MAP: beta*(alpha-1).

        DEFAULT_ALPHA = 1.073, DEFAULT_BETA = 0.779 → prior mode ≈ 0.057 cc.
        """
        single_obs = np.array([3.0])
        result = std_calc(single_obs, DEFAULT_ALPHA, DEFAULT_BETA)

        prior_mode = DEFAULT_BETA * (DEFAULT_ALPHA - 1)
        # Grid resolution is 0.001 cc; allow ±3 grid steps.
        assert abs(result - prior_mode) < 0.005, (
            f"n=1 result {result:.4f} should be near prior mode {prior_mode:.4f}"
        )

    def test_higher_sample_variance_gives_larger_std_estimate(self):
        """The MAP std should be monotonically larger for higher-variance data."""
        low_var  = np.array([3.0, 3.1, 2.9, 3.0, 3.1])   # var ≈ 0.004 cc²
        high_var = np.array([0.5, 5.5, 1.0, 6.0, 0.8])   # var ≈ 5.7  cc²

        result_low  = std_calc(low_var,  DEFAULT_ALPHA, DEFAULT_BETA)
        result_high = std_calc(high_var, DEFAULT_ALPHA, DEFAULT_BETA)

        assert result_high > result_low, (
            f"High-variance data (result={result_high:.3f}) should yield larger std "
            f"than low-variance data (result={result_low:.3f})"
        )

    def test_large_n_estimate_is_closer_to_sample_std_than_small_n(self):
        """With more observations the MAP estimate converges toward the sample std."""
        # Fixed data so the test is deterministic.
        data_n2  = np.array([1.0, 5.0])                         # sample_std ≈ 2.0
        data_n10 = np.array([1.0, 2.0, 3.0, 4.0, 5.0,
                              1.2, 2.2, 3.2, 4.2, 5.2])         # sample_std ≈ 1.47

        result_n2  = std_calc(data_n2,  DEFAULT_ALPHA, DEFAULT_BETA)
        result_n10 = std_calc(data_n10, DEFAULT_ALPHA, DEFAULT_BETA)

        sample_std_n2  = float(np.std(data_n2))
        sample_std_n10 = float(np.std(data_n10))

        rel_err_n2  = abs(result_n2  - sample_std_n2)  / max(sample_std_n2,  0.01)
        rel_err_n10 = abs(result_n10 - sample_std_n10) / max(sample_std_n10, 0.01)

        assert rel_err_n10 < rel_err_n2, (
            f"n=10 relative error ({rel_err_n10:.3f}) should be smaller than "
            f"n=2 relative error ({rel_err_n2:.3f})"
        )

    def test_extreme_variance_returns_finite_positive_result(self):
        """Very high-variance data must not produce nan or negative std."""
        extreme_data = np.array([0.0, 20.0, 0.0, 20.0])
        result = std_calc(extreme_data, DEFAULT_ALPHA, DEFAULT_BETA)
        assert np.isfinite(result) and result > 0, (
            f"Extreme-variance data should give finite positive std; got {result}"
        )


# ---------------------------------------------------------------------------
# 2. min_dose_to_deliver — formula branches
# ---------------------------------------------------------------------------

class TestMinDoseToDeliver:
    """Unit-test every branch of min_dose_to_deliver."""

    def test_clamps_to_min_dose_when_future_fractions_can_absorb_remainder(self):
        # (40-0) - 4*10 = 0 < 6 → clamp to min_dose
        result = min_dose_to_deliver(
            accumulated_dose=0.0, fractions_left=5,
            prescribed_dose=40.0, min_dose=6.0, max_dose=10.0,
        )
        assert result == pytest.approx(6.0), f"Expected 6.0, got {result}"

    def test_returns_calculated_value_when_above_min_dose(self):
        # (40-20) - 1*10 = 10 > 6 → return 10
        result = min_dose_to_deliver(
            accumulated_dose=20.0, fractions_left=2,
            prescribed_dose=40.0, min_dose=6.0, max_dose=10.0,
        )
        assert result == pytest.approx(10.0), f"Expected 10.0, got {result}"

    def test_last_fraction_no_future_budget(self):
        # (40-30) - 0*10 = 10 → return 10
        result = min_dose_to_deliver(
            accumulated_dose=30.0, fractions_left=1,
            prescribed_dose=40.0, min_dose=6.0, max_dose=10.0,
        )
        assert result == pytest.approx(10.0), f"Expected 10.0, got {result}"

    def test_last_fraction_under_min_dose_clamps(self):
        # (40-35) - 0*10 = 5 < 6 → clamp
        result = min_dose_to_deliver(
            accumulated_dose=35.0, fractions_left=1,
            prescribed_dose=40.0, min_dose=6.0, max_dose=10.0,
        )
        assert result == pytest.approx(6.0), f"Expected 6.0, got {result}"

    def test_large_deficit_with_few_fractions(self):
        # (45-10) - 1*12 = 35-12 = 23 > 6 → return 23
        result = min_dose_to_deliver(
            accumulated_dose=10.0, fractions_left=2,
            prescribed_dose=45.0, min_dose=6.0, max_dose=12.0,
        )
        assert result == pytest.approx(23.0), f"Expected 23.0, got {result}"


# ---------------------------------------------------------------------------
# 3. nearest_idx — new Stage-A grid-snapping utility
# ---------------------------------------------------------------------------

class TestNearestIdx:
    """nearest_idx should snap every query to its closest grid point."""

    def test_exact_grid_hit(self):
        grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nearest_idx(3.0, grid) == 2

    def test_rounds_to_lower_when_equidistant(self):
        # 2.5 is equidistant between grid[1]=2 and grid[2]=3; implementation
        # uses <=, so the lower neighbour wins.
        grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nearest_idx(2.5, grid) == 1

    def test_rounds_to_nearest(self):
        grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nearest_idx(2.3, grid) == 1   # closer to 2
        assert nearest_idx(2.7, grid) == 2   # closer to 3

    def test_below_grid_clamps_to_first(self):
        grid = np.array([1.0, 2.0, 3.0])
        assert nearest_idx(0.0, grid) == 0

    def test_above_grid_clamps_to_last(self):
        grid = np.array([1.0, 2.0, 3.0])
        assert nearest_idx(10.0, grid) == 2

    def test_batch_array_input(self):
        grid = np.array([0.0, 1.0, 2.0, 3.0])
        queries = np.array([0.1, 1.6, 2.9])
        result = nearest_idx(queries, grid)
        np.testing.assert_array_equal(result, [0, 2, 3])

    def test_output_shape_matches_input(self):
        grid = np.linspace(0, 10, 11)
        queries = np.array([[1.1, 2.2], [3.3, 4.4]])
        result = nearest_idx(queries, grid)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# 4. linear_interp — new Stage-A 1-D interpolation utility
# ---------------------------------------------------------------------------

class TestLinearInterp:
    """linear_interp wraps np.interp; verify shape preservation and values."""

    def test_exact_grid_point_returns_exact_value(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 10.0, 20.0, 30.0])
        assert linear_interp(x, y, 2.0) == pytest.approx(20.0)

    def test_midpoint_interpolation(self):
        x = np.array([0.0, 2.0])
        y = np.array([0.0, 4.0])
        assert linear_interp(x, y, 1.0) == pytest.approx(2.0)

    def test_1d_array_query(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])
        result = linear_interp(x, y, np.array([0.5, 1.5]))
        assert result.shape == (2,)
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(2.5)

    def test_2d_query_preserves_shape(self):
        x = np.linspace(0, 10, 11)
        y = x ** 2
        queries = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = linear_interp(x, y, queries)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, queries ** 2, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. actual_policy_plotter without the optional probabilities argument
# ---------------------------------------------------------------------------

class TestActualPolicyPlotterNoProbabilities:
    """The probabilities=None branch of actual_policy_plotter was not tested."""

    def test_no_probabilities_does_not_raise(self):
        policies_overlap = np.array([9.0, 8.5, 8.0, 7.5, 7.0])
        volume_space     = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        try:
            fig = actual_policy_plotter(policies_overlap, volume_space)
        except Exception as exc:
            pytest.fail(f"actual_policy_plotter(probabilities=None) raised: {exc}")

        # Should return a figure with exactly one axis (no twin axis added)
        assert len(fig.axes) == 1, "Without probabilities there should be exactly one axis"


# ---------------------------------------------------------------------------
# 6. analytic_plotting single-subplot branch (number_of_plots == 1)
# ---------------------------------------------------------------------------

class TestAnalyticPlottingSingleSubplot:
    """fraction=4, number_of_fractions=5 gives number_of_plots=1; the else-branch
    (single subplot) was never executed by existing tests."""

    def test_single_subplot_does_not_raise(self):
        fraction = 4
        number_of_fractions = 5          # number_of_plots = 5-4 = 1
        values     = np.random.rand(1, 8, 12)   # (1 future fraction, N_vol, N_dose)
        volume_space = np.linspace(0, 4, 8)
        dose_space   = np.linspace(24, 40, 12)

        try:
            fig = analytic_plotting(fraction, number_of_fractions, values, volume_space, dose_space)
        except Exception as exc:
            pytest.fail(f"analytic_plotting single-subplot raised: {exc}")

        # A single imshow creates one main axis plus one colorbar axis → 2 axes total.
        assert len(fig.axes) >= 1, "Figure should have at least one axis"


# ---------------------------------------------------------------------------
# 7. _build_dp_context infeasibility branches
# ---------------------------------------------------------------------------

class TestBuildDpContextInfeasibility:
    """_build_dp_context must return immediately (no DP) when the goal is
    unreachable, filling values with _INFEASIBILITY_SENTINEL."""

    def test_under_prescribed_is_infeasible_and_fixed_to_min_dose(self):
        # prescribed=20 (mean=4), remaining_fracs=5, 20 < 5*6=30 → infeasible
        ctx = _build_dp_context(1, 5, 0.0, 6.0, 10.0, 4.0, 0.5)
        assert ctx['is_infeasible'] is True
        assert ctx['fixed_dose'] == pytest.approx(6.0), (
            f"Under-prescribed should fix dose to min_dose=6.0, got {ctx['fixed_dose']}"
        )

    def test_over_prescribed_is_infeasible_and_fixed_to_max_dose(self):
        # prescribed=60 (mean=12), remaining_fracs=5, 60 > 5*10=50 → infeasible
        ctx = _build_dp_context(1, 5, 0.0, 6.0, 10.0, 12.0, 0.5)
        assert ctx['is_infeasible'] is True
        assert ctx['fixed_dose'] == pytest.approx(10.0), (
            f"Over-prescribed should fix dose to max_dose=10.0, got {ctx['fixed_dose']}"
        )

    def test_infeasible_values_array_filled_with_sentinel(self):
        # remaining_fracs = 5-1+1 = 5 > 1, so values[:] = _INFEASIBILITY_SENTINEL
        ctx = _build_dp_context(1, 5, 0.0, 6.0, 10.0, 4.0, 0.5)
        assert np.all(ctx['values'] == _INFEASIBILITY_SENTINEL), (
            "All future-state values must be _INFEASIBILITY_SENTINEL in an infeasible plan"
        )

    def test_feasible_terminal_fraction_has_empty_values_array(self):
        # Last fraction: remaining_fracs=1, no backward sweep; values shape[0]=0
        ctx = _build_dp_context(5, 5, 32.0, 6.0, 10.0, 8.0, 0.5)
        assert not ctx['is_infeasible']
        assert ctx['values'].shape[0] == 0, (
            f"Terminal fraction should have no future-state values; "
            f"got shape {ctx['values'].shape}"
        )

    def test_feasible_non_terminal_is_not_infeasible(self):
        # Normal mid-plan call; should not be flagged infeasible
        ctx = _build_dp_context(3, 5, 16.0, 6.0, 10.0, 8.0, 0.5)
        assert not ctx['is_infeasible']


# ---------------------------------------------------------------------------
# 8. adaptive_fractionation_core infeasibility via the public API
# ---------------------------------------------------------------------------

class TestAdaptiveFractionationCoreInfeasibility:
    """Public-API surface for the infeasibility branches tested in (7)."""

    _COMMON = dict(
        number_of_fractions=5, min_dose=6.0, max_dose=10.0,
        dose_steps=0.5, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
    )

    def test_under_prescribed_returns_min_dose(self):
        # prescribed=20, remaining_ptv < remaining_fracs*min_dose → fixed=min
        result = adaptive_fractionation_core(
            fraction_index_today=1, volumes=np.array([2.0]),
            accumulated_dose=0.0, mean_dose=4.0, **self._COMMON,
        )
        assert result[3] == pytest.approx(6.0), (
            f"Under-prescribed plan should return min_dose=6.0, got {result[3]}"
        )

    def test_over_prescribed_returns_max_dose(self):
        # prescribed=60, remaining_ptv > remaining_fracs*max_dose → fixed=max
        result = adaptive_fractionation_core(
            fraction_index_today=1, volumes=np.array([2.0]),
            accumulated_dose=0.0, mean_dose=12.0, **self._COMMON,
        )
        assert result[3] == pytest.approx(10.0), (
            f"Over-prescribed plan should return max_dose=10.0, got {result[3]}"
        )

    def test_normal_plan_dose_within_bounds(self):
        # Sanity check that a feasible first fraction returns a dose in [6, 10]
        result = adaptive_fractionation_core(
            fraction_index_today=1, volumes=np.array([1.5]),
            accumulated_dose=0.0, mean_dose=8.0, **self._COMMON,
        )
        assert DEFAULT_MIN_DOSE <= result[3] <= DEFAULT_MAX_DOSE, (
            f"Feasible plan dose {result[3]} should be in [{DEFAULT_MIN_DOSE}, {DEFAULT_MAX_DOSE}]"
        )


# ---------------------------------------------------------------------------
# 9. _hypothetical_belief_grid_indices — Welford update convergence
# ---------------------------------------------------------------------------

class TestWelfordBeliefUpdateConvergence:
    """The existing tests only check that indices are in-bounds and that one
    consistent observation does not increase sigma substantially.  We add:
      - mu updates toward the observed value
      - repeated consistent observations reduce sigma
    """

    def test_observing_above_mean_increases_updated_mu(self):
        """A new observation above mu should pull the updated belief mean upward."""
        mu, sigma = 3.0, 1.0
        high_obs = np.array([mu + 2.0])   # above current mean

        next_mi, _ = _hypothetical_belief_grid_indices(mu, sigma, high_obs, 3)

        assert _MU_GRID[next_mi[0]] > mu, (
            f"Updated belief mu {_MU_GRID[next_mi[0]]:.3f} should exceed "
            f"prior mu {mu} when observation > mu"
        )

    def test_observing_below_mean_decreases_updated_mu(self):
        """A new observation below mu should pull the updated belief mean downward."""
        mu, sigma = 5.0, 1.0
        low_obs = np.array([mu - 2.0])

        next_mi, _ = _hypothetical_belief_grid_indices(mu, sigma, low_obs, 3)

        assert _MU_GRID[next_mi[0]] < mu, (
            f"Updated belief mu {_MU_GRID[next_mi[0]]:.3f} should be below "
            f"prior mu {mu} when observation < mu"
        )

    def test_fewer_prior_observations_give_greater_sigma_reduction(self):
        """The Welford update weight is 1/(observation_count+1), so a consistent
        new observation has a larger impact when observation_count is small.

        With observation_count=n and obs=mu:
            sigma' = sqrt(n/(n+1)) * sigma
          → n=1  → sigma' ≈ 0.707 (large reduction)
          → n=10 → sigma' ≈ 0.954 (small reduction)

        Therefore the grid-snapped sigma must be ≤ for smaller n.
        """
        mu, sigma = 3.0, 1.0
        obs_at_mean = np.array([mu])

        _, si_n1  = _hypothetical_belief_grid_indices(mu, sigma, obs_at_mean, 1)
        _, si_n10 = _hypothetical_belief_grid_indices(mu, sigma, obs_at_mean, 10)

        assert _SIGMA_GRID[si_n1[0]] <= _SIGMA_GRID[si_n10[0]], (
            f"Fewer prior observations should cause greater sigma reduction: "
            f"n=1 snaps to {_SIGMA_GRID[si_n1[0]]:.3f} cc, "
            f"n=10 snaps to {_SIGMA_GRID[si_n10[0]]:.3f} cc"
        )


# ---------------------------------------------------------------------------
# 10. _bellman_expectation vs _bellman_expectation_full_grid consistency
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBellmanExpectationConsistency:
    """For a specific (mu, sigma) grid point, the single-belief and full-grid
    Bellman operators must agree to within floating-point rounding."""

    def test_single_belief_matches_full_grid_slice(self):
        rng = np.random.default_rng(42)
        N_dose   = 3   # keep memory low; shape (3, 441, 150, 11) ≈ 17 MB
        N_mu     = len(_MU_GRID)
        N_sigma  = len(_SIGMA_GRID)
        N_overlap = len(_VOLUME_SPACE)

        values_prev = rng.random((N_dose, N_overlap, N_mu, N_sigma))
        observation_count = 2

        # Full-grid result: (N_dose, N_mu, N_sigma)
        full_result = _bellman_expectation_full_grid(values_prev, observation_count)

        # Single-belief check at grid point (mi=50, si=3)
        mi, si = 50, 3
        mu    = _MU_GRID[mi]
        sigma = _SIGMA_GRID[si]
        p_branch = _P_BELIEF[mi, si]   # (N_overlap,)

        single_result = _bellman_expectation(
            values_prev, _VOLUME_SPACE, p_branch, mu, sigma, observation_count
        )  # (N_dose,)

        np.testing.assert_allclose(
            full_result[:, mi, si], single_result, rtol=1e-5,
            err_msg=(
                "_bellman_expectation_full_grid and _bellman_expectation must agree "
                f"at (mi={mi}, si={si})"
            ),
        )


# ---------------------------------------------------------------------------
# 11. adaptfx_full total-penalty consistency
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAdaptfxFullPenaltyConsistency:
    """The total_penalty scalar returned by adaptfx_full must equal the
    negated sum of per-fraction OAR costs computed from physical_doses and
    the treatment-fraction volumes.

    Uses n=2 fractions so _build_dp_context only runs the terminal state
    (no Numba kernel), keeping CI runtime well under a second.
    """

    def test_total_penalty_equals_negated_per_fraction_sum(self):
        volumes   = [1.0, 0.5, 2.0]   # planning scan + 2 treatment fractions
        min_dose  = 6.0
        n_frac    = 2

        physical_doses, _, total_penalty = adaptfx_full(
            volumes=volumes,
            number_of_fractions=n_frac,
            min_dose=min_dose,
            max_dose=10.0,
            mean_dose=8.0,
            dose_steps=0.5,
            alpha=DEFAULT_ALPHA,
            beta=DEFAULT_BETA,
        )

        # Replicate the formula used inside adaptfx_full
        expected = 0.0
        for i, dose in enumerate(physical_doses):
            expected -= float(
                penalty_calc_single(dose, min_dose, volumes[-n_frac + i])
            )

        assert total_penalty == pytest.approx(expected, abs=1e-12), (
            f"total_penalty={total_penalty} should equal sum of per-fraction "
            f"penalties={expected}"
        )

    def test_total_penalty_is_non_positive(self):
        """The penalty is always ≤ 0 because penalty_calc_single ≥ 0 and we
        negate.  A plan with any dose above min_dose must yield a negative total."""
        volumes = [0.5, 0.5, 0.5]   # low overlap → DP likely recommends > min_dose
        physical_doses, _, total_penalty = adaptfx_full(
            volumes=volumes,
            number_of_fractions=2,
            min_dose=6.0,
            max_dose=10.0,
            mean_dose=8.0,
            dose_steps=0.5,
            alpha=DEFAULT_ALPHA,
            beta=DEFAULT_BETA,
        )
        assert total_penalty <= 0.0, (
            f"total_penalty must be ≤ 0; got {total_penalty}"
        )
        # If any dose exceeds min_dose, penalty must be strictly negative
        if any(d > 6.0 for d in physical_doses):
            assert total_penalty < 0.0
