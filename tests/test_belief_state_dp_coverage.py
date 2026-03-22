"""Coverage gaps for the belief-state DP branch (16 tests, <5 s total)."""
import numpy as np
import pytest

from adaptive_fractionation_overlap.constants import DEFAULT_ALPHA, DEFAULT_BETA
from adaptive_fractionation_overlap.helper_functions import (
    std_calc, min_dose_to_deliver, nearest_idx,
    actual_policy_plotter, analytic_plotting, penalty_calc_single,
)
from adaptive_fractionation_overlap.core_adaptfx import adaptive_fractionation_core, adaptfx_full
from adaptive_fractionation_overlap.belief_model import (
    _MU_GRID, _SIGMA_GRID, _VOLUME_SPACE, _P_BELIEF,
    _hypothetical_belief_grid_indices, _bellman_expectation, _bellman_expectation_full_grid,
)


def test_std_calc_n1_near_prior_mode():
    result = std_calc(np.array([3.0]), DEFAULT_ALPHA, DEFAULT_BETA)
    assert abs(result - DEFAULT_BETA * (DEFAULT_ALPHA - 1)) < 0.005

def test_std_calc_higher_variance_gives_larger_estimate():
    low  = np.array([3.0, 3.1, 2.9, 3.0, 3.1])
    high = np.array([0.5, 5.5, 1.0, 6.0, 0.8])
    assert std_calc(high, DEFAULT_ALPHA, DEFAULT_BETA) > std_calc(low, DEFAULT_ALPHA, DEFAULT_BETA)

def test_min_dose_clamps_to_min():
    assert min_dose_to_deliver(0.0, 5, 40.0, 6.0, 10.0) == pytest.approx(6.0)

def test_min_dose_returns_calculated_value():
    assert min_dose_to_deliver(20.0, 2, 40.0, 6.0, 10.0) == pytest.approx(10.0)


class TestNearestIdx:
    G = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_exact(self):
        assert nearest_idx(3.0, self.G) == 2

    def test_equidistant_lower_wins(self):
        assert nearest_idx(2.5, self.G) == 1

    def test_rounds_to_nearest(self):
        assert nearest_idx(2.3, self.G) == 1
        assert nearest_idx(2.7, self.G) == 2

    def test_out_of_range_clamps(self):
        g = np.array([1.0, 2.0, 3.0])
        assert nearest_idx(0.0, g) == 0
        assert nearest_idx(10.0, g) == 2


def test_policy_plotter_no_probabilities():
    fig = actual_policy_plotter(np.array([9., 8.5, 8., 7.5, 7.]), np.array([.5, 1.5, 2.5, 3.5, 4.5]))
    assert len(fig.axes) == 1

def test_analytic_plotting_single_subplot():
    analytic_plotting(
        fraction=4, number_of_fractions=5,
        values=np.random.rand(1, 8, 12),
        volume_space=np.linspace(0, 4, 8),
        dose_space=np.linspace(24, 40, 12),
    )


_AFC_COMMON = dict(number_of_fractions=5, min_dose=6.0, max_dose=10.0,
                   dose_steps=0.5, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA)

def test_infeasible_under_prescribed_returns_min_dose():
    result = adaptive_fractionation_core(
        fraction_index_today=1, volumes=np.array([2.0]),
        accumulated_dose=0.0, mean_dose=4.0, **_AFC_COMMON)
    assert result[3] == pytest.approx(6.0)

def test_infeasible_over_prescribed_returns_max_dose():
    result = adaptive_fractionation_core(
        fraction_index_today=1, volumes=np.array([2.0]),
        accumulated_dose=0.0, mean_dose=12.0, **_AFC_COMMON)
    assert result[3] == pytest.approx(10.0)


def test_welford_above_mean_increases_mu():
    mu, sigma = 3.0, 1.0
    mi, _ = _hypothetical_belief_grid_indices(mu, sigma, np.array([mu + 2.0]), 3)
    assert _MU_GRID[mi[0]] > mu

def test_welford_fewer_observations_give_greater_sigma_reduction():
    mu, sigma, obs = 3.0, 1.0, np.array([3.0])
    _, si1  = _hypothetical_belief_grid_indices(mu, sigma, obs, 1)
    _, si10 = _hypothetical_belief_grid_indices(mu, sigma, obs, 10)
    assert _SIGMA_GRID[si1[0]] <= _SIGMA_GRID[si10[0]]

def test_bellman_single_matches_full_grid_slice():
    values = np.random.default_rng(42).random((3, len(_VOLUME_SPACE), len(_MU_GRID), len(_SIGMA_GRID)))
    full = _bellman_expectation_full_grid(values, observation_count=2)
    mi, si = 50, 3
    single = _bellman_expectation(values, _VOLUME_SPACE, _P_BELIEF[mi, si],
                                  _MU_GRID[mi], _SIGMA_GRID[si], observation_count=2)
    np.testing.assert_allclose(full[:, mi, si], single, rtol=1e-5)

def test_adaptfx_full_total_penalty_consistency():
    vols, n, dmin = [1.0, 0.5, 2.0], 2, 6.0
    doses, _, total = adaptfx_full(
        volumes=vols, number_of_fractions=n, min_dose=dmin,
        max_dose=10.0, mean_dose=8.0, dose_steps=0.5,
        alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA)
    expected = sum(-float(penalty_calc_single(d, dmin, vols[-n + i])) for i, d in enumerate(doses))
    assert total == pytest.approx(expected, abs=1e-12)
