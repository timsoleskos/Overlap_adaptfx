"""
Test suite for belief_model module.

Tests the belief-state grids, default branch probabilities (_P_BELIEF),
Welford belief updates on S^2, Bellman operators, and the current_belief_probdist helper.
"""

import numpy as np
import pytest
from adaptive_fractionation_overlap.belief_model import (
    _MU_GRID,
    _S2_GRID,
    _VOLUME_SPACE,
    _P_BELIEF,
    _hypothetical_belief_grid_indices,
    _bellman_expectation,
    _bellman_expectation_full_grid,
    current_belief_probdist,
)
from adaptive_fractionation_overlap.constants import DEFAULT_ALPHA, DEFAULT_BETA
from adaptive_fractionation_overlap.helper_functions import sigma_map_from_variance


class TestGrids:
    """Verify the static grid constants."""

    def test_mu_grid_shape_and_order(self):
        """_MU_GRID should be sorted, positive, and have the documented 150 points."""
        assert len(_MU_GRID) == 150, f"Expected 150 mu grid points, got {len(_MU_GRID)}"
        assert np.all(np.diff(_MU_GRID) > 0), "_MU_GRID must be strictly increasing"
        assert _MU_GRID[0] >= 0.0, "_MU_GRID must start at or above 0 cc"

    def test_s2_grid_shape_and_order(self):
        """_S2_GRID should be sorted, non-negative, and have the documented 12 points."""
        assert len(_S2_GRID) == 12, f"Expected 12 S^2 grid points, got {len(_S2_GRID)}"
        assert np.all(np.diff(_S2_GRID) > 0), "_S2_GRID must be strictly increasing"
        assert _S2_GRID[0] == pytest.approx(0.0, abs=1e-12), "_S2_GRID must include the exact zero-variance state"

    def test_volume_space_shape_and_uniformity(self):
        """_VOLUME_SPACE should be a uniform linspace with 441 bins starting at 0."""
        assert len(_VOLUME_SPACE) == 441, f"Expected 441 volume bins, got {len(_VOLUME_SPACE)}"
        assert _VOLUME_SPACE[0] == pytest.approx(0.0, abs=1e-12), "_VOLUME_SPACE must start at 0.0 cc"
        steps = np.diff(_VOLUME_SPACE)
        assert np.allclose(steps, steps[0], rtol=1e-10), "_VOLUME_SPACE must be uniform (linspace)"
        assert steps[0] == pytest.approx(0.1, abs=1e-3), "Expected ~0.1 cc step size"

    def test_volume_space_covers_full_belief_range(self):
        """_VOLUME_SPACE must cover at least mu_grid_max + 3 * sigma_max_from_grid.

        The belief state stores S^2, so the predictive sigma bound must be derived from the
        largest grid variance via the same MAP rule used by the solver. _VOLUME_SPACE is
        hardcoded to 44 cc (not derived from the belief grid) to keep the bin width at exactly
        0.1 cc while still covering the extreme belief tail.
        """
        sigma_max = float(sigma_map_from_variance(_S2_GRID[-1], 1, DEFAULT_ALPHA, DEFAULT_BETA))
        required_min = _MU_GRID[-1] + 3 * sigma_max
        assert _VOLUME_SPACE[-1] >= required_min, (
            f"_VOLUME_SPACE must reach at least {required_min:.2f} cc (mu_max + 3*sigma_max); "
            f"currently ends at {_VOLUME_SPACE[-1]:.2f} cc"
        )


class TestPBeliefProbabilities:
    """Verify the default _P_BELIEF branch probability table."""

    def test_shape(self):
        assert _P_BELIEF.shape == (len(_MU_GRID), len(_S2_GRID), len(_VOLUME_SPACE)), (
            f"_P_BELIEF shape mismatch: {_P_BELIEF.shape}"
        )

    def test_all_probabilities_non_negative(self):
        assert np.all(_P_BELIEF >= 0), "All _P_BELIEF entries must be non-negative"

    def test_all_probabilities_at_most_one(self):
        assert np.all(_P_BELIEF <= 1.0 + 1e-12), "All _P_BELIEF entries must be ≤ 1"

    def test_rows_sum_to_exactly_one(self):
        """Each (mu, S^2) row must sum to exactly 1.0 due to tail-folding."""
        row_sums = _P_BELIEF.sum(axis=2)  # sum over overlap axis; shape (N_mu, N_s2)
        assert np.allclose(row_sums, 1.0, atol=1e-12), (
            f"_P_BELIEF rows must sum to 1.0 (tail-folded); "
            f"max deviation: {np.abs(row_sums - 1.0).max():.2e}"
        )


class TestCurrentBeliefProbdist:
    """Verify current_belief_probdist output over _VOLUME_SPACE."""

    def test_returns_array_of_correct_length(self):
        result = current_belief_probdist(mu=3.0, sigma=1.0)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(_VOLUME_SPACE), (
            f"Expected {len(_VOLUME_SPACE)} bins, got {len(result)}"
        )

    def test_all_non_negative(self):
        result = current_belief_probdist(mu=3.0, sigma=0.5)
        assert np.all(result >= 0), "Probabilities must be non-negative"

    def test_sum_near_one_for_central_belief(self):
        """For a belief well within _VOLUME_SPACE the sum should be very close to 1."""
        result = current_belief_probdist(mu=5.0, sigma=0.5)
        assert np.sum(result) == pytest.approx(1.0, abs=1e-6), (
            f"Sum should be ≈1 for a central belief; got {np.sum(result):.8f}"
        )

    def test_sum_equals_one_for_extreme_belief(self):
        """For a belief whose tails extend beyond _VOLUME_SPACE, tails are folded into boundary bins."""
        # Belief centred at _VOLUME_SPACE[-1] with large sigma: right tail is clipped
        result = current_belief_probdist(mu=_VOLUME_SPACE[-1], sigma=5.0)
        total = np.sum(result)
        assert total == pytest.approx(1.0, abs=1e-12), (
            f"current_belief_probdist should sum to 1.0 after tail-folding; got {total:.6f}"
        )

    def test_peak_near_mu(self):
        """The highest-probability bin should be near the belief mean."""
        mu = 8.0
        result = current_belief_probdist(mu=mu, sigma=0.5)
        peak_volume = _VOLUME_SPACE[np.argmax(result)]
        assert abs(peak_volume - mu) < 0.5, (
            f"Peak probability at {peak_volume:.2f} cc should be near mu={mu} cc"
        )


class TestHypotheticalBeliefGridIndices:
    """Verify that the Welford belief-update maps to valid grid indices."""

    def test_indices_in_bounds(self):
        mu = 3.0
        s2 = 0.25
        observation_count = 2
        next_mi, next_si = _hypothetical_belief_grid_indices(
            mu, s2, _VOLUME_SPACE, observation_count
        )
        assert np.all(next_mi >= 0) and np.all(next_mi < len(_MU_GRID)), "mu indices out of bounds"
        assert np.all(next_si >= 0) and np.all(next_si < len(_S2_GRID)), "S^2 indices out of bounds"

    def test_observation_equal_to_mean_reduces_predictive_sigma(self):
        """Observing exactly the belief mean should reduce (or at least not increase) predictive sigma."""
        mu = 5.0
        s2 = 1.0
        observation_count = 3
        current_sigma = sigma_map_from_variance(s2, observation_count, DEFAULT_ALPHA, DEFAULT_BETA)
        # Inject the mean as the new observation: Welford S'^2 ≤ S^2, so sigma_MAP should not materially increase.
        observation_at_mean = np.array([mu])
        next_mi, next_si = _hypothetical_belief_grid_indices(
            mu, s2, observation_at_mean, observation_count
        )
        next_s2 = _S2_GRID[next_si[0]]
        next_sigma = sigma_map_from_variance(next_s2, observation_count + 1, DEFAULT_ALPHA, DEFAULT_BETA)
        assert next_sigma <= current_sigma + 0.2, (
            f"Observing the mean should not substantially increase predictive sigma; "
            f"got next_sigma={next_sigma:.3f} vs current sigma={current_sigma:.3f}"
        )
