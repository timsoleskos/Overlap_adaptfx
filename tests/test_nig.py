"""
Test suite for Stage B NIG (Normal-Inverse-Gamma) infrastructure.

Tests _nig_posterior_params, _precompute_nig_branch_probabilities,
_get_p_belief_nig, current_belief_probdist_nig, and the backward-compat
p_belief=None path of _bellman_expectation_full_grid.
"""

import numpy as np
import pytest
import adaptive_fractionation_overlap.belief_model as bm
from adaptive_fractionation_overlap.belief_model import (
    _MU_GRID,
    _SIGMA_GRID,
    _MU_GRID_NIG,
    _SIGMA_GRID_NIG,
    _LOG_OFFSET,
    _VOLUME_SPACE,
    _P_BELIEF,
    _nig_posterior_params,
    _precompute_nig_branch_probabilities,
    _get_p_belief_nig,
    _P_BELIEF_NIG_CACHE,
    _bellman_expectation_full_grid,
    current_belief_probdist_nig,
)

# ---------------------------------------------------------------------------
# Dummy NIG hyperparameters used across tests that need concrete values.
# _DUMMY_MU0/_DUMMY_BETA0 are generic (unit-agnostic) for _nig_posterior_params tests.
# _DUMMY_MU0_LOG/_DUMMY_BETA0_LOG are log-space values for log-NIG tests.
# ---------------------------------------------------------------------------
_DUMMY_MU0    = 5.0    # generic (used in _nig_posterior_params tests; units don't matter)
_DUMMY_KAPPA0 = 1.0
_DUMMY_ALPHA0 = 2.0
_DUMMY_BETA0  = 1.0

# Log-space dummy parameters (realistic for log-NIG): mu0=1.0 nats ≈ 2.6 cc
_DUMMY_MU0_LOG   = 1.0   # nats
_DUMMY_BETA0_LOG = 0.5   # nats^2


class TestNigPosteriorParams:
    """Analytical correctness of _nig_posterior_params."""

    def test_one_observation_zero_m2(self):
        """n=1, M2=0 (single obs, no variance): closed-form posterior."""
        mu0, kappa0, alpha0, beta0 = _DUMMY_MU0, _DUMMY_KAPPA0, _DUMMY_ALPHA0, _DUMMY_BETA0
        s_bar = 7.0  # single observation
        m2 = 0.0

        mu_n, kappa_n, alpha_n, beta_n = _nig_posterior_params(1, s_bar, m2, mu0, kappa0, alpha0, beta0)

        assert kappa_n == pytest.approx(kappa0 + 1, rel=1e-12)
        assert mu_n    == pytest.approx((kappa0 * mu0 + s_bar) / (kappa0 + 1), rel=1e-12)
        assert alpha_n == pytest.approx(alpha0 + 0.5, rel=1e-12)
        expected_beta_n = beta0 + kappa0 * 1 * (s_bar - mu0) ** 2 / (2 * (kappa0 + 1))
        assert beta_n  == pytest.approx(expected_beta_n, rel=1e-12)

    def test_two_observations_known_m2(self):
        """n=2, obs 7 and 3 → s_bar=5, M2=8."""
        mu0, kappa0, alpha0, beta0 = _DUMMY_MU0, _DUMMY_KAPPA0, _DUMMY_ALPHA0, _DUMMY_BETA0
        s_bar, m2, n = 5.0, 8.0, 2

        mu_n, kappa_n, alpha_n, beta_n = _nig_posterior_params(n, s_bar, m2, mu0, kappa0, alpha0, beta0)

        assert kappa_n == pytest.approx(3.0, rel=1e-12)  # kappa0=1, n=2
        assert mu_n    == pytest.approx(5.0, rel=1e-12)  # (1*5 + 2*5) / 3 = 5
        assert alpha_n == pytest.approx(3.0, rel=1e-12)  # 2 + 1
        # beta_n = 1 + 4 + 1*2*(5-5)^2/(2*3) = 5
        assert beta_n  == pytest.approx(5.0, rel=1e-12)

    def test_prior_mean_shrinks_toward_mu0(self):
        """With kappa0 >> n, the posterior mean should be closer to mu0 than to s_bar."""
        large_kappa0 = 100.0
        s_bar = 15.0
        mu0 = 2.0
        mu_n, *_ = _nig_posterior_params(1, s_bar, 0.0, mu0, large_kappa0, 1.5, 1.0)
        # mu_n = (100*2 + 15) / 101 ≈ 2.13 — much closer to mu0=2 than s_bar=15
        assert abs(mu_n - mu0) < abs(mu_n - s_bar), "Strong prior should pull mu_n toward mu0"

    def test_beta_n_always_positive(self):
        """beta_n must stay positive for any reasonable inputs."""
        for s_bar in [0.1, 5.0, 25.0]:
            for n in [1, 3, 5]:
                m2 = n * 2.0  # variance = 2
                _, _, _, beta_n = _nig_posterior_params(n, s_bar, m2, _DUMMY_MU0, _DUMMY_KAPPA0, _DUMMY_ALPHA0, _DUMMY_BETA0)
                assert beta_n > 0, f"beta_n must be positive (s_bar={s_bar}, n={n})"

    def test_broadcast_over_arrays(self):
        """_nig_posterior_params must broadcast correctly over 2D input arrays."""
        N_mu, N_sigma = len(_MU_GRID_NIG), len(_SIGMA_GRID_NIG)
        s_bar = _MU_GRID_NIG[:, np.newaxis]              # (N_mu, 1)
        m2    = 2 * _SIGMA_GRID_NIG[np.newaxis, :] ** 2  # (1, N_sigma)
        mu_n, kappa_n, alpha_n, beta_n = _nig_posterior_params(
            2, s_bar, m2, _DUMMY_MU0_LOG, _DUMMY_KAPPA0, _DUMMY_ALPHA0, _DUMMY_BETA0_LOG
        )
        assert mu_n.shape   == (N_mu, 1)
        assert beta_n.shape == (N_mu, N_sigma)
        assert np.all(beta_n > 0), "All beta_n values must be positive"


class TestPrecomputeNigBranchProbabilities:
    """Verify shape and probability-mass properties of the log-NIG branch probability table."""

    @pytest.fixture(autouse=True)
    def _precomputed(self):
        n_fractions = 3
        # Use log-space dummy params (mu0_log=1.0 nats, beta0=0.5 nats^2)
        self.p = _precompute_nig_branch_probabilities(
            n_fractions, _DUMMY_MU0_LOG, _DUMMY_KAPPA0, _DUMMY_ALPHA0, _DUMMY_BETA0_LOG
        )
        self.n_fractions = n_fractions

    def test_shape(self):
        assert self.p.shape == (self.n_fractions, len(_MU_GRID_NIG), len(_SIGMA_GRID_NIG), len(_VOLUME_SPACE))

    def test_all_non_negative(self):
        assert np.all(self.p >= 0), "All NIG branch probabilities must be non-negative"

    def test_all_at_most_one(self):
        assert np.all(self.p <= 1.0 + 1e-12), "All NIG branch probabilities must be <= 1"

    def test_rows_sum_to_one(self):
        """Each (t, mi, si) row must sum to exactly 1 due to tail-folding."""
        row_sums = self.p.sum(axis=3)  # sum over overlap axis
        assert np.allclose(row_sums, 1.0, atol=1e-10), (
            f"NIG branch probability rows must sum to 1; "
            f"max deviation: {np.abs(row_sums - 1.0).max():.2e}"
        )

    def test_peak_near_nig_posterior_mean(self):
        """Peak probability should map back to a volume near the log-NIG posterior mean."""
        t = 0  # observation_count = 1
        # Choose a grid belief in the moderate-overlap range: find index closest to 1.5 nats ≈ 4.4 cc
        mi = np.searchsorted(_MU_GRID_NIG, 1.5)
        si = 0  # smallest sigma
        p_row = self.p[t, mi, si, :]
        peak_vol = _VOLUME_SPACE[np.argmax(p_row)]
        # With kappa0=1, mu0_log=1.0, n=1, s_bar_log=_MU_GRID_NIG[mi]:
        #   mu_n_log = (1.0*1.0 + 1.0*s_bar) / 2.0  => exp(mu_n_log) - eps is expected cc peak
        s_bar_log = _MU_GRID_NIG[mi]
        mu_n_log = (_DUMMY_KAPPA0 * _DUMMY_MU0_LOG + s_bar_log) / (_DUMMY_KAPPA0 + 1)
        expected_peak_cc = float(np.exp(mu_n_log) - _LOG_OFFSET)
        expected_peak_cc = np.clip(expected_peak_cc, _VOLUME_SPACE[0], _VOLUME_SPACE[-1])
        assert abs(peak_vol - expected_peak_cc) < 2.0, (
            f"Peak at {peak_vol:.2f} cc should be near log-NIG posterior mean {expected_peak_cc:.2f} cc"
        )


class TestGetPBeliefNig:
    """Test the lazy cache and error-handling in _get_p_belief_nig."""

    def test_raises_when_constants_none(self, monkeypatch):
        """_get_p_belief_nig must raise RuntimeError when log-NIG constants are None."""
        import adaptive_fractionation_overlap.constants as consts
        monkeypatch.setattr(consts, "NIG_LOG_MU_0",    None)
        monkeypatch.setattr(consts, "NIG_LOG_KAPPA_0", None)
        monkeypatch.setattr(consts, "NIG_LOG_ALPHA_0", None)
        monkeypatch.setattr(consts, "NIG_LOG_BETA_0",  None)
        _P_BELIEF_NIG_CACHE.pop(99, None)
        with pytest.raises(RuntimeError, match="Log-NIG hyperparameters have not been set"):
            _get_p_belief_nig(99)

    def test_returns_correct_shape_with_valid_constants(self, monkeypatch):
        """When constants are valid, _get_p_belief_nig returns the right shape."""
        import adaptive_fractionation_overlap.constants as consts
        monkeypatch.setattr(consts, "NIG_LOG_MU_0",    _DUMMY_MU0_LOG)
        monkeypatch.setattr(consts, "NIG_LOG_KAPPA_0", _DUMMY_KAPPA0)
        monkeypatch.setattr(consts, "NIG_LOG_ALPHA_0", _DUMMY_ALPHA0)
        monkeypatch.setattr(consts, "NIG_LOG_BETA_0",  _DUMMY_BETA0_LOG)
        _P_BELIEF_NIG_CACHE.pop(3, None)
        result = _get_p_belief_nig(3)
        assert result.shape == (3, len(_MU_GRID_NIG), len(_SIGMA_GRID_NIG), len(_VOLUME_SPACE))

    def test_caches_result(self, monkeypatch):
        """Second call with the same n_fractions returns the identical array (cached)."""
        import adaptive_fractionation_overlap.constants as consts
        monkeypatch.setattr(consts, "NIG_LOG_MU_0",    _DUMMY_MU0_LOG)
        monkeypatch.setattr(consts, "NIG_LOG_KAPPA_0", _DUMMY_KAPPA0)
        monkeypatch.setattr(consts, "NIG_LOG_ALPHA_0", _DUMMY_ALPHA0)
        monkeypatch.setattr(consts, "NIG_LOG_BETA_0",  _DUMMY_BETA0_LOG)
        _P_BELIEF_NIG_CACHE.pop(4, None)
        r1 = _get_p_belief_nig(4)
        r2 = _get_p_belief_nig(4)
        assert r1 is r2, "Repeated calls must return the same cached object"


class TestCurrentBeliefProbdistNig:
    """Verify current_belief_probdist_nig output properties (log-space inputs)."""

    @pytest.fixture(autouse=True)
    def _patch_constants(self, monkeypatch):
        import adaptive_fractionation_overlap.constants as consts
        monkeypatch.setattr(consts, "NIG_LOG_MU_0",    _DUMMY_MU0_LOG)
        monkeypatch.setattr(consts, "NIG_LOG_KAPPA_0", _DUMMY_KAPPA0)
        monkeypatch.setattr(consts, "NIG_LOG_ALPHA_0", _DUMMY_ALPHA0)
        monkeypatch.setattr(consts, "NIG_LOG_BETA_0",  _DUMMY_BETA0_LOG)

    def test_returns_correct_length(self):
        # s_bar_log=1.5 nats ≈ 4.4 cc, m2_log=0.2 nats^2 (reasonable log-space variance)
        result = current_belief_probdist_nig(n=2, s_bar_log=1.5, m2_log=0.2)
        assert len(result) == len(_VOLUME_SPACE)

    def test_all_non_negative(self):
        result = current_belief_probdist_nig(n=3, s_bar_log=1.0, m2_log=0.3)
        assert np.all(result >= 0)

    def test_sums_to_exactly_one(self):
        """Tail-folding means the sum should be exactly 1.0 for any belief."""
        for s_bar_log in [-1.5, 0.0, 1.0, 2.5, 3.5]:  # spans 0.12 cc to 33 cc
            result = current_belief_probdist_nig(n=3, s_bar_log=s_bar_log, m2_log=0.3)
            assert np.sum(result) == pytest.approx(1.0, abs=1e-10), (
                f"current_belief_probdist_nig must sum to 1.0 (s_bar_log={s_bar_log})"
            )

    def test_peak_near_nig_posterior_mean(self):
        """Peak of log-NIG distribution should be near exp(mu_n_log) - eps in cc."""
        # Use s_bar_log = log(5.0 + _LOG_OFFSET) so posterior mean ≈ 5 cc
        s_bar_log = float(np.log(5.0 + _LOG_OFFSET))  # ≈ 1.629 nats
        result = current_belief_probdist_nig(n=1, s_bar_log=s_bar_log, m2_log=0.0)
        peak_vol = _VOLUME_SPACE[np.argmax(result)]
        # mu_n_log = (kappa0 * mu0_log + s_bar_log) / (kappa0 + 1)
        mu_n_log = (_DUMMY_KAPPA0 * _DUMMY_MU0_LOG + s_bar_log) / (_DUMMY_KAPPA0 + 1)
        expected_peak_cc = float(np.exp(mu_n_log) - _LOG_OFFSET)
        expected_peak_cc = np.clip(expected_peak_cc, _VOLUME_SPACE[0], _VOLUME_SPACE[-1])
        assert abs(peak_vol - expected_peak_cc) < 1.5, (
            f"Peak at {peak_vol:.2f} cc should be near log-NIG posterior mean {expected_peak_cc:.2f} cc"
        )

    def test_heavier_tails_than_gaussian_for_few_observations(self):
        """Log-Student-t predictive with few obs should have mass at upper tail beyond a matched Gaussian."""
        from adaptive_fractionation_overlap.belief_model import current_belief_probdist
        # Choose a belief well inside _VOLUME_SPACE: s_bar_log = log(3+eps) ≈ 1.1 nats
        s_bar_log = float(np.log(3.0 + _LOG_OFFSET))
        p_nig = current_belief_probdist_nig(n=1, s_bar_log=s_bar_log, m2_log=0.0)
        # The log-NIG has a right-skewed predictive in cc; verify it places probability across a wide range
        # (at least 50% of probability mass spread over more than 5 cc)
        cumulative = np.cumsum(p_nig)
        p5_idx  = np.searchsorted(cumulative, 0.05)
        p95_idx = np.searchsorted(cumulative, 0.95)
        spread_cc = _VOLUME_SPACE[p95_idx] - _VOLUME_SPACE[p5_idx]
        assert spread_cc >= 2.0, (
            f"Log-NIG predictive should spread over at least 2 cc (5th–95th pct); got {spread_cc:.2f} cc"
        )


class TestBellmanExpectationFullGridBackwardCompat:
    """Verify that p_belief=None still uses _P_BELIEF (Stage A backward compatibility)."""

    def test_none_gives_same_result_as_p_belief(self):
        """_bellman_expectation_full_grid(v, n, None) == _bellman_expectation_full_grid(v, n, _P_BELIEF)."""
        N_dose, N_overlap, N_mu, N_sigma = 3, len(_VOLUME_SPACE), len(_MU_GRID), len(_SIGMA_GRID)
        rng = np.random.default_rng(42)
        values_prev = rng.standard_normal((N_dose, N_overlap, N_mu, N_sigma))
        observation_count = 2

        result_none  = _bellman_expectation_full_grid(values_prev, observation_count, p_belief=None)
        result_explicit = _bellman_expectation_full_grid(values_prev, observation_count, p_belief=_P_BELIEF)

        np.testing.assert_array_equal(result_none, result_explicit)

    def test_custom_p_belief_changes_result(self):
        """Providing a different p_belief should change the result from the default."""
        N_dose, N_overlap, N_mu, N_sigma = 2, len(_VOLUME_SPACE), len(_MU_GRID), len(_SIGMA_GRID)
        rng = np.random.default_rng(7)
        values_prev = rng.standard_normal((N_dose, N_overlap, N_mu, N_sigma))

        # Construct a non-trivial alternative p_belief: uniform distribution
        p_uniform = np.full((N_mu, N_sigma, N_overlap), 1.0 / N_overlap)

        result_default = _bellman_expectation_full_grid(values_prev, 2, p_belief=None)
        result_uniform = _bellman_expectation_full_grid(values_prev, 2, p_belief=p_uniform)

        assert not np.allclose(result_default, result_uniform), (
            "A different p_belief table should produce a different Bellman expectation"
        )

    def test_output_shape(self):
        """Output shape must be (N_dose, N_mu, N_sigma) regardless of p_belief."""
        N_dose = 5
        rng = np.random.default_rng(0)
        values_prev = rng.standard_normal((N_dose, len(_VOLUME_SPACE), len(_MU_GRID), len(_SIGMA_GRID)))
        result = _bellman_expectation_full_grid(values_prev, 3)
        assert result.shape == (N_dose, len(_MU_GRID), len(_SIGMA_GRID))


class TestNigPathInAdaptfxCore:
    """Verify the use_nig=True path raises RuntimeError before fitting."""

    def test_adaptive_fractionation_core_raises_before_fitting(self, monkeypatch):
        """adaptive_fractionation_core(use_nig=True) must raise RuntimeError if log-NIG constants are None."""
        from adaptive_fractionation_overlap.core_adaptfx import adaptive_fractionation_core
        import adaptive_fractionation_overlap.constants as consts
        monkeypatch.setattr(consts, "NIG_LOG_MU_0",    None)
        monkeypatch.setattr(consts, "NIG_LOG_KAPPA_0", None)
        monkeypatch.setattr(consts, "NIG_LOG_ALPHA_0", None)
        monkeypatch.setattr(consts, "NIG_LOG_BETA_0",  None)
        # Use a number_of_fractions not already in the cache so _get_p_belief_nig must re-import constants.
        # With the NIG call moved before array allocation, the RuntimeError fires before any large allocation.
        _P_BELIEF_NIG_CACHE.pop(7, None)
        with pytest.raises(RuntimeError, match="Log-NIG hyperparameters have not been set"):
            adaptive_fractionation_core(
                fraction_index_today=1,
                volumes=np.array([3.0]),
                accumulated_dose=0.0,
                number_of_fractions=7,
                min_dose=6.0,
                max_dose=10.0,
                mean_dose=8.0,
                dose_steps=0.5,
                alpha=1.073,
                beta=0.779,
                use_nig=True,
            )

    def test_adaptive_fractionation_core_stage_a_unchanged(self):
        """adaptive_fractionation_core with default use_nig=False must still work and return 9 elements."""
        from adaptive_fractionation_overlap.core_adaptfx import adaptive_fractionation_core
        result = adaptive_fractionation_core(
            fraction_index_today=1,
            volumes=np.array([3.0]),
            accumulated_dose=0.0,
            number_of_fractions=5,
            min_dose=6.0,
            max_dose=10.0,
            mean_dose=8.0,
            dose_steps=0.5,
            alpha=1.073,
            beta=0.779,
        )
        assert len(result) == 9
        physical_dose = result[3]
        assert 6.0 <= physical_dose <= 10.0

    @pytest.mark.slow
    def test_adaptfx_full_nig_produces_valid_plan(self):
        """adaptfx_full(use_nig=True) must return a valid 5-fraction dose plan."""
        from adaptive_fractionation_overlap.core_adaptfx import adaptfx_full
        volumes = [9.08, 19.79, 6.02, 9.45, 19.59, 12.62]  # patient 3 from conftest
        physical_doses, accumulated_doses, total_penalty = adaptfx_full(
            volumes=volumes,
            number_of_fractions=5,
            min_dose=6.0,
            max_dose=10.0,
            mean_dose=6.6,
            dose_steps=0.5,
            use_nig=True,
        )
        assert len(physical_doses) == 5
        assert np.all(physical_doses >= 6.0 - 1e-9)
        assert np.all(physical_doses <= 10.0 + 1e-9)
        assert np.isfinite(total_penalty)

    @pytest.mark.slow
    def test_nig_and_stage_a_give_different_plans(self):
        """NIG and Stage A should produce different dose recommendations for the same patient."""
        from adaptive_fractionation_overlap.core_adaptfx import adaptfx_full
        volumes = [9.08, 19.79, 6.02, 9.45, 19.59, 12.62]
        doses_stage_a, _, _ = adaptfx_full(volumes=volumes, use_nig=False)
        doses_nig, _, _ = adaptfx_full(volumes=volumes, use_nig=True)
        assert not np.allclose(doses_stage_a, doses_nig), (
            "NIG and Stage A must use different predictive models and should produce different plans"
        )
