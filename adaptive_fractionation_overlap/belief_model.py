"""
Belief model for adaptive fractionation: grids, belief updates, and Bellman operators.

The patient's PTV-OAR overlap distribution is modelled as a Gaussian with running
mean (mu) and standard deviation (sigma), updated after each fraction via Welford's
online algorithm.  All beliefs are discretised onto fixed (_MU_GRID, _SIGMA_GRID)
grids so that the DP value tables can be indexed directly.

Branch probabilities _P_BELIEF[mi, si, j] = P(overlap in bin j | belief (mu_grid[mi],
sigma_grid[si])) are precomputed once at module load and reused across all calls.

Tail-folding note
-----------------
_P_BELIEF uses tail-folding: the probability mass that falls below _VOLUME_SPACE[0] or
above _VOLUME_SPACE[-1] is folded into the boundary bins, so each row sums to exactly
1.0.  current_belief_probdist (used for the actual observed patient belief at each
fraction) does NOT fold tails, so its output can sum to slightly less than 1.0 if the
patient's belief tails extend outside _VOLUME_SPACE.  In practice this is negligible
because _VOLUME_SPACE is fixed at 44 cc, far beyond the maximum observed overlap
of ~29 cc in the 58-patient ACTION cohort.  For NIG extensibility (Stage B): replace current_belief_probdist
with a Student-t CDF-difference; _P_BELIEF would then require a corresponding update.
"""

import numpy as np
from scipy.stats import norm

from .helper_functions import nearest_idx


# ---------------------------------------------------------------------------
# Belief grids
# ---------------------------------------------------------------------------

# Non-uniform mu grid optimized for the observed 58-patient cohort prefix-mean distribution
# under a 150-point budget.  A sweep over N_mu in [50, 500] showed quality saturates at ~150:
# above N_mu=150 the benefit fluctuates within ±0.005 ccGy (quantisation noise), making
# N_mu=150 the principled choice — 99.95% of N=500 benefit at 2.3× lower compute than N_mu=280.
_MU_GRID = np.unique(np.concatenate([
    np.linspace(0.0,  1.0,  38),  # 38 points, step ~0.0270 cc
    np.linspace(1.05, 4.0,  39),  # 39 points, step ~0.0776 cc
    np.linspace(4.1,  10.0, 42),  # 42 points, step ~0.1439 cc
    np.linspace(10.2, 16.0, 15),  # 15 points, step ~0.4143 cc
    np.linspace(16.5, 30.0, 16),  # 16 points, step ~0.9000 cc
]))  # 150 grid points total

# Non-uniform sigma grid: fine resolution in [0, 0.7] cc where 75% of patients fall,
# coarser in the tail.  Range extended to 4.5 cc to avoid clipping outlier patients
# (observed max σ ≈ 4.05 cc on the 58-patient ACTION cohort, patient 3).
# Peak memory: (4, 70, 441, 150, 11) × 8 bytes ≈ 1.63 GB (1.52 GiB).
_SIGMA_GRID = np.unique(np.concatenate([
    np.linspace(0.05, 0.7, 5),  # step ~0.163 cc  (p0–p75 of clinical σ)
    np.linspace(0.8,  1.8, 3),  # step ~0.500 cc  (p75–p90)
    np.linspace(2.0,  4.5, 3),  # step ~1.250 cc  (tail, covers max observed σ 4.05 cc)
]))  # 11 grid points total
_SIGMA_MIN = float(_SIGMA_GRID[0])

# Fixed overlap state space: 0 to 44 cc in 0.1 cc steps, matching TPS output resolution.
# Hardcoded to 44 cc (independent of _SIGMA_GRID) so that bin width stays exactly 0.1 cc.
# 44 cc covers mu_grid_max (30 cc) + 3 × sigma_grid_max (4.5 cc) = 43.5 cc with margin to spare;
# the tail probability beyond 44 cc is negligible for any belief on the grid.
_VOLUME_SPACE = np.linspace(0.0, 44.0, 441)  # 0.1 cc steps, fixed upper bound


# ---------------------------------------------------------------------------
# Log-NIG grids (used when use_nig=True; Stage B)
# ---------------------------------------------------------------------------

# Log-transform offset: observations are mapped as log(v + _LOG_OFFSET) before NIG update.
# eps=0.1 cc keeps the domain well away from log(0) and converts the right-skewed overlap
# distribution (skewness 2.73 in cc) to near-symmetric (skewness -0.15 in log-space).
_LOG_OFFSET = 0.1  # cc

# Pre-computed log-space volume grid used for Welford updates in NIG Bellman operators.
_LOG_VOLUME_SPACE = np.log(_VOLUME_SPACE + _LOG_OFFSET)  # (N_overlap,) in nats

# mu grid in log-space (nats): covers log(0+eps)=-2.30 to log(30+eps)=3.40 with margin.
# Non-uniform: finer near the cohort median (~0.2 nats ≈ 1.1 cc), coarser in the tails.
# 100-point budget: p1–p99 of NIG posterior mu covers [-2.15, 2.90] nats on the
# 58-patient ACTION cohort (all fractions), so the grid extends to ±0.4 nats of margin.
_MU_GRID_NIG = np.unique(np.concatenate([
    np.linspace(-2.5, -1.5, 18),  # very low overlap  (0.08–0.22 cc)
    np.linspace(-1.4,  0.5, 32),  # low overlap        (0.25–1.65 cc)
    np.linspace( 0.6,  2.0, 30),  # moderate overlap   (1.82–7.4 cc)
    np.linspace( 2.1,  3.5, 20),  # high overlap       (8.2–33 cc)
]))  # 100 grid points total

# sigma grid in log-space (nats): covers log-space sigma from very tight to very wide.
# Empirical range on the 58-patient cohort: p1=0.34, p99=0.75 nats.
# Grid extends to [0.05, 1.5] to handle edge cases.
# Peak memory: (4, 70, 441, 100, 10) × 8 bytes ≈ 0.99 GB.
_SIGMA_GRID_NIG = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.70, 0.90, 1.20, 1.50])  # 10 pts
_SIGMA_MIN_NIG = float(_SIGMA_GRID_NIG[0])


# ---------------------------------------------------------------------------
# Branch probabilities (precomputed once at module load)
# ---------------------------------------------------------------------------

def _precompute_branch_probabilities():
    """Return p[mi, si, j] = P(overlap in bin j | belief (mu_grid[mi], sigma_grid[si])).

    Uses CDF differences over each bin. Left/right tails are folded into the first/last
    bin so that probabilities sum to 1 for every belief.
    """
    spacing = _VOLUME_SPACE[1] - _VOLUME_SPACE[0]  # bin width of the overlap grid (uniform, since _VOLUME_SPACE is linspace)
    upper_bounds = _VOLUME_SPACE + spacing / 2
    lower_bounds = _VOLUME_SPACE - spacing / 2
    # P(bin j | mu, sigma) = CDF(upper_bounds[j]) - CDF(lower_bounds[j]), broadcast over all (mu, sigma) pairs
    probabilities = (
        norm.cdf(upper_bounds[None, None, :], loc=_MU_GRID[:, None, None], scale=_SIGMA_GRID[None, :, None])
        - norm.cdf(lower_bounds[None, None, :], loc=_MU_GRID[:, None, None], scale=_SIGMA_GRID[None, :, None])
    )
    # Fold left and right tails into the boundary bins so each row sums to 1
    probabilities[:, :, 0] += norm.cdf(lower_bounds[0], loc=_MU_GRID[:, None], scale=_SIGMA_GRID[None, :])
    probabilities[:, :, -1] += 1.0 - norm.cdf(upper_bounds[-1], loc=_MU_GRID[:, None], scale=_SIGMA_GRID[None, :])
    return probabilities

_P_BELIEF = _precompute_branch_probabilities()  # shape: (N_mu, N_sigma, N_overlap), computed once at module load


# ---------------------------------------------------------------------------
# Belief update (Welford) and Bellman operators
# ---------------------------------------------------------------------------

def _hypothetical_belief_grid_indices(mu, sigma, volume_space, observation_count):
    """For each hypothetical next overlap in volume_space, compute where the updated
    belief (mu, sigma) would land on the (_MU_GRID, _SIGMA_GRID) grid.

    Returns next_mi, next_si: (N_overlap,) integer index arrays into _MU_GRID and
    _SIGMA_GRID, used to look up future values in the DP values array.
    Uses Welford's online algorithm to update the running mean and variance.
    """
    mu_prime = (observation_count * mu + volume_space) / (observation_count + 1)  # new mean after observing each hypothetical overlap
    sigma2_prime = (observation_count * sigma ** 2 + (volume_space - mu) * (volume_space - mu_prime)) / (observation_count + 1)  # new variance after observing each hypothetical overlap
    sigma_prime = np.sqrt(np.maximum(sigma2_prime, _SIGMA_MIN ** 2))  # new sigma, clamped to avoid zero with few observations
    mu_prime = np.clip(mu_prime, _MU_GRID[0], _MU_GRID[-1])          # clamp new mu to grid's range
    sigma_prime = np.clip(sigma_prime, _SIGMA_GRID[0], _SIGMA_GRID[-1])  # clamp new sigma to grid's range
    next_mu_index = nearest_idx(mu_prime, _MU_GRID)        # nearest mu grid index for each hypothetical next overlap
    next_sigma_index = nearest_idx(sigma_prime, _SIGMA_GRID)  # nearest sigma grid index for each hypothetical next overlap
    return next_mu_index, next_sigma_index  # indices rather than values so callers can directly index into the DP values array


def _bellman_expectation(values_prev, volume_space, p_branch, mu, sigma, observation_count):
    """Bellman expectation over all hypothetical next overlaps for a single belief (mu, sigma).

    Sums future values weighted by branch probabilities:
        sum_j p_branch[j] * values_prev[d, j, next_mi[j], next_si[j]]

    values_prev: (N_dose, N_overlap, N_mu, N_sigma)
    p_branch:    (N_overlap,) probability of each next overlap under current belief
    Returns:     (N_dose,) expected future value for each accumulated dose state
    """
    next_mi, next_si = _hypothetical_belief_grid_indices(mu, sigma, volume_space, observation_count) # If overlap was j, where would the belief (mu, sigma) land on the grid?
    branch_vals = values_prev[:, np.arange(len(volume_space)), next_mi, next_si]  # Value function of the next belief state
    return (branch_vals * p_branch[None, :]).sum(axis=1) # Probability-weighted sum of next states' value functions


def _bellman_expectation_full_grid(values_prev, observation_count, p_belief=None):
    """Compute the expected future value for every (accumulated_dose, belief_mu, belief_sigma) triple.

    Vectorised full-grid counterpart of _bellman_expectation. Where _bellman_expectation
    operates on a single belief (mu, sigma), this function operates on all (N_mu × N_sigma)
    beliefs simultaneously, making it suitable for the backward-induction DP loop.

    Steps:
      1. Welford belief update: for each (mu_grid, sigma_grid, overlap_bin) triple, compute
         the updated belief (mu', sigma') after observing that overlap.
      2. Map (mu', sigma') to the nearest grid indices (next_mi, next_si).
      3. Accumulate the probability-weighted future values across all overlap bins.

    Args:
        values_prev: value function of the next DP state; shape (N_dose, N_overlap, N_mu, N_sigma).
        observation_count (int): number of overlap observations already made at this fraction
                                 (used as the sample count n in the Welford update formula).
        p_belief: optional branch probability table of shape (N_mu, N_sigma, N_overlap).
                  If None (default), uses _P_BELIEF (Stage A Gaussian). Pass a slice of
                  _P_BELIEF_NIG_CACHE[n_fractions][observation_count-1] for Stage B.

    Returns:
        np.ndarray: expected future value for each (dose, mu, sigma); shape (N_dose, N_mu, N_sigma).
    """
    N_dose, N_overlap, N_mu, N_sigma = values_prev.shape[0], len(_VOLUME_SPACE), len(_MU_GRID), len(_SIGMA_GRID)

    # ----- Step 1: belief transitions for all (mi, si, j) at this observation_count -----

    # Expand grids into 3D arrays so they can broadcast together:
    mu_vals = _MU_GRID[:, None, None]        # (N_mu, 1, 1)
    sigma_vals = _SIGMA_GRID[None, :, None]  # (1, N_sigma, 1)
    o_vals = _VOLUME_SPACE[None, None, :]    # (1, 1, N_overlap)

    mu_prime = (observation_count * mu_vals + o_vals) / (observation_count + 1)  # new mean after observing each hypothetical overlap
    delta = o_vals - mu_vals         # deviation of new observation from old mean
    delta_prime = o_vals - mu_prime  # deviation of new observation from new mean (Welford variance term)
    sigma2_prime = (observation_count * sigma_vals ** 2 + delta * delta_prime) / (observation_count + 1)  # new variance (Welford online update)
    sigma_prime = np.sqrt(np.maximum(sigma2_prime, _SIGMA_MIN ** 2))  # new std, floored at _SIGMA_MIN
    mu_prime = np.clip(mu_prime, _MU_GRID[0], _MU_GRID[-1])
    sigma_prime = np.clip(sigma_prime, _SIGMA_GRID[0], _SIGMA_GRID[-1])
    # Nearest grid indices — shape (N_mu, N_sigma, N_overlap)
    # searchsorted is O(N*log(G)) vs argmin's O(N*G), ~35x faster for G=280.
    next_mi = nearest_idx(mu_prime, _MU_GRID)
    next_si = nearest_idx(sigma_prime, _SIGMA_GRID)

    # ----- Steps 2+3: accumulate probability-weighted future values in a j-loop -----
    # This avoids materialising the full (N_dose, N_mu, N_sigma, N_overlap) = ~660 MB
    # branch_vals array; instead each j-slice is ~1.5 MB and is processed on the fly.
    next_b = next_mi * N_sigma + next_si  # flat belief index, shape (N_mu, N_sigma, N_overlap)
    vals_flat = values_prev.reshape(N_dose, N_overlap, -1)  # (N_dose, N_overlap, N_belief)
    p = _P_BELIEF if p_belief is None else p_belief  # (N_mu, N_sigma, N_overlap)
    future_value_prob_full = np.zeros((N_dose, N_mu, N_sigma))
    for j in range(N_overlap):
        # vals_flat[:, j, next_b[:, :, j]]: (N_dose, N_mu, N_sigma) — one j-slice
        future_value_prob_full += vals_flat[:, j, :][:, next_b[:, :, j]] * p[None, :, :, j]

    return future_value_prob_full


# ---------------------------------------------------------------------------
# Log-NIG Bellman operators (Stage B)
# ---------------------------------------------------------------------------

def _bellman_expectation_nig(values_prev, current_overlap_probs, initial_belief_mu, initial_belief_sigma, observation_count):
    """Single-belief Bellman expectation for log-NIG Stage B.

    Identical to _bellman_expectation but uses _MU_GRID_NIG, _SIGMA_GRID_NIG, and
    log-space Welford updates (observations transformed via log(v + _LOG_OFFSET)).

    Args:
        values_prev: (N_dose, N_overlap, N_mu_nig, N_sigma_nig)
        current_overlap_probs: (N_overlap,) probability of each overlap bin
        initial_belief_mu (float): current NIG posterior log-mean (nats)
        initial_belief_sigma (float): current NIG posterior log-sigma mode (nats)
        observation_count (int): number of overlap observations already made

    Returns:
        (N_dose,) expected future value for each accumulated dose state
    """
    log_o = _LOG_VOLUME_SPACE  # (N_overlap,) already in nats

    mu_prime = (observation_count * initial_belief_mu + log_o) / (observation_count + 1)
    delta = log_o - initial_belief_mu
    delta_prime = log_o - mu_prime
    sigma2_prime = (observation_count * initial_belief_sigma ** 2 + delta * delta_prime) / (observation_count + 1)
    sigma_prime = np.sqrt(np.maximum(sigma2_prime, _SIGMA_MIN_NIG ** 2))
    mu_prime = np.clip(mu_prime, _MU_GRID_NIG[0], _MU_GRID_NIG[-1])
    sigma_prime = np.clip(sigma_prime, _SIGMA_GRID_NIG[0], _SIGMA_GRID_NIG[-1])
    next_mi = nearest_idx(mu_prime, _MU_GRID_NIG)
    next_si = nearest_idx(sigma_prime, _SIGMA_GRID_NIG)

    branch_vals = values_prev[:, np.arange(len(log_o)), next_mi, next_si]
    return (branch_vals * current_overlap_probs[None, :]).sum(axis=1)


def _bellman_expectation_full_grid_nig(values_prev, observation_count, p_belief_nig):
    """Full-grid Bellman expectation for log-NIG Stage B.

    Identical to _bellman_expectation_full_grid but:
      - uses _MU_GRID_NIG and _SIGMA_GRID_NIG as the belief state grids, and
      - applies Welford updates in log-space: o_vals → log(o + _LOG_OFFSET).

    Args:
        values_prev: shape (N_dose, N_overlap, N_mu_nig, N_sigma_nig)
        observation_count (int): number of overlap observations already made
        p_belief_nig: shape (N_mu_nig, N_sigma_nig, N_overlap) — log-NIG branch probs

    Returns:
        np.ndarray of shape (N_dose, N_mu_nig, N_sigma_nig)
    """
    N_dose = values_prev.shape[0]
    N_overlap = len(_VOLUME_SPACE)
    N_mu_nig = len(_MU_GRID_NIG)
    N_sigma_nig = len(_SIGMA_GRID_NIG)

    mu_vals    = _MU_GRID_NIG[:, None, None]       # (N_mu_nig, 1, 1) nats
    sigma_vals = _SIGMA_GRID_NIG[None, :, None]    # (1, N_sigma_nig, 1) nats
    log_o      = _LOG_VOLUME_SPACE[None, None, :]  # (1, 1, N_overlap) nats

    # Welford update in log-space
    mu_prime = (observation_count * mu_vals + log_o) / (observation_count + 1)
    delta = log_o - mu_vals
    delta_prime = log_o - mu_prime
    sigma2_prime = (observation_count * sigma_vals ** 2 + delta * delta_prime) / (observation_count + 1)
    sigma_prime = np.sqrt(np.maximum(sigma2_prime, _SIGMA_MIN_NIG ** 2))
    mu_prime = np.clip(mu_prime, _MU_GRID_NIG[0], _MU_GRID_NIG[-1])
    sigma_prime = np.clip(sigma_prime, _SIGMA_GRID_NIG[0], _SIGMA_GRID_NIG[-1])

    next_mi = nearest_idx(mu_prime, _MU_GRID_NIG)       # (N_mu_nig, N_sigma_nig, N_overlap)
    next_si = nearest_idx(sigma_prime, _SIGMA_GRID_NIG)

    next_b = next_mi * N_sigma_nig + next_si
    vals_flat = values_prev.reshape(N_dose, N_overlap, -1)
    future_value_prob_full = np.zeros((N_dose, N_mu_nig, N_sigma_nig))
    for j in range(N_overlap):
        future_value_prob_full += vals_flat[:, j, :][:, next_b[:, :, j]] * p_belief_nig[None, :, :, j]

    return future_value_prob_full


# ---------------------------------------------------------------------------
# NIG (Normal-Inverse-Gamma) Stage B infrastructure
# ---------------------------------------------------------------------------

def _nig_posterior_params(n, s_bar, m2, mu0, kappa0, alpha0, beta0):
    """Compute NIG posterior parameters given Welford sufficient statistics.

    The NIG posterior after n observations with running mean s_bar and
    sum-of-squared-deviations M2 (= n * sample_var) is:
        kappa_n = kappa0 + n
        mu_n    = (kappa0*mu0 + n*s_bar) / kappa_n
        alpha_n = alpha0 + n/2
        beta_n  = beta0 + M2/2 + kappa0*n*(s_bar - mu0)^2 / (2*kappa_n)

    Args:
        n: number of observations (int or broadcastable array).
        s_bar: sample mean (float or array).
        m2: Welford M2 = sum of squared deviations from mean (float or array).
        mu0, kappa0, alpha0, beta0: NIG prior hyperparameters.

    Returns:
        (mu_n, kappa_n, alpha_n, beta_n): posterior parameters, same broadcast shape as inputs.
    """
    kappa_n = kappa0 + n
    mu_n    = (kappa0 * mu0 + n * s_bar) / kappa_n
    alpha_n = alpha0 + n / 2
    beta_n  = beta0 + m2 / 2 + kappa0 * n * (s_bar - mu0) ** 2 / (2 * kappa_n)
    return mu_n, kappa_n, alpha_n, beta_n


def _precompute_nig_branch_probabilities(n_fractions, mu0_log, kappa0, alpha0, beta0):
    """Return p[t, mi, si, j] = P(overlap bin j | obs_count=t+1, belief (MU_GRID_NIG[mi], SIGMA_GRID_NIG[si])).

    Log-NIG version: operates on log(v + _LOG_OFFSET) so the Student-t predictive is a
    log-Student-t in original cc space. CDF differences are computed at the log-transformed
    bin edges of _VOLUME_SPACE, giving probabilities that sum to exactly 1 via tail-folding.

    Grid convention: MU_GRID_NIG[mi] is the running log-mean (nats), SIGMA_GRID_NIG[si] is
    the running log-std; M2_log = n * sigma_log^2 at observation_count n = t+1.

    Shape: (n_fractions, N_mu_nig, N_sigma_nig, N_overlap).
    t=0 → observation_count=1; t=n_fractions-1 → observation_count=n_fractions.
    """
    from scipy.stats import t as student_t

    N_mu_nig    = len(_MU_GRID_NIG)
    N_sigma_nig = len(_SIGMA_GRID_NIG)
    N_overlap   = len(_VOLUME_SPACE)

    # Log-space bin edges: transform cc bin centres ± half-width to log scale
    spacing         = _VOLUME_SPACE[1] - _VOLUME_SPACE[0]
    upper_bounds_cc = _VOLUME_SPACE + spacing / 2
    lower_bounds_cc = np.maximum(_VOLUME_SPACE - spacing / 2, 0.0)  # never go below 0 cc
    upper_bounds_log = np.log(upper_bounds_cc + _LOG_OFFSET)  # (N_overlap,) nats
    lower_bounds_log = np.log(lower_bounds_cc + _LOG_OFFSET)  # (N_overlap,) nats

    result = np.zeros((n_fractions, N_mu_nig, N_sigma_nig, N_overlap))

    for t, n in enumerate(range(1, n_fractions + 1)):
        # Recover Welford sufficient statistics from grid coordinates (log-space)
        s_bar = _MU_GRID_NIG[:, np.newaxis]              # (N_mu_nig, 1): running log-mean
        m2    = n * _SIGMA_GRID_NIG[np.newaxis, :] ** 2  # (1, N_sigma_nig): Welford M2 in log-space

        # NIG posterior parameters (all in log-space)
        kappa_n = kappa0 + n                                                           # scalar
        mu_n    = (kappa0 * mu0_log + n * s_bar) / kappa_n                            # (N_mu_nig, 1)
        alpha_n = alpha0 + n / 2                                                       # scalar
        beta_n  = beta0 + m2 / 2 + kappa0 * n * (s_bar - mu0_log) ** 2 / (2 * kappa_n)  # (N_mu_nig, N_sigma_nig)

        # Log-Student-t predictive: Student-t on log(v + eps)
        dof    = 2 * alpha_n                                              # scalar
        center = mu_n                                                     # (N_mu_nig, 1)
        scale  = np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n))  # (N_mu_nig, N_sigma_nig)

        # CDF differences at log-space bin edges — broadcast to (N_mu_nig, N_sigma_nig, N_overlap)
        probs = (
            student_t.cdf(upper_bounds_log[np.newaxis, np.newaxis, :], df=dof,
                          loc=center[:, :, np.newaxis], scale=scale[:, :, np.newaxis])
            - student_t.cdf(lower_bounds_log[np.newaxis, np.newaxis, :], df=dof,
                            loc=center[:, :, np.newaxis], scale=scale[:, :, np.newaxis])
        )  # (N_mu_nig, N_sigma_nig, N_overlap)

        # Tail-fold: fold left log-tail (v < lower_bounds_cc[0]) into bin 0,
        #            fold right log-tail (v > upper_bounds_cc[-1]) into last bin
        probs[:, :, 0]  += student_t.cdf(lower_bounds_log[0], df=dof, loc=center[:, :], scale=scale)
        probs[:, :, -1] += 1.0 - student_t.cdf(upper_bounds_log[-1], df=dof, loc=center[:, :], scale=scale)

        result[t] = probs

    return result


_P_BELIEF_NIG_CACHE: dict = {}  # maps n_fractions → (n_fractions, N_mu_nig, N_sigma_nig, N_overlap) array


def _get_p_belief_nig(n_fractions: int) -> np.ndarray:
    """Return the log-NIG branch probability table for a given treatment length.

    Lazily computed and cached per n_fractions. Raises RuntimeError if log-NIG
    hyperparameters have not been set in constants.py.

    Returns:
        np.ndarray of shape (n_fractions, N_mu_nig, N_sigma_nig, N_overlap).
        Index [t, mi, si, j] = P(overlap bin j | obs_count=t+1,
                                  belief=(MU_GRID_NIG[mi], SIGMA_GRID_NIG[si])).
    """
    if n_fractions not in _P_BELIEF_NIG_CACHE:
        from .constants import NIG_LOG_MU_0, NIG_LOG_KAPPA_0, NIG_LOG_ALPHA_0, NIG_LOG_BETA_0
        if any(v is None for v in (NIG_LOG_MU_0, NIG_LOG_KAPPA_0, NIG_LOG_ALPHA_0, NIG_LOG_BETA_0)):
            raise RuntimeError(
                "Log-NIG hyperparameters have not been set. "
                "Run scripts/fit_nig_hyperparams.py --log-space to fit them from data."
            )
        _P_BELIEF_NIG_CACHE[n_fractions] = _precompute_nig_branch_probabilities(
            n_fractions, NIG_LOG_MU_0, NIG_LOG_KAPPA_0, NIG_LOG_ALPHA_0, NIG_LOG_BETA_0
        )
    return _P_BELIEF_NIG_CACHE[n_fractions]


# ---------------------------------------------------------------------------
# Current-belief branch probabilities (evaluated on-the-fly, not precomputed)
# ---------------------------------------------------------------------------

def current_belief_probdist(mu: float, sigma: float) -> np.ndarray:
    """Return P(overlap in bin j | current patient belief (mu, sigma)) over _VOLUME_SPACE.

    Evaluates Gaussian CDF differences over the fixed _VOLUME_SPACE grid, without
    tail-folding.  The probabilities may therefore sum to slightly less than 1.0 when
    the belief tails extend beyond _VOLUME_SPACE; in practice this is negligible
    because _VOLUME_SPACE is fixed at 44 cc, far beyond the maximum observed overlap.

    Unlike _P_BELIEF (precomputed for all grid points, with tail-folding), this
    function evaluates on-the-fly for the patient's actual current belief — which
    may lie off-grid between DP solver calls.

    For NIG extensibility (Stage B): replace this function with a Student-t
    CDF-difference over _VOLUME_SPACE, leaving _bellman_expectation unchanged.

    Args:
        mu (float): current belief mean (cc).
        sigma (float): current belief standard deviation (cc).

    Returns:
        np.ndarray: probability of each overlap bin j; shape (N_overlap,).
    """
    from .helper_functions import probdist
    return probdist((mu, sigma), _VOLUME_SPACE)


def current_belief_probdist_nig(n: int, s_bar_log: float, m2_log: float) -> np.ndarray:
    """Return log-Student-t predictive P(overlap in bin j | n log-obs, log-mean s_bar_log, M2_log) over _VOLUME_SPACE.

    Log-NIG version: the NIG model operates on log(v + _LOG_OFFSET), so the predictive is a
    log-Student-t (Student-t applied to log-space). Bin probabilities are computed via CDF
    differences at log-transformed bin edges. Uses tail-folding so the output sums to exactly 1.0.

    Args:
        n (int): number of overlap observations accumulated so far.
        s_bar_log (float): running mean of log(overlap + eps) in nats.
        m2_log (float): Welford M2 in log-space = sum of squared log-deviations from s_bar_log.

    Returns:
        np.ndarray: probability of each overlap bin j; shape (N_overlap,).
    """
    from scipy.stats import t as student_t
    from .constants import NIG_LOG_MU_0, NIG_LOG_KAPPA_0, NIG_LOG_ALPHA_0, NIG_LOG_BETA_0

    mu_n, kappa_n, alpha_n, beta_n = _nig_posterior_params(
        n, s_bar_log, m2_log, NIG_LOG_MU_0, NIG_LOG_KAPPA_0, NIG_LOG_ALPHA_0, NIG_LOG_BETA_0
    )
    dof    = float(2 * alpha_n)
    center = float(mu_n)
    scale  = float(np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n)))

    spacing         = _VOLUME_SPACE[1] - _VOLUME_SPACE[0]
    upper_bounds_cc = _VOLUME_SPACE + spacing / 2
    lower_bounds_cc = np.maximum(_VOLUME_SPACE - spacing / 2, 0.0)
    upper_bounds_log = np.log(upper_bounds_cc + _LOG_OFFSET)
    lower_bounds_log = np.log(lower_bounds_cc + _LOG_OFFSET)

    probs = (student_t.cdf(upper_bounds_log, df=dof, loc=center, scale=scale)
             - student_t.cdf(lower_bounds_log, df=dof, loc=center, scale=scale))
    probs[0]  += student_t.cdf(lower_bounds_log[0], df=dof, loc=center, scale=scale)
    probs[-1] += 1.0 - student_t.cdf(upper_bounds_log[-1], df=dof, loc=center, scale=scale)
    return probs
