"""Fit NIG hyperparameters (mu0, kappa0, alpha0, beta0) to the 58-patient ACTION cohort.

Uses empirical Bayes: maximise the sum of sequential NIG marginal log-likelihoods
over all patients. The sequential marginal likelihood factorises as:

    log p(o_1, ..., o_n) = sum_t log p(o_t | o_1, ..., o_{t-1})

where each predictive factor is a Student-t:

    o_t | o_1,...,o_{t-1} ~ t_{2 alpha_{t-1}}(mu_{t-1}, sqrt(beta_{t-1}*(kappa_{t-1}+1)/(alpha_{t-1}*kappa_{t-1})))

Flags
-----
--log-space
    Fit in log-space: each observation is transformed as log(v + LOG_OFFSET) before
    fitting. The NIG model then applies to log-transformed data, producing a
    log-Student-t predictive in the original cc space. This corrects for the
    right-skewed bimodal overlap distribution (skewness=2.73), which violates the
    symmetric Gaussian assumption of the NIG. In log-space the distribution is
    near-symmetric (skewness=-0.15), yielding an interior alpha0 solution rather
    than being forced to the floor of 1.0.

    When --log-space is set, the alpha0 constraint changes to alpha0 >= 0.5 instead
    of alpha0 >= 1.0, reflecting that the log-Student-t predictive has finite
    variance as long as alpha0 > 0 (log-dof > 1 for finite first moment in cc space).
    In practice the optimiser finds alpha0 ≈ 2.74.

Reparameterisation for L-BFGS-B positivity constraints (original-scale):
    kappa0  = exp(log_kappa0)              bounds: [1e-3, 5]
    alpha0  = 1 + exp(log_alpha_shift)     enforces alpha0 >= 1 (finite variance at fraction 1)
    mu0     = mu0                          bounds: [0, 30] cc
    beta0   = exp(log_beta0)              bounds: [exp(-5), exp(10)]

Reparameterisation (log-space):
    kappa0  = exp(log_kappa0)              bounds: [1e-3, 5]
    alpha0  = 0.5 + exp(log_alpha_shift)   enforces alpha0 >= 0.5
    mu0     = mu0_log                      bounds: [-3, 4] nats  (covers 0.05–55 cc)
    beta0   = exp(log_beta0_log)           bounds: [exp(-5), exp(5)]

After fitting, copy the printed constants into
adaptive_fractionation_overlap/constants.py.
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "docs", "papers", "2025", "table8.csv"
)
FRACTION_COLS = [
    "overlap_f1_cc", "overlap_f2_cc", "overlap_f3_cc", "overlap_f4_cc", "overlap_f5_cc"
]
LOG_OFFSET = 0.1  # epsilon for log(v + LOG_OFFSET) transform


def _nig_update_sequential(kappa, mu, alpha, beta, obs):
    """One-step NIG conjugate update after observing `obs`."""
    kappa_new = kappa + 1
    mu_new    = (kappa * mu + obs) / kappa_new
    alpha_new = alpha + 0.5
    beta_new  = beta + 0.5 * kappa * (obs - mu) ** 2 / kappa_new
    return kappa_new, mu_new, alpha_new, beta_new


def _sequential_log_marginal(observations, mu0, kappa0, alpha0, beta0):
    """Sum of log p(o_t | o_{<t}) for one patient under NIG(mu0,kappa0,alpha0,beta0) prior."""
    kappa, mu, alpha, beta = kappa0, mu0, alpha0, beta0
    log_lik = 0.0
    for obs in observations:
        dof   = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        log_lik += student_t.logpdf(obs, df=dof, loc=mu, scale=scale)
        kappa, mu, alpha, beta = _nig_update_sequential(kappa, mu, alpha, beta, obs)
    return log_lik


def _negative_total_log_marginal(params, all_patients):
    """Negative total log marginal likelihood (objective for L-BFGS-B minimisation)."""
    log_kappa0, log_alpha_shift, mu0, log_beta0 = params
    kappa0 = np.exp(log_kappa0)
    alpha0 = 1.0 + np.exp(log_alpha_shift)  # enforces alpha0 >= 1
    beta0  = np.exp(log_beta0)
    total = sum(
        _sequential_log_marginal(obs_seq, mu0, kappa0, alpha0, beta0)
        for obs_seq in all_patients
    )
    return -total


def _negative_total_log_marginal_logspace(params, all_patients_log):
    """Negative total log marginal likelihood in log-space."""
    log_kappa0, log_alpha_shift, mu0_log, log_beta0_log = params
    kappa0 = np.exp(log_kappa0)
    alpha0 = 0.5 + np.exp(log_alpha_shift)  # enforces alpha0 >= 0.5
    beta0  = np.exp(log_beta0_log)
    total = sum(
        _sequential_log_marginal(obs_seq_log, mu0_log, kappa0, alpha0, beta0)
        for obs_seq_log in all_patients_log
    )
    return -total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--log-space", action="store_true",
        help="Fit NIG in log(overlap + LOG_OFFSET) space (recommended; fixes right-skew issue)."
    )
    parser.add_argument(
        "--fix-kappa0", type=float, default=None,
        help="Fix kappa0 to this value and optimise only (mu0, alpha0, beta0). "
             "Only applies with --log-space. E.g. --fix-kappa0 0.001."
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    raw_values = df[FRACTION_COLS].values  # (58, 5)

    if args.log_space:
        all_patients = [np.log(row + LOG_OFFSET) for row in raw_values]
        overall_mean = float(np.mean([obs for seq in all_patients for obs in seq]))
        overall_var  = float(np.var([obs for seq in all_patients for obs in seq]))

        if args.fix_kappa0 is not None:
            # Fixed-kappa0 mode: optimise only (mu0_log, alpha0, beta0).
            fixed_kappa0 = args.fix_kappa0

            def _neg_logmarg_fixed_kappa(params_3, all_pts):
                log_alpha_shift, mu0_log, log_beta0_log = params_3
                kappa0 = fixed_kappa0
                alpha0 = 0.5 + np.exp(log_alpha_shift)
                beta0  = np.exp(log_beta0_log)
                total = sum(
                    _sequential_log_marginal(obs_seq, mu0_log, kappa0, alpha0, beta0)
                    for obs_seq in all_pts
                )
                return -total

            x0 = [
                np.log(2.0),            # log_alpha_shift: alpha0 = 0.5 + 2.0 = 2.5
                overall_mean,           # mu0_log
                np.log(overall_var),    # log_beta0_log
            ]
            bounds_3 = [
                (-10.0, 10.0),   # log_alpha_shift
                (-3.0, 4.0),     # mu0_log
                (-5.0, 5.0),     # log_beta0_log
            ]
            print(f"Fitting Log-NIG (kappa0 FIXED={fixed_kappa0}) to {len(all_patients)} patients "
                  f"(log-space, LOG_OFFSET={LOG_OFFSET})...")
            print(f"Log-space mean: {overall_mean:.4f} nats, variance: {overall_var:.4f} nats²")
            print()
            result = minimize(
                _neg_logmarg_fixed_kappa,
                x0,
                args=(all_patients,),
                method="L-BFGS-B",
                bounds=bounds_3,
                options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9},
            )
            log_alpha_shift, mu0, log_beta0 = result.x
            kappa0 = fixed_kappa0
            alpha0 = float(0.5 + np.exp(log_alpha_shift))
            beta0  = float(np.exp(log_beta0))
            is_constrained = log_alpha_shift < -9
            constraint_label = "constrained at 0.5" if is_constrained else "interior solution"

            print(f"Optimisation success : {result.success}")
            print(f"Message              : {result.message}")
            print(f"Neg log-marginal     : {result.fun:.6f}")
            print(f"Function evaluations : {result.nfev}")
            print()
            print("Fitted Log-NIG hyperparameters (log-space, kappa0 fixed):")
            print(f"  NIG_LOG_OFFSET  = {LOG_OFFSET}")
            print(f"  NIG_LOG_MU_0    = {mu0:.15f}   # nats")
            print(f"  NIG_LOG_KAPPA_0 = {kappa0:.15f}")
            print(f"  NIG_LOG_ALPHA_0 = {alpha0:.15f}")
            print(f"  NIG_LOG_BETA_0  = {beta0:.15f}   # nats²")
            print()
            print("Copy these five lines into adaptive_fractionation_overlap/constants.py")
            print(f"\n[Note] log_alpha_shift = {log_alpha_shift:.4f}, alpha0 = {alpha0:.4f} ({constraint_label})")
            print(f"[Note] Student-t dof at fraction 1 = 2*alpha_n = 2*(alpha0+0.5) = {2*(alpha0+0.5):.2f}")
            return mu0, kappa0, alpha0, beta0

        x0 = [
            np.log(0.1),            # log_kappa0: kappa0 = 0.1
            np.log(2.0),            # log_alpha_shift: alpha0 = 0.5 + 2.0 = 2.5
            overall_mean,           # mu0_log
            np.log(overall_var),    # log_beta0_log
        ]
        bounds = [
            (np.log(1e-3), np.log(5.0)),   # log_kappa0: kappa0 in [0.001, 5]
            (-10.0, 10.0),                  # log_alpha_shift: alpha0 = 0.5 + exp(...) in [0.5, ...]
            (-3.0, 4.0),                    # mu0_log: in nats (covers 0.05–55 cc)
            (-5.0, 5.0),                    # log_beta0_log: beta0 in nats^2
        ]
        objective = _negative_total_log_marginal_logspace

        print(f"Fitting Log-NIG hyperparameters to {len(all_patients)} patients "
              f"(log-space, LOG_OFFSET={LOG_OFFSET})...")
        print(f"Log-space mean: {overall_mean:.4f} nats, variance: {overall_var:.4f} nats²")
        print()
    else:
        all_patients = [row for row in raw_values]
        overall_mean = float(raw_values.mean())
        overall_var  = float(raw_values.var())

        x0 = [
            np.log(0.5),            # log_kappa0: kappa0 = 0.5
            np.log(0.5),            # log_alpha_shift: alpha0 = 1 + 0.5 = 1.5
            overall_mean,           # mu0
            np.log(overall_var),    # log_beta0: beta0 ≈ sample variance
        ]
        bounds = [
            (np.log(1e-3), np.log(5.0)),   # log_kappa0: kappa0 in [0.001, 5]
            (-10.0, 10.0),                  # log_alpha_shift: alpha0 = 1 + exp(...) in [1, 1+e^10]
            (0.0, 30.0),                    # mu0: prior mean in cc
            (-5.0, 10.0),                   # log_beta0: beta0 in [e^-5, e^10]
        ]
        objective = _negative_total_log_marginal

        print(f"Fitting NIG hyperparameters to {len(all_patients)} patients, "
              f"{len(FRACTION_COLS)} fractions each...")
        print(f"Cohort mean overlap: {overall_mean:.3f} cc, variance: {overall_var:.3f} cc²")
        print()

    result = minimize(
        objective,
        x0,
        args=(all_patients,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9},
    )

    log_kappa0, log_alpha_shift, mu0, log_beta0 = result.x
    kappa0 = float(np.exp(log_kappa0))
    beta0  = float(np.exp(log_beta0))

    if args.log_space:
        alpha0 = float(0.5 + np.exp(log_alpha_shift))
        is_constrained = log_alpha_shift < -9
        constraint_label = "constrained at 0.5" if is_constrained else "interior solution"
    else:
        alpha0 = float(1.0 + np.exp(log_alpha_shift))
        is_constrained = log_alpha_shift < -9
        constraint_label = "constrained at 1.0" if is_constrained else "interior solution"

    print(f"Optimisation success : {result.success}")
    print(f"Message              : {result.message}")
    print(f"Neg log-marginal     : {result.fun:.6f}")
    print(f"Function evaluations : {result.nfev}")
    print()

    if args.log_space:
        print("Fitted Log-NIG hyperparameters (log-space):")
        print(f"  NIG_LOG_OFFSET  = {LOG_OFFSET}")
        print(f"  NIG_LOG_MU_0    = {mu0:.15f}   # nats")
        print(f"  NIG_LOG_KAPPA_0 = {kappa0:.15f}")
        print(f"  NIG_LOG_ALPHA_0 = {alpha0:.15f}")
        print(f"  NIG_LOG_BETA_0  = {beta0:.15f}   # nats²")
        print()
        print("Copy these five lines into adaptive_fractionation_overlap/constants.py")
    else:
        print("Fitted NIG hyperparameters:")
        print(f"  NIG_MU_0    = {mu0:.15f}")
        print(f"  NIG_KAPPA_0 = {kappa0:.15f}")
        print(f"  NIG_ALPHA_0 = {alpha0:.15f}")
        print(f"  NIG_BETA_0  = {beta0:.15f}")
        print()
        print("Copy these four lines into adaptive_fractionation_overlap/constants.py")

    print(f"\n[Note] log_alpha_shift = {log_alpha_shift:.4f}, alpha0 = {alpha0:.4f} "
          f"({constraint_label})")
    print(f"[Note] Student-t dof at fraction 1 = 2*alpha_n = 2*(alpha0+0.5) = {2*(alpha0+0.5):.2f}")

    return mu0, kappa0, alpha0, beta0


if __name__ == "__main__":
    main()
