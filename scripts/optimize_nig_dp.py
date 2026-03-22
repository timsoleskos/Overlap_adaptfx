#!/usr/bin/env python3
"""Grid search over (alpha0, beta0) to maximise mean DP benefit vs Stage A.

Fixes kappa0=0.001 (suppresses NIG cross-term) and mu0 at the value from
re-fitting with kappa0 fixed (0.666 nats). Runs all 58 patients for each
(alpha0, beta0) combination and reports mean delta vs Stage A.

Results are saved incrementally to scripts/nig_dp_grid_results.csv.

Usage
-----
    python scripts/optimize_nig_dp.py [--stage-a-report PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import adaptive_fractionation_overlap as afx
import adaptive_fractionation_overlap.constants as C
import adaptive_fractionation_overlap.belief_model as BM

# ---------------------------------------------------------------------------
# Config (mirrors full_cohort_replay_ds05.json)
# ---------------------------------------------------------------------------
DATASET_PATH    = REPO / "evaluation" / "ACTION_patients_overlap_only.xlsx"
N_FRACTIONS     = 5
MIN_DOSE        = 6.0
DOSE_STEP       = 0.5
MAX_DOSE_POLICY = {"default": 10.0, "mean_gt_8_lte_9": 11.0, "mean_gt_9": 12.0}

FIXED_KAPPA0 = 0.001
FIXED_MU0    = 0.666160089085257   # refitted with kappa0 fixed; barely matters since cross-term suppressed


def select_max_dose(mean_dose: float) -> float:
    if mean_dose > 9.0:
        return MAX_DOSE_POLICY["mean_gt_9"]
    if mean_dose > 8.0:
        return MAX_DOSE_POLICY["mean_gt_8_lte_9"]
    return MAX_DOSE_POLICY["default"]


def load_cases() -> list[dict]:
    df = pd.read_excel(DATASET_PATH)
    df["P_NUMBER"] = df["P_NUMBER"].ffill()
    cases = []
    for patient_number, patient_df in df.groupby("P_NUMBER", sort=False):
        patient_df = patient_df.sort_values("FRAC_NUMBER")
        overlaps = patient_df["TOTAL_OVERLAP (cc)"].astype(float).tolist()
        import re as _re
        rx = patient_df["PRESCRIPTION_DOSE"].dropna().iloc[0]
        if isinstance(rx, str):
            m = _re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)", rx)
            prescription_dose = float(m.group(1)) * float(m.group(2)) if m else float(rx)
        else:
            prescription_dose = float(rx)
        cases.append({"patient_number": patient_number, "overlaps": overlaps,
                      "prescription_dose": prescription_dose})
    return cases


def run_nig_benefit(case: dict) -> float:
    """Run adaptfx_full with use_nig=True for one patient and return benefit vs flat."""
    overlaps      = case["overlaps"]
    mean_dose     = case["prescription_dose"] / N_FRACTIONS
    max_dose      = select_max_dose(mean_dose)
    _, _, total_penalty = afx.adaptfx_full(
        volumes=overlaps,
        number_of_fractions=N_FRACTIONS,
        min_dose=MIN_DOSE,
        max_dose=max_dose,
        mean_dose=mean_dose,
        dose_steps=DOSE_STEP,
        use_nig=True,
    )
    overlap_treatment = np.asarray(overlaps[1:], dtype=float)
    flat_penalties = afx.penalty_calc_single(
        mean_dose, MIN_DOSE, overlap_treatment,
        intercept=afx.INTERCEPT, slope=afx.SLOPE,
    )
    standard_penalty = -float(np.sum(np.asarray(flat_penalties, dtype=float)))
    return float(total_penalty) - standard_penalty


def eval_params(alpha0: float, beta0: float, cases: list[dict],
                stage_a_benefits: list[float]) -> tuple[float, float, list[float]]:
    """Evaluate (alpha0, beta0) on all patients. Returns (mean_delta_vs_stageA, mean_nig_benefit, per_patient_deltas)."""
    C.NIG_LOG_ALPHA_0 = alpha0
    C.NIG_LOG_BETA_0  = beta0
    BM._P_BELIEF_NIG_CACHE.clear()

    nig_benefits = [run_nig_benefit(c) for c in cases]
    deltas = [n - s for n, s in zip(nig_benefits, stage_a_benefits)]
    return float(np.mean(deltas)), float(np.mean(nig_benefits)), deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-a-report",
                        default=str(REPO / "benchmarks" / "reports" / "stage_a_final.json"))
    parser.add_argument("--output",
                        default=str(REPO / "scripts" / "nig_dp_grid_results.csv"))
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load Stage A baseline benefits
    # -----------------------------------------------------------------------
    with open(args.stage_a_report, encoding="utf-8") as f:
        stage_a_report = json.load(f)
    # Keyed by patient_number (normalised to int) → benefit_af
    stage_a_map = {int(c["patient_number"]): c["benefit_af"]
                   for c in stage_a_report["cases"]}

    # -----------------------------------------------------------------------
    # Load patient cases and align with Stage A report order
    # -----------------------------------------------------------------------
    all_cases = load_cases()
    # Keep only patients present in Stage A report
    # Normalise patient_number to int for lookup
    for c in all_cases:
        c["patient_number"] = int(float(c["patient_number"]))
    cases = [c for c in all_cases if c["patient_number"] in stage_a_map]
    stage_a_benefits = [stage_a_map[c["patient_number"]] for c in cases]
    print(f"Loaded {len(cases)} patients (matched to Stage A report).")
    print(f"Stage A mean benefit: {np.mean(stage_a_benefits):.4f} ccGy")
    print()

    # -----------------------------------------------------------------------
    # Fix kappa0 and mu0
    # -----------------------------------------------------------------------
    C.NIG_LOG_KAPPA_0 = FIXED_KAPPA0
    C.NIG_LOG_MU_0    = FIXED_MU0

    # -----------------------------------------------------------------------
    # Grid definition
    # -----------------------------------------------------------------------
    # alpha0 controls dof (= 2*alpha0 + n) and how fast scale shrinks with data.
    # beta0 controls the prior scale in log-space.
    # Parameterized by (alpha0, prior_scale_1) where:
    #   prior_scale_1 = sqrt(2 * beta0 / (alpha0 + 0.5))   [predictive scale at fraction 1, m2=0]
    # So beta0 = prior_scale_1^2 * (alpha0 + 0.5) / 2.
    # This covers the space more uniformly than a (alpha0, beta0) grid.
    alpha0_grid      = [0.55, 0.75, 1.0, 1.3, 1.527, 2.0, 2.74, 4.0, 6.0, 9.0]
    prior_scale_grid = [0.15, 0.25, 0.35, 0.492, 0.65, 0.85, 1.1, 1.4]  # nats

    combos = []
    for a0 in alpha0_grid:
        for s1 in prior_scale_grid:
            b0 = s1 ** 2 * (a0 + 0.5) / 2.0
            combos.append((a0, b0, s1))

    total = len(combos)
    print(f"Grid: {len(alpha0_grid)} alpha0 values x {len(prior_scale_grid)} prior_scale values = {total} combos")
    print(f"Estimated time: ~{total * len(cases) * 7.6 / 3600:.1f} hours")
    print()
    print(f"{'#':>4}  {'alpha0':>8}  {'beta0':>8}  {'scale_1':>8}  {'mean_delta':>12}  {'best_so_far':>12}  {'time_s':>8}")
    print("-" * 75)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    best_delta = -np.inf
    best_combo = None

    with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["alpha0", "beta0", "prior_scale_1", "mean_delta_vs_stageA",
                             "mean_nig_benefit"] +
                            [f"delta_p{c['patient_number']}" for c in cases])

        for i, (a0, b0, s1) in enumerate(combos, start=1):
            t0 = time.perf_counter()
            mean_delta, mean_benefit, deltas = eval_params(a0, b0, cases, stage_a_benefits)
            elapsed = time.perf_counter() - t0

            if mean_delta > best_delta:
                best_delta = mean_delta
                best_combo = (a0, b0, s1)

            writer.writerow([f"{a0:.6f}", f"{b0:.6f}", f"{s1:.6f}",
                             f"{mean_delta:.6f}", f"{mean_benefit:.6f}"] +
                            [f"{d:.6f}" for d in deltas])
            csvfile.flush()

            print(f"{i:>4}  {a0:>8.4f}  {b0:>8.4f}  {s1:>8.4f}  {mean_delta:>+12.5f}  "
                  f"{best_delta:>+12.5f}  {elapsed:>8.1f}s")

    print()
    if best_combo:
        a0, b0, s1 = best_combo
        print(f"Best combination: alpha0={a0:.4f}, beta0={b0:.6f}, prior_scale={s1:.4f}")
        print(f"Best mean delta vs Stage A: {best_delta:+.5f} ccGy")
        print()
        print("Update constants.py with:")
        print(f"  NIG_LOG_KAPPA_0 = {FIXED_KAPPA0}")
        print(f"  NIG_LOG_MU_0    = {FIXED_MU0:.15f}")
        print(f"  NIG_LOG_ALPHA_0 = {a0:.15f}")
        print(f"  NIG_LOG_BETA_0  = {b0:.15f}")


if __name__ == "__main__":
    main()
