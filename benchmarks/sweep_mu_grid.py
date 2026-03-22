#!/usr/bin/env python3
"""Sweep over N_mu values to measure sensitivity of AF improvement to mu-grid resolution.

For each target N_mu, the 5 existing mu-grid segments are scaled proportionally, keeping
the non-uniform density shape (fine at low mu, coarser in the tail).  All 58 patients
in the ACTION cohort are evaluated for each configuration.

Usage:
    python benchmarks/sweep_mu_grid.py [--progress] [--output PATH] [--config PATH]

Output:
    CSV with columns: n_mu_target, n_mu_actual, patient_number, benefit, runtime_sec
    Printed summary table at the end.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

N_MU_VALUES = [50, 100, 150, 200, 280, 350, 420, 500]

# Current 5-segment mu grid: (start, stop, n_points)
_BASE_SEGMENTS = [
    (0.0,  1.0,  70),
    (1.05, 4.0,  73),
    (4.1,  10.0, 79),
    (10.2, 16.0, 28),
    (16.5, 30.0, 30),
]
_BASE_N_MU = 280  # sum of base segment counts


# ---------------------------------------------------------------------------
# Grid construction and monkey-patching
# ---------------------------------------------------------------------------

def compute_mu_grid(n_mu_target: int) -> np.ndarray:
    """Scale each segment count proportionally and return the concatenated grid.

    Uses standard rounding (not banker's rounding) so totals stay close to n_mu_target.
    Each segment is clamped to a minimum of 2 points.
    """
    scale = n_mu_target / _BASE_N_MU
    segments = []
    for start, stop, base_count in _BASE_SEGMENTS:
        count = max(2, int(base_count * scale + 0.5))  # standard round
        segments.append(np.linspace(start, stop, count))
    return np.unique(np.concatenate(segments))


def patch_mu_grid(grid: np.ndarray) -> None:
    """Replace _MU_GRID (and derived _P_BELIEF) in every module that uses it."""
    import adaptive_fractionation_overlap.belief_model as belief_model
    import adaptive_fractionation_overlap.core_adaptfx as core_adaptfx

    belief_model._MU_GRID = grid
    belief_model._P_BELIEF = belief_model._precompute_branch_probabilities()
    # core_adaptfx imports _MU_GRID by value (from .belief_model import _MU_GRID),
    # so its local reference must also be updated so N_mu = len(_MU_GRID) is correct.
    core_adaptfx._MU_GRID = grid


# ---------------------------------------------------------------------------
# Per-patient benchmark helpers (replicate minimal logic from run_benchmark.py)
# ---------------------------------------------------------------------------

def _select_max_dose(mean_dose: float, policy: dict) -> float:
    if mean_dose > 9.0:
        return float(policy.get("mean_gt_9", 12.0))
    if mean_dose > 8.0:
        return float(policy.get("mean_gt_8_lte_9", 11.0))
    return float(policy.get("default", 10.0))


def _standard_penalty(overlaps: list[float], mean_dose: float, min_dose: float) -> float:
    import adaptive_fractionation_overlap as afx
    vals = afx.penalty_calc_single(
        mean_dose, min_dose,
        np.asarray(overlaps[1:], dtype=float),
        intercept=afx.INTERCEPT,
        slope=afx.SLOPE,
    )
    return -float(np.sum(np.asarray(vals, dtype=float)))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmarks/configs/full_cohort_replay_ds05.json")
    parser.add_argument("--output", default="benchmarks/reports/mu_grid_sweep.csv")
    parser.add_argument("--progress", action="store_true", help="Print per-patient and per-config progress.")
    parser.add_argument(
        "--configs", type=int, nargs="+", default=None,
        metavar="N_MU",
        help="Subset of N_mu values to run (e.g. --configs 50 280). Defaults to all.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    # Load benchmark config
    config_path = repo_root / args.config
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    n_fractions = int(config.get("number_of_fractions", 5))
    min_dose = float(config.get("min_dose", 6.0))
    dose_step = float(config.get("dose_step", 0.5))
    max_dose_policy = dict(config.get("max_dose_policy", {}))

    # Load patient data (import here so module is patched before any afx usage)
    sys.path.insert(0, str(repo_root))
    from benchmarks.run_benchmark import extract_replay_cases
    dataset_path = repo_root / config["dataset_path"]
    df = pd.read_excel(dataset_path)
    replay_cases = extract_replay_cases(df, n_fractions)
    n_patients = len(replay_cases)

    import adaptive_fractionation_overlap as afx

    n_mu_values = args.configs if args.configs is not None else N_MU_VALUES
    invalid = [v for v in n_mu_values if v not in N_MU_VALUES]
    if invalid:
        print(f"Warning: {invalid} not in the predefined N_MU_VALUES list; they will still be run.")

    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    sweep_start = time.perf_counter()
    n_configs = len(n_mu_values)

    for cfg_idx, n_mu_target in enumerate(n_mu_values, start=1):
        grid = compute_mu_grid(n_mu_target)
        n_mu_actual = len(grid)

        # High-level progress header
        elapsed_so_far = time.perf_counter() - sweep_start
        if args.progress:
            if cfg_idx > 1:
                avg_cfg_time = elapsed_so_far / (cfg_idx - 1)
                remaining_configs = n_configs - cfg_idx + 1
                eta_sec = avg_cfg_time * remaining_configs
                finish_str = (
                    datetime.now().astimezone() + timedelta(seconds=eta_sec)
                ).strftime("%H:%M:%S")
                print(
                    f"\n{'='*60}\n"
                    f"[Config {cfg_idx}/{n_configs}] N_mu={n_mu_target} (actual={n_mu_actual}) | "
                    f"elapsed={elapsed_so_far/60:.1f}min  ETA={eta_sec/60:.1f}min  "
                    f"finish_local={finish_str}  configs_remaining={n_configs - cfg_idx}"
                    f"\n{'='*60}",
                    flush=True,
                )
            else:
                print(
                    f"\n{'='*60}\n"
                    f"[Config {cfg_idx}/{n_configs}] N_mu={n_mu_target} (actual={n_mu_actual}) | "
                    f"starting sweep"
                    f"\n{'='*60}",
                    flush=True,
                )

        patch_mu_grid(grid)
        cfg_start = time.perf_counter()

        for pat_idx, case in enumerate(replay_cases, start=1):
            mean_dose = case.prescription_dose / float(n_fractions)
            max_dose = _select_max_dose(mean_dose, max_dose_policy)

            t0 = time.perf_counter()
            _, _, af_total_penalty = afx.adaptfx_full(
                volumes=case.overlaps,
                number_of_fractions=n_fractions,
                min_dose=min_dose,
                max_dose=max_dose,
                mean_dose=mean_dose,
                dose_steps=dose_step,
            )
            pat_elapsed = time.perf_counter() - t0

            std_pen = _standard_penalty(case.overlaps, mean_dose, min_dose)
            benefit = float(af_total_penalty) - std_pen

            all_rows.append({
                "n_mu_target": n_mu_target,
                "n_mu_actual": n_mu_actual,
                "patient_number": case.patient_number,
                "benefit": benefit,
                "runtime_sec": pat_elapsed,
            })

            if args.progress:
                cfg_elapsed = time.perf_counter() - cfg_start
                avg_pat = cfg_elapsed / pat_idx
                pat_eta = avg_pat * (n_patients - pat_idx)
                print(
                    f"  [{pat_idx}/{n_patients}] patient={case.patient_number}  "
                    f"benefit={benefit:+.4f} ccGy  t={pat_elapsed:.2f}s  "
                    f"cfg_elapsed={cfg_elapsed:.0f}s  pat_eta={pat_eta:.0f}s",
                    flush=True,
                )

    # Save results
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(output_path, index=False)

    total_elapsed = time.perf_counter() - sweep_start
    print(f"\n{'='*60}")
    print(f"Sweep complete: {n_configs} configs × {n_patients} patients in {total_elapsed/60:.1f} min")
    print(f"Results: {output_path}")
    print(f"\n{'N_mu_tgt':>10} {'N_actual':>10} {'Mean benefit':>14} {'Median':>10} {'N_improved':>12} {'t/patient':>12}")
    for n_mu_target in n_mu_values:
        sub = df_out[df_out["n_mu_target"] == n_mu_target]
        print(
            f"{n_mu_target:>10} {sub['n_mu_actual'].iloc[0]:>10} "
            f"{sub['benefit'].mean():>+14.4f} {sub['benefit'].median():>+10.4f} "
            f"{(sub['benefit'] > 0).sum():>12} {sub['runtime_sec'].mean():>11.2f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
