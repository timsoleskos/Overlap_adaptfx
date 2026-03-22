#!/usr/bin/env python3
"""Manual benchmark runner for adaptive fractionation reports.

This script is intentionally manual (not CI). It produces a structured JSON
artifact with per-case outcomes and timings so reports can be compared across
implementations (for example, across git branches).
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import adaptive_fractionation_overlap as afx


REQUIRED_COLUMNS = {
    "P_NUMBER",
    "FRAC_NUMBER",
    "TOTAL_OVERLAP (cc)",
    "PRESCRIPTION_DOSE",
}


@dataclass
class PatientCase:
    """Replay-case definition extracted from the cohort file."""

    case_id: str
    patient_number: int | float | str
    overlaps: list[float]
    prescription_dose: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manual replay benchmark and write report artifacts."
    )
    parser.add_argument(
        "--config",
        default="benchmarks/configs/full_cohort_replay_ds05.json",
        help="Path to JSON benchmark config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output report JSON path. Default: benchmarks/reports/<label>_<timestamp>.json",
    )
    parser.add_argument(
        "--cases-csv",
        default=None,
        help="Optional per-case CSV output path.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional run label stored in report metadata.",
    )
    parser.add_argument(
        "--patient-limit",
        type=int,
        default=None,
        help="Optional limit on number of replay cases (for smoke runs).",
    )
    parser.add_argument(
        "--disable-upper-bound",
        action="store_true",
        help="Disable upper-bound DP calculation for this run.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress updates as each patient case completes.",
    )
    parser.add_argument(
        "--use-nig",
        action="store_true",
        help="Use Stage B NIG belief model (use_nig=True) instead of Stage A Gaussian.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_prescription_total_dose(raw_value: Any) -> float:
    """Parse notebook-style prescription strings such as '5 x 6.6 Gy @80%'."""
    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        return float(raw_value)

    if isinstance(raw_value, str):
        match = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)", raw_value)
        if match:
            n_fractions = float(match.group(1))
            per_fraction = float(match.group(2))
            return n_fractions * per_fraction

    raise ValueError(f"Could not parse prescription dose from value={raw_value!r}")


def normalize_patient_number(value: Any) -> int | float | str:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return int(value) if float(value).is_integer() else float(value)
    return str(value)


def extract_replay_cases(df: pd.DataFrame, number_of_fractions: int) -> list[PatientCase]:
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")

    data = df.copy()
    data["P_NUMBER"] = data["P_NUMBER"].ffill()

    cases: list[PatientCase] = []
    for patient_number, patient_df in data.groupby("P_NUMBER", sort=False):
        patient_df = patient_df.sort_values("FRAC_NUMBER")
        overlaps = patient_df["TOTAL_OVERLAP (cc)"].astype(float).tolist()
        expected_len = number_of_fractions + 1  # planning scan + per-fraction observed overlaps
        if len(overlaps) != expected_len:
            raise ValueError(
                f"Patient {patient_number!r} has {len(overlaps)} overlaps, "
                f"expected {expected_len}."
            )

        non_null_rx = patient_df["PRESCRIPTION_DOSE"].dropna()
        if non_null_rx.empty:
            raise ValueError(f"Patient {patient_number!r} has no prescription dose.")
        prescription_dose = parse_prescription_total_dose(non_null_rx.iloc[0])
        normalized_number = normalize_patient_number(patient_number)

        cases.append(
            PatientCase(
                case_id=str(normalized_number),
                patient_number=normalized_number,
                overlaps=overlaps,
                prescription_dose=prescription_dose,
            )
        )

    return cases


def select_max_dose(mean_dose: float, policy_cfg: dict[str, Any]) -> float:
    default_max = float(policy_cfg.get("default", 10.0))
    max_gt_8_lte_9 = float(policy_cfg.get("mean_gt_8_lte_9", 11.0))
    max_gt_9 = float(policy_cfg.get("mean_gt_9", 12.0))

    if mean_dose > 9.0:
        return max_gt_9
    if mean_dose > 8.0:
        return max_gt_8_lte_9
    return default_max


def calc_standard_penalty(overlaps: list[float], mean_dose: float, min_dose: float) -> float:
    overlap_treatment = np.asarray(overlaps[1:], dtype=float)  # skip planning scan
    penalty_values = afx.penalty_calc_single(
        mean_dose,
        min_dose,
        overlap_treatment,
        intercept=afx.INTERCEPT,
        slope=afx.SLOPE,
    )
    return -float(np.sum(np.asarray(penalty_values, dtype=float)))


def calc_upper_bound_treatment_discrete_dp(
    overlaps: list[float],
    prescription_dose: float,
    number_of_fractions: int,
    min_dose: float,
    max_dose: float,
    step: float,
    intercept: float,
    slope: float,
) -> tuple[np.ndarray | None, float | None]:
    """Exact discrete optimization in dose-step tick space."""
    min_ticks = int(round(min_dose / step))
    max_ticks = int(round(max_dose / step))
    rx_ticks = int(round(prescription_dose / step))

    if rx_ticks < number_of_fractions * min_ticks or rx_ticks > number_of_fractions * max_ticks:
        return None, None

    overlap_treatment = np.asarray(overlaps[1:], dtype=float)
    steepness = np.abs(intercept + slope * overlap_treatment)

    # dp[sum_ticks] -> (min_cost, path_ticks)
    dp: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}

    for frac_idx in range(number_of_fractions):
        overlap_value = float(overlap_treatment[frac_idx])
        steepness_value = float(steepness[frac_idx])
        new_dp: dict[int, tuple[float, list[int]]] = {}

        for sum_ticks, (cost_so_far, path_so_far) in dp.items():
            for dose_ticks in range(min_ticks, max_ticks + 1):
                new_sum_ticks = sum_ticks + dose_ticks
                if new_sum_ticks > rx_ticks:
                    continue

                dose_value = dose_ticks * step
                dose_above_min = dose_value - min_dose
                incremental_penalty = (
                    dose_above_min * overlap_value
                    + (dose_above_min**2) * steepness_value / 2.0
                )
                new_cost = cost_so_far + incremental_penalty

                existing = new_dp.get(new_sum_ticks)
                if existing is None or new_cost < existing[0]:
                    new_dp[new_sum_ticks] = (new_cost, path_so_far + [dose_ticks])

        dp = new_dp
        if not dp:
            return None, None

    best = dp.get(rx_ticks)
    if best is None:
        return None, None

    best_cost, best_path_ticks = best
    best_doses = np.asarray(best_path_ticks, dtype=float) * step
    return best_doses, float(best_cost)


def run_git_command(repo_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def infer_git_info(repo_root: Path) -> dict[str, Any]:
    commit = run_git_command(repo_root, ["rev-parse", "HEAD"])
    short_commit = run_git_command(repo_root, ["rev-parse", "--short", "HEAD"])
    branch = run_git_command(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    status = run_git_command(repo_root, ["status", "--porcelain"])
    dirty = bool(status) if status is not None else None

    return {
        "commit": commit,
        "short_commit": short_commit,
        "branch": branch,
        "dirty": dirty,
    }


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
        }

    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def default_output_path(repo_root: Path, label: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports_dir = repo_root / "benchmarks" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir / f"{label}_{ts}.json"


def write_cases_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        timings = case.get("timings_sec", {})
        rows.append(
            {
                "case_id": case.get("case_id"),
                "patient_number": case.get("patient_number"),
                "prescription_dose": case.get("prescription_dose"),
                "mean_dose": case.get("mean_dose"),
                "max_dose": case.get("max_dose"),
                "benefit_af": case.get("benefit_af"),
                "standard_penalty": case.get("standard_penalty"),
                "total_penalty_af": case.get("total_penalty_af"),
                "upper_bound_benefit": case.get("upper_bound_benefit"),
                "upper_bound_penalty": case.get("upper_bound_penalty"),
                "af_runtime_sec": timings.get("adaptive_fractionation"),
                "upper_bound_runtime_sec": timings.get("upper_bound_discrete_dp"),
                "af_doses": json.dumps(case.get("af_doses")),
                "upper_bound_doses": json.dumps(case.get("upper_bound_doses")),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = load_json(config_path)

    label = args.label or str(config.get("name", "benchmark_run"))
    output_path = Path(args.output) if args.output else default_output_path(repo_root, label)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path

    n_fractions = int(config.get("number_of_fractions", 5))
    min_dose = float(config.get("min_dose", 6.0))
    dose_step = float(config.get("dose_step", 0.5))
    max_dose_policy = dict(config.get("max_dose_policy", {}))
    upper_cfg = dict(config.get("upper_bound", {}))
    upper_enabled = bool(upper_cfg.get("enabled", True)) and not args.disable_upper_bound
    upper_step = float(upper_cfg.get("step", dose_step))
    tail_threshold = float(config.get("metrics", {}).get("tail_disadvantage_threshold", -1.0))

    patient_limit = args.patient_limit
    if patient_limit is None and "patient_limit" in config and config["patient_limit"] is not None:
        patient_limit = int(config["patient_limit"])

    df = pd.read_excel(dataset_path)
    replay_cases = extract_replay_cases(df, n_fractions)
    if patient_limit is not None:
        replay_cases = replay_cases[:patient_limit]
    total_cases = len(replay_cases)

    run_started = time.perf_counter()
    case_reports: list[dict[str, Any]] = []
    af_case_times: list[float] = []
    ub_case_times: list[float] = []

    if args.progress:
        print(
            f"Starting benchmark: cases={total_cases}, "
            f"upper_bound={'enabled' if upper_enabled else 'disabled'}",
            flush=True,
        )

    for idx, case in enumerate(replay_cases, start=1):
        case_started = time.perf_counter()
        mean_dose = case.prescription_dose / float(n_fractions)
        max_dose = select_max_dose(mean_dose, max_dose_policy)

        af_start = time.perf_counter()
        af_doses, af_accum_doses, af_total_penalty = afx.adaptfx_full(
            volumes=case.overlaps,
            number_of_fractions=n_fractions,
            min_dose=min_dose,
            max_dose=max_dose,
            mean_dose=mean_dose,
            dose_steps=dose_step,
            use_nig=args.use_nig,
        )
        af_elapsed = time.perf_counter() - af_start
        af_case_times.append(float(af_elapsed))

        standard_penalty = calc_standard_penalty(case.overlaps, mean_dose, min_dose)
        benefit_af = float(af_total_penalty) - float(standard_penalty)

        upper_doses_list: list[float] | None = None
        upper_penalty: float | None = None
        upper_benefit: float | None = None
        upper_elapsed: float | None = None

        if upper_enabled:
            upper_start = time.perf_counter()
            upper_doses, _ = calc_upper_bound_treatment_discrete_dp(
                overlaps=case.overlaps,
                prescription_dose=case.prescription_dose,
                number_of_fractions=n_fractions,
                min_dose=min_dose,
                max_dose=max_dose,
                step=upper_step,
                intercept=float(afx.INTERCEPT),
                slope=float(afx.SLOPE),
            )
            upper_elapsed = time.perf_counter() - upper_start
            ub_case_times.append(float(upper_elapsed))

            if upper_doses is not None:
                upper_doses_list = [float(x) for x in upper_doses.tolist()]
                upper_penalty = float(
                    np.sum(
                        afx.penalty_calc_single(
                            np.asarray(upper_doses, dtype=float),
                            min_dose,
                            np.asarray(case.overlaps[1:], dtype=float),
                            intercept=afx.INTERCEPT,
                            slope=afx.SLOPE,
                        )
                    )
                )
                upper_benefit = -upper_penalty - standard_penalty

        case_reports.append(
            {
                "case_id": case.case_id,
                "patient_number": case.patient_number,
                "overlaps": [float(x) for x in case.overlaps],
                "prescription_dose": float(case.prescription_dose),
                "mean_dose": float(mean_dose),
                "max_dose": float(max_dose),
                "af_doses": [float(x) for x in np.asarray(af_doses, dtype=float).tolist()],
                "af_accumulated_doses": [
                    float(x) for x in np.asarray(af_accum_doses, dtype=float).tolist()
                ],
                "total_penalty_af": float(af_total_penalty),
                "standard_penalty": float(standard_penalty),
                "benefit_af": float(benefit_af),
                "upper_bound_doses": upper_doses_list,
                "upper_bound_penalty": upper_penalty,
                "upper_bound_benefit": upper_benefit,
                "timings_sec": {
                    "adaptive_fractionation": float(af_elapsed),
                    "upper_bound_discrete_dp": upper_elapsed,
                },
            }
        )

        if args.progress:
            elapsed = time.perf_counter() - run_started
            avg_per_case = elapsed / float(idx)
            eta = avg_per_case * float(total_cases - idx)
            finish_local = (datetime.now().astimezone() + timedelta(seconds=eta)).strftime(
                "%H:%M:%S"
            )
            ub_display = f"{upper_elapsed:.3f}s" if upper_elapsed is not None else "n/a"
            case_elapsed = time.perf_counter() - case_started
            print(
                f"[{idx}/{total_cases}] patient={case.patient_number} "
                f"af={af_elapsed:.3f}s ub={ub_display} "
                f"case={case_elapsed:.3f}s elapsed={elapsed:.1f}s eta={eta:.1f}s "
                f"finish_local={finish_local}",
                flush=True,
            )

    run_elapsed = time.perf_counter() - run_started

    af_benefits = [float(case["benefit_af"]) for case in case_reports]
    ub_benefits = [
        float(case["upper_bound_benefit"])
        for case in case_reports
        if case.get("upper_bound_benefit") is not None
    ]
    tail_count = sum(1 for value in af_benefits if value < tail_threshold)

    summary = {
        "case_count": len(case_reports),
        "benefit_af_stats": stats(af_benefits),
        "upper_bound_benefit_stats": stats(ub_benefits),
        "tail_disadvantage_threshold": float(tail_threshold),
        "tail_disadvantage_count": int(tail_count),
        "tail_disadvantage_fraction": (
            float(tail_count / len(case_reports)) if case_reports else None
        ),
        "timing_sec": {
            "total_runtime": float(run_elapsed),
            "adaptive_fractionation_total": float(sum(af_case_times)),
            "upper_bound_total": float(sum(ub_case_times)) if ub_case_times else None,
            "adaptive_fractionation_stats": stats(af_case_times),
            "upper_bound_stats": stats(ub_case_times),
        },
    }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "use_nig": args.use_nig,
        "git": infer_git_info(repo_root),
        "system": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "paths": {
            "config": str(config_path.relative_to(repo_root)),
            "dataset": str(dataset_path.relative_to(repo_root)),
        },
        "config": config,
        "summary": summary,
        "cases": case_reports,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(report), handle, indent=2)

    if args.cases_csv:
        csv_path = Path(args.cases_csv)
        if not csv_path.is_absolute():
            csv_path = repo_root / csv_path
        write_cases_csv(csv_path, case_reports)

    print(
        f"Wrote benchmark report: {output_path} "
        f"(cases={len(case_reports)}, total_runtime_sec={run_elapsed:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
