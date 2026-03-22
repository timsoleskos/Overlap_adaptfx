#!/usr/bin/env python3
"""Compare two benchmark JSON reports and produce a delta report artifact."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate benchmark reports."
    )
    parser.add_argument("--baseline", required=True, help="Baseline report JSON path.")
    parser.add_argument("--candidate", required=True, help="Candidate report JSON path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output comparison JSON path. Default: benchmarks/reports/comparisons/<timestamp>.json",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Optional markdown summary output path.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-9,
        help="Numerical safety epsilon for gap-closure denominator.",
    )
    parser.add_argument(
        "--tail-threshold",
        type=float,
        default=-1.0,
        help="Threshold for counting severe disadvantage cases in delta benefits.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def case_key(case: dict[str, Any], fallback_index: int) -> str:
    value = case.get("case_id", case.get("patient_number", fallback_index))
    return str(value)


def get_float(case: dict[str, Any], names: list[str]) -> float | None:
    for name in names:
        value = case.get(name)
        if value is None:
            continue
        return float(value)
    return None


def get_runtime(case: dict[str, Any]) -> float | None:
    timings = case.get("timings_sec")
    if not isinstance(timings, dict):
        return None
    value = timings.get("adaptive_fractionation")
    return float(value) if value is not None else None


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


def default_output_path(repo_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = repo_root / "benchmarks" / "reports" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"comparison_{ts}.json"


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


def write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    summary = comparison["summary"]
    def fmt(value: Any) -> str:
        return f"{float(value):.6f}" if value is not None else "n/a"

    lines = [
        "# Benchmark Comparison",
        "",
        f"- Baseline: `{comparison['baseline_report']}`",
        f"- Candidate: `{comparison['candidate_report']}`",
        f"- Compared cases: `{summary['compared_case_count']}`",
        "",
        "## Outcome Deltas",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean delta benefit | {fmt(summary['delta_benefit_stats']['mean'])} |",
        f"| Median delta benefit | {fmt(summary['delta_benefit_stats']['median'])} |",
        f"| Improved fraction | {fmt(summary['improved_fraction'])} |",
        f"| Severe disadvantage fraction | {fmt(summary['severe_disadvantage_fraction'])} |",
        "",
        "## Runtime",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Baseline total runtime (s) | {fmt(summary['runtime_sec']['baseline_total'])} |",
        f"| Candidate total runtime (s) | {fmt(summary['runtime_sec']['candidate_total'])} |",
        f"| Runtime ratio (candidate/baseline) | {fmt(summary['runtime_sec']['candidate_over_baseline'])} |",
    ]

    gap = summary.get("gap_to_ideal")
    if gap:
        lines.extend(
            [
                "",
                "## Gap-To-Ideal",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Mean old gap | {fmt(gap['old_gap_stats']['mean'])} |",
                f"| Mean new gap | {fmt(gap['new_gap_stats']['mean'])} |",
                f"| Mean normalized gap closure | {fmt(gap['gap_closure_stats']['mean'])} |",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    if not baseline_path.is_absolute():
        baseline_path = repo_root / baseline_path
    if not candidate_path.is_absolute():
        candidate_path = repo_root / candidate_path

    output_path = Path(args.output) if args.output else default_output_path(repo_root)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    baseline = load_json(baseline_path)
    candidate = load_json(candidate_path)

    baseline_cases = baseline.get("cases", [])
    candidate_cases = candidate.get("cases", [])

    baseline_map = {
        case_key(case, idx): case for idx, case in enumerate(baseline_cases)
    }
    candidate_map = {
        case_key(case, idx): case for idx, case in enumerate(candidate_cases)
    }

    common_keys = sorted(set(baseline_map) & set(candidate_map))

    per_case: list[dict[str, Any]] = []
    delta_benefits: list[float] = []
    old_gaps: list[float] = []
    new_gaps: list[float] = []
    gap_closures: list[float] = []
    runtime_baseline_case: list[float] = []
    runtime_candidate_case: list[float] = []

    improved_count = 0
    severe_disadvantage_count = 0

    for key in common_keys:
        base_case = baseline_map[key]
        cand_case = candidate_map[key]

        old_benefit = get_float(base_case, ["benefit_af", "benefit", "benefits"])
        new_benefit = get_float(cand_case, ["benefit_af", "benefit", "benefits"])

        if old_benefit is None or new_benefit is None:
            continue

        delta = new_benefit - old_benefit
        delta_benefits.append(delta)
        if delta > 0.0:
            improved_count += 1
        if delta < float(args.tail_threshold):
            severe_disadvantage_count += 1

        ideal = get_float(cand_case, ["upper_bound_benefit"])
        if ideal is None:
            ideal = get_float(base_case, ["upper_bound_benefit"])

        old_gap = None
        new_gap = None
        gap_closure = None
        if ideal is not None:
            old_gap = float(ideal - old_benefit)
            new_gap = float(ideal - new_benefit)
            denom = max(old_gap, float(args.epsilon))
            gap_closure = float((old_gap - new_gap) / denom)
            old_gaps.append(old_gap)
            new_gaps.append(new_gap)
            gap_closures.append(gap_closure)

        base_runtime = get_runtime(base_case)
        cand_runtime = get_runtime(cand_case)
        if base_runtime is not None:
            runtime_baseline_case.append(base_runtime)
        if cand_runtime is not None:
            runtime_candidate_case.append(cand_runtime)

        per_case.append(
            {
                "case_id": key,
                "patient_number": cand_case.get(
                    "patient_number", base_case.get("patient_number")
                ),
                "old_benefit": old_benefit,
                "new_benefit": new_benefit,
                "delta_benefit": delta,
                "ideal_benefit": ideal,
                "old_gap_to_ideal": old_gap,
                "new_gap_to_ideal": new_gap,
                "gap_closure_normalized": gap_closure,
                "baseline_runtime_sec": base_runtime,
                "candidate_runtime_sec": cand_runtime,
            }
        )

    baseline_total = get_float(
        baseline.get("summary", {}).get("timing_sec", {}), ["total_runtime"]
    )
    candidate_total = get_float(
        candidate.get("summary", {}).get("timing_sec", {}), ["total_runtime"]
    )

    compared_count = len(delta_benefits)
    summary: dict[str, Any] = {
        "compared_case_count": compared_count,
        "delta_benefit_stats": stats(delta_benefits),
        "improved_count": improved_count,
        "improved_fraction": (
            float(improved_count / compared_count) if compared_count else None
        ),
        "severe_disadvantage_threshold": float(args.tail_threshold),
        "severe_disadvantage_count": severe_disadvantage_count,
        "severe_disadvantage_fraction": (
            float(severe_disadvantage_count / compared_count) if compared_count else None
        ),
        "runtime_sec": {
            "baseline_total": baseline_total,
            "candidate_total": candidate_total,
            "candidate_over_baseline": (
                float(candidate_total / baseline_total)
                if baseline_total and candidate_total is not None
                else None
            ),
            "baseline_case_runtime_stats": stats(runtime_baseline_case),
            "candidate_case_runtime_stats": stats(runtime_candidate_case),
        },
    }

    if old_gaps and new_gaps and gap_closures:
        summary["gap_to_ideal"] = {
            "old_gap_stats": stats(old_gaps),
            "new_gap_stats": stats(new_gaps),
            "gap_closure_stats": stats(gap_closures),
        }

    comparison = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_report": str(baseline_path),
        "candidate_report": str(candidate_path),
        "summary": summary,
        "per_case": per_case,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(comparison), handle, indent=2)

    if args.markdown:
        md_path = Path(args.markdown)
        if not md_path.is_absolute():
            md_path = repo_root / md_path
        write_markdown(md_path, comparison)

    print(
        f"Wrote comparison report: {output_path} "
        f"(compared_cases={compared_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
