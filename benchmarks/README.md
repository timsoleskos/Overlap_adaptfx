# Benchmark Harness
This folder contains a manual (non-CI) benchmark harness for reproducible implementation-to-implementation comparisons.

The workflow is:
1. Run the benchmark on one implementation (for example `main` or an `old_af` branch) and save a JSON report artifact.
2. Run the same benchmark config on another implementation (for example `belief_state_dp`) and save another report artifact.
3. Compare the two reports with `compare_reports.py`.

`evaluation/Evaluation.ipynb` should consume these artifacts for plotting; it should not be the source of truth for comparisons.

## Files
- `run_benchmark.py`: runs replay benchmark and writes report JSON (and optional per-case CSV).
- `compare_reports.py`: compares baseline vs candidate reports and writes a delta report.
- `configs/full_cohort_replay_ds05.json`: full 58-patient replay config (`dose_step=0.5`).
- `configs/smoke_5patients_ds05.json`: quick smoke config for local checks.

## Replay Metrics (per case)
`run_benchmark.py` writes per-case:
- AF doses and AF total penalty (`adaptfx_full`).
- Standard penalty (uniform per-fraction dose = prescription/number_of_fractions).
- AF benefit: `benefit_af = total_penalty_af - standard_penalty`.
- Optional upper-bound doses/penalty/benefit from exact discrete DP.
- Per-case runtimes (AF and upper-bound DP).

## Run Examples
From repository root:

```bash
# Smoke run (first 5 patients)
python3 benchmarks/run_benchmark.py \
  --config benchmarks/configs/smoke_5patients_ds05.json \
  --label smoke_old \
  --output benchmarks/reports/smoke_old.json \
  --cases-csv benchmarks/reports/smoke_old_cases.csv

# Full cohort run
python3 benchmarks/run_benchmark.py \
  --config benchmarks/configs/full_cohort_replay_ds05.json \
  --label full_old \
  --output benchmarks/reports/full_old.json \
  --cases-csv benchmarks/reports/full_old_cases.csv
```

Then switch branch/implementation, rerun with same config and different label/output:

```bash
python3 benchmarks/run_benchmark.py \
  --config benchmarks/configs/full_cohort_replay_ds05.json \
  --label full_new \
  --output benchmarks/reports/full_new.json \
  --cases-csv benchmarks/reports/full_new_cases.csv
```

## Compare Reports
```bash
python3 benchmarks/compare_reports.py \
  --baseline benchmarks/reports/full_old.json \
  --candidate benchmarks/reports/full_new.json \
  --output benchmarks/reports/comparisons/full_old_vs_new.json \
  --markdown benchmarks/reports/comparisons/full_old_vs_new.md
```

The comparison report includes:
- `delta_benefit = new_benefit - old_benefit` per case.
- Improvement fraction and severe-disadvantage fraction (threshold configurable).
- Gap-to-ideal metrics when upper-bound benefit is present:
  - `G_old = B_ideal - B_old`
  - `G_new = B_ideal - B_new`
  - `R = (G_old - G_new)/max(G_old, epsilon)`
- Runtime comparison summary.

## Notes
- This harness currently runs replay cases from `evaluation/ACTION_patients_overlap_only.xlsx`.
- The upper-bound reference is exact discrete DP on the dose grid (`step` in config).
- For fast local checks, use smoke config or `--patient-limit`.
