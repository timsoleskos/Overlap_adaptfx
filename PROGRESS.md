# Progress: Review slow-marked tests for CI promotion

## Goal
Determine which `@pytest.mark.slow` tests are now fast enough (and memory-safe) to run
in GitHub Actions CI (≤7 GB RAM, 15-min timeout) after the belief-state DP algorithm was accelerated.

## Slow-marked tests — timing results (local, Win32 Python 3.12)

| # | Test | Location | Local time | CI estimate | Verdict |
|---|---|---|---|---|---|
| 1 | `TestAdaptfxFull::test_adaptfx_full_evaluation_style` | line 284 | **157s** | ~5–8 min | ❌ too slow |
| 2 | `TestPrecomputePlan::test_precompute_plan_basic` | line 384 | **>12 min (killed)** | ~1+ hour | ❌ way too slow |
| 3 | `TestPrecomputePlan::test_precompute_plan_different_fractions` | line 452 | (not timed, similar mechanism) | ~1+ hour | ❌ way too slow |
| 4 | `TestCoreAdaptfxIntegration::test_adaptfx_full_vs_core_consistency` | line 485 | **104s** | ~3–5 min | ❌ too slow |
| 5 | `TestCoreAdaptfxIntegration::test_evaluation_workflow_reproduction` | line 531 | **158s** | ~5–8 min | ❌ too slow |
| 6 | `TestCoreAdaptfxPerformance::test_adaptfx_full_multiple_patients` | line 753 | (not timed) | — | ❌ will FAIL (see below) |

CI timeout is 15 min total. Non-slow tests already use ~3 min. No headroom to add any of these.

## Root cause analysis

### precompute_plan tests (tests #2, #3) — fundamentally broken
`precompute_plan` calls `adaptive_fractionation_core` (full 5D DP rebuild) **once per 0.1 cc volume bin**
in a while loop. The test asserts 152 bins → 152 × ~30s ≈ **75 minutes**. These cannot be promoted
without refactoring `precompute_plan` to build the DP table once and reuse it.

### adaptfx_full tests (tests #1, #4, #5) — still too slow for CI
Each `adaptfx_full` call (5 fractions, 1 patient) = ~34s locally. 3 patients = ~103s.
CI runners are slower → estimate 5–8 min per 3-patient test. On a 15-min timeout
with 3-min baseline, adding even one of these risks timeout.

### Performance test (test #6) — will FAIL regardless of slow mark
`TestCoreAdaptfxPerformance::test_adaptfx_full_multiple_patients` asserts `elapsed < 10.0s`
for 3 patients. Actual time ≈ 103s. **The threshold is wrong and needs updating.**
Either update the threshold (e.g. 300s) or keep test slow-marked as a regression guard.

## Decision

**All 5 slow locations should remain `@pytest.mark.slow`.**

The algorithm speedup (79s → 34.5s per patient) did not make any of these tests CI-viable:
- `precompute_plan` tests: O(N_bins × DP_cost) — DP cost is still ~30s per call
- `adaptfx_full` tests: 3 patients × ~34s = ~103s is too risky on a 15-min CI budget
- Performance test: threshold needs fixing (separate issue)

## Actions needed

1. **Fix `TestCoreAdaptfxPerformance` time threshold** — update `elapsed < 10.0` to a realistic
   value (e.g. `300.0s` for 3 patients × ~100s). This is a bug independent of the slow/fast question.
2. **No slow marks removed** — all remain as-is.

## Status: COMPLETE (analysis done, action item #1 still pending)

---

# Progress: Grid ceiling bugs found in discretisation review (2026-03-19)

## Findings

Two issues found by computing per-patient statistics directly from
`evaluation/ACTION_patients_overlap_only.xlsx` (58-patient ACTION cohort).

### Finding 1 — σ grid ceiling clips the most variable patient

| Stat | Value |
|---|---|
| σ grid max | 3.5 cc |
| Comment in `belief_model.py` | "observed max σ ≈ 3.3 cc" |
| Actual MAP σ max (patient 3) | **4.049 cc** |

Patient 3's belief is initialised at σ = 3.5 cc (clipped) instead of 4.049 cc.
During the DP, any hypothetical branch that pushes σ above 3.5 cc is also clipped.
The `belief_model.py` comment is factually wrong.

**Expected impact:** small — patient 3 is 1 of 58, and at large σ the DP policy is
already near-saturated (high caution). But it is a correctness bug.

**Action:** extend σ grid to ~4.5 cc (add one point to the tail segment), and correct
the comment in `belief_model.py`.

### Finding 2 — μ grid headroom above 21 cc is intentional, not waste

| Stat | Value |
|---|---|
| μ grid max | 30.0 cc |
| Actual max patient mean (patient 5) | **21.18 cc** |
| Headroom | ~8.8 cc |

The top segment `[16.5, 30.0]` cc (30 points) extends ~9 cc beyond the most extreme
observed patient mean.  This headroom is appropriate: during DP backward induction,
hypothetical branches can push the running Welford mean above the patient's observed
mean.  For a belief at μ=21 cc, an overlap observation o' > 39 cc (probability ~2×10⁻⁷
under Normal(21, 3.5²)) would be needed to push μ' above 30 cc — negligible in practice,
but the margin costs nothing and protects against future patients outside the current cohort.

**Action:** none — 30 cc ceiling is appropriate. Close this finding.

## Additional data

```
Per-patient mean overlap:  min 0.01, median 1.17, mean 2.90, p90 6.90, max 21.18 cc
Per-patient MAP σ:         min 0.018, median 0.587, p90 1.52, max 4.049 cc
Max single observation:    28.56 cc (patient 5)
```

## Status: COMPLETE — Finding 1 fixed (σ grid extended to 4.5 cc, comment corrected, _VOLUME_SPACE hardcoded to 44 cc); Finding 2 closed (no action).
