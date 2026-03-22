# In Simple Terms

This repository computes radiation dose recommendations for adaptive fractionation when the overlap between target volume and organ-at-risk changes over time.

## The Big Picture

The core question is:

> Given the overlap measurements seen so far, how much dose should we deliver in the current fraction?

The code answers that by balancing two competing goals:

- Deliver enough total dose to hit the prescription.
- Avoid giving extra dose when overlap is large.

The main implementation lives in [../adaptive_fractionation_overlap/core_adaptfx.py](../adaptive_fractionation_overlap/core_adaptfx.py).

## Main Entry Points

- `adaptive_fractionation_core(...)`
  - Computes the best dose for one fraction.
- `adaptfx_full(...)`
  - Simulates the full treatment course by calling `adaptive_fractionation_core(...)` once per fraction.
- `precompute_plan(...)`
  - Calls `adaptive_fractionation_core(...)` repeatedly for many hypothetical next overlaps to build a lookup table.
- `policy_calc(...)`
  - Analytical helper that computes a policy from a fixed overlap mean and standard deviation.

## Who Calls What

```text
app.py
  -> adaptive_fractionation_core(...)   # one-fraction recommendation
  -> precompute_plan(...)               # lookup table for possible next overlaps
  -> adaptfx_full(...)                  # full-course simulation

adaptfx_full(...)
  -> adaptive_fractionation_core(...)   # recompute best action each fraction
  -> penalty_calc_single(...)           # total plan penalty summary

precompute_plan(...)
  -> std_calc(...)                      # estimate overlap uncertainty
  -> min_dose_to_deliver(...)           # compute minimum feasible dose
  -> adaptive_fractionation_core(...)   # evaluate many hypothetical next overlaps

adaptive_fractionation_core(...)
  -> std_calc(...)                      # estimate spread of future overlaps
  -> get_state_space(...)               # discretize possible overlaps
  -> probdist(...)                      # probability per overlap state
  -> max_action(...)                    # clip allowed dose actions
  -> penalty_calc_matrix(...)           # score many actions vs many overlaps
  -> penalty_calc_single(...)           # score the actual chosen action
```

## How `adaptive_fractionation_core(...)` Works

Think of it as a dynamic-programming optimizer.

## MDP Dimensionality

The current implementation is best understood as a finite-horizon MDP solved by dynamic programming.

There are two valid ways to describe its dimensionality:

- As implemented in the DP tables, it is effectively **3-dimensional**:
  - time-to-go / fraction index
  - accumulated dose state
  - current overlap state
- If you treat time as part of the finite-horizon setup rather than part of the state, then the decision state is effectively **2-dimensional**:
  - accumulated dose
  - current observed overlap

The important implementation detail is that `mean` and `std` of the overlap distribution are **not** explicit state variables right now. They are recomputed once from the observed history at the start of each call:

- `mean = volumes.mean()`
- `std = std_calc(volumes, alpha, beta)`

Then they are held fixed while the backward recursion runs.

That is why the main value tables in `adaptive_fractionation_core(...)` have shape:

`(number_of_fractions - fraction, len(dose_space), len(volume_space))`

So the recursion is indexed by:

- remaining time steps
- discretized accumulated dose
- discretized overlap

For the default settings at fraction 1, that is roughly:

- 4 future time steps
- about 70 accumulated-dose states
- 200 overlap states

If you extend the model so `mean` and `std` become explicit state variables, then the mathematical state would have **4 non-time coordinates**:

- accumulated dose
- current overlap
- overlap mean
- overlap std

If you also count time explicitly in the finite-horizon problem, that becomes **5 coordinates total**:

- time
- accumulated dose
- current overlap
- overlap mean
- overlap std

If you stored that model in a dense DP table, it would likely add two extra axes for `mean` and `std`.

The important point is not just that the state gets bigger. The important point is that the policy could start reasoning about **how much it has learned so far** about this patient's overlap distribution.

Right now, the solver recomputes

- `mu = volumes.mean()`
- `std = std_calc(volumes, alpha, beta)`

at the start of the current fraction, and then treats that overlap distribution as fixed while solving the remaining fractions.

That means the current model does **not** explicitly reason about this idea:

"After the next fraction, I will have one more measurement, so I should know the patient's overlap distribution better than I do right now."

If `mean` and `std` become part of the state, then the next state would carry not just:

- the next overlap
- the next accumulated dose

but also:

- the updated mean estimate
- the updated standard-deviation estimate

That creates room for a more realistic policy:

- early in treatment, be more cautious because the estimated distribution is still uncertain
- preserve flexibility while the model is still learning
- later in treatment, act more confidently because the patient's geometry pattern is better understood

So the real benefit of the extension is that the MDP can start modeling **learning over time**, not just dose allocation under a fixed guessed distribution.

In this document, the intended interpretation is that `time + mean + std` already gives a practical representation of confidence. Time tells us how many overlaps have been incorporated, and the running estimates become harder to move as more fractions are observed. Early in treatment, one new overlap can change the mean or spread estimate a lot. Later in treatment, the same new overlap usually changes them less.

So the extended state can already represent learning if we define the state-transition updates carefully. In other words, the confidence is not a separate number; it is reflected in how stable the belief state has become. Extra posterior-style variables would only be needed later if we decide we want a stricter Bayesian representation of uncertainty, rather than this practical estimator-based one.

That would still increase memory use and runtime substantially because both `values` and `policies` would need extra axes or a different representation.

### Inputs

- `fraction`: which treatment fraction we are on
- `volumes`: observed overlap values so far
- `accumulated_dose`: dose already delivered
- dose constraints: `min_dose`, `max_dose`, `mean_dose`

### Step 1: Define the treatment goal

The total prescribed dose is:

`goal = number_of_fractions * mean_dose`

For the default settings, that is `5 * 8 = 40 Gy`.

### Step 2: Model uncertainty in future overlap

The code assumes future overlap is approximately normally distributed.

- Mean = `volumes.mean()`
- Standard deviation = `std_calc(volumes, alpha, beta)`

This is a simple forecasting model: future overlap is expected to look like recent measured overlap, with uncertainty estimated from prior assumptions and observed variation.

### How `mu` and `sigma` change from fraction to fraction

This is the easiest way to think about it:

1. Before a fraction starts, you already have a history of measured overlaps, including the planning scan and any treatment fractions that have happened so far.
2. The current fraction's overlap is observed and included in `volumes`.
3. The code recomputes the overlap mean from that full history:
   - `mu = volumes.mean()`
4. The code recomputes the overlap standard deviation from that same history:
   - `sigma = std_calc(volumes, alpha, beta)`
5. It then assumes future overlaps are drawn from a normal distribution using that updated `mu` and `sigma`.
6. With that distribution fixed, it solves the dynamic program for the current fraction.
7. After the next fraction is observed, the history gets one new overlap value, and the whole update happens again.

In plain language: after every new fraction, the model says, "Given everything I have seen up to today, what do I now believe about the average future overlap and how variable it is?"

So `mu` and `sigma` are updated **between fractions**, not inside the backward recursion itself.
For one call to `adaptive_fractionation_core(...)`, they are treated as fixed inputs.

Code references:

- The main solver recomputes `sigma` and then builds the normal distribution from `volumes.mean()` and that `sigma` in [core_adaptfx.py#L161](/mnt/c/sources/Overlap_adaptfx/adaptive_fractionation_overlap/core_adaptfx.py#L161), [core_adaptfx.py#L162](/mnt/c/sources/Overlap_adaptfx/adaptive_fractionation_overlap/core_adaptfx.py#L162), and [core_adaptfx.py#L164](/mnt/c/sources/Overlap_adaptfx/adaptive_fractionation_overlap/core_adaptfx.py#L164).
- The actual `sigma` calculation happens in [helper_functions.py#L48](/mnt/c/sources/Overlap_adaptfx/adaptive_fractionation_overlap/helper_functions.py#L48) through [helper_functions.py#L79](/mnt/c/sources/Overlap_adaptfx/adaptive_fractionation_overlap/helper_functions.py#L79). That function searches over candidate standard deviations and returns the one with the largest posterior score.
- A useful cross-check is the plan-precomputation helper, which repeats the same update in [core_adaptfx.py#L313](/mnt/c/sources/Overlap_adaptfx/adaptive_fractionation_overlap/core_adaptfx.py#L313) and [core_adaptfx.py#L314](/mnt/c/sources/Overlap_adaptfx/adaptive_fractionation_overlap/core_adaptfx.py#L314).
- If you want to see where a new overlap gets folded into the history, [core_adaptfx.py#L321](/mnt/c/sources/Overlap_adaptfx/adaptive_fractionation_overlap/core_adaptfx.py#L321) calls `adaptive_fractionation_core(...)` with `np.append(volumes, volume)`, which is the code-level version of "observe one more overlap, then update `mu` and `sigma`."

### Step 3: Discretize the problem

Because the optimizer is not continuous, it turns the problem into grids:

- `volume_space`: 200 possible overlap states
- `dose_space`: possible accumulated tumor doses
- `delivered_doses`: allowed per-fraction doses, e.g. `6.0, 6.5, ..., 10.0`

This makes the problem finite and solvable with backward recursion.

### Step 4: Score candidate actions

The penalty functions are in [../adaptive_fractionation_overlap/helper_functions.py](../adaptive_fractionation_overlap/helper_functions.py).

In plain language:

- Higher overlap should make dose less attractive.
- Giving more than `min_dose` increases penalty.
- Penalty grows faster when overlap is high.

That is why the recommended dose tends to go down when overlap goes up.

### Step 5: Work backward from the end of treatment

The function builds `values` and `policies` arrays for the remaining fractions.

For each future state, it asks:

> If I have this accumulated dose now, and overlap may take these values next, which action gives the best long-term outcome?

This is the dynamic-programming part.

The code evaluates:

- immediate penalty for a dose decision
- plus expected future value from the remaining fractions

It then chooses the action with the best combined result.

### Step 6: Return the actual recommendation

The outputs include:

- `physical_dose`: the recommended dose for the actual observed overlap
- `policies_overlap`: the full policy curve over all possible overlaps
- `values`: value tables for future fractions
- `final_penalty`: projected outcome if this recommendation is followed

## The Helper Functions

These are the most important helpers in `helper_functions.py`:

- `std_calc(...)`
  - Estimates the standard deviation of overlap for the current patient history.
- `get_state_space(...)`
  - Builds the overlap grid used by the optimizer.
- `probdist(...)`
  - Converts the overlap distribution into probability mass on that grid.
- `max_action(...)`
  - Prevents actions that would overshoot the total prescribed dose.
- `penalty_calc_single(...)`
  - Computes penalty for one dose and one overlap.
- `penalty_calc_matrix(...)`
  - Computes the same penalty in bulk across many states.

## What `adaptfx_full(...)` Does

`adaptfx_full(...)` is a wrapper around the core function.

It loops from fraction 1 to the last fraction and repeatedly calls `adaptive_fractionation_core(...)` using only the overlap history available at that point. That makes it the easiest function to read if you want the high-level treatment workflow.

Use it when you want to simulate a whole patient course.

## What `precompute_plan(...)` Does

`precompute_plan(...)` is also a wrapper around the core function.

It takes the current patient state, tries many possible next overlap values, and records the dose recommendation for each one. The result is a table from overlap volume to recommended dose. This is useful when you want a precomputed decision rule instead of recomputing at treatment time.

## Important Constraints and Gotchas

- Invalid plans are punished with a very large negative number rather than explicit exceptions.
  - This is how underdose and overdose states are rejected.
- `dose_steps` changes both runtime and solution granularity.
  - Smaller steps give finer decisions but more computation.
- `adaptive_fractionation_core(...)` returns a lot of internal arrays.
  - For most product work, the most important outputs are `physical_dose`, `policies_overlap`, and `final_penalty`.
- `adaptfx_full(...)` stores accumulated dose before the next fraction.
  - Read it carefully if you change reporting.
- `pandas` is imported by `core_adaptfx.py` because `precompute_plan(...)` returns a DataFrame.

## Where to Start If You Want to Extend It

Read in this order:

1. `adaptfx_full(...)`
2. `adaptive_fractionation_core(...)`
3. `penalty_calc_single(...)` and `penalty_calc_matrix(...)`
4. `std_calc(...)`, `get_state_space(...)`, `probdist(...)`
5. `precompute_plan(...)`

If you want to change behavior, the usual extension points are:

- penalty definition
- overlap distribution model
- dose discretization (`dose_steps`)
- feasibility rules around total prescribed dose

## Practical Mental Model

This code is a dose recommender built on:

- a simple forecast of future overlap
- a penalty function for unsafe or inefficient dosing
- backward dynamic programming to choose the best action now

If you keep those three pieces straight, the rest of the file becomes much easier to modify safely.
