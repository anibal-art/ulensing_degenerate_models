# Fresh H1 holdout plan (selection specification only -- not executed)

No holdout is generated, materialized, or fit in this task. This
document specifies how it will be built later, so the selection rule is
frozen and reviewable before any holdout result is inspected.

## Target size

`N_H1_holdout = 200-300` **fitted/passing events** (not raw catalog
draws -- see "raw vs fitted" distinction below). Exact N within this
range to be set when the holdout is actually built, based on how much
compute is available at that time; not decided here.

## Selection procedure (to be executed later, exactly as specified)

1. Fixed random seed, distinct from every seed already used in this
   project (424242 profiling sample, 636363 H0-generated smoke sample,
   707070 H0-calibration sample, 838383 H1-basin-validation
   development sample). A new seed must be chosen and recorded at
   holdout-build time, before any candidate row is drawn.
2. Deterministic permutation over the full raw catalog (same mechanism
   as every other sample in this project: `np.random.default_rng(seed)
   .permutation(N_TOTAL_ROWS)`), walked in order, materializing
   `truth_parallax=True` (real catalog piE) events via the same
   `standalone_materialize.py` path already used and verified.
3. **Zero overlap**, verified programmatically (not by inspection)
   against:
   - the development N=100 sample (`h1_basin_validation_sample_frozen.csv`)
   - the 25-event profiling sample (`profiling_sample_frozen_100.csv`)
   - the H0-generated smoke sample (`h0_generated_sample_frozen.csv`)
   - the H0-calibration sample (`h0_calibration_sample_frozen.csv`)
   - the 134 development-contaminated rows (`extreme100` + tail/control)
4. IDs frozen (written to a manifest CSV) **before** any fit is run on
   the holdout -- no opportunistic selection after seeing results.
5. The holdout is fit with the frozen
   `lrt_simulation_policy_v1` policy only, plus an expensive independent
   reference (the same
   `old_final`-free controlled4-style multistart used throughout this
   session's validation work, or an equivalent), so the holdout
   provides a genuinely independent basin-gap/repair-rate estimate --
   not reusing the development-set trigger, panel, or any
   development-informed choice.

## Explicit non-negotiables

- Holdout membership is determined by the frozen seed/rule BEFORE
  scientific results are inspected -- never selected opportunistically
  from already-materialized or already-fit events.
- The holdout is not used to tune anything; if it reveals problems, the
  response is to report them, not to iterate the fitter against the
  holdout itself (that would consume its independence).
- This holdout is separate from, and must not be confused with, the H0
  calibration sample (`ALPHA_CALIBRATION_PLAN.md`) or the eventual
  million-event Sedighe production population.

## Not done in this task

No candidate rows drawn, no materialization, no fits. This is a
specification for later execution only.
