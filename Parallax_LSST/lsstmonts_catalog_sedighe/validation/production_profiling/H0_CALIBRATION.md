# H0 calibration -- NOT the 1M Sedighe sample

Distinct, deterministic sample (seed 707070), `truth_parallax=False` (a
genuine no-parallax simulation, not a truth override), zero overlap with
the H1-generated profiling sample, the earlier 5-event H0-generated smoke
sample, or any development-contaminated row. Fit with the EXACT frozen
final policy (`final_policy.py`) that will be used in production -- same
bounds, same morphology estimator, same nested rescue, same sanity-flag
logic, nothing recalibrated or retuned for this step.

## Sample size (explicit, time-boxed decision)

Target was N=150; the build was stopped at **N=60 materialized events**
(candidates evaluated: ~190, pass rate ~32%, `results/
h0_calibration_sample_candidate_log.csv` partial /
`results/h0_calibration_sample_frozen.csv` reconstructed from the 60 H5
files actually written to disk plus the build log, since the process was
stopped before its own end-of-run CSV write). This is a real time/scope
tradeoff made explicitly in this session, not a scientifically-driven
sample size -- it is enough to calibrate a workable threshold at a modest
alpha, but is explicitly **not** large enough for a well-resolved small-alpha
(e.g. 1%) tail estimate; see below.

Of 60, 1 event (865478) hit `morphology_not_measurable` (the H1 fit could
not run at all -- an honest estimator/domain inconsistency, not silently
substituted) -- **N=59 valid** for the calibration distribution, and this
1/60 = **1.7% estimator-failure rate** is itself a production-readiness
number worth carrying forward.

## Empirical p(Delta_chi2_LRT | H0), n=59

`results/h0_calibration_delta_lrt_60events.csv`.

- range: `[0.0, 11.369]` (0.0 achievable exactly because the nested
  rescue guarantees `chi2_H1 <= chi2_H0`).
- mean = 2.392, median = 1.516, std = 2.307.
- rescue fraction under the null: **57.6%** (34/59) -- much higher than
  the 16% seen on H1-generated events (`PRODUCTION_PROFILING_FINAL.md`).
  This is expected and reassuring: under a true null, chi2_H0 and the
  morphology-seeded chi2_H1 are close by construction (no real parallax
  signal to separate them), so which one lands lower by chance is close
  to a coin flip -- exactly the regime the rescue safeguard exists for,
  and it is doing real, frequent work here to keep the null
  distribution's lower tail honestly bounded at 0.
- 0/59 chi2-sanity flags fired (no catastrophic false convergence
  observed in this null sample, vs 3/25=12% in the H1-generated basin-gap
  sample -- plausibly because H0-generated light curves are intrinsically
  simpler for both models to fit; not conclusively explained here).
- avg TRF/event = 2.58 (including rescues).

Quantiles and bootstrap threshold uncertainty (20000 resamples,
90% percentile interval):

| alpha | quantile | point estimate | 90% CI |
|---|---|---|---|
| 0.10 | 0.90 | 4.717 | [4.098, 6.130] |
| **0.05** | **0.95** | **5.851** | **[4.672, 8.759]** |
| 0.01 | 0.99 | 9.687 | [6.239, 11.369] |

## Chosen alpha and threshold

**alpha = 5%, calibrated threshold = 5.85** (90% CI [4.67, 8.76]).
Reasoning: at N=59, the alpha=1% quantile is dominated by the single
highest observed value (11.37) and its bootstrap CI spans nearly the
entire observed range (6.2-11.4) -- not resolved with any real confidence
at this sample size. alpha=10% is well-resolved but a coarser
false-positive budget than typically wanted for a detection claim.
alpha=5% is the best-resolved choice that is still a scientifically
conventional false-positive budget. **Not the illustrative threshold=9**
used earlier for descriptive purposes only -- this is the first real
calibration pass.

**This is explicitly a provisional, first-pass calibration**, not a
final production threshold: N=59 gives single-digit-count precision at
alpha=5% (~3 order statistics inform the estimate) and essentially no
precision at alpha=1%. Before committing this threshold to the actual 1M
run, if the target alpha is 1% or smaller, or if the CI width above
[4.67, 8.76] is not acceptable, the calibration sample should be extended
substantially (several hundred to ~1000+ events, following exactly the
same procedure) -- this is a resourcing/time decision for the user, not
made here.
