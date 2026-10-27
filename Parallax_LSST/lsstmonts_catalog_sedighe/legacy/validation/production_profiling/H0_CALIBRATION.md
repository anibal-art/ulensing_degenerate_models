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

Re-run with the v2 final policy (adds safeguard 1, same-point
continuation on catastrophic false convergence -- `FINAL_POLICY.md`):
distribution is numerically IDENTICAL to the v1 run below, because
continuation fired on 0/59 H0 fits and only 2/59 (3.4%) H1 fits under the
null, each with negligible chi2 change (consistent with 0/59 chi2-sanity
flags observed either version) -- `results/
h0_calibration_delta_lrt_v2_60events.csv`. Average TRF/event rises
slightly to 2.61 (from 2.58) from those 2 extra continuation fits;
rescue fraction unchanged.

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

## CORRECTED (2026-09-15): alpha is NOT chosen here

An earlier version of this document picked alpha=5% and justified it by
what N=59 happens to resolve well. **That reasoning was backwards and has
been retracted.** Alpha is a scientific operating choice (how many false
positives are acceptable in the 1M-event population, and what that costs
in missed real detections) -- it must be specified independent of
whatever a convenience-sized calibration sample can currently measure,
not backed into from sample-size convenience.

**N=59 is retained only as a FIRST-PASS reference point**:
`q95 ~5.85, 90% bootstrap CI ~[4.67, 8.76]` -- not adopted as a threshold,
not implying alpha=5% is the chosen operating point.

### Sample size required per candidate alpha

Tail count (expected number of null events at or above the threshold)
governs the precision of the quantile/threshold estimate -- roughly,
relative uncertainty on the estimated tail probability scales like
`1/sqrt(tail_count)`. Order-of-magnitude guide (`tail_count = N * alpha`):

| N | alpha=0.05 -> tail count | alpha=0.01 -> tail count | alpha=0.001 -> tail count |
|---|---|---|---|
| 200 | 10 | 2 | 0.2 |
| 1000 | 50 | 10 | 1 |
| 2000 | 100 | 20 | 2 |
| 5000 | 250 | 50 | 5 |
| 10000 | 500 | 100 | 10 |

Matches the values given in the closure instructions exactly (N=1000:
alpha=0.05->~50, alpha=0.01->~10; N=2000: alpha=0.05->~100, alpha=0.01->
~20; N=5000: alpha=0.01->~50). As a rule of thumb, a tail count of
~20-50+ gives a reasonably stable quantile estimate; single-digit tail
counts (this project's current N=59 at alpha=0.01, tail count~0.6) should
not be treated as a resolved threshold.

### Stopping for the scientific alpha decision

**This is a stop point.** The calibration sample should not be extended
further, and no threshold should be adopted as final, until alpha is
explicitly specified (by the user / the science requirements this
detection is feeding into -- e.g. a target false-positive count in the 1M
population, or a specific p-value convention). Once alpha is specified,
the required N follows directly from the table above, and the H0
calibration sample can be extended to that N using the exact same,
unmodified procedure (`build_h0_calibration_sample.py`,
`final_policy.py`) -- no retuning, no algorithm changes.
