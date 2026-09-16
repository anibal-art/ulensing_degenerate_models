# Alpha / threshold calibration plan (no new calibration run)

This document does not choose alpha and does not run a new calibration.
It summarizes the existing first-pass H0 calibration result and lays out
the sample-size-vs-alpha trade-off so the user can make an informed,
explicit choice before the (separate, later) final calibration run.

## Existing first-pass result (unchanged, not extended here)

From `H0_CALIBRATION.md` (N=60 built, N=59 valid, seed 707070, using the
final policy as it stood at that point -- see caveat below): empirical
`p(Delta_chi2_LRT | H0)`, mean 2.39, median 1.52, range [0, 11.37].

| alpha | quantile | point estimate | 90% bootstrap CI |
|---|---|---|---|
| 0.10 | 0.90 | 4.72 | [4.10, 6.13] |
| 0.05 | 0.95 | 5.85 | [4.67, 8.76] |
| 0.01 | 0.99 | 9.69 | [6.24, 11.37] |

**This is a stale point estimate**, not because the definition of the
test statistic changed, but because it predates the final scalable
fitter policy. In particular, H1-generated events now use the
truth-started-H1 projected second H0 start, while H0-generated
calibration events use a truth-started H0 followed by an exact nested
H1 start from the settled H0 solution. The H1-blind alternative was
tested and rejected. Therefore this table is retained only as a
first-pass reference point and must not be used as the final calibrated
threshold.

## Sample-size-vs-alpha trade-off (tail-count table)

Tail count = number of null events at or above the candidate threshold
= `N_H0 * alpha`. Precision of the empirical quantile scales roughly
like `1/sqrt(tail_count)` -- a handful of order statistics (tail count
<10) gives a poorly resolved threshold; several dozen gives a
reasonably stable one.

| N_H0 | alpha=0.05 -> tail count | alpha=0.02 -> tail count | alpha=0.01 -> tail count |
|---|---|---|---|
| 1000 | 50 | 20 | 10 |
| 2000 | 100 | 40 | 20 |
| 5000 | 250 | 100 | 50 |
| 10000 | 500 | 200 | 100 |

As a rule of thumb: tail count >~20-50 gives a reasonably stable
quantile estimate for reporting a threshold with a defensible
confidence interval; tail count <10 (e.g. the current N=59 at
alpha=0.01, tail count~0.6) should not be treated as resolved.

## Required order (unchanged from the earlier freeze)

```
1. choose alpha scientifically (a false-positive budget for the
   eventual 1M-event population, or an equivalent operating-point
   argument) -- USER DECISION, not made here or automatically;
2. use the frozen `lrt_simulation_policy_v1` fitter policy for the
   calibration. The H1-blind/H0_2-blind alternative has been tested and
   rejected;
3. run H0 calibration with that exact, frozen policy, at N_H0 sized
   from the table above for the chosen alpha;
4. derive the empirical D_threshold(alpha) from that run, with its own
   bootstrap CI;
5. only then is `[4,9]` replaced by a real, calibrated threshold.
```

`[4,9]` remains a development-only bookkeeping interval until this
sequence completes. No step of this sequence is executed in this
session's work -- this document is planning only.
