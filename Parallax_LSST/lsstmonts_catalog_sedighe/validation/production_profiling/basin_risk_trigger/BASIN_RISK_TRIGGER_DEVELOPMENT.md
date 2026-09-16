# Demonstration 1: identify H0 basin-risk (development only)

**STOP.** Selective adaptive refitting is not sufficiently supported by
the tested production-safe diagnostics on this development set. No
individual diagnostic and no tested small combination reaches high
dangerous-case recall at <=20% activation (best single feature: 20-40%
worst-case recall across the 5-25% activation range; best combination:
53.3% recall, but only at 45.2% actual activation -- more than double
the 20% budget). This is a development-only, N=93 result on the
now-repurposed N=100 sample; no second H0 start is implemented; no new
holdout is generated. See §14 for the exact stop-rule statement and
what is/is not done next.

## Sample-status change

The N=100 H1-generated sample used for the corrected basin-gap
validation (`BASIN_GAP_VALIDATION.md`) is now a DEVELOPMENT SET for
trigger design. It must not later be presented as independent validation
of any adaptive policy that uses a trigger selected here -- a fresh
H1-generated holdout will be required before that.

## 1. Label definitions (frozen before any feature analysis)

```
D_s = Delta_chi2_LRT from the current final policy (final_policy.py v2)
D_r = Delta_chi2_LRT from the old_final-free controlled4 robust reference
```

Primary label:

```
dangerous = (D_r < 9) and (D_s >= 4) and (D_s > D_r)
```

equivalently: exists tau in [4,9] such that D_s >= tau > D_r -- the cheap
policy would claim detection at some plausible threshold for which the
robust reference would not.

Secondary labels (NOT the optimization target):
```
large_gap_abs = abs(D_s - D_r) > 5
large_gap_up  = (D_s - D_r) > 5
```

**`[4,9]` is a provisional development operating range, not the final
scientifically adopted threshold interval** (alpha has not been
scientifically frozen -- `H0_CALIBRATION.md`). The trigger characterized
here is demonstrated only for dangerous crossings inside `[4,9]`; if the
eventual calibrated threshold lands outside this interval, dangerous-case
recall must be re-evaluated before any production use.

## 2. Population (§0 of the task)

Development population: the **93 clean/unflagged** events, i.e.
`post-continuation chi2nu_h0 <= 50` under the ALREADY-frozen catastrophic
sanity mechanism (`FINAL_POLICY.md`). The 7 already-flagged events are
excluded from every activation-rate denominator, recall calculation,
trigger-selection step, and bootstrap -- reported separately, audit-only,
below.

- development n = 93, dangerous = **15** (16.1%), large_gap_abs = 30,
  large_gap_up = 28.
- dangerous event ids: 229716, 520062, 324312, 704717, 97760, 790880,
  788605, 909986, 963300, 236102, 798179, 837177, 159493, 114675, 703006.

### Excluded (catastrophic, chi2-sanity-flagged) events -- audit only

| event_id | D_s | D_r | dangerous | large_gap_abs | large_gap_up | chi2nu_h0 |
|---|---|---|---|---|---|---|
| 631745 | 21635.3 | 2976.1 | False | True | True | 69.19 |
| 553869 | 18650.7 | 18650.4 | False | False | False | 233.99 |
| 463414 | 34825.2 | 34648.2 | False | True | True | 95.36 |
| 632725 | 26121.6 | 0.5 | **True** | True | True | 74.48 |
| 585618 | 105699.8 | 213.0 | False | True | True | 755.90 |
| 190719 | 13335.4 | 13335.6 | False | False | False | 107.53 |
| 486919 | 30847.98 | 25965.0 | False | True | True | 110.75 |

`excluded_from_development=True` for all 7; none contribute to any
metric above. (Row 632725 IS technically `dangerous` by the label
formula -- included here for full transparency, per the task's explicit
instruction to report labels for excluded events, but per §0's rule it
plays no role in trigger development: it is already caught by the
existing catastrophic mechanism regardless of any new trigger.)

## 3. Leakage audit

Every candidate feature is computable from: the observed light curve,
the truth-blind morphology seed, and the nominal/continuation-repaired
H0 fit (which is itself already part of the frozen, pre-decision
pipeline). None uses truth, the robust reference, its chi2, its strategy
identity, or the development labels. Full table in
`feature_safety_audit.csv`; every row is `production_safe=True`,
`truth_reference_leakage=False`. The H1 fit was deliberately not used as
a predictor (per instruction "prefer initially NOT to use the H1 fit").

| feature | production_safe | source | leakage |
|---|---|---|---|
| chi2nu_h0 | True | H0 fit chi2 + own point/param count | False |
| morph_mismatch | True | morphology seed (already frozen) | False |
| dlogtE, dlogrho, du0, dt0_over_tE | True | H0 seed vs H0 fit params | False |
| d_bounds, h0_bound_active | True | H0 fit params + production bounds | False |
| R_max_resid, T25/T50/T75_resid, A_resid | True | H0 fit standardized residuals | False |
| dw_stat | True | H0 fit standardized residuals | False |
| logkappa_J | True | H0 fit Jacobian at solution | False |

## 4. Exact feature definitions

### dof_H0 / chi2nu_h0

Reused verbatim from `final_policy.py`: `dof_H0 = max(1, n_points - 4)`,
`n_points` = total photometric points across all bands with data
(`final_policy.n_photometry_points`), `4` = the H0 nonlinear parameter
count (`t0,u0,tE,rho`) -- profiled source/blend flux parameters are
excluded (analytically profiled out by the frozen `bounded_flux_profile`
patch, never counted as free parameters). `chi2nu_h0 = chi2_H0 / dof_H0`,
kept continuous; the existing catastrophic threshold (50) is NOT assumed
relevant here and was not used to pick a new cutoff.

### morph_mismatch

The existing, already-frozen `morphology_seed_from_curves(curves)
["mismatch"]` -- the FSPL dimensionless-lookup nearest-match residual
already computed operationally by the morphology estimator. Not
modified, not retuned, not rebuilt.

### Seed-displacement features

```
dlogtE      = abs(log(tE_fit / tE_seed))
dlogrho     = abs(log(rho_fit / rho_seed))
du0         = abs(u0_fit - u0_seed)
dt0_over_tE = abs(t0_fit - t0_seed) / tE_fit
```
`_seed` = the morphology seed; `_fit` = the (possibly continuation-
repaired) final H0 solution. 0/93 non-finite/invalid in this sample (all
`tE,rho > 0` by construction of the production bounds).

### Distance to bounds

The production fitter, in the frozen `physical` TRF coordinate mode,
optimizes `t0,u0,tE,rho` directly in raw physical units (no internal
log-transform); `x_scale='jac'` only affects the optimizer's internal
trust-region step scaling, not the parameterization itself. For this
diagnostic (defined before looking at any label), bound distance is
normalized **in log-scale for `tE,rho`** (bounds span 6-7 orders of
magnitude: `tE in [0.1,500000]`, `rho in [1e-7,10]`) and **in linear
scale for `t0,u0`** (bounds span a single scale: `t0` is a data-driven
`center +- half_width` window, `u0 in [-10,10]`):
```
linear: d = min(x-lo, hi-x) / (hi-lo)
log:    d = min(log(x)-log(lo), log(hi)-log(x)) / (log(hi)-log(lo))
d_bounds = min over the 4 parameters
```
`h0_bound_active` = whether scipy's own `optimizer_active_mask` (from the
already-computed TRF result, not re-derived) has any nonzero entry.
0/93 events had `h0_bound_active=True` in this sample.

### Standardized residuals

Confirmed exactly (both via the chi2 identity and via scipy's own least-
squares convention): `fit_object.fun` (the residual vector scipy's TRF
returns at the solution) already equals the standardized residual
`r_i = (F_i - F_H0(t_i))/sigma_i` used in chi2 -- verified numerically,
`sum(fun_i^2) == chi2_H0` exactly (`reproduction_chi2_diff=0.0` for all
100 events between a fresh in-memory re-run and the already-frozen
result). No separate sigma_i lookup or re-derivation was needed or
performed.

**Band segmentation** (needed to associate each residual with a time):
confirmed by direct source inspection of `fit_lc.py`'s
`add_rubin_telescopes`/`create_fit_event` that `fit_object.fun`/`.jac`
rows are ordered by telescope-addition order: Roman/W149 first (always
empty in this dataset, contributes 0 rows), then `LSST_BANDS = ["u",
"g","r","i","z","y"]` in that fixed order, skipping empty bands, each
contributing exactly `len(curves[band])` consecutive rows in the same
(already time-sorted, verified) order as the input light curve. Verified
empirically: `len(fun) == n_points` exactly for every event checked.

**Residual morphology** (R_max_resid, T25/T50/T75_resid, A_resid): all
bands pooled into one (time, |residual|) list, sorted by time. Peak =
`argmax(|residual|)`. For each `q in [0.25,0.50,0.75]`: walk outward from
the peak (in time) while `|residual| >= q * R_max_resid`; width = time
from peak to the LAST point still satisfying the threshold (not the
first failing point); a side that reaches the data boundary without
failing is CENSORED (never extrapolated) -- this is the exact same
convention `morphology_extraction.py` already uses for flux-excess
widths (same `q` values, same walk-until-first-failure rule), applied to
standardized residuals instead of flux excess. `T50_resid`/`T75_resid`/
`T25_resid` = `left_width + right_width` only if BOTH sides are
measurable, else missing (not zero, not imputed). `A_resid =
(right_width-left_width)/(right_width+left_width)` only when `T50_resid`
is fully measurable and the sum is nonzero. **T25_resid is audit-only**
(not added to the candidate trigger set without this explicit note).

Coverage note: `T50_resid`/`T75_resid`/`T25_resid` are exactly **0.0**
(not missing) for a large fraction of events (55/93 for `T50_resid`) --
the largest standardized-residual excursion is very often an isolated
single point with both neighbors already below the threshold, so the
temporal-width features are frequently degenerate on this dataset;
`R_max_resid` (peak magnitude) carries most of the residual-based
signal. `A_resid` is missing for 56/93 (60.2%) as a direct consequence.

### Temporal residual correlation (dw_stat)

Classic Durbin-Watson statistic per band (`sum(diff(r)^2)/sum(r^2)`,
time-ordered within band, no theoretical p-value attached -- used purely
as an empirical ranking feature), combined across bands via a
point-count-weighted average (weight = `n_band-1`, the number of
consecutive pairs each band contributes). A band with `n_band<2`
contributes nothing (not zero) to the combination. `dw_stat` is missing
only if no band has >=2 points -- did not occur in this sample (0/93
missing). Direction: **low value suspicious** (near 0 = strong positive
autocorrelation = systematically trending residuals, a sign of model-
shape misspecification; near 2 = no autocorrelation; near 4 = negative
autocorrelation).

### Jacobian conditioning (logkappa_J)

`fit_object.jac` at the final H0 solution (shape `(n_points, 4)`,
confirmed via direct inspection), in the raw physical-coordinate
parameterization (mode=`physical`, no internal log-transform; `x_scale=
'jac'` affects only the optimizer's internal step scaling, not the
returned Jacobian's units). `logkappa_J = log10(cond(J))` via SVD singular
values (`s.max()/s.min()`), undefined/unavailable if any singular value
is non-finite or the minimum is <=0. 0/93 unavailable in this sample.

## 5. Missing-value policy (defined before ranking)

Denominator is always the full 93-event development population.
Default: a missing diagnostic value means the event is **not** triggered
by that diagnostic (its suspicion score is set to `-infinity`, so it can
never enter a top-activation-fraction selection). No feature's coverage
was low enough in this sample to warrant testing "missingness itself" as
a separate risk signal, except implicitly for `A_resid` (60.2% missing,
reported and visible in its results, not elevated to its own diagnostic
since it was not among the primary candidate set to begin with).

## 6. Table audit

See §2 above for population counts and dangerous-event IDs; see the
Coverage note in §4 for missing rates; `feature_summary.csv` has full
min/median/p90/max per diagnostic. Confirmed: 0 truth or reference-
derived quantity enters any predictor column (§3).

## 7-8. Recall vs activation, with tie handling (development n=93)

Full tables: `recall_vs_activation_per_feature.csv`. Worst-case recall
(the conservative comparator per instruction) at each budget, best single
feature:

| activation | best single feature | recall (worst-case) |
|---|---|---|
| 5% | dlogrho | 20.0% |
| 10% | morph_mismatch | 20.0% |
| 15% | dlogrho | 26.7% |
| 20% | dlogrho | 33.3% |
| 25% | dlogrho | 40.0% |

`T75_resid` shows a large worst/best-case gap (13.3% vs 60-86.7% at
20-25% activation) because 80/93 events are exactly tied at
`T75_resid=0` -- essentially uninformative once tie-breaking is treated
conservatively, consistent with the coverage note in §4. `abs_A_resid`
(tested as its own diagnostic, magnitude of the ambiguous-sign
`A_resid`) similarly has a large tied group (29/93, from its 60%
missingness) and 0% worst-case recall throughout.

## 9. Stratified bootstrap (seed=20260915, 2000 resamples, 5-95% interval)

Full table: `bootstrap_recall.csv`. All individual features show wide,
overlapping intervals at every budget (e.g. `dlogrho` at 20%: median
33.3%, CI [13.3%, 53.3%]) -- consistent with only ~15 dangerous positives
in n=93; no individual feature's interval clears anywhere near 80%
recall at <=20% activation.

## 10. Small combinations (union of top-q-percentile rule, max 2 members)

| combination | per-member q | actual union activation | recall | precision |
|---|---|---|---|---|
| R_max_resid OR morph_mismatch | 20% | 38.7% | 46.7% | 19.4% |
| R_max_resid OR morph_mismatch | 25% | 45.2% | **53.3%** | 19.0% |
| R_max_resid OR dlogtE | 25% | 38.7% | 40.0% | 16.7% |
| R_max_resid OR d_bounds | 25% | 40.9% | 40.0% | 15.8% |
| chi2nu_h0 OR R_max_resid | 25% | 28.0% | 33.3% | 19.2% |

The best-performing combination reaches 53.3% recall, but only by
consuming 45.2% actual activation -- more than double the 20% budget the
whole exercise is meant to respect (an OR-of-two-top-20%-each union
naturally activates well above 20% whenever the two features are not
highly redundant, which is the case here). No combination was found that
reaches high recall within a ~20% actual activation budget.

## 11. Development success criteria (coarse categories, per instruction)

At <=20% activation: best single feature 20-33.3% recall; best
combination's *20%-target* recall is 46.7% at 38.7% actual activation.
None reach "VERY PROMISING" (all/all-but-one) or "POSSIBLY USEFUL"
(~80%). This falls squarely in **INSUFFICIENT**: no feature or small
combination reaches high recall at <=20% activation; several dangerous
events are missed at every tested budget up to 25%.

## 12. Explicit separation from Demonstration 2

`D_s2` and `repaired` columns are reserved in
`basin_risk_development_table.csv` but **not populated**. No second H0
start was implemented or run in this task.

## 13. Outputs

- `basin_risk_development_table.csv` (100 rows: 93 development + 7
  audit-only excluded)
- `feature_safety_audit.csv`, `feature_summary.csv`
- `recall_vs_activation_per_feature.csv`, `bootstrap_recall.csv`,
  `recall_vs_activation_combinations.csv`
- `build_basin_risk_table.py`, `analyze_basin_risk_trigger.py`
- this document

## 14. Stop-rule conclusion

**Selective adaptive refitting is not sufficiently supported by the
tested production-safe diagnostics on this development set.**

Per the task's explicit stop rule, this is NOT followed by: raising the
allowed activation budget, adding machine learning, tuning further
cutoffs, adding more fitter strategies, implementing multistart for
every event, modifying morphology, using truth/reference-derived
features, generating a new holdout, or implementing the second H0 start.
None of those were done. This document reports the negative result and
stops.
