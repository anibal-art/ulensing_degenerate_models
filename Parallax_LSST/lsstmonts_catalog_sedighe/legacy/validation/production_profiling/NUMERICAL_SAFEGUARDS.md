# Two minimal numerical safeguards on the 2-fit baseline

Tests whether the frozen 2-fit architecture (`TWO_FIT_ARCHITECTURE.md`) can
stay `2 nominal fits + rare/conditional objective safeguards` rather than
reverting to unconditional multistart. No bounds, starts, or the nominal
policy itself were changed; both safeguards are strictly conditional on an
explicit diagnostic of the nominal result, and each fires at most once per
fit/event. Implementation: `numerical_safeguards.py`,
`run_one_fit_full.py` (full `fit.fit_results` incl. `optimizer_nfev`,
`optimizer_njev`, `optimizer_status`, `optimizer_message`,
`optimizer_optimality`, `optimizer_active_mask` for every fit -- not the
trimmed dict `core.run_one_fit` returns), `run_two_fit_with_safeguards.py`.

## Safeguard 1: same-point continuation

Trigger: `optimizer_optimality` non-finite or `> 0.05` (the
`polish_optimality_threshold` already used historically for `old_final`).
Exactly one continuation TRF from the nominal fit's own final vector, same
model/objective/`production_candidate` bounds, all parameters free,
stricter tolerances (`xtol=ftol=1e-12, gtol=1e-10` vs the nominal
`1e-8/1e-8/1e-8`), no new physical seed. Kept only if it does not worsen
chi2.

## Safeguard 2: nested H1 rescue

Trigger: `chi2_H1 > chi2_H0 + 1e-6` after safeguard 1. Exactly one
additional H1 TRF, initialized at the (possibly continuation-repaired) H0
solution embedded with `piEN=piEE=0`, all six H1 parameters free, standard
tolerances. Final H1 = `min(current H1, rescue H1)`.

## Row 451245 (the headline case): is it repaired by premature-termination logic alone?

**Yes, essentially exactly.** Before -> after continuation:

| | chi2 | optimality | termination |
|---|---|---|---|
| H0 nominal | 247402.591 | 15964.75 | -- |
| H0 continuation | **1513.406** | 160.83 | `xtol` termination condition satisfied |
| H1 nominal | 334.7704 | 14.69 | -- |
| H1 continuation | 334.7704 (~unchanged) | 0.0857 | `ftol` termination condition satisfied |

H0 continuation moved the parameters by `Δt0=+92.1d, Δu0=-2.28, ΔtE=-944.2d,
Δrho=-0.904` from the bad nominal point -- large displacement confirming the
nominal fit was genuinely stuck far from its own achievable optimum, not
merely at a slightly-off point. Final `chi2_H0=1513.4064` matches the
robust-reference `h0_final_chi2=1513.4066` (diff `-0.0002`) almost exactly,
and `chi2_H1=334.7704` matches the reference `334.7704` (diff `+0.0001`).
**Caveat**: chi2 converged to (essentially) the reference value, but the
formal `optimizer_optimality` criterion did NOT drop below 0.05 even after
continuation (160.8 for H0) -- one continuation pass repairs the
practical/scientific result here, but does not generally certify strict
gradient-optimality convergence (see below).

## Rerun of both frozen samples, with safeguards

`results/two_fit_safeguards_h1generated_9events.json` (n=9, H1-generated),
`results/two_fit_safeguards_h0generated_5events.json` (n=5, H0-generated).

### 1. How many events needed same-point continuation

**Nearly all, at the 0.05 threshold -- not rare in practice.**
`continuation_h1` fired on **14/14** events (every H1 fit, both regimes).
`continuation_h0` fired on **12/14** events (all except 885956, 498910,
whose nominal H0 fit already had optimality `<0.05`). For most of these,
chi2 barely moved (e.g. row 929641 H1: `441.873509 -> 441.873358`) --
continuation is triggering on a strict optimality bar that the
`production_candidate` wide-domain fits routinely miss even when chi2 is
already stable, not only in genuinely broken cases like 451245. Even after
one continuation pass, final optimality stays well above 0.05 for most
events (e.g. row 42385: `25650` after continuation) while chi2 itself is
stable -- suggesting the 0.05 threshold (calibrated originally for
`old_final`'s narrow truth-relative bounds) may not be a well-calibrated
convergence criterion under the much wider `production_candidate` domain.
This is reported as an open question, not resolved here.

### 2. How many events needed nested H1 rescue

**4/14 (28.6%)**: rows 769381, 8681 (H1-generated) and 885956, 498910
(H0-generated) -- exactly the 4 events that had `chi2_H1 > chi2_H0` in the
nominal (pre-safeguard) run. This is a small, genuinely rare-condition
subset, triggering exactly on the diagnosed pathology and nothing else.

### 3. Average actual TRF/event, including rescues

- H1-generated (n=9): mean **4.22** TRF/event (nominal 2 + continuation on
  both fits for 7/9 events, + rescue for 2/9).
- H0-generated (n=5): mean **4.00** TRF/event.
- Combined (n=14): mean **4.14** TRF/event -- almost exactly double the
  nominal 2, dominated by continuation firing on nearly every fit rather
  than by rescue (which stayed rare, 4/14).

### 4. Is row 451245 repaired?

**Yes** -- see above. Chi2 and the LRT match the robust reference to
within `~1e-4`, via continuation alone, no new physical seed. Not repaired
in the sense of driving `optimizer_optimality` under the 0.05 threshold.

### 5. Does any final LRT remain significantly negative?

**No.** All 4 events that had a negative nominal `Delta_chi2_LRT`
(769381: -1.32, 8681: -1.76, 885956: -19.70, 498910: -15.89) are resolved
to small positive values by the nested rescue: 0.371, 1.063, 0.441, 1.412
respectively -- consistent with H1 nesting H0 (a correctly converged
comparison should never show negative LRT). `still_significantly_negative`
is `False` for every one of the 14 events in the final audit.

### 6. LRT difference vs the old robust reference (paired, n=9, H1-generated)

| row | TRF | Δχ² safeguarded | Δχ² reference | diff |
|---|---|---|---|---|
| 929641 | 4 | 0.399 | 3.804 | -3.405 |
| 835459 | 4 | 131.667 | 131.667 | -0.0001 |
| 378870 | 4 | 2.181 | 3.010 | -0.829 |
| 395956 | 4 | 1326.788 | 1024.982 | +301.806 |
| 769381 | 5 | 0.371 | 0.067 | +0.305 |
| 893531 | 4 | 23.292 | 10.145 | +13.147 |
| 42385 | 4 | 255.689 | 91.978 | +163.711 |
| 8681 | 5 | 1.063 | 1.063 | +0.0001 |
| 451245 | 4 | 1178.636 | 1178.636 | -0.0002 |

- **bias (mean diff) = 52.75**, **median diff = 0.0001** -- the median is
  now essentially exact (three events, including the former 451245
  outlier, match the reference to ~1e-4); the mean is still pulled by a
  few genuine basin-difference cases, not numerical failures.
- **p90 abs diff = 191.33, p95 abs diff = 246.57, max abs diff = 301.81**
  (row 395956).
- The remaining sizeable diffs (395956, 42385, 893531) are **not** the
  optimizer-failure pathology from before -- both H0 and H1 report
  `status=success`, and for 395956 specifically the H1 side matches the
  reference almost exactly (362.798 vs 362.798); the discrepancy is
  entirely on H0, where the single morphology-seeded start (chi2=1689.6)
  landed in a measurably worse local optimum than the robust reference's
  truth-anchored, differently-coordinatized search (chi2=1387.8, strategy
  `log_rho/truth/0.1`). This is a genuine single-start-vs-multistart basin
  gap for the (deliberately truth-blind) wrong-model fit, not a numerical
  bug -- exactly the kind of residual risk the stop rule is meant to
  surface for a scientific judgment call, not something to patch here.
- Classification at the same illustrative (non-final) threshold=9.0 is
  unchanged for all 9 events (0/9 reclassified), same as the nominal run.

### 7. Runtime, including conditional rescues

14 events, one process, physical mode, `production_candidate` bounds:
mean wall **0.94s**/event (batch: 9 events in 8.7s, 5 events in 4.4s ->
`13.1s / 14 = 0.936s` combined), roughly **1.8x** the pure-nominal 2-fit
runtime (0.53s/event) -- consistent with the ~2.07x average TRF-count
increase (4.14 vs 2). Still **~16x** faster wall-time than the old 31-TRF
robust architecture's 15.27s/event mean on the same 9 reference events.

## Conclusion

The catastrophic row-451245 failure is fully explained and repaired as
premature numerical termination, via same-point continuation alone -- no
new physical seed was needed. The negative-LRT pathology (4/14 events) is
fully resolved by the nested H1 rescue in every observed case. Both
safeguards fire on an explicit, narrow diagnostic condition, not
unconditionally -- rescue in particular stays genuinely rare (28.6%,
exactly the pathological subset). Continuation, however, fires far more
often than "rare" at the inherited 0.05 optimality threshold (14/14 and
12/14), though for most of those events it changes chi2 negligibly; this
suggests the threshold itself (inherited from a different, narrower-bounds
context) may need recalibration for `production_candidate`'s wider domain
-- an open question, not resolved or acted on here. The remaining LRT
differences against the robust reference (median ~0, but with real
outliers up to ~302) are, on inspection, genuine single-start-vs-multistart
local-optimum gaps for the truth-blind wrong-model fit, not numerical
failures -- whether that residual gap is scientifically acceptable is the
judgment call the stop rule asks for, not something resolved by this
experiment.
