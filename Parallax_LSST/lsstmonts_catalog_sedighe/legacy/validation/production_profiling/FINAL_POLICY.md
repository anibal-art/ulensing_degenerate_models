# Final frozen fit policy

Implementation: `final_policy.py` (`run_final_policy`). Supersedes
`fit_two_policy.py` as the production candidate; `fit_two_policy.py` and
`numerical_safeguards.py` remain as-is (used internally / preserved as
validation history for the continuation experiment).

## 1. Nominal architecture -- 1 H0 TRF + 1 H1 TRF per event

```
H0 = FSPL, no parallax:      (t0,u0,tE,rho)
H1 = FSPL + annual parallax: (t0,u0,tE,rho,piEN,piEE)
```

| generating model | H0 fit start | H1 fit start |
|---|---|---|
| H0-generated | truth `(t0,u0,tE,rho)` | morphology seed `(t0,u0,tE,rho)`, `piEN=piEE=0` |
| H1-generated | morphology seed `(t0,u0,tE,rho)` | truth `(t0,u0,tE,rho,piEN,piEE)` |

Bounds: `production_candidate` throughout (truth-independent absolute
`u0,tE,rho,piEN,piEE`; data-driven `t0`). No `old_H0` bridge, no
`old_final`, no oracle multistart, no coordinate-mode multistart, no
Fisher/scouting -- all abandoned as production components (preserved as
validation history in `bounds_convergence/` and the earlier
`production_profiling/` commits).

## 2. Numerical safeguard policy (v2, corrected 2026-09-15)

Two AUTOMATIC, conditional safeguards are adopted -- both fire only on an
explicit per-fit diagnostic, neither is unconditional multistart:

### Safeguard 1 (adopted): same-point continuation on catastrophic false convergence

Trigger: reduced chi2 (`chi2/dof`, using only that fit's own photometry
point count and parameter count -- no reference/other model needed)
`> CHI2_DOF_SANITY_THRESHOLD = 50.0`. Neither `optimizer_optimality` nor
`optimizer_nfev`/`njev` separates catastrophic false convergence (row
451245: `chi2=247402.6, optimality=15965, nfev=17`) from ordinary valid
fits -- across 74 fits inspected across three independent samples,
optimality legitimately ranges `0.001-32630` and nfev `5-159` on fits with
completely normal chi2. Reduced chi2 does separate cleanly: **0.75-4.46**
across every non-catastrophic fit inspected, vs `778.0` (row 451245,
before continuation) and other catastrophic cases found since -- 3
orders of magnitude clear.

When triggered: run exactly ONE same-point continuation (same model,
same objective, same `production_candidate` bounds, all parameters free,
stricter tolerances, no new physical seed); keep if it does not worsen
chi2. This is the same mechanism validated in `NUMERICAL_SAFEGUARDS.md`
that repaired row 451245 without any new physical seed.

**Measured trigger rate and cost**: on the 25-event basin-gap sample
(`results/final_policy_v2_h1generated_25events.json`), triggered on
**3/25 = 12%**, always on H0, never H1; average TRF/event rises 2.16 ->
2.28. **2/3 fully repaired** (451245: `chi2 247402.6 -> 1513.4`, matching
the robust reference to ~1e-4; 848426: `chi2 30784.5 -> 74.9`); **1/3
(row 793762) stayed flagged** (`chi2 18942.90 -> 18942.89`, displacement
~0 -- a genuine local-minimum trap, not premature termination). On the
larger, independent 100-event basin-validation sample
(`results/final_policy_h1basinvalidation_100events.json`,
`BASIN_GAP_VALIDATION.md`): triggered on **13/100 = 13%** (always H0,
consistent with the smaller sample), and **7/100 = 7% remained flagged
after continuation** -- i.e. continuation resolves roughly half of
triggered cases, the rest are genuine local-minimum traps that stay
flagged for downstream review, not silently accepted. Nested rescue did
not fire at all in this particular 100-event sample (0/100) -- sample
variance, consistent with its 16-58%-depending-on-regime range measured
elsewhere.

**NOT adopted**: `optimizer_optimality > 0.05` as a general continuation
trigger (`NUMERICAL_SAFEGUARDS.md`: fires on 14/14 H1 and 12/14 H0 of ALL
nominal fits while usually changing chi2 negligibly -- not exceptional
under `production_candidate`'s much wider domain than the narrow-bounds
context `0.05` was calibrated for).

### Safeguard 2 (adopted): nested-model-consistency rescue

Runs AFTER safeguard 1 settles (using whatever chi2 that leaves). Trigger:
`chi2_H1 > chi2_H0 + epsilon_numeric` (`epsilon_numeric=1e-6`). One
additional H1 TRF initialized at the final H0 solution embedded with
`piEN=piEE=0`, all six H1 parameters free; retain
`min(current_H1, rescue_H1)`. Fires on 16% (H1-generated,
`PRODUCTION_PROFILING_FINAL.md`) to 58% (H0-generated,
`H0_CALIBRATION.md`) of events depending on regime and fully resolves
every negative-LRT case observed in both regimes.

## Failure-handling rule (frozen, applies identically to H0-calibration and H1-production events)

1. Nominal 2 fits (truth-start for the matching model, morphology-seed
   start for the other).
2. **If either fit's reduced chi2 > 50**: run exactly 1 same-point
   continuation for that fit (automatic). Keep if it does not worsen
   chi2.
3. **If `chi2_H1 > chi2_H0 + 1e-6`** (after step 2): run exactly 1 nested
   H1 rescue TRF (automatic). Retain the minimum.
4. Recompute `sanity_flags` on the settled H0/H1. **If still flagged
   after step 2** (e.g. row 793762): the event is NOT silently accepted
   as a normal result -- it is logged with `sanity_flags[model].flagged
   = True` for batch-level review (exclude / manual re-run / accept with
   caveat is a downstream, per-population decision, not made
   automatically here). This is the only case requiring any human
   attention; it does not block automatic pipeline completion.
5. **If `morphology_seed_from_curves` returns `None`** (not measurable):
   the affected fit is skipped -- never given a fallback value, never
   given truth. The event is recorded with
   `estimator_failure="morphology_not_measurable"`, excluded from any
   `Delta_chi2_LRT`/classification statistic (there is no chi2 to
   compare), but counted in the total processed population for an
   explicit failure-rate metric (measured: 1/60 = 1.7%,
   `H0_CALIBRATION.md`). Identical rule, identical code path, whether the
   event is part of H0 calibration or H1 production -- no special-casing.

Total: 2-4 TRF/event depending on which safeguards fire; no unconditional
multistart anywhere; no step in this list requires manual intervention to
complete the pipeline for a given event (step 4's flag is advisory,
attached to the output, not a blocking gate).
