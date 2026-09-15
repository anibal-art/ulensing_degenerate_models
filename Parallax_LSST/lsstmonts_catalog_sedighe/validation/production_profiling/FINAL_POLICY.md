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

## 2. Numerical safeguard policy

**Adopted**: nested-model-consistency rescue. If
`chi2_H1 > chi2_H0 + epsilon_numeric` (`epsilon_numeric=1e-6`), run exactly
one additional H1 TRF initialized at the final H0 solution embedded with
`piEN=piEE=0`, all six H1 parameters free; retain
`min(nominal_H1, rescue_H1)`. Adopted because `NUMERICAL_SAFEGUARDS.md`
found it fires on a narrow, correctly-targeted subset (4/14 = 28.6% of
that validation batch, exactly the negative-LRT cases) and fully resolves
every one (no significantly negative LRT remained in any case).

**NOT adopted**: `optimizer_optimality > 0.05` as a general continuation
trigger. `NUMERICAL_SAFEGUARDS.md` found it fires on 14/14 (H1) and 12/14
(H0) of ALL nominal fits in the same batch while usually changing chi2
negligibly -- not a genuinely exceptional condition under
`production_candidate`'s much wider domain than the narrow-bounds context
`0.05` was originally calibrated for (`old_final`'s
`polish_optimality_threshold`).

### Characterizing catastrophic false convergence (row 451245) without a generic extra fit

Neither `optimizer_optimality` nor `optimizer_nfev`/`njev` separates
row 451245's catastrophic H0 fit (`chi2=247402.6, optimality=15965,
nfev=17`) from ordinary valid fits: across the 14-event validation batch,
optimality legitimately ranges `0.001` to `32630` and `nfev` ranges `5` to
`159` on fits with completely normal chi2. Neither is usable as a
reference-free trigger without an unacceptable false-positive rate.

What DOES separate it cleanly: **reduced chi2** (`chi2 / dof`, using only
that fit's own photometry point count and parameter count -- no other
model, no reference needed). Across all 27 non-catastrophic nominal fits
inspected (9 H1-generated + 5 H0-generated events, H0 and H1 each),
reduced chi2 ranges **0.85-4.46**. Row 451245's catastrophic H0 fit has
reduced chi2 **778.0** -- three orders of magnitude beyond anything else
observed. `CHI2_DOF_SANITY_THRESHOLD=50.0` sits with >11x headroom above
the highest valid value and >15x below the one observed failure.

Re-validated on the independent 25-event basin-gap sample (`BASIN_GAP_
VALIDATION.md`): the flag fired on exactly 3/50 fits (rows 848426,
793762, 451245 -- all H0, all with chi2 in the tens-of-thousands range)
and 0 false positives among the other 47.

**Per the stop rule, this is reported as a characterized diagnostic, not
wired into an automatic extra TRF.** `final_policy.py` computes it and
attaches `sanity_flags` to every event's output for batch-level review
and logging -- it never re-fits or alters the result. This satisfies
"do not introduce a generic extra fit unless necessary": the nested H1
rescue is the only automatic extra fit in the frozen policy; catastrophic
H0/H1 failures are flagged for downstream handling (re-run, exclude, or
manual inspection), not silently patched at fit time.

## Failure-handling rule (frozen)

1. Nominal 2 fits.
2. If `chi2_H1 > chi2_H0 + 1e-6`: 1 nested H1 rescue TRF (automatic).
3. Compute `reduced_chi2` for both final H0 and H1; if either exceeds 50,
   set `sanity_flags[model].flagged = True` and log the event for
   batch-level review. Do not re-fit, do not exclude automatically from
   any output file -- flagging is advisory metadata attached to the
   result record.

Total: 2 or 3 TRF/event, no unconditional multistart anywhere.
