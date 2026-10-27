# Hidden-parallax fitter validation status

**Date:** 2026-09-14

This document records the current validation state of the H0/H1 fitting
pipeline used for the LSST hidden-parallax experiment.

The purpose is to make the current scientific and numerical choices
reproducible without relying on conversation history.

---

## 1. Scientific problem

We compare two finite-source microlensing models:

- **H0:** FSPL without annual parallax
- **H1:** FSPL with annual parallax

with nonlinear parameters

\[
H0 = (t_0,u_0,t_E,\rho)
\]

and

\[
H1 =
(t_0,u_0,t_E,\rho,\pi_{E,N},\pi_{E,E}).
\]

The likelihood-ratio statistic is

\[
\Delta\chi^2
=
\chi^2_{H0}
-
\chi^2_{H1}.
\]

Both models use the same finite-source treatment and the same
box-constrained weighted least-squares profiling of the flux parameters.

The validation has three distinct goals:

1. **Detection:** determine whether annual parallax is distinguishable
   from H0.
2. **Hidden-parallax / confusion:** determine whether H0 can provide an
   acceptable fit while biasing inferred microlensing parameters.
3. **Characterization:** determine how accurately the H1 parameters and
   uncertainties are recovered.

Detection and characterization are intentionally treated as different
questions.

---

## 2. Important historical issue

The original production fits used truth-dependent and relatively narrow
bounds.

In particular, the old H1 parallax limits were approximately

\[
\pi_{E,N}\in[-1.9973,1.9973],
\qquad
\pi_{E,E}\in[-2.201096,2.201096].
\]

These limits exclude part of the simulated population and therefore the
old H1 fits cannot be used as final scientific measurements.

Old fits may still be used as **initial guesses / warm starts**, provided
that the fit is rerun with the new validated domain.

The old production chi2 values are diagnostic only.

---

## 3. Current candidate parameter domain

The current profile used for the production-candidate validation is

`production_candidate`.

The shared H0/H1 nuisance-parameter bounds are

\[
u_0\in[-10,10],
\]

\[
t_E\in[0.1,500000]\ {\rm d},
\]

\[
\rho\in[10^{-7},10].
\]

For H1,

\[
\pi_{E,N}\in[-40,40],
\qquad
\pi_{E,E}\in[-40,40].
\]

The ±40 parallax domain was chosen to safely contain the support of the
full simulated catalogue, whose individual parallax components reach
approximately +32.

### t0 domain

The default domain is data driven:

\[
t_0\in[t_{\min},t_{\max}],
\]

where the limits come from the finite Rubin time series for the event.

A global enlargement of this interval was tested and rejected because
it changed optimizer trajectories even for events whose final solution
remained inside the original interval.

The final Gate-1 policy is therefore adaptive:

1. fit with the base interval;
2. if the winning H0 solution has active `t0`,
   rerun the event using

\[
[t_{\min}-0.25T_{\rm obs},
 t_{\max}+0.25T_{\rm obs}];
\]

3. retain the better solution.

In the `extreme100` validation sample this rescue was triggered only for
catalog row `36103`.

For an LRT, H0 and H1 must use the same shared nuisance-parameter domain.
The Gate-2 H1 runs therefore read the event-specific `t0_margin_factor`
from the final H0 anchor manifest.

---

# GATE 1 — H0 validation

## 4. Gate-1 oracle

The broad H0 validation used four TRF coordinate systems:

- `physical`
- `log_te`
- `log_rho`
- `log_te_rho`

with 13 starts per coordinate system.

This gives a 52-fit validation oracle per event.

The coordinate transforms affect only the internal TRF coordinates.
Physical bounds, objective function, flux profiling, and reported
physical parameters remain unchanged.

No individual coordinate system reliably reproduced the four-mode
oracle.

Examples of large disagreements were found for catalog rows such as

- `62786`
- `71075`
- `82728`
- `578896`
- `52609`
- `577615`

which demonstrated that optimizer-basin effects are important for H0.

---

## 5. H0 bounds validation

Using the four-mode oracle on `extreme100`, the winning H0 parameters
were comfortably inside

\[
u_0\in[-10,10],
\quad
t_E\in[0.1,500000],
\quad
\rho\in[10^{-7},10].
\]

The hardest event in the previous tE-bound analysis was row `577615`.

Its H0 solution stabilized only after allowing very large timescales:

- `tE_max = 5000 d`: boundary dominated
- `tE_max = 20000 d`: boundary dominated
- `tE_max = 50000 d`: boundary dominated
- `tE_max = 100000 d`: boundary dominated
- `tE_max = 500000 d`: interior solution
- `tE_max = 1000000 d`: consistent with the 500000 d result to much
  better than the adopted chi2 tolerance

The winning `extreme100` H0 solution for this event has approximately

\[
t_E \simeq 1.9\times10^5\ {\rm d},
\]

so the 500000 d limit is intentionally very broad.

---

## 6. Final fixed H0 start set found in Gate 1

An exact set-cover analysis was performed over the 52
`(coordinate mode, start)` strategies.

The strict requirement was

\[
\chi^2_{\rm selected}
-
\chi^2_{\rm 52-fit\ oracle}
\leq0.1
\]

for every event in `extreme100`.

A fixed set of 18 strategies was sufficient.

The implemented `gate1_final18` semantic plan contains

- 3 `physical` starts
- 6 `log_te` starts
- 8 `log_rho` starts
- 1 `log_te_rho` start

for 18 H0 optimizations per ordinary event.

The adaptive t0 rescue reruns the same plan only if the winning H0
solution has active `t0`.

For `extreme100`, only row `36103` required this rescue.

The final adaptive result reproduced the Gate-1 reference oracle with

- `N(delta chi2 > 0.1) = 0`
- maximum delta chi2 approximately `0.02714`

and all final winning H0 solutions were interior to the final parameter
domain.

The exact 18-strategy solution is not unique. A later maximum-coverage
MILP found another 18-strategy set with the same strict coverage.
Therefore the robust result is the **required cardinality under this
fixed-strategy criterion**, not a claim that one particular list of 18
starts is uniquely preferred.

---

## 7. Can H0 use only 1–3 starts?

An exhaustive ablation over the already-computed 52-fit oracle gave:

| H0 fits | Best fixed strategy result | N(delta>0.1) | N(delta>1) | N(delta>10) | max delta |
|---:|---|---:|---:|---:|---:|
| 1 | best single strategy | 51 | 37 | 24 | 120955 |
| 2 | best pair | 37 | 24 | 14 | 24837 |
| 3 | best triplet | 27 | 18 | 7 | 11778 |

Therefore 2–3 fixed H0 fits are not sufficiently reliable for the LRT.

This is especially important because an upward error in H0 chi2 enters
directly as an artificial increase in

\[
\Delta\chi^2_{\rm LRT}.
\]

A poor H0 minimum can therefore produce a false parallax detection.

---

## 8. H0 fixed-strategy coverage curve

A maximum-coverage MILP was used to determine the best possible fixed
coverage at tolerance

\[
\Delta\chi^2\le0.1
\]

for each number of H0 strategies.

The number of failures was

| k | N(delta>0.1) |
|---:|---:|
| 1 | 51 |
| 2 | 37 |
| 3 | 27 |
| 4 | 21 |
| 5 | 17 |
| 6 | 14 |
| 7 | 12 |
| 8 | 10 |
| 9 | 9 |
| 10 | 8 |
| 11 | 7 |
| 12 | 6 |
| 13 | 5 |
| 14 | 4 |
| 15 | 3 |
| 16 | 2 |
| 17 | 1 |
| 18 | 0 |

Thus no small fixed multistart set reproduces the Gate-1 oracle at the
adopted tolerance.

This does **not** establish that production must always execute 18 H0
fits.

The next planned H0 optimization step is to investigate a sequential /
adaptive strategy:

- execute a small base set;
- use information available from those fits to detect unstable events;
- execute additional rescue starts only when needed.

No such trigger has yet been validated.

Optimizer `success`, active bounds, or optimality alone must not be
assumed to identify wrong basins: previous Gate-1 tests showed that
incorrect local minima can be interior and reported as successful.

---

# GATE 2 — H1 validation

## 9. Controlled H1 experiment

This is a controlled simulation.

The generating parameters are known, and using truth-assisted
initialization is intentional.

This is not intended to represent a blind observational-discovery
pipeline.

The H1 tests use

`production_candidate`

with

\[
u_0\in[-10,10],
\quad
t_E\in[0.1,500000],
\quad
\rho\in[10^{-7},10],
\]

and

\[
\pi_{E,N},\pi_{E,E}\in[-40,40].
\]

---

## 10. Initial controlled4 strategy

The first reduced H1 multistart strategy used four starts:

1. `truth`
2. `H0_NESTED_piE_0`
3. `truth_half_piE`
4. `truth_mirror_u0_piEN`

The definitions are:

### truth

Exact generating nonlinear parameters.

### H0_NESTED_piE_0

The final validated H0 nonlinear solution with

\[
\pi_{E,N}=\pi_{E,E}=0.
\]

### truth_half_piE

Truth parameters except

\[
\boldsymbol{\pi}_{E,\rm init}
=
0.5\boldsymbol{\pi}_{E,\rm true}.
\]

### truth_mirror_u0_piEN

Truth parameters with

\[
u_0\rightarrow-u_0,
\qquad
\pi_{E,N}\rightarrow-\pi_{E,N},
\]

while keeping the remaining truth parameters unchanged.

All four are rerun with the new broad production-candidate bounds.

---

## 11. controlled4 extreme100 result

All 100 `extreme100` events were fitted with the four H1 starts using
the physical parameterization.

The four starts all found useful minima in different events.

A comparison against historical H1 results showed three clear cases
where the old final solution pointed to a better basin than
`controlled4`:

- row `79700`
- row `85380`
- row `87786`

The old H1 solution was therefore tested as an additional **initial
guess only**.

It is not accepted as a final fit without rerunning the new fitter.

---

## 12. old_final_reseed

The fifth H1 start is

`old_final_reseed`.

It uses the nonlinear solution from the historical final H1 fit as the
initial point, but reruns the current fitter using the broad new bounds.

For the three initially problematic events it recovered or improved the
old basin:

### row 79700

`controlled4` best:

\[
\chi^2 \simeq133.91894
\]

`old_final_reseed`:

\[
\chi^2 \simeq132.94087.
\]

### row 85380

`controlled4` best:

\[
\chi^2 \simeq95.49552
\]

`old_final_reseed`:

\[
\chi^2 \simeq91.98922.
\]

### row 87786

`controlled4` best:

\[
\chi^2 \simeq269.34380
\]

`old_final_reseed`:

\[
\chi^2 \simeq266.73513.
\]

The `old_final_reseed` start was subsequently run once for all 100
events so that the five-start H1 ablation could be performed without
rerunning the original four fits.

---

## 13. Five-start H1 oracle used for the current ablation

The current H1 validation reference is the minimum over

1. `truth`
2. `H0_NESTED_piE_0`
3. `truth_half_piE`
4. `truth_mirror_u0_piEN`
5. `old_final_reseed`

plus the exact already-computed H0 point when evaluating mathematical
nestedness.

Among the five H1 optimizations, the oracle winners in `extreme100`
were:

- `H0_NESTED_piE_0`: 30
- `old_final_reseed`: 25
- `truth_mirror_u0_piEN`: 17
- `truth`: 15
- `truth_half_piE`: 13

Thus each of the five tested H1 starts contributes useful minima for
some events.

---

## 14. H1 start ablation

The best fixed strategies found for each number of H1 fits were:

| H1 fits | N(delta>0.1) | N(delta>1) | max delta |
|---:|---:|---:|---:|
| 1 | 48 | 26 | 195.9 |
| 2 | 19 | 9 | 4.25 |
| 3 | 10 | 5 | 3.66 |
| 4 | 4 | 3 | 3.66 |
| 5 | 0 | 0 | 0 |

The best four-fit subset excludes `truth`, but fails the 0.1 criterion
for four events:

- `35101`: delta approximately 3.659
- `81449`: delta approximately 1.311
- `579071`: delta approximately 0.215
- `579320`: delta approximately 2.873

In all four cases `truth` is the five-start oracle winner.

Therefore, under the current strict fixed-strategy criterion, all five
H1 starts are required on `extreme100`.

This is still only five H1 optimizations per event, rather than the
earlier experimental strategy that scaled the number of fits with the
number of historical piE basins.

---

## 15. Nestedness

Mathematically,

\[
H0\subset H1
\]

because H0 corresponds to

\[
\pi_{E,N}=\pi_{E,E}=0.
\]

The exact final H0 solution should therefore always be considered an
allowed H1 point.

For scientific LRT construction the final H1 chi2 must satisfy

\[
\chi^2_{H1}
\leq
\chi^2_{H0}.
\]

The H0 nested point should be available explicitly rather than relying
on a TRF optimization started at H0 to return exactly the same objective
value.

The raw H1 optimizer result must nevertheless still be monitored.
Taking the minimum with the exact H0 point must not be used to hide
systematic optimizer failures.

A production implementation should therefore store both

- raw best H1 optimized chi2;
- exact embedded-H0 chi2;

and report whether the raw optimizer itself satisfies nestedness within
the adopted numerical tolerance.

---

## 16. Important interpretation of old H1 fits

The old H1 fits were obtained with invalidly narrow parallax bounds and
must not be used as final measurements.

`old_final_reseed` is acceptable in the current controlled validation
only because:

1. it supplies an initial point;
2. the current optimizer is rerun;
3. the current broad bounds are used;
4. the resulting new fit, not the historical chi2, is evaluated.

A future self-contained fitter for entirely new events should not
depend on an old production result. If needed, this warm start will
eventually need to be replaced by an internally generated
initialization rule.

---

# CURRENT STATUS

## 17. What is currently validated

The following points have strong support from the `extreme100`
validation:

- broad shared domain:
  - `u0 = [-10,10]`
  - `tE = [0.1,500000] d`
  - `rho = [1e-7,10]`
- H1 parallax candidate domain:
  - `piEN = [-40,40]`
  - `piEE = [-40,40]`
- base data-driven t0 interval;
- adaptive `0.25*Tobs` t0 rescue for an active H0 t0 bound;
- the H0 problem has serious optimizer-basin structure;
- 2–3 fixed H0 starts are not adequate;
- 18 fixed H0 strategies reproduce the 52-fit Gate-1 oracle within
  delta chi2 <= 0.1 for `extreme100`;
- the reduced H1 problem can be handled with a small fixed multistart;
- among the five currently tested H1 starts, all five are needed to
  reproduce the current five-start oracle at delta chi2 <= 0.1 on
  `extreme100`.

---

## 18. What is NOT yet frozen

The following decisions are still under validation:

- whether H0 can use a substantially cheaper **adaptive** strategy
  instead of 18 fixed fits;
- whether an observable trigger based on the first few H0 fits can
  reliably identify events requiring rescue;
- whether H1 needs any additional coordinate-system rescue beyond the
  current physical-coordinate five-start strategy;
- symmetric t0-domain handling if H1 itself reaches a t0 boundary;
- final production representation of the exact embedded H0 candidate;
- final uncertainty / covariance / multimodality treatment for H1;
- empirical H0 LRT threshold calibration.

The empirical null-LRT calibration must be performed only after the
fitter and configuration are frozen.

---

# KEY REPOSITORY FILES

## 19. Main fitter

- `validation/bounds_audit/run_bounds_audit_refit_core.py`
- `validation/bounds_audit/run_bounds_audit_refit.py`

The core currently contains validation switches for

- bounds profiles;
- TRF coordinate parameterization;
- t0 margin;
- H0 start plans;
- H1 start plans;
- exact H0 anchor manifest.

---

## 20. Gate-1 analysis files

Important Gate-1 analysis/results include

- `results/gate1_all_start_fits.csv`
- `results/gate1_minimum_start_set.csv`
- `results/gate1_minimum_start_set_per_event.csv`
- `results/gate1_final_adaptive_t0.csv`
- `results/gate1_h0_small_start_ablation.csv`
- `results/gate1_h0_coverage_curve.csv`

and the corresponding analysis scripts in
`validation/bounds_convergence/`.

---

## 21. Gate-2 files

Important Gate-2 files include

- `data/gate2_h0_anchor_manifest_extreme100.csv`
- `results/gate2_h1_dryrun_counts.csv`
- `results/gate2_start_ablation_per_event.csv`
- `results/gate2_start_ablation_summary.csv`
- `analyze_gate2_start_ablation.py`
- `build_gate2_h0_anchor_manifest.py`

Runtime fit directories under `~/Downloads/...` are validation working
directories and are not the canonical repository record.

The important derived results must therefore be stored as repository
CSVs and documented here.

---

# NEXT STEP

## 22. Immediate next validation task

Before changing the final production pipeline, investigate whether the
18-fit H0 strategy can be replaced by a cheaper **adaptive sequential
strategy**.

The first proposed diagnostic is to determine whether the spread among
the first few H0 minima,

\[
\Delta_{\rm spread}
=
\chi^2_{\rm second\ best}
-
\chi^2_{\rm best},
\]

or another quantity available from the initial fits can predict which
events remain more than 0.1 above the Gate-1 oracle.

No conclusion about such a trigger has yet been established.

Do not remove or reduce the current 18-fit H0 validation strategy until
that analysis is complete.

---

## 23. Reproducibility note

The current validation is intentionally truth-assisted because it is a
controlled simulation study.

This must be stated explicitly in any later scientific description.

The final production fitter used for this simulation does not need to
behave like a blind fitting pipeline for real observational data, but
the use of truth information must remain transparent.
