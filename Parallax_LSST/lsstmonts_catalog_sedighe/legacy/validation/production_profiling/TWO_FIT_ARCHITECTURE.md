# Simplified 2-fit production architecture

Replaces, as the production candidate, everything previously explored under
`validation/bounds_convergence/` and the earlier `validation/production_profiling/`
work (`old_H0` bridge, `old_final` legacy producer, oracle-derived multistart
sets, truth-relative bounds applied to another model's seed, morphology
multistart families, Fisher/scouting/hybrid-start machinery). Those remain
preserved as validation history; none of them are part of this candidate.

## 1. Exact H0/H1 initialization matrix

```
H0 = FSPL, no parallax:      (t0,u0,tE,rho)
H1 = FSPL + annual parallax: (t0,u0,tE,rho,piEN,piEE)
```

| generating model | H0 fit start | H1 fit start | TRF count |
|---|---|---|---|
| H0-generated | truth `(t0,u0,tE,rho)` | morphology seed `(t0,u0,tE,rho)`, `piEN=piEE=0` | 1 + 1 |
| H1-generated | morphology seed `(t0,u0,tE,rho)` | truth `(t0,u0,tE,rho,piEN,piEE)` | 1 + 1 |

Truth is initialization only for the model that matches the generating model.
It never seeds, even partially, the other model. No hidden prerequisite fit,
no bridge, no conditional rescue, no multistart. Implemented in
`fit_two_policy.run_two_fit_policy`.

Bounds: `run_bounds_audit_refit_core`'s existing `production_candidate`
profile throughout -- `u0∈[-10,10]`, `tE∈[0.1,500000]`, `rho∈[1e-7,10]`,
`piEN,piEE∈[-40,40]` (fixed, truth-independent absolute intervals), `t0`
data-driven from the photometry time range (not truth-centered, not
morphology-centered). The same domain is used whichever model's start comes
from truth and whichever comes from morphology, and it is never re-centered
on the start -- satisfying "one consistent production bounds definition...
not centered on injected truth when fitting the wrong model."

## 2. Proof the morphology seed uses no truth

`morphology_seed.py`'s `morphology_seed_from_curves(curves)` has the single
positional parameter `curves` -- no truth/event_params argument exists in
its signature, so no injected truth value can enter it even by accident.
`smoke_test_two_fit_matrix.py` asserts this structurally via
`inspect.signature` before running anything (not just asserted in prose):

```
[proof] morphology_seed_from_curves signature = (curves) -- no truth/event_params argument exists.
```

It reuses the already-frozen morphology extraction (`extract_event_morphology`:
peak time, T25/T50/T75 excess-flux widths, asymmetry, coverage) and the
already-frozen FSPL dimensionless `(u0,rho)` lookup table
(`validation/bounds_convergence/results/fspl_dimensionless_lookup.csv`),
unchanged -- a deterministic rank-1 nearest match, identical code path for
H0-generated and H1-generated curves. If morphology is not measurable
(insufficient peak coverage), the seed is `None` and this is treated as a
genuine estimator/domain inconsistency (the affected fit is skipped and
`estimator_failure="morphology_not_measurable"` is recorded), never
silently substituted.

## 3. One-event smoke test, all 4 combinations

`smoke_test_two_fit_matrix.py`, run on one freshly-materialized H0-generated
event (catalog_row 813025, via `truth_parallax=False` -- a genuine
no-parallax simulation, not a truth override; `pyLIMA_parameters` has no
piEN/piEE at all) and one H1-generated event (catalog_row 929641, from the
existing profiling sample):

| combo | row | fit | start | chi2 | status |
|---|---|---|---|---|---|
| (a) H0-gen -> H0 | 813025 | H0 | truth | 236.696 | success |
| (b) H0-gen -> H1 | 813025 | H1 | morphology, piE=0 | 231.715 | success |
| (c) H1-gen -> H1 | 929641 | H1 | truth | 441.874 | success |
| (d) H1-gen -> H0 | 929641 | H0 | morphology | 442.273 | success |

Domain-containment check: 0 violations in both events (morphology seed fell
inside `production_candidate` bounds by construction, as expected).

## 4. Optimizer count

Both smoke-test events: `n_trf = 2` (asserted in code, not just printed).
Across the full 14-event validation batch (9 H1-generated + 5 H0-generated),
every event used exactly 2 TRF calls -- confirmed via
`results/two_fit_h1generated_9events.json` and
`results/two_fit_h0generated_5events.json`.

## 5. Small independent LRT comparison

Reference = the previously-frozen robust/multistart architecture (16-cost
H0 search + `old_final`+`controlled5` H1, 31 TRF/event), run on the 9
profiling-sample events confirmed executable by the feasibility audit
(`FEASIBILITY_AUDIT.md`) -- the only pre-existing robust reference available
in this project's history, and paired on the SAME 9 events
(`results/reference_robust_9events.json`). All H1-generated (real catalog
piE); no comparable robust reference exists for H0-generated events in this
project's history (the whole prior effort targeted real-parallax
detection), so that regime is reported on its own terms (§5b).

### 5a. H1-generated regime (n=9, paired)

`Delta_chi2_LRT_simple = chi2_H0 - chi2_H1` (2-fit policy) vs
`Delta_chi2_LRT_reference = h0_final_chi2 - h1_final_chi2` (robust,
`results/lrt_comparison_h1generated_9events.csv`):

| catalog_row | simple | reference | diff |
|---|---|---|---|
| 929641 | 0.399 | 3.804 | -3.405 |
| 835459 | 131.641 | 131.667 | -0.025 |
| 378870 | 2.173 | 3.010 | -0.837 |
| 395956 | 1326.799 | 1024.982 | 301.817 |
| 769381 | -1.317 | 0.067 | -1.383 |
| 893531 | 23.291 | 10.145 | 13.146 |
| 42385 | 255.686 | 91.978 | 163.708 |
| 8681 | -1.758 | 1.063 | -2.821 |
| **451245** | **247067.821** | **1178.636** | **245889.185** |

- **bias (mean diff) = 27373.26**, dominated entirely by row 451245.
- **median diff = -0.026** -- essentially zero; the typical-case agreement
  between the simplified and robust architectures is excellent.
- **p90 abs diff = 49419.29, p95 abs diff = 147654.24** -- both dominated by
  the single row-451245 outlier (n=9 is too small for these percentiles to
  be meaningful on their own; reported as requested, but see the outlier
  analysis below for the actual driver).
- **fraction reclassified @ illustrative threshold=9.0 (chi2(2)-like,
  NOT the final production threshold -- see §7): 0/9 = 0%.** Every event's
  H0-vs-H1 classification is unchanged between the two architectures in
  this sample, including row 451245 (both call it overwhelmingly H1).

**Catastrophic outlier -- row 451245.** The simplified policy's H0 fit
(morphology-seeded) landed at `chi2=247402.6` against a robust
`h0_final_chi2=1513.4` (~163x worse). Diagnostic
(`results/two_fit_h1generated_9events.json`): `optimizer_success=True` but
`optimizer_optimality=15964.7` -- two to four orders of magnitude above
every other event's optimality and far above the historical
`polish_optimality_threshold=0.05` used elsewhere in this project. This is
a **false-convergence failure**: TRF reported nominal success (satisfied an
internal step-size/cost tolerance) while the gradient-based optimality
criterion shows it did not reach a stationary point. `active_mask=[0,0,0,0]`
rules out a simple boundary lock; the morphology seed for this event
(`tE=2.04d`) was far from the fitted result (`tE=1019.9d`), i.e. a
poorly-conditioned single-start landscape. It did not flip this event's
classification only because both chi2 values are far above any plausible
threshold -- for a marginal event nearer the threshold, this failure mode
could change the scientific conclusion. This is the central risk the
"no multistart" simplification introduces: a single start has no built-in
redundancy against landing in a bad local optimum, unlike the robust
architecture's own multiple starts.

### 5b. H0-generated regime (n=5, null-injection distribution)

No robust reference exists; reported as the Delta_chi2_LRT distribution
itself, the quantity that matters for calibrating the null
(`results/lrt_h0generated_5events.csv`):

| catalog_row | Delta_chi2_LRT | chi2_H0 | chi2_H1 |
|---|---|---|---|
| 813025 | 4.981 | 236.696 | 231.715 |
| 885956 | -19.699 | 335.724 | 355.423 |
| 699469 | 8.976 | 320.492 | 311.516 |
| 532443 | 3.411 | 227.690 | 224.279 |
| 498910 | -15.895 | 116.781 | 132.676 |

mean = -3.65, median = 3.41, range = [-19.70, 8.98]. All 5 `optimizer_success
= True` on both fits; no domain violations. 0/5 false positives at the
illustrative threshold=9.0. Two of five events show a *negative*
`Delta_chi2_LRT` (H1's nested piE=0 fit converged to a slightly worse chi2
than H0's own truth-anchored fit) -- since H1 nests H0 exactly at
`piEN=piEE=0`, a perfect optimizer could never produce a negative LRT; this
is the same single-start local-optimum risk as §5a, at a smaller scale here
because these are well-behaved null events, not a resolved defect.
**n=5 is a smoke-scale sample, not a calibration sample** -- it demonstrates
the mechanism works and is roughly the right order of magnitude, nothing
more; the actual H0-calibration sample (§7) is a separate, later decision.

## 6. Runtime profile

14 events (9 H1-generated + 5 H0-generated), one process, physical
coordinate mode, `production_candidate` bounds (`run_two_fit_batch.py`):

| | wall (s) | CPU (s) |
|---|---|---|
| mean | 0.529 | 0.449 |
| median | 0.472 | 0.392 |
| p90 | 0.808 | 0.729 |
| max | 0.866 | 0.789 |

Compared to the same 9 events under the old 31-TRF robust architecture
(`results/reference_robust_9events.json`): mean wall 15.27s, mean CPU
31.39s. **~29x wall speedup, ~70x CPU speedup** (the CPU gap is larger
because the old architecture pays Python-import overhead 4 separate times
per event, once per H0 coordinate-mode subprocess; the new architecture
needs only one mode, one process, for both fits).

**Preliminary serial-throughput projection** (naive extrapolation from
these 14 measured events -- NOT a measured parallel-throughput sweep; P1/P2
benchmarking is deferred until requested, per the earlier stop rule, and
would need a much larger sample for reliable p90/p95 given how heavy-tailed
single-TRF wall time can be, per the §5a outlier):

- serial, single core: `1e6 x 0.529s ≈ 147 hours ≈ 6.1 days`.
- at 16 cores (naive linear scaling, unverified): `≈ 9.2 hours`.
- both comfortably inside the `1e6/(7*24) ≈ 5952 events/hour` 1-week
  target even before any real parallel-efficiency measurement -- unlike the
  old architecture, which the feasibility audit showed could not even
  complete 64% of events.

## 7. Scope notes

- Development/reference fitting (`validation/bounds_convergence/`,
  `validation/bounds_audit/`), the H0 null-calibration sample, and the
  1M-event Sedighe population production run remain three separate,
  distinct things. Nothing here launches or sizes any of them; the
  H0-calibration sample size and the actual production LRT threshold are
  separate decisions driven by alpha/tail-precision requirements, not
  touched by this validation.
- No bounds, starts, or selection policy were changed to produce any of
  the above numbers. No fix for the row-451245 failure mode is proposed or
  implemented here, per the stop rule.
