# Production-feasibility audit of the frozen `old_final` H1 legacy producer

Status: **STOP before P1/P2 throughput profiling.** Full event-level and
within-event parallel-throughput profiling (originally planned as the next
step after the serial baseline) is deferred pending resolution of the
finding below. This is a feasibility finding, not a fitter retune — no
bounds, starts, or selection policy were changed to produce it.

## Trigger

The serial baseline (5 events, `results/serial_baseline_5events.json`)
crashed in Stage 3 (H1) on 3/5 events (412859, 815061, 609457), all with
the same uncaught `ValueError` from `fit_lc.validate_initial_guess_inside_bounds`
on the `rho` parameter, before any H1 TRF ran. This audit checks whether
that 60% rate was representative or an unlucky draw, using only the
minimum diagnostic needed (no H1 TRF at all).

## Method

For each of the 25 frozen profiling-sample events:
1. load/setup,
2. the frozen `old_H0` bridge exactly as defined (1 real TRF; reused from
   an existing `results/pipeline_stage_json/{row}_bridge.json` for the 5
   events that already had one from the serial baseline / smoke test — 20
   fresh bridge TRFs run here),
3. resolve the exact `old_final` H1 bound specs
   (`configs/validation/lrt_benchmark/FAST.json`'s H1 block: `t0`/`u0`
   center-width around truth, `tE`/`rho` "relative" type with `frac=1.0`)
   using `fit_lc.resolve_bound_spec` itself — not a reimplementation,
   centered on truth (matching `fit_lc.apply_custom_bounds` /
   `_center_for_parameter`, which centers on `fit_params` = the event's
   true parameters in this pipeline),
4. check the bridge vector against those bounds directly (equivalent to
   `validate_initial_guess_inside_bounds`, but checking every parameter
   independently instead of stopping at the first failure, so partial
   failure classes are visible).

Zero H1 TRF calls were run. Script: `audit_old_final_feasibility.py`.
Raw per-event output: `results/old_final_feasibility_audit_25events.csv`.

## Result

| | |
|---|---|
| N audited | 25 |
| Executable (PASS) | 9/25 = **36.0%** |
| Would crash before any `old_final` TRF (FAIL) | 16/25 = **64.0%** |

Failure-class breakdown (of 25):
- PASS: 9
- `rho_only` invalid: 14
- `both_tE_rho` invalid: 2
- `tE_only` invalid: 0
- any other parameter (`t0`, `u0`) invalid: 0 (`t0`/`u0` bridge bounds are
  identical to the `old_final` bridge-centered bounds by construction, so
  they can never be the cause)

`bridge_rho / truth_rho` ratio distribution:
- FAIL group (n=16): median 575.8x, min 18.9x, max 42,411x — bridge `rho`
  always **overshoots** truth, often by 2-4 orders of magnitude.
- PASS group (n=9): median 0.0086x, min 7.4e-5x, max 0.48x — bridge `rho`
  always stays at or below truth.

The separation is essentially clean: since the truth-relative `rho` bound
is `[1e-7, 2×truth_rho]`, failure requires only `bridge_rho > ~2×truth_rho`
— and this happened in 16/25 sampled events, at magnitudes far beyond the
threshold (median 575x over).

`bridge_tE / truth_tE` stays close to 1 in nearly every event (0.37x-1.78x
range) except the 2 `both_tE_rho` cases (2.94x, 3.39x, against a `[0.1,
2×truth_tE]` bound) — `tE` divergence is rare and always co-occurs with
severe `rho` divergence, never occurs alone.

**Was the 3/5 (60%) serial-baseline crash rate representative?** Yes —
64.0% on the full N=25 audit is consistent with it, if anything slightly
higher. Not an unlucky draw.

## Minimal architectural inconsistency identified

The frozen `old_final` legacy producer combines two conventions that are
individually reasonable but jointly inconsistent:

1. **Initialization convention**: every grid candidate's `(t0,u0,tE,rho)`
   is embedded at the `old_H0` bridge's own best-fit estimate — an
   independent, truth-blind estimate from fitting the misspecified
   (no-parallax) FSPL model under wide bounds (`rho∈[1e-7,10]`).
2. **Bound convention**: the same candidates' `tE`/`rho` bounds are
   resolved as `"relative"` (`fit_lc.resolve_bound_spec`), which centers
   on **truth**, not on the initial guess — giving a narrow
   `[1e-7, 2×truth_rho]` band that assumes the starting point is already
   close to truth.

Nothing enforces agreement between (1) and (2). The well-known FSPL
finite-source/blend-flux degeneracy routinely lets an unparallaxed H0 fit
inflate `rho` to (partially) absorb an unmodeled long-timescale asymmetric
(parallax) distortion, so the bridge's `rho` has no reliable tendency to
stay within a factor of 2 of truth. When it doesn't, `fit_lc.
validate_initial_guess_inside_bounds` raises before the first of the 10
grid TRFs even starts.

This is not a profiling-harness artifact: the real production driver
(`install_h1_parallax_grid_multistart_runtime_patch` in
`run_lsstmonts_catalog_hidden_parallax.py`) wraps the per-candidate
`base_multi(...)` call only in `try/finally` (to reset a module-global
flux-embedding context), with **no `except`** — so this same uncaught
`ValueError` would abort the whole event's H1 stage in real production,
not just this audit.

## Explicitly not done here

Per the audit's scope: bounds were not changed, starts were not clipped,
the bridge was not replaced, `old_final` was not modified, no fallback
behavior was invented, and no new basin/hybrid strategies were tried. A
start outside bounds is marked invalid here exactly as the current
production code would treat it (uncaught crash).

## Decision

Incompatibility is **non-negligible** (64%). Per the agreed stop rule,
event-level (P1) and within-event (P2) parallel-throughput profiling are
**deferred** — running them now would either measure a majority-failing
pipeline or require inventing a policy for how to treat crashed events
that was never authorized. Resolving the initialization/bound-convention
inconsistency above is an architectural decision for the user to make; no
fix is proposed or implemented here.
