# Scientific policy freeze -- pre-1M-production checkpoint

Freezes every element the user's closure instructions asked to freeze,
each already fully specified in its own document; this file is the single
pointer/summary tying them together for the tag.

| element | frozen as | doc |
|---|---|---|
| fitting policy | 1 H0 TRF + 1 H1 TRF/event, truth start for the matching model, truth-blind morphology start for the other, `H0_NESTED_piE_0`-style embedding for the H0-generated->H1 case | `FINAL_POLICY.md` |
| bounds | `production_candidate` (`run_bounds_audit_refit_core.py` `BOUNDS_PROFILES["production_candidate"]`): `u0∈[-10,10]`, `tE∈[0.1,500000]`, `rho∈[1e-7,10]`, `piEN,piEE∈[-40,40]`, data-driven `t0` | `TWO_FIT_ARCHITECTURE.md` §1 |
| morphology estimator | `morphology_seed.py: morphology_seed_from_curves(curves)`, truth-blind by construction (signature has no truth argument), reuses the frozen `extract_event_morphology` + FSPL dimensionless lookup unchanged | `TWO_FIT_ARCHITECTURE.md` §2 |
| numerical nested rescue | if `chi2_H1 > chi2_H0 + 1e-6`: 1 extra H1 TRF at the H0 solution embedded with `piEN=piEE=0`; retain the min. `optimality>0.05` continuation explicitly NOT adopted. Catastrophic false convergence flagged (reduced chi2 > 50) for review, not auto-re-fit | `FINAL_POLICY.md` |
| LRT definition | `Delta_chi2_LRT = chi2_H0 - chi2_H1` (both from the final policy, rescue included) | `final_policy.py` |
| alpha | 5% | `H0_CALIBRATION.md` |
| calibrated threshold | 5.85 (90% bootstrap CI [4.67, 8.76], N=59 -- explicitly provisional, see caveat below) | `H0_CALIBRATION.md` |
| failure-handling rules | nested rescue triggers automatically and only on `chi2_H1>chi2_H0+eps`; `reduced_chi2>50` sets an advisory `sanity_flags` entry, logged, never auto-re-fit or silently dropped; `morphology_seed_from_curves`returning `None` (not measurable) skips the affected fit and records `estimator_failure`, never substitutes truth or another value | `FINAL_POLICY.md`, `H0_CALIBRATION.md` |

## Known, quantified, NOT-fixed residual risks (carried forward explicitly, not hidden)

1. **Wrong-model single-start basin gap** (`BASIN_GAP_VALIDATION.md`):
   systematic, directional (inflates `Delta_chi2_LRT`, i.e. biases toward
   claiming detection) risk from the single morphology-seeded start for
   whichever model is NOT the generating one. Quantified on n=25
   independent H1-generated events: median diff ~0.7 (unflagged),
   p90/p95 ~163/295, one clean reclassification (row 603127, no
   sanity-flag warning) at every threshold tested. Self-consistently
   absorbed into the H0-calibration threshold (same policy runs under
   both null and alternative), but costs statistical power relative to a
   robust multistart reference -- not eliminated by calibration.
2. **H0-calibration sample size** (`H0_CALIBRATION.md`): N=59 valid
   events is a time-boxed, first-pass calibration. alpha=5% is
   reasonably resolved (90% CI [4.67,8.76]); alpha=1% is NOT (CI
   [6.24,11.37], dominated by 1-2 order statistics). If the science needs
   alpha<=1%, extend the calibration sample substantially before
   committing to the 1M run.
3. **Estimator failure rate**: 1/60 = 1.7% of the calibration sample hit
   `morphology_not_measurable`. At 1M events this is ~17000 events that
   would need this same honest failure handling (skip + flag, not
   invented values) -- a real, small, already-measured tax on completion
   rate, not a blocker.

## Tag

`git tag -a scientific-policy-freeze-v1-2026-09-15` on the commit that
adds this file and all documents/scripts/results listed above. Message:
freezes the simplified 2(+1)-fit architecture, `production_candidate`
bounds, truth-blind morphology estimator, nested-rescue-only safeguard,
alpha=5%/threshold=5.85 first-pass H0 calibration, ahead of the 1M
H1-generated Sedighe production run. Explicitly notes the two residual
risks above as known and carried forward, not resolved.
