# Scientific policy freeze -- pre-1M-production checkpoint

## STATUS (2026-09-15): NOT approved for production. STOP per the basin-gap finding.

Commit `52dd116` (this document's first version) was explicitly NOT a
final production approval. Four methodological blockers were raised for
resolution without reopening a general fitter-strategy search:

| # | blocker | resolution |
|---|---|---|
| 1 | basin-gap risk mischaracterized as "absorbed by calibration" | **Corrected.** It is not absorbed; status was OPEN pending data -- `BASIN_GAP_VALIDATION.md` |
| 2 | materiality of the basin gap unknown (n=25 only) | **Measured, N=100 independent sample. NOT NEGLIGIBLE: ~10-14% clean, false-detection-direction classification disagreement across the whole plausible threshold range.** `BASIN_GAP_VALIDATION.md` |
| 3 | alpha=5% picked from what N=59 could resolve | **Corrected.** Alpha is not chosen; N=59 kept as a first-pass reference point only, sample-size-vs-alpha table provided, explicit stop for the user's scientific alpha decision -- `H0_CALIBRATION.md` |
| 4 | failure handling needed an explicit, automatic, production-safe rule | **Defined and measured.** Same-point continuation on `reduced_chi2>50` (adopted, trigger rate measured), nested rescue (already adopted), `morphology_not_measurable` handling rule (identical for calibration and production) -- `FINAL_POLICY.md` |
| 5 | throughput measured fitting-only, not end-to-end | **Corrected.** End-to-end (materialize+gate+fit) measured; the earlier "ready within a week" conclusion is retracted for at least one of two plausible readings of "1M" -- `PRODUCTION_PROFILING_FINAL.md` |

**Per the pre-agreed stop rule, blocker #2's result is decisive: the
basin gap is NOT negligible, so this checkpoint STOPS here and reports,
rather than proceeding to production or auto-adding a second H0 start.**

## What is frozen as characterized fact (not as a production go-ahead)

| element | current state | doc |
|---|---|---|
| nominal fitting policy | 1 H0 TRF + 1 H1 TRF/event, truth start for the matching model, truth-blind morphology start for the other | `FINAL_POLICY.md` |
| bounds | `production_candidate`, truth-independent absolute `u0,tE,rho,piEN,piEE`, data-driven `t0` | `TWO_FIT_ARCHITECTURE.md` |
| morphology estimator | `morphology_seed_from_curves(curves)`, truth-blind by construction | `TWO_FIT_ARCHITECTURE.md` |
| safeguard 1 (adopted) | same-point continuation on `reduced_chi2>50`; measured trigger rate 12% (n=25), repairs 2/3 catastrophic cases seen, 1/3 (genuine local-min trap) stays flagged | `FINAL_POLICY.md` |
| safeguard 2 (adopted) | nested H1 rescue on `chi2_H1>chi2_H0+eps`; fires 16-58% depending on regime, resolves every negative-LRT case | `FINAL_POLICY.md` |
| `morphology_not_measurable` handling | skip the affected fit, record `estimator_failure`, exclude from LRT stats, count in failure-rate denominator -- identical rule for calibration and production | `FINAL_POLICY.md`, measured 1.7% (n=60) |
| alpha | **NOT chosen.** N=59 first-pass reference point only (q95~5.85, 90% CI [4.67,8.76]); sample-size-vs-alpha table provided for the user's explicit decision | `H0_CALIBRATION.md` |
| calibrated threshold | **NOT finalized** -- depends on the alpha decision above | `H0_CALIBRATION.md` |
| wrong-model basin-gap risk | **NOT negligible**: ~10-14% clean classification disagreement (false-detection direction) across threshold 4-25, on an independent N=100 sample | `BASIN_GAP_VALIDATION.md` |
| end-to-end throughput | fitting alone: ~8000 events/hour/core. End-to-end (materialize+gate+fit): materialization dominates by ~14.7x; the 1-week target is NOT clearly met under either reading of "1M" at any efficiency measured this session | `PRODUCTION_PROFILING_FINAL.md` |

## Readiness verdict

**NOT ready for the 1M-event production launch.** Two independent,
measured blockers each individually justify holding, before any question
of fitter redesign:

1. The wrong-model single-start basin gap materially changes
   classification for ~10-14% of otherwise-clean events across the whole
   plausible threshold range -- this would bias any power/detectability
   estimate computed from the 1M run. Per the stop rule, do not
   auto-resurrect a 15-start H0 search; the only pre-authorized next
   step (one additional deterministic truth-blind H0 start) is
   unimplemented and untested.
2. Alpha and the final LRT threshold remain unspecified as a matter of
   explicit scientific choice, and end-to-end (not fitting-only)
   throughput does not clearly meet the 1-week target for the more
   demanding of two plausible readings of "1M events."

## No tag

The `scientific-policy-freeze-v1-2026-09-15` tag on `52dd116` stands as a
historical marker of the (now-corrected) first-pass state; it is
explicitly NOT re-affirmed as a production approval by this document, and
no new tag is created until the basin-gap finding is addressed and alpha
is specified.
