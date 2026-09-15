# Independent wrong-model basin-gap validation (H1-generated, n=25)

Question: does the single-start, truth-blind morphology seed for the
wrong model materially alter detectability decisions relative to a
robust/multistart search, and in which direction?

## Method

All 25 events of the frozen H1-generated profiling sample
(`results/profiling_sample_frozen_100.csv`, seed 424242, zero overlap
with any development/calibration/smoke sample). Not used to tune
anything -- the final policy (`final_policy.py`) was frozen before this
run and nothing here fed back into it.

- **Candidate**: `final_policy.py`'s frozen 2(+1)-fit policy
  (`results/final_policy_h1generated_25events.json`).
- **Reference**: an `old_final`-free robust multistart (1 bridge + 15 H0
  strategies across 4 coordinate modes + 4 controlled H1 starts = 20
  TRF/event; built specifically for this validation,
  `stage_h1_controlled4.py` / `orchestrator_robust_reference_controlled4.py`,
  `results/robust_reference_controlled4_25events.json`) -- chosen over the
  `old_final`-based 9-event reference used earlier because it runs
  uniformly on all 25 events without the 64% crash rate documented in
  `FEASIBILITY_AUDIT.md`. Not part of the frozen production policy;
  reference-only.

Full per-event table: `results/basin_gap_validation_25events.csv`.

## Distribution of Delta-LRT differences

n=25. 3/25 events (848426, 793762, 451245) triggered the frozen policy's
own `chi2_dof_sanity_flag` (`FINAL_POLICY.md`) -- self-diagnosed
catastrophic H0 false convergence, not silent. Reporting both with and
without them:

| | all n=25 | unflagged n=22 |
|---|---|---|
| bias (mean diff) | 11221.3 | 62.95 |
| median diff | 1.46 | 0.70 |
| p90 abs diff | 1751.8 | 163.4 |
| p95 abs diff | 25119.8 | 294.9 |
| max abs diff | 245889.2 (451245, flagged) | 662.2 (row 61773) |

The 3 flagged events are entirely responsible for the huge headline
bias/p90/p95 -- they are exactly the cases the policy's own sanity check
already catches and marks for review, so they should not be silently
included in an automated detectability decision. **They are not swept
under the rug**: they ARE real failures of the nominal fit, but they are
self-diagnosable and would be caught before any downstream use, which is
the entire point of the sanity flag.

## Sign/direction of bias

**Systematically positive, even excluding the 3 flagged events**: median
diff = +0.70 (not ~0), and of the meaningful (non-negligible) remaining
gaps, every one has `Delta_chi2_LRT_final_policy > Delta_chi2_LRT_reference`
(rows 603127: +161.1, 61773: +662.2, 625238: +86.2, 597046: +10.2,
893531: +13.1, 320298: +2.0, 185746: +1.1, 721826: +1.5). Root cause,
confirmed by inspection on every one of these: **the H1 side matches the
reference almost exactly** (truth-start already finds ~the same optimum a
robust multistart does, as expected -- e.g. row 603127: 355.20 vs
355.11), while **the H0 side (morphology-seeded, single start) lands in a
measurably worse-but-plausible-looking local optimum** than the robust
search's best (row 603127: chi2=519.6, reduced chi2=1.69 -- looks
completely fine on its own -- vs the reference's 358.4, found via a
truth-anchored, differently-coordinatized strategy the simplified policy
never tries). This inflated chi2_H0 systematically inflates
`Delta_chi2_LRT`, i.e. **the simplification's single-start wrong-model
risk biases toward CLAIMING parallax detection, not toward missing it.**

## Catastrophic (non-flagged) basin gap -- row 603127

The clearest case that a real, undetected-by-sanity-check basin gap can
flip a classification: `Delta_chi2_LRT_final_policy = 164.4` vs
`Delta_chi2_LRT_reference = 3.3`. At every illustrative threshold tested
(4, 9, 16, 25) this event reclassifies from "not detected" (reference) to
"strongly detected" (final policy) -- entirely because of the H0 basin
gap above, with BOTH chi2 values individually looking perfectly ordinary
(reduced chi2 1.69 and 1.16). **This is the residual risk the stop rule
asks to be judged**, not something fixed here.

## Classification disagreement vs threshold

| illustrative threshold | reclassified (of 25) | rows |
|---|---|---|
| 4 | 1 | 603127 |
| 9 | 2 | 848426 (flagged), 603127 |
| 16 | 3 | 848426 (flagged), 893531, 603127 |
| 25 | 2 | 848426 (flagged), 603127 |

Row 603127 reclassifies at every threshold tested -- it is not a
near-threshold borderline artifact, it is a large, threshold-independent
gap. Row 893531 (diff=+13.1) is genuinely near-threshold-sensitive: it
reclassifies only when the threshold falls between `d_ref=10.14` and
`d_simple=23.29` (illustrated at threshold=16). This is the kind of
near-threshold disagreement the validation was specifically asked to
surface.

## Implication for calibration (important, not a fix)

Because the H0 fit's single-start basin bias is systematic and directional
(inflates `Delta_chi2_LRT`), and because the H0-calibration step (§4,
`H0_CALIBRATION.md`) estimates `p(Delta_chi2_LRT | H0)` using this EXACT
same final policy (same single-start H0-basin behavior) on genuinely
H0-generated events, the resulting calibrated threshold will already
reflect however much this bias also inflates the NULL distribution --
i.e. the false-positive rate at the chosen alpha should still be correctly
controlled by construction. What calibration does NOT fix is the loss of
statistical POWER this adds: extra basin-driven variance in Delta_chi2_LRT
under H1 requires a higher threshold to hold alpha fixed, which costs
some true detections relative to a robust-multistart policy. This
efficiency cost is real, quantified above, and not eliminated by
calibration -- it is exactly the tradeoff the stop rule asks to be judged
against the ~16-29x runtime speedup.
