# Independent wrong-model basin-gap validation

## STATUS: NOT NEGLIGIBLE -- STOP, per the pre-agreed stop rule

The expanded, independent N=100 validation below found clean (chi2-
sanity-unflagged), classification-changing basin gaps in the
**false-detection direction at ~10-14% of clean events** across the
entire plausible calibration threshold range (4-25), 95% CI lower bound
never below ~5%. **This is not negligible.** Per the pre-agreed rule:
"If they are not negligible, STOP and report. Do not automatically
resurrect 15-start H0." This document reports the finding; no
architecture change is made here. The only pre-authorized next step (one
additional deterministic truth-blind H0 start, tested separately) has
NOT been implemented -- that is a decision for the user to authorize.

## Method (N=100, independent of every other sample used in this project)

- **Candidate**: `final_policy.py` v2 (2-4 TRF/event, truth start for the
  generating model, morphology-seed start for the other, both adopted
  safeguards), on a NEW frozen H1-generated sample, seed 838383, zero
  overlap with development, extreme100, the 25-event profiling sample, or
  any H0-generated sample (`results/h1_basin_validation_sample_frozen.csv`,
  verified programmatically before use). Nothing was retuned on this
  sample. `results/final_policy_h1basinvalidation_100events.json`.
- **Reference**: the same `old_final`-free, 20-TRF controlled4 robust
  multistart used for the earlier n=25 check (1 bridge + 15 H0 strategies
  across 4 coordinate modes + 4 controlled H1 starts), run on the SAME
  100 events. `results/robust_reference_controlled4_h1basin_100events.json`.
- Analysis: `analyze_basin_validation.py`,
  `results/basin_gap_validation_100events.csv`.

## Basin-gap distribution

n=100, all valid (0 `morphology_not_measurable`). **7/100 (7%)** triggered
the frozen policy's own chi2-sanity flag (reduced chi2 > 50) -- these are
excluded from the classification analysis below exactly as the policy's
own failure-handling rule intends (flagged for review, not silently used
in a detectability decision); see `FINAL_POLICY.md`.

Clean (unflagged) n=93: mean diff = 365.2 (dominated by a few huge
gaps), **median diff = 0.34** (typical-case agreement remains good),
p90 abs diff = 514.6, p95 abs diff = 1980.5, max abs diff = 8259.9.

## Classification disagreement vs threshold grid (clean events only, n=93)

| threshold | false_detect (simple>=thr, ref<thr) | opposite (simple<thr, ref>=thr) | total | fraction | 95% CI |
|---|---|---|---|---|---|
| 2 | 3 | 1 | 4/93 | 4.3% | [1.2%, 10.6%] |
| 3 | 4 | 7 | 11/93 | 11.8% | [6.1%, 20.2%] |
| 4 | 8 | 4 | 12/93 | 12.9% | [6.8%, 21.5%] |
| **5** | **9** | **3** | **12/93** | **12.9%** | **[6.8%, 21.5%]** |
| 6 | 9 | 4 | 13/93 | 14.0% | [7.7%, 22.7%] |
| **7** | **10** | **0** | **10/93** | **10.8%** | **[5.3%, 18.9%]** |
| 8 | 12 | 1 | 13/93 | 14.0% | [7.7%, 22.7%] |
| **9** | **11** | **0** | **11/93** | **11.8%** | **[6.1%, 20.2%]** |
| 12 | 12 | 0 | 12/93 | 12.9% | [6.8%, 21.5%] |
| 16 | 11 | 0 | 11/93 | 11.8% | [6.1%, 20.2%] |
| 20 | 13 | 0 | 13/93 | 14.0% | [7.7%, 22.7%] |
| 25 | 13 | 0 | 13/93 | 14.0% | [7.7%, 22.7%] |

**Across the entire plausible calibration interval (threshold 4-9,
`H0_CALIBRATION.md`'s alpha=5-10% region lands around 4.7-5.9): 9-12/93
clean events (~10-13%) disagree, essentially ALL in the false-detection
direction** (`simple>=threshold, reference<threshold` -- the simplified
policy claims detection where a robust search does not). The opposite
(conservative) direction is small and inconsistent (0-7, no clear
pattern), confirming the directional bias found in the n=25 sample: this
is not symmetric threshold noise, it is a one-sided inflation of
`Delta_chi2_LRT`.

**Material disagreement (|diff| > 5), independent of any threshold**:
**30/93 = 32.3%** of clean events (95% CI [22.9%, 42.7%]).

## Root cause, confirmed at n=100 exactly as at n=25

Every large disagreement traces to the same mechanism: the H1
(truth-seeded, correctly-specified) side matches the robust reference
closely, while the H0 (morphology-seeded, wrong-model) side lands in a
measurably worse local optimum that a 15-strategy/4-coordinate-mode
search finds and a single morphology start does not. Three representative
clean (unflagged, reduced chi2 well under 50) cases:

| row | chi2_H0 (simple) | reduced chi2 | reference chi2_H0 (strategy) | chi2_H1 (simple) | reference chi2_H1 |
|---|---|---|---|---|---|
| 236102 | 3396.2 | 10.5 | 311.2 (`log_te/truth/0.01`) | 308.1 | 308.1 (`truth_half_piE`) |
| 114675 | 2082.3 | 7.4 | 266.8 (`physical/truth/0.1`) | 263.2 | 261.4 (`H0_NESTED_piE_0`) |
| 788605 | 1912.7 | 17.9 | 109.5 (`log_rho/old_H0/0.01`) | 108.9 | 107.0 (`truth_mirror_u0_piEN`) |

**Important refinement of the sanity-flag's coverage**: these three (and
most of the classification-flipping disagreements) have reduced chi2 in
the **7-18 range** -- clearly elevated above the normal 0.75-4.5 band,
but well under the `CHI2_DOF_SANITY_THRESHOLD=50` used for catastrophic
false convergence. **The chi2-sanity flag catches only the most extreme
failures; it does not catch, and was never intended to catch, this
"medium-severity" basin-gap band, which is exactly where most of the
classification-flipping disagreements live.** This is a distinct failure
mode from the catastrophic false-convergence cases in `FINAL_POLICY.md`
-- these are stable, converged (`optimizer_success=True`, plausible
reduced chi2 for a merely-imperfect fit), just landing in a real, wrong,
local optimum relative to what a multistart search reaches.

## Verdict

**Not negligible.** Per the stop rule: STOP here. The 2-fit architecture,
as currently specified (single deterministic morphology-seed start for
the wrong model), produces classification-changing, false-detection-
direction disagreements on the order of 10-14% of clean events across the
entire plausible operating threshold range -- this would materially
inflate any power/detectability estimate computed from the 1M
H1-generated production run relative to what a robust search would find,
exactly the concern raised in the corrected interpretation above (this
document's earlier version, now superseded).

**Not resolved here.** Per the stop rule, no fitter redesign was
attempted. The only pre-authorized next step -- at most one additional
deterministic, truth-blind H0 start -- has NOT been implemented; it would
need its own separate test (does adding it reduce the false-detection
rate above to a negligible level, at what TRF cost) before being
considered for adoption.
