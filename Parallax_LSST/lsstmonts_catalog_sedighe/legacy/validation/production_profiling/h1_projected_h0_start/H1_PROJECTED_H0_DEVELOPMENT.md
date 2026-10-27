# Demonstration 2A: one H1-projected second H0 start

## Result: PROMISING BUT INCOMPLETE

Repairs **11/15 (73.3%)** of the dangerous events from Demonstration 1,
but leaves 4 unrepaired, and introduces a material new opposite-
direction risk (5 `new_missed` events, 4 of which persist even under the
most charitable reference-quality correction). Per the predefined,
not-redefined-after-the-fact categories (§11 of the task), this is
**not** "all or all-but-one with no material new opposite-direction
problem" (VERY PROMISING), and it is well above "only a modest
fraction" (INSUFFICIENT) -- it lands squarely in **PROMISING BUT
INCOMPLETE**. Per the stop rule: **STOP here.** No rank-2 morphology, no
second additional start, no per-event fixes, no automatic production
adoption, no new holdout.

## 1-2. Exact H0_2 seed and its methodological status

```
H0_2_seed = (t0, u0, tE, rho) from the SETTLED nominal H1 fit
            (final_h1 in Demonstration 1's frozen output --
            already includes any continuation/nested-rescue the
            existing final policy applied to H1)
piEN, piEE dropped entirely before fitting H0.
```

### Methodological status of H1_projected (required caveat, stated prominently)

1. H1 is truth-started in H1-generated simulations.
2. H0_2 therefore receives information indirectly downstream of a
   truth-started fit -- there is a causal path truth -> H1 fit -> H0_2
   seed.
3. H0_2 does **NOT** directly receive truth parameters; it never sees
   `t0,u0,tE,rho` from `meta["truth"]` at any point.
4. This experiment is mechanistic/diagnostic: it answers "does a
   parameter-space location inferred by the well-converged H1 fit lead
   H0 into the missing basin?", not "is this an acceptable production
   policy?"
5. **A positive (or, as here, partially positive) result does not by
   itself settle whether this start is acceptable for production.** It
   is NOT described as truth-blind anywhere in this document without
   this qualification.
6. That scientific-policy decision -- accept the indirect truth
   dependence, or design a genuinely truth-blind proxy for the same
   geometry -- is explicitly deferred, not made here.

## 3. Reuse of frozen Demonstration 1 results (not recomputed)

`H0_1` and the settled nominal H1 fit were loaded, not recomputed, from:
- `basin_risk_trigger/basin_risk_development_table.csv`: `D_s`, `D_r`,
  `dangerous`, `chi2nu_h0`, `continuation_ran_h0`, H0_1 seed/fit params,
  `n_points`, `excluded_from_development`.
- `results/final_policy_h1basinvalidation_100events.json`: raw
  `chi2_H0_1` (`chi2_h0`), `chi2_H1_current` (`chi2_h1`), and the
  settled H1 fitted `(t0,u0,tE,rho,piEN,piEE)` (`final_h1`) -- these
  fields are not in the CSV, so they are read from this OTHER
  already-frozen Demonstration-1-lineage file; no fit is recomputed.
- `results/robust_reference_controlled4_h1basin_100events.json`:
  `chi2_H0_reference`, `chi2_H1_reference`.

**Verification**: Demonstration 1 already established bit-exact
reproducibility of the frozen H0_1 values for all 100 events
(`reproduction_chi2_diff=0.0` throughout, `BASIN_RISK_TRIGGER_
DEVELOPMENT.md` §4) -- this is cited, not repeated, per the task's own
instruction that such a check is a "separate, clearly labeled
verification step," not the primary source of the numbers used here.

## 4. Bounds, objective, tolerances -- identical to H0_1

H0_2 uses the same H0 physical model, `production_candidate` bounds,
profiled-flux machinery, parameter scaling (raw physical coordinates,
mode=`physical`), and nominal TRF tolerances as H0_1 (`run_one_fit_full`,
unchanged). All four H0 nonlinear parameters free.

### Out-of-bounds handling

Checked per event: **0/93** H1-projected seeds fell outside the
production H0 bounds (t0 data-driven window, `u0∈[-10,10]`,
`tE∈[0.1,500000]`, `rho∈[1e-7,10]`) -- expected, since H1's own bounds
for these 4 shared parameters are identical to H0's under
`production_candidate` (confirmed by inspection of `core.py`'s
`apply_bounds_profile`), so a valid H1 solution is essentially
guaranteed to already satisfy H0's bounds. No event was excluded on this
basis; the out-of-bounds exclusion machinery is implemented and
verified to correctly report zero violations, not simply skipped.

## 5. Same catastrophic-failure safeguard applied to H0_2

Exactly the existing rule (`reduced_chi2 > 50` -> one same-point
continuation, `final_policy.chi2_dof_sanity_flag` /
`numerical_safeguards.run_same_point_continuation`, unmodified) was
applied to H0_2. **Trigger rate: 0/93 (0%)** -- H0_2's nominal fit never
produced a catastrophic reduced chi2 in this sample, so its continuation
never fired. `chi2_H0_final = min(chi2_H0_1_settled, chi2_H0_2_settled)`
for all 93 evaluated events (no out-of-bounds exclusions).

## 6. Post-selection nested-consistency recheck

`epsilon_numeric = 1e-6` (source: `numerical_safeguards.EPSILON_NUMERIC`,
the exact same project-wide constant already used by the existing
nested-model rescue -- no second epsilon was defined). Rechecked
`chi2_H1_current > chi2_H0_final + eps` for all 93 events AFTER H0
selection: **triggered 0/93 times** -- H0_2 never improved chi2_H0
enough to violate the nesting relation the existing final policy had
already established for H1. `chi2_H1_final == chi2_H1_before_second_
start` for every one of the 93 evaluated events.

## 7. Labels (unchanged definitions, reused exactly)

```
dangerous(D, D_r) = (D_r < 9) and (D >= 4) and (D > D_r)
repaired = dangerous_before and (not dangerous_after)
```
`D_s2 = chi2_H0_final - chi2_H1_final`.

## 8. Opposite-direction risk (formalized and measured)

```
missed(D, D_r) = (D_r >= 4) and (D < 9) and (D < D_r)
```
Evaluated on all 93 (no out-of-bounds exclusions):
- `n_missed_before = 8`
- `n_missed_after = 13`
- **`n_new_missed = 5`**: event ids 704717, 97760, 443054, 963300,
  114675.

This is a real, material cost: 5 events that were not at risk of an
opposite-direction (missed-detection) disagreement before H0_2 now are.

## 9. Reference-quality audit

`n_reference_beaten_h0 = 19/93` -- **H0_2 (or the H0_1/H0_2 minimum)
finds a chi2 lower than the existing 20-TRF/4-mode "robust" controlled4
reference in 19 of 93 events.** This is a genuinely important secondary
finding: the "robust" reference used throughout this project's basin-gap
validation is itself not always the true minimum. `n_reference_beaten_h1
= 0/93` (H1 was not touched, since the nested rescue never fired -- see
§6). `n_best_known_nesting_violation = 0/93` -- no inconsistency between
`chi2_H0_best_known` and `chi2_H1_best_known` was found in this sample.

Cross-checking the 5 `new_missed` events against `D_best_known`
(`chi2_H0_best_known = min(chi2_H0_reference, chi2_H0_final)`,
`chi2_H1_best_known = min(chi2_H1_reference, chi2_H1_final)`):

| event_id | D_s | D_s2 | D_r | D_best_known | reference_beaten_h0 | missed(D_s2) | missed(D_best_known) |
|---|---|---|---|---|---|---|---|
| 704717 | 9.99 | 7.46 | 7.46 | 7.46 | True | True | True |
| 97760 | 12.56 | 4.76 | 7.74 | 5.04 | True | True | True |
| 443054 | 14.44 | 8.63 | 10.38 | 10.38 | False | True | False |
| 963300 | 20.11 | 6.69 | 6.70 | 6.69 | True | True | True |
| 114675 | 1819.1 | 3.63 | 5.48 | 5.45 | True | True | True |

**4/5 persist as `missed` even under the charitable `D_best_known`
comparison.** This is not merely a reference-quality artifact:
when `chi2_H0_final` genuinely beats `chi2_H0_reference`
(`reference_beaten_h0=True`), `D_best_known` becomes even SMALLER than
`D_r`, not larger -- crediting the better H0 fit makes these events look
*even less* like parallax detections, not more. This suggests the
original 20-TRF reference itself may have somewhat overstated the
parallax significance for these specific borderline events (its own H0
search did not find as good a fit as a single H1-projected start does),
which is a substantive, separate finding about reference quality, not a
simple flaw in H0_2. Only event 443054 (`reference_beaten_h0=False`)
resolves under `D_best_known` (its apparent "miss" was specific to the
`D_s2` vs the original `D_r` comparison).

## 10-11. Per-event table and primary outcome

`h1_projected_h0_results.csv`, 100 rows (93 evaluated + 7 audited-only
excluded events from Demonstration 1's catastrophic mechanism, per §0 --
none of the 7 excluded events were among the 15 dangerous cases, and 0
of the 93 had an out-of-bounds H0_2 seed).

```
N_clean = 93
N_excluded_out_of_bounds = 0
N_evaluated = 93

N_dangerous_before = 15
N_dangerous_after  = 4

N_repaired   = 11
repair_rate  = 11/15 = 73.3%
```

**Repaired** (11): 229716, 704717, 97760, 788605, 909986, 963300, 236102,
798179, 837177, 114675, 703006.

**Unrepaired** (4): 520062, 324312, 790880, 159493. Of these, 520062 and
324312 show H0_2 not winning at all (`delta_chi2_H0_gain=0`, identical
`D_s2==D_s`) -- the H1-projected seed converged back to essentially the
same point as H0_1 (or worse). 790880 and 159493 show H0_2 winning by a
small margin (gain 0.74 and 0.07 respectively) -- not enough to cross
below `D_r` into "not dangerous."

## 12. Secondary outcomes

**H0_2 usefulness (all 93)**: wins **44.1%** of the time; gain
distribution (chi2 units): median 0.0, p90 438.3, max 8259.9 (most
non-dangerous events see no or negligible improvement; the tail is
driven by a handful of severe basin escapes).

**For the 15 dangerous events specifically**: H0_2 wins **86.7%
(13/15)** of the time; gain median 11.18, mean 521.4, max 3085.0 (see
full per-event table in §8/§9 above and the CSV). Winning does not
guarantee repair (790880, 159493 won but by too little).

**Opposite-direction risk**: see §8 (8 before, 13 after, 5 new).

**Reference improvement**: see §9 (19 reference-beaten-H0, 0
reference-beaten-H1, 0 nesting violations).

**Numerical behavior**: H0_2 continuation trigger rate 0%, nested-rescue
trigger rate after H0 selection 0%, remaining final chi2-sanity-flag
rate among the 93 evaluated events: 0% (none of the 93 clean events
regressed into a catastrophic state under H0_2). Out-of-bounds: 0/93.

**Cost** (added by this experiment on top of the existing 2-TRF nominal
policy; diagnostic development-machine cost only, NOT extrapolated to
the 1M run):
- average TRF added per event: exactly 1.0 (H0_2 nominal only;
  continuation and rescue never fired in this sample) -- i.e. the
  nominal architecture under this policy would be 3 TRF/event (2
  existing + 1 universal H0_2), before any conditional safeguard.
- median added wall time: 0.197s; p90: 0.282s.

## 13. Cost-interpretation caveat (explicit, per instruction)

The added TRF/wall-time cost above is diagnostic-development-machine
context only. It is explicitly NOT used here as evidence that a
universal third fit is acceptable -- that architecture decision must
depend first on repair rate, new-missed risk, and reference consistency
(§§8-11 above), all of which are mixed (partial repair, material new-
missed risk). Cluster-scale throughput is not evaluated in this task and
should only be measured later, if the scientific behavior is judged
satisfactory after review.

## 14. No tuning performed

The seed rule `(t0,u0,tE,rho)_H0_2 = (t0,u0,tE,rho)_H1_final` was frozen
before running any event and was not altered after seeing which events
repaired. No scaling, perturbation, mirroring, or per-event
modification was applied.

## Conclusion / stop

**PROMISING BUT INCOMPLETE.** Repairs a strong majority (73.3%) of the
dangerous basin failures identified in the corrected N=100 validation,
using a single, cheap, deterministic (if not strictly truth-blind)
extra H0 start -- but leaves 4 dangerous cases unrepaired and introduces
a materially new opposite-direction risk (5 events), most of which
persist even after crediting H0_2 for beating the existing robust
reference on 19/93 events (itself a notable finding about that
reference's quality). Per the stop rule: this result is reported, and
no further action is taken automatically -- no rank-2 morphology, no
additional start, no per-event fix, no fresh holdout, no production
adoption. This awaits review of two separate open questions: (1) is the
73%/4-remaining/5-new-missed tradeoff acceptable, or does it need a
different mechanism; and (2) is the indirect truth-dependence of
`H1_projected` itself acceptable for production, or does it require a
genuinely truth-blind proxy for the same geometry.
