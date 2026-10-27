# Independent H0 reference after Demonstration 2A

## 1-2. Old-reference limitation and the circularity problem

Demonstration 2A found H0_2 (or, more precisely, `chi2_H0_final =
min(H0_1, H0_2)`) below the old controlled4 20-TRF reference on 19/93
clean events. `chi2_H0_best_known = min(controlled4, H0_1, H0_2, ...)`
is valid bookkeeping for "best chi2 found so far," but grading H0_2
against a quantity that can itself include H0_2 is circular. This task
builds `chi2_H0_ref2_independent`, which **excludes H0_2 and
H1_projected entirely**, from a small frozen panel of already-existing,
H0_2-independent H0 starts, and re-evaluates both the original
single-start problem and Demonstration 2A against it.

## 3. Frozen verification panel (documented before running)

4 members, fixed identically for all selected events, chosen only
from already-implemented, historically-existing strategy/mechanism
code -- no new perturbation family invented:

| strategy_name | source/module | seed definition | coordinate mode | truth dependence | independence from H0_2 |
|---|---|---|---|---|---|
| `old_H0_bridge` | `stage_bridge_morphology.py` (unmodified); REUSED from `results/pipeline_stage_json/{row}_bridge.json`, already computed for this same 100-event sample -- zero new TRF | truth start, OLD narrow FAST bounds | physical | truth-anchored (explicitly reference-only, historically existing since the project's very first architecture) | independent -- predates H0_2 entirely |
| `physical/truth/truth_rho` | `stage_h0_downstream.py`'s truth-anchor construction pattern (unmodified), extended with the already-established `"truth_rho"` fraction token (used elsewhere in `log_rho`/`log_te` modes, just never in `physical` mode) | truth `(t0,u0,tE)`, `rho`=truth rho | physical | truth-anchored | independent -- no H1 fit involved anywhere |
| `log_te_rho/truth/1` | same pattern, extended to the `log_te_rho` coordinate mode (already implemented in `core.py`, never used for a non-morphology start in this project -- the frozen 16-cost architecture only ever ran the morphology strategy in this mode) | truth `(t0,u0,tE,rho)` | log_te_rho | truth-anchored | independent |
| `log_te_rho/old_H0/1` | same, old_H0-anchored | `(t0,u0,tE)` from the bridge vector, `rho`=truth rho | log_te_rho | truth-anchored (via the bridge) | independent |

Truth-anchored strategies are explicitly used here as reference-only,
per the task's allowance -- none is proposed as production policy.
Same H0 physical model, profiled-flux objective, `production_candidate`
bounds (except the bridge, which correctly keeps its own historical old
narrow bounds), and the already-adopted `reduced_chi2>50` -> one
same-point continuation safeguard, applied identically
(`run_verification_panel.py`, reusing `run_one_fit_full`,
`chi2_dof_sanity_flag`, `run_same_point_continuation` unmodified).

**One panel entry failed** for one event (`log_te_rho/old_H0/1` on row
883597: the bridge's own `t0`, fit under its own old bounds, fell
~7 days outside the CURRENT data-driven H0 `t0` window) -- recorded as
`failed=True`, not patched, not imputed; the remaining 3 panel members
still cover that event.

## 4. Confirmation: H0_2/H1_projected excluded from `chi2_H0_ref2_independent`

`build_reference_v2_independent.py`'s candidate list for the 19 events
is exactly `[old_controlled4_reference, old_H0_bridge,
physical/truth/truth_rho, log_te_rho/truth/1, log_te_rho/old_H0/1]` --
`chi2_H0_2`/`chi2_H0_H0_2` never enters this list; it is loaded
separately, purely for comparison.

## 5. Selected events

Verified from stored results (not hardcoded): **19** events, exactly
matching Demonstration 2A's `reference_beaten_h0` flag
(`chi2_H0_final = min(H0_1,H0_2) < chi2_H0_reference - eps`).
`verification_panel_events_19.csv`.

**Important clarifying nuance** (not in the original task framing, found
during this analysis): of these 19, only **7** are cases where H0_2
*itself* (not H0_1) is strictly below both H0_1 and the reference
(`chi2_H0_2 < chi2_H0_1 - eps` and `chi2_H0_2 < chi2_H0_reference -
eps`): 704717, 738063, 97760, 87599, 900304, 963300, 114675. For the
other 12/19, **H0_1 alone** (already part of the existing frozen
policy, computed identically in every event) already beat the old
reference, and H0_2 was not the actual discovery for those events. This
distinction matters for §7 below and is kept explicit throughout
(`h0_2_basin_replication_classification.csv`'s `h0_2_was_actual_winner`
column).

## 6. Per-strategy verification results

`physical_panel_results.csv` (19 rows), `log_te_rho_panel_results.csv`
(38 rows, 1 marked `failed=True`). All succeeded fits report
`optimizer_success=True`; 0/56 successful fits triggered the
continuation safeguard.

## 7. H0_2 basin replication classification

**Precise definition used** (distinct from the broader "best-known
including the old reference" question in §9 below): for each of the 19
events, compare `chi2_H0_2` only against the 4 NEW panel members
(excluding the old reference, which was already known and is not a
"verification start" in the sense of this question):

- **A. independently reproduced** (`best_new_chi2 <= chi2_H0_2 + eps`,
  not strictly lower): **1/19** (919990).
- **C. independent panel improves beyond H0_2** (`best_new_chi2 <
  chi2_H0_2 - eps`): **7/19** (366338, 78241, 97760, 442306, 883597,
  904318, 448056) -- but cross-referencing `h0_2_was_actual_winner`:
  6 of these 7 are cases where H0_1 (not H0_2) was already the actual
  discovery, so the panel "improving beyond H0_2" there is not a
  surprising new result; only **97760** is a genuine case of H0_2 being
  the real discovery AND the panel finding something even lower
  (163.777 vs H0_2's 163.778 -- a marginal, near-tie improvement).
- **B. H0_2 still best-known** (no new panel member matches or beats
  it): **11/19**: 87599, 114675, 354627, 670244, 704717, 738063,
  815761, 897838, 900304, 940849, 963300.

`n_independent_starts_reproducing_H0_2` (candidates, including the old
reference, at or below `chi2_H0_2+eps`) is also reported per event in
`reference_v2_independent.csv` for the broader bookkeeping question.

**Among the 7 events where H0_2 was the genuine discovery** (§5's
clarifying nuance): only **1/7 (97760)** is even marginally matched by
the small independent panel; the other **6/7 remain
H0_2-only-best-known, unreproduced** by this panel
(`h0_2_basin_replication_classification.csv`).

## 8-9. Independent H0 and H1 reference construction; nesting audit

For the 19 selected events: `chi2_H0_ref2_independent = min(old
controlled4 reference, old_H0_bridge, physical/truth/truth_rho,
log_te_rho/truth/1, log_te_rho/old_H0/1)` (excludes H0_2). For the
other 74: `chi2_H0_ref2_independent = chi2_H0_old_reference` unchanged
(no additional historical reference was available or added for these,
per the predeclared rule -- not selectively improved).

Nesting audit (`chi2_H1_old_reference <= chi2_H0_ref2_independent +
eps`, using `epsilon_numeric=1e-6` from `numerical_safeguards.
EPSILON_NUMERIC`, the exact same project-wide constant, reused
unchanged): **0/93 violations before any rescue** -- the existing
`chi2_H1_old_reference` already satisfies nesting against the (weakly
improved) independent H0 reference for every one of the 93 events, so
**the reference-H1 nested rescue never fired (0/93)**, and
`chi2_H1_ref2_independent = chi2_H1_old_reference` for all 93 events.
**0/93 violations remain** (trivially, since none occurred).

## 10-12. D_ref2_independent and non-circular re-evaluation

`reference_v2_independent.csv`, 93 rows, columns exactly as specified
in the task (§10) plus `delta_H0`/`delta_H1` (§14).

### Q2: corrected original single-start dangerous/missed counts

```
old_reference dangerous count:        15
independent_ref2 dangerous count:     15   (UNCHANGED)
removed from dangerous (old->ref2):   []   (none)
newly added to dangerous (old->ref2): []   (none)
old_reference missed count:           8
independent_ref2 missed count:        8    (UNCHANGED)
```

**The original "15 dangerous" label is NOT an artifact of a weak
reference** -- the independent panel's improvements over the old
reference (where they exist) were too small in magnitude to flip any
event across the `dangerous`/`missed` threshold definitions. The old
controlled4 reference, while demonstrably beatable on 19/93 events
(mostly by the ALREADY-EXISTING H0_1, not a new discovery), was
apparently good enough for the specific purpose of labeling the
original 15 dangerous single-start events.

### Q3: non-circular Demonstration 2A repair rate

```
N_dangerous_before_ref2 = 15
N_dangerous_after_ref2  = 4
N_repaired_ref2          = 11
repair_rate_ref2         = 73.3%    (IDENTICAL to the original circular D_best_known result)

N_missed_before_ref2 = 8
N_missed_after_ref2  = 13
N_new_missed_ref2    = 5            (IDENTICAL to the original result)
```

**The non-circular re-evaluation reproduces Demonstration 2A's headline
numbers exactly.** This is a reassuring finding in its own right: the
73.3% repair rate and the 5 new-missed events reported in Demonstration
2A were NOT artifacts of grading against a reference that H0_2 itself
had contaminated -- they hold up against a reference explicitly built
to exclude H0_2.

## 13. Special treatment of H0_2-best-known events

Flagged via `independent_reference_beats_H0_2=False` combined with
`H0_2_beats_independent_reference` in `reference_v2_independent.csv`
(the 6-11 event distinction depends on which comparison, broad §9 vs
strict §7 -- see both tables). For all such events, both `D_ref2_
independent` and `D_best_known` are reported side by side (never
merged); per instruction, `D_best_known` is NOT used to claim unbiased
repair-rate validation anywhere in this document -- only `D_ref2_
independent` is used for §§10-12.

## 14. delta_H0/delta_H1 decomposition for discrepant events

`discrepant_event_decomposition.csv` (also inline below); `delta_H0 =
chi2_H0_candidate(=chi2_H0_final from 2A) - chi2_H0_ref2_independent`,
`delta_H1 = chi2_H1_candidate - chi2_H1_ref2_independent`.

| event_id | D_s | D_s2 | D_ref2_independent | delta_H0 | delta_H1 | category | source of discrepancy |
|---|---|---|---|---|---|---|---|
| 520062 | 10.16 | 10.16 (unrepaired) | 3.45 | **7.23** | 0.52 | unrepaired dangerous | mostly H0-limited |
| 324312 | 8.99 | 8.99 (unrepaired) | 4.25 | **5.14** | 0.40 | unrepaired dangerous | mostly H0-limited |
| 790880 | 5.93 | 5.19 (unrepaired) | 4.82 | 0.38 | 0.00 | unrepaired dangerous | both small -- borderline/threshold case, not a fitter gap |
| 159493 | 4.91 | 4.85 (unrepaired) | 4.84 | 0.01 | 0.00 | unrepaired dangerous | both ~0 -- borderline/threshold case, not a fitter gap |
| 704717 | 9.99 | 7.46 | 7.46 | ~0.00 | 0.01 | new_missed | both ~0 -- borderline/threshold case; H0_2-best-known (§7) |
| 97760 | 12.56 | 4.76 | 5.04 | ~0.00 | 0.28 | new_missed | small H1-side gap; the ONE event where the panel marginally improves on H0_2 |
| 443054 | 14.44 | 8.63 | 10.38 | 1.34 | **3.08** | new_missed | **mostly H1-limited** (not one of the original 15 dangerous) |
| 963300 | 20.11 | 6.69 | 6.70 | ~0.00 | 0.00 | new_missed | both ~0 -- borderline/threshold case; H0_2-best-known (§7) |
| 114675 | 1819.1 | 3.63 | 5.48 | ~0.00 | **1.82** | new_missed | H0 side matches/slightly beats reference; **mostly H1-limited**; H0_2-best-known (§7) |

**Pattern**: of the 4 unrepaired dangerous events, exactly 2 (520062,
324312) show a real, sizeable H0-side gap against the independent
reference -- genuine H0 basin-search insufficiency that neither H0_1
nor H0_2 nor this small panel resolves. The other 2 (790880, 159493)
and 3 of the 5 new_missed events (704717, 963300, and to a lesser
extent 97760) show the candidate essentially MATCHING the independent
reference on both H0 and H1 sides (`delta_H0`, `delta_H1` both ~0) --
these are not candidate/fitter failures; they are events whose true
`D_ref2_independent` itself sits right at the edge of the provisional
`[4,9]` interval, so small numerical differences alone flip the coarse
label. Only 2 events (443054, 114675) show a real, non-trivial H1-side
gap (`delta_H1` = 3.08, 1.82) with a near-zero H0-side gap -- these
point toward H1 (not H0) as the locally limiting factor for those two
specific cases, not a broad pattern (n=2).

## 15. Answers to Q1/Q2/Q3 (kept separate, not merged)

### Q1: Was the old controlled4 H0 reference too weak?

Yes, measurably, but mostly for a reason already known before this
task: **12/19 "H0_2 beats reference" events are actually H0_1 (already
in the existing frozen final policy) beating the reference**, not a
new discovery. Of the 7 events where H0_2 specifically was the
discovery, the small 4-member independent panel reproduces/improves on
it in only 1/7 (97760, marginally); H0_2 remains the sole best-known
solution, unreproduced by this panel, on 6/7. So: the reference was
"too weak" mainly in the sense that it didn't already include H0_1
itself as a candidate (an existing, cheap, always-computed fit) --
that gap is now closed for these 93 events. Whether H0_2's specific,
additional discoveries (the 6 unreproduced cases) reflect a genuinely
better basin or an as-yet-unverified one remains open; this small
panel cannot settle it (§4 of the earlier task: a panel of 3-5 starts
"cannot mathematically prove the global minimum").

### Q2: How many of the original 15 dangerous single-start events remain dangerous against the stronger independent reference?

**All 15.** The corrected label set is identical to the original
(`chi2_H0_ref2_independent` improved too little, where it improved at
all, to move any event across the dangerous/missed boundary). The
original "15 dangerous" finding from Demonstration 1 is not a
reference-weakness artifact.

### Q3: How well does H1_projected actually repair them when graded against an independent reference?

**Identical to the original (circular) result: 11/15 repaired (73.3%),
4 unrepaired, 5 new_missed.** The non-circularity concern raised at the
start of this task is resolved: it does not change Demonstration 2A's
headline numbers. What it DOES add is the delta_H0/delta_H1
decomposition above, which shows the 4 unrepaired + 5 new-missed
events split roughly into: 2 genuine H0-limited failures, 2 genuine
(if narrow, n=2) H1-limited cases, and 5 cases that are better
described as borderline/threshold-sensitive than as fitter failures on
either side.

## 16. Explicit uncertainties / limitations

- The 4-member panel is small; per the task's own instruction, it
  cannot prove any event's true global H0 minimum. "Independently
  reproduced" and "still best-known" describe this specific panel's
  behavior, not a settled fact about the underlying optimization
  landscape.
- One panel entry failed for one event (log_te_rho/old_H0/1, row
  883597) due to a bridge-vs-data-window bounds mismatch -- itself a
  minor, separate domain-consistency finding, not investigated further
  here (out of scope, per the task's stop rule).
- The δH0/δH1 "H1-limited" finding rests on only 2 events (443054,
  114675) -- not a statistically established pattern.
- `chi2_H0_ref2_independent` improves on `chi2_H0_old_reference` for
  74/93 events not at all (no new candidate available/added for them,
  per the predeclared rule) -- this task's improvement is concentrated
  entirely on the 19 pre-selected events.

## 17. No production-policy change

No fitter redesign, no new production policy, no new holdout, no H0
calibration extension, no large-scale profiling, no change to the
H1_projected seed rule, no rank-2 morphology, no additional H0 start,
no adaptive/trigger logic, no threshold changes. This task only built
an independent grading reference and re-evaluated already-frozen
results against it.

## 18. Stop-rule outcome

Per the task's final stop rule: the updated, non-circular decomposition
shows the remaining discrepancies are **mixed** -- 2 clearly H0-limited,
2 clearly (if narrowly) H1-limited, and 5 better characterized as
threshold-boundary sensitivity rather than a fitter-side deficiency at
all. This does not cleanly match any single one of the three
predefined stop-rule branches ("primarily H1-side" / "primarily
H0-side" / "very strong repair"). **Reporting this mixed, decomposed
result and stopping here**, as instructed. No automatic next
experiment is launched.
