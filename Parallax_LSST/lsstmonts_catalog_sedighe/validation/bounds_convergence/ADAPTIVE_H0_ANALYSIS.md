# Adaptive H0 start-strategy analysis — working log

**Status:** IN PROGRESS
**Started from checkpoint:** commit `9796d97`, tag
`validation-gates-1-2-2026-09-14`
**Relation to `VALIDATION_STATUS_2026-09-14.md`:** that document is the
historical Gate-1/Gate-2 checkpoint and is **not** edited retroactively
by this analysis. This document is the canonical, evolving record of
the H0 adaptive-strategy investigation that the checkpoint's §22
("Immediate next validation task") calls for. When a stage of this
analysis closes, its conclusion will be folded back into
`VALIDATION_STATUS_2026-09-14.md` or into a new, explicitly dated
checkpoint — never by silently rewriting the 2026-09-14 text.

This file must let another person or agent, with no access to prior
conversation, understand what was tried, what worked, what was
rejected and why, and what to do next.

---

## 0. Inherited state (from checkpoint `9796d97`)

### 0.1 What is validated

- Shared H0/H1 nuisance domain `production_candidate`:
  `u0=[-10,10]`, `tE=[0.1,500000] d`, `rho=[1e-7,10]`,
  `piEN,piEE=[-40,40]` for H1.
- Data-driven `t0` base interval with an adaptive `0.25*Tobs` rescue
  triggered when the winning H0 solution has an active `t0` bound
  (triggered once in `extreme100`, catalog row `36103`).
- H0 has strong optimizer-basin structure: 1-3 fixed starts are not
  reliable; the fixed 18-strategy plan `gate1_final18` (3 `physical`,
  6 `log_te`, 8 `log_rho`, 1 `log_te_rho`) reproduces the 52-fit
  4-coordinate oracle with `N(delta_chi2>0.1)=0` on `extreme100`
  (max delta approx `0.02714`).
- H1 controlled-simulation multistart (`truth`, `H0_NESTED_piE_0`,
  `truth_half_piE`, `truth_mirror_u0_piEN`, `old_final_reseed`)
  reproduces its own 5-start oracle with `N(delta_chi2>0.1)=0` on
  `extreme100`; 4 starts already lose 4 events (`35101`, `81449`,
  `579071`, `579320`), all requiring `truth`.
- Nestedness: the exact H0 solution must be inserted explicitly as an
  H1 candidate (`H0_NESTED_piE_0`), not assumed recoverable by TRF.

### 0.2 What is explicitly NOT yet validated

- Whether an **adaptive/sequential** H0 strategy can replace the fixed
  18-fit plan at substantially lower mean cost while preserving
  `N(delta_chi2>0.1)=0` on `extreme100`.
- Whether any observable available during a real run (no oracle)
  reliably predicts that the current H0 minimum is still wrong.
- Whether the 18-strategy execution *order* used by `gate1_final18` is
  a good order for a sequential policy — it was selected as a
  fixed-cardinality **set-cover** solution, not to minimize the
  expected cost of a sequential stopping rule. **This must not be
  assumed.**
- Any H1 rescue beyond the current 5-start physical-coordinate plan.
- Empirical LRT threshold calibration (deliberately deferred until the
  fitter is frozen).

### 0.3 Ground rule for this analysis

Every predictor, trigger, or policy studied here must be computable
from information available during a real run. The 52-fit and 5-fit
oracles are used **only** offline, to compute
`delta_chi2 = chi2_policy - chi2_oracle` for evaluating a candidate
policy after the fact. No oracle-derived quantity may be used as a
policy feature. This separation is tracked explicitly in every
experiment entry below (§ "Information allowed in production").

### 0.4 Clarification: `gate1_final18_rescue_t0margin0.25` is a diagnostic run, not the validated production policy

`gate1_final18_rescue_t0margin0.25` (one of the three `source` values
aggregated in Experiment 0) contains 18 H0 fits for **all 100**
`extreme100` events at `t0_margin_factor=0.25`. This is a **global
diagnostic run**, executed to have the rescue-domain result available
for every event for validation purposes.

It does **not** represent what the validated production policy
actually does. The validated policy (checkpoint `9796d97`, §6) is:

1. run `gate1_final18` with `t0_margin_factor=0` (the base interval)
   for the event;
2. take the winner of that **full 18-fit** run;
3. only if that winner has an active `t0` bound, rerun the **same
   event** with `t0_margin_factor=0.25`;
4. keep the better of the two results.

In `extreme100` this condition was true for exactly one event,
`catalog_row=36103` (confirmed again in Experiment 0, §7: the
`gate1_final18_base_t0margin0` winner-level `t0_active_rate` is
`0.0100 = 1/100`, and it is row `36103`; for `gate1_oracle_52` the
same single-event result additionally matches the pre-existing,
independently computed `gate1_oracle_active_bounds.csv` on that one
count — see §7 for the exact scope of that comparison). The pipeline
therefore reruns 1/100 events at margin 0.25, not 100/100.

**Consequence for Phase A/B:** any policy that decides to fire the
`t0`-margin rescue based on the winner of a **partial** run (i.e.
after `k<18` starts, before the full `gate1_final18` plan has been
executed) is a genuinely new adaptive rule that has not been
validated and must not be presented as already covered by the
checkpoint's t0-rescue result. The checkpoint only validates
triggering the rescue from the **full-18-fit** winner. Testing an
earlier trigger point is an explicit, separate hypothesis for Phase
A/B to evaluate offline (against the oracle, never in production
without that offline evidence).

### 0.4b `optimizer_success` in this dataset

`optimizer_success` is `True` for 100% of rows across all three
`source` values in `extreme100` (Experiment 0, §7). This means it has
**no discriminative power on this development sample** and is
excluded from the Phase A predictor list for that reason. This is
purely a statement about `extreme100`, not a statement about the
signal's general usefulness: `optimizer_success=False` must continue
to be treated as a failure/rescue condition in the fitter and in any
independent validation sample (Gate 3) or production run, where it
may occur even though it never did here.

### 0.5 Explicitly out of scope for this analysis (do not modify)

- `validation/bounds_audit/run_bounds_audit_refit_core.py`
- `validation/bounds_audit/run_bounds_audit_refit.py`
- `run_lsstmonts_catalog_hidden_parallax.py`
- `production_candidate` bounds
- The H1 controlled-5 start plan
- LRT threshold calibration

Any future proposal to change one of these must first be written up
in this document (change, evidence, before/after behaviour) and
agreed before code is touched.

---

## 1. Experiment log

### Experiment 0 — Build and audit the per-start Gate-1 diagnostics table

**1. Question**

Is there a single, unambiguous, per-start table of H0 optimizer
diagnostics (chi2, status, optimizer success/optimality/active bounds)
for the three relevant Gate-1 runs, with a strategy key that is
provably stable **across events**, so that it can be used as the raw
input for the sequential/adaptive analysis without silently mixing
runs that are not scientifically equivalent? In particular: can a
canonical `strategy_id` be constructed for the 52-fit oracle so that
`(mode, strategy_id)` identifies one of the 52 strategies consistently
for every event, given that the raw `label` text is known to embed
per-event numeric values for some starts?

**2. Motivation**

The adaptive-strategy analysis (Phase A/B, below) needs per-start
optimizer diagnostics that are not present in the existing
`gate1_all_start_fits.csv` (which only carries `chi2`, the
already-matched oracle chi2, and their delta). It also must not treat
the 52-fit oracle, the base `gate1_final18` run, and its `t0`-rescue
rerun as three interchangeable "strategies" — they are three distinct
runs with different start-plans and, for the rescue run, a different
`t0` domain.

**3. Inputs**

- `aggregate_gate1_oracle_diagnostics.py` (pre-existing in the working
  tree at the start of this session, authored to support exactly this
  need; verified and documented here, not rewritten).
- Local per-event `all_refits.csv` files under three root families on
  this machine (paths recorded inside the script's `ROOTS` dict):
  - `~/Downloads/hidden_parallax/production_validation/gate1_coordinates`
    (the 4x13 = 52-strategy oracle, `HIDDEN_PARALLAX_H0_START_PLAN=all`)
  - `~/Downloads/hidden_parallax/production_validation/gate1B_final18_t0base`
    (`gate1_final18` plan, `HIDDEN_PARALLAX_T0_MARGIN_FACTOR=0`)
  - `~/Downloads/hidden_parallax/production_validation/gate1B_final18`
    (`gate1_final18` plan, `HIDDEN_PARALLAX_T0_MARGIN_FACTOR=0.25`,
    i.e. the adaptive t0-rescue rerun)
- New audit script written in this session:
  `audit_gate1_oracle_diagnostics.py`.

**4. Method**

`aggregate_gate1_oracle_diagnostics.py` reads every event's
`all_refits.csv` under each root/mode, in on-disk row order (the
execution order of that call), and:

- tags each row with `source` (one of the three run names above),
  `root`, `h0_start_plan`, `t0_margin_factor`, `mode`, `catalog_row`;
- assigns `start_slot = 1..N` as the row's position within that
  specific `(source, mode, catalog_row)` call — **not** a property of
  the fitter itself, purely an on-disk ordering label, retained as
  provenance only;
- assigns a canonical `strategy_id`:
  - for `gate1_final18` sources: `strategy_id = label` (already
    event-independent by construction — see §7 below);
  - for `gate1_oracle_52` (`h0_start_plan="all"`): `strategy_id` is
    reconstructed from `start_slot` via a fixed 13-entry lookup table
    (`GATE1_ORACLE_52_SLOT_STRATEGY_ID`) transcribed directly from
    the start-construction loop in `run_bounds_audit_refit_core.py`
    (`h0_anchor_vectors` = `[("old_H0", ...), ("truth", ...)]`
    crossed with `unique_rhos([v[3], truth_rho, *RHO_GRID])`, lines
    2436-2492, with `RHO_GRID = [1e-6,1e-4,1e-2,1e-1,1.0]` at
    lines 1489-1495), giving:

    | slot | strategy_id | meaning |
    |---:|---|---|
    | 1 | `old_H0/rho_from_old_H0` | anchor=old_H0, rho=that anchor's own rho |
    | 2 | `old_H0/truth_rho` | anchor=old_H0, rho=this event's truth rho |
    | 3-7 | `old_H0/1e-06` ... `old_H0/1` | anchor=old_H0, rho=fixed grid value |
    | 8 | `truth/truth_rho` | anchor=truth, rho=that anchor's own (=truth) rho |
    | 9-13 | `truth/1e-06` ... `truth/1` | anchor=truth, rho=fixed grid value |

    This mapping is guarded, not assumed: it is applied only when a
    `(mode, catalog_row)` call has exactly 13 rows, and raises a
    `RuntimeError` naming the offending event/mode otherwise (a
    `unique_rhos()` dedup could in principle collapse a slot for some
    event and shift the positional mapping — this guard exists so
    such a case fails loudly instead of being silently mislabeled).
  - keeps a fixed column subset (see schema, §2 below) plus a
    wall-clock time table keyed by `(source, mode, catalog_row)`.

It performs no fitting; it is a pure re-aggregation of already-computed
local results into two repository-tracked CSVs. Root directories are
resolvable via `--root SOURCE=PATH` (repeatable) or
`GATE1_DIAG_ROOT_<SOURCE_UPPER>` environment variables, instead of
only via in-file edits to `ROOTS`.

`audit_gate1_oracle_diagnostics.py` then computes, purely from that
aggregated table, per `source`:

- row and event counts;
- number of distinct `(mode, label)` pairs **and** number of distinct
  `(mode, strategy_id)` pairs, compared against the expected plan
  size (52 for `gate1_oracle_52`, 18 for each `gate1_final18` source);
- whether every `(mode, strategy_id)` pair covers all events in its
  source (no event silently missing a strategy);
- whether `(mode, start_slot) -> label` is a function (single label
  per slot) across all 100 events, checked by literally grouping on
  `(source, mode, start_slot)` and counting distinct labels observed
  — not assumed;
- duplicate `(mode, catalog_row, start_slot)` / `(mode, catalog_row,
  label)` / `(mode, catalog_row, strategy_id)` keys, and
  `(mode, catalog_row)` combinations whose row count is short of,
  over, or entirely missing relative to the expected count for that
  source;
- `optimizer_success` rate and `optimizer_active_mask` component-wise
  active-bound rate (t0, u0, tE, rho), parsed from the stringified
  4-vector scipy writes, reported **separately** as:
  - `per_fit_*` — over every individual H0 start;
  - `winner_*` — over only the per-`(source, catalog_row)`
    minimum-chi2 row, i.e. one row per event (this is the winner
    *within that source's own starts*, not the cross-source oracle).

`audit_gate1_oracle_diagnostics.py` additionally runs a **hard**
integrity check (`audit_integrity`), per source, that RAISES
`RuntimeError` (not merely prints) on any violation of:
- `hypothesis` is `"H0"` for every row;
- `status` is `"success"` for every row counted as usable, and
  `chi2`/`t0`/`u0`/`tE`/`rho` are finite (`np.isfinite`) on every
  `status=="success"` row;
- `h0_start_plan` equals the source's own declared plan
  (`"all"` for `gate1_oracle_52`, `"gate1_final18"` for both
  `gate1_final18` sources) for every row;
- `t0_margin_factor` equals the source's own declared value
  (`0.0`, `0.0`, `0.25` respectively) for every row, to floating-point
  tolerance;
- the set of `catalog_row` values for that source is exactly the
  100-event `extreme100` reference set read from
  `data/extreme100_refit_manifest.csv` (no extra events, none
  missing).

This check was manually verified to actually raise (not silently
pass) by re-running it in this session against three deliberately
corrupted copies of the table (one bad `t0_margin_factor` value, one
event dropped from `gate1_final18_base_t0margin0`, one `NaN` `chi2`
on a `status=="success"` row) — each raised the expected
`RuntimeError` naming the violation; the unmodified table raises
nothing.

**5. Information allowed in production**

This is a data-integrity audit, not a policy. `gate1_oracle_diagnostics.csv`
itself contains only per-start fields that a real run already produces
(`status, chi2, t0,u0,tE,rho, optimizer_success, optimizer_optimality,
optimizer_active_mask`), plus run-identifying metadata
(`source, root, h0_start_plan, t0_margin_factor, mode, catalog_row,
start_slot, label, strategy_id`) that describes *which validation run*
a row came from and *which canonical strategy* it corresponds to —
this metadata is a property of the offline experiment design and of
the fitter's own start-construction code, not of the online fit
result, and must not be treated as a fit feature. `strategy_id` in
particular is derived only from which anchor/rho rule a given slot
uses (visible before the fit runs), never from any fit's outcome. The
table does **not** contain any oracle chi2 column; oracle comparison
happens only in downstream analysis scripts that explicitly join
against `gate1_all_start_fits.csv` / `gate1B_final18_per_event.csv`
(pre-existing checkpoint results, which already store the 52-fit
oracle winner per event).

**6. Metrics**

Structural/audit metrics only (see method above); no `delta_chi2`,
no cost/coverage metrics at this stage.

**7. Result**

Re-ran `aggregate_gate1_oracle_diagnostics.py` fresh in this session
(all three local root families still present on this machine):
`missing files: 0`, `row-count mismatches: 0`, `8800` fit rows,
`1200` call rows written, `strategy_id` column added without error
(the 13-row guard in `_gate1_oracle_52_strategy_id` never fired,
i.e. every `gate1_oracle_52` (mode, catalog_row) call had exactly the
assumed 13 rows) — confirming the repo-tracked
`gate1_oracle_diagnostics.csv` matches its declared local sources.

`audit_gate1_oracle_diagnostics.py` output (full text also
reproducible by re-running the script; summary CSVs listed in §2).

**Hard integrity check result — all three sources passed with zero
violations** (`audit_integrity`, part of the same script run):

| source | hypothesis | status | n_nonfinite (chi2/t0/u0/tE/rho) | h0_start_plan mismatches | t0_margin_factor mismatches | catalog_row set vs. reference |
|---|---|---|---|---:|---:|---|
| `gate1_oracle_52` | `['H0']` | `{success: 5200}` | 0/0/0/0/0 | 0 | 0 | exact match (100/100, 0 extra, 0 missing) |
| `gate1_final18_base_t0margin0` | `['H0']` | `{success: 1800}` | 0/0/0/0/0 | 0 | 0 | exact match (100/100, 0 extra, 0 missing) |
| `gate1_final18_rescue_t0margin0.25` | `['H0']` | `{success: 1800}` | 0/0/0/0/0 | 0 | 0 | exact match (100/100, 0 extra, 0 missing) |

Structural/strategy counts (same run):

| source | n_rows | n_events | n (mode,label) | n (mode,strategy_id) | strategy_id covers all events? | start_slot→label constant? | duplicates | short/over/absent (mode,row) |
|---|---:|---:|---:|---:|---|---|---:|---:|
| `gate1_oracle_52` | 5200 | 100 | 1184 | **52** (expected 52, OK) | YES — 52/52 | **NO** — 40/52 constant | 0 | 0/0/0 |
| `gate1_final18_base_t0margin0` | 1800 | 100 | 18 | **18** (expected 18, OK) | YES — 18/18 | YES — 18/18 | 0 | 0/0/0 |
| `gate1_final18_rescue_t0margin0.25` | 1800 | 100 | 18 | **18** (expected 18, OK) | YES — 18/18 | YES — 18/18 | 0 | 0/0/0 |

No duplicate `(mode, catalog_row, start_slot)` / `(mode, catalog_row,
label)` / `(mode, catalog_row, strategy_id)` keys in any source; every
`(mode, catalog_row)` cell has exactly the expected number of rows.
`strategy_id` gives `gate1_oracle_52` exactly the 52 expected
canonical strategies, each covering all 100 events — the raw
1184-label count is now understood and superseded, not treated as the
true strategy count.

Root cause of the `gate1_oracle_52` **label** instability (checked
directly, not inferred; this motivated building `strategy_id` rather
than trusting label): for every mode, exactly 3 of the 13 slots embed
a per-event numeric rho — slot 1 (`old_H0/rho_from_old_H0`), slot 2
(`old_H0/truth_rho`), and slot 8 (`truth/truth_rho`). These three
slots' *rule* is constant across events, but the resulting numeric
rho differs per event, so the raw label (e.g.
`old_H0_rho_1.0001e-07`) differs too, while `strategy_id` (e.g.
`old_H0/rho_from_old_H0`) does not. The other 10/13 slots per mode use
a fixed rho grid value and were already label-stable; `strategy_id`
is a distinct string spelling for those too (`old_H0/1e-06` vs. raw
label `old_H0_rho_1e-06`) but encodes the same strategy.

Component-wise active-bound rate (of the 4-parameter H0 mask,
`t0,u0,tE,rho`), reported both ways as required:

| source | per_fit t0/u0/tE/rho | winner t0/u0/tE/rho | winner success rate |
|---|---|---|---:|
| `gate1_oracle_52` | 0.0013 / 0.0000 / 0.0000 / 0.0650 | **0.0100 / 0.0000 / 0.0000 / 0.0000** | 1.0000 |
| `gate1_final18_base_t0margin0` | 0.0028 / 0.0000 / 0.0000 / 0.0194 | **0.0100 / 0.0000 / 0.0000 / 0.0000** | 1.0000 |
| `gate1_final18_rescue_t0margin0.25` | 0.0000 / 0.0000 / 0.0000 / 0.0233 | **0.0000 / 0.0000 / 0.0000 / 0.0000** | 1.0000 |

`results/gate1_oracle_active_bounds.csv` is the pre-existing,
independently computed winner-level active-bound audit for the
`gate1_oracle_52` winners only (built earlier by
`analyze_gate1_oracle_active_bounds.py`). This session's new
`winner_t0_active_rate = 0.0100 = 1/100` for `gate1_oracle_52`
reproduces that file's `t0_active` sum on the single point actually
compared: both identify the same one event, `catalog_row=36103`, as
the only `gate1_oracle_52` winner with an active `t0` bound. This is
a match on that specific count/event, not a full-table comparison of
every column or every event between the two files — no broader
row-by-row equivalence between `gate1_oracle_active_bounds.csv` and
the new audit output was checked. Separately (not a cross-check
against that pre-existing file, which only covers `gate1_oracle_52`),
the new audit finds the same event, `catalog_row=36103`, as the sole
`t0`-active winner for `gate1_final18_base_t0margin0` too — consistent
with, and using the same underlying data as, the checkpoint's
statement (§6) that row `36103` was the only `extreme100` event
requiring the t0 rescue.

`optimizer_success` is `True` for 100% of rows and 100% of winners in
all three sources — see §0.4b: excluded as a Phase A predictor for
lack of variation on this development sample, but this is **not** a
recommendation to drop the check in production or in an independent
sample.

**8. Interpretation**

- The three `source` values are not interchangeable "strategies" of
  one bigger multistart pool: `gate1_oracle_52` is a different,
  4x-larger, `t0_margin_factor=0` plan; the two `gate1_final18` runs
  share the same 18-strategy plan but differ only in `t0_margin_factor`
  (0 vs 0.25), and `gate1_final18_rescue_t0margin0.25` is a **global
  diagnostic run**, not a claim that production reruns all 100 events
  at margin 0.25 — see §0.4 for the exact validated policy (only
  `catalog_row=36103` is actually rescued).
- `(mode, strategy_id)` is the canonical semantic strategy key,
  proven stable **within and across all 100 events**, for all three
  sources (52/18/18 pairs, full coverage). It should be used in place
  of raw `label` for any cross-event grouping or matching in Phase
  A/B, because it expresses the semantics (anchor + rho rule)
  directly and does not implicitly rely on a fixed execution order
  being preserved.
- Precise three-way distinction (corrected from an earlier overly
  broad statement in this document that called `start_slot` "unsafe"
  for 3/13 slots — that conflated `start_slot` with raw `label`):
  - **raw `label`** is event-dependent, hence NOT a valid cross-event
    key, for exactly 3/13 slots per mode of `gate1_oracle_52` (the
    slots whose rho rule embeds a per-event numeric value: slots 1,
    2, 8 — see §7). It is fully stable for the other 10/13 slots and
    for all 18 slots of both `gate1_final18` sources.
  - **`start_slot`** IS a stable positional key for all 13/13 slots
    per mode of `gate1_oracle_52`, *given* the verified invariant
    that every `gate1_oracle_52` (mode, catalog_row) call produced
    exactly 13 rows in the fixed construction order (checked, not
    assumed — see §7 and the guard in
    `_gate1_oracle_52_strategy_id`). Under that invariant,
    `(mode, start_slot) -> (mode, strategy_id)` is constant across
    all 52 cases, including the 3 slots where raw `label` is not.
    `start_slot` is likewise stable for both `gate1_final18` sources.
  - **`strategy_id`** remains the preferred canonical key for Phase
    A/B regardless, because it is semantically self-describing and
    does not depend on an implicit ordering guarantee holding for
    every future run (a new run where `unique_rhos()` happens to
    collapse a slot for some event would silently break a
    `start_slot`-based join, while `strategy_id` is computed with an
    explicit guard that fails loudly instead — see
    `_gate1_oracle_52_strategy_id`).
- `optimizer_success` must be excluded as a standalone Phase A trigger
  feature (no variation on `extreme100`), but is retained as a
  pipeline sanity check to be exercised again on the Gate-3
  independent sample and in production.
- **Per-fit vs. winner active-bound rates tell different stories, and
  must not be conflated.** The per-fit `rho_active_rate` of 6.5%
  (`gate1_oracle_52`) / 2-2.3% (`gate1_final18` sources) reflects that
  some individual *exploratory* starts land on the rho bound; the
  winner-level `rho_active_rate` is **0.0000** in all three sources —
  no event's actually-selected H0 solution sits on a rho bound. This
  is exactly the distinction the checkpoint's separate active-bounds
  audit (`gate1_oracle_active_bounds.csv`) already captured for
  `gate1_oracle_52` winners, now reproduced independently and
  extended to the two `gate1_final18` sources. A high per-fit
  active-bound rate is therefore diagnostic of individual starts
  probing near a boundary, not evidence that the current `rho` domain
  is insufficient at the winner level; it does not by itself justify
  widening `rho` bounds (which remains out of scope, §0.5). `t0` is
  the only component with a nonzero winner-level active rate
  (`0.0100`, i.e. row `36103`), and it is already handled by the
  validated, full-18-fit-winner-triggered rescue.

**9. Decision**

**ACCEPTED** as the input table for Experiment/Phase A. Acceptance
rests on two independent results: (a) the hard integrity check passed
with zero violations for all three sources (hypothesis, status,
finiteness, `h0_start_plan`, `t0_margin_factor`, and the exact
100-event `catalog_row` set), so the table's basic contents are
trustworthy; and (b) the caveats below are treated as binding
constraints on how the table may be joined and grouped: use
`(source, mode, strategy_id)` as the strategy key (not raw `label`;
`start_slot` is also valid within `gate1_oracle_52`, per §8, but
`strategy_id` remains preferred); never pool `source` values as
equivalent strategies; treat `gate1_final18_rescue_t0margin0.25`
strictly as the diagnostic all-events rescue-domain run, not the
production policy itself.

**10. Consequence**

- Phase A's sequential-execution simulation must be built from the two
  `gate1_final18` sources specifically (`t0_margin0` as the base
  sequence, `t0_margin0.25` only for the one event where the existing,
  full-18-fit-winner-triggered t0-rescue is validated to apply), using
  `(mode, strategy_id)` in the plan's own fixed order — not from
  `gate1_oracle_52`.
- `gate1_oracle_52` remains reserved for two roles only: (a) the
  offline oracle chi2 per event (already available via
  `gate1_all_start_fits.csv` / `gate1B_final18_per_event.csv`), and
  (b) in Phase B, a candidate pool of additional
  `(mode, strategy_id)` strategies to test in a redesigned
  sequence/order.
- `optimizer_success` is dropped from the candidate-predictor list for
  Phase A (documented here so it is not re-tested from scratch later)
  but is NOT removed as a pipeline-level check going forward (§0.4b).
- `optimizer_active_mask` is retained as a Phase A candidate, but any
  trigger built on it must be evaluated, and reported, separately at
  the per-fit level and at the running-winner level, since they
  disagree sharply here (rho: 6.5% per-fit vs. 0% winner).
- Any Phase A/B proposal to trigger the `t0`-margin rescue from a
  **partial** (`k<18`) winner is explicitly flagged as a new,
  not-yet-validated policy (§0.4), to be evaluated against the oracle
  like any other candidate trigger — never presented as inheriting the
  checkpoint's existing t0-rescue validation.

**11. Next step**

Experiment A1 (Phase A), to be run as a **DIAGNOSTIC BASELINE** (its
purpose is to understand which observables predict
`delta_chi2_vs_oracle > 0.1`, not to propose a production sequence):
simulate the sequential execution of `gate1_final18` in its current
per-mode `(mode, strategy_id)` order (`gate1_final18_base_t0margin0`,
falling back to `gate1_final18_rescue_t0margin0.25` only for row
`36103`, per the validated full-winner t0 rule — see §0.4), and
compute, after each step, `Delta_spread`, running best-chi2
trajectory, cross-mode agreement in `(t0,u0,tE,rho)`, and
`optimizer_active_mask`-derived flags (kept separate from any
winner-level-only reasoning), then measure
`P(delta_chi2_vs_oracle > 0.1 | predictor)` empirically, without
presupposing the sign or threshold of any predictor, and without
presupposing that `gate1_final18`'s current order is a good order for
a sequential policy (that redesign is explicitly deferred to Phase
B, which may draw on all 52 `gate1_oracle_52` `(mode, strategy_id)`
candidates).

---

### Experiment A1 — Sequential H0 stopping diagnostics (DIAGNOSTIC BASELINE)

**Status: DIAGNOSTIC BASELINE.** This experiment characterizes
predictors; it does not propose or freeze any production stopping
rule or rescue trigger. No conclusion here authorizes reducing the
18-fit `gate1_final18` plan.

**1. Question**

Using only observables available from a sequential run of
`gate1_final18`'s own fixed 18-start plan, which predictors,
computed after each partial step k=1..18, are associated with the
running-best H0 solution still being more than 0.1 above the Gate-1
oracle? Separately: does operationalizing the checkpoint's validated
t0-rescue rule directly from the full-18-fit winner's own
`optimizer_active_mask` (rather than from a hardcoded event id)
correctly reproduce the checkpoint's result that only one
`extreme100` event needed it?

**2. Motivation**

This is the concrete first step toward a cheaper adaptive H0
strategy (checkpoint §22 / this document §0.2): before designing any
trigger, characterize empirically which signals available mid-run
actually correlate with "still wrong", without presupposing the
answer, and without smuggling in any oracle-derived or per-event
hardcoded information.

**3. Inputs**

- `results/gate1_oracle_diagnostics.csv`, sources
  `gate1_final18_base_t0margin0`, `gate1_final18_rescue_t0margin0.25`
  (both from Experiment 0, unmodified) and `gate1_oracle_52` (used
  only for offline labeling — see §5).
- New script:
  `validation/bounds_convergence/analyze_a1_sequential_h0_diagnostics.py`.

**4. Method**

*Sequence order (a stated A1 convention, not a validated production
order):* `gate1_final18` was historically run as four independent
per-mode multistarts, not one interleaved sequence. A1 concatenates
the four modes in the fixed order `physical, log_te, log_rho,
log_te_rho` and, within each mode, uses that mode's own `start_slot`
order — giving a global `sequence_index` 1..18 that is identical for
every event (guaranteed by the Experiment-0 result that
`(mode, start_slot) -> strategy_id` is constant across all 100 events
for this source). This is an explicit, documented choice for A1, not
a claim that this is the best order — Phase B is free to study
others.

*Per-step predictors (k=1..18, using only fits 1..k):* running best
chi2 (`chi2_best_k`), `delta_spread_k = chi2_second_best_k -
chi2_best_k` (undefined at k=1), `n_modes_seen_k`,
`n_modes_agreeing_k` (number of modes whose own best-so-far chi2 is
within 0.1 of the global best-so-far — a cross-coordinate-system
agreement proxy), `steps_since_improvement_k` (how many consecutive
recent steps failed to lower the running best), the winning row's
`optimizer_active_mask` split into `t0/u0/tE/rho_active_k`, its
`optimizer_success_k` / `optimizer_optimality_k`, and best-vs-second-best
parameter differences (`diff_t0_best2nd_k`, `diff_u0_best2nd_k`,
`absdiff_log10_tE_best2nd_k`, `absdiff_log10_rho_best2nd_k`).

*Offline-only labels:* `chi2_oracle_52` (pooled minimum over the
52-fit oracle for that event), `delta_chi2_vs_oracle_k = chi2_best_k
- chi2_oracle_52`, `fail_k = delta_chi2_vs_oracle_k > 0.1`. These are
never inputs to any predictor computation.

*Full-18 policy, rescue operationalized from active_mask:* for each
event, take the winner of all 18 base-plan fits; parse ITS
`optimizer_active_mask`; if and only if its `t0` component is active,
look up that same event's `gate1_final18_rescue_t0margin0.25` 18
fits, take their winner, and set the policy chi2 to the minimum of
the two winners (else the policy chi2 is just the base-18 winner).
Per the explicit instruction for this experiment, **the rescue is
applied only at the full k=18 boundary** — no rescue of any kind is
applied for k<18, even if a partial winner already shows an active
t0 bound; testing an earlier trigger point is explicitly left to
Phase B as a new, separately-evaluated policy.

**5. Information allowed in production**

Per-step predictors use only fields already present in
`gate1_oracle_diagnostics.csv` for fits 1..k of the event's own base
run (chi2, t0/u0/tE/rho, `optimizer_success`, `optimizer_optimality`,
`optimizer_active_mask`, mode) — exactly what a real sequential run
would have on hand after k starts. `catalog_row` is read only to
group rows by event and to join the rescue table; it is never used
as a condition in the rescue rule or in any predictor (in particular,
`catalog_row==36103` never appears as a rule anywhere in the script —
the single rescued event emerges from the active-mask condition, see
§7). `chi2_oracle_52` and everything derived from it
(`delta_chi2_vs_oracle_k`, `fail_k`) are strictly offline-only
evaluation labels.

**Terminology: two distinct reference objects, never call both
"the oracle" unqualified.**

1. **Base-domain oracle, `chi2_oracle_52`.** The pooled minimum over
   the 52-fit oracle, all of it computed at `t0_margin_factor=0`. This
   is the correct and only reference for evaluating the k=1..18 base
   sequence (`delta_chi2_vs_oracle_k`, `fail_k`, the `N(fail_k)` by k
   table), because every one of those base-plan fits also shares
   `t0_margin_factor=0` — comparing like domains to like domains.
2. **Final validated H0 reference (the policy itself, not a wider
   oracle).** The base-18 winner, with the checkpoint's t0 rescue
   applied on top of it when that winner's own `active_mask` shows
   `t0` active (§4, operationalized from `optimizer_active_mask`, not
   from `catalog_row`). This reference can legitimately explore
   `t0_margin_factor=0.25` for the one event it rescues.
   `delta_chi2_vs_oracle_final` therefore compares object 2 (the final
   policy, domain-mixed by construction for the rescued event) against
   object 1 (the base-domain oracle, always `t0_margin_factor=0`). For
   `catalog_row=36103` this is precisely why
   `delta_chi2_vs_oracle_final < 0` is expected and correct, not an
   inconsistency: the final policy is allowed to use a strictly wider
   `t0` domain than `chi2_oracle_52` was ever computed in, for that
   one event. `chi2_oracle_52` is not, and was never intended to be, a
   `t0_margin=0.25` oracle. **Phase B must keep these two references
   separate**: any k=1..18 per-step predictor analysis must compare
   against the base-domain oracle only (object 1); only a genuine
   final-policy comparison (object 2, after any rescue this
   experiment or Phase B applies) may legitimately see negative
   deltas for rescued events.

**6. Metrics**

`N(fail_k)` by k; empirical `P(fail_k | predictor bucket)` per
predictor (pooled over all 1800 `(catalog_row, k)` pairs); the
full-18-policy `N(fail_final)`, max `delta_chi2_vs_oracle_final`, and
the identity/count of rescued events (checked against, not derived
from, the checkpoint's `catalog_row=36103` statement).

**7. Result**

Ran `analyze_a1_sequential_h0_diagnostics.py` (100 events x 18 steps
= 1800 per-step rows).

*Full-18 policy, rescue from active_mask alone:* exactly **1** event
had an active `t0` bound on its base-18 winner —
`catalog_row=36103` — matching the checkpoint's statement (§6 of
`VALIDATION_STATUS_2026-09-14.md`) without that event id ever being
read by the script. After applying the rescue for that one event,
`N(fail_final, delta>0.1) = 0` and `max(delta_chi2_vs_oracle_final) =
0.0271387`, reproducing the checkpoint's Gate-1 result. This does
**not** mean `delta_chi2_vs_oracle_final` is exactly `0` for all 100
events — full distribution, read directly from
`a1_sequential_h0_full18_policy.csv`:

| min | median | max | N(delta>0) | N(delta>1e-6) | N(delta>0.01) | N(delta>0.1) |
|---:|---:|---:|---:|---:|---:|---:|
| -6.586858 | 0.000000 | 0.027139 | 22 | 20 | 5 | **0** |

The minimum is negative (`catalog_row=36103`, the rescued event) by
construction, per the base-domain-oracle vs. final-policy distinction
in §5 above: its `t0_margin=0.25` rescue chi2 is compared against
`chi2_oracle_52`, the **base-domain** (`t0_margin_factor=0`) oracle,
which never explored the wider rescue domain for that event — a
negative delta here reflects the final policy legitimately using a
wider `t0` domain than the base-domain oracle, not an error, and not
evidence that `chi2_oracle_52` is somehow wrong. The 5 events with
`delta_chi2_vs_oracle_final > 0.01` (all still `< 0.1`, so all still
pass the adopted tolerance) are:

| catalog_row | chi2_policy_final | chi2_oracle_52 | delta |
|---:|---:|---:|---:|
| 71949 | 1486.571357 | 1486.554579 | 0.016777 |
| 81185 | 3289.732938 | 3289.722289 | 0.010648 |
| 81449 | 397.352111 | 397.332558 | 0.019553 |
| 557806 | 447.776362 | 447.755102 | 0.021261 |
| 567691 | 324.894979 | 324.867840 | 0.027139 |

The correct summary is therefore **0 events with `delta_chi2 > 0.1`
but `max delta_chi2 ≈ 0.0271387` (not 0)**. The breakdown behind
`N(delta>0)=22`: for the 99 non-rescued events, `chi2_policy_final =
chi2_base18_winner ≥ chi2_oracle_52` by construction (the 18-strategy
plan is a subset of the same-domain 52-strategy oracle), so their
delta is always `≥ 0`; 77 of those 99 match the oracle exactly
(`delta = 0`) and the remaining 22 have small positive delta, up to
`0.027139`, still below the `0.1` tolerance. The 1 rescued event
(`catalog_row=36103`) is the sole negative-delta case (explained
above), giving `100 - 22 = 78` events with `delta ≤ 0` overall
(77 exact ties + 1 negative). "0 failures at the 0.1 tolerance" must
not be read as "delta is 0 for every event".

Separately: `N(fail)` at k=18 **without** any rescue was already 0 —
the t0-active condition triggered the rescue for row 36103 even
though its un-rescued delta was already ≤0.1; the rescue rule is a
conservative boundary-condition trigger, not one calibrated to fire
exactly when `delta_chi2>0.1` would otherwise occur, and the two
criteria are not the same thing.

*`N(fail_k)` by k* (pure running-min over the stated sequence order,
no rescue applied at any k):

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N(fail) | 54 | 38 | 29 | 25 | 24 | 22 | 21 | 18 | 16 | 14 | 13 | 10 | 9 | 8 | 7 | 4 | 1 | 0 |

For context only (not a claim of equivalence: the checkpoint's
coverage curve, §8, is the *best possible* fixed subset of size k
chosen by MILP over the full 52-strategy oracle, while this row is
one specific fixed *prefix* of `gate1_final18`'s own particular
18-of-52 selection and order): checkpoint coverage-curve N(delta>0.1)
at the same k values is `51, 37, 27, 21, 17, 14, 12, 10, 9, 8, 7, 6,
5, 4, 3, 2, 1, 0`. The two tracks are close at every k (this
particular sequence is not badly inefficient), with the MILP-optimal
subset consistently at or below the sequential prefix, as expected
since it is chosen with full hindsight over all 52 strategies rather
than following one fixed plan's order. The single k=17 failure
(`N(fail)=1`) is `catalog_row=578896` — one of the checkpoint's own
named examples of large cross-mode disagreement (§4) — resolved only
by the 18th, single `log_te_rho` start.

*Empirical `P(fail_k | predictor bucket)`, pooled over all 1800
`(catalog_row, k)` rows* (full table in
`results/a1_predictor_vs_failure.csv`):

| predictor | bucket | n_obs | fail_rate |
|---|---|---:|---:|
| `n_modes_agreeing_k` | 1 mode | 1041 | 0.249 |
| `n_modes_agreeing_k` | 2 modes | 470 | 0.072 |
| `n_modes_agreeing_k` | 3 modes | 283 | 0.071 |
| `n_modes_agreeing_k` | 4 modes | 6 | 0.000 |
| `steps_since_improvement_k` | 0 | 306 | 0.366 |
| `steps_since_improvement_k` | 1-2 | 394 | 0.236 |
| `steps_since_improvement_k` | 3-5 | 395 | 0.152 |
| `steps_since_improvement_k` | 6+ | 705 | 0.068 |
| `delta_spread_k` | [0, 0.01] | 776 | 0.075 |
| `delta_spread_k` | (0.01, 0.1] | 331 | 0.109 |
| `delta_spread_k` | (0.1, 1] | 195 | 0.159 |
| `delta_spread_k` | (1, 10] | 152 | 0.178 |
| `delta_spread_k` | (10, 100] | 118 | 0.432 |
| `delta_spread_k` | (100, inf) | 128 | 0.438 |
| `delta_spread_k` | NA (k=1) | 100 | 0.540 |
| `winner_t0_active_k` | 0 | 1782 | 0.175 |
| `winner_t0_active_k` | 1 | 18 | 0.056 |
| `winner_rho_active_k` | 0 (always) | 1800 | 0.174 |
| `winner_optimizer_success_k` | True (always) | 1800 | 0.174 |

**8. Interpretation**

**Overarching methodological warning (applies to every predictor
below, not just `delta_spread_k`): all three of `delta_spread_k`,
`n_modes_agreeing_k`, and `steps_since_improvement_k` are confounded
with `k` itself AND with the fixed, mode-blocked execution order
defined in §4 (`physical`(3) → `log_te`(6) → `log_rho`(8) →
`log_te_rho`(1)).** Because the sequence is not an interleaving of
modes but four contiguous blocks, `n_modes_seen_k` (and therefore
`n_modes_agreeing_k`) can only step from 1→2→3→4 at the fixed block
boundaries `k=4, 10, 18` — it is mechanically impossible to "see" a
second mode before `k=4` under this order, regardless of how the
first mode's fits actually behaved. Likewise `steps_since_improvement_k`
and `delta_spread_k` both trend with `k` simply because more fits have
had a chance to find or confirm the running best. None of the three
pooled associations reported below has been shown to carry
information beyond "how far into this particular fixed order are we,
and which block boundary have we crossed" — this has **not** been
tested at fixed cost, nor under an alternative order or start subset.
A1 is **diagnostic / hypothesis-generating only**; establishing
genuine, order-independent predictive power is explicitly deferred to
Phase B, which must evaluate these (and other) predictors at matched
`k`/cost across alternative orders or start sets before any of them
can be treated as informative in their own right.

- `n_modes_agreeing_k` and `steps_since_improvement_k` show the
  strongest, most monotonic association with failure, in the
  *expected* direction (more independent agreement, or more
  repeated confirmation without improvement, associates with lower
  failure): both are genuine hypotheses worth carrying into Phase B,
  not yet validated triggers, and both are subject to the k/block-order
  confound above. **This is a marginal, pooled association, not a
  per-event rule**: `catalog_row=578896` (the single k=17 failure,
  §7) is a direct counterexample — `n_modes_agreeing_k` stays at
  exactly 1 for all 18 steps (the other discovered modes' own best
  chi2 never comes within 0.1 of the running best), yet the event
  still resolves correctly at k=18, because the single `log_te_rho`
  start lands on a materially better minimum on its first and only
  try, not because multiple coordinate systems converged to
  agreement. A future trigger built on `n_modes_agreeing_k` alone
  would need a fallback for exactly this kind of case.
- `delta_spread_k` also increases with failure rate, but is heavily
  confounded with `k` itself (both shrink as more fits accumulate;
  `NA (k=1)` alone has fail_rate 0.540, matching `N(fail_k=1)=54`
  exactly). Its apparent predictive power in this pooled, marginal
  table has **not** been separated from a pure "more fits done so
  far" effect — this must be checked (e.g. conditioning on k, or a
  multivariate model) before treating `delta_spread_k` as informative
  beyond what `k` alone already tells you. No claim is made here
  about `delta_spread_k` having power independent of `k`.
- `winner_t0_active_k` shows a **lower**, not higher, failure rate
  when active (0.056 vs 0.175) — the opposite of a naive "active
  bound implies bad fit" prior, and consistent with §7's finding that
  the checkpoint's t0-rescue trigger is a conservative
  boundary-condition safeguard rather than a `delta_chi2>0.1`
  predictor. This is exactly the kind of non-presupposed-direction
  result the experiment was designed to surface, and it is a caution
  against using "active bound therefore untrustworthy" as an
  unexamined heuristic for any parameter.
- `winner_rho_active_k` is 0 for every one of the 1800 rows: within
  `gate1_final18_base_t0margin0`'s own running-best solution, `rho`
  is never at a bound at any step — consistent with Experiment 0's
  winner-level finding (§7-§8) and reinforcing that the per-fit rho
  active rate (individual exploratory starts) is not informative
  about the winner.
- `winner_optimizer_success_k` is constant `True`; it carries no
  information in this dataset (consistent with Experiment 0, §0.4b)
  and is not a usable Phase-A predictor here either, for the same
  reason: no variation to learn from on `extreme100`.
- Operationalizing the t0-rescue rule purely from the k=18 winner's
  own `active_mask` reproduced the checkpoint's single-event result
  (`catalog_row=36103`) without reading any event id — this is a
  successful reproducibility check on the *existing validated* rule,
  not a new result about adaptivity.

**9. Decision**

**DIAGNOSTIC ONLY / hypothesis-generating.** No predictor here is
accepted, rejected, or frozen as a trigger, and none of them —
including `n_modes_agreeing_k` and `steps_since_improvement_k` — has
demonstrated predictive power independent of `k` and the fixed
mode-blocked order (§8's overarching warning). `n_modes_agreeing_k`
and `steps_since_improvement_k` are the most promising *candidates*
to carry into Phase B precisely because of this experiment; they are
not validated triggers. `delta_spread_k` additionally needs
deconfounding from `k` before its apparent signal can be trusted at
all; `winner_t0_active_k` (at partial k) and
`winner_optimizer_success_k` are not usable as early-stopping signals
on this evidence. The full-18 active-mask-triggered rescue rule is
CONFIRMED to reproduce the checkpoint's validated result (§7,
including the base-domain-oracle-vs-final-policy distinction in §5)
and is safe to keep exactly as specified (applied only at k=18).

**10. Consequence**

- No change to the current 18-fit `gate1_final18` production-candidate
  plan. It remains the validated baseline.
- Phase B must treat `n_modes_agreeing_k` and
  `steps_since_improvement_k` as candidate predictors ONLY, and must
  test whether either carries predictive power independent of `k` and
  of this particular mode-blocked order — e.g. by evaluating them at
  fixed `k`/cost under a different order or a different start subset,
  or with an explicit multivariate model that includes `k` (and, for
  `n_modes_agreeing_k`, block position) as a control variable — before
  either is treated as informative on its own. The same applies, more
  strongly, to `delta_spread_k`, which must additionally be
  deconfounded from `k` before its apparent signal can be trusted at
  all. None of these three may be reused as-is from this pooled
  marginal table as evidence of standalone predictive value.
- Any proposal to fire the t0 rescue (or any new rescue) before k=18
  must be written up and evaluated against the oracle as an
  explicitly new policy — this experiment deliberately did not do
  that, per instruction.
- Phase B evaluation must keep the two references from §5 separate:
  compare candidate sequential policies' intermediate steps against
  the base-domain oracle (`chi2_oracle_52`), and only compare a
  policy's final, possibly-rescued result against a like-for-like
  final reference — never mix the two under one unqualified "oracle".

**11. Next step**

Phase B: design candidate sequential policies (start order,
stopping rule, rescue conditions) using `n_modes_agreeing_k` and
`steps_since_improvement_k` as candidate hypotheses to test — not
pre-validated triggers — explicitly checking for the k/block-order
confound identified in §8 (e.g. by testing them at matched cost under
an alternative order or start subset drawn from the 52-strategy
pool), deconfound `delta_spread_k` from `k` before deciding whether to
include it, evaluate candidates against the base-domain 52-fit oracle
for intermediate steps and against the final validated H0 reference
for policy-level results (§5 terminology), with the full metric set
requested (mean/percentile `N_fits`, `N(delta>0.1/1/10)`, max delta,
false/missed-rescue rate), and only then compare cost/robustness
against the fixed 18-fit baseline.

---

### Experiment B1 — Fixed-budget base sets and fixed-cost predictors (DIAGNOSTIC / CANDIDATE GENERATION)

**Status: DIAGNOSTIC / CANDIDATE GENERATION**, with one component
(`diff_u0_signed`, and the always-zero-variance `winner_u0/tE/rho_active`
predictors) **REJECTED** as fixed-cost predictors on the evidence
below. This experiment does not select, tune, or freeze any
stopping/rescue policy. `extreme100` remains the development sample;
nothing here is claimed to generalize without Gate 3.

**1. Question**

At a fixed H0 fit budget `k` (k=2..6), (a) which small sets of the 52
canonical `(mode, strategy_id)` strategies come closest to the
base-domain oracle, across the full severity profile
(`N(delta>0.1)`, `N(delta>1)`, `N(delta>10)`, `max delta`) rather than
`N(delta>0.1)` alone; and (b) holding that budget (and, per candidate,
its exact composition) fixed, do observables computable only from
that budget's own fits carry information about whether the resulting
event is still `>0.1` from the base-domain oracle — beyond what
Experiment A1 already showed is confounded with `k` and with A1's
particular mode-blocked order?

**2. Motivation**

Direct continuation of the Phase-B mandate (§0.2, Experiment A1 §9):
before any adaptive/sequential policy can be designed (Phase B2), we
need (a) a small pool of genuinely competitive fixed-size candidate
sets — not assumed to be `gate1_final18`'s own prefixes — and (b)
predictors whose apparent power survives being evaluated at constant
cost, since A1 explicitly could not separate predictor signal from
"how many fits have we done" for its own sequential design.

**3. Inputs**

- `results/gate1_oracle_diagnostics.csv`, `source=="gate1_oracle_52"`
  (Experiment 0's canonical, integrity-checked table; not modified).
- `results/gate1_h0_coverage_curve.csv` (pre-existing exact-MILP
  maximum-coverage-at-0.1 result, k=1..18, used only as an
  independent cross-check — not re-derived, not blindly trusted
  either).
- New scripts:
  `validation/bounds_convergence/analyze_b1_fixed_budget_base_sets.py`
  and
  `validation/bounds_convergence/analyze_b1_fixed_cost_predictors.py`.

**4. Method**

*B1a — fixed-budget base-set search.* For each `k` in `{2,3,4,5,6}`,
**full exhaustive** enumeration of every `C(52,k)` subset (no MILP, no
heuristic, no early stopping, for any `k` studied — see script
docstring for the two-pass vectorized implementation that keeps this
cheap even at `C(52,6)=20,358,520`; measured wall time on this
machine: pass 1 (primary objective only) 22.3s at k=6, pass 2 (full
metrics on the top 300 candidates) 2.9s at k=6, well under a second
for k≤5). Candidates are ranked **lexicographically**, exactly per
the requested correction: primary `N(delta>0.1)` (minimize), then
among primary-tied subsets, secondary `N(delta>1)`, tertiary
`N(delta>10)`, quaternary `max delta`, quinary summed positive excess
— never calling the primary-only winner "the best set" unqualified.
Per `k`, the script saves every exactly lexicographic-tied-optimal
subset plus up to 5 additional near-optimal subsets that introduce a
mode-composition signature (e.g. `{physical:2}` vs
`{log_rho:1, physical:1}`) not already saved, so that predictor
stability can be checked across different compositions, not just at
one arbitrary optimal point (§7-§8).

Selecting these fixed sets from `chi2_oracle_52` is a **design-time,
offline** choice (§5) — legitimate under the same logic already
established for `gate1_final18` itself.

*B1b — fixed-cost predictors.* For every saved `(k, candidate_rank)`
set and every event, using ONLY that candidate's own `k` fits for
that event: `chi2_best`, `chi2_second_best`, `delta_spread`, `chi2`
dispersion (`std`, `range`) among the `k` fits, `n_modes_in_set`,
`n_modes_agreeing` (modes whose own best-within-set chi2 is within
0.1 of the overall best-within-set chi2 — same definition as A1's
`n_modes_agreeing_k`, now evaluated on a fixed unordered set instead
of an incrementally-revealed sequence), the winning row's
`optimizer_active_mask` (split into t0/u0/tE/rho active),
`optimizer_optimality`, `optimizer_success`, and best-vs-second-best
parameter agreement using **physically-motivated, degeneracy-aware**
diagnostics per the explicit correction — not raw differences alone:
`diff_abs_u0 = ||u0_best|-|u0_2nd||` (insensitive to the `u0 -> -u0`
mirror degeneracy) alongside the raw signed `diff_u0_signed`;
`abs_log_ratio_tE = |log(tE_best/tE_2nd)|` and
`abs_log_ratio_rho = |log(rho_best/rho_2nd)|` (log-ratio, since `tE`
and `rho` are strictly positive and span orders of magnitude);
`diff_t0_raw` and `diff_t0_norm_by_tE = diff_t0_raw / tE_best`
(t0 offsets are only meaningful relative to the event's own
timescale). `steps_since_improvement` is explicitly **not** computed
here — B1's sets are evaluated as unordered batches, and this
predictor requires an execution order that only Phase B2 will define.

For every predictor, safe (`delta<=0.1`) vs unsafe (`delta>0.1`)
distributions are compared **within one fixed `(k, candidate_rank)`**
— never pooled across `k` the way A1's marginal tables were. Bucketed
predictors (integer/binary) get a frequency table
(`n_obs, n_safe, n_unsafe, unsafe_rate`); continuous predictors get
group means/medians plus a rank-based separation score
(`auc_unsafe_gt_safe`, the Mann-Whitney probability that a random
unsafe-event value exceeds a random safe-event value; `0.5` = no
separation). `N_safe`/`N_unsafe` are reported on every single row of
`b1_predictor_vs_failure_fixed_k.csv`, per instruction — no bare rate
without its denominators. No threshold is fit or proposed anywhere in
this step.

**5. Information allowed in production**

Every B1b predictor is a function only of the `k` fits belonging to
one fixed, pre-chosen candidate set, for one event — exactly what a
production run using that fixed set would have on hand (no
`catalog_row`-based branching, no oracle access at decision time).
`chi2_oracle_52` is read only to build the offline `safe`/`unsafe`
label. Choosing WHICH fixed set of `k` strategies to use at all
(B1a) is a design-time decision made once, offline, on the
development sample — it is not a per-event runtime decision and does
not leak the oracle into any simulated production choice. This
remains `extreme100` development-set analysis throughout; nothing
here is validated for generalization.

**6. Metrics**

B1a: `N(delta>0.1/1/10)`, `max delta`, summed/mean positive excess,
per candidate; agreement with the pre-existing MILP coverage curve at
the primary objective. B1b: `N_safe`, `N_unsafe`, unsafe-rate per
bucket, and `auc_unsafe_gt_safe` for continuous predictors, per fixed
`(k, candidate_rank)`; residual-failing-event counts and per-strategy
rescue rates from the rescue matrix.

**7. Results**

*B1a.* Every `k`'s lexicographic-optimal primary value (`N(delta>0.1)`)
**exactly matched** the pre-existing `gate1_h0_coverage_curve.csv`
MILP result (37, 27, 21, 17, 14 for k=2..6) — this exhaustive,
independently-implemented search agrees exactly with the earlier,
differently-computed MILP on every `k` tested.

| k | lex-optimal set (mode/strategy_id) | N(>0.1) | N(>1) | N(>10) | max delta | range across the 6 saved candidates (N(>0.1)) |
|---|---|---:|---:|---:|---:|---|
| 2 | physical/truth/0.1 + physical/truth/1 | 37 | 24 | 14 | 24836.8 | 37–46 |
| 3 | log_rho/truth/1 + physical/old_H0/1 + physical/truth/0.1 | 27 | 18 | 7 | 11777.6 | 27–31 |
| 4 | log_rho/truth/0.1 + log_rho/truth/1 + physical/old_H0/1 + physical/truth/0.1 | 21 | 14 | 5 | 969.6 | 21–24 |
| 5 | + log_te/truth/0.01 (5-strategy set) | 17 | 10 | 3 | 969.6 | 17–19 |
| 6 | + log_te/old_H0/truth_rho (6-strategy set) | 14 | 9 | 4 | 969.6 | 14–16 |

(Full sets, all 30 saved candidates with exact strategy lists and
severity metrics: `results/b1_fixed_budget_candidate_sets.csv`.) The
5 alternative compositions saved per `k` are close but strictly worse
on the primary objective (e.g. k=4: 21 vs 22–24) — no alternative
composition ties the lexicographic optimum at any `k` here, but they
remain useful as a composition-diversity contrast set for B1b (below).

Residual-failing-event counts under the lexicographic-optimal
candidates equal their `N(delta>0.1)` exactly (e.g. 21/100 events for
k=4), confirming `b1_residual_failing_events.csv` is internally
consistent with `b1_fixed_budget_candidate_sets.csv`.

That every one of those 21 residual events can be rescued by adding
*some* single strategy from the remaining 48 is **not itself an
empirical finding** — it follows directly from the definition of
`chi2_oracle_52` as the pooled minimum over all 52 strategies: if a
21-of-52 result differs from that pooled minimum, the strategy (or
strategies) achieving the minimum necessarily lies outside the
21-set and, added back in, necessarily reproduces it. The informative
result from `b1_rescue_matrix.csv` (`results/b1_rescue_strategy_summary.csv`
for the per-strategy counts) is the **distribution** of that
rescuability, not its mere existence: for k=4's optimal set, **no
single additional strategy rescues more than 4/21≈19%** of the 21
residual events (top two: `log_te/truth/1` and `log_te/truth/0.01`,
each rescuing 4/21) — i.e. the 21 residual events do not cluster
around one dominant fix; different events need different additional
strategies. This is what tells us a single fixed fallback strategy is
unlikely to be a competitive rescue design, and it is the direct
input B2 needs to design a rescue SET (covering the full spread of
residual events) rather than one fallback fit — the coverage
distribution itself, in `b1_rescue_strategy_summary.csv`, is what B2
should consult, not just this one summary statistic.

*B1b — predictors at fixed k, for the lexicographic-optimal candidate
of each k* (full table, all 6 candidates x 5 k values:
`results/b1_predictor_vs_failure_fixed_k.csv`):

| predictor | AUC(unsafe>safe), k=2..6 | direction |
|---|---|---|
| `abs_log_ratio_tE` | 0.591, 0.751, 0.778, 0.734, 0.762 | consistently >0.5; the most promising and comparatively stable candidate here |
| `delta_spread` | 0.627, 0.730, 0.725, 0.669, 0.746 | consistently >0.5; moderate and comparatively stable |
| `abs_log_ratio_rho` | 0.526, 0.711, 0.703, 0.681, 0.689 | >0.5 for k≥3, weak at k=2 |
| `diff_t0_raw` | 0.580, 0.704, 0.708, 0.687, 0.710 | consistently >0.5, moderate |
| `diff_t0_norm_by_tE` | 0.591, 0.655, 0.638, 0.594, 0.623 | consistently >0.5, weaker |
| `chi2_std_among_set` | 0.627, 0.595, 0.540, 0.523, 0.590 | weak, fading toward k=5 |
| `chi2_range_among_set` | 0.627, 0.599, 0.545, 0.527, 0.586 | weak, fading toward k=5 |
| `diff_abs_u0` | 0.531, 0.609, 0.600, 0.616, 0.614 | weak but consistently >0.5 |
| `winner_optimizer_optimality` | 0.616, 0.527, 0.561, 0.544, 0.644 | unstable, no consistent direction |
| `diff_u0_signed` | 0.387, 0.511, 0.512, 0.525, 0.368 | **no signal / not consistently >0.5** |

**Statistical caution on the AUC values above**: `n_unsafe` shrinks
from 37 (k=2) to 14 (k=6), so every AUC at higher `k` is estimated
from a small sample and carries a correspondingly wide uncertainty;
none of these AUC values should be read as a precise, stable estimate
of separation power, only as a same-direction, order-of-magnitude
signal across `k`. No predictor here is described as "strong" for
this reason — see §9 for the qualified language used in the decision.

Bucketed `n_modes_agreeing` (lexicographic-optimal candidate): unsafe
rate drops sharply with more agreeing modes at every k — e.g. k=4:
30.5% (1 mode) → 7.3% (2 modes); k=6: 20.0% (1) → 17.9% (2) → **0.0%**
(3, n=27). `winner_t0_active`: only 1/100 events per `k` (always the
same structural rarity as Experiment 0/A1), and that one event is
always safe — too sparse to conclude anything, consistent with A1's
counter-intuitive finding. `winner_u0_active`, `winner_tE_active`,
`winner_rho_active`: **zero variance** across all 3000
`(k, candidate_rank, event)` rows — the winner never sits on those
bounds regardless of which fixed set is used. `winner_optimizer_success`:
constant `True` throughout, as in every prior experiment.

*Composition-dependence check* (per instruction, point 3): comparing
`abs_log_ratio_tE` and `delta_spread` AUC across all 6 candidates at
k=4 shows both stay in a fairly narrow band (0.68–0.78 and 0.67–0.75
respectively) regardless of composition — comparatively stable, not
described here as "robust" given the small-sample caution above.
`n_modes_agreeing`, by contrast, is **not** stable
across compositions at k=4: candidate 1 (optimal) shows a clean
monotonic drop (30.5%→7.3%), candidate 3 is much weaker
(23.8%→18.8%), and candidate 4 actually **reverses** at its
3-mode-agreement bucket (22.4%→21.1%→40.0%, though that last bucket
has only n=5 — likely noise, but not demonstrated to be noise here).

**8. Interpretation**

- `abs_log_ratio_tE` and `delta_spread` are the most promising /
  moderate and comparatively stable fixed-cost predictor candidates
  identified so far — not "strong predictors": their separation is
  not an artifact of `k` (by construction, `k` is fixed within each
  comparison) and is reasonably stable across different set
  compositions at a given `k`, which is a methodologically cleaner
  result than A1's pooled, k-confounded association for
  `delta_spread`, but the AUC values themselves are small-sample
  estimates (§7) and B1 remains DIAGNOSTIC / CANDIDATE GENERATION —
  no trigger is validated here.
- `n_modes_agreeing` (bucketed) shows the sharpest unsafe-rate
  contrast of any predictor tested, but its stability across
  different compositions of the same `k` is NOT established here —
  it works cleanly for some sets and only weakly, or with a reversal
  in a small bucket, for others. This must not be read as a validated
  trigger; it needs testing across more/different compositions in
  B2 before being trusted the way `abs_log_ratio_tE`/`delta_spread`
  currently can be.
- `diff_u0_signed` (AUC clustering around/below 0.5, including two
  values `<0.5`) shows **no usable signal**, while the
  degeneracy-aware `diff_abs_u0` shows a weak but consistently `>0.5`
  signal — direct empirical confirmation that the raw signed
  difference was the wrong quantity to use, exactly the failure mode
  the instruction anticipated.
- `winner_u0_active`, `winner_tE_active`, `winner_rho_active` carry
  zero information (no variance) at any tested `k` or composition —
  consistent with, and now extended from, Experiment 0/A1's
  winner-level finding that these bounds are essentially never active
  for the winning H0 solution in `extreme100`.
- `winner_optimizer_success` remains constant `True` in every
  experiment run so far on `extreme100` (Experiment 0, A1, B1) — it
  is not a usable development-sample predictor, but per §0.4b it must
  remain a pipeline sanity check for Gate 3 / production, where it
  may actually vary.
- The rescue-matrix **distribution** (no single strategy rescues more
  than ~19% of a candidate's residual events, not merely "some
  strategy always exists" — which is definitional, see §7) is an
  important input for B2 design: a fixed small base set plus a single
  fixed fallback strategy is unlikely to be competitive; B2 will
  likely need either a small rescue SET or an event-conditional choice
  among rescue candidates, driven by an actually-validated predictor
  (not yet available at that confidence level here).

**9. Decision**

**DIAGNOSTIC / CANDIDATE GENERATION** for the base-set search (B1a)
and for `abs_log_ratio_tE`, `delta_spread`, `abs_log_ratio_rho`,
`diff_t0_raw`, `diff_t0_norm_by_tE`, `diff_abs_u0`, and bucketed
`n_modes_agreeing` (all show `>0.5` separation at fixed cost, none
validated as a threshold or trigger). **REJECTED** as fixed-cost
predictors: `diff_u0_signed` (no consistent signal; superseded by
`diff_abs_u0`), `winner_u0_active`/`winner_tE_active`/`winner_rho_active`
(zero variance), and `winner_optimizer_success` (zero variance on
this development sample; retained only as a pipeline sanity check,
not a predictor). `winner_optimizer_optimality` and
`chi2_std/range_among_set` are **INCONCLUSIVE** — weak and
k-unstable, not clearly usable but not cleanly rejected either.

**10. Consequence**

- B2 should prioritize `abs_log_ratio_tE` and `delta_spread` as the
  leading fixed-cost trigger candidates, given their AUC stability
  across both `k` and composition.
- B2 may still consider `n_modes_agreeing` but must test its
  stability more broadly (more compositions, and eventually the
  independent Gate-3 sample) before relying on it alone.
- B2's rescue-set design must account for the finding that no single
  additional strategy rescues most residual events under a small base
  set — `results/b1_rescue_matrix.csv` /
  `results/b1_rescue_strategy_summary.csv` are the direct input for
  choosing a rescue SET (or an event-conditional rescue choice) rather
  than one fixed fallback strategy.
- `diff_u0_signed` should be dropped from further predictor
  candidate lists; `diff_abs_u0` may replace it if a u0-based
  predictor is wanted at all (it is currently weak either way).
- No change to `gate1_final18` or any production/fitter code. This
  remains entirely offline, development-set (`extreme100`) evidence.

**11. Next step**

Phase B2: design candidate sequential/adaptive policies that combine
a small fixed base set (from `b1_fixed_budget_candidate_sets.csv`, k
in the 4–6 range looks most promising given its severity profile) with
a stopping/rescue decision built on `abs_log_ratio_tE` and/or
`delta_spread` (and, cautiously, `n_modes_agreeing`), plus the
existing validated full-set-winner t0 rescue rule (§0.4, unchanged),
and a rescue set informed by `b1_rescue_strategy_summary.csv` for
whichever events remain hard. Every B2 policy must be evaluated with
the full metric set already specified (mean/percentile `N_fits`,
`N(delta>0.1/1/10)`, max delta, false/missed-rescue rate) against the
base-domain oracle for intermediate steps and the final validated H0
reference for policy-level results (§5 terminology from Experiment
A1), and no threshold may be selected by looking only at the mean.

---

### Experiment B1c — Simulation-aware event difficulty (DIAGNOSTIC / CANDIDATE GENERATION)

**Status: DIAGNOSTIC / CANDIDATE GENERATION**, with an explicit
**REJECTED** finding for one specific claim (pre-H0 simulation
features alone, or combined with B1's fit diagnostics, as a useful
exploratory classifier at N=100 — see §9). `extreme100` remains the
development sample; nothing here authorizes a production policy or
Phase B2 design decision by itself.

**1. Question**

Do truth/simulation properties and generated-light-curve properties —
all available in memory after simulating an event and *before* any H0
fit runs — predict (a) whether Experiment B1's small fixed H0 base
sets already reach the base-domain oracle within 0.1, (b) which
coordinate-mode family of additional strategy would rescue the event
if not, and (c) whether this "pre-H0" information adds anything beyond
what Experiment B1's own fixed-cost fit diagnostics already show once
the small base set has actually been run?

**2. Motivation**

This project optimizes a Monte Carlo simulation pipeline for parallax
detectability, not a fitter for real survey data. In that setting,
truth and generated-data properties are not "leakage" in the
production-relevant sense — the outcome we must never use as a
feature is the exhaustive/oracle result itself (§0.3, reaffirmed
below). If simulation-aware information can cheaply route an event to
an appropriate H0 strategy *before* spending any fits, that is a
genuinely different, and potentially much cheaper, kind of adaptive
policy than the fit-sequential ideas from Experiment A1/B1.

**3. Inputs**

- Local, not repository-tracked (same class of dependency as the
  Gate-1 raw fit directories already used by
  `aggregate_gate1_oracle_diagnostics.py`): per-event `Event_*.h5`
  (generated light curves, one group per band with `time, mag,
  err_mag, flux, err_flux, photometry_keep, ...`) and
  `true_rr_manual_*.parquet` (truth + simulation/survey metadata),
  under
  `~/Downloads/hidden_parallax/hidden_parallax_refit_test/artifacts/extreme100/<catalog_row>/...`
  (configurable via `--root` / `B1C_ARTIFACTS_ROOT`). Verified present
  for all 100 `extreme100` events (0 missing, 0 extra) before use.
- `validation/bounds_convergence/data/extreme100_refit_manifest.csv`
  (canonical truth `u0/tE/rho/piEN/piEE` — the same values the fitter
  itself uses as its "truth" anchor).
- `results/b1_predictors_per_event.csv`, `b1_fixed_budget_candidate_sets.csv`,
  `b1_rescue_matrix.csv` (Experiment B1, unmodified) — read only to
  build offline labels and, for Phase F scenario B/C, B1's own
  fixed-cost fit diagnostics.
- New script:
  `validation/bounds_convergence/analyze_b1c_simulation_difficulty.py`.

**4. Method**

*Leakage boundary, traced from code, not assumed.* Read
`validation/bounds_audit/run_bounds_audit_refit_core.py`'s
`load_case()` (loads the H5 via `load_h5_lightcurves()`, lines
1540-1606, and the truth parquet) and its `DATA-DRIVEN t0 BOUND` block
(lines ~2091-2168): `Tobs` is computed from `meta["curves"]`
*before* any H0 bounds/starts/optimizer setup, and `meta["curves"]` is
passed to the fit call (lines ~2246-2264) **unmodified** — no
additional filtering between load and fit. This is direct, in-repo
evidence that Tobs, Ndata, per-band counts, sampling and
photometric-precision quantities derived from the same arrays are
genuinely pre-H0, not merely plausible-sounding proxies.

*`Ndata_H0_input` — resolved, not assumed.* Defined as
`Σ_band len(meta["curves"][band])`, i.e. exactly
`load_h5_lightcurves`'s own filter
(`photometry_keep & isfinite(t,mag,err_mag) & err_mag>0`), replicated
read-only in the new script (not imported from core, to keep this
script fully decoupled from any production code path). Verified
against the truth parquet's `fit_n_points_total` for **all 100/100
events: exact match, 0 mismatches** (audit output, §7) — resolving the
earlier open question: `n_data_true` is a *different*, occasionally
diverging quantity from an earlier/external simulation stage (not
traceable further, since it originates outside this repository) and
is **not** used as a sampling feature anywhere in this experiment.

*Feature set — core vs. exploratory, deliberately small given N=100*
(see module docstring for exact derivations):
- **Core truth/geometry (5):** `log10_tE_true, log10_rho_true,
  abs_u0_true, abs_u0_over_rho (= abs_u0_true/rho_true, not the
  signed ratio), piE_true`.
- **Core sampling (9), computed from the H5, per-band first and then
  aggregated (never concatenating interleaved-band timestamps):**
  `Tobs_over_tE, Ndata_H0_input, n_filters_H0, frac_within_0p5_tE,
  frac_within_1_tE, frac_within_2_tE, pre_post_imbalance,
  median_cadence_across_bands` (median of each band's own median
  cadence), `max_gap_within_1tE_across_bands` (max of each band's own
  max gap inside `±tE_true`).
- **Core photometry (2):** `median_err_mag` (photometric precision,
  median-of-per-band-medians), `median_abs_flux_over_err_flux`
  (`|flux|/err_flux` per point, median-of-per-band-medians — an
  explicit **flux-precision proxy, not a microlensing-signal or
  parallax-detection S/N**, and not called "S/N" anywhere in outputs).
- **Core brightness/blending (4):** `brightest_source_mag,
  median_source_mag_visible, min_source_fraction_visible,
  median_source_fraction_visible`, all aggregated only over bands
  actually present in that event's H5 (not the parquet's static
  6-band columns, several of which are `NaN` per event).
- **Exploratory (10, kept in `b1c_event_features.csv` but not fed to
  Phase D/F models):** `lens_mass_msun, D_L, D_S, mu_rel, thetaE_mas,
  source_radius_rsun_catalog, l_deg, b_deg, piEN_true, piEE_true`.
- **Raw audit columns** (also in `b1c_event_features.csv`, to let the
  derivations above be independently reproduced/verified):
  `t0_true, tE_true, rho_true, u0_true, piEN_true, piEE_true,
  Tobs_global, Ndata_<band>` (7 bands), `fit_n_points_total_parquet`,
  `ndata_h0_input_matches_fit_n_points_total`, `ndata_mismatch_abs`.
- `delta_chi2_catalog` remains **excluded** (§7 of the prior planning
  turn): its provenance relative to this repo's own H0 input is
  unresolved (it may or may not require an auxiliary fit upstream),
  so it is not treated as a free pre-H0 quantity here.

*Labels (offline only, never features):* `unsafe_k{4,5,6}` and
`delta_vs_oracle_k{4,5,6}` read directly from
`b1_predictors_per_event.csv` (`candidate_rank==1`, i.e. each k's own
lexicographic-optimal candidate from Experiment B1 — reused, not
recomputed). `min_candidate_budget_reaching_tol` = smallest
k∈{2,3,4,5,6} whose **own independently-optimized** candidate already
has `delta<=0.1` for that event; **the k=2..6 candidates are not
nested**, so this characterizes B1's specific candidate policies, not
an intrinsic "minimum starts truly required" property of the event.
Rescue-family labels (`rescued_by_{physical,log_te,log_rho,log_te_rho}`,
`n_rescuing_strategies_<mode>`, `n_rescuing_modes`) are multilabel, built
from `b1_rescue_matrix.csv` at `k=4, candidate_rank=1`, grouping
`remaining_strategy` by its `mode` — no forced single "preferred
rescue" class.

*Analysis order, exactly as specified:* Phase A (build + hard-audit
the event table) → Phase B (univariate association of every feature
vs. `unsafe_k{4,5,6}`: AUC, medians/quartiles, `n_safe`/`n_unsafe`
always shown; Spearman rho of the feature vs. continuous
`delta_vs_oracle_k`) → Phase C (the same table's rows for the 6
declared scientific-risk variables — `tE_true, piE_true, rho_true,
abs_u0_true`, brightness, blending — examined explicitly, regardless
of whether they turn out predictive) → Phase D (logistic regression +
shallow decision tree only, repeated stratified CV, ROC AUC / PR AUC,
class sizes and fold-to-fold variability reported; no random
forest/boosting used, since nothing in the descriptive phases
justified adding non-interpretable models) → Phase E (rescue-family
associations, `INCONCLUSIVE` when a mode's rescued/not-rescued class
has `<3` residual events) → Phase F (three feature scenarios — A:
core pre-H0 only, B: B1's fixed-cost fit diagnostics only, C:
combined — compared via the same CV machinery as Phase D; these rows
live inside `b1c_difficulty_models_cv.csv`, tagged by
`feature_scenario`, rather than a separate combined-models file, since
a separate file would duplicate the same table with no new
information).

**5. Information allowed in "simulation-aware" policy design**

Every core/exploratory feature is a function only of the truth
parquet and the H5 light curve for that event — available before any
H0 fit, per the traced leakage boundary (§4). `catalog_row` is used
only to join tables, never as a feature or as a special-case
condition. `chi2_oracle_52` and everything derived from it
(`unsafe_k*, delta_vs_oracle_k*, min_candidate_budget_reaching_tol`,
rescue-family labels) are read **only** to build offline evaluation
labels in Phase A's label-loading step — Phase A's feature-building
step (`build_event_features`) never touches them. B1's fixed-cost fit
diagnostics (Phase F scenario B/C) are explicitly **not** pre-H0 —
they require the small base set to have already run — and are kept in
a separate, clearly-labeled block, never merged into the "core"
feature set silently.

**6. Metrics**

Phase A: row/uniqueness counts, `Ndata_H0_input` vs.
`fit_n_points_total` match rate, per-feature missingness. Phase
B/C: `n_safe`, `n_unsafe`, medians, IQR, `auc_unsafe_gt_safe`
(Mann-Whitney), Spearman rho/p vs. continuous delta. Phase D/F: ROC
AUC, PR AUC (mean ± std over repeated-CV folds), class sizes, number
of fold evaluations. Phase E: `n_rescued`, `n_not_rescued`, medians,
AUC, explicit `INCONCLUSIVE` marking.

**7. Results**

*Phase A audit* (`analyze_b1c_simulation_difficulty.py` run in this
session): 100/100 rows, 100/100 distinct `catalog_row`, no
duplicates. **`Ndata_H0_input` matched `fit_n_points_total` for
100/100 events, 0 mismatches** — the H5-derived definition is not
just consistent with the parquet's own recorded fit-input count in
the 4 events spot-checked during planning, it holds for the entire
sample. No missing/non-finite values in any core or exploratory
feature for any event.

*Label cross-check:* `unsafe_k4/k5/k6` counts from this table (21,
17, 14) reproduce Experiment B1's own `N(delta>0.1)` for the k=4/5/6
lexicographic-optimal candidates exactly — confirms the label-loading
path is wired correctly. `min_candidate_budget_reaching_tol`: 63
events already safe at k=2, 13 more first safe at k=3, 5 at k=4, 4 at
k=5, 4 at k=6, and **11/100 events are not reached by any of the five
independently-optimized k=2..6 candidates** (consistent with
non-nestedness — a larger k's own candidate is not guaranteed to be a
superset of a smaller k's).

*Phase B/C — univariate, k=4* (full table, all k and all features:
`results/b1c_feature_vs_difficulty.csv`): every single feature's
`auc_unsafe_gt_safe` falls in **0.42-0.63** — i.e. no feature shows
anything beyond weak separation on its own. The largest: `piEN_true`
(exploratory) 0.629, `abs_u0_true` (core, scientific-risk) 0.596,
`max_gap_within_1tE_across_bands` 0.574, `frac_within_1_tE` 0.565.
`median_source_fraction_visible` is **constant at 1.0 for all 100
events** (Spearman/AUC undefined, `NaN`) — this simulated catalog has
essentially no blending at the median-band level; `min_source_fraction_visible`
is not constant but is saturated at 1.0 for 75% of events (weak signal,
AUC 0.458, i.e. not even in the expected direction).

*Phase C — scientific-risk stratification* (mandatory regardless of
predictive value): of the 6 declared risk variables, `piE_true`,
`log10_tE_true`, `log10_rho_true`, and `median_source_mag_visible`
(brightness) are all within AUC≈0.48-0.58 at every k — **no strong
concentration of `unsafe` events detected in high-piE, long-tE,
small-rho, or bright-source regions within `extreme100`.**
`abs_u0_true` shows a consistent, mild/moderate signal across all
three k (AUC 0.596-0.619; unsafe events have larger median
`|u0_true|` — e.g. k=4: 0.744 unsafe vs. 0.460 safe) — flagged to
monitor, not alarming at this AUC. **Blending
(`median_source_fraction_visible`) cannot be assessed at all: it is
constant at 1.0 for all 100 events, and `min_source_fraction_visible`
is saturated at 1.0 for 75/100 — `extreme100` simply does not contain
the strongly-blended regime this risk check would need to be
informative.** This is a statement about the development sample's
coverage, not a (positive or negative) finding about blending itself.

*CV pipeline audit (before trusting Phase D/F's ROC AUC<0.5 cells).*
Verified explicitly in this session, on the actual fitted objects
(not by inspection alone):
- **positive class is `unsafe==True`**: `y = unsafe_k{k}.astype(int)`,
  confirmed `sum(y)` equals the known `unsafe_k{k}` count (e.g. 21 for
  k=4) for every k;
- **the score used is `P(class==1)`**, read via
  `predict_proba(...)[:, positive_col]` with `positive_col` located
  from `model.classes_ == 1` rather than assumed to be column 1 —
  confirmed `classes_ == [0, 1]` for every fitted model, so column 1
  was already correct, but the code no longer assumes it;
- **logistic regression is scaled** (`StandardScaler`), fit on the
  training fold and applied to the test fold, inside the CV loop --
  confirmed in code;
- **imputation was fit on the full dataset before the CV split** in
  the first version of this script -- a genuine leakage bug in
  general, but empirically inert here: `n_nan_total == 0` for every
  one of the 9 (k, scenario) combinations (core, B1-diagnostics, and
  combined feature matrices all have zero missing values for these
  100 events), confirmed by direct inspection and now also printed by
  the script itself. The code was corrected to fit imputation inside
  the training fold regardless, so it cannot become a live leakage
  path if a future re-run (e.g. on a Gate-3 sample) does have missing
  values. Re-running after the fix reproduced **byte-identical**
  `roc_auc_mean`/`pr_auc_mean` values to the pre-fix run for every
  row, confirming the fix changed nothing here;
- no label/score inversion found anywhere in the pipeline.

Per-cell audit printout (prevalence = fraction unsafe = the PR-AUC
random-classifier baseline):

| k | model | scenario | prevalence | roc_auc | pr_auc | pr_auc baseline |
|---|---|---|---:|---:|---:|---:|
| 4 | logistic | A | 0.210 | 0.417 | 0.295 | 0.210 |
| 4 | tree | A | 0.210 | 0.531 | 0.246 | 0.210 |
| 6 | logistic | A | 0.140 | 0.321 | 0.146 | 0.140 |
| 6 | tree | A | 0.140 | 0.441 | 0.172 | 0.140 |

(full 18-row table with every k/model/scenario:
`results/b1c_difficulty_models_cv.csv`, now including explicit
`prevalence_unsafe` and `pr_auc_baseline_prevalence` columns). **The
reported B1c CV metrics were unaffected by the latent
imputation-leakage bug because all evaluated feature matrices
contained zero missing values. After moving imputation inside each
training fold, all ROC-AUC and PR-AUC results were reproduced
exactly. The remaining ROC-AUC < 0.5 cells (worst: 0.321 at k=6,
scenario A, logistic regression) therefore reflect poor out-of-sample
generalization of the tested model/feature set, not score inversion
or CV leakage.**

*Phase D/F — exploratory CV models* (`results/b1c_difficulty_models_cv.csv`,
repeated stratified CV, up to 5-fold x 10 repeats): **scenario A (core pre-H0 features only)
gives ROC AUC 0.32-0.53 across k=4/5/6 and both models — at or below
chance in most cells, including 0.32 (worse than random) for logistic
regression at k=6.** Scenario B (B1's 5 fixed-cost fit diagnostics
only) gives ROC AUC 0.61-0.72, consistent with Experiment B1's own
marginal AUCs. **Scenario C (combined) gives ROC AUC 0.49-0.56 —
worse than scenario B alone at every k**, consistent with adding 14
weak/noisy core features diluting or overfitting relative to B1's
already-moderate 5-feature signal at N=100.

*Phase E — rescue-family associations* (`results/b1c_rescue_family_associations.csv`,
21 residual k=4 events): 60/80 (feature, mode) combinations were
`DIAGNOSTIC`, 20/80 `INCONCLUSIVE` (all involving `log_te_rho`, whose
residual-event count in either class is `<3` for every feature — too
few to say anything). Among `DIAGNOSTIC` rows, the strongest are for
`mode=physical` (only 4/21 residual events rescued by physical,
**very small sample**): `Tobs_over_tE` AUC 0.838 (rescued events have
much larger `Tobs_over_tE`, median 60 vs. 30) and `log10_rho_true`
AUC 0.750. `mode=log_te` (12 rescued / 9 not, better balanced) shows
moderate associations with `median_err_mag` (0.685) and
`abs_u0_over_rho` (0.667). `mode=log_rho` (7/14) shows weaker,
scattered associations (max 0.663).

**8. Interpretation**

- The Ndata/leakage-boundary work (§4, §7) is a clean, fully verified
  methodological result independent of everything else: pre-H0
  sampling features are exactly reconstructible from local artifacts,
  and `Ndata_H0_input` is now unambiguous and repo-code-verified.
- Pre-H0 simulation/truth features, **alone**, do not show useful
  out-of-sample predictive power for `unsafe_k` at N=100 (Phase D
  scenario A). This is a materially different, and more informative,
  result than "no strong univariate signal" (Phase B) — it shows the
  weak marginal associations do not combine into anything usable
  multivariately at this sample size, most likely because 14
  core features chasing 14-21 positive cases overfits badly under CV.
- Combining pre-H0 features with B1's fixed-cost fit diagnostics
  **hurts** performance relative to fit diagnostics alone at this N
  (scenario C < scenario B). This is a genuine, not-hypothesized,
  negative finding: at N=100, a simulation-aware layer does not
  currently earn its complexity on top of what Experiment B1 already
  extracts from a small executed base set.
- The rescue-family results are the most interesting exploratory
  finding (§7), but rest on very small samples (as few as 4 rescued
  events for `physical`) — real signal is plausible (e.g. `physical`-mode
  rescues concentrating in large-`Tobs_over_tE`, large-`rho_true`
  events is physically not unreasonable) but cannot be distinguished
  from noise at this N. `log_te_rho` cannot be assessed at all here.
- The scientific-risk stratification (§7, Phase C) is reassuring: no
  strong evidence that this line of computational optimization would
  concentrate errors in the high-piE / long-tE / small-rho regime that
  matters most for the detectability science question. The one caveat
  (`abs_u0_true`, moderate) should be re-examined once/if a
  simulation-aware policy is ever designed, not ignored.

**9. Decision**

**DIAGNOSTIC / CANDIDATE GENERATION** overall. Specifically:
- Phase A event table: **ACCEPTED** as a reproducible, audited
  artifact (100/100 rows, 0 duplicates, 0 Ndata mismatches, full
  feature coverage).
- Pre-H0 features as a **standalone** difficulty classifier at N=100:
  **REJECTED** (Phase D scenario A, ROC AUC at/below chance across
  k=4/5/6/both models tested) — this is a claim about *this*
  exploratory attempt at *this* sample size, not a claim that no
  pre-H0 signal can ever exist.
- Pre-H0 features **combined** with B1's fit diagnostics at N=100:
  **REJECTED** (scenario C consistently worse than scenario B alone).
- B1's fixed-cost fit diagnostics, reproduced here as scenario B:
  status unchanged from Experiment B1 (DIAGNOSTIC / CANDIDATE
  GENERATION, `abs_log_ratio_tE`/`delta_spread` the leading
  candidates).
- Rescue-family associations (Phase E): **DIAGNOSTIC / CANDIDATE
  GENERATION**. The single strongest cell (`physical` mode x
  `Tobs_over_tE`, AUC≈0.838) is specifically tagged **CANDIDATE
  HYPOTHESIS — NOT VALIDATED**: it comes from only 4 rescued events
  and is the best of 60 inspected (feature x mode) `DIAGNOSTIC`
  cells, so it is exactly the kind of result likely to look stronger
  than it is. `log_te_rho` is **INCONCLUSIVE** throughout (too few
  residual events rescued by it in either class, for every feature).
- Scientific-risk stratification (Phase C): **ACCEPTED as a risk
  read** for `extreme100` on `piE_true, tE_true, rho_true,` and
  brightness — no strong concentration detected. `abs_u0_true`:
  **mild/moderate concentration, monitor.** Blending: **INCONCLUSIVE
  / NOT ASSESSABLE** — `extreme100` has essentially no variation in
  `source_fraction` (constant 1.0 at the median, saturated at 1.0 for
  75/100 at the minimum), so this check has no power to detect a
  blending-driven risk even if one existed; it is not evidence that
  none exists.

**10. Consequence**

- **B2 should use B1's fixed-cost fit diagnostics
  (`abs_log_ratio_tE`, `delta_spread`) as its primary evidence for
  adaptive routing.** B1c does not justify adding a standalone
  pre-H0 simulation-aware classifier at N=100: the tested core
  feature block, evaluated on its own (scenario A) or added en bloc
  to B1's diagnostics (scenario C), did not improve on B1's own
  diagnostic model and degraded CV performance at N=100 (scenario C <
  scenario B at every k tested). This is a statement about the
  specific feature block and models tested here, at this sample size
  — it does not claim that no individual pre-H0 simulation feature
  could add value under a different design, a different feature
  selection, or a larger sample; that has not been tested, and this
  experiment deliberately does not re-search `extreme100` for a
  better-performing subset (that would be retuning on the development
  set, not evidence).
- The `physical`-mode rescue association (`Tobs_over_tE` AUC≈0.838,
  `log10_rho_true` AUC≈0.750) is a **CANDIDATE HYPOTHESIS — NOT
  VALIDATED**, not a rule: it rests on only 4 rescued events out of 21
  residual events, and it is the strongest result among many
  (feature x mode) combinations inspected (60 `DIAGNOSTIC` cells in
  `b1c_rescue_family_associations.csv`), which is exactly the setting
  where the single best-looking cell is expected to look better than
  it is. **B2 must not adopt it automatically as a tie-breaker or
  routing rule from this result alone** — it may be worth testing
  explicitly in B2 as one candidate among others, evaluated against
  the oracle like any other trigger. `log_te_rho` remains
  **INCONCLUSIVE** (too few residual events rescued by it to assess
  at all).
- No production, core-fitter, or SLURM changes result from this
  experiment. It remains fully offline, `extreme100` development-set
  evidence.
- This negative result should not be silently dropped if B2 or a
  later Gate 3 sample is examined: a genuinely larger sample (Gate 3's
  200-500 events) may reveal pre-H0 signal that N=100 cannot resolve,
  particularly for the rescue-family question, whose classes are
  currently too small to test at all for `log_te_rho` and barely
  testable for `physical`.

**11. Next step**

Proceed to Phase B2 design using B1's fixed-cost diagnostics as the
primary basis (per Experiment B1's own §11), not a simulation-aware
front-end. If Gate 3's independent sample is later assembled, re-run
this same B1c pipeline on it (same script, `--root` pointed at the new
sample's local artifacts) before deciding whether simulation-aware
routing deserves a second look — do not re-tune B1c's features on
`extreme100` in the meantime.

---

### Experiment B2 — Minimal conservative H0 adaptive-policy closure (CANDIDATE — FROZEN FOR INDEPENDENT VALIDATION)

**Status: CANDIDATE — FROZEN FOR INDEPENDENT VALIDATION.** A single,
minimal policy is identified that achieves zero false-safe on
`extreme100` with a modest (not large) reduction in mean H0 cost. The
exact policy (§4, §9) is now frozen -- Gate 3 evaluates it as-is, and
does not retune it (see §10 for exactly what "frozen" requires if
Gate 3 finds a failure). The threshold is fit exactly to the hardest
development-set case (§7-§8) and could plausibly fail on new data;
that is precisely what Gate 3 must test.

**Exact frozen policy (full precision; every other occurrence of this
number in this document is rounded for readability):**

```
k = 6
base set = results/b1_fixed_budget_candidate_sets.csv, k=6, candidate_rank=1
predictor = abs_log_ratio_tE  (computed from the k=6 base set's own fits)
rule: if abs_log_ratio_tE >= 0.0004123330728713: complete gate1_final18
                                                  (+ validated t0 rescue
                                                   when triggered)
      else: STOP, report the k=6 base set's own winner
```

This exact triple `(k=6, abs_log_ratio_tE, 0.0004123330728713)` is
pinned in code as `FROZEN_B2_POLICY_*` in
`analyze_b2_h0_policy.py`, with a consistency check
(`assert_frozen_policy_matches_computed_threshold`) that fails loudly
if it ever stops matching a fresh computation from the same inputs,
rather than silently drifting.

**1. Question**

Does a small fixed H0 base set (k=4, 5, or 6, from Experiment B1's
own `candidate_rank==1` sets, not re-optimized) plus a simple
threshold rule on Experiment B1's fixed-cost fit diagnostics
reproduce the validated H0 reference (`gate1_final18` + validated t0
rescue) within `delta_opt<=0.1` for every `extreme100` event, at a
lower mean H0-fit cost than always running the full 18-strategy plan?

**2. Motivation**

Direct continuation of Phase B (Experiment A1 §9, Experiment B1 §11):
this is the first concrete attempt at an actual policy, deliberately
scoped as a single, minimal, conservative iteration so that H0 can be
frozen and the project can move to H1/Gate 3/production without
another open-ended exploratory phase.

**3. Inputs**

- `results/b1_predictors_per_event.csv` (Experiment B1;
  `candidate_rank==1`, k∈{4,5,6} only: `chi2_best`,
  `abs_log_ratio_tE`, `delta_spread`, plus `chi2_std_among_set`,
  `chi2_range_among_set`, `n_modes_agreeing` inspected only as a
  clear-improvement check, not a new search).
- `results/b1_fixed_budget_candidate_sets.csv` (Experiment B1; to
  confirm, not re-derive, that each k's candidate is a subset of
  `gate1_final18`).
- `results/a1_sequential_h0_full18_policy.csv` (Experiment A1;
  `chi2_policy_final` — the validated reference this policy tries to
  reproduce cheaply — and `rescue_applied`, for N_fits accounting).
- New script:
  `validation/bounds_convergence/analyze_b2_h0_policy.py`.

**4. Method**

*Reference, precisely.* `chi2_policy_final` from Experiment A1 — the
`gate1_final18` full-18 winner, with the already-validated t0 rescue
applied when (and only when) that winner's own `active_mask` shows
`t0` active. This is **not** `chi2_oracle_52`; per Experiment A1 §5,
the two references must not be conflated.

*N_fits accounting, verified not assumed.* Confirmed in this session,
programmatically, that every one of the k=4/5/6 `candidate_rank==1`
strategy sets is a **subset** of `gate1_final18`'s own 18 strategies
(`analyze_b2_h0_policy.py:verify_subset_of_final18`, run against the
actual candidate lists — 4/4, 5/5, 6/6 members found in
`gate1_final18`). Therefore a SAFE decision costs `k` fits; an
ESCALATE decision costs `18` fits total (the `k` already spent are
reused, not duplicated), plus `18` more for the one `extreme100`
event whose full-18 winner triggers the t0 rescue (`catalog_row
36103`, per Experiment A1). Because escalation always completes the
exact validated reference procedure, an escalated event's resulting
`delta_opt` is `0` by construction — `N(delta_opt>0.1)` after a
policy is applied equals exactly that policy's `N_false_safe`, with
no additional failure mode from escalation itself.

*Threshold selection (design-time use of the reference, never a
runtime feature).* For each predictor P found positively associated
with risk (`delta_spread`, `abs_log_ratio_tE` — the two mandated
predictors — plus `chi2_std_among_set`, `chi2_range_among_set`,
`n_modes_agreeing`, checked only for a clear improvement), the
minimal zero-false-safe threshold is `t* = min(P over
delta_vs_reference>0.1 events)`, rule "escalate if `P >= t*`". This
is the cheapest single-threshold rule that catches every
development-set risky event; also tested, the simple OR of the two
primary predictors' individual thresholds. No joint/grid optimization
beyond this was performed (per instruction, to keep the rule
auditable and avoid a new open search).

**5. Information allowed in the policy**

Every rule's "escalate" condition is a function only of the k base
fits already run (`abs_log_ratio_tE`, `delta_spread`, etc., all
already validated in Experiment B1 as computable from a fixed
candidate set's own fits). `chi2_policy_final` / `chi2_oracle_52` /
`catalog_row` are used only to (a) select the threshold at design
time and (b) evaluate the resulting policy offline — never inside a
rule's escalate condition.

**6. Metrics**

Per policy: `frac_stopped_at_base`, `frac_escalated`, `n_false_safe`,
`N(delta_opt>0.1/1/10)`, `max_delta_opt`, `mean/p50/p90/max N_fits`,
exact rule text.

**7. Results**

*Verification:* k=4 (4 strategies), k=5 (5), k=6 (6) candidates are
each fully contained in `gate1_final18`'s 18 — confirmed
programmatically, not assumed.

*Threshold search* (`results/b2_policy_threshold_search.csv`): for
**every** predictor tested, at **every** k, the minimal zero-false-safe
threshold is forced down to a near-numerical-floor value by a single
recurring event, **`catalog_row=35927`**: at k=4/5/6 its
`delta_vs_reference` is a constant `0.343` (its winning candidate
strategy does not change as k grows from 4 to 6), while its
`delta_spread` (`0.000676`) and `abs_log_ratio_tE` (`0.000412`) are
10-100x smaller than the next-smallest risky event's — i.e. multiple
strategies in its own candidate set converge to the *same*,
slightly-wrong local minimum, so the fit "looks" fully converged and
confident despite being `0.34` above the reference. Because of this
single case, achieving zero false-safe forces large escalation
fractions for every tested predictor (**thresholds below are rounded
for display**; full precision is in
`results/b2_policy_threshold_search.csv` and, for the frozen k=6
policy specifically, in the exact box above):

| k | predictor | min-risky threshold | n_escalate/100 |
|---|---|---:|---:|
| 4 | `abs_log_ratio_tE` | 0.000412 | 81 |
| 4 | `delta_spread` | 0.000676 | 87 |
| 4 | `chi2_std_among_set` | 0.021 | 99 |
| 4 | `chi2_range_among_set` | 0.055 | 99 |
| 4 | `n_modes_agreeing` | 1 | 100 |
| 6 | `abs_log_ratio_tE` | 0.000412 | 75 |
| 6 | `delta_spread` | 0.000676 | 82 |

`n_modes_agreeing`, `chi2_std_among_set`, `chi2_range_among_set` are
all **worse** than the two mandated predictors (97-100% escalation) —
no clear improvement found among the additionally-inspected B1
diagnostics.

*Policy comparison* (`results/b2_policy_comparison.csv`, full 19-row
table):

| policy | k | rule | frac stopped | mean N_fits | p50/p90/max N_fits | N(delta_opt>0.1) | max delta_opt |
|---|---|---|---:|---:|---|---:|---:|
| fixed baseline | — | always run gate1_final18 (+t0 rescue) | 0.00 | **18.18** | 18/18/36 | 0 | 0 |
| best single, k=4 | 4 | `abs_log_ratio_tE >= 0.000412` | 0.19 | 15.52 | 18/18/36 | 0 | 0.030 |
| best single, k=5 | 5 | `abs_log_ratio_tE >= 0.000412` | 0.20 | 15.58 | 18/18/36 | 0 | 0.030 |
| **best single, k=6** | **6** | **`abs_log_ratio_tE >= 0.000412`** | **0.25** | **15.18** | 18/18/36 | **0** | **0.021** |
| OR rule, k=6 | 6 | `abs_log_ratio_tE>=0.000412 OR delta_spread>=0.000676` | 0.12 | 16.74 | 18/18/36 | 0 | 0.00003 |

The OR rule is **strictly worse** (higher mean N_fits) than
`abs_log_ratio_tE` alone at every k tested — `delta_spread`'s
threshold adds escalations without catching any risky event
`abs_log_ratio_tE` did not already catch. `N(delta_opt>1)` and
`N(delta_opt>10)` are `0` for every policy in the table (full table:
`results/b2_policy_comparison.csv`). Every policy's `p50`/`p90`/`max
N_fits` are unchanged from the fixed baseline (`18`/`18`/`36`) —
escalation dominates the majority of events for every rule tested;
only the mean shifts, and only modestly.

**8. Interpretation**

- The best closeable policy found is **k=6 base set, escalate if
  `abs_log_ratio_tE >= 0.000412`**: mean N_fits `15.18` vs. `18.18`
  fixed — a **16.5% reduction**, not the larger (5-8 mean fits)
  reduction hypothesized at the start of Phase B. This is a real, but
  modest, saving.
- The saving is concentrated in the minority (25%) of events that
  stop at the 6-fit base set; the majority still escalate to the full
  18(+18) procedure. Median cost is unchanged.
- The threshold is set **exactly** at `catalog_row=35927`'s observed
  diagnostic value — the smallest margin possible while still
  catching it. This is expected behavior for a min-over-risky
  threshold rule, but it means the policy is tuned to the single
  hardest case *in this specific 100-event sample*; a new sample
  (Gate 3) could contain a different event whose diagnostics are even
  smaller while still being risky, which would silently reintroduce a
  false-safe failure. This is a real risk to test, not yet resolved.
- Combining predictors (OR) does not help here: both mandated
  predictors are forced to their respective floors by the *same*
  event, so OR-ing them only adds each rule's independent false
  alarms without any additional true-risk coverage.

**9. Decision**

**CANDIDATE — FROZEN FOR INDEPENDENT VALIDATION**: *k=6 base set +
escalate if `abs_log_ratio_tE >= 0.0004123330728713` (else stop)*
(exact value; rounded to `0.000412` elsewhere in prose), with
escalation always completing `gate1_final18` and its validated t0
rescue exactly as today. Zero false-safe and zero
`N(delta_opt>0.1/1/10)` on `extreme100`; mean N_fits `15.18` vs.
`18.18` fixed (16.5% reduction, i.e. roughly 3 fewer H0 optimizations
per event — at production scale (e.g. ~300k events) this is on the
order of ~10^6 optimizations avoided, which is why this modest
percentage is still worth an independent-sample test). This satisfies
the stated lexicographic priority (zero false-safe → simplicity →
lower `E[N_fits]`) better than any other rule tested, including the
OR combination. **This is not yet a validated production policy** —
it is frozen exactly as stated, for Gate 3 to evaluate, not to retune.

**10. Consequence**

- This policy is **frozen, not adopted**: it is a candidate for Gate
  3 to test as-is. Nothing about it (the base set, the predictor, or
  the threshold `0.0004123330728713`) may be changed before or during
  Gate 3 without that action itself invalidating Gate 3 as an
  independent test (see below).
- **If Gate 3 finds a false-safe case for this policy, while the
  `gate1_final18 + validated t0 rescue` reference itself remains
  correct for that event:** the conservative default is to **abandon
  this adaptive policy** for production and produce with the fixed
  `gate1_final18` (+ t0 rescue) reference, which does not depend on
  this threshold at all.
- **If instead we choose to modify the threshold (or any other part
  of this policy) using what Gate 3 revealed:** Gate 3 has, by that
  choice, been converted into development data — it no longer counts
  as the independent validation this policy needs. The modified
  policy must then be evaluated on a **new, independent sample** before
  any production use. Retuning silently on Gate 3 and treating that
  same run as having validated the retuned policy is not acceptable
  and must not be done.
- The modest size of the saving (16.5%, concentrated in 25% of
  events) should be weighed against Gate 3B's not-yet-measured
  per-fit wall time (§5 of this document): if per-fit cost is small
  relative to fixed overhead, this saving may not be worth the added
  policy complexity in production; that trade-off cannot be resolved
  until Gate 3B actually measures wall time.
- No production, core-fitter, or SLURM change is made by this
  experiment. H1 is untouched (`controlled5` unchanged).

**11. Next step**

Per the project's stated path: H0 candidate policy proposed above →
final H1 audit (winner bounds, t0/piE, nestedness bookkeeping, per
checkpoint §Gate-2-pendiente) → Gate 3 independent validation
(200-500 events, testing this exact frozen policy without retuning)
→ Gate 3B cluster profiling → empirical H0/null LRT calibration →
staged production.

---

## 2. Reference: `gate1_oracle_diagnostics.csv` schema

Regenerated by `aggregate_gate1_oracle_diagnostics.py` (re-run,
verified in Experiment 0; not modified this session). Not committed
until reviewed together.

**Row meaning:** one row = one H0 optimization (one TRF start) for one
`(source, mode, catalog_row)` call.

**Keys:**
- **Canonical semantic key: `(mode, strategy_id)`.** Proven stable and
  fully event-covering for all three sources (52/18/18 distinct pairs;
  see Experiment 0, §7). Use this, not raw `label`, for any grouping
  or matching across events.
- Provenance fields, not the preferred grouping key:
  - `label`: raw fitter output; event-dependent (NOT usable as a
    cross-event key) for exactly 3/13 slots per mode in
    `gate1_oracle_52` (see Experiment 0), fully stable otherwise.
  - `start_slot`: on-disk execution order. Unlike `label`, it IS a
    stable positional key for all 13/13 slots of `gate1_oracle_52`
    (and for both `gate1_final18` sources), given the verified
    invariant that every `gate1_oracle_52` call has exactly 13 rows
    in the fixed construction order (see Experiment 0, §7-§8). It is
    still not the preferred key: `strategy_id` is semantically
    explicit and does not rely on that ordering invariant continuing
    to hold in future runs.
- Row-uniqueness key: `(source, mode, catalog_row, start_slot)`,
  equivalently `(source, mode, catalog_row, label)` or
  `(source, mode, catalog_row, strategy_id)` (all three verified
  duplicate-free in Experiment 0).

**Runs combined (`source` values), from `ROOTS` in
`aggregate_gate1_oracle_diagnostics.py`:**

| `source` | plan | `t0_margin_factor` | starts/event | expected rows (100 events) |
|---|---|---:|---:|---:|
| `gate1_oracle_52` | `HIDDEN_PARALLAX_H0_START_PLAN=all` (4 coords x 13 starts) | 0.0 | 52 | 5200 |
| `gate1_final18_base_t0margin0` | `gate1_final18` | 0.0 | 18 | 1800 |
| `gate1_final18_rescue_t0margin0.25` (**diagnostic run, all 100 events — see §0.4; NOT the validated per-event rescue policy**) | `gate1_final18` | 0.25 | 18 | 1800 |

**Columns directly from the fitter's per-fit record** (written by
`run_bounds_audit_refit_core.py`, unchanged by aggregation):
`hypothesis, label, status, chi2, t0, u0, tE, rho, optimizer_success,
optimizer_optimality, optimizer_active_mask, log`.

**Columns added by `aggregate_gate1_oracle_diagnostics.py`** (run
identity / derived, not part of the fitter's own output):
`source, root, h0_start_plan, t0_margin_factor, mode, catalog_row,
start_slot, strategy_id`. `strategy_id` is derived from `label`
directly for the `gate1_final18` sources, and from a fixed
`start_slot -> strategy_id` lookup table (transcribed from the
fitter's own start-construction code) for `gate1_oracle_52` — see
Experiment 0, §4, and the module docstring of
`aggregate_gate1_oracle_diagnostics.py` for the full derivation.

**Verified counts (this session, re-aggregated from local disk):**
8800 total rows = 5200 + 1800 + 1800; 100 events in every source;
0 missing files, 0 row-count mismatches; 52/18/18 distinct
`(mode, strategy_id)` pairs per source, each covering all 100 events.
Companion file `gate1_oracle_call_wall_times.csv`: 1200 rows = 100
events x (4 modes x 1 for `gate1_oracle_52` + ... ), one row per
`(source, mode, catalog_row)` call, `wall_seconds_call` plus a
**naive, call-level average** `wall_seconds_per_start_avg` — per the
cost-methodology note (§3 below), this average must not be read as a
true per-start runtime.

**Audit outputs (this session, `audit_gate1_oracle_diagnostics.py`):**
- `results/gate1_oracle_diagnostics_audit_by_source.csv` — one row per
  `source` with the counts in Experiment 0 §7: the hard integrity
  check's counters (hypothesis/status/finiteness/h0_start_plan/
  t0_margin_factor/catalog_row-set), strategy/label counts, and both
  `per_fit_*` and `winner_*` optimizer diagnostics.
- `results/gate1_oracle_diagnostics_audit_start_slot_map.csv` — one
  row per `(source, mode, start_slot)` with the distinct labels
  observed at that slot across all 100 events.
- `results/gate1_oracle_diagnostics_audit_strategy_id_coverage.csv` —
  one row per `(source, mode, strategy_id)` with the number of events
  that strategy_id is observed for in that source, and whether that
  equals the source's total event count.

---

## 3. Cost-accounting note

`wall_seconds_call / n_starts_in_call` in
`gate1_oracle_call_wall_times.csv` is a coarse, call-level estimator
(all starts in one mode's call share one measured wall time). It is
**not** a true per-start runtime and must only be reported as a
secondary/approximate cost metric, explicitly labelled as such. `N_fits`
(count of H0 optimizations run) is the primary cost metric for Phase
A/B until Gate 3B (§5 below) adds real per-fit timing (and, if
feasible, `nfev`/`njev`) to the fitter itself, on the machine/cluster
production will actually run on.

---

## 4. Validation-sample discipline

`extreme100` is the **development** sample for the adaptive H0 policy.
`N(delta_chi2>0.1)=0` on `extreme100` is a necessary condition to
freeze a candidate policy, not sufficient to call it validated for
production. Once a policy is selected on `extreme100`, it must be
frozen (no further threshold tuning) and re-evaluated on an
independent 200-500 event sample — this is Gate 3 and has not started.

---

## 5. Cluster-production reproducibility and profiling status

**This section is a status/limitation statement, not an experiment.**
No profiling is run here; it records what is and is not currently
known about production runtime, and defers the actual measurement to
an explicitly named future gate.

### 5.1 What the current wall-time numbers are, and are not

Every wall-time number referenced anywhere in this document or in
`gate1_oracle_call_wall_times.csv` (and the `wall_*` columns derived
from it, e.g. in `gate1B_final18_per_event.csv`) is a **historical,
approximate diagnostic**, not a characterization of the production
fitter:

- it was measured on whatever local machine happened to run each
  Gate-1 validation call (see `ROOTS` in
  `aggregate_gate1_oracle_diagnostics.py` — local `~/Downloads/...`
  directories), not on the cluster hardware production will actually
  run on;
- it is call-level, not per-fit/per-start (§3);
- part of the historical CHE profiling this project has referenced in
  the past corresponds to **earlier versions of the pipeline** (prior
  bounds, prior start plans, prior coordinate/x_scale handling) and
  must **not** be read as profiling of the H0/H1 fitter configuration
  this document is currently validating. Any number carried over from
  that earlier profiling is diagnostic-only context, never a
  production benchmark.

**There is currently no final characterization of the runtime of the
production fitter.** In particular, none of the following are known
with production-grade confidence yet: wall time per event at
production scale, cluster throughput, total wall-clock time for the
full simulated population, or how runtime scales with the number of
concurrent workers.

### 5.2 Why profiling must be repeated, and when

Cluster profiling is only meaningful once it measures the fitter
configuration that will actually ship. Every one of the following is
still an open decision in this document (§0.2, §18 of
`VALIDATION_STATUS_2026-09-14.md`) and each one can change the number
and cost of fits per event, so profiling done before all of them are
frozen would have to be redone anyway:

- the H0 adaptive policy (Phase A/B of this document — not yet
  designed);
- the H1 start strategy (currently 5 fixed starts, §0.1, but not
  frozen against further evidence);
- nested-H0-in-H1 handling / how the embedded-H0 candidate is stored
  and reported (§0.1, §15 of the checkpoint);
- the shared and H1-specific bounds (`production_candidate`, §0.1);
- the final production outputs/diagnostics the fitter writes per
  event (raw vs. nested H1 chi2, optimizer diagnostics kept, etc.).

**No final decision about production compute resources, wall-clock
budget, or total expected production duration may be made from the
historical profiling data currently in this repository.** Any such
statement made informally in discussion (e.g. rough past estimates
from CHE runs) is provisional context only, not a commitment.

### 5.3 GATE 3B — Production profiling (not started)

A dedicated profiling gate, run only after the fitter configuration
above is frozen (i.e. after Phase A/B conclude and are accepted, and
the resulting policy is written into the actual production code
path — not just validated offline). This is distinct from Gate 3
(§4), which re-validates scientific correctness (`delta_chi2`
coverage) on an independent 200-500 event sample; Gate 3B measures
**operational cost**, not scientific correctness, though it should
reuse the same frozen fitter and, where practical, the same
independent sample.

Gate 3B must measure, on the actual cluster/hardware production will
run on:

- wall time per event (end-to-end, including all H0 and H1 starts and
  any rescue);
- wall time per fit/start, individually, if the fitter can be
  instrumented to record it (current per-call-only timing, §3, is not
  sufficient for this);
- `N_fits` actually executed per event, separately for H0 and H1
  (including rescues);
- runtime distribution: p50 / p90 / p95 / p99, not just the mean;
- CPU usage per event/worker;
- peak memory usage;
- I/O (read of photometry/catalog inputs, write of per-event outputs);
- `nfev` / `njev` (function/Jacobian evaluation counts) from the
  optimizer, if obtainable, as a hardware-independent complement to
  wall time;
- how runtime varies with event type/difficulty (e.g. large tE, large
  rho, events that trigger the t0 rescue or any future H0 rescue);
- scaling of throughput with number of concurrent workers;
- resulting throughput in events/hour at the chosen worker count;
- an estimate of total wall-clock time for the full production
  population, derived from the above, not from historical numbers.

**Status: NOT STARTED.** This section documents Gate 3B as a required
future stage; it does not run any profiling itself. Do not treat the
existing `gate1_oracle_call_wall_times.csv` numbers, or any prior CHE
profiling of earlier pipeline versions, as a substitute for it.

---

## 6. Final H1 Audit — `controlled5` freeze decision (Gate 2 closure)

**Scope note:** this document's title and Experiment log (§1) are
H0-specific; this section is the one exception, added here (rather
than in a new file) because it is the direct next step after
Experiment B2 and closes the same overall Phase-B-to-Gate-3 arc. It
audits H1's already-run `controlled5` fits (checkpoint
`VALIDATION_STATUS_2026-09-14.md` §9-16); it proposes no new H1
search.

**Status: CANDIDATE — FROZEN FOR INDEPENDENT VALIDATION** for
`H1 = controlled5 + shared-domain t0 rescue when the H1 winner's own
t0 bound is active`. §6.1-6.9 record the initial audit and the
resulting `PENDING` state (one material t0 truncation found,
`catalog_row=565924`, §6.5-6.6); §6.10 records the shared-domain
rescue design, its Step 1-3 closure, and the final decision. §6.1-6.9
are kept exactly as originally written -- the `PENDING` conclusion
they reached was correct at the time and is not retroactively
edited; §6.10 supersedes only the open decision in §6.7, not the
facts in §6.5-6.6.

### 6.1 Question

Can `controlled5` (the 5-start H1 plan: `truth`,
`H0_NESTED_piE_0`, `truth_half_piE`, `truth_mirror_u0_piEN`,
`old_final_reseed`) be frozen as the H1 strategy for Gate 3, or does
a winner-bound / t0 / piE / nestedness audit reveal a structural
problem that must be fixed first?

### 6.2 Inputs

- `~/Downloads/hidden_parallax/production_validation/gate2_controlled4_extreme100/...all_refits.csv`
  (4 starts: `truth`, `H0_NESTED_piE_0`, `truth_half_piE`,
  `truth_mirror_u0_piEN`) and
  `gate2_oldfinal_extreme100/...all_refits.csv` (1 start:
  `old_final_reseed`) — local, not repository-tracked, same class of
  dependency as the Gate-1 raw fit directories.
- `~/Downloads/hidden_parallax/hidden_parallax_refit_test/artifacts/extreme100/<catalog_row>/models/*/*/Event_*.h5`
  — read-only, only for the pooled time arrays needed to reconstruct
  each event's data-driven t0 domain (same artifact tree Experiment
  B1c already used).
- `validation/bounds_convergence/data/gate2_h0_anchor_manifest_extreme100.csv`
  — the exact embedded-H0 reference per event (`chi2_h0`,
  `t0_margin_factor`, `rescue_triggered`). Verified in this session:
  for `catalog_row=36103`, `chi2_h0=178.6896431771567` matches
  Experiment A1's `chi2_policy_final` for that event exactly — i.e.
  this manifest already encodes the validated t0-rescue-adjusted H0
  reference, not a plain unrescued value.
- `results/b2_policy_comparison.csv` (Experiment B2; read only for
  the frozen H0 policy's `mean_n_fits`, for the cost section).
- New script: `validation/bounds_convergence/analyze_h1_final_audit.py`.
- **One authorized, minimal, targeted diagnostic run** (not a new
  start search): after confirming no existing local output covers
  `catalog_row=565924` at a widened t0 domain (searched
  `~/Downloads/hidden_parallax/production_validation/` and
  `hidden_parallax_refit_test/` broadly; only H0-only runs and an
  unrelated 3-event `controlled5_problem3` directory exist there, none
  matching), ran the existing, unmodified
  `validation/bounds_audit/run_bounds_audit_refit.py` /
  `run_bounds_audit_refit_core.py` for exactly this one event, H1
  only (`--fit-scope h1`, so 0 H0 fits executed), `controlled5` starts
  (`HIDDEN_PARALLAX_H1_START_PLAN=controlled5`), same
  `production_candidate` bounds, `HIDDEN_PARALLAX_T0_MARGIN_FACTOR=0.25`.
  The core script's own built-in consistency guard requires the H0
  anchor manifest's recorded `t0_margin_factor` to match the H1
  env var exactly (enforcing shared H0/H1 nuisance domain for a valid
  LRT); satisfied here with a scratch copy of
  `gate2_h0_anchor_manifest_extreme100.csv` with only that one row's
  `t0_margin_factor` set to `0.25` (not committed; the repo-tracked
  manifest is unchanged). Output written to a new, separate, local
  (not repository-tracked) directory,
  `gate2_t0margin025_565924_diag/`, isolated from all other runs.
  Exactly 5 H1 fits were executed, nothing else. The raw
  `all_refits.csv` from that one run is copied into
  `results/h1_t0margin025_565924_diagnostic.csv` (repository-tracked)
  so this result does not depend on the local scratch directory
  persisting. Exact reproduction command:
  ```
  HIDDEN_PARALLAX_H1_START_PLAN=controlled5 \
  HIDDEN_PARALLAX_H0_ANCHOR_MANIFEST=<scratch copy of gate2_h0_anchor_manifest_extreme100.csv with catalog_row=565924's t0_margin_factor set to 0.25> \
  HIDDEN_PARALLAX_T0_MARGIN_FACTOR=0.25 \
  python3 validation/bounds_audit/run_bounds_audit_refit.py \
    --catalog-row 565924 \
    --manifest validation/bounds_convergence/data/extreme100_refit_manifest.csv \
    --bounds-profile production_candidate \
    --fit-scope h1 \
    --work-root ~/Downloads/hidden_parallax/production_validation/gate2_t0margin025_565924_diag \
    --force
  ```

### 6.3 Method

Aggregated the two local H1 run roots into one canonical,
integrity-checked table (`gate2_controlled5_diagnostics.csv`, 100
events x 5 starts = 500 rows), hard-audited the same way as
Experiment 0 (row/event counts, exactly the 5 expected start labels
per event, `hypothesis=="H1"` throughout, `status=="success"`
throughout, all params finite, no duplicate `(catalog_row,label)`
rows — every check passed, 0 violations). Determined the
`controlled5` winner per event (`argmin chi2` over the 5 starts) —
this raw winner chi2 is `chi2_h1_raw_optimized`, never floored before
being reported.

*Winner-bound audit:* for each winner, read its 6-element
`optimizer_active_mask` (order `t0,u0,tE,rho,piEN,piEE`) and computed
distance-to-bound for the four fixed shared bounds
(`u0,tE,rho,piEN,piEE`) plus the event-specific, data-driven t0
domain (reconstructed from the H5, `t0_margin_factor` from the anchor
manifest — same construction as the core fitter's own `DATA-DRIVEN
t0 BOUND` block, read-only here). **`tE` and `rho` distances are
computed in log10 space**, not linear — their bounds span 6-7 orders
of magnitude, so a linear fraction-of-range is not physically
meaningful (an early linear-scale run of this audit spuriously
flagged 98/100 events as "near" the `tE` bound purely because the
range is huge in linear units, despite 0 of them showing an active
`tE` mask; switching to log-scale reproduced the mask-based result
exactly, 0/100, and was kept as the corrected method — this
correction is recorded here rather than silently applied).

*Nestedness audit:* compared each event's `chi2_h1_raw_optimized`
against that event's `chi2_h0` (the anchor manifest's exact embedded
value, itself already t0-rescue-adjusted), **before** any
`min(...)` floor, reporting the raw difference.

*Cost:* `controlled5` is fixed at 5 H1 fits/event; combined with
Experiment B2's frozen H0 policy mean.

### 6.4 Information allowed

This is a pure audit of already-run, already-committed fits; it reads
no new features and proposes no new H1 starts. `chi2_h0` is used only
as (a) the historical input to `H0_NESTED_piE_0` (already baked into
the fits being audited) and (b) the nestedness comparison target —
never as a criterion for selecting among the 5 `controlled5` starts.

### 6.5 Results

**Winner-bound audit** (`results/h1_final_audit_winner_bounds.csv`,
100 rows):

| param | n_active_mask | n_near_bound (log-scale for tE/rho, linear else, ≤1% of range) |
|---|---:|---:|
| t0 | 1/100 | 3/100 |
| u0 | 0/100 | 0/100 |
| tE | 0/100 | 0/100 |
| rho | 10/100 | 10/100 |
| piEN | 0/100 | 0/100 |
| piEE | 0/100 | 0/100 |

`u0`, `tE`, `piEN`, `piEE`: **no winner anywhere near any bound.**
Max `|piEN|`/`|piEE|` among winners: `10.74`/`7.86` (bound `±40`);
even pooling **all 500** individual H1 fits (every start, every
event, not just winners), the most extreme values found are
`|piEN|=20.81`, `|piEE|=21.20` — still barely half the `±40` bound.
**No evidence of piE bound truncation anywhere in the H1 fit
population, winners or not.**

`rho`: 10/100 winners sit at the **lower** floor (`rho≈1e-7`,
`active_mask` confirms exactly these 10, log-scale distance agrees);
**none** approach the upper bound (`rho=10`; max winner `rho=4.14`).
Full list (all 10, `results/h1_final_audit_winner_bounds.csv` joined
with `extreme100_refit_manifest.csv`'s `true_rho`):

| catalog_row | fitted rho | side | true_rho | winner start | chi2 |
|---:|---:|---|---:|---|---:|
| 64956 | 1.026e-7 | lower | 0.000264 | old_final_reseed | 132.15 |
| 79853 | 1.001e-7 | lower | 0.061391 | old_final_reseed | 111.28 |
| 85380 | 1.005e-7 | lower | 0.000053 | old_final_reseed | 91.99 |
| 86451 | 1.000e-7 | lower | 0.000059 | old_final_reseed | 301.23 |
| 87786 | 1.000e-7 | lower | 0.000211 | old_final_reseed | 266.74 |
| 557860 | 1.000e-7 | lower | 0.000008 | truth_mirror_u0_piEN | 288.75 |
| 562093 | 1.000e-7 | lower | 0.000166 | old_final_reseed | 232.52 |
| 565924 | 1.000e-7 | lower | 0.000101 | truth_mirror_u0_piEN | 111.21 |
| 572190 | 1.000e-7 | lower | 0.000524 | old_final_reseed | 520.43 |
| 578896 | 1.001e-7 | lower | 0.000078 | old_final_reseed | 521.31 |

**All 10/10 are lower-bound, 0/10 upper-bound — confirmed explicitly,
not assumed.** Every one of these events also has a tiny `true_rho`
(`8e-6` to `0.061`, all well below `1`) — i.e. the winner is
correctly recovering that the *true* finite-source signal is itself
close to the point-source limit for exactly these events; the data
cannot resolve `rho` below the floor, which is the expected numerical
signature of a real `rho→0` degeneracy, not a sign that a better,
larger-`rho` minimum exists beyond the bound. No rho ladder or new
fits are warranted by this. This is the well-known point-source
degeneracy (`rho→0` is consistent with "not distinguishable from a
point source" given the data, a genuine physical result at this
bound's floor, which is
itself a physically motivated limit, not an arbitrary cutoff) — not
evidence of upper-bound truncation.

`t0`: **1/100** winners exactly active (`catalog_row=565924`, at the
domain's upper edge, distance `4.7e-10` days — effectively exact).
Two more (`75965`, `80852`) are within 1% of the domain span without
being flagged active by the optimizer (`0.83%` and `0.45%` from the
nearest edge respectively) — close but not pinned.

**`catalog_row=565924`, targeted t0-widening diagnostic (5 H1 fits,
`t0_margin=0.25`, this event only — see §6.2/§6.3 for exactly how):**

| | base (`t0_margin=0`, current `controlled5`) | widened (`t0_margin=0.25`, diagnostic) |
|---|---|---|
| winner start | `truth_mirror_u0_piEN` | `truth_mirror_u0_piEN` |
| chi2 | `111.209716` | `109.846734` |
| t0 | at domain upper edge (active) | `2464680.07` (interior; new domain `[2460531.95, 2465709.36]`) |
| tE | `137.1` d | `861.1` d |
| u0 | `0.974` | `0.516` |
| rho | `1.0e-7` (floor) | `1.0e-7` (floor) |
| active_mask | `t0` active | `rho` only (lower) |

**`delta_chi2 = chi2_current - chi2_wider_t0 = 111.209716 - 109.846734
= 1.362982`** — **material**, not negligible (more than 10x the
`delta_opt<=0.1` development-set acceptance criterion Experiment
A1/B1/B2 used for the H0 policy; that criterion does not itself apply
to H1, but its scale is the only calibrated reference point available
for judging "material" here). The wider-domain winner is not a
small perturbation of the current one: `tE` moves from `137` to
`861` days, and `t0` becomes fully interior (no longer active) — this
is a genuinely different, better-fitting solution that the current
`t0_margin=0` domain cannot reach for this one event. Per instruction,
this result is reported here and NOT converted into a rescue policy;
see §6.7 for the decision this leaves open.

**Nestedness audit** (`results/h1_final_audit_nestedness.csv`, 100
rows): **`N(chi2_H1_raw > chi2_H0_exact) = 0/100`.** Every single
event's raw, unfloored `chi2_h1_raw_optimized` is already `<=
chi2_h0`. The closest raw nestedness margin is `-0.0163`: still on
the correct side of zero, with no raw nestedness violations. The
median margin is `-47.8`, and the minimum (most negative, i.e.
furthest from zero) is `-99625`. (This margin is a separate quantity
from the `delta_opt=0.1` tolerance used to validate the H0
optimizer elsewhere in this project; no comparison to that tolerance
is implied or needed for nestedness.)

**Cost:** `controlled5` = 5 H1 fits/event (fixed). Combined with
Experiment B2's frozen H0 policy (`mean_n_fits=15.18`): **expected
combined H0+H1 cost ≈ 20.18 fits/event** in development.

### 6.6 Interpretation

- No shared-bound truncation evidence anywhere: `u0`, `tE`, `piEN`,
  `piEE` winners are comfortably interior for all 100 events, and
  `piE` stays far from `±40` even across all 500 individual fits, not
  just winners.
- The 10/100 `rho` lower-bound hits are the expected point-source
  degeneracy, not a bounds problem — confirmed explicitly (§6.5): all
  10 are lower-bound, 0 upper-bound, and all 10 have a tiny
  `true_rho`, consistent with a genuine `rho→0` degeneracy rather than
  a truncated, larger-`rho` minimum. No event approaches the upper
  bound, so widening `rho`'s range would not be justified by this
  data, and narrowing the lower floor is a separate physical/numerical
  question outside this audit's scope. No rho ladder was run, per
  instruction, since this is exactly the case where one is not
  warranted.
- **The single `t0`-active event (`565924`) is NOT merely an isolated
  cosmetic boundary touch — it is a confirmed, material truncation.**
  The targeted diagnostic (§6.5) shows the current `controlled5`
  (`t0_margin=0`) result for this event is `1.36` chi2 worse than a
  reachable, materially different solution (`tE` `137d → 861d`) that
  only becomes visible once the t0 domain is widened. This is the H1
  analogue of H0's `36103` case, but unlike H0 (which has a validated,
  already-frozen t0-rescue mechanism, Experiment A1 §0.4), **H1
  currently has no equivalent rescue** — `controlled5` as specified
  does not know to widen t0 for this event. Per instruction, no rescue
  policy is proposed here; this is reported as an open finding.
- Nestedness holds with a comfortable margin for all 100 events — no
  numerical borderline case, let alone a real optimizer failure (this
  holds using the *current*, `t0_margin=0`, `chi2_h1_raw_optimized`
  for `565924`; the wider-domain diagnostic value, `109.85`, is even
  further from `chi2_h0=454.55`, so nestedness is not put at risk by
  the t0 finding either way). The raw-vs-floored bookkeeping principle
  (§0.1, §15 of the checkpoint) is upheld: this audit reports
  `chi2_h1_raw_optimized` directly, with no floor applied before
  checking it.
- `controlled5` continues to reproduce the best known H1 result on
  `extreme100` for 99/100 events (checkpoint §13-14; this audit did
  not re-run the ablation, only the winner/bound/nestedness properties
  of the already-established winners) — the t0 finding affects exactly
  one event out of 100 in this sample.

### 6.7 Decision

**PENDING — not yet frozen.** Three of the four checks (winner-bound
for `u0/tE/piEN/piEE`, piE, nestedness) found no structural problem.
The t0 check found one **material**, not negligible, truncation
(`catalog_row=565924`, `delta_chi2=1.36`, §6.5-6.6). Per the
conditions set for this audit, that rules out an unconditional
"no problem found, freeze as-is" close. This leaves an explicit choice
that this document does not make unilaterally:

- **(a) Freeze `controlled5` exactly as specified**, documenting the
  `565924` truncation as a known, quantified, single-event
  (1/100 in `extreme100`) gap, and let Gate 3 reveal whether it recurs
  at a similar rate in an independent sample before deciding whether
  H1 needs a rescue mechanism at all; or
- **(b) Design and validate an H1 t0-rescue mechanism** (structurally
  analogous to H0's already-frozen one, Experiment A1 §0.4) before
  freezing `H1`, so that Gate 3 evaluates a policy that already
  handles this case.

Both preserve everything else already audited clean in §6.5-6.6
unchanged; neither has been chosen here.

### 6.8 Consequence

- No `H1` freeze decision is recorded until §6.7 is resolved by
  explicit instruction.
- The 10/100 `rho`-floor population is confirmed benign (§6.5-6.6)
  and needs no further action regardless of how §6.7 is resolved.
- Combined expected per-event cost entering Gate 3 (unaffected by the
  §6.7 choice, since it only concerns one event's H1 result, not the
  fit count): `H0(B2, mean 15.18) + H1(controlled5, fixed 5) ≈ 20.18`
  H0/H1 fits per event — this is the number to carry into Gate 3B's
  eventual wall-time-based production budget estimate (§5), once real
  per-fit timing exists.
- No production, core-fitter, or SLURM change is made by this audit.
  The one diagnostic run (§6.2-6.3) used the existing, unmodified
  fitter for exactly 5 fits on one event and is not itself a change to
  any frozen policy.

### 6.9 Next step

Resolve §6.7 (a) vs (b) by explicit instruction. If (a): `H1 =
controlled5` is frozen as **CANDIDATE — FROZEN FOR INDEPENDENT
VALIDATION** (matching Experiment B2's status for H0) with the
`565924` finding documented as a known gap, and the project proceeds
directly to Gate 3. If (b): design the minimal H1 t0-rescue mechanism
first (a new, small, explicitly scoped task, not opened here), then
re-run this same audit's t0 check before freezing. Either way, H0
(Experiment B2) is unaffected and remains frozen as already decided.

---

### 6.10 Shared-domain t0 rescue — design and closure (resolves §6.7 option (b))

**Resolution of §6.7:** option **(b)** was chosen. This section
designs the minimal shared-domain t0 rescue, closes it with Steps 1-3
below, and records the final freeze decision.

**Design principle (as specified):** never run H1 alone at a t0
domain different from H0's — the core fitter's own consistency guard
already enforces this (Experiment A1 §0.4; triggered directly by this
session's diagnostic run, §6.2, before the correct anchor-manifest
override was used). The rescue is:

1. run the base domain per the frozen policies (H0: Experiment B2;
   H1: `controlled5`);
2. **if the H1 `controlled5` winner's own `t0` component of
   `optimizer_active_mask` is active**, promote that event to
   `t0_margin=0.25`, and re-evaluate **both** H0 (`gate1_final18`) and
   H1 (`controlled5`) in that shared, widened domain;
3. use the widened-domain results for that event's LRT.

This complements, and does not replace, the existing validated
H0-only t0 rescue (Experiment A1 §0.4, triggered by the H0 winner's
own `t0` active flag, independently). The new trigger reads only
`winner_H1_t0_active` (from `h1_final_audit_winner_bounds.csv`'s
`t0_active` column) — no `catalog_row` appears in the trigger logic
anywhere in `analyze_h1_final_audit.py`.

**Step 1 — `565924` case study** (`results/h1_t0_rescue_565924_case_study.csv`,
built entirely from already-existing tables: H0 base/wide from
`gate1_oracle_diagnostics.csv`'s `gate1_final18_base_t0margin0` /
`gate1_final18_rescue_t0margin0.25` sources — no new H0 fits were
needed, that global t0=0.25 diagnostic already covers all 100 events
— and H1 wide from this session's one 5-fit diagnostic, §6.5):

| quantity | value |
|---|---:|
| `chi2_H0_base` | 454.550938 |
| `chi2_H1_base` | 111.209716 |
| `DeltaChi2_LRT_base` | 343.341221 |
| `chi2_H0_wide` | 454.500884 |
| `chi2_H1_wide` | 109.846734 |
| `DeltaChi2_LRT_wide` | 344.654150 |
| change in H0 | `-0.050054` (negligible — H0 was never t0-bound for this event) |
| change in H1 | `-1.362982` (material, §6.5) |
| **change in LRT** | **`+1.312928`** |
| nestedness raw, widened (`chi2_H1_wide - chi2_H0_wide`) | `-344.654150` (no violation, comfortable margin) |
| H1 wide winner active_mask | `[0,0,0,-1,0,0]` (t0 resolved; `rho` still at its lower floor, consistent with this event's tiny `true_rho`, §6.5) |
| H1 base winner active_mask | `[1,0,0,-1,0,0]` (t0 **and** rho both active at base — the rescue resolves t0 only, as expected; the rho floor is the already-understood, benign point-source degeneracy, §6.5-6.6) |

Widening the shared domain for this event changes the **LRT by
+1.31**, driven almost entirely by H1's improvement (H0 barely
moves). This is a real, if modest, change to a real quantity used for
parallax detection — exactly the kind of case this rescue exists to
catch.

**Step 2 — trigger verification on extreme100**
(`results/h1_t0_rescue_extreme100_trigger_check.csv`, computed from
`h1_final_audit_winner_bounds.csv`'s `t0_active` column and
`gate2_h0_anchor_manifest_extreme100.csv`'s `rescue_triggered` column
only — no new fits, no hardcoded rows):

- `n_h1_winner_t0_active = 1/100` → row `565924` (this is *found*,
  not assumed — the same number already reported in §6.5).
- `n_h0_rescue_triggered` (existing, validated H0-only rescue) `=
  1/100` → row `36103`.
- `n_triggering_both = 0` — **the two rescues are disjoint in
  extreme100**: no event requires both simultaneously. (Nothing
  prevents an event from doing so in principle — the two triggers are
  independent conditions — this is simply what is observed here.)

**Step 3 — cost** (`analyze_h1_final_audit.py`, cost section, no new
thresholds or optimization):

- Conservative, non-optimized cost when the new rescue fires: complete
  `gate1_final18` (18 H0 fits, widened; no credit taken for any
  base-domain fits that might be reusable at the wider bound) + rerun
  `controlled5` (5 H1 fits, widened) = **23 extra fits**, upper-bound,
  not tuned.
- Triggers on `1/100 = 1.0%` of extreme100 events →
  **expected overhead = 0.01 x 23 = 0.230 fits/event.**
- Combined expected cost entering Gate 3:
  `H0(B2, 15.18) + H1(controlled5, 5) + shared-domain-rescue overhead
  (0.23) ≈ 20.41 fits/event` (vs. `20.18` without the new rescue) —
  small, as expected for a ~1%-triggered rescue.

**Combined H0+H1 t0-domain precedence — exact control flow for Gate 3.**
extreme100 happens to show the H0-only rescue (`36103`) and the new
H1-triggered rescue (`565924`) as disjoint (§6.10 Step 2), but Gate 3
could produce an event where they interact. The combined policy is
defined unambiguously as follows, so Gate 3 can implement it exactly,
with no ambiguity about precedence or which domain the final LRT
uses:

1. Run H0 per the frozen Experiment B2 policy (base `k=6`; escalate to
   complete `gate1_final18` only if B2's own `abs_log_ratio_tE`
   trigger fires). The validated H0-only t0 rescue (Experiment A1
   §0.4) can only be evaluated once `gate1_final18`'s full winner
   exists — i.e. only for events where B2 already escalated to
   completion; this is pre-existing B2/A1 behavior, unchanged here.
2. `shared_t0_margin := 0.0`. If the validated H0-only t0 rescue
   triggers (the `gate1_final18` winner's own `t0` component of
   `optimizer_active_mask` is active), set
   `shared_t0_margin := 0.25`.
3. Run H1 `controlled5` using `shared_t0_margin` as set by step 2 —
   i.e. if H0's own rescue already widened the domain, H1's *first*
   run already uses the widened domain, not the base one.
4. **Only if `shared_t0_margin` is still `0.0`** after step 3, and the
   H1 `controlled5` winner's own `t0` component of
   `optimizer_active_mask` is active: set
   `shared_t0_margin := 0.25`; re-run H0 robust (`gate1_final18`, full
   18 strategies) at `t0_margin=0.25`; re-run H1 `controlled5` at
   `t0_margin=0.25`; use these two widened results for the event's
   final LRT. (This is exactly the `565924` case, §6.10 Steps 1-3.)
5. **If, after `shared_t0_margin` is already `0.25`** (from step 2 or
   step 4), the H1 winner's `t0` is *still* active: **do not widen
   again and do not invent a new rule now.** Record this as a policy
   anomaly/failure for Gate 3 to report explicitly (event id, both
   chi2 values, both active masks) — it is evidence the frozen policy
   is insufficient for that event, to be evaluated after Gate 3, not
   patched during it.

**Invariant, enforced by construction in every branch above: the H0
and H1 chi2 values that enter the final LRT for an event must always
come from the identical `shared_t0_margin` for that event** — never
H0 at one t0 domain and H1 at another. Every condition above reads
only `optimizer_active_mask` (H0's or H1's own winner); **`catalog_row`
never appears in any runtime condition** of this combined policy, in
this document or in `analyze_h1_final_audit.py`. No result already
computed for `extreme100` (§6.1-6.10 above) changes under this
precedence definition — it is a restatement of exactly what Steps
1-3 already did for `565924`, made explicit and general so Gate 3 does
not have to re-derive it from one worked example.

**Freeze decision:** all four conditions set for this closure hold:
the widened shared-domain result for `565924` is internally
consistent (H0 barely moves, H1 improves materially, LRT changes by a
sensible, bounded amount); nestedness remains correct (widened margin
`-344.65`, no violation); the trigger is exactly
`winner_H1_t0_active`, verified on all 100 events with no
`catalog_row` hardcoded anywhere in the logic; no other anomaly
appeared (rho/piE were already closed clean in §6.5-6.6, unaffected
by this rescue).

**`H1 = controlled5 + shared-domain t0 rescue when the H1 winner's t0
bound is active` is frozen as `CANDIDATE — FROZEN FOR INDEPENDENT
VALIDATION`.** Gate 3 must evaluate exactly this rule (base
`controlled5`, check `winner_H1_t0_active`, and when true, widen both
H0 and H1 to `t0_margin=0.25` and use those widened results for the
LRT) without retuning the trigger, the `0.25` margin, or any bound —
the same no-retuning discipline already established for Experiment
B2's H0 policy (§10 there) applies here identically. If Gate 3 finds
this rule fails for some event (e.g. `t0_margin=0.25` is not wide
enough, or the rescue fires but does not resolve the truncation), the
same principle from Experiment B2 §10 applies: either fall back to
not rescuing (documenting the gap) or treat Gate 3 as consumed
development data and require a further independent sample — never
retune silently on Gate 3.

**No production, core-fitter, or SLURM change was made.** The design
above describes a rule; implementing it as an automatic step in
`run_bounds_audit_refit_core.py` (mirroring how the existing H0-only
t0 rescue is implemented there) is production-adjacent work explicitly
deferred, not done in this session — Gate 3 can be run by manually
applying this rule's two-branch logic (base run; widen only if
triggered) using the existing, unmodified fitter, exactly as Step 1's
case study did for `565924`.

**Next step:** H0 (Experiment B2, frozen) + H1 (`controlled5` +
shared-domain t0 rescue, frozen) → **Gate 3 independent validation**
(200-500 events, evaluating both frozen policies and this rescue rule
exactly as specified, without retuning) → Gate 3B cluster profiling →
empirical H0/null LRT calibration → staged production.

---

## 7. Optimization search closed before morphology experiment

**Status:** all sub-experiments below are **CLOSED — FAIL** or
**CLOSED — EXACT RESULT**, not to be repeated on `extreme100` without
new evidence. This section is the checkpoint required before starting
the morphology-informed-start experiment (§8).

**Context.** After freezing H0 (B2, §Experiment B2 elsewhere in this
log) and H1 (`controlled5` + t0 rescue, §6) for Gate 3, a Gate-3
preflight investigation found that both frozen policies structurally
depend on `old_H0` (a legacy TRF fit under old, invalid, narrow bounds)
as a seed: 8 of `gate1_final18`'s 18 H0 strategies and 1 of
`controlled5`'s 5 H1 strategies (`old_final_reseed`) are anchored on
it. Since Gate 3 needs this to work on **new**, previously-unfitted
events, a `legacy seed generation` stage (2 extra TRF fits/event) was
initially identified as unavoidable overhead (§7.1 below). Everything
in this section is the subsequent search for a way to **avoid or
reduce** that overhead — reusing already-paid computation instead —
before accepting it. All of it failed to reduce the cost; the exact
worst-case H0 cost is now proven to be irreducible within the studied
strategy pool (§7.3).

**`extreme100` remains development-only** throughout this section — no
Gate 3 event was touched. No production code, core fitter
(`run_bounds_audit_refit_core.py`, `run_bounds_audit_refit.py`), bounds
profile, `oracle52`, or frozen policy was modified. Gate 3 remains
paused.

**Reproducibility.** All scripts referenced below live in this
directory (`validation/bounds_convergence/`) and read/write
`results/*.csv`; run them with this directory as the working directory
(matches the convention of every other `analyze_*.py` script in this
log). They import the frozen fitter from `../bounds_audit/` read-only
and never modify it. Where a script needs to fix the TRF coordinate
mode (`physical`/`log_te`/`log_rho`/`log_te_rho`), it sets
`HIDDEN_PARALLAX_TRF_COORDS` **before** importing
`run_bounds_audit_refit_core`, because that module installs its
coordinate-mode monkeypatch once, at import time — this is why some of
the scripts below are split one-mode-per-process rather than looping
over modes in one run.

### 7.1 Legacy-seed cost audit (`old_H0` / `old_final` provenance)

Traced `old_H0` (→ `TRF_FSPL_NoParallax.npy`) and `old_final`
(→ `TRF_FSPL_Parallax.npy`) to their exact producer in
`~/microlensing/simulation_Rubin/roman_rubin/functions_roman_rubin.py`
(`sim_fit_multi_fits`) and `fit_lc.py`. Findings:

- `old_H0` = **exactly 1** real nonlinear-optimizer (TRF) call, start
  = truth, old narrow bounds (t0=truth±30d, u0=±1, tE∈[0.1,20000],
  rho∈[1e-7,10]). No forward-only part.
- `old_final`: re-audited in this checkpoint (§0 correction below) —
  **exactly 10 unconditional real TRF calls** (the grid) **+ 0 or 1
  conditional real TRF call** (the polish), i.e. **10-11 real
  optimizer calls total, not ~1**. None of it is forward-only. See
  the corrected breakdown immediately below — the original text here
  said "≈1 TRF-equivalent for `old_final`'s grid+polish" in the same
  paragraph that also said "each grid point a real TRF fit", which is
  self-contradictory; that was a documentation error, now fixed.
- Only the raw parameter vectors are consumed downstream
  (`old_h0[:4]`, `old_h1` full 6-vector) — `old_h0_chi2`/`old_h1_chi2`
  are print-only diagnostics, never used in any control-flow decision.
- Reusing old-bounds provenance is not scientifically required — the
  vector is just an initial guess, fully re-optimized under
  `production_candidate` bounds by the frozen fitter.

**Correction (this checkpoint): exact `old_final` call count,
traced from `run_lsstmonts_catalog_hidden_parallax.py`
(`install_h1_parallax_grid_multistart_runtime_patch`), no new fits
run to determine this — read-only trace of the historical producer
and its config (`configs/validation/lrt_benchmark/FAST.json`):**

- Candidate grid: `piE_grid_fractions=[-0.5,0,0.5]` → 3×3=9 grid
  points, plus `include_exact_H0_center=true` adds a 10th
  (`center_full`) point at exactly `(0,0)`. `include_auto_center=true`
  means the grid's own `(0,0)` point is **not** deduplicated against
  `center_full` — both are kept as distinct candidates. **→ 10
  candidates.**
- Each candidate is run through `base_multi(...)` → `sim_fit` →
  `fit_rubin_roman` with `initial_guess=start_guess` — **a genuine,
  unconditional, real TRF optimization for every one of the 10**, not
  a forward/cheap evaluation. Confirmed by direct code trace
  (`H1_multistart` candidate loop), not inferred.
- After the loop, `polish_winner="adaptive"` (FAST.json's actual
  setting): a further real TRF call, from the coarse winner's full
  solution vector with tighter tolerances
  (`xtol=ftol=1e-12` vs the grid's default), is run **only if** the
  coarse winner's `optimizer_optimality` is non-finite or exceeds
  `polish_optimality_threshold=0.05` — i.e. **conditionally 1
  additional real TRF call, not unconditionally, and not zero either**
  (this session did not re-run the historical production data to
  measure how often the threshold actually triggered, since that
  would require new fits; the correct statement is "0 or 1 depending
  on the coarse winner's fit quality", not a fixed number).
- **"TRF-equivalent" in the superseded text was not a wall-time
  measurement** — no wall-time profiling of the historical `old_final`
  production run was performed in this or any prior session. It was
  an unjustified cost simplification and has been removed.

**Corrected conclusion: legacy-seed generation, if it had to be
reproduced from scratch for a new event, costs 1 real TRF call for
`old_H0` + 10-11 real TRF calls for `old_final` ≈ 11-12 real
optimizer calls/event — not the "≈2 TRF/event" this section
previously claimed.** This is a **documentation-only correction**: it
does not touch or weaken §7.3's exact MILP result (`minimum true H0
cost = 19`), because that MILP's "bridge" is `old_H0` alone (cost 1) —
`old_final` was never part of the H0 cost model; it only ever fed
H1's `old_final_reseed`, accounted separately in the H1 policy
(§6/`controlled5`). No legacy fit was run to produce this correction —
it is a code/config trace only.

### 7.2 Self-seeding: reuse already-paid fits instead of a legacy stage — CLOSED, FAIL

**Question:** can `old_H0`/`old_final` be replaced by re-seeding from a
fit the frozen policy runs *anyway* (zero extra TRF), instead of a
separate legacy-bounds fit?

**H0 `selfseed_final18` — FAIL.** `current_H0_anchor` = min-chi2 of the
4 truth-anchored `k=6` fits (`physical/truth/{0.1,1}`,
`log_rho/truth/{0.1,1}`, `log_te/truth/1`; picked once/event, no extra
fit). Re-ran the 8 `old_H0`-anchored `gate1_final18` slots from this
anchor instead of `old_H0` — 96 real TRF fits (12 essential events ×
8 strategies; script `run_h0_selfseed_batch.py` /
`run_dynamic_h0_fit.py`, data `results/h0_selfseed_results.csv`).
Result vs `gate1_final18`: **N(Δ>0.1)=8/12, N(>1)=4, N(>10)=3,
worst=3522.1** (row 62786) — improves on pure legacy-free (12/12
failing) but does not reach zero. **Closed FAIL.**

**H1 `current_H0_grid_reseed` — FAIL, and *why* matters.** Proposed
replacement for `old_final_reseed`: current H0 final (already paid) +
the same 3×3 piE grid, but evaluated with a **true fixed-point forward
objective** (`fit.objective_function(x)`, zero optimizer steps, exact
`bounded_flux_profile`) instead of a partially-optimized probe, then
one real TRF polish of the winner (`run_h1_grid_reseed_v2.py`, data
`results/h1_grid_reseed_results_v2.csv`; a first version,
`run_h1_grid_reseed.py`, used `max_nfev=3` "grid" probes and produced
internally inconsistent results — grid-best chi2 sometimes *beaten* by
the full polish from the same nominal point — which is impossible for
a monotonic trust-region method and was traced to those probes not
being true fixed-point evaluations; that version's results are
superseded and not kept).

With a genuine forward evaluation, **the grid winner is always the
exact center `(piEN,piEE)=(0,0)`, for all 5 test events** — moving off
center while holding `(t0,u0,tE,rho)` fixed at the H0-only optimum is
essentially always worse, because those backbone parameters were
optimized *for* `piE=0`. The resulting "5th H1 start" is therefore
always numerically identical to `H0_NESTED_piE_0`, already inside
`controlled4`, so it rescues nothing beyond what `controlled4` already
has: **N(Δ_H1>0.1)=5/5 unchanged from `controlled4` alone**
(85380: Δ=3.506, 79700: Δ=0.978, 50179: Δ=0.420, 86451: Δ=0.261;
87786: Δ=−2.939, i.e. this one already happens to beat the legacy
reference via `controlled4`, unrelated to the grid).

**Mechanistic takeaway, applies to both H0 and H1:** any diagnostic or
seed-construction scheme that holds the nonlinear backbone fixed and
only perturbs a subset of parameters (a grid, a forward evaluation)
cannot discover an improvement that requires those parameters to
**co-adapt**. This is the single root cause behind every FAIL in this
section (§7.2, §7.4, §7.5).

### 7.3 Dependency-aware H0 cost — CLOSED, EXACT RESULT

**Question:** B1/B2's original strategy count treated every strategy
as costing 1, ignoring that the 8 `old_H0`-anchored `gate1_final18`
members share one `old_H0` fit (bridge). Re-derive the true minimum
H0 cost under `cost(S) = |S| + 1{S contains any old_H0-anchored
strategy}`.

All results below are **exact** (scipy `milp`, HiGHS branch-and-cut,
`status=0`/proven-optimal in every solve, not a greedy heuristic —
verified against brute-force exhaustive search for small k too),
computed only from the 52 already-existing oracle strategy results
(`results/gate1_oracle_diagnostics.csv`) plus one new read-only
quantity (below). Script: `h0_exact_milp.py`.

1. **Baseline exact minimum strategy count (no bridge concept)
   = 18**, requiring exactly 8 `old_H0`-anchored strategies — this
   reproduces (and formally validates) the original
   `analyze_gate1_start_cover.py` MILP result that produced
   `gate1_final18`; that original result was itself already an exact
   MILP solve, not greedy.
2. **Truth-only coverage of all 100 events is proven infeasible**
   (`status=2`, HiGHS-proven, not a search-budget artifact) — at least
   one `old_H0`-anchored strategy is mathematically mandatory for
   zero-failure coverage within this pool.
3. **`old_H0[:4]` re-evaluated with the current objective** (forward,
   zero optimizer steps, exact `bounded_flux_profile`, all 100 events
   — script `eval_old_h0_vector_current_objective.py`, data
   `results/old_h0_vector_current_objective.csv`) rescues only
   **6/100 events on its own** (median Δ≈109) — not usable as a
   general-purpose free candidate.
4. **Exact dependency-aware MILP, allowing that re-evaluated bridge
   point as a free coverage option once `bridge=1`: minimum total
   true TRF cost = 19** (18 downstream + 1 bridge), **unchanged** by
   including the free bridge-vector option — the 6 events it covers
   are already covered redundantly by other mandatory strategies.

**`minimum true H0 worst-case cost = 19 TRF` is an exact result within
the 52-strategy oracle pool studied on `extreme100`, not a
mathematical statement about any conceivable fitter or strategy
design** — a genuinely new strategy family (not in the 52-pool) could
in principle do better; none tested in this session did.

### 7.4 Fisher/curvature as a stopping diagnostic — CLOSED, FAIL

**Question:** after 1-2 real TRF fits, does local curvature
(Gauss-Newton Fisher of the profiled-flux objective, dimensionless
coordinates `(Δt0/tE, u0, log tE, log rho[, piEN, piEE])`, documented
transform in `fisher_engine.py`) let us **certify** a subset of events
safe to stop on, with zero false-safes?

Implementation (`fisher_engine.py`, `run_fisher_h0.py`,
`run_fisher_h1.py`, `analyze_fisher.py`; data
`results/fisher_h0_results.csv`, `results/fisher_h1_results.csv`)
validated bit-exact against the frozen fitter's own chi2 (<1e-7 H0,
<1e-11 H1 agreement).

- **H0** (step1 = best single truth strategy, step2 = best truth pair,
  both already-existing fits, no new TRF): every Fisher predictor
  tested (condition number, min eigenvalue, logdet, max correlation,
  per-coordinate σ, weakest-eigenvector components, inter-step Fisher
  distance) sits at AUC 0.50-0.63. Best zero-false-safe stop fraction:
  **6% after 1 TRF, 13% after 2 TRF** (best 2-predictor OR).
- **H1** (truth → full TRF): somewhat higher AUC (up to 0.65,
  Schur-complement `F_pi|eta` diagnostics ≈0.62-0.64) but zero-false-safe
  stop fraction still collapses to **4% individually, 5% best 2-combo
  OR**.
- Fisher is cheap (≈1/20-1/32 of one TRF in wall time) but that is
  irrelevant given the predictive power.

**Root cause (§7.2's mechanistic takeaway applies again): local
curvature at a solution is blind to the existence of a distant,
better basin** — a sharply-determined local optimum looks identical
in Fisher terms whether or not a better one exists elsewhere.
**Closed FAIL as a standalone stopping diagnostic.**

### 7.5 Multi-fidelity scouting (cheap global scout → rank → full TRF only for top-k) — CLOSED, FAIL

**Question:** does *some* cheap global signal (not local curvature)
correlate with final optimized quality well enough to safely skip
full optimization on most strategies?

**Stage A — fixed-start forward chi2 — FAIL.** For each event,
forward-evaluated (zero optimizer steps) the literal initial vector of
all 13 distinct physical starts used by the 52-strategy oracle (mode
does not affect the initial vector, only the optimizer's internal
path, so 52 strategies collapse to 13 distinct starting points ×
4 modes). Script `run_h0_scout_stageA.py` (1300 forward evals, data
`results/h0_scout_stageA_starts.csv`), ranking analysis
`analyze_h0_scout_stageA.py` (`results/h0_scout_stageA_summary.csv`).
Result: Spearman correlation between start-chi2 rank and final
optimized-chi2 rank, per event, **mean −0.02, median −0.04** —
essentially zero/noise. Winner localization: true winner within
top-10 only **17%** of events, top-1 only **2%**. Top-k simulation
(existing full-TRF results, k=1..10) never gets close to zero failures
(62/100 still failing at k=10). **Closed FAIL** — an un-adapted
starting chi2 does not predict optimized quality.

**Stage B — short bounded-TRF scouts, `gate1_final18`'s exact 18
strategies, run under their own coordinate mode, budgets
`max_nfev∈{3,5,10}` frozen before running — CLOSED, FAIL.**
Scripts: `run_h0_scout_stageB.py` (one process per mode, since TRF
coordinate mode is fixed at `core.py` import time),
`audit_stageB.py`, `analyze_stageB_topk.py`,
`calibrate_full_trf_timing.py`. Data:
`results/h0_scout_stageB_{physical,log_te,log_rho,log_te_rho}.csv`
(5400 rows total).

- **Audit: 5400/5400 scout records present and correct** — 18 unique
  `(mode,anchor,rho_tag)` strategies × 100 events × 3 budgets, zero
  duplicates, zero missing/extra combinations, zero monotonicity
  violations (chi2 non-increasing as budget grows, confirming TRF's
  trust-region descent property held throughout), nfev never exceeds
  its budget, and scouts that converged before their budget cap
  reproduce the original oracle52 chi2 to ~1e-7. (An earlier verbal
  status update mis-summed the per-mode row counts as 4500/5400 — a
  simple arithmetic slip in a chat message, not a data or code bug;
  the underlying files were always complete, as this audit confirms.)
- **No `(budget, k)` combination for k=1..10 reaches zero failures.**
  Floor: **10/100 failures**, reached at budget=10, k≥9, and it does
  not improve further with more scouting.
  Worst-case delta at that floor is still ≈71 — nowhere near the 0.1
  tolerance.
- **Cost: scouting is not even cheaper for what it does achieve.**
  Scouting all 18 strategies at budget=10 alone (≈382s/100 events)
  already costs ≈87% of the fixed 19-TRF policy (≈439s/100 events,
  measured); the best-performing scout configuration (budget=10, k=9,
  still 10/100 failing) costs **≈566s — more than the fixed policy**,
  before even counting that it still leaves 10 events unresolved.

**Closed FAIL: no tested combination reaches zero failures, and none
of the ones that get close cost less than the fixed policy — the two
conditions that would have justified keeping this direction.**

### 7.6 What NOT to repeat without new evidence

- Do not re-run H0 self-seeding or H1 fixed-backbone-grid reseeding —
  both are structurally blocked by §7.2's mechanistic finding
  (co-adaptation is required; nothing that holds parameters fixed can
  see it), not by an insufficient sample or an untried variant.
- Do not re-derive the minimum H0 strategy count/cost by hand or by
  greedy search — §7.3 is an exact, HiGHS-proven result. A different
  answer would only ever come from changing the strategy pool itself
  (a genuinely new starting-point family), not from re-optimizing the
  selection over the existing 52.
- Do not retune Fisher thresholds, add multivariate/ML combination of
  Fisher predictors, or extend it to other coordinates — §7.4's root
  cause (local curvature can't see distant basins) does not depend on
  which particular curvature summary is used.
- Do not retune Stage A/B budgets, grids, or extend Stage B to more
  events — §7.5 is closed on both economic grounds (cost) and
  coverage grounds (never reaches zero), independently.
- Any new heuristic in this family should be evaluated first against
  the same root cause identified in §7.2: does it let the parameters
  that actually determine basin identity co-adapt, or does it hold
  something fixed and only look at a cheap proxy?

### 7.7 Reference costs used throughout this section

- Fixed robust H0 policy (`gate1_final18` + bridge): **19 TRF**,
  exact minimum (§7.3), ≈439s measured per 100 events (mean full-TRF
  wall time ≈0.23s, averaged across the 4 coordinate modes). §8
  later finds an exact 16-TRF architecture that supersedes this.
- H1 `controlled5` + t0 rescue: **5 TRF nominal is the current-policy
  fit cost only** (running `controlled5` on an event that already has
  a usable `old_final` vector to reseed from) — it is **not** the true
  deployable-from-scratch cost for a new event. See §9 for the
  corrected accounting; the "5 TRF nominal" figure here was accurate
  as written but is easy to misread as the total H1 cost, which it
  never was.
- No cheaper alternative to either was found in this session.

**Next step:** morphology-informed H0 starts — see §8.

---

## 8. Morphology-informed H0 starts

**Status:** offline diagnostic (M0-M6) **PASSED**; minimal TRF test
(M7) gives a **partial, non-trivial rescue (5/12 critical events)**,
not a clean pass — decision on whether to extend to `extreme100` is
left open below, not made unilaterally in this session.

**Question:** can the *observed* light-curve shape (not truth, not
Fisher, not a short scout) be used to construct an H0 start
`(t0,u0,tE,rho)` that lands closer to the H0 oracle basin than the
truth-anchored starts already tried — specifically on the 12 events
where `old_H0` is essential (§7)? Truth/oracle are used only to
*evaluate* the result, never to build it.

### M0 — frozen morphology extraction

`morphology_extraction.py`. Per band: robust (inverse-variance-weighted
median) baseline magnitude, dimensionless excess-flux proxy
`excess(t) = 10^(-0.4(m(t)-m_base)) - 1`, error-propagated. A
heteroscedastic weighted smoothing spline (`UnivariateSpline`,
`w=1/err_excess`, `k=3`, `s=N_band` — scipy's own documented
rule-of-thumb for `w=1/sigma` weights) per band; **no Gaussian
Process anywhere**. Bands combined only *after* this per-band
normalization, via a single per-band scalar SNR weight — never mixing
raw fluxes across filters. Peak and `T25/T50/T75` are the width of
the largest contiguous region above a threshold fraction of peak
excess; a side that never drops below threshold within the
data-supported range is recorded `censored`/`not measurable`, never
extrapolated. Full definition frozen and documented in the module
docstring before any oracle comparison. Run on all 100 events
(`run_morphology_extraction_all_events.py` →
`results/morphology_M0_M1_M2_extreme100.csv`): **100/100 measurable**
at the top level; T50 fully uncensored (both sides) for 92/100;
`peak_SN` computable for 90/100 (the other 10 have zero data points
falling inside the geometric T50 window despite a measurable spline
peak — flagged as `NaN`, not fabricated; a real "poorly measured"
case, not a bug).

### M1/M2 — observables and quality

`t_peak_morph`, `T25`, `T50`, `T75` (+ `T50_left`/`T50_right`,
`A50 = (T50_right-T50_left)/(T50_right+T50_left)`, diagnostic only,
never an FSPL parameter), `peak_SN`, `integrated_SN`, point counts,
`n_contributing_bands` — all in the same CSV above. No trained
classifier built from these; they are reported and correlated
descriptively only (see M5).

### M3/M4 — FSPL dimensionless lookup + inversion

`build_fspl_dimensionless_lookup.py`: grid over `|u0|∈[1e-3,3]` ×
`rho∈[1e-4,2]` (60×60, log-spaced, generic — not tuned against
`extreme100`), pyLIMA's `magnification_FSPL_Yoo` (Yoo et al. 2004,
no limb darkening, `gamma=0`), extracting `T25/T50` and `T75/T50` in
units of `tE`. **Confirmed exactly, not assumed:** H0's FSPL
magnification is bit-identical under `u0 -> -u0`
(`max|A(u0)-A(-u0)|=0.0` over a test grid) — H0 is exactly degenerate
in `u0` sign, so only `u0_morph > 0` (canonical) is ever produced, no
mirrored duplicate starts. 2760/3600 grid points usable (others: peak
never drops below threshold within the fixed `tau∈[-15,15]` range, or
grid resolution insufficient).

`invert_morphology_to_h0_start.py`: inverts observed `(T25/T50,
T75/T50)` against the grid (nearest-match + up to 2 more genuinely
distinct local minima, ≥0.3 dex apart in `(log u0, log rho)` and
within 3× the best mismatch — M4's explicit anti-duplication rule).
**100/100 events invertible**, 98/100 produced all 3 candidate slots
(real degeneracy in the 2-ratio inversion is common and not hidden).
`t0_morph = t_peak_morph`; `tE_morph` from the absolute `T50` scale
divided by the grid point's dimensionless `T50/tE`.

### M5 — offline diagnostic (zero new TRF)

`analyze_morphology_vs_oracle.py` → `results/morphology_M5_offline_diagnostic.csv`.
Dimensionless joint distance `D = sqrt(Δ(t0/tE)^2 + Δu0^2 + Δ(log tE)^2
+ Δ(log rho)^2)` from each event's rank-1 morphology candidate to
truth, to the oracle52 winner, and to the raw `old_H0` vector.

**Critical group (n=12):** median `D(morph,oracle) = 3.05` vs median
`D(truth,oracle) = 5.97` — a clear reduction. **10/12 events move
closer to the oracle basin than truth does** (the 2 that don't:
563210, 567800). Median `D(morph,truth) = 5.83` — morphology is
**not** a trivial truth-proxy (if it were, this would be ≈0). Rest
group (n=88) shows the same direction (70/88 closer), so the effect
is not an artifact of the critical group's own construction.
Descriptive correlation of the improvement with quality diagnostics:
weak negative with `peak_SN` (−0.40, i.e. no evidence that only
high-SNR events benefit — if anything mildly the opposite), weak
positive with band count (+0.13), negligible with `A50` (−0.03) — no
strong, simple explanatory driver identified; not pursued further
(explicitly out of scope: no classifier built on this).

### M6 — preregistered decision (frozen before this run)

Criteria (documented in `analyze_morphology_vs_oracle.py` before
execution): (a) median distance on the 12 critical events improves by
more than 30% (`morph < 0.7×truth`), (b) a clear majority (≥60%,
i.e. ≥8/12) move closer, (c) not a trivial truth-proxy. **All three
hold** (49% median reduction, 10/12, `D(morph,truth)` far from zero).
**M6 = PASS.**

### M7 — minimal TRF test (12 events only, full convergence)

`run_h0_morphology_trf.py`, one process per coordinate mode (mode
fixed at `core.py` import time, as throughout this log) ×
`production_candidate` bounds × current `bounded_flux_profile` ×
default (unbudgeted) `OPTIMIZER_OPTIONS` — **48 new real TRF fits**
(12 events × 4 modes), the rank-1 morphology candidate as the only
start tested, no reduced budget, no new parametrization.

| row | chi2 (best of 4 modes) | oracle52 | delta | winner mode |
|---|---|---|---|---|
| 83189 | 163.178141 | 163.178116 | 0.00003 | log_rho |
| 567800 | 347.952416 | 347.952406 | 0.00001 | log_rho |
| 574423 | 257.312533 | 257.312224 | 0.00031 | physical |
| 35927 | 200.890880 | 200.887454 | 0.00343 | log_te |
| 79501 | 117.665383 | 117.623327 | 0.04206 | log_te |
| 82728 | 410.223349 | 409.967004 | 0.256 | log_te_rho |
| 567365 | 385.231833 | 383.598845 | 1.633 | physical |
| 558189 | 1434.196973 | 1429.085570 | 5.111 | log_te |
| 579320 | 28929.346972 | 28895.268509 | 34.08 | log_te_rho |
| 563210 | 1741.288128 | 1190.740949 | 550.5 | log_te_rho |
| 557860 | 946.547991 | 296.115322 | 650.4 | log_te_rho |
| 62786 | 103384.846317 | 100020.801881 | 3364.0 | log_te_rho |

**N(Δ>0.1)=7/12, N(>1)=6, N(>10)=4, worst=3364.0 (row 62786).
5/12 rescued exactly (Δ≤0.1).** This is a genuine, non-trivial partial
result — better than pure legacy-free (0/12, §7.3's truth-only proof)
and comparable to H0 self-seeding (4/12, §7.2) — but **not a clean
pass**: 3 of the 7 remaining failures are still catastrophic
(550-3364 in chi2), not near-misses.

### Decision (left explicit, not made unilaterally here)

Per the preregistered M7/M8 rule ("only if this test really rescues
an important fraction of the 12" → evaluate extending to
`extreme100`; otherwise accept the 19-TRF worst-case and stop): 5/12
is a real but partial signal, consistent with the pattern of every
other direction in §7 (real improvement, never full closure). This
session did **not** extend the test to the full 100 events or
recompute the exact set-cover cost with morphology as an added
candidate family — that decision is left to the next turn. **No
retuning of the smoothing rule, width thresholds, lookup grid range,
or SNR definitions was done to try to improve this result** — every
parameter above was frozen before M5 was run.

**Cost so far this section:** 0 TRF for M0-M6 (pure forward/offline
work) + 48 TRF for M7 = **48 new TRF fits total**, all confined to the
12 critical events, none on the other 88.

### M8-extension — structural capacity check before extending to `extreme100` (no new fits)

Before running the other 88 events, the 48 already-existing M7 fits
were turned into **4 new standalone strategies** —
`morphology/{physical,log_te,log_rho,log_te_rho}` (cost=1 each, no
`old_H0` bridge dependency) — and re-solved with the same exact,
HiGHS-proven dependency-aware MILP from §7.3, extended with these 4
columns. For the 88 not-yet-tested events, two bracketing scenarios
were solved exactly: **Scenario P** (pessimistic: morphology coverage
= False for all 88, real observed coverage for the 12 tested) and
**Scenario O** (optimistic lower bound: coverage = True for all 88,
tested-12 coverage never overridden). Preregistered decision rule:
`optimistic ≥ 19` → close without testing the 88 (Case A);
`pessimistic < 19` → the already-observed rescues alone are
structurally sufficient, extend to the full 100 (Case B);
`pessimistic = 19` but `optimistic < 19` → depends on the untested 88,
extend to justify (Case C).

Result: **Scenario P (pessimistic) = 18 < 19** (replacing 2 of
`gate1_final18`'s `old_H0`-anchored strategies with 1 morphology
strategy, `log_te_rho`). Force-no-bridge was infeasible under both
scenarios (7/12 critical events are, in reality, uncovered by any
morphology mode, independent of any assumption about the other 88, so
the bridge cannot be eliminated even optimistically). **Case B: the
decision to extend to `extreme100` did not depend on any assumption
about the untested events — it was already proven by the 12 tested
ones alone.**

### Extension to the full `extreme100` (352 new TRF fits: 88 × 4 modes)

Exactly the same frozen procedure as M7 (rank-1 morphology candidate,
full convergence, `production_candidate` bounds, current
`bounded_flux_profile`, one process per coordinate mode), run on the
remaining 88 events (`run_h0_morphology_trf_remaining88.py`). No new
morphology definition, no rank-2/3, no new width levels, no smoothing
or lookup-grid change, no GP, no scouts, no Fisher, no classifier, no
new coordinate parametrization — the only new "strategy family" is
the one already frozen: rank-1 morphology start × the 4 existing
modes.

**Audit (`audit_and_combine_morphology100.py`): PASS.** 400/400
records (100 events × 4 modes), zero duplicates, zero missing/extra
combinations, every row's initial vector verified to match the frozen
rank-1 morphology candidate for its event exactly, all 400 fits
`status=success`, combined into
`results/h0_morphology_trf_all100.csv`.

**Coverage (`analyze_morphology100_coverage.py`), 100 events:**

| mode | N covered/100 | N(Δ>0.1) | N(>1) | N(>10) | worst |
|---|---|---|---|---|---|
| physical | 43 | 57 | 47 | 33 | 118114 |
| log_te | 40 | 60 | 48 | 32 | 81463 |
| log_rho | 35 | 65 | 55 | 35 | 463929 |
| log_te_rho | 21 | 79 | 68 | 43 | 29544 |

**Union coverage (≥1 mode) = 63/100.** Unique (non-redundant)
contributions: physical 6, log_te 3, log_rho 8, log_te_rho 3;
43/100 events are covered redundantly by ≥2 modes — running all 4
modes is not simple duplication, each contributes real, distinct
coverage. Split: critical(12) union = 5/12 (unchanged from M7); the
other 88 = **58/88 (66%)**, markedly higher than the critical group's
rate, consistent with "critical" being specifically the events where
truth-anchored search already failed hardest.

**Final exact MILP** (`h0_exact_milp_final_with_morphology100.py`,
real 100-event coverage, no assumptions, bridge-vector free-coverage
option included exactly as in §7.3, HiGHS `status=0`/proven optimal):

- **Exact minimum total H0 TRF cost = 16** (down from the baseline
  19; the 18-pessimistic bound from the P/O check is now superseded
  by this exact, assumption-free result).
- Selected: 15 downstream strategies + bridge = **16**. Composition:
  7 truth-anchored, 6 `old_H0`-anchored, **2 morphology**
  (`morphology/log_rho`, `morphology/log_te_rho`).
- **`old_H0` bridge remains required** (6 `old_H0`-anchored strategies
  still selected). `force bridge=0` with the real, fully-observed
  100-event morphology coverage is **exactly infeasible**
  (HiGHS-proven, not assumption-based this time) — morphology, even
  now fully tested, cannot eliminate the legacy `old_H0` dependency,
  only reduce the downstream cost around it.
- Relative to the original 18-strategy `gate1_final18`: 5 strategies
  drop out (`log_rho/truth/1e-06`, `log_te/old_H0/0.01`,
  `log_te/old_H0/0.1`, `log_te_rho/truth/1e-06`, `physical/truth/1`),
  replaced by the 2 morphology strategies — net −3 in strategy count,
  −3 in total cost (19→16).

**352 new TRF fits were run in this extension** (88 × 4); combined
with M7's 48, morphology's total footprint in this investigation is
**400 TRF fits**, none of it touching the other 88 events' truth or
`old_H0` reference computations (those were already-existing oracle52
data, reused read-only throughout).

### Freeze — H0 strategy-family search closed

Per the explicit stop rule for this cycle: **the search for new H0
strategy families on `extreme100` is closed here.** No further
morphology variants (candidate rank ≥2, alternative width thresholds,
different smoothing, a re-tuned lookup grid, GP, classifiers) will be
tried, even though 37/100 events remain uncovered by the current
morphology family — that gap is accepted, not chased further in this
cycle.

**H0 candidate architecture to carry into independent Gate 3
validation:** the exact-cost-optimal 16-strategy set above (7 truth +
6 `old_H0` + 2 morphology, + the shared `old_H0` bridge), superseding
`gate1_final18`'s 19-cost architecture as the current best-known,
exactly-verified-on-`extreme100` H0 policy. This is a candidate for
independent validation, not yet a frozen production policy — the same
"CANDIDATE — FROZEN FOR INDEPENDENT VALIDATION" discipline used for B2
and `controlled5` applies once this is formally adopted as the H0
policy (not done automatically by this checkpoint; a deliberate
freeze step, analogous to B2's, is still needed before Gate 3 can use
it).

---

## 9. Corrected H1 deployment-cost accounting (documentation only — no new H1 fits)

**Trigger:** §7.1's correction of `old_final`'s real producer cost
(10 unconditional + 0-1 conditional real TRF calls, not "~1
TRF-equivalent") changes the interpretation of H1's true deployment
cost for a *new* event, even though nothing about `controlled5`
itself changed. This section makes that distinction explicit
everywhere it matters. **No H1 fit was run to produce this
correction** — it is a direct consequence of §7.1's code trace,
applied to the H1 side of the same legacy-seed problem.

### Two different numbers, previously conflated

**Current-policy fit cost** — what `controlled5` costs *given that an
event already has a usable `old_final` vector* (true for every
`extreme100` development event, since all of them carry the
historical legacy artifacts):

> `controlled5` = **5 current-bounds H1 TRF optimizations**
> (`truth`, `H0_NESTED_piE_0`, `truth_half_piE`,
> `truth_mirror_u0_piEN`, `old_final_reseed` — the last one re-seeded
> from the existing `old_final` vector and re-optimized under
> `production_candidate` bounds by the frozen fitter). This is the
> number used correctly throughout §6 and §7 for `extreme100` work.

**Deployable-from-scratch prerequisite cost** — what `controlled5`
actually requires for a genuinely **new** event (e.g. a Gate 3 event)
that has no pre-existing `old_final` artifact:

> `controlled5` (5) **+ 10-11 legacy H1 TRF optimizations** to
> fabricate `old_final` from scratch (§7.1's corrected count: the
> FAST 3×3+1 grid, all real TRF calls, plus the conditional polish) =
> **15-16 real TRF optimizations for H1 alone**, not 5.

The `old_H0` bridge (1 TRF, needed to fabricate `old_H0` itself,
which `old_final`'s grid embeds as its center) is **shared with H0**
if H0's own bridge has already been paid for that event — it must
**not** be double-counted in a combined H0+H1 per-event budget. A
combined from-scratch H0+H1 budget for one new event is therefore
approximately:

> `H0 (16, §8, includes 1 bridge) + H1 downstream (5 controlled5 + 10-11
> old_final grid+polish) = ~31-32 real TRF optimizations/event`,
> of which only **1** (the shared `old_H0` bridge) is common to both
> hypotheses — not `19 + 5 = 24` as a naive pre-correction estimate
> would have suggested, and not `16 + 5 = 21` either.

### Why this matters now

Before this correction, the working assumption was that legacy-seed
overhead was small (≈2 TRF/event total, H0+H1 combined) and that H0's
`old_H0` dependency was therefore the dominant open cost problem.
**That is no longer true.** `old_final`'s real cost (10-11 TRF) is
larger than H0's entire exact-optimal downstream budget (15,
excluding bridge, per §8) and larger than `controlled5` itself (5).
**H1 legacy-seed removal/replacement is, after this correction, the
larger of the two deployment-cost problems facing Gate 3** — larger
in absolute TRF count than anything still open on the H0 side.

### What is already closed and must not be repeated

The corrected fixed-backbone `current_H0_grid_reseed` test (§7.2)
remains **CLOSED — FAIL**: it does not become more attractive under
this cost correction, because its failure mode (the grid collapses to
the exact center, rescuing nothing beyond `H0_NESTED_piE_0`, already
in `controlled4`) is structural, not a cost artifact. **Do not repeat
it.** Whatever eventually replaces `old_final_reseed` will have to let
`(t0,u0,tE,rho,piEN,piEE)` co-adapt during the search, exactly as the
§7.2 root-cause finding already established.

### Left open, not resolved here

This section documents the corrected cost, it does not solve it. No
H1 self-seeding, scouting, Fisher, or morphology variant was tried or
re-tried for H1 in this session. That is explicitly the next
open problem for a future session, not started here.

---

## 10. H0 freeze and H1 legacy-seed replacement attempt (co-adaptive candidates)

**H0 freeze:** the 16-cost architecture from §8 is tagged
`h0-morphology-freeze-2026-09-15` on commit `7dadcbd`, message
"H0 candidate frozen for independent validation... cost=16... H0
strategy-family search closed on extreme100 as of this tag." No
further H0 strategy search is planned on `extreme100`.

**H1 objective:** replace `old_final_reseed` (deployable-from-scratch
cost 10-11 legacy TRF, §9) with a start built from information already
available at H1-fit time — no legacy producer, no fixed-backbone grid
(that remains **CLOSED FAIL**, §7.2 — this is a *different* mechanism
because here **all six H1 parameters co-adapt** during a full TRF,
nothing is held fixed). **Truth `piEN`/`piEE` is legitimate pre-fit
information in this Monte Carlo simulation only** — this strategy is
explicitly not a template for real-data fitting, where truth is
unknown; it is being used here purely to test whether the *basin* that
`old_final` finds is reachable at all once parallax is
seeded roughly correctly, as a necessary first check before
addressing the harder “piE also unknown” problem.

### H1-A — `H0_FINAL_TRUE_PIE`

Initial vector: `(t0,u0,tE,rho)` = the final robust H0 solution
already available under the current architecture (the same reference
`H0_NESTED_piE_0` already uses, `gate2_h0_anchor_manifest_extreme100.csv`
— no new H0 fit), `(piEN,piEE)` = truth. All 6 H1 parameters free,
`production_candidate` bounds, current `bounded_flux_profile`, same
optimizer options as every other H1 fit in this log, physical mode
(matching `controlled4`/`controlled5`'s own precedent). **5 new H1
TRF fits**, exactly the 5 `old_final`-essential events:

| row | chi2 new | chi2 controlled5 | delta | pass |
|---|---|---|---|---|
| 85380 | 92.086 | 91.989 | **0.096** | **PASS** |
| 87786 | 269.519 | 266.735 | 2.784 | fail |
| 79700 | 134.622 | 132.941 | 1.681 | fail |
| 50179 | 124.555 | 124.143 | 0.412 | fail |
| 86451 | 359.923 | 301.232 | 58.691 | fail |

**1/5 → PARTIAL.** No retuning of this start. Per the preregistered
rule, moved directly to H1-C.

### H1-C — `MORPH_TRUE_PIE`

Same construction, `(t0,u0,tE,rho)` = the frozen rank-1 H0 morphology
candidate (unchanged, no re-smoothing/re-lookup), `sign(u0)` set to
`sign(truth u0)` (documented, deliberate — H0's exact `u0` degeneracy
does not survive under H1, since `piE` breaks the `u0→-u0` symmetry;
for all 4 events tested `truth u0>0`, so no sign flip was actually
needed here, but the rule is general), `(piEN,piEE)` = truth. Run only
on the 4 events H1-A did not rescue. **4 new H1 TRF fits:**

| row | chi2 new | chi2 controlled5 | delta | pass |
|---|---|---|---|---|
| 87786 | 269.957 | 266.735 | 3.222 | fail |
| 79700 | 134.430 | 132.941 | 1.489 | fail |
| 50179 | 124.866 | 124.143 | 0.723 | fail |
| 86451 | 407.741 | 301.232 | 106.509 | fail |

**0/4 → both candidates together rescue only 1/5.** Per H1-E, this is
a joint FAIL: neither co-adaptive re-seeding from the current H0
final nor from morphology, even with truth `piE` handed to it for
free, reaches `old_final`'s basin on 4 of the 5 events.

### H1-E — what `old_final` is actually finding (diagnostic dump, no new fits beyond the 9 above)

| row | best `controlled4` | `old_final_reseed` | gap | note |
|---|---|---|---|---|
| 87786 | 269.344 (`H0_NESTED_piE_0`) | **266.735** | 2.6 | `old_final`'s own raw legacy vector already sits near `rho≈0`, `piE≈(-0.63,-1.95)` — a basin none of truth/H0-final/morphology-seeded fits reach; truth's own `piE≈(-3.9,-6.3)` is far larger in magnitude and different in balance |
| 79700 | 133.919 (`truth_half_piE`) | **132.941** | 1.0 | small but consistent gap; `MORPH_TRUE_PIE` gets closest of the new candidates (Δ=1.49) but not within 0.1 |
| 50179 | 124.563 (`H0_NESTED_piE_0`) | **124.143** | 0.4 | tightest gap of the 4; still not closed by either new candidate |
| 86451 | 301.493 (`truth`/`truth_half_piE`, tied) | 301.232 | 0.26 | **H0-final-seeded candidates do worse here than plain truth** (359.9, 407.7) — this event's current H0 final is itself a poor basin (`chi2_h0≈5209`, established earlier), so seeding H1 from it actively hurts; truth alone is already nearly as good as `old_final_reseed` |

**Pattern:** `old_final`'s advantage is small in absolute chi2 terms
for 3 of the 4 (Δ 0.26-1.0) but not reached by any tested
co-adaptive alternative; only 87786 shows a materially different,
apparently genuinely distinct low-rho/moderate-piE basin. This is
consistent with a **basin-reconstruction problem, not a starting-point
problem** — per the user's own framing, not pursued further as blind
start-guessing in this session.

### Totals and decision

- **New H1 TRF fits run this section: 9** (5 for H1-A + 4 for H1-C).
- `old_final` legacy dependency **cannot** yet be removed — neither
  candidate reaches a clean pass, and per the explicit stop rule no
  further algebraic-transform starts were invented on these same 5
  events.
- **Left open, explicitly, for a future session:** basin
  reconstruction for `old_final` — understanding *what* information
  (beyond truth `piE` and the current H0 final) actually determines
  that this specific low-rho / moderate-piE region is favored, rather
  than continuing to guess additional fixed starts.

### H1-F — local convergence audit (`controlled4` winner, tighter tolerances)

Before treating the 3 remaining events (87786, 79700, 50179, 86451 —
wait, 4 events survived H1-A/H1-C, see previous subsection) as
distinct-basin cases, checked whether `controlled4`'s own winning
solution was simply under-converged. For each, took `controlled4`'s
winning **final** vector (not its start) and ran exactly one H1 TRF
polish from that exact point, tolerances tightened to the historical
adaptive-polish philosophy (`xtol=ftol=1e-12`, `gtol=1e-8`, same
`max_nfev`/`x_scale`), all 6 parameters free, same bounds/profiling.
`run_h1_controlled4_polish.py`, `results/h1_controlled4_winner_polish_4events.csv`.

| row | winner | χ² before | χ² after | improvement | Δ vs controlled5 | pass |
|---|---|---|---|---|---|---|
| 87786 | `H0_NESTED_piE_0` | 269.344 | 269.338 | 0.006 | 2.603 | fail |
| **79700** | `truth_half_piE` | 133.919 | **131.689** | **2.230** | **−1.252** | **PASS** |
| 50179 | `H0_NESTED_piE_0` | 124.563 | 124.552 | 0.011 | 0.409 | fail |
| 86451 | `truth`/`truth_half_piE` | 301.493 | 301.493 | ~0 | 0.261 | fail |

**79700 was genuinely sub-converged** — tighter tolerances alone
close the gap and the polished point *beats* `old_final_reseed`
itself. **79700 is no longer `old_final`-essential.** The other 3
barely moved and proceeded to H1-F2.

### H1-F2 — convergence integrity audit (correction: `status=success` ≠ stationary)

An initial read of `optimizer_optimality` (0.137, 1.672, 0.001 for
87786/50179/86451) was mislabeled "well-converged" purely because
`status=success`. That is wrong on its face: none of these approach
`gtol=1e-8`, and 0.137/1.672 are also far above the 0.05 reference
threshold used elsewhere in this log for the historical adaptive
polish trigger. This needed resolving *before* calling any of them a
distinct basin. `run_h1_f2_convergence_audit.py`,
`results/h1_f2_convergence_audit.csv`. For each: recovered the full
scipy `OptimizeResult` (`status`, `message`, `nfev`, `njev`) via
`fit_lc.fit_rubin_roman` directly (`core.run_one_fit`'s trimmed
return dict does not carry these); computed an independent numerical
gradient (`J^T r`, same dimensionless scaling as `fisher_engine.py`)
at the returned point; computed a **bound-aware projected gradient**
(zeroing components at an active bound whose gradient points further
into the bound — the correct first-order stationarity condition,
not the raw gradient); and, for any event whose gradient was not
convincingly near zero, ran **one continuation** TRF from the exact
same final vector to test empirically whether it can still descend.

| row | scipy status | scipy optimality | raw grad (L∞) | bound-aware projected grad | continuation Δχ² | verdict |
|---|---|---|---|---|---|---|
| 87786 | 3 (`xtol`) | 0.137 | 0.0149 | 0.0149 (no active bound) | **0** (269.337642→269.337642, identical) | **genuinely stationary** |
| 50179 | 3 (`xtol`) | 1.672 | 0.0604 | 0.0604 | **0.0055** (still descending) | **NOT confirmed stationary** |
| 86451 | 3 (`xtol`) | 0.00098 | 1.037 (misleading — ρ at bound) | **0.00043** (ρ correctly excluded; `active_mask=[0,0,0,-1,0,0]`) | not run (already ≈0) | **genuinely stationary** (KKT-stationary at the ρ floor) |

All three terminated via `xtol` (small step), never `gtol` — exactly
why the original "well-converged" label was premature. The
continuation run is the decisive, model-independent check: 87786 and
86451 cannot improve further from their own final point (86451's
large raw gradient is fully explained by ρ sitting on its lower bound,
not by non-stationarity); 50179 still can, and its optimality stays
large (0.142) even after one continuation. **50179 is excluded from
further basin experiments — a real but small (Δ≈0.4) unresolved
convergence-quality question, not established as a distinct basin,
and deliberately not mixed with the basin-reconstruction problem.**
**Corrected survivors: 87786, 86451.**

### Block-ablation — necessity of `old_final`'s piE block

`run_h1_block_ablation.py`, `results/h1_block_ablation.csv`. Baseline
seed = `old_final` **raw** vector (proven to reach the target basin,
since `old_final_reseed` — its current-bounds re-optimization —
already does). Three leave-one-block-out hybrids per event
(backbone=`(t0,u0,tE)`, `rho`, piE=`(piEN,piEE)`), each replacing
exactly one block with `controlled4`'s polished value, everything
else kept from raw `old_final`. Full H1 TRF, all 6 parameters free,
standard optimizer options (not the tightened polish ones — this is a
normal fit). **6 new TRF fits** (2 events × 3 blocks).

| row | hybrid | χ² | Δ vs `old_final_reseed` | pass |
|---|---|---|---|---|
| 87786 | OLD_except_BACKBONE | 262.714 | −4.021 | PASS |
| 87786 | OLD_except_RHO | 263.797 | −2.938 | PASS |
| 87786 | **OLD_except_PIE** | 269.520 | **+2.784** | **FAIL** |
| 86451 | OLD_except_BACKBONE | 301.232 | ~0.000 | PASS |
| 86451 | OLD_except_RHO | 301.232 | 0.000 | PASS |
| 86451 | **OLD_except_PIE** | 301.493 | **+0.261** | **FAIL** |

Identical pattern on both events: replacing backbone or ρ alone from
raw `old_final` never breaks basin access (both even *beat*
`old_final_reseed` for 87786); replacing piE alone reproduces
`controlled4`'s own (wrong) basin almost exactly. **This demonstrates
piE necessity via leave-one-block-out perturbation around
`old_final`'s own raw vector** — a local statement, valid as such.

### H1-H — piE sufficiency test (mixed result; does not reinterpret the ablation above)

`run_h1_pie_sufficiency.py`, `results/h1_pie_sufficiency.csv`.
**Important distinction, not a contradiction of the block-ablation
result above:** the ablation tested necessity by perturbing *one
block away from `old_final`'s own full raw vector* — a small,
local move. H1-H tests **sufficiency** by transplanting *only*
`old_final`'s piE onto a vector otherwise entirely built from
`controlled4`'s own (distant) converged backbone/ρ — a much larger,
non-local move. These are different, complementary questions; the
ablation result stands unmodified.

Construction: `(t0,u0,tE,rho)` = `controlled4`-polished final,
`(piEN,piEE)` = raw `old_final`. Full H1 TRF, all 6 free, standard
options. **2 new TRF fits** (87786, 86451 only; 50179 excluded, per
H1-F2).

| row | χ² | Δ vs `old_final_reseed` | pass | final piE |
|---|---|---|---|---|
| 86451 | 301.232 | **≈0** | **PASS** | (0.416, 4.100) — essentially identical to `old_final_reseed`'s own final piE |
| 87786 | 269.418 | **2.683** | **FAIL** | (−7.09, −3.42) — drifted away from *both* the seeded `old_final` piE (−0.63,−1.95) and `old_final_reseed`'s own final piE |

**1/2 PASS.** `old_final`'s piE is **sufficient** for basin access on
86451 (the fit reproduces `old_final_reseed` to near machine
precision from a completely different backbone/ρ starting point) but
**not sufficient** on 87786, where the optimizer — seeded with
`controlled4`'s backbone/ρ context — pulls piE away during
optimization and re-converges near `controlled4`'s own basin instead.

**Global conclusion: `old_final` piE is necessary (block-ablation,
both events) but not universally sufficient (H1-H, 1/2 events) — for
at least 87786, basin access depends on the *joint* nonlinear
context (`old_final`'s specific backbone/ρ/piE combination together),
not on piE transplanted onto an arbitrary distant backbone.** No
further manual hybrid starts (pairwise swaps, rho+piE hand-built
combinations, truth or morphology hybrids, backbone-fixed grids) were
run or are planned — per the explicit stop rule, this line of
one-off parameter-swap experiments is closed. New H1 TRF fits this
round: **4 (H1-F polish) + 5 (H1-F2: 3 polish re-runs with full
diagnostics + 2 continuations, for 87786 and 50179) + 6 (block-ablation)
+ 2 (H1-H) = 17 new H1 TRF fits**, none of it a legacy-producer fit
(on top of H1-A/H1-C's 9, documented in the previous subsection).
**Next step (deliberately not started here):** reframe as a *cheap
family/template-based piE-basin-localization* problem, not another
round of individually-designed hybrid starts.
