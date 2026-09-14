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
