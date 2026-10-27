#!/usr/bin/env python3

"""
Final H1 audit -- winner-bound / t0 / piE / nestedness / cost audit
of the controlled5 H1 start plan on extreme100.

STATUS: audit of an already-run, already-committed set of H1 fits
(Gate 2). This script runs no fits. Its purpose is to decide whether
`controlled5` can be frozen as CANDIDATE -- FROZEN FOR INDEPENDENT
VALIDATION, the same status Experiment B2 gave the H0 policy, or
whether it reveals a structural problem that must be fixed before
Gate 3.

PURPOSE
    1. Aggregate the two local `controlled5` H1 run roots (4 starts
       from `gate2_controlled4_extreme100`, 1 start --
       `old_final_reseed` -- from `gate2_oldfinal_extreme100`) into
       one canonical, integrity-checked per-fit table, mirroring
       Experiment 0's treatment of the H0 oracle.
    2. Determine the controlled5 winner per event (min chi2 over the
       5 starts) and audit its distance to every shared bound
       (u0, tE, rho, piEN, piEE) and to the event-specific,
       data-driven t0 domain.
    3. Check nestedness: compare the RAW controlled5-optimized chi2
       (no floor) against the exact embedded-H0 chi2
       (`gate2_h0_anchor_manifest_extreme100.csv`, which already
       encodes the validated t0-rescue-adjusted H0 reference -- see
       Experiment A1/B2) -- without hiding any violation behind
       `min(chi2_H0_exact, chi2_H1_raw)`.
    4. Report the H0(B2)+H1 combined expected per-event fit cost.
    5. Design closure step: define and verify a SHARED-DOMAIN t0
       rescue -- if the H1 controlled5 winner (base, t0_margin=0) has
       an active t0 bound, promote that event to t0_margin=0.25 and
       re-evaluate BOTH H0 (gate1_final18) and H1 (controlled5) in
       that shared, widened domain (never H1 alone at a different t0
       domain than H0 -- the core fitter's own consistency guard
       already enforces this, see Experiment A1 section 0.4). This
       complements, and never replaces, the existing validated H0-only
       t0 rescue (Experiment A1). The trigger is exactly
       `winner_H1_t0_active` from `h1_final_audit_winner_bounds.csv`
       -- no catalog_row is ever hardcoded in the trigger logic; row
       565924 is simply the one event that happens to satisfy it in
       extreme100, discovered, not assumed.

INPUTS
    Local, not repository-tracked (same class of dependency as the
    Gate-1 raw fit directories):
      ~/Downloads/hidden_parallax/production_validation/
          gate2_controlled4_extreme100/.../all_refits.csv
          gate2_oldfinal_extreme100/.../all_refits.csv
      ~/Downloads/hidden_parallax/hidden_parallax_refit_test/
          artifacts/extreme100/<catalog_row>/models/*/*/Event_*.h5
          (read-only, for data-driven t0 bounds -- same artifact tree
          Experiment B1c already used; only the H5 time arrays are
          read here, not full light curves)
    Repository-tracked:
      validation/bounds_convergence/data/
          gate2_h0_anchor_manifest_extreme100.csv
          (chi2_h0 = the exact embedded-H0 reference per event,
          ALREADY including the validated t0 rescue when triggered --
          confirmed in this session: for catalog_row 36103,
          chi2_h0=178.6896431771567 exactly matches Experiment A1's
          chi2_policy_final for that event)
      validation/bounds_convergence/results/
          b2_policy_comparison.csv (Experiment B2; only its
          mean_n_fits for the frozen H0 policy is read, for the cost
          section -- no H0 policy logic is touched)
          gate1_oracle_diagnostics.csv (Experiment 0; sources
          `gate1_final18_base_t0margin0` and
          `gate1_final18_rescue_t0margin0.25` give the H0 winner at
          t0_margin=0 and t0_margin=0.25 respectively, for every
          extreme100 event including 565924 -- this table already
          contains the widened-H0 result needed for the 565924 case
          study; no new H0 fits are run for it)
          h1_t0margin025_565924_diagnostic.csv (this script's own
          prior output: the one targeted, already-run H1
          widened-domain diagnostic for catalog_row=565924)

OUTPUTS (validation/bounds_convergence/results/)
    gate2_controlled5_diagnostics.csv
        one row per (catalog_row, start label): the canonical,
        integrity-checked per-fit H1 table (100 events x 5 starts =
        500 rows). Columns directly from the fitter's own record
        (t0,u0,tE,rho,piEN,piEE,chi2,status,optimizer_*) plus
        `source` (which local root) and `label` (start name).
    h1_final_audit_winner_bounds.csv
        one row per catalog_row: the controlled5 winner's full
        parameter vector, active_mask-derived bound flags, and
        distance-to-bound (absolute and relative) for every shared
        parameter.
    h1_final_audit_nestedness.csv
        one row per catalog_row: chi2_h1_raw_optimized (winner,
        unfloored), chi2_h0_exact, the raw difference, and whether it
        violates chi2_H1<=chi2_H0.
    h1_t0margin025_565924_diagnostic.csv
        NOT produced by this script automatically -- the raw
        `all_refits.csv` from one manually-invoked, targeted diagnostic
        run (existing unmodified fitter, catalog_row=565924 only, H1
        `controlled5`, `t0_margin=0.25`; exact command recorded in
        ADAPTIVE_H0_ANALYSIS.md section 6.2), copied here so the
        t0-truncation finding for this event does not depend on the
        local scratch run directory persisting.
    h1_t0_rescue_565924_case_study.csv
        one row: chi2_H0/H1 base and widened, DeltaChi2_LRT base and
        widened, and the widened-domain nestedness check, for the one
        event that triggers the new shared-domain t0 rescue in
        extreme100 -- built entirely from already-existing tables, no
        new fits run by this function.
    h1_t0_rescue_extreme100_trigger_check.csv
        one row: how many extreme100 H1 winners have an active t0
        bound (the new rescue's trigger count), how many H0 winners
        have the existing validated t0 rescue triggered, and how many
        events trigger both (checked, not assumed).

WHAT QUESTION THIS ANSWERS
    Can `controlled5` be frozen as-is for Gate 3, or does this audit
    reveal truncated bounds, a t0-domain problem, or real (not just
    numerical) nestedness violations that must be fixed first?

ORACLE/REFERENCE USAGE
    `chi2_h0` (the embedded-H0 reference) is read only to (a) build
    the `H0_NESTED_piE_0` start (already done, upstream, when these
    fits were run) and (b) evaluate nestedness here, offline. This
    script proposes no new H1 search and does not touch H0.

COMBINED POLICY PRECEDENCE -- frozen, exact control flow for Gate 3
    (mirrored in ADAPTIVE_H0_ANALYSIS.md section 6.10; the two copies
    must be kept in sync if either changes). extreme100 shows the
    H0-only rescue (row 36103) and this H1-triggered rescue (row
    565924) as disjoint, but Gate 3 could produce an event where they
    interact, so precedence is fixed unambiguously here:
      1. Run H0 per the frozen Experiment B2 policy (base k=6;
         escalate to complete gate1_final18 only if B2's own
         abs_log_ratio_tE trigger fires). The validated H0-only t0
         rescue (Experiment A1 section 0.4) can only be evaluated once
         gate1_final18's full winner exists -- i.e. only for events
         where B2 already escalated; unchanged, pre-existing behavior.
      2. shared_t0_margin := 0.0. If the validated H0-only t0 rescue
         triggers (gate1_final18 winner's own t0 active_mask
         component is active), set shared_t0_margin := 0.25.
      3. Run H1 controlled5 using shared_t0_margin from step 2 -- if
         H0's own rescue already widened the domain, H1's FIRST run
         already uses the widened domain, not the base one.
      4. Only if shared_t0_margin is still 0.0 after step 3, and the
         H1 controlled5 winner's own t0 active_mask component is
         active: set shared_t0_margin := 0.25; re-run H0 robust
         (gate1_final18, full 18) and H1 controlled5, both at
         t0_margin=0.25; use these two widened results for the LRT.
         (Exactly the row-565924 case below.)
      5. If, after shared_t0_margin is already 0.25 (from step 2 or
         step 4), the H1 winner's t0 is STILL active: do NOT widen
         again and do NOT invent a new rule now -- record it as a
         policy anomaly/failure for Gate 3 to report explicitly.
    INVARIANT (enforced by construction above): the H0 and H1 chi2
    entering an event's final LRT must always come from the identical
    shared_t0_margin. Every condition reads only optimizer_active_mask
    (H0's or H1's own winner) -- catalog_row never appears in any
    runtime condition of this combined policy, here or in the
    markdown doc. This is a documentation-only control-flow
    specification: implementing it as automatic logic in
    run_bounds_audit_refit_core.py is production-adjacent work
    explicitly deferred, not done by this script.

NO FITS, NO pyLIMA, NO CHE, NO core/production/SLURM changes, NO new
H1 start search: this script only reads already-committed/already-run
artifacts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


RESULTS_DIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results"
)
DATA_DIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/data"
)

ROOT_CONTROLLED4 = Path(
    "~/Downloads/hidden_parallax/production_validation/"
    "gate2_controlled4_extreme100"
).expanduser()
ROOT_OLDFINAL = Path(
    "~/Downloads/hidden_parallax/production_validation/"
    "gate2_oldfinal_extreme100"
).expanduser()

SUBPATH = Path(
    "refits/bounds_convergence/production_candidate_h1only/extreme100"
)

ARTIFACTS_ROOT = Path(
    "~/Downloads/hidden_parallax/hidden_parallax_refit_test/"
    "artifacts/extreme100"
).expanduser()

H0_ANCHOR_MANIFEST = (
    DATA_DIR / "gate2_h0_anchor_manifest_extreme100.csv"
)
B2_COMPARISON = RESULTS_DIR / "b2_policy_comparison.csv"
GATE1_ORACLE_DIAGNOSTICS = (
    RESULTS_DIR / "gate1_oracle_diagnostics.csv"
)

OUT_DIAGNOSTICS = RESULTS_DIR / "gate2_controlled5_diagnostics.csv"
OUT_WINNER_BOUNDS = RESULTS_DIR / "h1_final_audit_winner_bounds.csv"
OUT_NESTEDNESS = RESULTS_DIR / "h1_final_audit_nestedness.csv"

H1_T0_DIAGNOSTIC_565924 = (
    RESULTS_DIR / "h1_t0margin025_565924_diagnostic.csv"
)
OUT_565924_CASE_STUDY = (
    RESULTS_DIR / "h1_t0_rescue_565924_case_study.csv"
)
OUT_TRIGGER_CHECK = (
    RESULTS_DIR / "h1_t0_rescue_extreme100_trigger_check.csv"
)

CONTROLLED4_STARTS = [
    "truth",
    "H0_NESTED_piE_0",
    "truth_half_piE",
    "truth_mirror_u0_piEN",
]
OLDFINAL_STARTS = ["old_final_reseed"]
ALL_STARTS = CONTROLLED4_STARTS + OLDFINAL_STARTS

# Shared production_candidate bounds (checkpoint, VALIDATION_STATUS
# section 3). H1 param order matches optimizer_active_mask order:
# t0, u0, tE, rho, piEN, piEE.
BOUNDS = {
    "u0": (-10.0, 10.0),
    "tE": (0.1, 500000.0),
    "rho": (1.0e-7, 10.0),
    "piEN": (-40.0, 40.0),
    "piEE": (-40.0, 40.0),
}

PARAM_ORDER = ["t0", "u0", "tE", "rho", "piEN", "piEE"]

# tE and rho span many orders of magnitude; bound-distance for them
# is evaluated in log10 space (see winner_bound_audit).
LOG_SCALE_PARAMS = {"tE", "rho"}

TOL = 0.1

# "close to bound" for continuous params without an exact active
# flag: within this fraction of the bound's full range.
NEAR_BOUND_FRAC = 0.01


def _parse_active_mask(raw) -> tuple[int, ...] | None:
    if not isinstance(raw, str):
        return None
    try:
        val = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    return tuple(int(x) for x in val)


def load_all_refits(
    root: Path, catalog_row: int, source: str
) -> pd.DataFrame:

    p = root / SUBPATH / str(catalog_row) / "all_refits.csv"

    if not p.exists():
        raise RuntimeError(f"Missing {p}")

    df = pd.read_csv(p)
    df["source"] = source

    return df


def build_diagnostics_table() -> pd.DataFrame:

    anchor = pd.read_csv(H0_ANCHOR_MANIFEST)
    rows = sorted(anchor["catalog_row"].unique().tolist())

    if len(rows) != 100:
        raise RuntimeError(
            f"Expected 100 catalog_row in {H0_ANCHOR_MANIFEST}, "
            f"found {len(rows)}."
        )

    frames = []

    for row in rows:
        frames.append(
            load_all_refits(ROOT_CONTROLLED4, row, "controlled4")
        )
        frames.append(
            load_all_refits(ROOT_OLDFINAL, row, "oldfinal")
        )

    df = pd.concat(frames, ignore_index=True)

    keep = [
        "source",
        "catalog_row",
        "hypothesis",
        "label",
        "status",
        "chi2",
        "t0",
        "u0",
        "tE",
        "rho",
        "piEN",
        "piEE",
        "optimizer_success",
        "optimizer_optimality",
        "optimizer_active_mask",
    ]

    return df[keep]


def audit_diagnostics_table(df: pd.DataFrame) -> None:

    print("=" * 70)
    print("H1 DIAGNOSTICS TABLE -- INTEGRITY AUDIT")
    print("=" * 70)

    errors = []

    n_rows = len(df)
    n_events = df["catalog_row"].nunique()

    print(f"n_rows={n_rows} (expected 500 = 100 events x 5 starts)")
    print(f"n_distinct_catalog_row={n_events} (expected 100)")

    if n_rows != 500:
        errors.append(f"expected 500 rows, found {n_rows}")
    if n_events != 100:
        errors.append(f"expected 100 events, found {n_events}")

    counts = df.groupby("catalog_row")["label"].apply(
        lambda s: sorted(s.tolist())
    )
    bad_events = [
        row for row, labels in counts.items() if labels != sorted(ALL_STARTS)
    ]
    if bad_events:
        errors.append(
            f"{len(bad_events)} events do not have exactly the 5 "
            f"expected start labels: {bad_events[:10]}"
        )
    else:
        print("every event has exactly the 5 expected start labels -- OK")

    hyp_values = sorted(df["hypothesis"].unique().tolist())
    print(f"hypothesis values: {hyp_values}")
    if hyp_values != ["H1"]:
        errors.append(f"hypothesis values != ['H1']: {hyp_values}")

    status_counts = df["status"].value_counts().to_dict()
    print(f"status counts: {status_counts}")
    n_non_success = int((df["status"] != "success").sum())
    if n_non_success:
        errors.append(f"{n_non_success} rows have status != 'success'")

    for col in ["chi2", "t0", "u0", "tE", "rho", "piEN", "piEE"]:
        n_bad = int((~np.isfinite(df[col])).sum())
        if n_bad:
            errors.append(f"{n_bad} non-finite values in {col}")

    n_dup = int(
        df.duplicated(subset=["catalog_row", "label"]).sum()
    )
    print(f"duplicate (catalog_row,label) rows: {n_dup}")
    if n_dup:
        errors.append(f"{n_dup} duplicate (catalog_row,label) rows")

    if errors:
        raise RuntimeError(
            "H1 diagnostics table integrity violation(s): "
            + "; ".join(errors)
        )

    print()
    print("H1 diagnostics table audit: PASSED")
    print()


def compute_winners(df: pd.DataFrame) -> pd.DataFrame:

    idx = df.groupby("catalog_row")["chi2"].idxmin()
    winners = df.loc[idx].reset_index(drop=True)
    winners = winners.rename(
        columns={"chi2": "chi2_h1_raw_optimized", "label": "winner_label"}
    )

    return winners


def read_h5_time_bounds(catalog_row: int) -> tuple[float, float]:
    """
    Data-driven t0 domain input, replicating (read-only) the core
    fitter's own pooled-times min/max (see
    run_bounds_audit_refit_core.py:2104-2140 and
    analyze_b1c_simulation_difficulty.py's identical use of the same
    artifact tree).
    """

    event_dir = ARTIFACTS_ROOT / str(catalog_row)
    matches = sorted(event_dir.glob("models/*/*/Event_*.h5"))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly 1 H5 for catalog_row={catalog_row}, "
            f"found {len(matches)}."
        )

    all_times = []

    with h5py.File(matches[0], "r") as f:
        for band in ["W149", "u", "g", "r", "i", "z", "y"]:
            if band not in f:
                continue
            g = f[band]
            t = np.asarray(g["time"], dtype=float)
            m = np.asarray(g["mag"], dtype=float)
            e = np.asarray(g["err_mag"], dtype=float)
            if "photometry_keep" in g:
                keep = np.asarray(g["photometry_keep"], dtype=bool)
            else:
                keep = np.ones(len(t), dtype=bool)
            good = (
                keep
                & np.isfinite(t)
                & np.isfinite(m)
                & np.isfinite(e)
                & (e > 0)
            )
            all_times.append(t[good])

    all_times = np.concatenate(all_times)

    return float(all_times.min()), float(all_times.max())


def winner_bound_audit(
    winners: pd.DataFrame, anchor: pd.DataFrame
) -> pd.DataFrame:

    anchor_idx = anchor.set_index("catalog_row")

    records = []

    for _, w in winners.iterrows():

        row = int(w["catalog_row"])
        mask = _parse_active_mask(w["optimizer_active_mask"])

        if mask is None or len(mask) != 6:
            raise RuntimeError(
                f"catalog_row={row}: expected a 6-element H1 "
                f"active_mask, got {mask}."
            )

        active = dict(zip(PARAM_ORDER, mask))

        t0_margin_factor = float(
            anchor_idx.loc[row, "t0_margin_factor"]
        )
        t_min_data, t_max_data = read_h5_time_bounds(row)
        t_obs = t_max_data - t_min_data
        margin = t0_margin_factor * t_obs
        t0_lo = t_min_data - margin
        t0_hi = t_max_data + margin

        rec = {
            "catalog_row": row,
            "winner_label": w["winner_label"],
            "chi2_h1_raw_optimized": float(
                w["chi2_h1_raw_optimized"]
            ),
            "t0": float(w["t0"]),
            "u0": float(w["u0"]),
            "tE": float(w["tE"]),
            "rho": float(w["rho"]),
            "piEN": float(w["piEN"]),
            "piEE": float(w["piEE"]),
            "t0_lo": t0_lo,
            "t0_hi": t0_hi,
            "t0_active": int(active["t0"] != 0),
            "u0_active": int(active["u0"] != 0),
            "tE_active": int(active["tE"] != 0),
            "rho_active": int(active["rho"] != 0),
            "piEN_active": int(active["piEN"] != 0),
            "piEE_active": int(active["piEE"] != 0),
        }

        # distance to bound, absolute and relative-to-range, for the
        # fixed shared bounds. tE and rho span many orders of
        # magnitude (0.1..500000 and 1e-7..10) -- a LINEAR
        # fraction-of-range is not physically meaningful for them
        # (nearly every plausible value would register as "close" to
        # the lower bound purely because the range is huge in linear
        # units); those two are evaluated in log10 space instead. u0,
        # piEN, piEE have bounds symmetric about a physically
        # meaningful zero, where linear distance is the natural scale.
        for p, (lo, hi) in BOUNDS.items():
            v = rec[p]

            if p in LOG_SCALE_PARAMS:
                lo_s, hi_s, v_s = (
                    np.log10(lo),
                    np.log10(hi),
                    np.log10(v),
                )
            else:
                lo_s, hi_s, v_s = lo, hi, v

            rng = hi_s - lo_s
            dist_lo = v - lo
            dist_hi = hi - v
            dist_lo_s = v_s - lo_s
            dist_hi_s = hi_s - v_s

            rec[f"{p}_dist_to_lower"] = dist_lo
            rec[f"{p}_dist_to_upper"] = dist_hi
            rec[f"{p}_frac_of_range_to_nearest_bound"] = (
                min(dist_lo_s, dist_hi_s) / rng
            )
            rec[f"{p}_near_bound"] = bool(
                (dist_lo_s / rng <= NEAR_BOUND_FRAC)
                or (dist_hi_s / rng <= NEAR_BOUND_FRAC)
            )

        # t0: event-specific domain
        rng_t0 = t0_hi - t0_lo
        dist_lo_t0 = rec["t0"] - t0_lo
        dist_hi_t0 = t0_hi - rec["t0"]
        rec["t0_dist_to_lower"] = dist_lo_t0
        rec["t0_dist_to_upper"] = dist_hi_t0
        rec["t0_frac_of_range_to_nearest_bound"] = (
            min(dist_lo_t0, dist_hi_t0) / rng_t0
        )
        rec["t0_near_bound"] = bool(
            (dist_lo_t0 / rng_t0 <= NEAR_BOUND_FRAC)
            or (dist_hi_t0 / rng_t0 <= NEAR_BOUND_FRAC)
        )

        records.append(rec)

    return pd.DataFrame.from_records(records)


def nestedness_audit(
    winners: pd.DataFrame, anchor: pd.DataFrame
) -> pd.DataFrame:

    m = winners[["catalog_row", "chi2_h1_raw_optimized"]].merge(
        anchor[["catalog_row", "chi2_h0"]], on="catalog_row"
    )
    m = m.rename(columns={"chi2_h0": "chi2_h0_exact"})
    m["violation"] = (
        m["chi2_h1_raw_optimized"] - m["chi2_h0_exact"]
    )
    m["violates_h1_le_h0"] = m["violation"] > 0

    return m


def _h0_winner_chi2(source: str, catalog_row: int) -> float:

    g1 = pd.read_csv(GATE1_ORACLE_DIAGNOSTICS)
    sub = g1[
        (g1["source"] == source) & (g1["catalog_row"] == catalog_row)
    ]

    if sub.empty:
        raise RuntimeError(
            f"No rows for source={source!r}, "
            f"catalog_row={catalog_row} in "
            f"{GATE1_ORACLE_DIAGNOSTICS}."
        )

    return float(sub["chi2"].min())


def shared_domain_t0_rescue_case_study(
    winners: pd.DataFrame, catalog_row: int
) -> pd.DataFrame:
    """
    Step 1 of the shared-domain t0 rescue closure: for the one event
    that triggers it in extreme100, assemble base vs. widened H0/H1
    results and the resulting LRT change, entirely from already-run
    fits (Experiment 0's global t0_margin=0.25 diagnostic run for H0,
    and this session's single targeted H1 diagnostic -- see module
    docstring). No new fits are run here.
    """

    w = winners[winners["catalog_row"] == catalog_row].iloc[0]

    chi2_h1_base = float(w["chi2_h1_raw_optimized"])

    wide = pd.read_csv(H1_T0_DIAGNOSTIC_565924)
    chi2_h1_wide = float(wide["chi2"].min())
    wide_winner = wide.loc[wide["chi2"].idxmin()]

    chi2_h0_base = _h0_winner_chi2(
        "gate1_final18_base_t0margin0", catalog_row
    )
    chi2_h0_wide = _h0_winner_chi2(
        "gate1_final18_rescue_t0margin0.25", catalog_row
    )

    lrt_base = chi2_h0_base - chi2_h1_base
    lrt_wide = chi2_h0_wide - chi2_h1_wide

    rec = {
        "catalog_row": catalog_row,
        "chi2_H0_base": chi2_h0_base,
        "chi2_H1_base": chi2_h1_base,
        "DeltaChi2_LRT_base": lrt_base,
        "chi2_H0_wide": chi2_h0_wide,
        "chi2_H1_wide": chi2_h1_wide,
        "DeltaChi2_LRT_wide": lrt_wide,
        "change_in_H0": chi2_h0_wide - chi2_h0_base,
        "change_in_H1": chi2_h1_wide - chi2_h1_base,
        "change_in_LRT": lrt_wide - lrt_base,
        "nestedness_raw_wide": chi2_h1_wide - chi2_h0_wide,
        "nestedness_violated_wide": (
            chi2_h1_wide > chi2_h0_wide
        ),
        "H1_wide_winner_label": wide_winner["label"],
        "H1_wide_winner_active_mask": (
            wide_winner["optimizer_active_mask"]
        ),
        "H1_base_winner_active_mask": w["optimizer_active_mask"],
    }

    return pd.DataFrame.from_records([rec])


def trigger_check(
    winner_bounds: pd.DataFrame, anchor: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 2: verify the new rescue's trigger (winner_H1_t0_active,
    read directly from the already-computed H1 winner-bound audit's
    own `t0_active` column -- no catalog_row is hardcoded, and no
    active_mask is re-parsed here) against extreme100, and check its
    overlap with the already-validated H0-only t0 rescue trigger
    (`rescue_triggered` in the H0 anchor manifest). No new fits.
    """

    h1_t0_active_rows = set(
        winner_bounds.loc[
            winner_bounds["t0_active"] == 1, "catalog_row"
        ].tolist()
    )

    h0_rescue_rows = set(
        anchor.loc[
            anchor["rescue_triggered"], "catalog_row"
        ].tolist()
    )

    both = h1_t0_active_rows & h0_rescue_rows

    rec = {
        "n_events": len(anchor),
        "n_h1_winner_t0_active": len(h1_t0_active_rows),
        "h1_winner_t0_active_rows": sorted(h1_t0_active_rows),
        "n_h0_rescue_triggered": len(h0_rescue_rows),
        "h0_rescue_triggered_rows": sorted(h0_rescue_rows),
        "n_triggering_both": len(both),
        "rows_triggering_both": sorted(both),
    }

    return pd.DataFrame.from_records([rec])


def main() -> None:

    print("Building H1 controlled5 diagnostics table...")
    df = build_diagnostics_table()
    audit_diagnostics_table(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIAGNOSTICS, index=False)
    print(f"wrote {OUT_DIAGNOSTICS} ({len(df)} rows)")
    print()

    winners = compute_winners(df)
    anchor = pd.read_csv(H0_ANCHOR_MANIFEST)

    # --- 1/2/3: winner-bound / t0 / piE audit ---
    print("Reading H5 time bounds for t0 domain (100 events)...")
    wb = winner_bound_audit(winners, anchor)
    wb.to_csv(OUT_WINNER_BOUNDS, index=False)
    print(f"wrote {OUT_WINNER_BOUNDS} ({len(wb)} rows)")
    print()

    print("=" * 70)
    print("WINNER-BOUND AUDIT SUMMARY")
    print("=" * 70)
    for p in ["t0", "u0", "tE", "rho", "piEN", "piEE"]:
        n_active = int(wb[f"{p}_active"].sum())
        n_near = int(wb[f"{p}_near_bound"].sum())
        print(
            f"  {p}: n_active_mask={n_active}/100, "
            f"n_near_bound(<= {NEAR_BOUND_FRAC:.0%} of range)="
            f"{n_near}/100"
        )
    print()

    # --- 4: nestedness audit ---
    nest = nestedness_audit(winners, anchor)
    nest.to_csv(OUT_NESTEDNESS, index=False)
    print(f"wrote {OUT_NESTEDNESS} ({len(nest)} rows)")
    print()

    print("=" * 70)
    print("NESTEDNESS AUDIT SUMMARY")
    print("=" * 70)
    n_violate = int(nest["violates_h1_le_h0"].sum())
    print(f"N(chi2_H1_raw > chi2_H0_exact) = {n_violate}/100")
    if n_violate:
        v = nest[nest["violates_h1_le_h0"]].sort_values(
            "violation", ascending=False
        )
        print(v[["catalog_row", "chi2_h1_raw_optimized",
                  "chi2_h0_exact", "violation"]].to_string(index=False))
    print(f"max violation = {nest['violation'].max():.6g}")
    print(f"median violation (all events, can be negative) = "
          f"{nest['violation'].median():.6g}")
    print()

    # --- Step 2 (checked before Step 1 so we know which row(s) to
    # report in the case study, without hardcoding one) ---
    print("=" * 70)
    print(
        "STEP 2 -- SHARED-DOMAIN t0 RESCUE TRIGGER CHECK (extreme100)"
    )
    print("=" * 70)
    tc = trigger_check(wb, anchor)
    tc.to_csv(OUT_TRIGGER_CHECK, index=False)
    print(f"wrote {OUT_TRIGGER_CHECK}")
    tcr = tc.iloc[0]
    print(
        f"  n_h1_winner_t0_active = {tcr['n_h1_winner_t0_active']}/"
        f"{tcr['n_events']} -> rows {tcr['h1_winner_t0_active_rows']}"
    )
    print(
        f"  n_h0_rescue_triggered (existing, validated) = "
        f"{tcr['n_h0_rescue_triggered']}/{tcr['n_events']} -> rows "
        f"{tcr['h0_rescue_triggered_rows']}"
    )
    print(
        f"  n_triggering_both = {tcr['n_triggering_both']} -> rows "
        f"{tcr['rows_triggering_both']}"
    )
    print()

    # --- Step 1: case study for whichever row(s) Step 2 found ---
    trigger_rows = list(tcr["h1_winner_t0_active_rows"])

    if trigger_rows:
        print("=" * 70)
        print("STEP 1 -- SHARED-DOMAIN t0 RESCUE CASE STUDY")
        print("=" * 70)

        case_frames = []
        for row in trigger_rows:
            if row == 565924:
                case_frames.append(
                    shared_domain_t0_rescue_case_study(winners, row)
                )
            else:
                print(
                    f"  catalog_row={row} also triggers the new "
                    "rescue but has no pre-computed widened-H1 "
                    "diagnostic in this session -- not included in "
                    "the case-study table below (would require "
                    "running the same 5-fit diagnostic for this row "
                    "specifically before it can be reported)."
                )

        if case_frames:
            case_df = pd.concat(case_frames, ignore_index=True)
            case_df.to_csv(OUT_565924_CASE_STUDY, index=False)
            print(f"wrote {OUT_565924_CASE_STUDY}")
            print(case_df.to_string(index=False))
        print()
    else:
        print("No event triggers the new shared-domain t0 rescue -- "
              "no case study to report.")
        print()

    # --- 5: cost ---
    print("=" * 70)
    print("COST SUMMARY")
    print("=" * 70)
    print(f"H1 starts/event = {len(ALL_STARTS)} (fixed, controlled5)")

    try:
        b2 = pd.read_csv(B2_COMPARISON)
        b2_best = b2[
            b2["rule"].str.startswith("escalate if abs_log_ratio_tE")
            & ~b2["rule"].str.contains(" OR ")
            & (b2["k"].astype(str) == "6")
        ]
        if len(b2_best) == 1:
            mean_h0 = float(b2_best.iloc[0]["mean_n_fits"])
            baseline_combined = mean_h0 + len(ALL_STARTS)
            print(f"Frozen B2 H0 policy mean N_fits/event = {mean_h0:.2f}")
            print(
                "Combined expected H0(B2)+H1(controlled5) "
                f"fits/event (before shared-domain rescue) = "
                f"{mean_h0:.2f} + {len(ALL_STARTS)} = "
                f"{baseline_combined:.2f}"
            )

            n_trigger = int(tcr["n_h1_winner_t0_active"])
            n_events = int(tcr["n_events"])
            frac_trigger = n_trigger / n_events

            # Conservative, non-optimized cost of the shared-domain
            # rescue when it fires: complete gate1_final18 (18 H0
            # fits total, widened) + rerun controlled5 (5 H1 fits,
            # widened). No credit is taken for any base-domain fits
            # that might be reusable at the wider domain -- this is
            # an upper-bound estimate, not a tuned one, per
            # instruction not to optimize the trigger or its cost.
            n_h0_wide_fits = 18  # gate1_final18 strategy count
            n_h1_wide_fits = len(ALL_STARTS)
            rescue_cost_per_trigger = n_h0_wide_fits + n_h1_wide_fits
            overhead_per_event = frac_trigger * rescue_cost_per_trigger
            combined_with_rescue = baseline_combined + overhead_per_event

            print()
            print(
                "Shared-domain t0 rescue overhead "
                f"(triggers on {n_trigger}/{n_events} = "
                f"{frac_trigger:.1%} of extreme100 events): "
                f"conservative cost when triggered = "
                f"{n_h0_wide_fits} (H0, full gate1_final18, widened) "
                f"+ {n_h1_wide_fits} (H1, controlled5, widened) = "
                f"{rescue_cost_per_trigger} extra fits "
                "(no reuse credited)."
            )
            print(
                f"Expected overhead = {frac_trigger:.4f} x "
                f"{rescue_cost_per_trigger} = "
                f"{overhead_per_event:.3f} fits/event."
            )
            print(
                "Combined expected H0(B2)+H1(controlled5)+shared-"
                f"domain-t0-rescue fits/event = {baseline_combined:.2f}"
                f" + {overhead_per_event:.3f} = "
                f"{combined_with_rescue:.3f}"
            )
        else:
            print(
                "Could not uniquely identify the frozen B2 policy row "
                f"in {B2_COMPARISON} ({len(b2_best)} matches) -- "
                "skipping combined cost estimate."
            )
    except FileNotFoundError:
        print(f"{B2_COMPARISON} not found -- skipping combined cost.")


if __name__ == "__main__":
    main()
