#!/usr/bin/env python3
"""
Builds chi2_H0_ref2_independent / chi2_H1_ref2_independent / D_ref2_
independent (explicitly excluding H0_2/H1_projected), re-evaluates the
original single-start H0 problem and Demonstration 2A against it
non-circularly, and produces the delta_H0/delta_H1 decomposition for
remaining discrepant events. See REFERENCE_V2_INDEPENDENT.md for the
full write-up; this script only computes numbers.
"""
import sys
import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.dirname(HERE)
BRT = os.path.join(PP, "basin_risk_trigger")
H1P = os.path.join(PP, "h1_projected_h0_start")
BA = os.path.join(PP, "..", "bounds_audit")
BC = os.path.join(PP, "..", "bounds_convergence")
sys.path.insert(0, BA)
sys.path.insert(0, BC)
sys.path.insert(0, PP)

os.environ["HIDDEN_PARALLAX_TRF_COORDS"] = "physical"
os.environ["HIDDEN_PARALLAX_T0_MARGIN_FACTOR"] = "0"
sys.argv = ["run_bounds_audit_refit_core.py", "--catalog-row", "71181",
            "--bounds-profile", "production_candidate", "--fit-scope", "h0", "--dry-run"]

import run_bounds_audit_refit_core as core  # noqa: E402
import fit_lc  # noqa: E402
from load_new_event import load_new_case  # noqa: E402
from numerical_safeguards import run_nested_h1_rescue, EPSILON_NUMERIC  # noqa: E402

print(f"epsilon_numeric = {EPSILON_NUMERIC} (source: numerical_safeguards.EPSILON_NUMERIC), reused unchanged.")

dev_table = pd.read_csv(os.path.join(BRT, "basin_risk_development_table.csv"))
h1proj = pd.read_csv(os.path.join(H1P, "h1_projected_h0_results.csv"))
ref_events = {e["catalog_row"]: e for e in
              json.load(open(os.path.join(PP, "results", "robust_reference_controlled4_h1basin_100events.json")))["events"]}
fp_events = {e["catalog_row"]: e for e in
             json.load(open(os.path.join(PP, "results", "final_policy_h1basinvalidation_100events.json")))["events"]}
manifest = pd.read_csv(os.path.join(PP, "results", "h1_basin_validation_sample_frozen.csv")).set_index("catalog_row")

physical_panel = pd.read_csv(os.path.join(HERE, "physical_panel_results.csv"))
log_te_rho_panel = pd.read_csv(os.path.join(HERE, "log_te_rho_panel_results.csv"))
panel = pd.concat([physical_panel, log_te_rho_panel], ignore_index=True)
panel_ok = panel[panel["failed"] != True]  # noqa: E712

BRIDGE_JSON_DIR = os.path.join(PP, "results", "pipeline_stage_json")


def dangerous_label(D, D_ref):
    return bool((D_ref < 9) and (D >= 4) and (D > D_ref))


def missed_label(D, D_ref):
    return bool((D_ref >= 4) and (D < 9) and (D < D_ref))


h1proj_dev = h1proj[~h1proj["excluded_from_development"]].copy()
h1proj_dev["reference_beaten_h0"] = h1proj_dev["reference_beaten_h0"].map(
    {"True": True, "False": False, True: True, False: False, np.nan: np.nan})
panel_event_ids = set(h1proj_dev.loc[h1proj_dev["reference_beaten_h0"] == True, "event_id"])  # noqa: E712
print(f"verification panel applies to {len(panel_event_ids)} events "
      f"(expected 19, verified from stored Demonstration 2A results).")

rows = []
for _, dr in dev_table.iterrows():
    row = int(dr["event_id"])
    if bool(dr["excluded_from_development"]):
        continue  # 7 catastrophic events stay outside the main analysis throughout

    D_s = float(dr["D_s"])
    D_r_old = float(dr["D_r"])

    ref = ref_events[row]
    chi2_H0_old_reference = float(ref["h0_final_chi2"])
    chi2_H1_old_reference = float(ref["h1_final_chi2"])

    h1p_row = h1proj_dev[h1proj_dev["event_id"] == row].iloc[0]
    chi2_H0_H0_2 = h1p_row["chi2_H0_2"]
    chi2_H0_H0_2 = float(chi2_H0_H0_2) if pd.notna(chi2_H0_H0_2) else None
    chi2_H0_final_2A = float(h1p_row["chi2_H0_final"])
    chi2_H1_final_2A = float(h1p_row["chi2_H1_final"])
    D_s2 = h1p_row["D_s2"]
    D_s2 = float(D_s2) if pd.notna(D_s2) else None

    fp = fp_events[row]
    chi2_H1_candidate = float(fp["chi2_h1"])

    # ---- chi2_H0_ref2_independent: EXCLUDES H0_2/H1_projected ----
    candidates = [chi2_H0_old_reference]
    strategies_used = ["old_controlled4_reference"]
    winner_params_by_strategy = {"old_controlled4_reference":
                                  {"t0": ref["t0"], "u0": ref["u0"], "tE": ref["tE"], "rho": ref["rho"]}
                                  if "t0" in ref else None}

    bridge_data = None
    if row in panel_event_ids:
        bridge_json = os.path.join(BRIDGE_JSON_DIR, f"{row}_bridge.json")
        with open(bridge_json) as f:
            bridge_data = json.load(f)
        bv = bridge_data["bridge_vector"]
        candidates.append(float(bridge_data["bridge_chi2"]))
        strategies_used.append("old_H0_bridge")
        winner_params_by_strategy["old_H0_bridge"] = {"t0": bv["t0"], "u0": bv["u0"], "tE": bv["tE"], "rho": bv["rho"]}

        prow = panel_ok[panel_ok["event_id"] == row]
        for _, prec in prow.iterrows():
            candidates.append(float(prec["chi2_H0"]))
            strategies_used.append(prec["strategy_name"])
            winner_params_by_strategy[prec["strategy_name"]] = {
                "t0": float(prec["t0"]), "u0": float(prec["u0"]), "tE": float(prec["tE"]), "rho": float(prec["rho"])}

    chi2_H0_ref2_independent = min(candidates)
    best_independent_strategy = strategies_used[int(np.argmin(candidates))]

    h0_2_beats_old_reference = row in panel_event_ids
    h0_2_beats_independent_reference = (chi2_H0_H0_2 is not None) and \
        (chi2_H0_H0_2 < chi2_H0_ref2_independent - EPSILON_NUMERIC)
    independent_reference_beats_h0_2 = (chi2_H0_H0_2 is not None) and \
        (chi2_H0_ref2_independent < chi2_H0_H0_2 - EPSILON_NUMERIC)
    h0_2_basin_independently_reproduced = (chi2_H0_H0_2 is not None) and (
        chi2_H0_ref2_independent <= chi2_H0_H0_2 + EPSILON_NUMERIC)
    n_independent_starts_reproducing_H0_2 = None
    if row in panel_event_ids and chi2_H0_H0_2 is not None:
        n_independent_starts_reproducing_H0_2 = int(
            sum(1 for c in candidates if c <= chi2_H0_H0_2 + EPSILON_NUMERIC))

    # ---- nesting audit + reference-H1 rescue (seeded from the INDEPENDENT
    # H0 winner's own params -- never from H0_2) ----
    nesting_violation_before = chi2_H1_old_reference > (chi2_H0_ref2_independent + EPSILON_NUMERIC)
    reference_H1_rescue_ran = False
    chi2_H1_ref2_independent = chi2_H1_old_reference
    nesting_violation_after = nesting_violation_before

    if nesting_violation_before:
        winner_params = winner_params_by_strategy.get(best_independent_strategy)
        if winner_params is None or winner_params.get("t0") is None:
            # old_controlled4_reference JSON does not carry (t0,u0,tE,rho) for
            # the winning strategy -- fall back to re-deriving them is out of
            # scope; flag and leave unresolved rather than silently guessing.
            reference_H1_rescue_ran = False
        else:
            h5_path = manifest.loc[row, "h5_path"]
            meta = load_new_case(h5_path, core)
            rescue = run_nested_h1_rescue(core, fit_lc, meta, winner_params)
            reference_H1_rescue_ran = True
            chi2_H1_ref2_independent = min(chi2_H1_old_reference, float(rescue["chi2"]))
        nesting_violation_after = chi2_H1_ref2_independent > (chi2_H0_ref2_independent + EPSILON_NUMERIC)

    D_ref2_independent = chi2_H0_ref2_independent - chi2_H1_ref2_independent

    # ---- non-circular re-evaluation ----
    dangerous_before_ref2 = dangerous_label(D_s, D_ref2_independent)
    dangerous_after_ref2 = dangerous_label(D_s2, D_ref2_independent) if D_s2 is not None else None
    repaired_ref2 = (dangerous_before_ref2 and (dangerous_after_ref2 is False)) if D_s2 is not None else None
    missed_before_ref2 = missed_label(D_s, D_ref2_independent)
    missed_after_ref2 = missed_label(D_s2, D_ref2_independent) if D_s2 is not None else None
    new_missed_ref2 = ((not missed_before_ref2) and missed_after_ref2) if D_s2 is not None else None

    delta_H0 = (chi2_H0_final_2A - chi2_H0_ref2_independent)
    delta_H1 = (chi2_H1_candidate - chi2_H1_ref2_independent)

    D_best_known_val = h1p_row["D_best_known"]
    D_best_known_val = float(D_best_known_val) if pd.notna(D_best_known_val) else None

    rows.append({
        "event_id": row,
        "chi2_H0_old_reference": chi2_H0_old_reference, "chi2_H0_ref2_independent": chi2_H0_ref2_independent,
        "chi2_H0_H0_2": chi2_H0_H0_2, "chi2_H0_best_known": min(chi2_H0_final_2A, chi2_H0_old_reference),
        "chi2_H1_old_reference": chi2_H1_old_reference, "chi2_H1_ref2_independent": chi2_H1_ref2_independent,
        "chi2_H1_candidate": chi2_H1_candidate, "chi2_H1_best_known": min(chi2_H1_final_2A, chi2_H1_old_reference),
        "D_s": D_s, "D_s2": D_s2, "D_r_old": D_r_old, "D_ref2_independent": D_ref2_independent,
        "D_best_known": D_best_known_val,
        "dangerous_before_ref2": dangerous_before_ref2, "dangerous_after_ref2": dangerous_after_ref2,
        "repaired_ref2": repaired_ref2,
        "missed_before_ref2": missed_before_ref2, "missed_after_ref2": missed_after_ref2,
        "new_missed_ref2": new_missed_ref2,
        "delta_H0": delta_H0, "delta_H1": delta_H1,
        "H0_2_beats_old_reference": h0_2_beats_old_reference,
        "H0_2_beats_independent_reference": h0_2_beats_independent_reference,
        "independent_reference_beats_H0_2": independent_reference_beats_h0_2,
        "H0_2_basin_independently_reproduced": h0_2_basin_independently_reproduced,
        "best_independent_H0_strategy": best_independent_strategy,
        "n_independent_starts_reproducing_H0_2": n_independent_starts_reproducing_H0_2,
        "nesting_violation_before_reference_rescue": bool(nesting_violation_before),
        "reference_H1_rescue_ran": reference_H1_rescue_ran,
        "nesting_violation_after_reference_rescue": bool(nesting_violation_after),
        "in_verification_panel_subset": row in panel_event_ids,
    })
    print(f"row={row} panel={row in panel_event_ids} chi2_H0_ref2ind={chi2_H0_ref2_independent:.3f} "
          f"(best={best_independent_strategy}) D_ref2ind={D_ref2_independent:.3f} D_r_old={D_r_old:.3f} "
          f"nesting_viol_before={nesting_violation_before} nesting_viol_after={nesting_violation_after}",
          flush=True)

df = pd.DataFrame(rows)
out_csv = os.path.join(HERE, "reference_v2_independent.csv")
df.to_csv(out_csv, index=False)
print(f"\nWrote {len(df)} rows -> {out_csv}")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"N events (93 minus catastrophic) = {len(df)}")
print(f"nesting violations before rescue: {df['nesting_violation_before_reference_rescue'].sum()}")
print(f"reference H1 rescues run: {df['reference_H1_rescue_ran'].sum()}")
print(f"nesting violations remaining after rescue: {df['nesting_violation_after_reference_rescue'].sum()}")
print()

n_dang_old = int(df.apply(lambda r: dangerous_label(r["D_s"], r["D_r_old"]), axis=1).sum())
n_dang_ref2 = int(df["dangerous_before_ref2"].sum())
print(f"Q2: original single-start dangerous count: old_reference={n_dang_old}, independent_ref2={n_dang_ref2}")

removed = df[df.apply(lambda r: dangerous_label(r["D_s"], r["D_r_old"]), axis=1) & (~df["dangerous_before_ref2"])]
added = df[(~df.apply(lambda r: dangerous_label(r["D_s"], r["D_r_old"]), axis=1)) & (df["dangerous_before_ref2"])]
print(f"removed from dangerous (old->ref2): {removed['event_id'].tolist()}")
print(f"newly added to dangerous (old->ref2): {added['event_id'].tolist()}")

n_missed_old = int(df.apply(lambda r: missed_label(r["D_s"], r["D_r_old"]), axis=1).sum())
n_missed_ref2 = int(df["missed_before_ref2"].sum())
print(f"missed count: old_reference={n_missed_old}, independent_ref2={n_missed_ref2}")

evaluated_2a = df[df["D_s2"].notna()]
n_dang_before_ref2_2a = int(evaluated_2a["dangerous_before_ref2"].sum())
n_dang_after_ref2_2a = int(evaluated_2a["dangerous_after_ref2"].sum())
n_repaired_ref2 = int(evaluated_2a["repaired_ref2"].sum())
print()
print(f"Q3 (non-circular 2A): N_dangerous_before_ref2={n_dang_before_ref2_2a} "
      f"N_dangerous_after_ref2={n_dang_after_ref2_2a} N_repaired_ref2={n_repaired_ref2} "
      f"repair_rate_ref2={n_repaired_ref2/n_dang_before_ref2_2a:.1%}" if n_dang_before_ref2_2a else "N/A")
print(f"N_missed_before_ref2={int(evaluated_2a['missed_before_ref2'].sum())} "
      f"N_missed_after_ref2={int(evaluated_2a['missed_after_ref2'].sum())} "
      f"N_new_missed_ref2={int(evaluated_2a['new_missed_ref2'].sum())}")
