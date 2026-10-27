#!/usr/bin/env python3
"""
Demonstration 2A: test ONE additional, H1-projected H0 start.

H0_1 and the nominal/settled H1 fit are NOT recomputed -- they are
loaded from Demonstration 1's already-frozen outputs:
  - basin_risk_development_table.csv (D_s, D_r, dangerous, chi2nu_h0,
    H0_1 seed/fit params, continuation_ran_h0, n_points,
    excluded_from_development)
  - final_policy_h1basinvalidation_100events.json (raw chi2_H0_1,
    chi2_H1_current, the settled H1 fitted (t0,u0,tE,rho,piEN,piEE))
  - robust_reference_controlled4_h1basin_100events.json
    (chi2_H0_reference, chi2_H1_reference)
basin_risk_development_table.csv alone does not carry raw chi2_H0_1,
chi2_H1_current, the H1 fitted parameters, or the reference chi2 values
-- these are read from the OTHER two files Demonstration 1's own
pipeline already produced (never recomputed here). Bit-exact
reproducibility of these frozen values was already established in
Demonstration 1 (reproduction_chi2_diff=0.0 for all 100 events); not
repeated here as a "primary source" check, only cited.

Only H0_2 (the H1-projected start) and everything derived from it
(H0_2 continuation, H0_final selection, the post-selection nested-
consistency recheck, D_s2, dangerous_after/missed_after/repaired,
reference-beaten and best-known-nesting audits) are computed fresh in
this task.
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.dirname(HERE)
BRT = os.path.join(PP, "basin_risk_trigger")
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
from run_one_fit_full import run_one_fit_full  # noqa: E402
from final_policy import n_photometry_points, chi2_dof_sanity_flag, CHI2_DOF_SANITY_THRESHOLD  # noqa: E402
from numerical_safeguards import run_same_point_continuation, run_nested_h1_rescue, EPSILON_NUMERIC  # noqa: E402

assert core.BOUNDS_PROFILE == "production_candidate"

print(f"Reusing the project-wide epsilon_numeric = {EPSILON_NUMERIC} "
      f"(source: numerical_safeguards.EPSILON_NUMERIC) for all nesting/reference comparisons.")

DEV_TABLE_PATH = os.environ.get("H1PROJ_DEV_TABLE", os.path.join(BRT, "basin_risk_development_table.csv"))
dev_table = pd.read_csv(DEV_TABLE_PATH)
fp_events = {e["catalog_row"]: e for e in
             json.load(open(os.path.join(PP, "results", "final_policy_h1basinvalidation_100events.json")))["events"]}
ref_events = {e["catalog_row"]: e for e in
              json.load(open(os.path.join(PP, "results", "robust_reference_controlled4_h1basin_100events.json")))["events"]}
manifest = pd.read_csv(os.path.join(PP, "results", "h1_basin_validation_sample_frozen.csv")).set_index("catalog_row")


def dangerous_label(D, D_r):
    return bool((D_r < 9) and (D >= 4) and (D > D_r))


def missed_label(D, D_r):
    return bool((D_r >= 4) and (D < 9) and (D < D_r))


rows = []
for _, dr in dev_table.iterrows():
    row = int(dr["event_id"])
    excluded = bool(dr["excluded_from_development"])

    fp = fp_events[row]
    ref = ref_events[row]

    D_s = float(dr["D_s"])
    D_r = float(dr["D_r"])
    dangerous_before = bool(dr["dangerous"])
    missed_before = missed_label(D_s, D_r)

    chi2_H0_1 = float(fp["chi2_h0"])
    chi2_H1_before = float(fp["chi2_h1"])
    chi2_H0_reference = float(ref["h0_final_chi2"])
    chi2_H1_reference = float(ref["h1_final_chi2"])
    h1_final = fp["final_h1"]

    record = {
        "event_id": row, "D_s": D_s, "D_r": D_r,
        "dangerous_before": dangerous_before, "missed_before": missed_before,
        "chi2_H0_1": chi2_H0_1, "chi2_H1_before_second_start": chi2_H1_before,
        "chi2_H0_reference": chi2_H0_reference, "chi2_H1_reference": chi2_H1_reference,
        "h0_1_continuation_ran": bool(dr["continuation_ran_h0"]),
        "excluded_from_development": excluded,
    }

    if excluded:
        # audited separately (§0/§11): not part of the primary repair analysis,
        # but still record what we can without running H0_2 on it.
        record.update({
            "D_s2": None, "D_best_known": None, "dangerous_after": None, "repaired": None,
            "missed_after": None, "new_missed": None,
            "chi2_H0_2": None, "chi2_H0_final": chi2_H0_1, "chi2_H1_final": chi2_H1_before,
            "h0_2_wins": None, "h0_2_continuation_ran": None,
            "nested_h1_rescue_after_h0_selection": None,
            "reference_beaten_h0": None, "reference_beaten_h1": None,
            "best_known_nesting_violation": None,
            "h0_2_seed_out_of_bounds": None, "h0_2_violating_param": None,
            "delta_chi2_H0_gain": None,
            "delta_D_vs_reference_before": D_s - D_r, "delta_D_vs_reference_after": None,
            "n_total_trf_event": None, "event_wall_s": None,
            "h0_2_seed_t0": None, "h0_2_seed_u0": None, "h0_2_seed_tE": None, "h0_2_seed_rho": None,
            "h0_2_fit_t0": None, "h0_2_fit_u0": None, "h0_2_fit_tE": None, "h0_2_fit_rho": None,
        })
        rows.append(record)
        print(f"row={row} EXCLUDED (already chi2-sanity-flagged in Demonstration 1) -- audit only", flush=True)
        continue

    t_event0 = time.time()
    h5_path = manifest.loc[row, "h5_path"]
    meta = load_new_case(h5_path, core)
    n_points = n_photometry_points(meta["curves"])

    # ---- H0_2 seed: (t0,u0,tE,rho) from the settled H1 fit, piE dropped ----
    seed2 = {"t0": float(h1_final["t0"]), "u0": float(h1_final["u0"]),
             "tE": float(h1_final["tE"]), "rho": float(h1_final["rho"])}

    # ---- out-of-bounds check (production_candidate bounds, same as always) ----
    all_times = []
    for b in ["u", "g", "r", "i", "z", "y"]:
        lc = meta["curves"][b]
        if len(lc):
            all_times.extend(np.asarray(lc[:, 0], dtype=float).tolist())
    t_lo, t_hi = float(np.min(all_times)), float(np.max(all_times))
    bounds_check = {
        "t0": (t_lo, t_hi), "u0": (-10.0, 10.0), "tE": (0.1, 500000.0), "rho": (1.0e-7, 10.0),
    }
    violations = []
    for p, (lo, hi) in bounds_check.items():
        if not (lo <= seed2[p] <= hi):
            violations.append(p)

    if violations:
        record.update({
            "D_s2": None, "D_best_known": None, "dangerous_after": None, "repaired": None,
            "missed_after": None, "new_missed": None,
            "chi2_H0_2": None, "chi2_H0_final": chi2_H0_1, "chi2_H1_final": chi2_H1_before,
            "h0_2_wins": False, "h0_2_continuation_ran": None,
            "nested_h1_rescue_after_h0_selection": None,
            "reference_beaten_h0": None, "reference_beaten_h1": None,
            "best_known_nesting_violation": None,
            "h0_2_seed_out_of_bounds": True, "h0_2_violating_param": ",".join(violations),
            "delta_chi2_H0_gain": None,
            "delta_D_vs_reference_before": D_s - D_r, "delta_D_vs_reference_after": None,
            "n_total_trf_event": None, "event_wall_s": time.time() - t_event0,
            "h0_2_seed_t0": seed2["t0"], "h0_2_seed_u0": seed2["u0"],
            "h0_2_seed_tE": seed2["tE"], "h0_2_seed_rho": seed2["rho"],
            "h0_2_fit_t0": None, "h0_2_fit_u0": None, "h0_2_fit_tE": None, "h0_2_fit_rho": None,
        })
        rows.append(record)
        print(f"row={row} H0_2 SEED OUT OF BOUNDS: {violations} -- excluded from repair-rate denominator", flush=True)
        continue

    # ---- run H0_2 nominal ----
    h0_2_nominal = run_one_fit_full(core, fit_lc, meta, "H0", seed2, "h1_projected")
    n_trf = 1

    flagged2, _ = chi2_dof_sanity_flag(h0_2_nominal, n_points, 4)
    h0_2_continuation_ran = False
    h0_2_settled = h0_2_nominal
    if flagged2:
        h0_2_continuation_ran = True
        cont2 = run_same_point_continuation(core, fit_lc, meta, "H0", h0_2_nominal,
                                             label_suffix="chi2dof_continuation")
        n_trf += 1
        if cont2["fit_record"]["chi2"] <= h0_2_nominal["chi2"]:
            h0_2_settled = cont2["fit_record"]

    chi2_H0_2 = float(h0_2_settled["chi2"])
    h0_2_wins = chi2_H0_2 < chi2_H0_1
    chi2_H0_final = min(chi2_H0_1, chi2_H0_2)

    # ---- recheck nested consistency AFTER H0 selection ----
    nested_rescue_ran = False
    chi2_H1_final = chi2_H1_before
    if chi2_H1_before > chi2_H0_final + EPSILON_NUMERIC:
        nested_rescue_ran = True
        # build a minimal dict with the winning H0's (t0,u0,tE,rho) for the rescue seed
        if h0_2_wins:
            h0_final_params = {"t0": h0_2_settled["t0"], "u0": h0_2_settled["u0"],
                                "tE": h0_2_settled["tE"], "rho": h0_2_settled["rho"]}
        else:
            h0_final_params = {"t0": float(dev_table.loc[dev_table["event_id"] == row, "h0_fit_t0"].iloc[0]),
                                "u0": float(dev_table.loc[dev_table["event_id"] == row, "h0_fit_u0"].iloc[0]),
                                "tE": float(dev_table.loc[dev_table["event_id"] == row, "h0_fit_tE"].iloc[0]),
                                "rho": float(dev_table.loc[dev_table["event_id"] == row, "h0_fit_rho"].iloc[0])}
        rescue = run_nested_h1_rescue(core, fit_lc, meta, h0_final_params)
        n_trf += 1
        chi2_H1_final = min(chi2_H1_before, float(rescue["chi2"]))

    D_s2 = chi2_H0_final - chi2_H1_final
    dangerous_after = dangerous_label(D_s2, D_r)
    repaired = dangerous_before and (not dangerous_after)
    missed_after = missed_label(D_s2, D_r)
    new_missed = (not missed_before) and missed_after

    reference_beaten_h0 = chi2_H0_final < (chi2_H0_reference - EPSILON_NUMERIC)
    reference_beaten_h1 = chi2_H1_final < (chi2_H1_reference - EPSILON_NUMERIC)

    chi2_H0_best_known = min(chi2_H0_reference, chi2_H0_final)
    chi2_H1_best_known = min(chi2_H1_reference, chi2_H1_final)
    best_known_nesting_violation = chi2_H0_best_known < (chi2_H1_best_known - EPSILON_NUMERIC)
    D_best_known = chi2_H0_best_known - chi2_H1_best_known

    event_wall_s = time.time() - t_event0

    record.update({
        "D_s2": D_s2, "D_best_known": D_best_known,
        "dangerous_after": dangerous_after, "repaired": repaired,
        "missed_after": missed_after, "new_missed": new_missed,
        "chi2_H0_2": chi2_H0_2, "chi2_H0_final": chi2_H0_final, "chi2_H1_final": chi2_H1_final,
        "h0_2_wins": bool(h0_2_wins), "h0_2_continuation_ran": h0_2_continuation_ran,
        "nested_h1_rescue_after_h0_selection": nested_rescue_ran,
        "reference_beaten_h0": bool(reference_beaten_h0), "reference_beaten_h1": bool(reference_beaten_h1),
        "best_known_nesting_violation": bool(best_known_nesting_violation),
        "h0_2_seed_out_of_bounds": False, "h0_2_violating_param": None,
        "delta_chi2_H0_gain": chi2_H0_1 - chi2_H0_final,
        "delta_D_vs_reference_before": D_s - D_r, "delta_D_vs_reference_after": D_s2 - D_r,
        "n_total_trf_event": n_trf, "event_wall_s": event_wall_s,
        "h0_2_seed_t0": seed2["t0"], "h0_2_seed_u0": seed2["u0"],
        "h0_2_seed_tE": seed2["tE"], "h0_2_seed_rho": seed2["rho"],
        "h0_2_fit_t0": h0_2_settled["t0"], "h0_2_fit_u0": h0_2_settled["u0"],
        "h0_2_fit_tE": h0_2_settled["tE"], "h0_2_fit_rho": h0_2_settled["rho"],
    })
    rows.append(record)
    print(f"row={row} D_s={D_s:.3f} D_s2={D_s2:.3f} D_r={D_r:.3f} dangerous_before={dangerous_before} "
          f"dangerous_after={dangerous_after} repaired={repaired} h0_2_wins={h0_2_wins} "
          f"gain={record['delta_chi2_H0_gain']:.3f} trf={n_trf} wall={event_wall_s:.2f}s", flush=True)

df = pd.DataFrame(rows)
out_csv = os.path.join(HERE, "h1_projected_h0_results.csv")
df.to_csv(out_csv, index=False)
print(f"\nWrote {len(df)} rows -> {out_csv}")
